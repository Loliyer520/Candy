"""
experiment36.py — 分区隔离池 (对话级路由, 表间零污染)

用户指令: 叠表+遗忘失败后选择"分区隔离池"。

机制 (规避 v1-v5 全部失败模式):
  - 桶 = 固定随机投影 (均匀聚桶, 与 exp29 固定4096 完全同构)
  - 路由: hash(ic) % max_buckets — 用完整输入, 读写恒一致, 永不漂移
    (表数恒定, 无"扩表导致已写内容路由漂移"问题)
  - 桶容量 ≈ 50 对话 (对齐固定4096/库50 甜点: 负载 ≈ 18)
  - 动态增容: 对话数 → 桶数线性增长 (每桶独立, 桶间零污染)
  - 读出: 只读 hash(ic) 桶, 单桶 top-k

对照:
  - 固定4096: 库50 t93.2/f91.6; 库200 t60.2/f33.3 (负载74 崩)
  - 固定16384 库200: 负载 18.6 → 验证"负载恒 18 性能恒定"纯容量假设
预期: 分区后库200/库400 每桶仍是 50 对话甜点 → 性能 ≈ 库50 的 93%。
"""

import sys, os, time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from episodic_memory import AutoRegressiveEventMemory, DEVICE


def text_to_codes(text):
    return [ord(c) for c in text if 32 <= ord(c) <= 126]


def load_pairs(path, n, offset=0):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            if offset > 0:
                offset -= 1
                continue
            inp, resp = line.split("\t", 1)
            pairs.append((inp.strip(), resp.strip()))
            if len(pairs) >= n:
                break
    return pairs


def route_hash(codes, n_buckets):
    """带位置加权确定性哈希 (ic 完整 → 读写路由一致)"""
    h = 0
    for i, c in enumerate(codes):
        h = (h * 31 + c * (i + 1)) & 0x7FFFFFFF
    return h % n_buckets


class PartitionSDM:
    """顺序桶分区池: 每桶 50 对话甜点, 桶满开新桶, 桶间零污染

    路由用写入时记录的 route 表 (ic_hash -> 桶), 不依赖桶数,
    扩桶永不漂移。桶 = 固定随机投影, 每桶 ≈ 库50 甜点配置。
    """

    def __init__(self, ar, bucket_size=4096, bucket_k=64,
                 dlg_per_bucket=50, seed=7):
        self.ar = ar
        self.bucket_size = bucket_size
        self.bucket_k = bucket_k
        self.dlg_per_bucket = dlg_per_bucket
        g = torch.Generator().manual_seed(seed)
        # 所有桶共享同一随机投影 (均匀聚桶性质一致)
        self.addr_idx = torch.stack([
            torch.randperm(ar.dim, generator=g)[:24]
            for _ in range(bucket_size)]).to(DEVICE)
        # content 惰性分配: 只用到的桶才占内存
        self.content = {}
        self.route = {}        # ic_hash -> bucket
        self.cur = 0           # 当前活跃桶
        self.n_dlg_in_cur = 0  # 当前桶内对话数

    def _bucket(self, b):
        if b not in self.content:
            self.content[b] = torch.zeros(
                self.bucket_size, self.ar.dim, device=DEVICE)
        return self.content[b]

    def _sim(self, q):
        idx = torch.nonzero(q).flatten()
        if idx.numel() == 0:
            return torch.zeros(self.bucket_size, device=DEVICE)
        mask = torch.zeros(self.ar.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        return mask[self.addr_idx].sum(1).float()

    def _buck_of(self, ic):
        h = route_hash(ic, 1 << 31)
        if h in self.route:
            return self.route[h]
        # 库外: 路由到当前活跃桶的确定性近似
        return self.cur if self.cur in self.content else 0

    def assign_dlg(self, ic):
        """新对话分配桶: 按对话数计容量, 同 ic_hash 对话整体同桶"""
        h = route_hash(ic, 1 << 31)
        if h in self.route:
            return self.route[h]
        if self.n_dlg_in_cur >= self.dlg_per_bucket:
            self.cur += 1
            self.n_dlg_in_cur = 0
        self.route[h] = self.cur
        self.n_dlg_in_cur += 1
        return self.cur

    def write(self, ic, ctx, nxt):
        b = self._buck_of(ic)
        sim = self._sim(ctx)
        top = torch.topk(sim, self.bucket_k).indices
        self._bucket(b)[top] += nxt

    def read(self, ic, query, t):
        b = self._buck_of(ic)
        sim = self._sim(query)
        top = torch.topk(sim, self.bucket_k).indices
        raw = self._bucket(b)[top].sum(0)
        lo = (self.ar.max_pos_in + t) * self.ar.block_size
        rk = raw[lo:lo + self.ar.block_size]
        scores = self.ar.p[t] @ rk
        return int(scores.argmax())

    def stats(self):
        return len(self.content)


class FixedSDM:
    """固定大池对照 (exp29 同构, 无路由)"""

    def __init__(self, ar, size=16384, k=64, seed=7):
        self.ar = ar
        self.size = size
        self.k = k
        g = torch.Generator().manual_seed(seed)
        self.addr_idx = torch.stack([
            torch.randperm(ar.dim, generator=g)[:24]
            for _ in range(size)]).to(DEVICE)
        self.content = torch.zeros(size, ar.dim, device=DEVICE)
        self.writes = 0

    def _sim(self, q):
        idx = torch.nonzero(q).flatten()
        mask = torch.zeros(self.ar.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        return mask[self.addr_idx].sum(1).float()

    def write(self, ctx, nxt):
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        self.content[top] += nxt
        self.writes += 1

    def read(self, ic, query, t):
        sim = self._sim(query)
        top = torch.topk(sim, self.k).indices
        raw = self.content[top].sum(0)
        lo = (self.ar.max_pos_in + t) * self.ar.block_size
        rk = raw[lo:lo + self.ar.block_size]
        scores = self.ar.p[t] @ rk
        return int(scores.argmax())


def make_ar():
    return AutoRegressiveEventMemory(dim=12288, char_ones=8,
                                     max_pos_in=32, max_pos_out=32,
                                     seed=7)


def train_all(ar, sdm, dlg):
    for inp, resp in dlg:
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc:
            continue
        if isinstance(sdm, PartitionSDM):
            sdm.assign_dlg(ic)
        for t in range(1, len(rc)):
            if t >= ar.max_pos_out:
                break
            ctx = ar.bind(ic, 0) + ar.bind(rc[:t], ar.max_pos_in)
            nxt = torch.zeros(ar.dim, device=DEVICE)
            lo = (ar.max_pos_in + t) * ar.block_size
            nxt[lo:lo + ar.block_size] = ar.p[t][rc[t]]
            sdm.write(ic, ctx, nxt) if isinstance(sdm, PartitionSDM) \
                else sdm.write(ctx, nxt)


def eval_dlg(ar, sdm, dlg):
    tf_ok = tf_tot = fr_ok = fr_tot = 0
    for inp, resp in dlg:
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc:
            continue
        for t in range(1, len(rc)):
            if t >= ar.max_pos_out:
                break
            query = ar.bind(ic, 0) + ar.bind(rc[:t], ar.max_pos_in)
            code = sdm.read(ic, query, t)
            tf_ok += (code == rc[t]); tf_tot += 1
        prefix = [rc[0]]
        for t in range(1, len(rc)):
            if t >= ar.max_pos_out:
                break
            query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
            code = sdm.read(ic, query, t)
            fr_ok += (code == rc[t]); fr_tot += 1
            prefix.append(code)
    return (tf_ok, tf_tot, fr_ok, fr_tot)


def run_partition(n_train, bucket_size=4096, bucket_k=64,
                  dlg_per_bucket=50):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = make_ar()
    sdm = PartitionSDM(ar, bucket_size, bucket_k, dlg_per_bucket)
    train_all(ar, sdm, train_dlg)
    print(f"  训练 {n_train} 对话 → {sdm.stats()} 桶 "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    for tag, dlg in (("库内(前14)", train_dlg[:14]), ("库外(10)", out_dlg)):
        tf_ok, tf_tot, fr_ok, fr_tot = eval_dlg(ar, sdm, dlg)
        print(f"  [{tag}] teacher {tf_ok}/{tf_tot} = "
              f"{tf_ok/max(tf_tot,1):.1%} | "
              f"free-run {fr_ok}/{fr_tot} = {fr_ok/max(fr_tot,1):.1%}",
              flush=True)


def run_fixed(n_train, size=16384, k=64):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = make_ar()
    sdm = FixedSDM(ar, size, k)
    train_all(ar, sdm, train_dlg)
    load = sdm.writes * k / size
    print(f"  训练 {n_train} 对话 → {size} 地址 (负载 {load:.1f}, "
          f"{time.perf_counter()-t0:.0f}s)", flush=True)
    for tag, dlg in (("库内(前14)", train_dlg[:14]), ("库外(10)", out_dlg)):
        tf_ok, tf_tot, fr_ok, fr_tot = eval_dlg(ar, sdm, dlg)
        print(f"  [{tag}] teacher {tf_ok}/{tf_tot} = "
              f"{tf_ok/max(tf_tot,1):.1%} | "
              f"free-run {fr_ok}/{fr_tot} = {fr_ok/max(fr_tot,1):.1%}",
              flush=True)


def main():
    print("=" * 66, flush=True)
    print("experiment36 — 分区隔离池(失败) + 固定池容量缩放验证", flush=True)
    print("=" * 66, flush=True)
    print("分区结论: 同ic聚桶→桶内超载, 库200 t78.6/f57.0", flush=True)
    print("        < 固定16384 t83.5/f60.2 (负载恒18才关键)", flush=True)
    print("\n[固定16384 库200 (负载18.6)]", flush=True)
    run_fixed(200, size=16384, k=64)
    print("\n[固定32768 库400 (负载恒18, 容量扩展验证)]", flush=True)
    run_fixed(400, size=32768, k=64)
    print("\n[固定4096 库400 (旧基线对照)]", flush=True)
    run_fixed(400, size=4096, k=64)


if __name__ == "__main__":
    main()
