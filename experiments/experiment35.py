"""
experiment35.py — 叠表并行容量: 动态扩表 + 遗忘

用户指令: "能不能叠表并行容量? 动态扩表以及增加遗忘?"

机制 (保留固定池统计性质, 规避 v4 失败):
  - 表 = 独立预生成随机投影 (均匀聚桶), 每表 1024 地址, k=16
    (激活比例 1.6%, 对齐固定4096/k=64 的甜点负载)
  - 动态扩表: 平均负载 = 总写入/总地址 > 目标(18) → 新增一张表
    (预生成随机, 不破坏统计) → 总地址随学习增长
  - 路由: hash(ctx) → 确定性分配到表 (同 ctx 同类事件同表, 聚桶完整)
  - 读出: 所有表 top-k 汇总求和 → 位型内积 argmax
  - 遗忘: 每事件 content ×= decay (指数衰减 = 近因窗口, 容量循环,
    旧记忆淡出, 新记忆权重 1)

对照 (exp29/32): 固定4096 库50 t93.2/f91.6; 库200 t60.2/f33.3。
预期: 扩表把库200 负载压回 18 → 容量显著提升; 遗忘提供近因性。
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


class SDMTable:
    """独立预生成随机投影表 (均匀聚桶)"""

    def __init__(self, ar, size=1024, k=16, seed=7):
        self.ar = ar
        self.dim = ar.dim
        self.size = size
        self.k = k
        g = torch.Generator().manual_seed(seed)
        self.addr_idx = torch.stack([
            torch.randperm(self.dim, generator=g)[:24]
            for _ in range(size)]).to(DEVICE)
        self.content = torch.zeros(size, self.dim, device=DEVICE)
        self.writes = 0

    def _sim(self, q):
        idx = torch.nonzero(q).flatten()
        if idx.numel() == 0:
            return torch.zeros(self.size, device=DEVICE)
        mask = torch.zeros(self.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        return mask[self.addr_idx].sum(1).float()

    def write(self, ctx, nxt):
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        self.content[top] += nxt
        self.writes += 1

    def read_raw(self, query):
        sim = self._sim(query)
        top = torch.topk(sim, self.k).indices
        return self.content[top].sum(0)

    def forget(self, decay):
        self.content *= decay


class SDMStack:
    """叠表: 动态扩表 (hash 路由) + 遗忘"""

    def __init__(self, ar, table_size=1024, max_tables=8,
                 load_target=18.0, decay=1.0, seed=7):
        self.ar = ar
        self.table_size = table_size
        self.max_tables = max_tables
        self.load_target = load_target
        self.decay = decay
        self.k_per = max(8, table_size // 64)   # 1.6% 激活比例
        self.tables = [SDMTable(ar, table_size, self.k_per, seed)]
        self._seed = seed

    def _hash(self, ctx):
        return int(ctx.sum().item()) % len(self.tables)

    def _avg_load(self):
        total_w = sum(t.writes for t in self.tables)
        total_a = sum(t.size for t in self.tables)
        return total_w * self.k_per / max(total_a, 1)

    def write(self, ctx, nxt):
        # 动态扩表: 平均负载超目标 → 新增预生成随机表
        if (len(self.tables) < self.max_tables
                and self._avg_load() > self.load_target):
            self.tables.append(
                SDMTable(self.ar, self.table_size, self.k_per,
                         self._seed + len(self.tables)))
        # 遗忘: 全局内容指数衰减 (近因窗口)
        if self.decay < 1.0:
            for t in self.tables:
                t.forget(self.decay)
        # hash 路由写入
        t = self.tables[self._hash(ctx)]
        t.write(ctx, nxt)

    def read(self, query, t):
        raw = torch.zeros(self.ar.dim, device=DEVICE)
        for tb in self.tables:
            raw += tb.read_raw(query)
        lo = (self.ar.max_pos_in + t) * self.ar.block_size
        rk = raw[lo:lo + self.ar.block_size]
        scores = self.ar.p[t] @ rk
        return int(scores.argmax())


def run_scale(n_train, decay):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    sdm = SDMStack(ar, decay=decay)
    n_ev = 0
    for inp, resp in train_dlg:
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc:
            continue
        for t in range(1, len(rc)):
            if t >= ar.max_pos_out:
                break
            ctx = ar.bind(ic, 0) + ar.bind(rc[:t], ar.max_pos_in)
            nxt = torch.zeros(ar.dim, device=DEVICE)
            lo = (ar.max_pos_in + t) * ar.block_size
            nxt[lo:lo + ar.block_size] = ar.p[t][rc[t]]
            sdm.write(ctx, nxt)
            n_ev += 1
    total_a = sum(t.size for t in sdm.tables)
    print(f"  decay={decay}: {n_ev} 事件 → {len(sdm.tables)} 表 × "
          f"{sdm.table_size} = {total_a} 地址 (平均负载 "
          f"{sdm._avg_load():.1f}, {time.perf_counter()-t0:.0f}s)",
          flush=True)
    for tag, dlg in (("库内(前14)", train_dlg[:14]), ("库外(10)", out_dlg)):
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
                code = sdm.read(query, t)
                tf_ok += (code == rc[t]); tf_tot += 1
            prefix = [rc[0]]
            for t in range(1, len(rc)):
                if t >= ar.max_pos_out:
                    break
                query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
                code = sdm.read(query, t)
                fr_ok += (code == rc[t]); fr_tot += 1
                prefix.append(code)
        print(f"  [{tag}] teacher {tf_ok}/{tf_tot} = "
              f"{tf_ok/max(tf_tot,1):.1%} | "
              f"free-run {fr_ok}/{fr_tot} = {fr_ok/max(fr_tot,1):.1%}",
              flush=True)


def main():
    print("=" * 66, flush=True)
    print("experiment35 — 叠表并行容量 (动态扩表 + 遗忘)", flush=True)
    print("=" * 66, flush=True)
    print("对照固定4096: 库50 t93.2/f91.6; 库200 t60.2/f33.3", flush=True)
    print("\n[库50 decay=1.0 (扩表无遗忘)]", flush=True)
    run_scale(50, 1.0)
    print("\n[库200 decay=1.0 (扩表无遗忘)]", flush=True)
    run_scale(200, 1.0)
    print("\n[库200 decay=0.9995 (扩表+遗忘)]", flush=True)
    run_scale(200, 0.9995)


if __name__ == "__main__":
    main()
