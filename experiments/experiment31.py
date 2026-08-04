"""
experiment31.py — SDM 动态容量修复: 随机扩增 (v2)

experiment30 失败根因: 新地址位型取 ctx 最活跃位 (= 共享位, 多次叠加
值最大), 被大量不同类事件共享命中 → top-k 被新地址垄断 (库50 93%→
20%; teacher==free 同值证实读出路径被垄断)。

修复: _spawn 位型 = 纯随机 (与初始池同分布, 保持均匀性), 扩增由
负载阈值 θ 驱动 (热点地址被写 θ 次才诞生新地址, 容量随学习增长,
但新地址与旧地址公平竞争 top-k)。

变体: θ ∈ {6, 30}; 对照 experiment29 固定 4096 (库50 t93.2/f91.6,
库100 t79.0/f62.1)。先库50 快速验证恢复, 再库100/200。
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


class DynamicSDMv2:
    """随机扩增的稀疏分布式记忆 (v2, 修复 experiment30 垄断问题)"""

    def __init__(self, ar, init_addrs=1024, addr_ones=24, k=64,
                 max_addrs=16384, load_theta=30, seed=7):
        self.ar = ar
        self.dim = ar.dim
        self.k = k
        self.addr_ones = addr_ones
        self.max_addrs = max_addrs
        self.load_theta = load_theta
        g = torch.Generator().manual_seed(seed)
        self._g = g
        self.addr_idx = [
            torch.randperm(self.dim, generator=g)[:addr_ones].to(DEVICE)
            for _ in range(init_addrs)]
        self.content = [torch.zeros(self.dim, device=DEVICE)
                        for _ in range(init_addrs)]
        self.load = [0] * init_addrs
        self.n_spawn = 0

    def _sim(self, q):
        idx = torch.nonzero(q).flatten()
        if idx.numel() == 0:
            return torch.zeros(len(self.addr_idx), device=DEVICE)
        mask = torch.zeros(self.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        A = torch.stack(self.addr_idx)
        return mask[A].sum(1).float()

    def _spawn(self):
        """纯随机位型: 与初始池同分布, 均匀参与竞争"""
        if len(self.addr_idx) >= self.max_addrs:
            return False
        self.addr_idx.append(
            torch.randperm(self.dim, generator=self._g)[:self.addr_ones].to(DEVICE))
        self.content.append(torch.zeros(self.dim, device=DEVICE))
        self.load.append(0)
        self.n_spawn += 1
        return True

    def write(self, ctx, nxt):
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        if self.load[top[0].item()] >= self.load_theta:
            if self._spawn():
                self.content[-1] += nxt
                self.load[-1] += 1
        for a in top:
            self.content[a] += nxt
            self.load[a] += 1

    def read(self, query, t):
        sim = self._sim(query)
        top = torch.topk(sim, self.k).indices
        raw = torch.stack([self.content[a] for a in top]).sum(0)
        lo = (self.ar.max_pos_in + t) * self.ar.block_size
        rk = raw[lo:lo + self.ar.block_size]
        scores = self.ar.p[t] @ rk
        return int(scores.argmax())


def run_scale(n_train, theta):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    sdm = DynamicSDMv2(ar, load_theta=theta)

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
    avg_load = sum(sdm.load) / max(len(sdm.addr_idx), 1)
    print(f"  θ={theta}: {n_ev} 事件 → {len(sdm.addr_idx)} 地址 "
          f"(诞生 {sdm.n_spawn}, 平均负载 {avg_load:.1f}, "
          f"{time.perf_counter()-t0:.0f}s)", flush=True)

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
    print("experiment31 — SDM 动态容量 v2 (随机扩增)", flush=True)
    print("=" * 66, flush=True)
    print("对照 (exp29 固定4096): 库50 t93.2/f91.6; 库100 t79.0/f62.1",
          flush=True)
    # 库50 先验证恢复
    for theta in (30, 6):
        print(f"\n[库50 θ={theta}]", flush=True)
        run_scale(50, theta)
    # 最优 θ 跑大库 (先定 θ=30)
    for n in (100, 200):
        print(f"\n[库{n} θ=30]", flush=True)
        run_scale(n, 30)


if __name__ == "__main__":
    main()
