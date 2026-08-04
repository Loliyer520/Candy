"""
experiment30.py — SDM 动态容量: 随学习自行扩增地址池

用户指令: "增加动态容量, 随着学习自行扩增"。

机制 (生物约束内, DG 成年神经发生类比 Aimone 2014):
  - 初始地址池小 (1024), 不预分配大池
  - 写入事件时, 若激活的 top-1 地址"负载"(已写入事件数) ≥ 阈值 θ
    → 诞生新地址: 位型 = 当前上下文最活跃的 addr_ones 位 (活性依赖
      募集, 新神经元对当前刺激类敏感) → 冲突热点分化 (聚类分裂)
  - 新地址立即参与当前事件写入 (Hebbian 就近叠加), 后续相似事件
    命中新地址 → 容量随学习扩增, 且扩增发生在"需要的地方"
  - 上限 max_addrs (资源安全); 解码不变 (就近读出 + 位型内积 argmax)

对照: experiment29 固定 4096 地址 (库50 teacher 93.2% / free 91.6%;
库100 teacher 79.0% / free 62.1%)。本实验: 初始 1024, 动态扩增,
看容量扩增是否缓解规模衰减 + 记录最终地址数与每地址负载。
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


class DynamicSDM:
    """负载驱动动态扩增的稀疏分布式记忆"""

    def __init__(self, ar, init_addrs=1024, addr_ones=24, k=64,
                 max_addrs=8192, load_theta=6, seed=7):
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
        A = torch.stack(self.addr_idx)          # 动态 stack (地址数少时快)
        return mask[A].sum(1).float()

    def _spawn(self, ctx):
        """活性依赖募集: 新地址位型 = ctx 最活跃位 (+ 随机补充)"""
        if len(self.addr_idx) >= self.max_addrs:
            return False
        nz = torch.nonzero(ctx).flatten()
        if nz.numel() == 0:
            return False
        topv, topi = torch.topk(ctx[nz], min(self.addr_ones, nz.numel()))
        new_idx = nz[topi].to(DEVICE)
        if new_idx.numel() < self.addr_ones:
            need = self.addr_ones - new_idx.numel()
            extra = torch.randperm(self.dim, generator=self._g)[:need].to(DEVICE)
            new_idx = torch.cat([new_idx, extra])
        self.addr_idx.append(new_idx)
        self.content.append(torch.zeros(self.dim, device=DEVICE))
        self.load.append(0)
        self.n_spawn += 1
        return True

    def write(self, ctx, nxt):
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        # 热点过载 → 诞生新地址并立即写入当前事件
        if self.load[top[0].item()] >= self.load_theta and self._spawn(ctx):
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


def run_scale(n_train):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    sdm = DynamicSDM(ar)

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
    avg_load = (sum(sdm.load) / max(len(sdm.addr_idx), 1))
    print(f"  {n_ev} 事件 → {len(sdm.addr_idx)} 地址 "
          f"(诞生 {sdm.n_spawn}, 平均负载 {avg_load:.1f}, "
          f"写入 {time.perf_counter()-t0:.0f}s)", flush=True)

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
    print(f"  (共 {time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    print("=" * 66, flush=True)
    print("experiment30 — SDM 动态容量 (负载驱动地址扩增)", flush=True)
    print("=" * 66, flush=True)
    print("对照 (experiment29 固定4096): 库50 t93.2/f91.6; "
          "库100 t79.0/f62.1", flush=True)
    for n in (50, 100, 200):
        print(f"\n[库{n}]", flush=True)
        run_scale(n)


if __name__ == "__main__":
    main()
