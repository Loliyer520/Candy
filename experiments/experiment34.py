"""
experiment34.py — SDM 动态容量 v4: 负载目标驱动批量扩增

v3 教训: spawn 触发看单热点负载 → 新地址分流热点后停止扩增, 地址数
不足 (库50 1142, 负载65)。正确"随学习扩增" = 维持全局平均负载恒定:
  平均负载 = 总写入 / 地址数 > target → 批量诞生新地址 (活性偏置8 +
  成熟期), 直到负载回落。地址数随事件数线性增长。

对照: 固定4096 (预生成均匀采样, 库50 负载18 甜点 t93.2/f91.6;
库200 负载74 崩 t60.2/f33.3)。v4 负载目标=18 → 库50 地址≈4144
(对齐固定规模), 库200≈16935 (封顶16384) → 直接对比
"动态生成的池" vs "预生成池"。
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


class DynamicSDMv4:
    """负载目标驱动的批量动态扩增"""

    def __init__(self, ar, init_addrs=1024, addr_ones=24, k=64,
                 max_addrs=16384, load_target=18, batch=128,
                 act_bias=8, mature_events=30, seed=7):
        self.ar = ar
        self.dim = ar.dim
        self.k = k
        self.addr_ones = addr_ones
        self.max_addrs = max_addrs
        self.load_target = load_target
        self.batch = batch
        self.act_bias = act_bias
        self.mature_events = mature_events
        g = torch.Generator().manual_seed(seed)
        self._g = g
        self.addr_idx = [
            torch.randperm(self.dim, generator=g)[:addr_ones].to(DEVICE)
            for _ in range(init_addrs)]
        self.content = [torch.zeros(self.dim, device=DEVICE)
                        for _ in range(init_addrs)]
        self.load = [0] * init_addrs
        self.birth = [0] * init_addrs
        self.n_spawn = 0
        self.global_ev = 0

    def _sim(self, q):
        idx = torch.nonzero(q).flatten()
        if idx.numel() == 0:
            return torch.zeros(len(self.addr_idx), device=DEVICE)
        mask = torch.zeros(self.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        A = torch.stack(self.addr_idx)
        return mask[A].sum(1).float()

    def _spawn(self, ctx):
        if len(self.addr_idx) >= self.max_addrs:
            return False
        nz = torch.nonzero(ctx).flatten()
        if nz.numel() < self.act_bias:
            return False
        perm = torch.randperm(nz.numel(), generator=self._g)
        ctx_bits = nz[perm[:self.act_bias]].cpu()
        all_pos = torch.randperm(self.dim, generator=self._g)
        excl = ~torch.isin(all_pos, ctx_bits)
        rand_bits = all_pos[excl][:self.addr_ones - self.act_bias]
        new_idx = torch.cat([ctx_bits, rand_bits]).to(DEVICE)
        self.addr_idx.append(new_idx)
        self.content.append(torch.zeros(self.dim, device=DEVICE))
        self.load.append(0)
        self.birth.append(self.global_ev)
        self.n_spawn += 1
        return True

    def write(self, ctx, nxt):
        self.global_ev += 1
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        # 负载目标: 平均负载超 target → 批量扩增
        total = sum(self.load) + self.k
        if total / len(self.addr_idx) > self.load_target:
            for _ in range(self.batch):
                if not self._spawn(ctx):
                    break
        for a in top:
            self.content[a] += nxt
            self.load[a] += 1

    def read(self, query, t):
        sim = self._sim(query)
        if self.mature_events > 0:
            age = self.global_ev - torch.tensor(self.birth, device=DEVICE)
            sim = sim.masked_fill(age < self.mature_events, -1e9)
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
    sdm = DynamicSDMv4(ar)
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
    avg = sum(sdm.load) / max(len(sdm.addr_idx), 1)
    print(f"  {n_ev} 事件 → {len(sdm.addr_idx)} 地址 "
          f"(诞生 {sdm.n_spawn}, 平均负载 {avg:.1f}, "
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
    print("experiment34 — SDM 动态容量 v4 (负载目标批量扩增)", flush=True)
    print("=" * 66, flush=True)
    print("对照固定4096: 库50 t93.2/f91.6 (负载18); 库200 t60.2/f33.3",
          flush=True)
    for n in (50, 200):
        print(f"\n[库{n}]", flush=True)
        run_scale(n)


if __name__ == "__main__":
    main()
