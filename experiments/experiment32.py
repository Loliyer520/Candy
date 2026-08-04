"""
experiment32.py — 固定 SDM 池容量上限 (库200/400) + 地址密度

判断动态扩增需求: experiment31 两种动态扩增 (活性 v1 20%, 随机 v2
~70%) 均不如固定 4096 (93.2%)。若固定池在更大库仍能扛, 则容量非
瓶颈, 动态扩增无必要; 若崩, 则需求存在但需新机制。

扫描: 固定 4096 × 库200/400; 固定 8192 × 库400。
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


class SDMem:
    def __init__(self, ar, n_addrs=4096, addr_ones=24, k=64, seed=7):
        self.ar = ar
        self.dim = ar.dim
        self.n = n_addrs
        self.k = k
        g = torch.Generator().manual_seed(seed)
        self.addr_idx = torch.stack([
            torch.randperm(self.dim, generator=g)[:addr_ones]
            for _ in range(n_addrs)]).to(DEVICE)
        self.content = torch.zeros(n_addrs, self.dim, device=DEVICE)

    def _sim(self, q):
        idx = torch.nonzero(q).flatten()
        if idx.numel() == 0:
            return torch.zeros(self.n, device=DEVICE)
        mask = torch.zeros(self.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        return mask[self.addr_idx].sum(1).float()

    def write(self, ctx, nxt):
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        self.content[top] += nxt

    def read(self, query, t):
        sim = self._sim(query)
        top = torch.topk(sim, self.k).indices
        raw = self.content[top].sum(0)
        lo = (self.ar.max_pos_in + t) * self.ar.block_size
        rk = raw[lo:lo + self.ar.block_size]
        scores = self.ar.p[t] @ rk
        return int(scores.argmax())


def run_scale(n_train, n_addrs):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    sdm = SDMem(ar, n_addrs=n_addrs)
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
    print(f"  {n_ev} 事件 → 固定 {n_addrs} 地址 "
          f"(平均负载 {n_ev*64/n_addrs:.1f}, {time.perf_counter()-t0:.0f}s)",
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
    print("experiment32 — 固定 SDM 池容量上限 (库200/400)", flush=True)
    print("=" * 66, flush=True)
    print("对照 (exp29 固定4096): 库50 t93.2/f91.6; 库100 t79.0/f62.1",
          flush=True)
    for n, a in ((200, 4096), (400, 4096), (400, 8192)):
        print(f"\n[库{n} 固定{a}]", flush=True)
        run_scale(n, a)


if __name__ == "__main__":
    main()
