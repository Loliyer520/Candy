"""
experiment29.py — SDM 非叠加存储 (Kanerva) vs AR E2 叠加

用户指令: "试试" — 顺着 experiment28 结论 (3-gram 80.3% > AR E2 73.1%,
E2 是 2-gram 级转移统计) 试 SDM 稀疏分布式记忆。

机制 (Kanerva 1988, 生物约束内):
  - 硬地址: n_addrs 个固定随机稀疏位型 (不学习, 类比 DG 稀疏码)
  - 写入 (Hebbian 外积到就近槽): ctx 激活 top-k 最相似地址,
    content[a] += nxt 位型 (分散写入 → 冲突局部化, 替代 E2 全局叠加)
  - 读出: query 激活 top-k 地址 → 内容槽加权求和 → 块内位型内积 argmax
  - 无梯度/无BP/无连续信号/无批量/无偏置 (红线合规)

对照 (库50, experiment28): E2 teacher 73.1% / free 59.5%; 3-gram 80.3%。
规模: 库50 / 库100 各跑 teacher + free-run, 对比容量衰减。
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
    """稀疏分布式记忆: 就近写入 + 就近读出 (AR 位置块解码)"""

    def __init__(self, ar, n_addrs=4096, addr_ones=24, k=64, seed=7):
        self.ar = ar
        self.dim = ar.dim
        self.n = n_addrs
        self.k = k
        g = torch.Generator().manual_seed(seed)
        self.addr_idx = torch.stack([
            torch.randperm(self.dim, generator=g)[:addr_ones]
            for _ in range(n_addrs)]).to(DEVICE)  # (n_addrs, addr_ones) int
        self.content = torch.zeros(n_addrs, self.dim, device=DEVICE)

    def _sim(self, q):
        """q (稀疏实值) 与各地址的重叠计数"""
        idx = torch.nonzero(q).flatten()
        if idx.numel() == 0:
            return torch.zeros(self.n, device=DEVICE)
        mask = torch.zeros(self.dim, dtype=torch.bool, device=DEVICE)
        mask[idx] = True
        return mask[self.addr_idx].sum(1).float()

    def write(self, ctx, nxt):
        sim = self._sim(ctx)
        top = torch.topk(sim, self.k).indices
        self.content[top] += nxt                # Hebbian 就近叠加

    def read(self, query, t):
        sim = self._sim(query)
        top = torch.topk(sim, self.k).indices
        raw = self.content[top].sum(0)
        lo = (self.ar.max_pos_in + t) * self.ar.block_size
        rk = raw[lo:lo + self.ar.block_size]
        scores = self.ar.p[t] @ rk
        top2 = torch.topk(scores, 2)
        return int(top2.indices[0]), float(top2.values[0] - top2.values[1])


def run_scale(n_train, n_addrs, k):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    sdm = SDMem(ar, n_addrs=n_addrs, k=k)

    # 写入 (与 E2 相同的事件集: 每对话每前缀 t≥1)
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
    print(f"  SDM 写入完成: {n_train} 库, {n_addrs} 地址 k={k} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

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
                code, _ = sdm.read(query, t)
                tf_ok += (code == rc[t]); tf_tot += 1
            prefix = [rc[0]]
            for t in range(1, len(rc)):
                if t >= ar.max_pos_out:
                    break
                query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
                code, _ = sdm.read(query, t)
                if code is None:
                    break
                fr_ok += (code == rc[t]); fr_tot += 1
                prefix.append(code)
        print(f"  [{tag}] teacher {tf_ok}/{tf_tot} = "
              f"{tf_ok/max(tf_tot,1):.1%} | "
              f"free-run {fr_ok}/{fr_tot} = {fr_ok/max(fr_tot,1):.1%}",
              flush=True)


def main():
    print("=" * 66, flush=True)
    print("experiment29 — SDM 非叠加存储 vs AR E2 叠加", flush=True)
    print("=" * 66, flush=True)
    print("对照 (库50, experiment28/26): E2 teacher 73.1% / free 59.5%; "
          "3-gram 80.3%", flush=True)
    for n in (50, 100):
        print(f"\n[库{n}]", flush=True)
        run_scale(n, n_addrs=4096, k=64)


if __name__ == "__main__":
    main()
