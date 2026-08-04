"""
experiment22.py — 自回归表示参数扫描 (块大小 × 位型强度)

experiment21 诊断结论: teacher forcing 单步上限 73.1% (库内) / 27.7%
(库外), 误差累积损失仅 13.6% → 主瓶颈是关联存储表示, 非生成路径。

扫描表示参数, 找 teacher 单步上限的硬顶:
  - dim 12288 (块 192) → 16384 (块 256, P1 甜点)
  - char_ones 8 → 16 (位型信号强度)
基线 (dim=12288, ones=8): 库内 teacher 73.1% / free 59.5%; 库外 27.7% / 21.1%。

纯事件记忆秒级构建, 查询 CPU ~0.2-0.3s/步, 4 配置 ≈ 10 分钟。
"""

import sys, os, time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from episodic_memory import AutoRegressiveEventMemory


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


def run(mem, dlg):
    """返回 (teacher_ok, teacher_tot, free_ok, free_tot)"""
    t_ok = t_tot = f_ok = f_tot = 0
    for inp, resp in dlg:
        rc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if not rc or not ic:
            continue
        for t in range(1, len(rc)):
            code, _ = mem.next_char(ic, rc[:t])
            if code is None:
                break
            t_ok += (code == rc[t]); t_tot += 1
        prefix = [rc[0]]
        for t in range(1, len(rc)):
            code, _ = mem.next_char(ic, prefix)
            if code is None:
                break
            f_ok += (code == rc[t]); f_tot += 1
            prefix.append(code)
    return t_ok, t_tot, f_ok, f_tot


def main():
    print("=" * 70, flush=True)
    print("experiment22 — 自回归表示参数扫描 (块大小 × 位型强度)", flush=True)
    print("=" * 70, flush=True)

    train_dlg = load_pairs("english_pairs_1000.txt", 50)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=50)
    in14 = train_dlg[:14]

    configs = [
        ("dim=16384/ones=8 (块256)", 16384, 8),
        ("dim=12288/ones=16", 12288, 16),
        ("dim=16384/ones=16", 16384, 16),
    ]
    print(f"  基线 dim=12288/ones=8: 库内 teacher 73.1% / free 59.5%; "
          f"库外 27.7% / 21.1%", flush=True)

    for tag, dim, ones in configs:
        t0 = time.perf_counter()
        mem = AutoRegressiveEventMemory(dim=dim, char_ones=ones,
                                        max_pos_in=32, max_pos_out=32, seed=7)
        for inp, resp in train_dlg:
            oc = text_to_codes(resp)
            ic = text_to_codes(inp)
            if oc and ic:
                mem.store_dialogue(ic, oc)
        ti_ok, ti_tot, fi_ok, fi_tot = run(mem, in14)
        to_ok, to_tot, fo_ok, fo_tot = run(mem, out_dlg)
        print(f"\n[{tag}]  (构建 {time.perf_counter()-t0:.0f}s)", flush=True)
        print(f"  库内: teacher {ti_ok}/{ti_tot} = {ti_ok/max(ti_tot,1):.1%}  "
              f"free {fi_ok}/{fi_tot} = {fi_ok/max(fi_tot,1):.1%}", flush=True)
        print(f"  库外: teacher {to_ok}/{to_tot} = {to_ok/max(to_tot,1):.1%}  "
              f"free {fo_ok}/{fo_tot} = {fo_ok/max(fo_tot,1):.1%}", flush=True)


if __name__ == "__main__":
    main()
