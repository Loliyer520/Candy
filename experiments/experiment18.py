"""
experiment18.py — P1 事件联合记忆验证 (v14)

对照基线 = experiment17 A0B2 (DG64 位置头, 同环境已跑):
   位置头回忆字符级: 库14 100% / 库50 35% / 库100 16%
本实验: P1 事件联合记忆 (episodic_memory.py), 同 N=14/50/100。

假说 (可证伪, 文献驱动):
  H1 (容量): P1 在 N=50/100 的整串回忆准确率显著高于基线 35%/16%。
     机制: 稀疏高维 (4096) 事件码互不干扰, 串扰随维度降低。
  H2 (绑定): 完整句匹配率高于逐字符独立预测的期望 —
     顺序编入位置排列, 整体补全, 无逐字符联合目标问题。

评估 (与 experiment17 同口径):
  - 回忆: 输入上下文 → P1 整体补全 → 与目标回复逐字符/整串比较
  - 上下文特征: 每对话 W_coact 归零 (v_peak 累积) 的 256 维 acc_state
    (P1 自带 4096 维投影判别层, 不依赖 LIF 记忆训练链)

★ 红线检查: 存储 = Hebbian 外积一次性写入 (imprinting, 无 RPE);
  解码 = 固定位型内积 argmax (联想匹配); 无梯度/无BP/无连续信号/无批量。
"""

import sys, os, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, DEVICE
from episodic_memory import EpisodicEventMemory
from test_recurrent_learning import DIALOGUES

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HIDDEN = 256
BASIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"


def load_pairs(path, n):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            inp, resp = line.split("\t", 1)
            pairs.append((inp.strip(), resp.strip()))
            if len(pairs) >= n:
                break
    return pairs


def make_sim():
    sim = RecurrentLIFSimulator(hidden_size=HIDDEN, output_size=8,
                                input_bias=1.0, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=3)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    return sim


def run_P1(dialogues, dim, max_pos, label):
    n_dlg = len(dialogues)
    t0 = time.perf_counter()
    print(f"\n--- {label}  (P1 v3 块绑定, D={dim}, max_pos={max_pos}, 库={n_dlg}) ---",
          flush=True)
    sim = make_sim()
    pairs = []
    for inp, resp in dialogues:
        in_codes = sim._text_to_codes(inp)
        out_codes = sim._text_to_codes(resp)
        if not in_codes or not out_codes:
            continue
        pairs.append((in_codes, out_codes))
    mem = EpisodicEventMemory(dim=dim, char_ones=8, max_pos=max_pos, seed=7)

    # 存储: 每对话一次性写入 (输入序列 ↔ 回复序列 联想绑定)
    for in_c, out_c in pairs:
        mem.store(in_c, out_c)
    print(f"  存储 {mem._n_events} 事件 (E 非零占比 "
          f"{(mem.E != 0).float().mean():.1%}), 存储耗时 {time.perf_counter()-t0:.0f}s",
          flush=True)

    # 回忆: 库内整体补全
    c_ok = c_tot = 0
    full_ok = 0
    trunc = 0
    for in_c, out_c in pairs:
        if len(out_c) > max_pos:
            trunc += 1
        pred = mem.recall(in_c, len(out_c))
        if pred == out_c:
            full_ok += 1
        for a, b in zip(pred, out_c):
            c_tot += 1
            if a == b:
                c_ok += 1
    print(f"  [回忆] 完整句 {full_ok}/{len(pairs)}, 字符级 {c_ok}/{c_tot} "
          f"({c_ok/max(c_tot,1):.1%}), 超长截断 {trunc}/{len(pairs)}", flush=True)
    print(f"  耗时: {time.perf_counter()-t0:.0f}s", flush=True)
    return dict(full=full_ok, cchar=c_ok/max(c_tot,1), n=len(pairs))


def main():
    print("=" * 72, flush=True)
    print("experiment18 — P1 事件联合记忆 (同时解无限上下文 + 时间序列)", flush=True)
    print("基线(exp17 A0B2 DG64): 库14 100% / 库50 35% / 库100 16%", flush=True)
    print("=" * 72, flush=True)

    runs = [
        # (对话, dim, max_pos, 标签)
        (DIALOGUES,                                    8192, 32, "B256 M32 库14"),
        (DIALOGUES,                                    12288, 48, "B256 M48 库14"),
        (load_pairs("english_pairs_1000.txt", 50),     8192, 32, "B256 M32 库50"),
        (load_pairs("english_pairs_1000.txt", 100),    8192, 32, "B256 M32 库100"),
        (load_pairs("english_pairs_1000.txt", 200),    8192, 32, "B256 M32 库200"),
        (load_pairs("english_pairs_1000.txt", 50),     12288, 48, "B256 M48 库50"),
        (load_pairs("english_pairs_1000.txt", 100),    12288, 48, "B256 M48 库100"),
    ]
    results = []
    for dlg, dim, mp, label in runs:
        results.append((label, run_P1(dlg, dim, mp, label)))

    print(f"\n{'='*72}", flush=True)
    print("汇总 (P1 完整句 / 字符级 vs 基线 位置头DG64 字符级):", flush=True)
    for label, r in results:
        scope = label.split()[-1]
        base = {"库14": "100%", "库50": "35%", "库100": "16%",
                "库200": "≈14%"}[scope]
        print(f"  {label:<14} P1: {r['full']}/{r['n']} 完整, "
              f"{r['cchar']:.1%} 字符   (基线 {base})", flush=True)


if __name__ == "__main__":
    main()
