"""
experiment17.py — 文献驱动架构升级 v14: 三因子资格迹 (A) + DG 稀疏分离 (B)

背景: 用户指令"上网调研学习一下, 不再盲目试错"。经文献调研
(SNN 信用分配 / 海马模式补全 / Hopfield 容量理论 / 海马序列生成),
已将 experiment16 记忆库扩容崩溃 (14→50 对话 69.8%→20.9%) 归因于
"稠密 Hebbian 关联记忆的串扰 (crosstalk)" + "即时 RPE 权重抖动"。
本实验用文献依据的两个升级做可证伪对照:

  H1 (资格迹, 文献: Gerstner & Lehmann 2018; E-prop):
     三因子资格迹 (e_ji ← λ·e_ji + pre×post; Δw = lr × M_j × e_ji)
     相比即时 RPE 调制 Hebbian, 在记忆库扩容时提升位置头回忆准确率。
     机制: 迹保留共激活时间历史, 权重更新平滑, 减少对话间干扰。
  H2 (稀疏分离, 文献: CLS Schapiro 2017; HiCL 2025):
     DG top-k 稀疏分离相比稠密状态, 降低记忆头串扰, 提升记忆库容量。
     机制: 不同状态 top-k 后支持集重叠期望降至 k²/N, Hebbian 更新互扰小。

对照矩阵 (2×2 + 稀疏度变体, 单变量判定):
  A0B0: 即时 Hebbian + 稠密状态   ← 基线 (= experiment16 位置头)
  A1B0: 资格迹(λ=0.9) + 稠密状态   → 判 H1
  A0B1: 即时 Hebbian + DG k=32     → 判 H2
  A1B1: 资格迹 + DG k=32
  A0B2: 即时 Hebbian + DG k=64    (稀疏度变体)

记忆库: N ∈ {14, 50, 100} (english_pairs_1000 前 N 对话)
指标: 解码 / 首字符 / 位置头回忆 (字符级) / 端到端 (前 min(14,N), 修正开)

★ 关键一致性: 启用 DG 分离时, 记忆头训练与评估必须用同一特征入口
  _mem_feature (内部做 _dg_separate), 逐位一致 (v12.3 快照同理)。
"""

import sys, os, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, train_w_h2o_stdp_gpu, DEVICE
from test_recurrent_learning import DIALOGUES

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HIDDEN = 256
INPUT_BIAS = 1.0
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


def run_config(dialogues, label, use_trace, use_dg, dg_k):
    n_dlg = len(dialogues)
    print(f"\n--- {label}  (迹={use_trace}, DG={use_dg}, k={dg_k}, 记忆库={n_dlg}) ---",
          flush=True)
    t0 = time.perf_counter()

    sim = RecurrentLIFSimulator(hidden_size=HIDDEN, output_size=8,
                                input_bias=INPUT_BIAS, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=3,
                                use_eligibility_trace=use_trace,
                                eligibility_lambda=0.9,
                                use_dg_separation=use_dg, dg_k=dg_k)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_codes = [ord(c) for c in BASIC_CHARS]

    # 1) 基础解码 + 渐进式深度训练 (与 experiment16 完全一致)
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=False)
    sim.train_multi_layer_stdp(train_codes, num_epochs=120, lr_layer=0.3,
                               lr_out=0.5, verbose=False, n_loops=1)
    n_ok = 0
    for c in train_codes:
        sim.reset_state()
        out = sim._multi_layer_forward(sim._get_char_code(chr(c)), n_loops=1)
        if sim.check_decode(out, c):
            n_ok += 1
    print(f"  [解码] {n_ok}/72", flush=True)

    # 2) W_ctx_to_first + 首字符 (库内, 快照恢复)
    sim.train_context_to_first(dialogues, lr=0.05, n_iter=400, n_loops=1)
    first_ok = first_total = 0
    for i, (inp, resp) in enumerate(dialogues):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, acc_state = sim.encode_text_lif(inp, update_memory=True, n_loops=1)
        pred = sim._binary_decode(sim.W_ctx_to_first, sim._mem_feature(acc_state))
        pred_code = sum((1 << j) for j in range(8) if pred[j] >= 0.5)
        first_total += 1
        if pred_code == rc[0]:
            first_ok += 1
    print(f"  [首字符] {first_ok}/{first_total}", flush=True)

    # 3) 位置记忆头 + 回忆 (库内)
    sim.train_pos_heads(dialogues, lr=0.05, n_iter=200, n_loops=1)
    ok = tot = 0
    for i, (inp, resp) in enumerate(dialogues):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, st = sim.encode_text_lif(inp, update_memory=True, n_loops=1)
        for k in range(len(rc)):
            mem_code, _ = sim.pos_head_recall(st, k)
            tot += 1
            if mem_code == rc[k]:
                ok += 1
    pos_acc = ok / max(tot, 1)
    print(f"  [位置头回忆] {ok}/{tot} ({pos_acc:.1%})", flush=True)

    # 4) 端到端生成 (前 min(14, n), 快照恢复, 修正开, oracle 长度)
    eval_n = min(14, n_dlg)
    fc = full = cchar = ctot = 0
    for i in range(eval_n):
        inp, resp = dialogues[i]
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        expected = resp
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, cf = sim.encode_text_lif(inp, update_memory=True, n_loops=1)
        if cf.sum().item() == 0:
            continue
        result = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                        update_memory=True, use_pos_memory=True,
                                        n_loops=1)
        if result and expected and result[0] == expected[0]:
            fc += 1
        for a, b in zip(result, expected):
            ctot += 1
            if a == b:
                cchar += 1
        if result == expected:
            full += 1
    print(f"  [端到端] 首字符 {fc}/{eval_n}, 完整 {full}/{eval_n}, 字符级 "
          f"{cchar}/{max(ctot,1)} ({cchar/max(ctot,1):.1%})", flush=True)
    print(f"  耗时: {time.perf_counter()-t0:.0f}s", flush=True)
    return dict(decode=n_ok, first=first_ok, first_tot=first_total,
                pos=pos_acc, full=full, cchar=cchar/max(ctot,1), eval_n=eval_n)


def main():
    print("=" * 72, flush=True)
    print("experiment17 — v14 文献升级: 三因子资格迹(A) × DG 稀疏分离(B)", flush=True)
    print("对照矩阵 5 配置 × 记忆库 {14, 50, 100}; 数据: english_pairs_1000", flush=True)
    print("=" * 72, flush=True)

    results = {}
    runs = [
        # (规模, 对话, 标签, 迹, DG, k)
        ("库14", DIALOGUES,        "A0B0 基线   ", False, False, 0),
        ("库14", DIALOGUES,        "A1B0 迹     ", True,  False, 0),
        ("库14", DIALOGUES,        "A0B1 DG32   ", False, True,  32),
        ("库14", DIALOGUES,        "A1B1 迹+DG32", True,  True,  32),
        ("库14", DIALOGUES,        "A0B2 DG64   ", False, True,  64),
        ("库50", load_pairs("english_pairs_1000.txt", 50), "A0B0 基线   ", False, False, 0),
        ("库50", load_pairs("english_pairs_1000.txt", 50), "A1B0 迹     ", True,  False, 0),
        ("库50", load_pairs("english_pairs_1000.txt", 50), "A0B1 DG32   ", False, True,  32),
        ("库50", load_pairs("english_pairs_1000.txt", 50), "A1B1 迹+DG32", True,  True,  32),
        ("库50", load_pairs("english_pairs_1000.txt", 50), "A0B2 DG64   ", False, True,  64),
        ("库100", load_pairs("english_pairs_1000.txt", 100), "A0B0 基线  ", False, False, 0),
        ("库100", load_pairs("english_pairs_1000.txt", 100), "A1B0 迹    ", True,  False, 0),
        ("库100", load_pairs("english_pairs_1000.txt", 100), "A0B1 DG32  ", False, True,  32),
        ("库100", load_pairs("english_pairs_1000.txt", 100), "A1B1 迹+DG32", True,  True,  32),
        ("库100", load_pairs("english_pairs_1000.txt", 100), "A0B2 DG64  ", False, True,  64),
    ]
    for scope, dlg, label, trace, dg, k in runs:
        r = run_config(dlg, label, trace, dg, k)
        results[(scope, label.strip())] = r

    print(f"\n{'='*72}", flush=True)
    print(f"汇总 (位置头回忆 / 端到端完整 / 字符级):", flush=True)
    print(f"{'配置':<18} {'库14':>20} {'库50':>20} {'库100':>20}", flush=True)
    for label in ["A0B0 基线", "A1B0 迹", "A0B1 DG32", "A1B1 迹+DG32", "A0B2 DG64"]:
        row = []
        for scope in ["库14", "库50", "库100"]:
            r = results.get((scope, label))
            if r:
                row.append(f"{r['pos']:.0%}/{r['full']}/{r['cchar']:.0%}")
            else:
                row.append("-")
        print(f"{label:<18} {row[0]:>20} {row[1]:>20} {row[2]:>20}", flush=True)


if __name__ == "__main__":
    main()
