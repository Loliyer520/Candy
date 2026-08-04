"""
experiment15.py — 8 层隐藏层 + 自回归循环验证 (v13.1)

用户方向: "隐藏层扩展到 8 层, 每次输入后自回归循环一次再传入输入"
机制 (已实现于 lif_pytorch.py, 参数化 n_loops):
  - num_layers=8: 渐进式训练自动扩展到 8 阶段 (每阶段恒等初始化)
  - n_loops=2: 输入 → 8 层 → 输出1 → (输出1 作为输入) → 8 层 → 输出2
    (轮次间 V 膜电位继续累积, 不 reset)

配置矩阵 (对比):
  1. L=4,  loops=1  — 现有 v13 基线
  2. L=8,  loops=1  — 纯 8 层 (无循环)
  3. L=8,  loops=2  — 8 层 + 自回归循环一次 (用户指定)

每配置指标: 解码准确率 / 首字符 14/14 / 位置头回忆 / 端到端修正字符级
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

HIDDEN_SIZE = 256
INPUT_BIAS = 1.0
BASIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"


def run_config(num_layers, n_loops, label):
    print(f"\n{'#'*72}", flush=True)
    print(f"# 配置: {label}  (num_layers={num_layers}, n_loops={n_loops})", flush=True)
    print(f"{'#'*72}", flush=True)
    t0 = time.perf_counter()

    sim = RecurrentLIFSimulator(hidden_size=HIDDEN_SIZE, output_size=8,
                                input_bias=INPUT_BIAS, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=num_layers)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_codes = [ord(c) for c in BASIC_CHARS]

    # 1) 基础解码 + 渐进式深度训练 (训练阶段与推理同 n_loops)
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=False)
    sim.train_multi_layer_stdp(train_codes, num_epochs=120, lr_layer=0.3,
                               lr_out=0.5, verbose=False, n_loops=n_loops)
    n_ok = 0
    for c in train_codes:
        ch = chr(c) if 32 <= c <= 126 else '?'
        sim.reset_state()
        out = sim._multi_layer_forward(sim._get_char_code(ch), n_loops=n_loops)
        if sim.check_decode(out, c):
            n_ok += 1
    print(f"  [解码] {n_ok}/{len(train_codes)}", flush=True)

    # 2) W_ctx_to_first + 首字符评估 (v12.3 快照恢复)
    sim.train_context_to_first(DIALOGUES, lr=0.05, n_iter=400, n_loops=n_loops)
    first_ok = first_total = 0
    for i, (inp, resp) in enumerate(DIALOGUES):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, acc_state = sim.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
        pred = sim._binary_decode(sim.W_ctx_to_first, acc_state)
        pred_code = sum((1 << j) for j in range(8) if pred[j] >= 0.5)
        first_total += 1
        if pred_code == rc[0]:
            first_ok += 1
    print(f"  [首字符] {first_ok}/{first_total}", flush=True)

    # 3) 位置记忆头 + 回忆准确率
    sim.train_pos_heads(DIALOGUES, lr=0.05, n_iter=200, n_loops=n_loops)
    ok = tot = 0
    for i, (inp, resp) in enumerate(DIALOGUES):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, st = sim.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
        for k in range(len(rc)):
            mem_code, _ = sim.pos_head_recall(st, k)
            tot += 1
            if mem_code == rc[k]:
                ok += 1
    print(f"  [位置头回忆] {ok}/{tot} ({ok/max(tot,1):.1%})", flush=True)

    # 4) 端到端生成 (快照恢复, 修正开) — oracle 长度
    fc = full = cchar = ctot = 0
    for i, (inp, resp) in enumerate(DIALOGUES):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        expected = resp
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, cf = sim.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
        if cf.sum().item() == 0:
            continue
        result = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                        update_memory=True, use_pos_memory=True,
                                        n_loops=n_loops)
        if result and expected and result[0] == expected[0]:
            fc += 1
        for a, b in zip(result, expected):
            ctot += 1
            if a == b:
                cchar += 1
        if result == expected:
            full += 1
    print(f"  [端到端] 首字符 {fc}/14, 完整 {full}/14, 字符级 {cchar}/{max(ctot,1)} "
          f"({cchar/max(ctot,1):.1%})", flush=True)
    print(f"  耗时: {time.perf_counter()-t0:.0f}s", flush=True)
    return dict(decode=n_ok, first=first_ok, pos=ok/max(tot,1),
                full=full, cchar=cchar/max(ctot,1), ctot=ctot)


def main():
    print(f"{'='*72}", flush=True)
    print(f"experiment15 — 8 层 + 自回归循环 (n_loops) 验证", flush=True)
    print(f"约束: RPE 调制 Hebbian / 纯二值阈值 / 无梯度 / 无批量优化", flush=True)
    print(f"{'='*72}", flush=True)
    results = {}
    results["L4_loops1"] = run_config(4, 1, "4 层基线 (v13)")
    results["L8_loops1"] = run_config(8, 1, "纯 8 层")
    results["L8_loops2"] = run_config(8, 2, "8 层 + 自回归循环一次 ★")

    print(f"\n{'='*72}", flush=True)
    print(f"汇总:", flush=True)
    print(f"{'配置':<16} {'解码':>6} {'首字符':>8} {'位置头':>8} {'完整':>6} {'字符级':>8}",
          flush=True)
    for key, r in results.items():
        print(f"{key:<16} {r['decode']:>5}/72 {r['first']:>5}/14 "
              f"{r['pos']:>8.1%} {r['full']:>5}/14 {r['cchar']:>8.1%}",
              flush=True)


if __name__ == "__main__":
    main()
