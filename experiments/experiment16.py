"""
experiment16.py — 三层 + 记忆库扩容 (超大记忆) 验证 (v13.1)

用户方向: "改为三层加循环加超大记忆"

★ 循环结论 (前置, experiment15/16 v1): n_loops=2 在 L3/L8 均崩溃 (解码 1/72,
  根因: 第二轮输入依赖未训练输出 → RPE Hebbian 训练目标不稳定 + 判别性
  逐轮压缩)。循环在本 0-1 膜电位系统是死路, 本实验不再含循环配置。

本实验聚焦其余两个要素:
- 三层: num_layers=3 (浅层, 基线 71.9%)
- 超大记忆: 位置记忆头记忆库扩容 — english_pairs_1000 前 N 个对话,
  测记忆库规模 N ∈ {14, 50, 100} 时位置头回忆/端到端如何变化
  (experiment12 先例: W_ctx_to_first 在 50+ 规模从 100% 崩到 10-18%,
  本实验验证位置记忆头在大记忆库下的可扩展性)

每配置: 解码 / 首字符 / 位置头回忆 / 端到端修正 (均对库内对话评估)
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


def run_config(dialogues, label):
    num_layers, n_loops = 3, 1
    n_dlg = len(dialogues)
    print(f"\n{'#'*72}", flush=True)
    print(f"# 配置: {label}  (L={num_layers}, loops={n_loops}, 记忆库={n_dlg})", flush=True)
    print(f"{'#'*72}", flush=True)
    t0 = time.perf_counter()

    sim = RecurrentLIFSimulator(hidden_size=HIDDEN, output_size=8,
                                input_bias=INPUT_BIAS, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=num_layers)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_codes = [ord(c) for c in BASIC_CHARS]

    # 1) 基础解码 + 渐进式深度训练
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

    # 2) W_ctx_to_first + 首字符 (库内评估)
    sim.train_context_to_first(dialogues, lr=0.05, n_iter=400, n_loops=n_loops)
    first_ok = first_total = 0
    for i, (inp, resp) in enumerate(dialogues):
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

    # 3) 位置记忆头 + 回忆 (库内)
    sim.train_pos_heads(dialogues, lr=0.05, n_iter=200, n_loops=n_loops)
    ok = tot = 0
    for i, (inp, resp) in enumerate(dialogues):
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

    # 4) 端到端生成 (库内前 min(14, n) 个对话, 快照恢复, 修正开, oracle 长度)
    eval_n = min(14, n_dlg)
    fc = full = cchar = ctot = 0
    for i in range(eval_n):
        inp, resp = dialogues[i]
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
    print(f"  [端到端] 首字符 {fc}/{eval_n}, 完整 {full}/{eval_n}, 字符级 "
          f"{cchar}/{max(ctot,1)} ({cchar/max(ctot,1):.1%})", flush=True)
    print(f"  耗时: {time.perf_counter()-t0:.0f}s", flush=True)
    return dict(decode=n_ok, first=first_ok, first_tot=first_total,
                pos=ok/max(tot,1), full=full, cchar=cchar/max(ctot,1),
                eval_n=eval_n)


def main():
    print(f"{'='*72}", flush=True)
    print(f"experiment16 — 三层 + 超大记忆 (记忆库扩容)", flush=True)
    print(f"数据源: english_pairs_1000.txt (前 N 对话为记忆库)", flush=True)
    print(f"{'='*72}", flush=True)
    results = {}
    results["库14 (DIALOGUES)"] = run_config(DIALOGUES, "14 主对话基线")
    pairs50 = load_pairs("english_pairs_1000.txt", 50)
    results["库50"] = run_config(pairs50, "超大记忆库 50")
    pairs100 = load_pairs("english_pairs_1000.txt", 100)
    results["库100"] = run_config(pairs100, "超大记忆库 100")

    print(f"\n{'='*72}", flush=True)
    print(f"汇总:", flush=True)
    print(f"{'配置':<18} {'解码':>6} {'首字符':>10} {'位置头':>8} {'完整':>8} {'字符级':>8}",
          flush=True)
    for key, r in results.items():
        print(f"{key:<18} {r['decode']:>5}/72 {r['first']:>4}/{r['first_tot']} "
              f"{r['pos']:>8.1%} {r['full']:>4}/{r['eval_n']} {r['cchar']:>8.1%}",
              flush=True)


if __name__ == "__main__":
    main()
