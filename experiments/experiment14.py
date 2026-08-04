"""
experiment14.py — 位置记忆头 + 修正机制诊断 (v13)

背景: W_ctx_to_first (首字符) 在 14 对话 memorize 场景可靠 (14/14,
      experiment10); W_seq 结构性不可解 (experiment11: 5.4-11.5% 字符级);
      异联想联想链 W_trans 受 bigram 混合限制 (experiment13: 3-8%)。

用户方向: "增加记忆层, 对非首字结果进行修正"。
方案: 位置记忆头 W_ctx_to_pos[k] (上下文状态 → 回复第 k 字符, RPE Hebbian),
      生成时 margin = min_j|raw_j| 超过门控阈值 → 覆盖 W_seq 候选。

本实验回答三个问题:
  Q1 位置头回忆本身多准? (逐位置准确率 / 加权平均)
  Q2 margin 是否有判别力? (回忆正确 vs 错误样本的 margin 分离度)
  Q3 端到端修正收益随门控阈值 θ 如何变化? (θ 扫描对照 W_seq 基线)
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


def main():
    t0 = time.perf_counter()
    sim = RecurrentLIFSimulator(hidden_size=HIDDEN_SIZE, output_size=8,
                                input_bias=INPUT_BIAS, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=4)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_codes = [ord(c) for c in BASIC_CHARS]
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=False)
    sim.train_multi_layer_stdp(train_codes, num_epochs=200, lr_layer=0.3,
                               lr_out=0.5, verbose=False)
    sim.train_context_to_first(DIALOGUES, lr=0.05, n_iter=500)
    # W_seq 基线 (缩短迭代, 足够体现"修正"对照; test 用 1000)
    sim.train_sequence(DIALOGUES, lr=0.5, n_iter=300)
    print(f"基础训练完成: {time.perf_counter()-t0:.1f}s", flush=True)

    # 1) 快照状态收集 (与 test Step 2 一致)
    states, resp_lens = [], []
    for i, (inp, resp) in enumerate(DIALOGUES):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, st = sim.encode_text_lif(inp, update_memory=True)
        if st.sum().item() == 0:
            continue
        states.append((st, rc))
        resp_lens.append(len(rc))
    n_dlg = len(states)
    max_len = max(resp_lens)
    print(f"对话数: {n_dlg}, 回复长度 {min(resp_lens)}-{max_len}", flush=True)

    # 2) 训练位置记忆头 (与 test Step 2.5 同一实现路径)
    sim.train_pos_heads(DIALOGUES, lr=0.05, n_iter=500)
    print(f"位置头训练完成: {len(sim.W_ctx_to_pos)} 位置", flush=True)

    # 3) Q1 逐位置准确率 + Q2 margin 判别力
    print(f"\n{'位置':>4} {'样本':>4} {'准确率':>8} {'margin_ok':>10} {'margin_bad':>10}",
          flush=True)
    all_ok, all_bad = [], []
    per_pos_ok = per_pos_tot = 0
    for k in range(max_len):
        data = [(st, rc[k]) for st, rc in states if len(rc) > k]
        if not data:
            break
        acc = 0
        for st, code in data:
            mem_code, margin = sim.pos_head_recall(st, k)
            if mem_code == code:
                acc += 1
                all_ok.append(margin)
            else:
                all_bad.append(margin)
        acc /= len(data)
        per_pos_ok += acc * len(data)
        per_pos_tot += len(data)
        m_ok = f"{np.mean(all_ok):.2f}" if all_ok else "-"
        m_bad = f"{np.mean(all_bad):.2f}" if all_bad else "-"
        print(f"{k:>4} {len(data):>4} {acc:>8.0%} {m_ok:>10} {m_bad:>10}",
              flush=True)
    print(f"加权平均: {per_pos_ok/per_pos_tot:.1%}", flush=True)
    ok_a = np.array(all_ok)
    bad_a = np.array(all_bad)
    print(f"\nmargin 分布 (回忆正确 {len(ok_a)} 样本 vs 错误 {len(bad_a)}):", flush=True)
    print(f"  correct: min={ok_a.min():.2f} p25={np.percentile(ok_a,25):.2f} "
          f"med={np.median(ok_a):.2f} p75={np.percentile(ok_a,75):.2f} max={ok_a.max():.2f}",
          flush=True)
    if len(bad_a):
        print(f"  wrong:   min={bad_a.min():.2f} p25={np.percentile(bad_a,25):.2f} "
              f"med={np.median(bad_a):.2f} p75={np.percentile(bad_a,75):.2f} max={bad_a.max():.2f}",
              flush=True)

    # 4) Q3 端到端修正: θ 扫描
    print(f"\n端到端生成对照 (14 训练对话, 快照恢复, oracle 步数):", flush=True)
    print(f"{'θ':>6} {'首字符':>8} {'字符级':>8} {'完整':>6} {'字符数':>6}", flush=True)
    results = {}
    for theta in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        fc = full = cchar = ctot = nchar = 0
        for i, (inp, resp) in enumerate(DIALOGUES):
            rc = sim._text_to_codes(resp)
            if not rc:
                continue
            expected = resp
            sim.W_coact = sim._coact_snapshots[i].clone()
            _, cf = sim.encode_text_lif(inp, update_memory=True)
            if cf.sum().item() == 0:
                continue
            result = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                            update_memory=True, use_pos_memory=True,
                                            pos_margin_thresh=theta)
            nchar += len(result)
            if result and expected and result[0] == expected[0]:
                fc += 1
            for a, b in zip(result, expected):
                ctot += 1
                if a == b:
                    cchar += 1
            if result == expected:
                full += 1
        char_acc = cchar / max(ctot, 1)
        results[theta] = char_acc
        print(f"{theta:>6} {fc:>5}/14 {char_acc:>8.1%} {full:>4}/14 {nchar:>6}",
              flush=True)
    best = max(results, key=results.get)
    print(f"\n最佳 θ = {best} (字符级 {results[best]:.1%}); "
          f"W_seq 基线 39.7% (test 快照一致), 5.4% (experiment11 理想态)",
          flush=True)
    print(f"\n总耗时: {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
