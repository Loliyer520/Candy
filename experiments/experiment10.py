"""
experiment10.py — v12.4 状态判别性修复的验证记录 (原 diag_ctx_discrim)

背景: v12.3 下 W_ctx_to_first 首字符预测上限 6/14, 瓶颈 = 状态判别性
      (两两余弦 0.971)。本实验在 v12.3 快照方案下分解状态信号:
      state = max(v_peak, recall)

回答三个问题 (假说演绎法 — 先复现, 再定位根因):
  [Q1] recall 是否被饱和的 v_peak 覆盖 (max 后 recall 的贡献占比)?
  [Q2] state / v_peak-only / recall-only 三组信号的判别性 (两两余弦) 谁更高?
  [Q3] 分别用多组信号训练独立 W_ctx_to_first → 准确率?
       → 定位判别性损失发生在哪一层

结论 (实测):
  - recall > v_peak 占比 69.2% → recall 主导 state, 未被 max 吞掉
  - 二值化 (raw > 0.5) 是判别性杀手: 独立 W 仅 5/14
  - v_peak 分级 / 分级回忆 min-max / raw/128 / min(raw/128,1) → 全部 14/14
  - 直接 clamp(raw, 0, 1) 失败 (4-5/14): raw 量级常 > 1, 直接钳位大量饱和
  - ★ 修复: recall = clamp(raw / (H/2), 0, 1) → Step 2 首字符 6/14 → 14/14

★ 诊断指标 (余弦/统计) 仅用于分析, 不进入网络机制 (网络内禁止余弦检索)。
"""

import sys, os, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, train_w_h2o_stdp_gpu, DEVICE
from test_recurrent_learning import DIALOGUES  # 复用测试数据 (模块级定义)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HIDDEN_SIZE = 256
INPUT_BIAS = 1.0
BASIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"


def cos_pairwise(M):
    """M: (n, dim) 行向量 → 两两余弦矩阵 (仅诊断用)"""
    M = M / (M.norm(dim=1, keepdim=True) + 1e-12)
    return M @ M.T


def train_fresh_W(sim, signals, labels, lr=0.05, n_iter=500):
    """独立训练一个 W_ctx_to_first (奖赏调制 Hebbian), 返回在 signals 上的准确率"""
    n = len(labels)
    W = torch.empty(8, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
    W.uniform_(-0.1, 0.1)
    tgt = torch.tensor([[float((l >> j) & 1) for j in range(8)] for l in labels],
                       dtype=torch.float32, device=DEVICE)
    for _ in range(n_iter):
        idx = list(range(n))
        random.shuffle(idx)
        for k in idx:
            pred = sim._binary_decode(W, signals[k])
            rpe = tgt[k] - pred
            for j in range(8):
                W[j] += lr * rpe[j] * signals[k]
            W.clamp_(-10.0, 10.0)
    ok = sum(1 for k in range(n)
             if (sim._binary_decode(W, signals[k]) == tgt[k]).all().item())
    return ok


def main():
    t0 = time.perf_counter()

    # ---- 复现 Step 1 + Step 1.5 + Step 2 (与 test_recurrent_learning 完全一致) ----
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
    print(f"复现训练完成: {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- 快照重放, 逐字符复刻 encode_text_lif, 收集 state/v_peak/recall/raw_recall ----
    states, v_peaks, recalls, raw_recalls, labels = [], [], [], [], []
    for i, (inp, resp) in enumerate(DIALOGUES):
        resp_codes = sim._text_to_codes(resp)
        if not resp_codes:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        sim.reset_state()
        sim.reset_memory()
        for ch in [c for c in inp if 32 <= ord(c) <= 126]:
            vec = sim._char_to_8bit(ch) + sim.input_bias
            output = sim._multi_layer_forward(vec)
            sim.update_coactivation(output)
            raw = torch.mv(sim.W_coact, output)          # 未阈值化的回忆 (连续强度)
            recall = (raw > 0.5).float()                  # 二值回忆 (网络实际用)
            v_peak = sim.V_deep[-1]
            sim.MemWork = torch.max(v_peak, recall)
        states.append(sim.MemWork)
        v_peaks.append(v_peak)
        recalls.append(recall)
        raw_recalls.append(raw)
        labels.append(resp_codes[0])

    states = torch.stack(states)
    v_peaks = torch.stack(v_peaks)
    recalls = torch.stack(recalls)
    raw_recalls = torch.stack(raw_recalls)
    n = len(labels)

    print(f"对话数: {n}")

    # [Q1] recall 贡献占比
    contrib = (recalls > v_peaks).float().mean(dim=1)
    print(f"\n[Q1] recall 被 max 覆盖程度")
    print(f"  recall > v_peak 平均占比: {contrib.mean().item():.1%} "
          f"(范围 {contrib.min().item():.1%}~{contrib.max().item():.1%})")
    eq = (states == v_peaks).all(dim=1).sum().item()
    print(f"  state 与 v_peak 逐位完全相同的对话: {eq}/{n} (max 完全吞掉 recall)")

    # [Q2] 判别性
    print(f"\n[Q2] 三组信号的两两余弦 (越低越有判别性)")
    for name, M in [("state  ", states), ("v_peak ", v_peaks), ("recall ", recalls)]:
        C = cos_pairwise(M)
        off = C[~torch.eye(n, dtype=torch.bool)].cpu().numpy()
        act = (M > 0.5).float().sum(dim=1)
        print(f"  {name}: 两两余弦均 {off.mean():.3f} (min {off.min():.3f} max {off.max():.3f}), "
              f"活性 {act.min().item():.0f}~{act.max().item():.0f}/256")

    # [Q3] 独立 W 准确率 — 扩展变体: 分级 recall 组合。
    #   注意: min-max 归一化仅诊断可用 (需全序列统计), 网络内需逐点可计算方案。
    print(f"\n[Q3] 独立 W_ctx_to_first 在多种信号上的准确率 (n_iter=500)")
    raw_n = (raw_recalls - raw_recalls.min(dim=1, keepdim=True).values) / \
            (raw_recalls.max(dim=1, keepdim=True).values -
             raw_recalls.min(dim=1, keepdim=True).values + 1e-12)
    raw_c = torch.clamp(raw_recalls, 0.0, 1.0)   # 逐点: 钳位到 [0,1] (与 V 处理一致)
    raw_s = raw_recalls / 128.0                   # 逐点: 固定缩放 (活性 ~50%=128)
    variants = [
        ("state  (max(v_peak, bin_recall))", states),
        ("v_peak-only", v_peaks),
        ("bin_recall-only", recalls),
        ("raw_recall-only (min-max 归一化)", raw_n),
        ("max(v_peak, raw 归一化)", torch.max(v_peaks, raw_n)),
        ("clamp(raw,0,1)-only", raw_c),
        ("max(v_peak, clamp(raw,0,1))", torch.max(v_peaks, raw_c)),
        ("raw/128-only", raw_s),
        ("max(v_peak, raw/128)", torch.max(v_peaks, raw_s)),
        ("min(raw/128,1)-only", torch.min(raw_s, torch.ones_like(raw_s))),
        ("max(v_peak, min(raw/128,1))",
         torch.max(v_peaks, torch.min(raw_s, torch.ones_like(raw_s)))),
    ]
    for name, M in variants:
        ok = train_fresh_W(sim, M, labels)
        print(f"  {name}: {ok}/{n} ({ok/n:.0%})")

    # 对照: 训练用 W 在 state 上的准确率 (Step 2 评估)
    tgt = torch.tensor([[float((l >> j) & 1) for j in range(8)] for l in labels],
                       dtype=torch.float32, device=DEVICE)
    ok0 = sum(1 for k in range(n)
              if (sim._binary_decode(sim.W_ctx_to_first, states[k]) == tgt[k]).all().item())
    print(f"\n对照 (train_context_to_first 的 W 在 state 上, 即 Step 2 评估): {ok0}/{n}")
    print(f"诊断完成: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
