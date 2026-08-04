"""
experiment12.py — W_ctx_to_first 规模泛化验证 (v12.5 探查)

问题: v12.4 下 W_ctx_to_first 首字符预测 14/14 (100%), 但仅 14 个对话 —
      是真实学习还是过拟合?

方法: 复用 Step 1 (W_h2o) + Step 1.5 (W_deep 渐进四层) 训练 (与对话集无关),
      仅对 english_pairs_1000.txt 的不同规模子集训练 W_ctx_to_first
      (train_context_to_first, lr=0.05, n_iter=500):
        N ∈ {14, 50, 100, 200}
      每个子集前重置 W_coact (清零) 与 W_ctx_to_first (重初始化),
      评估用 v12.3 确定性快照重放 (与 test Step 2 一致)。

对照: 多数类基线 (预测最频繁首字符) — 若准确率显著高于基线且随规模
      缓慢下降 → 真实学习; 若崩塌到基线附近 → 过拟合。

结论 (实测):
  - N=14: 14/14 (100%) vs 基线 42.9% — 显著"学会"
  - N=50/100/200: 18%/13%/10% vs 基线 28%/26%/23% — **被多数类基线反超**
  - 状态两两余弦随规模升至 0.994-0.995 (14 规模 0.990) — 判别性崩塌
  - ★ 结论: 14/14 是过拟合 (memorization), 非真实泛化
  - ★ 与 experiment11 交叉印证: 系统级"逐 bit 独立决策 vs 8-bit 联合目标"
    结构性限制 — 位级学习存在 (~74%), 但 8-bit 联合全对崩溃 (0.74^8≈10%)
  - 位级 vs 联合: 评估按 8-bit 联合时, 学习的联合准确率 = 位准确率乘积,
    天然 < 多数一致基线 (整模式查表)
  - 方向: 状态判别性提升 (余弦 0.995) 不足以突破 — 规则限制是主因
"""

import sys, os, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, train_w_h2o_stdp_gpu, DEVICE

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HIDDEN_SIZE = 256
INPUT_BIAS = 1.0
BASIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"
DATA_FILE = os.path.join(os.path.dirname(__file__), "english_pairs_1000.txt")
SCALES = [14, 50, 100, 200]


def load_pairs(path, n=None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pairs.append((parts[0].strip(), parts[1].strip()))
            if n and len(pairs) >= n:
                break
    return pairs


def evaluate_first_char(sim, dialogues):
    """v12.3 快照重放评估 — 与 test Step 2 一致"""
    ok = total = 0
    snap_i = 0
    for inp, resp in dialogues:
        resp_codes = sim._text_to_codes(resp)
        if not resp_codes:
            continue
        sim.W_coact = sim._coact_snapshots[snap_i].clone()
        snap_i += 1
        _, acc_state = sim.encode_text_lif(inp, update_memory=True)
        if acc_state.sum().item() == 0:
            continue
        bits = sim._binary_decode(sim.W_ctx_to_first, acc_state)
        pred = sum((1 << j) for j in range(8) if bits[j] > 0.5)
        total += 1
        ok += (pred == resp_codes[0])
    return ok, total


def majority_baseline(dialogues):
    from collections import Counter
    codes = []
    for inp, resp in dialogues:
        rc = [ord(c) for c in resp if 32 <= ord(c) <= 126]
        if rc:
            codes.append(rc[0])
    if not codes:
        return 0.0
    return max(Counter(codes).values()) / len(codes)


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
    print(f"基础训练完成 (Step 1 + 1.5): {time.perf_counter() - t0:.1f}s", flush=True)

    pairs = load_pairs(DATA_FILE)
    print(f"数据文件: {len(pairs)} 对话对\n", flush=True)
    print(f"{'规模':>5} {'有效':>4} {'首字符':>8} {'多数基线':>8} {'状态余弦':>8}", flush=True)
    print("-" * 40, flush=True)

    for N in SCALES:
        t1 = time.perf_counter()
        dialogues = pairs[:N]
        # 子集间隔离: 清零 W_coact (连续累积) + 重初始化 W_ctx_to_first
        sim.W_coact.zero_()
        sim.W_ctx_to_first.uniform_(-0.1, 0.1)
        sim.reset_state()
        sim.reset_memory()
        sim.train_context_to_first(dialogues, lr=0.05, n_iter=500)

        ok, total = evaluate_first_char(sim, dialogues)
        base = majority_baseline(dialogues)

        # 状态判别性 (快照重放收集, 诊断指标)
        states = []
        snap_i = 0
        for inp, resp in dialogues:
            rc = sim._text_to_codes(resp)
            if not rc:
                continue
            sim.W_coact = sim._coact_snapshots[snap_i].clone()
            snap_i += 1
            _, st = sim.encode_text_lif(inp, update_memory=True)
            if st.sum().item() > 0:
                states.append(st)
        if len(states) > 1:
            M = torch.stack(states)
            M = M / (M.norm(dim=1, keepdim=True) + 1e-12)
            C = (M @ M.T)
            off = C[~torch.eye(len(states), dtype=torch.bool)].cpu().numpy()
            cos = off.mean()
        else:
            cos = float("nan")

        print(f"{N:>5} {total:>4} {ok}/{total:>5} ({ok/total:>4.0%}) "
              f"{base:>8.1%} {cos:>8.3f}  ({time.perf_counter()-t1:.0f}s)", flush=True)

    print(f"\n总耗时: {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
