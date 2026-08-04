"""
experiment11.py — W_seq 容量决定性实验 (v12.5 探查)

问题: W_seq 失败 (评估 5.4%) 的瓶颈是"状态信息不足"还是"学习规则容量极限"?

方法: 固定 RPE Hebbian + 纯二值阈值解码 (与 train_sequence 逐行一致:
      out = (W·x + b > 0), RPE = target - out, W[j] += lr×RPE_j×x,
      无 b 更新, lr 每 200 epoch ×0.9), 替换输入为理想化字符状态:
      1) one-hot 当前字符   (信息论极限: 每个字符可完全区分)
      2) 随机稠密编码当前字符 (网络输入层编码风格: ~50% 活跃, 256 维)

对照: 多数一致精确匹配上限 (信息论天花板)
  - 每唯一输入取出现最多的目标, 统计精确匹配率

结论判据:
  - one-hot 学习 ≈ 上限 → 规则容量够用, 瓶颈 = 状态不携带字符身份信息
  - one-hot 学习 << 上限 → 规则本身容量不足 → 确认 W_seq 机制极限

结论 (实测):
  - one-hot 当前字符: 在线 22.7% / 批量 18.4-22.2% << 天花板 34.1%
  - 随机稠密当前字符: 15.6-20.5% (与 one-hot 同量级)
  - 网络 MemWork 状态: 5.4% (test Step 3) — 状态信息缺失是次瓶颈
  - ★ 根因: RPE Hebbian 每 bit 独立决策 + 8-bit 联合目标
    → "逐 bit 多数" ≠ "整模式多数" (Naive-Bayes 式独立决策 vs 联合最优),
    one-hot 下每字符的 8 列权重独立收敛, 组合出的字符 ≠ 该字符的多数目标
  - 结论: 逐字符 next-char 预测在"纯二值 RPE Hebbian + 8-bit 独立解码"
    约束下结构性不可解 — 即使状态完美 (one-hot) 也无法逼近天花板
  - 方向: W_seq 关闭 (除非放宽学习规则约束); 状态改造只能到 ~20% (规则上限),
    收益有限且风险高 (可能破坏 W_ctx_to_first 14/14)
"""

import sys, os, random
from collections import Counter, defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import DEVICE
from test_recurrent_learning import DIALOGUES

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

OUT = 8
DIM = 256


def extract_transitions():
    """与 train_sequence 一致: 响应内相邻字符转移 (当前字符 → 下一字符)"""
    pairs = []
    for inp, resp in DIALOGUES:
        codes = [ord(c) for c in resp if 32 <= ord(c) <= 126]
        for i in range(len(codes) - 1):
            pairs.append((codes[i], codes[i + 1]))
    return pairs


def encode_onehot(c):
    x = torch.zeros(DIM, dtype=torch.float32, device=DEVICE)
    x[c] = 1.0
    return x


def encode_dense(c):
    """随机稠密编码 (与网络输入层同风格: RandomState(c) 种子, ~50% 活跃)"""
    return torch.tensor((np.random.RandomState(c).rand(DIM) < 0.5),
                        dtype=torch.float32, device=DEVICE)


def majority_ceiling(pairs):
    """每唯一输入 → 多数目标 → 精确匹配率 (信息论天花板)"""
    per = defaultdict(Counter)
    for c, nxt in pairs:
        per[c][nxt] += 1
    ok = sum(max(cnt.values()) for cnt in per.values())
    return ok / len(pairs), len(per)


def train_rpe_online(pairs, encode, n_iter=200, lr=0.05):
    """逐行复刻 train_sequence 的逐样本在线更新 (小规模对照)"""
    n = len(pairs)
    xs = [encode(c) for c, _ in pairs]
    tgts = torch.tensor([[float((nxt >> j) & 1) for j in range(OUT)]
                         for _, nxt in pairs], dtype=torch.float32, device=DEVICE)
    W = torch.empty(OUT, DIM, dtype=torch.float32, device=DEVICE)
    W.uniform_(-0.1, 0.1)
    b = torch.empty(OUT, dtype=torch.float32, device=DEVICE)
    b.uniform_(-0.1, 0.1)
    seq = list(zip(xs, tgts))
    best_acc = 0.0
    lr_current = lr
    for epoch in range(n_iter):
        random.shuffle(seq)
        correct = 0
        for fr, target in seq:
            out = (W @ fr + b > 0).float()
            rpe = target - out
            for j in range(OUT):
                W[j] += lr_current * rpe[j] * fr
            W.clamp_(-10.0, 10.0)
            if (out == target).all().item():
                correct += 1
        best_acc = max(best_acc, correct / n)
        if (epoch + 1) % 200 == 0:
            lr_current *= 0.9
    return best_acc


def train_rpe_batch(pairs, encode, n_iter=1000, lr=0.05):
    """批量版 RPE Hebbian (在线规则的批量等价, 容量上界估计):
    ΔW = lr × Σ_k RPE_k ⊗ x_k — 逐样本更新在 GPU 上太慢, 批量版给出
    容量上界: 若批量版都不能突破上限, 在线版更不能"""
    n = len(pairs)
    X = torch.stack([encode(c) for c, _ in pairs])   # n×256
    T = torch.tensor([[float((nxt >> j) & 1) for j in range(OUT)]
                      for _, nxt in pairs], dtype=torch.float32, device=DEVICE)
    W = torch.empty(OUT, DIM, dtype=torch.float32, device=DEVICE)
    W.uniform_(-0.1, 0.1)
    b = torch.empty(OUT, dtype=torch.float32, device=DEVICE)
    b.uniform_(-0.1, 0.1)
    best = 0.0
    lr_cur = lr
    for epoch in range(n_iter):
        out = (X @ W.T + b > 0).float()    # n×8
        rpe = T - out                      # n×8 (正确样本 RPE=0)
        W += lr_cur * (rpe.T @ X)          # 8×256
        W.clamp_(-10.0, 10.0)
        best = max(best, (out == T).all(dim=1).float().mean().item())
        if (epoch + 1) % 200 == 0:
            lr_cur *= 0.9
    return best


def main():
    pairs = extract_transitions()
    n = len(pairs)
    ceil, u = majority_ceiling(pairs)
    print(f"转移总数: {n}", flush=True)
    print(f"唯一输入(字符): {u}", flush=True)
    print(f"多数一致精确匹配上限 (信息论天花板): {ceil:.1%}", flush=True)
    print(f"多数类基线 (预测全局最频繁下一字符): "
          f"{max(Counter(nxt for _, nxt in pairs).values()) / n:.1%}", flush=True)
    print(f"\nRPE Hebbian 训练 (输入 → 下一字符, best_acc):", flush=True)
    for name, encode in [("one-hot 当前字符", encode_onehot),
                         ("随机稠密 当前字符", encode_dense)]:
        print(f"  {name}:", flush=True)
        # 在线版 (小规模对照, 与 train_sequence 逐行一致)
        onl = train_rpe_online(pairs, encode, n_iter=200)
        print(f"    在线版 n_iter=200: best_acc = {onl:.1%}", flush=True)
        # 批量版 (容量上界扫描)
        for n_iter in (500, 1000, 2000):
            bat = train_rpe_batch(pairs, encode, n_iter=n_iter)
            print(f"    批量版 n_iter={n_iter}: best_acc = {bat:.1%}", flush=True)


if __name__ == "__main__":
    main()
