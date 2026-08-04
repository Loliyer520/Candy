"""
experiment13.py — 异联想转移记忆层 W_trans 可行性验证 (v13 探查)

背景: 首字符预测 (W_ctx_to_first) 可靠 (14 样本 memorize);
      后续字符 (W_seq) 结构性不可解 (experiment11: 逐 bit 独立 Hebbian
      vs 8-bit 联合目标, one-hot 也仅 22.7%)。

方向: 增加异联想转移记忆层 W_trans — 序列学习的海马体联想机制:
      训练 (时序 Hebbian): 回复内相邻字符对 (c_i → c_{i+1}),
        记录其 L4 输出对的共发放:
        W_trans += outer(o_{i+1}, o_i)   (后 ⊗ 前, "先因后果")
      回忆: r = W_trans · o_i  →  256 维联合模式 (矩阵联想)
            → W_h2o 解码 → 下一字符
      ★ 关键: 回忆是 256 维整体联想 (非逐 bit 独立), 绕开 experiment11
        证实的规则限制; 解码用已验证的 W_h2o (72/72)。

指标: 下一字符准确率 vs W_seq 5.4% / one-hot RPE 22.7% / bigram 天花板 34.1%
"""

import sys, os, random, time
from collections import Counter, defaultdict
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
from test_recurrent_learning import DIALOGUES


def extract_transitions():
    pairs = []
    for inp, resp in DIALOGUES:
        codes = [ord(c) for c in resp if 32 <= ord(c) <= 126]
        for i in range(len(codes) - 1):
            pairs.append((codes[i], codes[i + 1]))
    return pairs


def encode_char(sim, c):
    """单字符独立编码 → L4 输出 (与渐进训练一致: reset + 无 bias); c 为 ASCII 码"""
    sim.reset_state()
    vec = sim._char_to_8bit(chr(c))
    return sim._multi_layer_forward(vec)


def decode_char(sim, W, o):
    """W_h2o 二值阈值解码 → 字符码 (W 可指定: L4 版或输入编码版)"""
    bits = sim._binary_decode(W, o)
    return sum((1 << j) for j in range(8) if bits[j] > 0.5)


def main():
    t0 = time.perf_counter()
    sim = RecurrentLIFSimulator(hidden_size=HIDDEN_SIZE, output_size=8,
                                input_bias=INPUT_BIAS, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=4)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_codes = [ord(c) for c in BASIC_CHARS]
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=False)
    W_h2o_enc = sim.W_h2o.clone()          # Step 1: 解码输入编码
    sim.train_multi_layer_stdp(train_codes, num_epochs=200, lr_layer=0.3,
                               lr_out=0.5, verbose=False)
    print(f"基础训练完成: {time.perf_counter()-t0:.1f}s", flush=True)

    # 1) 验证: 单字符 L4 输出 → W_h2o (L4 版) 解码; 输入编码 → W_h2o_enc 解码
    ok = sum(1 for c in BASIC_CHARS
             if decode_char(sim, sim.W_h2o, encode_char(sim, ord(c))) == ord(c))
    print(f"单字符 L4→W_h2o(L4版) 解码: {ok}/{len(BASIC_CHARS)}", flush=True)
    ok2 = sum(1 for c in BASIC_CHARS
              if decode_char(sim, W_h2o_enc, sim._char_to_8bit(c)) == ord(c))
    print(f"输入编码→W_h2o(编码版) 解码: {ok2}/{len(BASIC_CHARS)}", flush=True)

    # 2) 收集相邻字符对 (两种表示), 构造 W_trans
    pairs = extract_transitions()
    o_cache = {}
    for c_cur, c_next in pairs:
        if c_cur not in o_cache:
            o_cache[c_cur] = encode_char(sim, c_cur)
        if c_next not in o_cache:
            o_cache[c_next] = encode_char(sim, c_next)

    # 表示重叠诊断: 唯一字符的 L4 输出两两余弦 (仅分析用)
    uniq = sorted(o_cache)
    M = torch.stack([o_cache[c] for c in uniq])
    C = M / (M.norm(dim=1, keepdim=True) + 1e-12)
    off = (C @ C.T)[~torch.eye(len(uniq), dtype=torch.bool)].cpu().numpy()
    print(f"L4 输出唯一字符数: {len(uniq)}, 两两余弦均值 {off.mean():.3f} "
          f"(min {off.min():.3f} max {off.max():.3f})", flush=True)
    enc = torch.stack([sim._char_to_8bit(chr(c)) for c in uniq])
    C2 = enc / (enc.norm(dim=1, keepdim=True) + 1e-12)
    off2 = (C2 @ C2.T)[~torch.eye(len(uniq), dtype=torch.bool)].cpu().numpy()
    print(f"输入编码两两余弦均值 {off2.mean():.3f} (min {off2.min():.3f} max {off2.max():.3f})", flush=True)

    # 3) 两种 W_trans: L4 层 vs 输入编码层
    n = len(pairs)
    Wt_l4 = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
    Wt_enc = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
    for c_cur, c_next in pairs:
        Wt_l4 += torch.outer(o_cache[c_next], o_cache[c_cur])
        e_cur = sim._char_to_8bit(chr(c_cur))
        e_next = sim._char_to_8bit(chr(c_next))
        Wt_enc += torch.outer(e_next, e_cur)
    Wt_l4 = Wt_l4 / Wt_l4.sum(dim=0, keepdim=True).clamp(min=1e-6)
    Wt_enc = Wt_enc / Wt_enc.sum(dim=0, keepdim=True).clamp(min=1e-6)

    per_char = defaultdict(Counter)
    for c_cur, c_next in pairs:
        per_char[c_cur][c_next] += 1
    bigram_ok = sum(per_char[c].most_common(1)[0][0] == nxt for c, nxt in pairs)

    print(f"\n下一字符预测准确率 (in-sample, {n} 转移):", flush=True)
    print(f"  bigram 多数一致天花板       : {bigram_ok/n:.1%}", flush=True)
    for name, Wt, o_get, dec in [("L4 层 W_trans  ", Wt_l4,
                                  lambda c: o_cache[c], lambda o: decode_char(sim, sim.W_h2o, o)),
                                 ("编码层 W_trans ", Wt_enc,
                                  lambda c: sim._char_to_8bit(chr(c)), lambda o: decode_char(sim, W_h2o_enc, o))]:
        ok = sum(1 for c, nxt in pairs
                 if dec(Wt @ o_get(c)) == nxt)
        print(f"  {name} (平均回忆): {ok/n:.1%}", flush=True)

    # max 池化回忆变体 (抗混合: 保留任一源位的强关联, OR 逻辑 — 与 max(v_peak, recall) 同风格)
    for name, o_get, dec in [("L4 层 max 回忆 ", lambda c: o_cache[c],
                              lambda o: decode_char(sim, sim.W_h2o, o)),
                             ("编码层 max 回忆", lambda c: sim._char_to_8bit(chr(c)),
                              lambda o: decode_char(sim, W_h2o_enc, o))]:
        ok = 0
        for c, nxt in pairs:
            o_cur = o_get(c)
            cols = Wt_l4 if name.startswith("L4") else Wt_enc
            r = cols[:, o_cur > 0.5].max(dim=1).values
            ok += (dec(r) == nxt)
        print(f"  {name} (max 池化): {ok/n:.1%}", flush=True)
    print(f"  对照: W_seq 网络状态 5.4% | one-hot RPE 22.7%", flush=True)
    print(f"\n总耗时: {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
