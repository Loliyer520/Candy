"""实验脚本 2 — 方案 G: b_o 正初始化 (不改规则) vs 方案 F (改规则 da 基于 eff)"""
import random
import numpy as np
import torch
from lif_pytorch import TorchLIFSimulator, DEVICE

random.seed(42); np.random.seed(42); torch.manual_seed(42)

basic_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"
train_codes = [ord(c) for c in basic_chars]
n_vocab = len(train_codes)

targets = torch.zeros(n_vocab, 8, dtype=torch.float32, device=DEVICE)
for i, c in enumerate(train_codes):
    for j in range(8):
        targets[i, j] = float((c >> j) & 1)

input_vecs = torch.zeros(n_vocab, 256, dtype=torch.float32, device=DEVICE)
for i, c in enumerate(train_codes):
    ch = chr(c) if 32 <= c <= 126 else '?'
    input_vecs[i] = TorchLIFSimulator._get_char_code(ch)


def run(mode, lr, num_epochs=200, b_o_val=0.0, decay=0.0, verbose=True):
    sim = TorchLIFSimulator()
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    sim.W_h2o.uniform_(-0.1, 0.1)
    sim.b_o.fill_(b_o_val)  # 正初始化 (不更新)

    def check(i):
        out = sim._binary_decode(sim.W_h2o, input_vecs[i], sim.b_o)
        code = 0
        for j in range(8):
            if out[j] >= 0.5: code |= (1 << j)
        return code == train_codes[i]

    init_ok = sum(1 for i in range(n_vocab) if check(i))
    for epoch in range(num_epochs):
        idxs = list(range(n_vocab)); random.shuffle(idxs)
        ok = 0
        for i in idxs:
            vec = input_vecs[i]; tgt = targets[i]
            out = sim._binary_decode(sim.W_h2o, vec, sim.b_o)
            pred_bits = (out > 0.5).int()
            tgt_bits = (tgt > 0.5).int()
            if mode == 'F':
                # 方案 F: post=max(out,tgt), da 基于 eff 匹配
                post = torch.max(out, tgt)
                eff_correct = (post == tgt).float()
                da = 2.0 * eff_correct - 1.0
            else:
                # 原规则 (方案 G: 靠 b_o 正初始化)
                correct_mask = (pred_bits == tgt_bits).float()
                da = 2.0 * correct_mask - 1.0
                post = out
            for j in range(8):
                sim.W_h2o[j] += lr * da[j] * post[j] * vec
            if decay > 0:
                sim.W_h2o *= (1.0 - decay)
            sim.W_h2o.clamp_(-10.0, 10.0)
            if correct_mask.sum().item() == 8 if mode != 'F' else (pred_bits == tgt_bits).all().item():
                ok += 1
        acc = ok / n_vocab
        if (epoch + 1) % 50 == 0 or acc == 1.0:
            print(f"    [{mode}] b_o={b_o_val} lr={lr} decay={decay} epoch {epoch+1}: acc={ok}/{n_vocab} ({acc:.1%})", flush=True)
            if acc == 1.0: break
    final_ok = sum(1 for i in range(n_vocab) if check(i))
    print(f"  [{mode}] b_o={b_o_val} lr={lr} decay={decay}: init={init_ok} → final={final_ok}/{n_vocab}")
    return final_ok


print("=== G1: 原规则, b_o=+1.0, lr=0.1 ===")
run('G', 0.1, b_o_val=1.0)

print("\n=== G2: 原规则, b_o=+1.0, lr=0.2 ===")
run('G', 0.2, b_o_val=1.0)

print("\n=== G3: 原规则, b_o=+0.5, lr=0.1 ===")
run('G', 0.1, b_o_val=0.5)

print("\n=== G4: 原规则, b_o=+1.0, lr=0.1, decay=0.002 ===")
run('G', 0.1, b_o_val=1.0, decay=0.002)

print("\n=== F1: 方案F规则, b_o=0, lr=0.1 ===")
run('F', 0.1, b_o_val=0.0)

print("\n=== F2: 方案F规则, b_o=+1.0, lr=0.1 ===")
run('F', 0.1, b_o_val=1.0)
