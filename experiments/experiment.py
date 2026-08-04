"""实验脚本 — 验证死神经元修复方案 (out_eff = max(out, target))"""
import random, time
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


def run(mode, lr, num_epochs=200, decay=0.0, verbose=True):
    """mode: 'orig' 原规则 / 'eff' out_eff=max(out,t)"""
    sim = TorchLIFSimulator()
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    sim.W_h2o.uniform_(-0.1, 0.1)
    sim.b_o.uniform_(-0.1, 0.1)

    def check(i):
        out = sim._binary_decode(sim.W_h2o, input_vecs[i], sim.b_o)
        code = 0
        for j in range(8):
            if out[j] >= 0.5: code |= (1 << j)
        return code == train_codes[i]

    init_ok = sum(1 for i in range(n_vocab) if check(i))
    best = init_ok
    for epoch in range(num_epochs):
        idxs = list(range(n_vocab)); random.shuffle(idxs)
        ok = 0
        for i in idxs:
            vec = input_vecs[i]; tgt = targets[i]
            out = sim._binary_decode(sim.W_h2o, vec, sim.b_o)
            pred_bits = (out > 0.5).int()
            tgt_bits = (tgt > 0.5).int()
            correct_mask = (pred_bits == tgt_bits).float()
            da = 2.0 * correct_mask - 1.0
            if mode == 'orig':
                post = out
            else:  # eff
                post = torch.max(out, tgt)  # 实际发放 OR 期望发放
            for j in range(8):
                sim.W_h2o[j] += lr * da[j] * post[j] * vec
            if decay > 0:
                sim.W_h2o *= (1.0 - decay)
            sim.W_h2o.clamp_(-10.0, 10.0)
            if correct_mask.sum().item() == 8:
                ok += 1
        acc = ok / n_vocab
        best = max(best, ok)
        if (epoch + 1) % 50 == 0 or acc == 1.0:
            print(f"    {mode} epoch {epoch+1}: acc={ok}/{n_vocab} ({acc:.1%})", flush=True)
            if acc == 1.0: break
    final_ok = sum(1 for i in range(n_vocab) if check(i))
    print(f"  [{mode}] lr={lr} decay={decay}: init={init_ok}/{n_vocab} → final={final_ok}/{n_vocab} (best={best})")
    return final_ok


print("=== 实验 A: 原规则 (复现基线) ===")
run('orig', 0.5)

print("\n=== 实验 B: out_eff=max(out,t), lr=0.5 ===")
run('eff', 0.5)

print("\n=== 实验 C: out_eff=max(out,t), lr=0.1 ===")
run('eff', 0.1)

print("\n=== 实验 D: out_eff=max(out,t), lr=0.2 + 稳态衰减0.002 ===")
run('eff', 0.2, decay=0.002)

print("\n=== 实验 E: out_eff=max(out,t), lr=0.2 ===")
run('eff', 0.2)
