"""实验脚本 3 — 决定性对照 + MemWork 区分度分析"""
import random
import numpy as np
import torch
from lif_pytorch import TorchLIFSimulator, RecurrentLIFSimulator, DEVICE

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

print("=== 对照实验 I: 监督式 Hebbian Δw=lr×(2t-1)×pre (验证规则缺陷假说) ===")
sim = TorchLIFSimulator()
sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
sim.W_h2o.uniform_(-0.1, 0.1); sim.b_o.uniform_(-0.1, 0.1)
for epoch in range(10):
    for i in range(n_vocab):
        vec = input_vecs[i]; tgt = targets[i]
        post_eff = 2.0 * tgt - 1.0  # 监督信号
        for j in range(8):
            sim.W_h2o[j] += 0.5 * post_eff[j] * vec
    sim.W_h2o.clamp_(-10.0, 10.0)

ok = 0
for i in range(n_vocab):
    out = sim._binary_decode(sim.W_h2o, input_vecs[i], sim.b_o)
    code = 0
    for j in range(8):
        if out[j] >= 0.5: code |= (1 << j)
    if code == train_codes[i]: ok += 1
print(f"  监督式 Hebbian 10 epoch 后: {ok}/{n_vocab} ({ok/n_vocab:.1%})")

print("\n=== MemWork 输入区分度分析 ===")
sim2 = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
sim2.init_random_weights(scale=0.8, connection_sparsity=0.5)

inputs = ["Hi!", "Hello!", "How are you?", "What is your name?", "Who you are?",
          "What is 7 times 8?", "I feel sad today.", "Goodbye my friend.",
          "Thank you so much.", "What's up?", "Tell me a joke.", "I am so happy!"]
states = []
for inp in inputs:
    sim2.reset_state(); sim2.reset_memory()
    _, acc_state = sim2.encode_text_lif(inp)
    states.append(acc_state)
    print(f"  '{inp[:20]}': mean={acc_state.mean().item():.3f}, >0.5={ (acc_state>0.5).sum().item() }, >0.9={ (acc_state>0.9).sum().item() }")

# 成对重叠度 (活跃神经元重合比例)
print("\n  状态间重叠度 (Jaccard, >0.5 活跃):")
sets = [(s > 0.5).int() for s in states]
for a in range(4):
    row = []
    for b in range(4):
        inter = (sets[a] * sets[b]).sum().item()
        union = ((sets[a] + sets[b]) > 0).sum().item()
        row.append(f"{inter/max(union,1):.2f}")
    print(f"    {inputs[a][:12]:14} | " + " ".join(row))
