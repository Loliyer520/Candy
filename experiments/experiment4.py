"""实验脚本 4 — W_ctx_to_first 路径决定性实验"""
import random
import numpy as np
import torch
from lif_pytorch import RecurrentLIFSimulator, DEVICE

random.seed(42); np.random.seed(42); torch.manual_seed(42)

DIALOGUES = [
    ("Hi!", "Hello!"), ("Hello!", "Hi there! How are you?"),
    ("How are you?", "I am doing well, thanks!"),
    ("What is your name?", "My name is Candy, nice to meet you!"),
    ("Who you are?", "I'm Candy, your AI assistant!"),
    ("What is 7 times 8?", "7 times 8 equals 56."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("I feel sad today.", "I'm sorry to hear that. Would you like to talk?"),
    ("I am so happy!", "That's wonderful! I'm glad you're feeling great today!"),
    ("Goodbye my friend.", "See you later! Take good care!"),
    ("Thank you so much.", "You're welcome friend. Happy to help!"),
    ("What's up?", "Not much, just relaxing. How about you?"),
    ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side!"),
]

sim = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
sim.init_random_weights(scale=0.8, connection_sparsity=0.5)

# 收集 (acc_state, first_code)
ctx_data = []
for inp, resp in DIALOGUES:
    resp_codes = sim._text_to_codes(resp)
    if not resp_codes: continue
    _, acc_state = sim.encode_text_lif(inp)
    ctx_data.append((acc_state.clone(), resp_codes[0]))

print(f"样本数: {len(ctx_data)}")
# 检查目标位分布
from collections import Counter
for j in range(8):
    c = Counter(1 if (code >> j) & 1 else 0 for _, code in ctx_data)
    print(f"  bit{j} 目标分布: {dict(c)}")

# 状态区分度: 成对重叠
print("\n状态两两重叠 (点积/活跃数):")
states = [s for s, _ in ctx_data]
for a in range(3):
    row = []
    for b in range(3):
        inter = (states[a] * states[b]).sum().item()
        row.append(f"{inter:.0f}")
    print(f"  {DIALOGUES[a][0][:12]:14} | " + " ".join(row))

# 决定性实验: 监督式 Hebbian (对照) 学 W_ctx_to_first
print("\n=== 对照: 监督式 Hebbian 学 W_ctx_to_first ===")
W = torch.zeros(8, 256, dtype=torch.float32, device=DEVICE)
for epoch in range(500):
    for s, code in ctx_data:
        t = torch.tensor([float((code >> j) & 1) for j in range(8)], dtype=torch.float32, device=DEVICE)
        post_eff = 2.0 * t - 1.0
        for j in range(8):
            W[j] += 0.1 * post_eff[j] * s

def eval_w(W):
    ok = 0
    for s, code in ctx_data:
        raw = torch.mv(W, s)
        bits = (raw > 0).float()
        pred = 0
        for j in range(8):
            if bits[j] >= 0.5: pred |= (1 << j)
        if pred == code: ok += 1
    return ok

print(f"  监督式 Hebbian (500 iter): {eval_w(W)}/{len(ctx_data)}")

# 感知器式对照 (仅实验, 验证可分性)
print("\n=== 对照: 感知器式 (验证数据可分性) ===")
W2 = torch.zeros(8, 256, dtype=torch.float32, device=DEVICE)
for epoch in range(500):
    for s, code in ctx_data:
        t = torch.tensor([float((code >> j) & 1) for j in range(8)], dtype=torch.float32, device=DEVICE)
        raw = torch.mv(W2, s)
        pred = (raw > 0).float()
        err = t - pred  # 感知器误差 (仅实验用)
        for j in range(8):
            W2[j] += 0.1 * err[j] * s
print(f"  感知器式 (500 iter): {eval_w(W2)}/{len(ctx_data)}")
