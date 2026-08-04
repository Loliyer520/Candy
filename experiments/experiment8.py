"""实验 8 — MemWork 饱和修复验证: max累积 vs 随机替换"""
import random
import numpy as np
import torch
import torch.nn.functional as F
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

# 复刻 encode_text_lif 的核心: 逐字符前向 + 记忆更新
def encode_variant(sim, text, mode, replace_gate=0.3, forget_ratio=0.3):
    sim.reset_state(); sim.reset_memory()
    chars = text
    state = torch.zeros(sim.hidden_size, device=DEVICE)
    for ch in chars:
        vec = sim._char_to_8bit(ch) + sim.input_bias
        output = sim._multi_layer_forward(vec)
        sim.update_coactivation(output)
        recall = sim.recall_from_memassoc(output)
        v_peak = sim.V_deep[-1] if sim.num_layers > 1 else sim.V
        if mode == 'max':
            forget_mask = (torch.rand_like(state) > forget_ratio).float()
            state = torch.max(torch.max(v_peak, recall), state * forget_mask)
        elif mode == 'replace':
            replace_mask = (torch.rand_like(state) < replace_gate).float()
            new_val = torch.max(v_peak, recall)  # 新信息
            state = state * (1.0 - replace_mask) + new_val * replace_mask
    return state

sim = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
sim.init_random_weights(scale=0.8, connection_sparsity=0.5)

for mode in ['max', 'replace']:
    print(f"\n=== MemWork 模式: {mode} ===")
    states = []
    for inp, resp in DIALOGUES:
        s = encode_variant(sim, inp, mode)
        states.append(s)
        print(f"  '{inp[:22]:24}' sum={s.sum().item():7.1f} mean={s.mean().item():.3f} "
              f">0.5={ (s>0.5).sum().item():4d} >0.9={ (s>0.9).sum().item():4d}")
    # 区分度: 两两余弦 + 平均
    cos_sim = []
    for a in range(len(states)):
        for b in range(a+1, len(states)):
            cos = F.cosine_similarity(states[a].unsqueeze(0), states[b].unsqueeze(0)).item()
            cos_sim.append(cos)
    print(f"  两两余弦均值: {np.mean(cos_sim):.3f} (min={min(cos_sim):.3f}, max={max(cos_sim):.3f})")

# 用 replace 模式做 W_ctx_to_first 学习验证
print("\n=== replace 模式下 W_ctx_to_first 学习 ===")
ctx_data = []
for inp, resp in DIALOGUES:
    resp_codes = sim._text_to_codes(resp)
    if not resp_codes: continue
    s = encode_variant(sim, inp, 'replace')
    if s.sum().item() == 0: continue
    ctx_data.append((s, resp_codes[0]))

def decode(W, s):
    bits = (torch.mv(W, s) > 0).float()
    code = 0
    for j in range(8):
        if bits[j] >= 0.5: code |= (1 << j)
    return chr(code) if 32 <= code <= 126 else '?'

best = 0
for trial in range(5):
    W = torch.zeros(8, 256, device=DEVICE); W.uniform_(-0.1, 0.1)
    for e in range(500):
        for s, code in ctx_data:
            t = torch.tensor([float((code >> j) & 1) for j in range(8)], dtype=torch.float32, device=DEVICE)
            pred = (torch.mv(W, s) > 0).float()
            rpe = t - pred
            W += torch.outer(rpe, s) * 0.05
            W.clamp_(-10.0, 10.0)
    ok = sum(1 for s, c in ctx_data if decode(W, s) == chr(c))
    best = max(best, ok)
    print(f"  trial {trial+1}: {ok}/{len(ctx_data)}")
print(f"  best: {best}/{len(ctx_data)}")
