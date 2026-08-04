"""实验 9 — MemWork 累积变体对比 (解决饱和)"""
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

sim = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
sim.init_random_weights(scale=0.8, connection_sparsity=0.5)

def encode_variant(sim, text, mode, decay=0.5):
    sim.reset_state(); sim.reset_memory()
    state = torch.zeros(sim.hidden_size, device=DEVICE)
    for ch in text:
        if not (32 <= ord(ch) <= 126): continue
        vec = sim._char_to_8bit(ch) + sim.input_bias
        output = sim._multi_layer_forward(vec)
        sim.update_coactivation(output)
        recall = sim.recall_from_memassoc(output)
        v_peak = sim.V_deep[-1] if sim.num_layers > 1 else sim.V
        if mode == 'now':  # D: 只取当前 (无累积)
            state = torch.max(v_peak, recall)
        elif mode == 'decay':  # G: 指数衰减累积
            state = torch.max(torch.max(v_peak, recall), state * decay)
        elif mode == 'forget_mask':  # 现状
            forget_mask = (torch.rand_like(state) > 0.3).float()
            state = torch.max(torch.max(v_peak, recall), state * forget_mask)
    return state

def learn_and_eval(mode, decay=0.5, trials=5):
    ctx_data = []
    for inp, resp in DIALOGUES:
        resp_codes = sim._text_to_codes(resp)
        if not resp_codes: continue
        s = encode_variant(sim, inp, mode, decay)
        if s.sum().item() == 0: continue
        ctx_data.append((s, resp_codes[0]))
    # 区分度
    cos_sim = []
    for a in range(len(ctx_data)):
        for b in range(a+1, len(ctx_data)):
            cos_sim.append(F.cosine_similarity(ctx_data[a][0].unsqueeze(0), ctx_data[b][0].unsqueeze(0)).item())
    # 学习
    results = []
    for t in range(trials):
        W = torch.zeros(8, 256, device=DEVICE); W.uniform_(-0.1, 0.1)
        for e in range(500):
            for s, code in ctx_data:
                tgt = torch.tensor([float((code >> j) & 1) for j in range(8)], dtype=torch.float32, device=DEVICE)
                pred = (torch.mv(W, s) > 0).float()
                rpe = tgt - pred
                W += torch.outer(rpe, s) * 0.05
                W.clamp_(-10.0, 10.0)
        ok = 0
        for s, code in ctx_data:
            bits = (torch.mv(W, s) > 0).float()
            pred_code = sum((1 << j) for j in range(8) if bits[j] >= 0.5)
            ok += (pred_code == code)
        results.append(ok)
    return np.mean(cos_sim), results

for mode, decay in [('now', 0.0), ('decay', 0.5), ('decay', 0.7), ('decay', 0.9), ('forget_mask', 0.0)]:
    cos, res = learn_and_eval(mode, decay)
    print(f"  {mode:12} decay={decay}: 余弦均值={cos:.3f} | 学习结果 x5: {res}")
