"""针对性实验 — W_ctx_to_first 为何学不会 (vs 实验4感知器 12/14)"""
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


def make_sim(init_mode):
    sim = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    if init_mode == 'zeros':
        sim.W_ctx_to_first.zero_()
    return sim


def eval_first(sim):
    ok, tot = 0, 0
    outs = []
    for inp, resp in DIALOGUES:
        resp_codes = sim._text_to_codes(resp)
        if not resp_codes: continue
        _, acc_state = sim.encode_text_lif(inp)
        if acc_state.sum().item() == 0: continue
        bits = sim._binary_decode(sim.W_ctx_to_first, acc_state)
        code = 0
        for j in range(8):
            if bits[j] >= 0.5: code |= (1 << j)
        ch = chr(code) if 32 <= code <= 126 else '?'
        outs.append(ch)
        ok += (code == resp_codes[0]); tot += 1
    return ok, tot, outs


# A: uniform init (现状)
print("=== A: uniform init, lr=0.05, 500 iter (现状) ===")
sim = make_sim('uniform')
sim.train_context_to_first(DIALOGUES, lr=0.05, n_iter=500)
ok, tot, outs = eval_first(sim)
print(f"  {ok}/{tot} | preds: {outs}")

# B: zeros init
print("\n=== B: zeros init, lr=0.05, 500 iter ===")
sim = make_sim('zeros')
sim.train_context_to_first(DIALOGUES, lr=0.05, n_iter=500)
ok, tot, outs = eval_first(sim)
print(f"  {ok}/{tot} | preds: {outs}")

# C: zeros init, lr=0.1
print("\n=== C: zeros init, lr=0.1, 500 iter ===")
sim = make_sim('zeros')
sim.train_context_to_first(DIALOGUES, lr=0.1, n_iter=500)
ok, tot, outs = eval_first(sim)
print(f"  {ok}/{tot} | preds: {outs}")

# D: uniform init, lr=0.2, 1000 iter
print("\n=== D: uniform init, lr=0.2, 1000 iter ===")
sim = make_sim('uniform')
sim.train_context_to_first(DIALOGUES, lr=0.2, n_iter=1000)
ok, tot, outs = eval_first(sim)
print(f"  {ok}/{tot} | preds: {outs}")

# E: zeros init, lr=0.2, 1000 iter
print("\n=== E: zeros init, lr=0.2, 1000 iter ===")
sim = make_sim('zeros')
sim.train_context_to_first(DIALOGUES, lr=0.2, n_iter=1000)
ok, tot, outs = eval_first(sim)
print(f"  {ok}/{tot} | preds: {outs}")

# F: 权重统计 (训练后)
print("\n=== F: uniform init 训练后 W_ctx_to_first 权重统计 ===")
sim = make_sim('uniform')
sim.train_context_to_first(DIALOGUES, lr=0.05, n_iter=500)
for j in range(8):
    w = sim.W_ctx_to_first[j]
    print(f"  bit{j}: mean={w.mean().item():+.3f}, std={w.std().item():.3f}, pos={(w>0).float().mean().item():.1%}")
