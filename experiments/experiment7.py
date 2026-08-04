"""实验 7 — 控制变量: init / lr / clamp / 顺序 对 W_ctx_to_first 学习的影响"""
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
ctx_data = []
for inp, resp in DIALOGUES:
    resp_codes = sim._text_to_codes(resp)
    if not resp_codes: continue
    _, acc_state = sim.encode_text_lif(inp)
    if acc_state.sum().item() == 0: continue
    ctx_data.append((acc_state, resp_codes[0]))

def decode(W, s):
    bits = (torch.mv(W, s) > 0).float()
    code = 0
    for j in range(8):
        if bits[j] >= 0.5: code |= (1 << j)
    return chr(code) if 32 <= code <= 126 else '?'

def train(init, lr, n_epoch, do_clamp, do_shuffle, order_epochs=False):
    W = torch.zeros(8, 256, device=DEVICE)
    if init == 'uniform':
        W.uniform_(-0.1, 0.1)
    for e in range(n_epoch):
        idxs = list(range(len(ctx_data)))
        if do_shuffle:
            random.shuffle(idxs)
        for i in idxs:
            s, code = ctx_data[i]
            t = torch.tensor([float((code >> j) & 1) for j in range(8)], dtype=torch.float32, device=DEVICE)
            pred = (torch.mv(W, s) > 0).float()
            rpe = t - pred
            W += torch.outer(rpe, s) * lr
            if do_clamp:
                W.clamp_(-10.0, 10.0)
    ok = sum(1 for s, c in ctx_data if decode(W, s) == chr(c))
    return ok, W

cases = [
    ("uniform lr=0.05 clamp shuffle", 'uniform', 0.05, 500, True, True),
    ("uniform lr=0.05 clamp 顺序", 'uniform', 0.05, 500, True, False),
    ("uniform lr=0.1  clamp shuffle", 'uniform', 0.1, 500, True, True),
    ("zeros   lr=0.1  clamp shuffle", 'zeros', 0.1, 500, True, True),
    ("zeros   lr=0.1  NOclamp shuffle", 'zeros', 0.1, 500, False, True),
    ("uniform lr=0.1  NOclamp shuffle", 'uniform', 0.1, 500, False, True),
    ("uniform lr=0.1  NOclamp 顺序", 'uniform', 0.1, 500, False, False),
    ("uniform lr=0.2  NOclamp shuffle", 'uniform', 0.2, 1000, False, True),
]
for name, init, lr, ne, cl, sh in cases:
    ok, W = train(init, lr, ne, cl, sh)
    print(f"  {name:34} → {ok}/{len(ctx_data)}")
