"""快速验证 — RPE 规则在 W_h2o 与 W_ctx_to_first 上的效果"""
import random
import numpy as np
import torch
from lif_pytorch import RecurrentLIFSimulator, train_w_h2o_stdp_gpu, DEVICE

random.seed(42); np.random.seed(42); torch.manual_seed(42)

basic_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"
train_codes = [ord(c) for c in basic_chars]

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

# ---- W_h2o ----
print("=== W_h2o (RPE 规则) ===")
sim = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=True)

# ---- W_ctx_to_first ----
print("\n=== W_ctx_to_first (RPE 规则) ===")
sim2 = RecurrentLIFSimulator(hidden_size=256, output_size=8, input_bias=1.0, num_layers=4)
sim2.init_random_weights(scale=0.8, connection_sparsity=0.5)
sim2.train_context_to_first(DIALOGUES, lr=0.05, n_iter=500)

def decode(sim, state):
    bits = sim._binary_decode(sim.W_ctx_to_first, state)
    code = 0
    for j in range(8):
        if bits[j] >= 0.5: code |= (1 << j)
    return chr(code) if 32 <= code <= 126 else '?'

ok, tot = 0, 0
for inp, resp in DIALOGUES:
    resp_codes = sim2._text_to_codes(resp)
    if not resp_codes: continue
    _, acc_state = sim2.encode_text_lif(inp)
    if acc_state.sum().item() == 0: continue
    pred = decode(sim2, acc_state)
    exp = chr(resp_codes[0])
    mark = "✓" if pred == exp else "✗"
    print(f"    {mark} '{inp}' → pred={pred} expected={exp}")
    ok += (pred == exp); tot += 1
print(f"  首字符准确率: {ok}/{tot}")
