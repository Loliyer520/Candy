"""实验 6 — 训练状态 vs 重新编码状态的评估差异"""
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

# 训练数据 (一次性 encode)
ctx_data = []
for inp, resp in DIALOGUES:
    resp_codes = sim._text_to_codes(resp)
    if not resp_codes: continue
    _, acc_state = sim.encode_text_lif(inp)
    if acc_state.sum().item() == 0: continue
    ctx_data.append((acc_state, resp_codes[0]))
print(f"训练样本: {len(ctx_data)}")

def decode_w(sim, state):
    bits = sim._binary_decode(sim.W_ctx_to_first, state)
    code = 0
    for j in range(8):
        if bits[j] >= 0.5: code |= (1 << j)
    return chr(code) if 32 <= code <= 126 else '?'

# 训练并观察准确率 (用训练状态)
W_ctx = sim.W_ctx_to_first
print("\n训练过程 (训练状态评估):")
for it in range(500):
    random.shuffle(ctx_data)
    for acc_state, first_code in ctx_data:
        target = torch.tensor(
            [float((first_code >> j) & 1) for j in range(8)],
            dtype=torch.float32, device=DEVICE)
        pred = sim._binary_decode(W_ctx, acc_state)
        pred_bits = (pred > 0.5).float()
        target_bits = (target > 0.5).float()
        rpe = target_bits - pred_bits
        for j in range(8):
            W_ctx[j] += 0.05 * rpe[j] * acc_state
        W_ctx.clamp_(-10.0, 10.0)
    if (it + 1) % 100 == 0:
        ok = sum(1 for s, c in ctx_data if decode_w(sim, s) == chr(c))
        print(f"  iter {it+1}: train-state acc = {ok}/{len(ctx_data)}")

# 评估: 训练状态 vs 重新编码状态 (每次不同随机遗忘)
def eval_state(reuse=True, trials=5):
    results = []
    for t in range(trials):
        ok, tot = 0, 0
        for inp, resp in DIALOGUES:
            resp_codes = sim._text_to_codes(resp)
            if not resp_codes: continue
            if reuse:
                s = dict((i, st) for i, (inp2, r) in enumerate(DIALOGUES) for st in [None])  # noqa
                st = None
                for idx, (inp2, _) in enumerate(DIALOGUES):
                    if inp2 == inp:
                        st = ctx_data[idx][0]
                        break
            else:
                sim.reset_state(); sim.reset_memory()
                _, st = sim.encode_text_lif(inp)
            if st is None or st.sum().item() == 0: continue
            ok += (decode_w(sim, st) == chr(resp_codes[0])); tot += 1
        results.append(f"{ok}/{tot}")
    return results

print("\n训练状态评估 x5:", eval_state(reuse=True, trials=5))
print("重新编码评估 x5:", eval_state(reuse=False, trials=5))

# 检查状态对之间的相似度 (训练状态)
print("\n训练状态两两相似度 (余弦):")
import torch.nn.functional as F
for a in range(3):
    row = []
    for b in range(3):
        cos = F.cosine_similarity(ctx_data[a][0].unsqueeze(0), ctx_data[b][0].unsqueeze(0)).item()
        row.append(f"{cos:.3f}")
    print(f"  {DIALOGUES[a][0][:14]:16} | " + " ".join(row))
