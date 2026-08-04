"""
diag_p1.py — P1 初版失败根因诊断 (experiment18 前置定位)
三个假说:
  D1 管线本身有 bug (单事件也无法回忆)
  D2 上下文投影层 (ctx_sparse) 判别性差 (两两重叠过高)
  D3 多事件串扰增长过快 (增量回忆曲线陡降)
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from episodic_memory import EpisodicEventMemory
from experiment18 import make_sim, encode_contexts
from test_recurrent_learning import DIALOGUES

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

sim = make_sim()
states, responses = encode_contexts(sim, DIALOGUES)
n = len(states)
print(f"库 {n} 对话, 编码完成")

# D1: 单事件回忆 (管线正确性)
mem1 = EpisodicEventMemory(dim=4096, seed=7)
mem1.store(states[0], responses[0])
pred = mem1.recall(states[0], len(responses[0]))
print(f"\n[D1 单事件] 目标={responses[0][:8]} 预测={pred[:8]} "
      f"字符级={sum(a==b for a,b in zip(pred, responses[0]))}/{len(responses[0])}")

# D2: ctx_sparse 两两重叠
mem = EpisodicEventMemory(dim=4096, seed=7)
ctxs = [mem.ctx_sparse(s) for s in states]
ov = []
for i in range(n):
    for j in range(i+1, n):
        ov.append(int(ctxs[i] @ ctxs[j]))
print(f"\n[D2 ctx重叠] 两两内积 (共{len(ov)}对): "
      f"min={min(ov)} med={int(np.median(ov))} max={max(ov)} "
      f"(ctx_ones=200, 理想 < 20)")

# D3: 增量存储 → 逐样本回忆曲线 (每新增一个事件, 回忆已存全部)
memN = EpisodicEventMemory(dim=4096, seed=7)
print(f"\n[D3 增量串扰] 存 k 个后回忆前 k 个的字符级:")
for i, (st, rc) in enumerate(zip(states, responses)):
    memN.store(st, rc)
    accs = []
    for j in range(i+1):
        pred = memN.recall(states[j], len(responses[j]))
        accs.append(sum(a==b for a,b in zip(pred, responses[j]))/len(responses[j]))
    k = i + 1
    print(f"  k={k:2d}: 平均 {np.mean(accs):.0%}  | 最近样本 {accs[-1]:.0%}")
