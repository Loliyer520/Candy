"""
experiment19.py — P1 事件记忆 + LIF 逐字对话集成 (v14.1)

用户指令: "这个框架方向还能够逐字对话嘛?" → "实验一下"

架构 (双系统: 海马快记忆 + 皮层逐字生成):
  - LIF 链路 (generate_recurrent): 逐字循环生成 (首字符 → 编码 →
    工作记忆 → W_seq 候选 → 记忆修正 → 下一字符), 保留逐字推理形态
  - P1 事件记忆 (episodic_memory): 大库整体回忆, 通过新参数
    event_guide (callable) 逐字注入每一步, 优先于位置头覆盖 W_seq

对照 (每个库场景):
  A: 位置头 DG64 修正 (experiment17 最优配置, 基线)
  B: P1 事件记忆引导 (experiment18 v3 块绑定)

指标: 端到端逐字生成 (首字符/完整句/字符级), 库内快照恢复 (Step 4 场景)
★ 库50 是 P1 优势区 (P1 回忆 63.6% vs 位置头 35%) → 预期 B 显著 > A
★ P1 margin 统计: 同时记录回忆的 argmax margin, 为后续门控决策留数据
"""

import sys, os, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, train_w_h2o_stdp_gpu
from episodic_memory import EpisodicEventMemory
from test_recurrent_learning import DIALOGUES

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HIDDEN = 256
BASIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"


def load_pairs(path, n):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            inp, resp = line.split("\t", 1)
            pairs.append((inp.strip(), resp.strip()))
            if len(pairs) >= n:
                break
    return pairs


def train_lif(sim, dialogues, verbose=True):
    t0 = time.perf_counter()
    train_codes = [ord(c) for c in BASIC_CHARS]
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=False)
    sim.train_multi_layer_stdp(train_codes, num_epochs=120, lr_layer=0.3,
                               lr_out=0.5, verbose=False, n_loops=1)
    sim.train_context_to_first(dialogues, lr=0.05, n_iter=400, n_loops=1)
    sim.train_pos_heads(dialogues, lr=0.05, n_iter=200, n_loops=1)
    if verbose:
        print(f"  LIF 训练完成: {time.perf_counter()-t0:.0f}s", flush=True)


def train_p1(sim, dialogues):
    mem = EpisodicEventMemory(dim=8192, char_ones=8, max_pos=32, seed=7)
    for inp, resp in dialogues:
        in_codes = sim._text_to_codes(inp)
        out_codes = sim._text_to_codes(resp)
        if in_codes and out_codes:
            mem.store(in_codes, out_codes)
    return mem


def make_guide(p1_codes):
    """P1 整条回忆 → 逐字注入 (超范围回退 None → 位置头/W_seq)"""
    def guide(step):
        return p1_codes[step] if step < len(p1_codes) else None
    return guide


def run_scenario(dialogues, label):
    n_dlg = len(dialogues)
    print(f"\n{'='*60}\n--- {label} (库={n_dlg}) ---", flush=True)
    sim = RecurrentLIFSimulator(hidden_size=HIDDEN, output_size=8,
                                input_bias=1.0, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=3,
                                use_dg_separation=True, dg_k=64,
                                use_eligibility_trace=False)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_lif(sim, dialogues)
    mem = train_p1(sim, dialogues)

    # 库内端到端逐字生成对照 (快照恢复)
    eval_n = min(14, n_dlg)
    statA = dict(full=0, cchar=0, ctot=0, first=0)
    statB = dict(full=0, cchar=0, ctot=0, first=0)
    margins = {"ok": [], "bad": []}
    for i in range(eval_n):
        inp, resp = dialogues[i]
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        in_codes = sim._text_to_codes(inp)
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, cf = sim.encode_text_lif(inp, update_memory=True, n_loops=1)
        if cf.sum().item() == 0:
            continue
        # A: 位置头 DG64
        rA = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                    update_memory=True, use_pos_memory=True,
                                    n_loops=1)
        # B: P1 事件记忆引导 (整体回忆 → 逐字注入)
        p1_codes, margins_i = mem.recall_with_margin(in_codes, len(rc))
        rB = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                    update_memory=True, event_guide=make_guide(p1_codes),
                                    n_loops=1)
        for tag, r in (("A", rA), ("B", rB)):
            st = statA if tag == "A" else statB
            if r and resp and r[0] == resp[0]:
                st["first"] += 1
            if r == resp:
                st["full"] += 1
            for a, b in zip(r, resp):
                st["ctot"] += 1
                if a == b:
                    st["cchar"] += 1
        # P1 margin 判别力记录 (正确 vs 错误字符)
        for k, c in enumerate(p1_codes[:len(rc)]):
            m = margins_i[k] if k < len(margins_i) else 0.0
            key = "ok" if c == rc[k] else "bad"
            margins[key].append(m)

    def fmt(st):
        c = st["cchar"] / max(st["ctot"], 1)
        return f"{st['first']}/{eval_n} 首字, {st['full']}/{eval_n} 完整, {c:.1%} 字符"
    print(f"  [A 位置头DG64] {fmt(statA)}", flush=True)
    print(f"  [B P1引导    ] {fmt(statB)}", flush=True)
    if margins["ok"] and margins["bad"]:
        import numpy as _np
        print(f"  [P1 margin] 正确 med={_np.median(margins['ok']):.2f} "
              f"(n={len(margins['ok'])}), 错误 med={_np.median(margins['bad']):.2f} "
              f"(n={len(margins['bad'])}), 重叠判据可用性待评估", flush=True)
    return dict(A=statA, B=statB)


def main():
    print("=" * 60, flush=True)
    print("experiment19 — P1 事件记忆 + LIF 逐字对话集成 (v14.1)", flush=True)
    print("=" * 60, flush=True)
    run_scenario(DIALOGUES, "库14 (DIALOGUES, 位置头优势区)")
    run_scenario(load_pairs("english_pairs_1000.txt", 50), "库50 (english_pairs, P1 优势区)")


if __name__ == "__main__":
    main()
