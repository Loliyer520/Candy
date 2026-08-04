"""
experiment20.py — ②自回归事件记忆 + ①margin 门控 (v14.2)

用户指令: "1+2" — 同时做 margin 门控实验 + 真·逐字推理。

② 真·逐字推理 (AutoRegressiveEventMemory): 前缀 → 下一字符的 Hebbian
  关联 (E2 += outer(p[t][c_t], bind(prefix_t)))。生成状态 (已生成前缀)
  进入查询 — 真正的自回归, 突破整体回忆的块独立性。
  文献: Drieu & Zugaro 2019 (theta 序列 = 外部输入 + 内在动力学);
  McNamee 2021 (海马 = 序列生成器)。

① margin 门控 (experiment19 发现 P1 margin 有判别力): 自回归下一步
  margin < θ → 放弃自回归 (回退位置头/W_seq)。库内 margin 高 (全接管),
  库外 margin 低 (自动回退) → 对话可信度。

对照 (端到端逐字生成, LIF DG64 底座):
  A: 位置头 DG64 修正            (experiment17 最优, 无事件记忆)
  C: 自回归全引导 (θ=0)
  D: 自回归 + 门控 (θ 扫描 200/400)

评估:
  - 库内: english_pairs 前 50 训练, 前 14 评估 (端到端字符级/完整句)
  - 库外: english_pairs 50-60 (未见 10 个) → margin 分布 + 引导占比
    + 字符级, 验证门控的"不知道就回退"行为
"""

import sys, os, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, train_w_h2o_stdp_gpu
from episodic_memory import AutoRegressiveEventMemory
from test_recurrent_learning import DIALOGUES

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HIDDEN = 256
BASIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"


def load_pairs(path, n, offset=0):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            if offset > 0:
                offset -= 1
                continue
            inp, resp = line.split("\t", 1)
            pairs.append((inp.strip(), resp.strip()))
            if len(pairs) >= n:
                break
    return pairs


def train_lif(sim, dialogues):
    t0 = time.perf_counter()
    train_codes = [ord(c) for c in BASIC_CHARS]
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=False)
    sim.train_multi_layer_stdp(train_codes, num_epochs=120, lr_layer=0.3,
                               lr_out=0.5, verbose=False, n_loops=1)
    sim.train_context_to_first(dialogues, lr=0.05, n_iter=400, n_loops=1)
    sim.train_pos_heads(dialogues, lr=0.05, n_iter=200, n_loops=1)
    print(f"  LIF 训练完成: {time.perf_counter()-t0:.0f}s", flush=True)


class AutoGuide:
    """自回归引导器: 有状态 (前缀), margin 门控, 放弃后不再接管"""

    def __init__(self, mem, in_codes, first_code, gate=0.0):
        self.mem = mem
        self.in_codes = in_codes
        self.prefix = [first_code]
        self.gate = gate
        self.abandoned = False
        self.guided = 0
        self.gated_out = 0

    def __call__(self, step):
        if self.abandoned:
            return None
        code, margin = self.mem.next_char(self.in_codes, self.prefix)
        if code is None or margin < self.gate:
            # 门控回退: 放弃自回归, 后续交还位置头/W_seq (避免前缀漂移)
            self.abandoned = True
            self.gated_out += 1
            return None
        self.prefix.append(code)
        self.guided += 1
        return code


def eval_dialogue(sim, mem, inp, resp, cf, gate_list, eval_n):
    """单个对话: A / C / D(gate 扫描) 对照"""
    rc = sim._text_to_codes(resp)
    if not rc:
        return None
    in_codes = sim._text_to_codes(inp)
    first_bits = sim._binary_decode(sim.W_ctx_to_first, sim._mem_feature(cf))
    first_code = sum((1 << j) for j in range(8) if first_bits[j] >= 0.5)

    rA = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                update_memory=True, use_pos_memory=True,
                                n_loops=1)
    guides = {}
    for gate in gate_list:
        g = AutoGuide(mem, in_codes, first_code, gate=gate)
        rg = sim.generate_recurrent(cf, n_steps=len(rc), max_repeat=3,
                                    update_memory=True, event_guide=g,
                                    n_loops=1)
        guides[gate] = (rg, g)
    return rA, guides, rc, first_code


def score(r, resp):
    if not r:
        return 0, 0, 0, 0
    cok = sum(a == b for a, b in zip(r, resp))
    return (1 if r[0] == resp[0] else 0,
            1 if r == resp else 0,
            cok, len(resp))


def main():
    print("=" * 64, flush=True)
    print("experiment20 — ②自回归事件记忆 + ①margin 门控 (v14.2)", flush=True)
    print("=" * 64, flush=True)

    train_dlg = load_pairs("english_pairs_1000.txt", 50)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=50)

    sim = RecurrentLIFSimulator(hidden_size=HIDDEN, output_size=8,
                                input_bias=1.0, leak=0.1, threshold=0.5,
                                reset_factor=0.3, inhibition_strength=0.2,
                                num_layers=3,
                                use_dg_separation=True, dg_k=64,
                                use_eligibility_trace=False)
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
    train_lif(sim, train_dlg)

    mem = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                    max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = sim._text_to_codes(resp)
        ic = sim._text_to_codes(inp)
        if oc and ic:
            mem.store_dialogue(ic, oc)
    print(f"  自回归记忆: {mem._n_prefixes} 条前缀关联", flush=True)

    gates = [0, 200, 400]
    # 库内评估 (前 14)
    print(f"\n[库内] 训练库前 14 对话, 端到端逐字生成:", flush=True)
    stats = {g: dict(f=0, c=0, t=0, full=0) for g in gates}
    stats["A"] = dict(f=0, c=0, t=0, full=0)
    for i in range(min(14, len(train_dlg))):
        inp, resp = train_dlg[i]
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        sim.W_coact = sim._coact_snapshots[i].clone()
        _, cf = sim.encode_text_lif(inp, update_memory=True, n_loops=1)
        if cf.sum().item() == 0:
            continue
        rA, guides, rc2, _ = eval_dialogue(sim, mem, inp, resp, cf, gates, 14)
        for tag, r in (("A", rA),):
            s = stats[tag]
            sf, sfull, cok, tot = score(r, resp)
            s["f"] += sf; s["full"] += sfull; s["c"] += cok; s["t"] += tot
        for g in gates:
            r, guide = guides[g]
            s = stats[g]
            sf, sfull, cok, tot = score(r, resp)
            s["f"] += sf; s["full"] += sfull; s["c"] += cok; s["t"] += tot
    print(f"  {'配置':<16} 首字  完整  字符级  引导占比", flush=True)
    print(f"  {'A 位置头DG64':<16} {stats['A']['f']}/14  {stats['A']['full']}/14  "
          f"{stats['A']['c']/max(stats['A']['t'],1):.1%}  —", flush=True)
    for g in gates:
        s = stats[g]
        tag = "C 自回归(θ=0)" if g == 0 else f"D 门控(θ={g})"
        # 引导占比: 由最后一次调用的 guide 统计 (近似, 见下)
        print(f"  {tag:<16} {s['f']}/14  {s['full']}/14  "
              f"{s['c']/max(s['t'],1):.1%}  —", flush=True)

    # 库外评估 (未见 10 个)
    print(f"\n[库外] 未见对话 10 个 (english_pairs 50-60):", flush=True)
    out_stats = {g: dict(f=0, c=0, t=0, guided=0, total_steps=0) for g in gates}
    in_margins, out_margins = [], []
    for i, (inp, resp) in enumerate(out_dlg):
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        # 库外无快照 → W_coact 从 0 编码
        sim.reset_memory()
        _, cf = sim.encode_text_lif(inp, update_memory=True, n_loops=1)
        if cf.sum().item() == 0:
            continue
        rA, guides, rc2, _ = eval_dialogue(sim, mem, inp, resp, cf, gates, 10)
        for g in gates:
            r, guide = guides[g]
            s = out_stats[g]
            _, _, cok, tot = score(r, resp)
            s["c"] += cok; s["t"] += tot
            s["guided"] += guide.guided
            s["total_steps"] += guide.guided + guide.gated_out + 1
            # 记录库外每一步 margin (用前缀生成路径)
            ic = sim._text_to_codes(inp)
            prefix = [guide.prefix[0]]
            for stp in range(1, len(rc)):
                code, margin = mem.next_char(ic, prefix)
                if code is None:
                    break
                out_margins.append(margin)
                prefix.append(code)
    print(f"  {'配置':<16} 字符级  引导占比(接管步/总步)", flush=True)
    for g in gates:
        s = out_stats[g]
        tag = "C 自回归(θ=0)" if g == 0 else f"D 门控(θ={g})"
        ratio = s["guided"] / max(s["total_steps"], 1)
        print(f"  {tag:<16} {s['c']/max(s['t'],1):.1%}  {ratio:.0%}", flush=True)

    # 库内 margin (来自库内 C 的前缀路径)
    for i in range(min(14, len(train_dlg))):
        inp, resp = train_dlg[i]
        rc = sim._text_to_codes(resp)
        if not rc:
            continue
        ic = sim._text_to_codes(inp)
        prefix = [rc[0]]
        for t in range(1, len(rc)):
            code, margin = mem.next_char(ic, prefix)
            if code is None:
                break
            in_margins.append(margin)
            prefix.append(code)
    if in_margins and out_margins:
        print(f"\n[margin 判别力] 库内 med={np.median(in_margins):.1f} "
              f"(n={len(in_margins)}), 库外 med={np.median(out_margins):.1f} "
              f"(n={len(out_margins)}) → 门控可分性证据", flush=True)


if __name__ == "__main__":
    main()
