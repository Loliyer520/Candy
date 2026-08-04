"""
experiment28.py — AR 错误本质诊断: 语义 vs 马尔可夫

用户提问: 神经元错误是"语义错误导致稳定错误"还是"类似马尔可夫链仅能
通过上一字预测"?

判别设计 (库50, teacher forcing, 库内前14):
  1. 上下文消融 (解码位置固定为 t, 改变查询上下文):
     完整前缀 rc[:t]           → 长程历史 + 输入 (基准)
     仅输入 (无前缀)            → 输入语义标签的单独预测力
     仅前缀 (无输入)            → 前缀转移概率 (v1 的裸前缀, 输入消融)
     位置感知 last-k           → 只给最近 k 字符 (保留绝对位置)
     位置无关 last-k           → 只给最近 k 字符内容 (纯 n-gram 类比)
  2. 马尔可夫基线 (训练库频率, 不看输入):
     1/2/3-gram argmax + 输入标签频率 P(c|input) (无位置)
  判别规则:
     完整 ≈ last-1 / last-2 且 > n-gram → 马尔可夫式 (只靠最近字符)
     完整 >> last-k            → 长程上下文/语义参与
     仅前缀 ≈ 完整             → 输入语义无用 (马尔可夫式)
     仅前缀 << 完整            → 输入 (语义情境) 是关键
     AR >> 3-gram 基线         → 超越纯转移统计, 有语义成分
"""

import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from episodic_memory import AutoRegressiveEventMemory


def text_to_codes(text):
    return [ord(c) for c in text if 32 <= ord(c) <= 126]


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


def decode_at(ar, ic, ctx_chars, ctx_offset, t):
    """位置 t 解码; ctx_chars 可空; 返回 (code, scores)"""
    query = ar.bind(ic, 0)
    if ctx_chars:
        query = query + ar.bind(ctx_chars, ctx_offset)
    raw = ar.E2 @ query
    lo = (ar.max_pos_in + t) * ar.block_size
    rk = raw[lo:lo + ar.block_size]
    scores = ar.p[t] @ rk
    return int(scores.argmax())


def main():
    print("=" * 66, flush=True)
    print("experiment28 — AR 错误本质: 语义 vs 马尔可夫", flush=True)
    print("=" * 66, flush=True)

    train_dlg = load_pairs("english_pairs_1000.txt", 50)
    in14 = train_dlg[:14]

    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if oc and ic:
            ar.store_dialogue(ic, oc)

    # 预收集评估样本 (ic, rc) 及 t
    samples = []
    for inp, resp in in14:
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc:
            continue
        for t in range(1, len(rc)):
            if t >= ar.max_pos_out:
                break
            samples.append((ic, rc, t))
    n = len(samples)
    print(f"  评估样本: {n} 个 (t≥1 且 t<32)", flush=True)

    # ---- 1. 上下文消融 ----
    print("\n[上下文消融] 预测命中 / 总 (库内前14, teacher forcing):",
          flush=True)
    configs = [
        ("完整前缀 rc[:t]",          "full"),
        ("仅输入 (无前缀)",          "in_only"),
        ("仅前缀 rc[:t] (无输入)",   "pref_only"),
        ("位置感知 last-1",          "pl1"),
        ("位置感知 last-3",          "pl3"),
        ("位置无关 last-1 (内容)",   "nl1"),
        ("位置无关 last-3 (内容)",   "nl3"),
    ]
    res = {}
    for name, key in configs:
        ok = 0
        for ic, rc, t in samples:
            if key == "full":
                code = decode_at(ar, ic, rc[:t], ar.max_pos_in, t)
            elif key == "in_only":
                code = decode_at(ar, ic, None, 0, t)
            elif key == "pref_only":
                code = decode_at(ar, [], rc[:t], ar.max_pos_in, t)
            elif key == "pl1":
                code = decode_at(ar, ic, rc[t-1:t], ar.max_pos_in + t - 1, t)
            elif key == "pl3":
                k = min(3, t)
                code = decode_at(ar, ic, rc[t-k:t], ar.max_pos_in + t - k, t)
            elif key == "nl1":
                code = decode_at(ar, ic, rc[t-1:t], ar.max_pos_in, t)
            else:  # nl3
                k = min(3, t)
                code = decode_at(ar, ic, rc[t-k:t], ar.max_pos_in, t)
            ok += (code == rc[t])
        res[key] = ok
        print(f"  {name:<24} {ok}/{n} = {ok/n:.1%}", flush=True)

    # ---- 2. 马尔可夫基线 (训练库频率) ----
    print("\n[马尔可夫基线] 训练库频率 argmax (不看输入):", flush=True)
    n1 = defaultdict(lambda: defaultdict(int))
    n2 = defaultdict(lambda: defaultdict(int))
    n3 = defaultdict(lambda: defaultdict(int))
    inp_freq = defaultdict(lambda: defaultdict(int))
    for inp, resp in train_dlg:
        rc = text_to_codes(resp)
        ic = text_to_codes(inp)
        for i in range(1, len(rc)):
            n1[(rc[i-1],)][rc[i]] += 1
            if i >= 2:
                n2[(rc[i-2], rc[i-1])][rc[i]] += 1
            if i >= 3:
                n3[(rc[i-3], rc[i-2], rc[i-1])][rc[i]] += 1
        for i, c in enumerate(rc):
            inp_freq["".join(chr(x) for x in ic)][c] += 1

    def ngram_eval(order, table):
        ok = 0
        for ic, rc, t in samples:
            ctx = rc[:t]
            if len(ctx) >= order:
                key = tuple(ctx[-order:])
                if key in table and table[key]:
                    code = max(table[key].items(), key=lambda x: x[1])[0]
                    ok += (code == rc[t])
        return ok

    for tag, order, table in (("1-gram (上一字)", 1, n1),
                              ("2-gram", 2, n2), ("3-gram", 3, n3)):
        ok = ngram_eval(order, table)
        print(f"  {tag:<16} {ok}/{n} = {ok/n:.1%}", flush=True)
    # 输入标签频率: P(c|input) 无位置
    ok = 0
    for ic, rc, t in samples:
        inp_key = "".join(chr(x) for x in ic)
        code = max(inp_freq[inp_key].items(), key=lambda x: x[1])[0]
        ok += (code == rc[t])
    print(f"  {'输入标签P(c|input)':<16} {ok}/{n} = {ok/n:.1%}", flush=True)


if __name__ == "__main__":
    main()
