"""
experiment24.py — 对话生成展示 (P1 轨道 / AR free-run / 混合 B)

experiment23 最优配置的逐字生成文本展示 (库内前 6 + 库外 5 对话)。
纯事件记忆 (无 LIF 底座, 秒级), 首字符由外部链路提供 (真值)。
错误标注: [真→预测]; 空格显示为 ·; 每行附字符级得分。

配置:
  P1 纯轨道   — EpisodicEventMemory 整体回忆
  AR 纯       — AutoRegressiveEventMemory v2 free-run
  混合 B θ=800 — AR 优先, 高 margin 修正 P1 轨道 (experiment23 库内最优)
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from episodic_memory import EpisodicEventMemory, AutoRegressiveEventMemory


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


def gen_p1(p1, ic, rc):
    codes, _ = p1.recall_with_margin(ic, len(rc))
    return codes


def gen_ar(ar, ic, rc):
    prefix = [rc[0]]
    for t in range(1, len(rc)):
        code, _ = ar.next_char(ic, prefix)
        if code is None:
            break
        prefix.append(code)
    return prefix


def gen_hybrid(p1, ar, ic, rc, theta):
    p1_seq, _ = p1.recall_with_margin(ic, len(rc))
    prefix = [rc[0]]
    for t in range(1, len(rc)):
        ar_code, ar_m = ar.next_char(ic, prefix)
        p1_code = p1_seq[t] if t < len(p1_seq) else None
        if ar_code is None:
            code = p1_code
        elif p1_code is None:
            code = ar_code
        else:
            code = ar_code if (ar_code == p1_code or ar_m >= theta) else p1_code
        prefix.append(code)
    return prefix


def render(codes, rc):
    """逐字符标注: 一致显示真值, 不一致 [真→预测]"""
    out = []
    for a, b in zip(codes, rc):
        ch = chr(b) if b != 32 else "·"
        if a == b:
            out.append(ch)
        else:
            out.append(f"[{chr(a) if a != 32 else '·'}→{ch}]")
    return "".join(out)


def show(tag, ic, rc, p1, ar, theta):
    g1 = gen_p1(p1, ic, rc)
    g2 = gen_ar(ar, ic, rc)
    g3 = gen_hybrid(p1, ar, ic, rc, theta)
    print(f"\n=== {tag} ===", flush=True)
    print(f"  输入  : {''.join(chr(c) for c in ic)}", flush=True)
    print(f"  真实  : {''.join(chr(c) for c in rc)}", flush=True)
    ok1 = sum(a == b for a, b in zip(g1, rc))
    ok2 = sum(a == b for a, b in zip(g2, rc))
    ok3 = sum(a == b for a, b in zip(g3, rc))
    n = len(rc)
    print(f"  P1 纯 : {render(g1, rc)}   [{ok1}/{n} = {ok1/n:.0%}]", flush=True)
    print(f"  AR 纯 : {render(g2, rc)}   [{ok2}/{n} = {ok2/n:.0%}]", flush=True)
    print(f"  混合B : {render(g3, rc)}   [{ok3}/{n} = {ok3/n:.0%}]", flush=True)


def main():
    print("=" * 66, flush=True)
    print("experiment24 — 对话生成展示 (P1 轨道 vs AR vs 混合B θ=800)", flush=True)
    print("=" * 66, flush=True)

    train_dlg = load_pairs("english_pairs_1000.txt", 50)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=50)

    p1 = EpisodicEventMemory(dim=8192, char_ones=8, max_pos=32, seed=7)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if oc and ic:
            p1.store(ic, oc)
            ar.store_dialogue(ic, oc)

    shown = 0
    for i, (inp, resp) in enumerate(train_dlg[:14]):
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc or len(rc) > 34:
            continue
        show(f"库内 #{i+1} (训练对话)", ic, rc, p1, ar, 800.0)
        shown += 1
        if shown >= 4:
            break
    shown = 0
    for i, (inp, resp) in enumerate(out_dlg):
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc or len(rc) > 34:
            continue
        show(f"库外 #{i+51} (未见对话)", ic, rc, p1, ar, 800.0)
        shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    main()
