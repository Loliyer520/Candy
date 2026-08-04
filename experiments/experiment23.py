"""
experiment23.py — P1 轨道约束自回归 (混合引导仲裁)

前提 (experiment21/22): AR teacher 单步 75.4% / free 59.5% (误差累积
13.6%), E2 线性叠加是结构性硬顶; P1 整体回忆库50 63.7% (experiment19)。

融合: P1 (EpisodicEventMemory) 提供整句轨道 recall_with_margin, AR
(AutoRegressiveEventMemory v2) 提供单步候选, 仲裁选择:
  A P1 优先: p1_margin >= θ → P1, 否则 AR
  B AR 优先: ar_margin >= θ → AR, 否则 P1
  C margin 仲裁: 两者一致取共同; 不一致取 margin 高者
基线: P1 纯 (全轨道) / AR 纯 (全 free-run)。
评估: 字符级 (库内前14 / 库外10), 纯事件记忆秒级构建。
"""

import sys, os, time

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


def gen(p1, ar, ic, rc, rule, theta):
    """返回生成序列 (首字符由外部提供), rule ∈ {P1, AR, P1first, ARfirst, C, D}
    D: P1 轨道作导航前缀 (AR 查询基于 P1 前缀, 近 teacher forcing),
       输出 = AR 高 margin 修正, 否则 P1 轨道。"""
    if rule == "P1":
        codes, _ = p1.recall_with_margin(ic, len(rc))
        return codes
    if rule == "AR":
        prefix = [rc[0]]
        for t in range(1, len(rc)):
            code, _ = ar.next_char(ic, prefix)
            if code is None:
                break
            prefix.append(code)
        return prefix
    # 混合仲裁
    p1_seq, p1_m = p1.recall_with_margin(ic, len(rc))
    prefix = [rc[0]]
    nav = [rc[0]]
    for t in range(1, len(rc)):
        ar_code, ar_m = ar.next_char(ic, nav if rule == "D" else prefix)
        p1_code = p1_seq[t] if t < len(p1_seq) else None
        p1_mg = p1_m[t] if t < len(p1_m) else 0.0
        if ar_code is None:
            code = p1_code
        elif p1_code is None:
            code = ar_code
        elif rule == "P1first":
            code = p1_code if p1_mg >= theta else ar_code
        elif rule in ("ARfirst", "D"):
            # 一致取共同; 不一致且 AR 有把握 → 修正; 否则跟随轨道
            code = ar_code if (ar_code == p1_code or ar_m >= theta) else p1_code
        else:  # C: margin 仲裁
            code = ar_code if (ar_code == p1_code or ar_m >= p1_mg) else p1_code
        prefix.append(code)
        nav.append(p1_code)
    return prefix


def char_acc(gen_codes, rc):
    if not gen_codes:
        return 0, 0
    ok = sum(a == b for a, b in zip(gen_codes, rc))
    return ok, len(rc)


def main():
    print("=" * 66, flush=True)
    print("experiment23 — P1 轨道约束自回归 (混合仲裁)", flush=True)
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
    print(f"  P1 事件: {p1._n_events}, AR 前缀: {ar._n_prefixes}", flush=True)

    configs = [
        ("P1 纯轨道",        "P1",      0.0),
        ("AR 纯 free-run",   "AR",      0.0),
        ("B AR优先 θ=200",   "ARfirst", 200.0),
        ("B AR优先 θ=400",   "ARfirst", 400.0),
        ("B AR优先 θ=800",   "ARfirst", 800.0),
        ("D P1导航 θ=200",   "D",       200.0),
        ("D P1导航 θ=400",   "D",       400.0),
        ("D P1导航 θ=800",   "D",       800.0),
        ("C margin仲裁",     "C",       0.0),
    ]
    for tag, dlg in (("库内(前14)", train_dlg[:14]), ("库外(未见10)", out_dlg)):
        res = {c[0]: [0, 0] for c in configs}
        for inp, resp in dlg:
            rc = text_to_codes(resp)
            ic = text_to_codes(inp)
            if not rc or not ic:
                continue
            for name, rule, theta in configs:
                ok, tot = char_acc(gen(p1, ar, ic, rc, rule, theta), rc)
                res[name][0] += ok; res[name][1] += tot
        print(f"\n[{tag}] 字符级:", flush=True)
        for name, _, _ in configs:
            ok, tot = res[name]
            print(f"  {name:<18} {ok}/{tot} = {ok/max(tot,1):.1%}", flush=True)


if __name__ == "__main__":
    main()
