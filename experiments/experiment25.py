"""
experiment25.py — 规模 × 训练量扫描 (P1 / AR / 混合B)

用户指令: "略微扩大规模, 并增加训练量测试"。

维度 1 规模: 训练库 50 (基线, experiment23) → 100 → 200, 库内前 14 评估,
            库外取 offset 之后的未见 10 对话。
维度 2 训练量: 重复印刻 n× (同一对话多次 Hebbian 写入)。
            ★ 关键假说: 纯叠加结构 (E += outer) 下重复印刻只放大权重,
            不改变 argmax 解码 → 训练量在该架构中无效 (容量取决于
            对话数, 而非每对话写入次数)。用库100 P1 1× vs 5× 验证。

评估: 字符级 (P1 纯轨道 / AR 纯 free-run / 混合 B θ=800)。
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


def gen_hybrid(p1, ar, ic, rc, theta=800.0):
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


def build(n_train, repeat=1):
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    p1 = EpisodicEventMemory(dim=8192, char_ones=8, max_pos=32, seed=7)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if oc and ic:
            for _ in range(repeat):
                p1.store(ic, oc)
                ar.store_dialogue(ic, oc)
    return p1, ar, train_dlg, out_dlg


def eval_all(p1, ar, dlg):
    res = {"P1": [0, 0], "AR": [0, 0], "B": [0, 0]}
    for inp, resp in dlg:
        rc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if not rc or not ic:
            continue
        for name, g in (("P1", gen_p1(p1, ic, rc)),
                        ("AR", gen_ar(ar, ic, rc)),
                        ("B", gen_hybrid(p1, ar, ic, rc))):
            res[name][0] += sum(a == b for a, b in zip(g, rc))
            res[name][1] += len(rc)
    return res


def main():
    print("=" * 66, flush=True)
    print("experiment25 — 规模 × 训练量扫描 (P1 / AR / 混合B)", flush=True)
    print("=" * 66, flush=True)
    print("基线 (库50, experiment23): P1 64.1% / AR 61.3% / B 65.3% "
          "(库内14); 19.5% / 23.4% / 24.1% (库外10)", flush=True)

    for n in (100, 200):
        t0 = time.perf_counter()
        p1, ar, train_dlg, out_dlg = build(n)
        ri = eval_all(p1, ar, train_dlg[:14])
        ro = eval_all(p1, ar, out_dlg)
        print(f"\n[库{n}]  P1 {p1._n_events} 事件, AR {ar._n_prefixes} 前缀 "
              f"(构建 {time.perf_counter()-t0:.0f}s)", flush=True)
        print(f"  库内(前14): P1 {ri['P1'][0]}/{ri['P1'][1]} = "
              f"{ri['P1'][0]/max(ri['P1'][1],1):.1%} | "
              f"AR {ri['AR'][0]}/{ri['AR'][1]} = "
              f"{ri['AR'][0]/max(ri['AR'][1],1):.1%} | "
              f"B {ri['B'][0]}/{ri['B'][1]} = "
              f"{ri['B'][0]/max(ri['B'][1],1):.1%}", flush=True)
        print(f"  库外(10) : P1 {ro['P1'][0]}/{ro['P1'][1]} = "
              f"{ro['P1'][0]/max(ro['P1'][1],1):.1%} | "
              f"AR {ro['AR'][0]}/{ro['AR'][1]} = "
              f"{ro['AR'][0]/max(ro['AR'][1],1):.1%} | "
              f"B {ro['B'][0]}/{ro['B'][1]} = "
              f"{ro['B'][0]/max(ro['B'][1],1):.1%}", flush=True)

    # 训练量诊断: 库100 重复印刻 5× 是否改变解码 (假说: 不变)
    t0 = time.perf_counter()
    p1a, ar, train_dlg, out_dlg = build(100, repeat=1)
    p1b, _, _, _ = build(100, repeat=5)
    r1 = eval_all(p1a, ar, train_dlg[:14])["P1"]
    r5 = eval_all(p1b, ar, train_dlg[:14])["P1"]
    print(f"\n[训练量诊断 库100] P1 1× {r1[0]}/{r1[1]} = "
          f"{r1[0]/max(r1[1],1):.1%} vs 5× {r5[0]}/{r5[1]} = "
          f"{r5[0]/max(r5[1],1):.1%} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print("  → 若 1× == 5×: 纯叠加下重复印刻无效, 训练量=存储对话数", flush=True)


if __name__ == "__main__":
    main()
