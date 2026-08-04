"""
experiment21.py — 自回归误差累积诊断 (teacher forcing vs free-run)

假说 (experiment20 v2 后): 库内自回归 53.9% < P1 整体回忆 63.7% 的主
瓶颈是误差累积 — 自回归每步用"已生成前缀"查询, 一旦错一个字符, 前缀
绑定偏离 → 后续查询全偏 (错误传播)。

诊断: 同一 mem (v2 上下文消歧), 对比两种查询路径:
  - teacher forcing: 查询用真实前缀 out_codes[:t] → 单步能力上限
  - free-run: 查询用已生成前缀 (experiment20 方式) → 实际端到端
  - 1-gram 基线: 忽略上下文, 取库内边缘频率最高的字符 → 随机下限参照

  teacher >> free → 误差累积是主瓶颈 → 修复方向是"约束生成路径"
  (P1 整句模板候选 / 高 margin 锚点 / 前缀回退), 而非改关联本身。
  teacher ≈ free → 关联本身容量不足 (块大小/位型/长尾), 方向是改存储。

纯事件记忆 (无 LIF 底座, store_dialogue 只吃字符码) → 秒级运行。
与 experiment20 同数据同 mem 构造 (dim=12288, max_pos_in/out=32)。
"""

import sys, os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from episodic_memory import AutoRegressiveEventMemory


def text_to_codes(text):
    """同 lif_pytorch._text_to_codes: 可打印 ASCII"""
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


def main():
    print("=" * 64, flush=True)
    print("experiment21 — 自回归误差累积诊断 (teacher vs free-run)", flush=True)
    print("=" * 64, flush=True)

    train_dlg = load_pairs("english_pairs_1000.txt", 50)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=50)

    mem = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                    max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if oc and ic:
            mem.store_dialogue(ic, oc)
    print(f"  前缀关联: {mem._n_prefixes} 条", flush=True)

    # 1-gram 基线: 库内回复的边缘字符频率 (含首字符偏置)
    freq = Counter()
    for inp, resp in train_dlg:
        for c in text_to_codes(resp):
            freq[c] += 1
    top1 = freq.most_common(1)[0][0]
    print(f"  1-gram 基线: 常预测字符={chr(top1)!r} ({freq[top1]} 次)", flush=True)

    for tag, dlg in (("库内(前14)", train_dlg[:14]),
                     ("库外(未见10)", out_dlg)):
        tf_ok = tf_tot = fr_ok = fr_tot = g_ok = g_tot = 0
        tf_m, fr_m = [], []
        for inp, resp in dlg:
            rc = text_to_codes(resp)
            ic = text_to_codes(inp)
            if not rc or not ic:
                continue
            # teacher forcing: 真实前缀 rc[:t]
            for t in range(1, len(rc)):
                code, margin = mem.next_char(ic, rc[:t])
                if code is None:
                    break
                tf_ok += (code == rc[t]); tf_tot += 1
                tf_m.append(margin)
            # free-run: 已生成前缀 (首字符 rc[0] 由外部链路提供)
            prefix = [rc[0]]
            for t in range(1, len(rc)):
                code, margin = mem.next_char(ic, prefix)
                if code is None:
                    break
                fr_ok += (code == rc[t]); fr_tot += 1
                fr_m.append(margin)
                prefix.append(code)
            # 1-gram 基线
            for t in range(1, len(rc)):
                g_ok += (top1 == rc[t]); g_tot += 1
        print(f"\n[{tag}]", flush=True)
        print(f"  teacher forcing: {tf_ok}/{tf_tot} = {tf_ok/max(tf_tot,1):.1%} "
              f"(margin med={sorted(tf_m)[len(tf_m)//2] if tf_m else 0:.0f})",
              flush=True)
        print(f"  free-run      : {fr_ok}/{fr_tot} = {fr_ok/max(fr_tot,1):.1%} "
              f"(margin med={sorted(fr_m)[len(fr_m)//2] if fr_m else 0:.0f})",
              flush=True)
        print(f"  1-gram 基线   : {g_ok}/{g_tot} = {g_ok/max(g_tot,1):.1%}",
              flush=True)
        if tf_tot:
            print(f"  → 误差累积损失: teacher - free-run = "
                  f"{(tf_ok/tf_tot - fr_ok/max(fr_tot,1)):.1%}", flush=True)

    # 分位置分解 (teacher forcing): 前 8 字符 vs 后半段 — 关联是否随前缀增长衰减
    print("\n[分位置 teacher forcing 单步准确率 (库内)]:", flush=True)
    pos_ok = {}
    pos_tot = {}
    for inp, resp in train_dlg[:14]:
        rc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if not rc or not ic:
            continue
        for t in range(1, len(rc)):
            code, _ = mem.next_char(ic, rc[:t])
            if code is None:
                break
            pos_ok[t] = pos_ok.get(t, 0) + (code == rc[t])
            pos_tot[t] = pos_tot.get(t, 0) + 1
    for t in sorted(pos_tot):
        if pos_tot[t] >= 5:
            print(f"  pos {t:>2}: {pos_ok[t]}/{pos_tot[t]} = "
                  f"{pos_ok[t]/pos_tot[t]:.0%}", flush=True)


if __name__ == "__main__":
    main()
