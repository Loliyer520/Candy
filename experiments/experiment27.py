"""
experiment27.py — 动态修正神经元规模测试 + 对话效果

用户指令: "进行规模训练测试对话效果"。

规模: 训练库 50 (基线, experiment26: 59.5→67.3%) → 100 → 200。
每规模:
  1. AR v2 存储 + teacher forcing 募集修正神经元 (DynCorr, θ=0/α=1)
  2. 库内前14 / 库外10: AR 纯 vs AR+修正 (字符级 + 修正决策统计)
  3. 对话文本展示: 库内 #1 / 库外 #1 (AR 纯 vs AR+修正, [真→预测] 标注)

假说: 规模增大时修正神经元依然捕获系统性错误 (库内提升), 但募集
容量 (max_cells) 与上下文相似度可能饱和 → 观察提升幅度衰减。
"""

import sys, os, time
import torch

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


class DynCorr:
    """动态修正神经元池 (experiment26 验证有效, 内嵌副本)"""

    def __init__(self, dim, max_cells=4096, theta=0.0, alpha=1.0):
        self.dim = dim
        self.max_cells = max_cells
        self.theta = theta
        self.alpha = alpha
        self.used = 0
        self.ctx = []
        self.wrong = []
        self.right = []

    def learn(self, query, wrong, right):
        if self.used >= self.max_cells:
            return
        self.ctx.append(query.clone())
        self.wrong.append(wrong)
        self.right.append(right)
        self.used += 1

    def apply(self, query, scores):
        best_act = -1.0
        best_i = -1
        for i in range(self.used):
            act = float((self.ctx[i] * query).sum())
            if act > best_act:
                best_act, best_i = act, i
        if best_i >= 0 and best_act > self.theta:
            w, r = self.wrong[best_i], self.right[best_i]
            scores[w] -= best_act * self.alpha
            scores[r] += best_act * self.alpha
        return scores


def next_char_scores(ar, ic, prefix):
    t = len(prefix)
    if t <= 0 or t >= ar.max_pos_out:
        return None, None
    query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
    raw = ar.E2 @ query
    lo = (ar.max_pos_in + t) * ar.block_size
    rk = raw[lo:lo + ar.block_size]
    scores = ar.p[t] @ rk
    return int(scores.argmax()), scores


def gen(ar, corr, ic, rc, use_corr):
    prefix = [rc[0]]
    out = [rc[0]]
    for t in range(1, len(rc)):
        code, scores = next_char_scores(ar, ic, prefix)
        if code is None:
            break
        if use_corr:
            query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
            scores = corr.apply(query, scores)
            code = int(scores.argmax())
        out.append(code)
        prefix.append(code)
    return out


def render(codes, rc):
    out = []
    for a, b in zip(codes, rc):
        ch = chr(b) if b != 32 else "·"
        if a == b:
            out.append(ch)
        else:
            out.append(f"[{chr(a) if a != 32 else '·'}→{ch}]")
    return "".join(out)


def run_scale(n_train):
    t0 = time.perf_counter()
    train_dlg = load_pairs("english_pairs_1000.txt", n_train)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=n_train)
    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if oc and ic:
            ar.store_dialogue(ic, oc)
    print(f"  AR 存储: {ar._n_prefixes} 前缀 ({time.perf_counter()-t0:.0f}s)",
          flush=True)

    # 募集
    t1 = time.perf_counter()
    corr = DynCorr(ar.dim, max_cells=4096)
    n_err = 0
    for inp, resp in train_dlg:
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc:
            continue
        for t in range(1, len(rc)):
            code, _ = next_char_scores(ar, ic, rc[:t])
            if code is None:
                break
            if code != rc[t]:
                n_err += 1
                query = ar.bind(ic, 0) + ar.bind(rc[:t], ar.max_pos_in)
                corr.learn(query, code, rc[t])
    print(f"  募集: {n_err} 错误 → {corr.used} 修正神经元 "
          f"({time.perf_counter()-t1:.0f}s)", flush=True)

    # 评估
    for tag, dlg in (("库内(前14)", train_dlg[:14]), ("库外(10)", out_dlg)):
        ok_pure = ok_fix = tot = 0
        hit = miss = 0
        for inp, resp in dlg:
            ic = text_to_codes(inp)
            rc = text_to_codes(resp)
            if not ic or not rc:
                continue
            g1 = gen(ar, corr, ic, rc, False)
            g2 = gen(ar, corr, ic, rc, True)
            for i in range(1, len(rc)):
                a = g1[i] if i < len(g1) else None
                b = g2[i] if i < len(g2) else None
                if a is None or b is None:
                    break
                if a == rc[i]:
                    ok_pure += 1
                if b == rc[i]:
                    ok_fix += 1
                tot += 1
                if a != b:
                    hit += (b == rc[i])
                    miss += (b != rc[i])
        print(f"  [{tag}] AR 纯 {ok_pure}/{tot} = {ok_pure/max(tot,1):.1%} | "
              f"AR+修正 {ok_fix}/{tot} = {ok_fix/max(tot,1):.1%} | "
              f"修正 {hit}对/{miss}错", flush=True)

    # 对话展示 (库内 #1, 库外 #1)
    for tag, dlg in (("库内#1", train_dlg[:14]), ("库外#1", out_dlg)):
        inp, resp = dlg[0]
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc or len(rc) > 40:
            continue
        g1 = gen(ar, corr, ic, rc, False)
        g2 = gen(ar, corr, ic, rc, True)
        n = len(rc)
        ok1 = sum(a == b for a, b in zip(g1, rc))
        ok2 = sum(a == b for a, b in zip(g2, rc))
        print(f"\n  [{tag}] 输入: {inp}", flush=True)
        print(f"    真实  : {resp}", flush=True)
        print(f"    AR纯  : {render(g1, rc)}  [{ok1}/{n}]", flush=True)
        print(f"    AR+修正: {render(g2, rc)}  [{ok2}/{n}]", flush=True)
    print(f"  完成 (总 {time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    print("=" * 66, flush=True)
    print("experiment27 — 动态修正神经元规模测试 + 对话效果", flush=True)
    print("=" * 66, flush=True)
    print("基线 (库50, experiment26): 库内 59.5→67.3% (+7.8pp), "
          "库外 21.1→21.5% (+0.4pp)", flush=True)
    for n in (100, 200):
        print(f"\n[库{n}]", flush=True)
        run_scale(n)


if __name__ == "__main__":
    main()
