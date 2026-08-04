"""
experiment26.py — 动态修正神经元 (错误驱动募集)

用户指令: 实现动态神经元 — 现有网络无法达到输出预期时, 诞生新的
修正神经元, 抑制错误并引导正确通路。

机制 (纯生物约束, 无梯度/无BP/无连续信号):
  - 储备池: 预生成稀疏随机位型的"祖细胞" (类比 DG 成年神经发生
    Aimone 2014; 一次性生成, 不学习)
  - 募集: teacher forcing 扫描训练库, AR 预测 ≠ 真实字符 (输出未达
    预期) → 激活一个新修正神经元, 一次性 Hebbian 印刻:
      Δscore[wrong] -= act  (抑制错误通路 = 负 RPE 调制, RPE=-1)
      Δscore[right] += act  (引导正确通路 = 正 Hebbian, RPE=+1)
  - 解码 (WTA): 查询上下文与各细胞内积 → 激活最相似细胞 → 修正
    候选字符分数 → argmax。生成阶段无 ground truth: 修正神经元回忆
    训练期的系统性错误, 相同上下文再次预测 wrong → 被抑制引导到 right。

假说: teacher 74% > free 61% 的差中, 一部分是"同上下文稳定错同一
字符"的系统性错误 → 修正神经元可捕获 → free-run 向 teacher 靠近;
库外错误事件未学过 → 相似度低 → 修正不激活 → 不误伤。

对照: AR 纯 vs AR+修正 (库内前14 / 库外10); 募集数/修正命中统计。
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
    """动态修正神经元池: 错误驱动募集 + WTA 修正 (实验内嵌, 待验证后迁移)"""

    def __init__(self, dim, max_cells=2048, theta=0.0, alpha=1.0, seed=7):
        self.dim = dim
        self.max_cells = max_cells
        self.theta = theta
        self.alpha = alpha
        self.used = 0
        self.ctx = []    # 每细胞绑定的上下文位型 (dim 维实值)
        self.wrong = []  # 抑制的字符码
        self.right = []  # 引导的字符码
        self._n_learn = 0
        self._n_apply = 0
        self._n_correct = 0  # 修正后命中真实

    def learn(self, query, wrong, right):
        """输出未达预期 → 募集新修正神经元 (一次性 Hebbian 印刻)"""
        if self.used >= self.max_cells:
            return
        self.ctx.append(query.clone())
        self.wrong.append(wrong)
        self.right.append(right)
        self.used += 1
        self._n_learn += 1

    def apply(self, query, scores):
        """解码修正: WTA 激活最相似细胞, 抑制 wrong / 引导 right"""
        self._n_apply += 1
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
    """AR v2 解码副本 (不动核心模块): 返回 (code, margin, scores)"""
    t = len(prefix)
    if t <= 0 or t >= ar.max_pos_out:
        return None, 0.0, None
    query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
    raw = ar.E2 @ query
    lo = (ar.max_pos_in + t) * ar.block_size
    rk = raw[lo:lo + ar.block_size]
    scores = ar.p[t] @ rk
    top2 = torch.topk(scores, 2)
    return (int(top2.indices[0]),
            float(top2.values[0] - top2.values[1]), scores)


def gen(ar, corr, ic, rc, use_corr):
    """free-run 生成 (首字符外部提供); use_corr=True 时应用修正"""
    prefix = [rc[0]]
    out = [rc[0]]
    for t in range(1, len(rc)):
        query = ar.bind(ic, 0) + ar.bind(prefix, ar.max_pos_in)
        code, _, scores = next_char_scores(ar, ic, prefix)
        if code is None:
            break
        if use_corr:
            scores = corr.apply(query, scores)
            code = int(scores.argmax())
        out.append(code)
        prefix.append(code)
    return out


def main():
    print("=" * 66, flush=True)
    print("experiment26 — 动态修正神经元 (错误驱动募集)", flush=True)
    print("=" * 66, flush=True)

    train_dlg = load_pairs("english_pairs_1000.txt", 50)
    out_dlg = load_pairs("english_pairs_1000.txt", 10, offset=50)

    ar = AutoRegressiveEventMemory(dim=12288, char_ones=8, max_pos_in=32,
                                   max_pos_out=32, seed=7)
    for inp, resp in train_dlg:
        oc = text_to_codes(resp)
        ic = text_to_codes(inp)
        if oc and ic:
            ar.store_dialogue(ic, oc)

    # ---- 募集阶段: teacher forcing 扫描训练库错误 ----
    t0 = time.perf_counter()
    corr = DynCorr(ar.dim, max_cells=2048, theta=0.0, alpha=1.0, seed=7)
    n_err = 0
    for inp, resp in train_dlg:
        ic = text_to_codes(inp)
        rc = text_to_codes(resp)
        if not ic or not rc:
            continue
        for t in range(1, len(rc)):
            query = ar.bind(ic, 0) + ar.bind(rc[:t], ar.max_pos_in)
            code, _, _ = next_char_scores(ar, ic, rc[:t])
            if code is None:
                break
            if code != rc[t]:
                n_err += 1
                corr.learn(query, code, rc[t])
    print(f"  募集: {n_err} 个错误事件 → {corr.used} 个修正神经元 "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- 评估: AR 纯 vs AR+修正 ----
    for tag, dlg in (("库内(前14)", train_dlg[:14]),
                     ("库外(未见10)", out_dlg)):
        ok_pure = ok_fix = tot = 0
        hit = miss = noact = 0
        for inp, resp in dlg:
            ic = text_to_codes(inp)
            rc = text_to_codes(resp)
            if not ic or not rc:
                continue
            g1 = gen(ar, corr, ic, rc, use_corr=False)
            g2 = gen(ar, corr, ic, rc, use_corr=True)
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
                if a != b:  # 修正改变了决策
                    if b == rc[i]:
                        hit += 1
                    else:
                        miss += 1
        print(f"\n[{tag}] 字符级:", flush=True)
        print(f"  AR 纯    : {ok_pure}/{tot} = {ok_pure/max(tot,1):.1%}", flush=True)
        print(f"  AR+修正  : {ok_fix}/{tot} = {ok_fix/max(tot,1):.1%}", flush=True)
        print(f"  修正决策 : {hit} 正确改对 / {miss} 正确改错 / "
              f"{tot - hit - miss} 未改动", flush=True)


if __name__ == "__main__":
    main()
