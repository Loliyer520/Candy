# -*- coding: utf-8 -*-
"""P55 v3.9 验收 — 200 条多块续写训练 (扩充词表 15×15).

在 v3.9 N-slot 多块路由架构 (nslot_v39 模块) 上, 用扩充词表
15 形容词 × 15 动物构建 200 对多块续写语料训练, 留出新组合验收
组合泛化与容量。4 块 (8-slot) 目标 ~76-82 字, 超过 v3.9 60 字目标。

运行: python _p55_v39_200accept.py  (工作目录含 nslot_v39.py / slot_route_v38.py)
"""
import os
import time
import random
from collections import defaultdict

import torch

import nslot_v39 as v39
import slot_route_v38 as sr

random.seed(1)
torch.manual_seed(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = open(os.path.join(_HERE, "_p55_result.txt"), "w", encoding="utf-8")


def log(m):
    _LOG.write(str(m) + "\n")
    _LOG.flush()


def prog(m):
    print(str(m), flush=True)


MODEL_BASE = os.path.join(_HERE, "_p37_v38_96sim.pt")
MODEL_OUT = os.path.join(_HERE, "_p55_nslot_200.pt")


# ---- 词表: 15×15 = 225 (扩充, 排除 v3.8 已用 100 拉丁方), 支持 200 训练 + 留出 ----
ADJS = v39.EXT_ADJS
ANS = v39.EXT_ANS
log(f"[corpus] 词表 adj={len(ADJS)} x an={len(ANS)} = {len(ADJS) * len(ANS)} 配对")


# ---- 构建 4 块 (8-slot) 语料: 每句 4 动物 + 4 形容词 ----
def build(n_blocks, n_train, n_held, seed):
    rng = random.Random(seed)
    seen = set()
    combos = []
    while len(combos) < n_train + n_held:
        ans_s = rng.sample(ANS, n_blocks)
        adjs_s = rng.sample(ADJS, n_blocks)
        key = (tuple(adjs_s), tuple(ans_s))
        if key in seen:
            continue
        seen.add(key)
        combos.append(list(zip(adjs_s, ans_s)))
    pairs = [v39.make_pair(c) for c in combos]
    train, held = pairs[:n_train], pairs[n_train:]
    return train, held


def coverage(pairs, sigma, n_roles):
    cov = defaultdict(int)
    for inp, _ in pairs:
        words = inp.split()
        for r in range(1, n_roles + 1):
            cov[(r, words[sigma[r]])] += 1
    return cov


def spell_report(gen, train, words, cls, slots):
    log(f"\n   == 跨协议拼写测试 (cls={cls}) ==")
    cov = coverage(train, gen.sigma, gen.n_roles)
    n_ok = 0
    for w in words:
        got = v39.spell_word(gen.sim, w, cls, gen.n_roles)
        want = w + " " if cls == "lead" else (w if cls == "mid" else " " + w + "!")
        ok = (got == want)
        n_ok += ok
        mark = "OK  " if ok else "FAIL"
        log(f"   [{mark}] {w:6s} 在位覆盖{cov.get(w, 0):2d}次 -> {got!r}")
    log(f"   拼写 {n_ok}/{len(words)}")
    return n_ok, len(words)


def prefix_probe(gen, pairs, fracs=(0.25, 0.5, 0.75)):
    log("\n   == 前缀续写探针 ==")
    total = 0
    hits = 0
    for inp, tgt in pairs[:6]:
        for f in fracs:
            p = max(1, min(len(tgt) - 1, int(len(tgt) * f)))
            o = gen.generate(inp, prefix=tgt[:p])
            ok = (o == tgt)
            total += 1
            hits += ok
            if not ok:
                log(f"   [prefix P={p}] {inp!r} -> {o!r} (want {tgt!r})")
    log(f"   前缀续写: {hits}/{total}")
    return hits, total


def main():
    t0 = time.perf_counter()
    log("[P55] v3.9 验收 — 200 条多块续写训练 (扩充 15×15 词表)")
    log(f"[base] {os.path.basename(MODEL_BASE)}")
    log(f"[模块] nslot_v39 (P54d 统一 cont + 统一 done + '!' 归模板)")

    n_blocks = 4
    train, held = build(n_blocks, 200, 24, seed=13)
    lens = [len(t) for _, t in train + held]
    log("=" * 72)
    log(f"[4-block] 8-slot | 训练 200 + 留出 {len(held)} | "
        f"tgt {min(lens)}-{max(lens)} 字")
    for inp, tgt in train[:2]:
        log(f"   例: {inp!r}")
        log(f"       -> {tgt!r}")

    prog("[P55] from_base... (200 train, 4-block)")
    gen = v39.NSlotGenerator.from_base(MODEL_BASE, train, held)
    log(f"   sigma = {gen.sigma}")
    log(f"   n_roles = {gen.n_roles}")
    for r in range(1, gen.n_roles + 1):
        ws = [w for (rr, w) in coverage(train, gen.sigma, gen.n_roles) if rr == r]
        mn = min((coverage(train, gen.sigma, gen.n_roles)[(r, w)] for w in ws),
                 default=0)
        log(f"   [coverage] slot{r}: {len(ws)}/15 词, 最少 {mn} 次")

    log("")
    gen.init_heads()
    log(f"   头维度: W_cont F+3={gen.sim.W_cont.shape}")

    iters = 60
    gen.train(iters=iters, log_every=10, log_fn=log)
    log(f"   训练耗时 {time.perf_counter() - t0:.0f}s")

    ftr, fails_tr = gen.evaluate(train, tag="训练内", log_fn=log)
    fhe, fails_he = gen.evaluate(held, tag="留出", log_fn=log)
    prog(f"[P55] train {ftr}/{len(train)}, held {fhe}/{len(held)} "
         f"({time.perf_counter() - t0:.0f}s)")

    if fails_he:
        inp, o, tgt = fails_he[0]
        log(f"\n   [trace] 首个留出失败 {inp!r} 路由轨迹:")
        tr = []
        gen.generate(inp, trace=tr)
        for e in tr:
            log(f"      step={e['step']:3d} ch={e['ch']!r} "
                f"cur_role={e['cur_role']} r_pred={e['r_pred']} "
                f"done={e['done']}")

    # 拼写测试: mid 类 (4块下 mid 词为主)
    spell_report(gen, train, ANS, "mid", list(range(2, gen.n_roles, 2)))
    prefix_probe(gen, held)

    gen.save(MODEL_OUT)
    log(f"   模型已保存: {os.path.basename(MODEL_OUT)}")
    log(f"总耗时 {time.perf_counter() - t0:.0f}s")
    prog("[P55] ALL DONE")


if __name__ == "__main__":
    main()