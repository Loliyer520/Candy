# -*- coding: utf-8 -*-
"""P51 v3.8 验收 — 60 条续写训练 (扩充词表的 2-slot 拉丁方)。

在 v3.8 slot 角色路由架构上, 用扩充的形容词/动物词表生成 60 对续写语料
训练, 留出新组合验收组合泛化与容量。复用 slot_route_v38 模块
from_base(_p37 基础配方) 统一训练全部头。
"""
import os
import time
import random

import torch

import slot_route_v38 as sr

random.seed(1)
torch.manual_seed(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = open(os.path.join(_HERE, "_p51_result.txt"), "w", encoding="utf-8")


def log(msg):
    _LOG.write(msg + "\n")
    _LOG.flush()


# ---- 扩充词表 15×15=225 组合 ----
ADJS = sr.ADJS10 + ["dark", "brave", "smart", "quiet", "gold"]
ANS = sr.ANS10 + ["tiger", "horse", "mouse", "duck", "goat"]

CORPUS_ALL = [(f"{a} {an}", f"the {an} is {a}!") for a in ADJS for an in ANS]
assert len(set(k for k, _ in CORPUS_ALL)) == 225, "键歧义!"

# 去掉已在 P49 验证过的 100 拉丁方, 保留全新组合 (真正 OOD 泛化验收)
KNOWN100 = set(sr.TRAIN96) | set(sr.HELD4)
NEW = [p for p in CORPUS_ALL if p not in KNOWN100]   # 125 全新组合
log(f"[corpus] 全部={len(CORPUS_ALL)} 全新(排除P49已见100)={len(NEW)}")

# 语义正确、语法合理、且为全新组合的留出
HELD_FINAL = [("red tiger", "the tiger is red!"),
              ("gold goat", "the goat is gold!"),
              ("smart fox", "the fox is smart!"),
              ("quiet bear", "the bear is quiet!"),
              ("blue horse", "the horse is blue!"),
              ("dark wolf", "the wolf is dark!")]
HELD_FINAL = [p for p in HELD_FINAL if p not in KNOWN100]
log(f"[held] 留出组合 {len(HELD_FINAL)}")

# 60 条训练: 从全新组合中随机抽, 并保证与留出不重叠
pool = [p for p in NEW if p not in set(HELD_FINAL)]
random.shuffle(pool)
TRAIN = pool[:60]
log(f"[train] 60 条 (全新组合, 排除留出)")

MODEL_BASE = os.path.join(_HERE, "_p37_v38_96sim.pt")
MODEL_OUT = os.path.join(_HERE, "_p51_slotroute_60.pt")


def eval_held(gen, tag):
    hits = 0
    for inp, tgt in HELD_FINAL:
        o = gen.generate(inp)
        hits += (o == tgt)
        log(f"   {inp!r:12} -> {o!r:28} 期望 {tgt!r} "
            f"({'MATCH' if o == tgt else 'miss'})")
    log(f"{tag}. 留出 FULL: {hits}/{len(HELD_FINAL)}")
    return hits


def main():
    t0 = time.perf_counter()
    log("[P51] v3.8 验收 — 60 条续写训练 (扩充词表 2-slot 拉丁方)")
    log(f"[training corpus] {len(TRAIN)}")
    for inp, tgt in TRAIN[:10]:
        log(f"   {inp!r:12} -> {tgt!r}")

    gen = sr.SlotRouteGenerator.from_base(MODEL_BASE, TRAIN, HELD_FINAL)
    log(f"[sigma] {gen.sigma} n_roles={gen.n_roles}")
    gen.train(iters=120, log_every=20, log_fn=log)
    gen.save(MODEL_OUT)
    log(f"[save] {MODEL_OUT}")

    full, fails = gen.evaluate(TRAIN)
    log(f"C. 训练内 FULL: {full}/{len(TRAIN)}")
    for inp, o, tgt in fails[:10]:
        log(f"   [miss] {inp!r:12} -> {o!r:28} 期望 {tgt!r}")
    eval_held(gen, "D")
    log(f"[time] total {time.perf_counter() - t0:.0f}s")
    log("DONE")
    _LOG.close()


if __name__ == "__main__":
    main()