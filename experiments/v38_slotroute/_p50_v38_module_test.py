# -*- coding: utf-8 -*-
"""P50 v3.8 模块化封装验证 — slot_route_v38 统一训练 + 复用性测试。

Phase A: from_pretrained(_p49e) 仅评估 — 验证模块 generate() 与实验链
         gen_full_v5 行为一致 (期望 96/96 + 4/4)。
Phase B: from_base(_p37) 全头统一从头训练 — 验证摆脱分阶段脚本链,
         六头一次训练收敛 (期望 96/96 + 4/4), 保存统一模型。
"""
import os
import time
import random

import torch

import slot_route_v38 as sr

random.seed(1)
torch.manual_seed(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = open(os.path.join(_HERE, "_p50_result.txt"), "w", encoding="utf-8")


def log(msg):
    _LOG.write(msg + "\n")
    _LOG.flush()


MODEL_P49E = os.path.join(_HERE, "_p49e_v38_96done.pt")
MODEL_BASE = os.path.join(_HERE, "_p37_v38_96sim.pt")
MODEL_OUT = os.path.join(_HERE, "_p50_slotroute_unified.pt")

TRAIN = sr.TRAIN96
HELD = sr.HELD4


def eval_held(gen, tag):
    hits = 0
    for inp, tgt in HELD:
        o = gen.generate(inp)
        hits += (o == tgt)
        log(f"   {inp!r:12} -> {o!r:28} 期望 {tgt!r} "
            f"({'MATCH' if o == tgt else 'miss'})")
    log(f"{tag}. 留出 FULL: {hits}/{len(HELD)}")
    return hits


def main():
    t0 = time.perf_counter()
    log("[P50] slot_route_v38 模块化封装验证")
    log(f"[corpus] train={len(TRAIN)} held={len(HELD)}")

    # ---- Phase A: 复用 _p49e 仅评估 ----
    log(f"\n== Phase A: from_pretrained({os.path.basename(MODEL_P49E)}) 仅评估 ==")
    gen = sr.SlotRouteGenerator.from_pretrained(MODEL_P49E, TRAIN, HELD)
    log(f"[sigma] {gen.sigma} n_roles={gen.n_roles} "
        f"heads_ready={gen._heads_ready}")
    full, fails = gen.evaluate(TRAIN)
    log(f"A1. 训练内 FULL: {full}/{len(TRAIN)}")
    for inp, o, tgt in fails[:5]:
        log(f"   [miss] {inp!r:12} -> {o!r:28} 期望 {tgt!r}")
    eval_held(gen, "A2")
    log(f"[time] Phase A {time.perf_counter() - t0:.0f}s")

    # ---- Phase B: 统一从头训练 ----
    log(f"\n== Phase B: from_base({os.path.basename(MODEL_BASE)}) 全头统一训练 ==")
    random.seed(1)
    torch.manual_seed(1)
    gen2 = sr.SlotRouteGenerator.from_base(MODEL_BASE, TRAIN, HELD)
    log(f"[sigma] {gen2.sigma} n_roles={gen2.n_roles} "
        f"heads_ready={gen2._heads_ready}")
    gen2.train(iters=60, log_every=10, log_fn=log)
    gen2.save(MODEL_OUT)
    log(f"[save] {MODEL_OUT}")
    full2, fails2 = gen2.evaluate(TRAIN)
    log(f"B1. 训练内 FULL: {full2}/{len(TRAIN)}")
    for inp, o, tgt in fails2[:8]:
        log(f"   [miss] {inp!r:12} -> {o!r:28} 期望 {tgt!r}")
    eval_held(gen2, "B2")

    # ---- 快速复载冒烟: 保存的模型能被 from_pretrained 读回 ----
    log(f"\n== Phase C: 复载冒烟 ({os.path.basename(MODEL_OUT)}) ==")
    gen3 = sr.SlotRouteGenerator.from_pretrained(MODEL_OUT, TRAIN, HELD)
    o = gen3.generate("red dog")
    log(f"   'red dog' -> {o!r} ({'MATCH' if o == 'the dog is red!' else 'miss'})")
    log(f"[time] total {time.perf_counter() - t0:.0f}s")
    log("DONE")
    _LOG.close()


if __name__ == "__main__":
    main()
