# -*- coding: utf-8 -*-
"""candyfish — r1 版本对外 API (内核: v3.9 N-slot 多块路由生成器).

把 v3.9 的 nslot_v39.NSlotGenerator 封装为可从外部直接调用的顶层接口,
提供 加载模型 / 训练 / 续写 / 保存模型 四类功能。

r1 相对裸用 nslot_v39 的改进:
  - 路由元数据 (sigma/types/n_roles/max_steps) 内嵌进模型文件
    (`sim._candy_meta`), 加载不再需要重建同 seed 语料;
  - 统一入口, 输入格式校验 + 清晰报错;
  - 兼容旧 v3.9 模型 (无 meta 时需提供 corpus 重建路由)。

## 快速上手

    from candyfish import CandyFish

    # --- 训练新模型 ---
    train, held = CandyFish.build_corpus(n_blocks=4, n_train=30, n_held=6,
                                         seed=13)
    cf = CandyFish.train_new(train, held=held, iters=60)
    cf.save("my_model.pt")

    # --- 加载模型 (无需 corpus) ---
    cf = CandyFish.load("my_model.pt")

    # --- 续写 ---
    text = cf.continue_text("big deer green hare calm fox fast wolf")
    # 'the deer is big and the hare is green and the fox is calm and the wolf is fast!'

    # --- 前缀续写 (给定开头强制续写) ---
    text = cf.continue_text("big deer green hare calm fox fast wolf",
                            prefix="the deer is ")

    # --- 继续训练 ---
    cf.train_more(train, iters=5)
    cf.save("my_model_v2.pt")

## 依赖
    同目录需有 nslot_v39.py / slot_route_v38.py / core/lif_v36.py。
    底层为纯 spiking LIF + margin 斜坡 Hebbian 学习 (无梯度/无查表/
    无规则匹配)。

## 输入格式 (4 块 / 8-slot 例)
    inp  = "a1 n1 a2 n2 a3 n3 a4 n4"   (形容词 动物 交替, 每 slot 一词)
    tgt  = "the n1 is a1 and the n2 is a2 and ... the n4 is a4!"
    句长约 n_blocks * 19 字 (4 块 ≈ 76-82 字)。
"""
import os
import time

import torch

import nslot_v39 as v39
import slot_route_v38 as sr

VERSION = "r1"

# 默认基模型 (v3.8 P37 配方产物), 训练新模型的起点
_DEFAULT_BASE = "_p37_v38_96sim.pt"


class CandyFish:
    """r1 顶层接口: 封装 v3.9 N-slot 生成器的 加载/训练/续写/保存."""

    def __init__(self, sim, sigma, types, n_roles, max_steps=120,
                 lr=0.5, proj_scale=0.125, vocab=None):
        self.sim = sim
        self.sigma = dict(sigma)
        self.types = dict(types)
        self.n_roles = int(n_roles)
        self.max_steps = int(max_steps)
        self.lr = float(lr)
        self.proj_scale = float(proj_scale)
        self.vocab = vocab or {}

    # ---------------------------------------------------------- 加载

    @classmethod
    def load(cls, path, corpus=None):
        """加载已保存的 r1 模型.

        path:   模型文件 (.pt)
        corpus: 仅旧 v3.9 模型 (无内嵌 meta) 需要 — 传入训练时同 seed
                重建的语料用于推导路由; r1 保存的模型传 None 即可。
        """
        sim = sr._load_sim(path)
        meta = getattr(sim, "_candy_meta", None)
        if meta is not None:
            return cls(sim, meta["sigma"], meta["types"], meta["n_roles"],
                       meta.get("max_steps", 120), meta.get("lr", 0.5),
                       meta.get("proj_scale", 0.125),
                       meta.get("vocab"))
        if corpus is None:
            raise ValueError(
                "该模型无内嵌路由元数据 (旧 v3.9 格式): "
                "请用训练时同 seed 重建的 corpus 传入 load(path, corpus=...)")
        sigma = v39.fit_sigma(corpus)
        n_roles = max(max(v39.align_roles_u(i, t)) for i, t in corpus)
        types = v39.fit_slot_types(corpus, sigma)
        max_steps = max(len(t) for _, t in corpus) + 12
        return cls(sim, sigma, types, n_roles, max_steps)

    # ---------------------------------------------------------- 训练

    @classmethod
    def train_new(cls, corpus, held=None, iters=60, lr=0.5,
                  base_model=None, log_every=10, log_fn=print,
                  proj_scale=0.125):
        """从基模型训练新 r1 模型 (统一头从头训练).

        corpus:     [(inp, tgt), ...] 训练对 (见 build_corpus)
        held:       留出评估对 (可选)
        iters:      训练轮数 (经验: 30 条约 60 轮, 200 条约 60 轮收敛)
        base_model: v3.8 P37 基模型路径, 默认同目录 _p37_v38_96sim.pt
        """
        if base_model is None:
            here = os.path.dirname(os.path.abspath(__file__))
            base_model = os.path.join(here, _DEFAULT_BASE)
        gen = v39.NSlotGenerator.from_base(base_model, corpus, held,
                                           lr=lr, proj_scale=proj_scale)
        gen.train(iters=iters, log_every=log_every, log_fn=log_fn)
        cf = cls(gen.sim, gen.sigma, gen.types, gen.n_roles, gen.max_steps,
                 lr, proj_scale)
        cf._store_meta()
        return cf

    def train_more(self, pairs, iters=1, lr=None, log_every=10,
                   log_fn=print):
        """在已加载模型上继续训练 (不重建头).

        pairs 的块数必须与模型 n_roles 一致 (sigma 路由固定)。
        """
        lr = self.lr if lr is None else lr
        self._check_corpus(pairs)
        keys = ("wfirst", "role", "first", "cont", "tmpl", "done")
        t0 = time.perf_counter()
        for it in range(iters):
            agg = {k: [0, 0] for k in keys}
            for inp, tgt in pairs:
                cnt = v39.teacher_pass_u(self.sim, inp, tgt, self.sigma,
                                         self.types, learn=True, lr=lr)
                for k in keys:
                    agg[k][0] += cnt[k][0]
                    agg[k][1] += cnt[k][1]
            if it == 0 or (it + 1) % log_every == 0:
                parts = " ".join(f"{k}={agg[k][0] / max(1, agg[k][1]):.3f}"
                                 for k in keys)
                log_fn(f"[r1+ it={it + 1}/{iters}] {parts} "
                       f"time={time.perf_counter() - t0:.0f}s")
        self._store_meta()
        return self

    # ---------------------------------------------------------- 续写

    def continue_text(self, inp, prefix=None, trace=None, max_steps=None):
        """续写: 输入词序列 → 生成完整句 (或从 prefix 强制续写).

        inp:    "a1 n1 a2 n2 ..." 词数须为 2*n_blocks
        prefix: 可选开头 (如 "the deer is "), 生成将从其后续写
        trace:  传入 list 可获得逐步路由轨迹 (调试用)
        """
        n_words = len(inp.split())
        need = max(self.sigma.values()) + 1  # 每 slot 一个词
        if n_words != need:
            raise ValueError(
                f"输入需 {need} 个词 ({self.n_roles} slot), 得到 {n_words}: {inp!r}")
        return v39.generate_u(self.sim, inp, self.sigma, self.types,
                              max_steps=self.max_steps if max_steps is None
                              else max_steps,
                              end_role=self.n_roles, trace=trace,
                              prefix=prefix)

    def evaluate(self, pairs, show=4, log_fn=print):
        """整句精确匹配评估, 返回 (n_full, fails)."""
        full = 0
        fails = []
        for inp, tgt in pairs:
            o = self.continue_text(inp)
            if o == tgt:
                full += 1
            else:
                fails.append((inp, o, tgt))
        log_fn(f"[r1 eval] FULL {full}/{len(pairs)}")
        for inp, o, tgt in fails[:show]:
            log_fn(f"  MISS {inp!r}\n       got  {o!r}\n       want {tgt!r}")
        return full, fails

    # ---------------------------------------------------------- 保存

    def save(self, path):
        """保存模型 (权重 + 固定投影 + 路由元数据 一并 pickle)."""
        self._store_meta()
        torch.save(self.sim, path)

    def _store_meta(self):
        self.sim._candy_meta = {
            "version": VERSION,
            "sigma": dict(self.sigma),
            "types": dict(self.types),
            "n_roles": self.n_roles,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "proj_scale": self.proj_scale,
            "vocab": self.vocab,
        }

    # ---------------------------------------------------------- 工具

    def _check_corpus(self, pairs):
        need = max(self.sigma.values()) + 1  # 每 slot 一个词
        for inp, tgt in pairs:
            if len(inp.split()) != need:
                raise ValueError(
                    f"train_more 语料词数与模型不符 (需 {need} 词): {inp!r}")
            if max(v39.align_roles_u(inp, tgt)) != self.n_roles:
                raise ValueError(f"目标句 slot 数与模型不符: {tgt!r}")

    @staticmethod
    def build_corpus(n_blocks, n_train, n_held, seed,
                     adjs=None, ans=None):
        """构建训练语料. 词表默认 15 形容词 × 15 动物 (EXT)."""
        if adjs is None:
            adjs = v39.EXT_ADJS
        if ans is None:
            ans = v39.EXT_ANS
        return v39.build_blocks_corpus(n_blocks, n_train, n_held, seed,
                                       adjs=adjs, ans=ans)

    # 词表 (供外部构建自定义语料)
    ADJS = v39.EXT_ADJS
    ANS = v39.EXT_ANS
