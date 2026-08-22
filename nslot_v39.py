# -*- coding: utf-8 -*-
"""nslot_v39 — v3.9 N-slot 多块路由生成器 (可复用模块)

v3.9 最终架构的完整封装, 整合 P54 → P54b → P54c → P54d 修复链, 在纯
spiking LIF 框架 (core/lif_v36.py) 上实现多块长文本续写复述 (60 字目标,
实证达 76-82 字), 且留出组合组合泛化 6/6。

设计: 每个语义块 (slot) = 'the X is A and ... the Z is W!' 中的一个
'the <n> is <a>' 片段。块边界由 W_done 头判定 (slot 转移), 块内沿用
v3.8 的角色专用读出头, 块间通过 slots_done 计数器路由。

## 读出头体系 (全 margin 斜坡 Hebbian 学习, 无梯度/无查表/无规则匹配)
- W_first          句首字符 (输入 cf+direction)
- W_first_t[an/adj] 类型分离词首字符头 (lead/mid slot 词首)
- W_first_last      末 slot 段首协议空格 ' '
- W_cont            统一续写头 [wsrc + proj(剥离计数)] ⊕ onehot3(协议类)
- W_done            统一段完成头 [wsrc + proj(原始计数)] ⊕ onehot3(协议类)
- W_tmpl            模板链头 (输入=当前字符onehot + 模板段计数 + done 计数)
- W_role            模板态角色判定 (融合态 + slot/tmpl 进度投影)
- W_slot_proj       计数通道投影 (scale 放大解决弱计数信号)

## 终止轴 (P54d 核心修复)
- 统一 done 头: 词长知识跨协议类 (lead/mid/last) 共享, 解决零覆盖词
  done 提前/延迟
- '!' 由模板头读出 (done 触发且 slot 全完成), 不再依赖 cont 头满词计数
  泛化

## 用法
    from nslot_v39 import NSlotGenerator, build_blocks_corpus

    train, held = build_blocks_corpus(n_blocks=4, n_train=30, n_held=6, seed=13)
    gen = NSlotGenerator.from_base(base_path, train, held)
    gen.train(iters=60)
    gen.save(out_path)
    print(gen.generate("big deer green hare calm fox fast wolf"))
    # 'the deer is big and the hare is green and the fox is calm and the wolf is fast!'

注意: 模型 pickle 引用 slot_route_v38.R36NoPos, 需保证 slot_route_v38.py
在本模块同目录或 sys.path 上 (运行脚本同目录即可)。
"""
import os
import random
import time
from collections import defaultdict

import torch

import slot_route_v38 as sr

_LOAD = False
if not _LOAD:
    # 复用 v3.8 的 R36NoPos / _load_sim / encode / margin_update / fit_sigma
    R36NoPos = sr.R36NoPos
    _load_sim = sr._load_sim
    encode_words = sr.encode_words
    encode_text = sr.encode_text
    margin_update = sr.margin_update
    fit_sigma = sr.fit_sigma

CLS_I = {"lead": 0, "mid": 1, "last": 2}

# ---------------- 语料 (可扩展词表多块续写) ----------------

ADJS10 = sr.ADJS10          # 10 形容词
ANS10 = sr.ANS10            # 10 动物
EXT_ADJS = ADJS10 + ["dark", "brave", "smart", "quiet", "gold"]      # 15
EXT_ANS = ANS10 + ["tiger", "horse", "mouse", "duck", "goat"]        # 15
# 默认拼接词表 (与 P54d 可比: 10×10); 大语料测试用 EXT_XX
ADJS = ADJS10
ANS = ANS10


def build_blocks_corpus(n_blocks, n_train, n_held, seed,
                        adjs=None, ans=None):
    """构建多块续写语料 (inp = 'a1 n1 a2 n2 ...', tgt = 连接块 + '!').

    n_blocks: 语义块 (slot) 数; 输出 target 约 n_blocks * 16-20 字符
    (2 块 ~35 字, 4 块 ~76-82 字). 4 块即达 60 字目标。
    """
    if adjs is None:
        adjs = ADJS
    if ans is None:
        ans = ANS
    rng = random.Random(seed)
    ans_pool = list(ans)
    adj_pool = list(adjs)
    seen = set()
    combos = []
    while len(combos) < n_train + n_held:
        ans_s = rng.sample(ans_pool, n_blocks)
        adjs_s = rng.sample(adj_pool, n_blocks)
        key = (tuple(adjs_s), tuple(ans_s))
        if key in seen:
            continue
        seen.add(key)
        combos.append(list(zip(adjs_s, ans_s)))
    train = [make_pair(c) for c in combos[:n_train]]
    held = [make_pair(c) for c in combos[n_train:]]
    return train, held


def make_pair(blocks):
    inp = " ".join(f"{a} {n}" for a, n in blocks)
    tgt = " and ".join(f"the {n} is {a}" for a, n in blocks) + "!"
    return (inp, tgt)


# ---------------- 角色协议 (v3.9: '!' 归模板) ----------------

def align_roles_u(inp, tgt):
    """slot1=词+尾空格, 末slot=首空格+词, '!'=模板 (done 触发后由 tmpl 读出).

    与 v3.8 的差异: '!' 不再归 last slot, 使 '!' 读出由 done(全类共享)+tmpl
    承载, 摆脱 cont 头满词计数泛化的依赖。
    """
    words = [w for w in inp.split() if w]
    spans = []
    for w in words:
        i = tgt.find(w)
        if i >= 0:
            spans.append((i, i + len(w)))
    spans.sort()
    roles = [0] * len(tgt)
    for r, (s, e) in enumerate(spans, start=1):
        for j in range(s, e):
            roles[j] = r
        if r == 1:
            if e < len(tgt) and tgt[e] == ' ':
                roles[e] = 1
        if r == len(spans):
            if s > 0 and tgt[s - 1] == ' ':
                roles[s - 1] = r
    return roles


def cls_of(r, n_roles):
    """协议类: lead=slot1(词+尾空格), last=末slot(首空格+词), mid=其余(裸词)."""
    if r == 1:
        return "lead"
    if r == n_roles:
        return "last"
    return "mid"


def fit_slot_types(corpus, sigma):
    """slot 角色 → 词类型 ('an'/'adj'), 从语料统计."""
    types = {}
    for r, idx in sigma.items():
        words = {inp.split()[idx] for inp, _ in corpus}
        if words & set(ANS10):
            types[r] = "an"
        else:
            types[r] = "adj"
    return types


# ---------------- 头初始化 (统一 cont + 统一 done) ----------------

def init_unified_heads(sim, n_roles, proj_scale=0.125):
    dev = sim.W_first.device
    F = sim.feat_dim
    sim.W_role = torch.randn(n_roles + 1, F, dtype=torch.float32,
                             device=dev) * 0.05
    sim.b_role = torch.zeros(n_roles + 1, dtype=torch.float32, device=dev)
    sim.W_slot_proj = torch.randn(F, 256, dtype=torch.float32,
                                  device=dev) * proj_scale
    # 类型分离首字符头 (lead/mid/last 的词首字符共用)
    sim.W_first_t = {}
    sim.b_first_t = {}
    for t in ("an", "adj"):
        sim.W_first_t[t] = torch.randn(256, F, dtype=torch.float32,
                                       device=dev) * 0.05
        sim.b_first_t[t] = torch.zeros(256, dtype=torch.float32, device=dev)
    # 末 slot 段首协议空格 ' '
    sim.W_first_last = torch.randn(256, F, dtype=torch.float32,
                                   device=dev) * 0.05
    sim.b_first_last = torch.zeros(256, dtype=torch.float32, device=dev)
    # 统一 cont 头: [wsrc + proj(剥离计数)] ⊕ onehot3(协议类)
    sim.W_cont = torch.randn(256, F + 3, dtype=torch.float32,
                             device=dev) * 0.05
    sim.b_cont = torch.zeros(256, dtype=torch.float32, device=dev)
    # 统一 done 头: [wsrc + proj(原始计数)] ⊕ onehot3(协议类)
    sim.W_done = torch.randn(2, F + 3, dtype=torch.float32,
                             device=dev) * 0.05
    sim.b_done = torch.zeros(2, dtype=torch.float32, device=dev)
    sim.W_tmpl = torch.randn(256, 512 + n_roles + 1, dtype=torch.float32,
                             device=dev) * 0.05
    sim.b_tmpl = torch.zeros(256, dtype=torch.float32, device=dev)
    embed_scale = 0.15 * (proj_scale / (1.0 / 16.0))
    sim.W_done_embed = torch.randn(n_roles + 1, F, dtype=torch.float32,
                                   device=dev) * embed_scale


def _onehot_cls(cls, dev):
    oh = torch.zeros(3, dtype=torch.float32, device=dev)
    oh[CLS_I[cls]] = 1.0
    return oh


def _cont_input(sim, wsrc, slot_chars, cls):
    """统一 cont 输入: 剥离 last 协议前导空格 + 拼接协议类 one-hot."""
    cv = slot_chars.clone()
    if cls == "last" and cv[ord(' ')] > 0:
        cv[ord(' ')] -= 1.0
    return torch.cat([wsrc.clone() + torch.mv(sim.W_slot_proj, cv),
                      _onehot_cls(cls, sim.W_cont.device)])


def _done_input(sim, wsrc, slot_chars, cls):
    """统一 done 输入: 原始计数 (含协议空格) + 拼接协议类 one-hot."""
    return torch.cat([wsrc.clone() + torch.mv(sim.W_slot_proj, slot_chars),
                      _onehot_cls(cls, sim.W_done.device)])


def _ci(ch):
    return ord(ch) if 0 <= ord(ch) <= 255 else 0


# ---------------- teacher 轨迹 (统一 cont + 统一 done) ----------------

def teacher_pass_u(sim, inp, tgt, sigma, types, learn=True, lr=0.5):
    word_feats = encode_words(sim, inp)
    encode_text(sim, inp)
    direction = sim.dmd_committed_direction
    context = sim.dmd_committed_context
    if context is None:
        context = direction
    goal = (sim._last_goal.clone()
            if getattr(sim, "_last_goal", None) is not None
            else context.clone())
    direction_sel, _ = sim._select_sentence(goal)
    cf = context.detach().clone()
    direction = direction_sel.detach().clone()
    sim.reset_state()
    sim.reset_memory()
    sim._slot_fatigue_reset()
    sim._seqctx_reset()
    if getattr(sim, "dmd_direction", None) is None:
        sim._dmd_reset()
    sim.dmd_direction = direction.clone()

    roles = align_roles_u(inp, tgt)
    tgt_codes = [ord(c) for c in tgt]
    n_roles = max(roles)
    dev = sim.W_tmpl.device

    cnt = {k: [0, 0] for k in ("wfirst", "role", "first", "cont", "tmpl", "done")}

    # ---- W_first 句首字符 ----
    cf_feat = sim._mem_feature(cf)
    first_in = torch.max(cf_feat, direction * 1.0)
    fs = torch.mv(sim.W_first, first_in) + sim.b_first
    f_pred = int(fs.argmax().item())
    cnt["wfirst"][1] += 1
    cnt["wfirst"][0] += (f_pred == tgt_codes[0])
    if learn and f_pred != tgt_codes[0]:
        sr.margin_update(sim.W_first, sim.b_first, tgt_codes[0], f_pred,
                         fs, lr, first_in)

    cur_role = roles[0]
    slots_done = 0
    slot_chars = torch.zeros(256, dtype=torch.float32, device=dev)
    tmpl_chars = torch.zeros(256, dtype=torch.float32, device=dev)
    state = cf.clone()
    prev_state = None
    ch = tgt[0]
    for step in range(1, len(tgt)):
        vec = sim._char_to_8bit_bias(ch)
        sim._multi_layer_forward(vec, n_loops=1)
        v_curr = sim.V_deep[-1] if sim.num_layers > 1 else sim.V
        sparse_feat = sim._dg_separate(v_curr)
        sim.update_coactivation(sparse_feat)
        r_prev = roles[step - 1]
        if r_prev == cur_role:
            if r_prev > 0:
                slot_chars[_ci(ch)] += 1.0
            else:
                tmpl_chars[_ci(ch)] += 1.0
        else:
            cur_role = r_prev
            slot_chars = torch.zeros(256, dtype=torch.float32, device=dev)
            tmpl_chars = torch.zeros(256, dtype=torch.float32, device=dev)
            if r_prev > 0:
                slot_chars[_ci(ch)] += 1.0
            else:
                tmpl_chars[_ci(ch)] += 1.0
        recall = sim.recall_from_memassoc(sparse_feat, sparse_hint=True)
        seqctx = sim._seqctx_contrib(sparse_feat, step)
        state = sim._fuse_state_kwt(
            sparse_feat, recall,
            sim._chain_transition(prev_state) if prev_state is not None else None,
            direction, seqctx=seqctx, content=sim._content_quota_src(cf))
        sim.MemWork = state
        sim._dmd_step(state)
        feat = state
        r_tgt = roles[step]
        # ---- 路由: slot 段内统一 done 头 / 模板态角色头 ----
        if cur_role > 0:
            cls = cls_of(cur_role, n_roles)
            wsrc_d = word_feats[sigma[cur_role]]
            din = _done_input(sim, wsrc_d, slot_chars, cls)
            ds = torch.mv(sim.W_done, din) + sim.b_done
            t_done = 1 if r_tgt != cur_role else 0
            d_pred = int(ds.argmax().item())
            cnt["done"][1] += 1
            cnt["done"][0] += (d_pred == t_done)
            if learn and d_pred != t_done:
                sr.margin_update(sim.W_done, sim.b_done,
                                 t_done, d_pred, ds, lr, din)
            r_pred = 0 if d_pred == 1 else cur_role
            if t_done == 1:
                slots_done += 1
        else:
            feat_role = (feat + torch.mv(sim.W_slot_proj, slot_chars)
                         + torch.mv(sim.W_slot_proj, tmpl_chars)
                         + sim.W_done_embed[slots_done])
            rs = torch.mv(sim.W_role, feat_role) + sim.b_role
            r_pred = int(rs.argmax().item())
            cnt["role"][1] += 1
            cnt["role"][0] += (r_pred == r_tgt)
            if learn and r_pred != r_tgt:
                sr.margin_update(sim.W_role, sim.b_role, r_tgt, r_pred,
                                 rs, lr, feat_role)
        t_code = tgt_codes[step]
        if r_tgt > 0:
            wsrc = word_feats[sigma[r_tgt]]
            cls = cls_of(r_tgt, n_roles)
            t = types[r_tgt]
            seg_start = (r_tgt != r_prev)
            # 末 slot 词首字符: 协议空格后第一个字符, 走类型首字符头
            word_start = (cls == "last" and not seg_start and ch == " ")
            if seg_start or word_start:
                if seg_start and cls == "last":
                    W = sim.W_first_last
                    b = sim.b_first_last
                else:
                    W = sim.W_first_t[t]
                    b = sim.b_first_t[t]
                slot_in = wsrc.clone()
                cnt["first"][1] += 1
            else:
                slot_in = _cont_input(sim, wsrc, slot_chars, cls)
                W = sim.W_cont
                b = sim.b_cont
                cnt["cont"][1] += 1
            sc = torch.mv(W, slot_in) + b
            pred = int(sc.argmax().item())
            if seg_start or word_start:
                cnt["first"][0] += (pred == t_code)
            else:
                cnt["cont"][0] += (pred == t_code)
            if learn and pred != t_code:
                sr.margin_update(W, b, t_code, pred, sc, lr, slot_in)
        else:
            x = torch.zeros(512 + n_roles + 1, dtype=torch.float32,
                            device=dev)
            x[_ci(ch)] = 1.0
            x[256:512] = tmpl_chars
            x[512 + slots_done] = 1.0
            sc = torch.mv(sim.W_tmpl, x) + sim.b_tmpl
            pred = int(sc.argmax().item())
            cnt["tmpl"][1] += 1
            cnt["tmpl"][0] += (pred == t_code)
            if learn and pred != t_code:
                sr.margin_update(sim.W_tmpl, sim.b_tmpl, t_code, pred,
                                 sc, lr, x)
        ch = tgt[step]
        prev_state = state.clone()
    return cnt


# ---------------- 生成 (统一 cont + 统一 done) ----------------

def generate_u(sim, inp, sigma, types, max_steps=120, end_role=None,
               end_char='!', trace=None, prefix=None):
    word_feats = encode_words(sim, inp)
    encode_text(sim, inp)
    direction = sim.dmd_committed_direction
    context = sim.dmd_committed_context
    if context is None:
        context = direction
    goal = (sim._last_goal.clone()
            if getattr(sim, "_last_goal", None) is not None
            else context.clone())
    direction, _ = sim._select_sentence(goal)
    cf = context.detach().clone()
    direction = direction.detach().clone()
    sim.reset_state()
    sim.reset_memory()
    sim._slot_fatigue_reset()
    sim._seqctx_reset()
    if getattr(sim, "dmd_direction", None) is None:
        sim._dmd_reset()
    sim.dmd_direction = direction.clone()

    n_roles = max(sigma) if sigma else 0
    if end_role is None:
        end_role = n_roles
    dev = sim.W_tmpl.device

    result = []
    cf_feat = sim._mem_feature(cf)
    first_in = torch.max(cf_feat, direction * 1.0)
    f_scores = torch.mv(sim.W_first, first_in) + sim.b_first
    ch = chr(int(f_scores.argmax().item()))
    if prefix:
        ch = prefix[0]
    result.append(ch)

    cur_role = 0
    slots_done = 0
    slot_chars = torch.zeros(256, dtype=torch.float32, device=dev)
    tmpl_chars = torch.zeros(256, dtype=torch.float32, device=dev)
    state = cf.clone()
    prev_state = None
    for step in range(1, max_steps):
        vec = sim._char_to_8bit_bias(ch)
        sim._multi_layer_forward(vec, n_loops=1)
        v_curr = sim.V_deep[-1] if sim.num_layers > 1 else sim.V
        sparse_feat = sim._dg_separate(v_curr)
        sim.update_coactivation(sparse_feat)
        if cur_role > 0:
            slot_chars[_ci(ch)] += 1.0
        else:
            tmpl_chars[_ci(ch)] += 1.0
        recall = sim.recall_from_memassoc(sparse_feat, sparse_hint=True)
        seqctx = sim._seqctx_contrib(sparse_feat, step)
        state = sim._fuse_state_kwt(
            sparse_feat, recall,
            sim._chain_transition(prev_state) if prev_state is not None else None,
            direction, seqctx=seqctx, content=sim._content_quota_src(cf))
        sim.MemWork = state
        sim._dmd_step(state)
        feat = state
        # ---- 路由 ----
        if cur_role > 0:
            cls = cls_of(cur_role, n_roles)
            wsrc_d = word_feats[sigma[cur_role]]
            din = _done_input(sim, wsrc_d, slot_chars, cls)
            ds = torch.mv(sim.W_done, din) + sim.b_done
            d_pred = int(ds.argmax().item())
            r_pred = 0 if d_pred == 1 else cur_role
            if d_pred == 1:
                slots_done = min(slots_done + 1, n_roles)
        else:
            feat_role = (feat + torch.mv(sim.W_slot_proj, slot_chars)
                         + torch.mv(sim.W_slot_proj, tmpl_chars)
                         + sim.W_done_embed[slots_done])
            rs = torch.mv(sim.W_role, feat_role) + sim.b_role
            r_pred = int(rs.argmax().item())
        if trace is not None:
            trace.append({"step": step, "ch": ch, "r_pred": r_pred,
                          "cur_role": cur_role, "done": slots_done})
        if r_pred > 0:
            wsrc = word_feats[sigma[r_pred]]
            cls = cls_of(r_pred, n_roles)
            if r_pred != cur_role:
                cur_role = r_pred
                slot_chars = torch.zeros(256, dtype=torch.float32, device=dev)
                tmpl_chars = torch.zeros(256, dtype=torch.float32, device=dev)
                if cls == "last":
                    W = sim.W_first_last
                    b = sim.b_first_last
                else:
                    t = types[r_pred]
                    W = sim.W_first_t[t]
                    b = sim.b_first_t[t]
                slot_in = wsrc.clone()
            elif cls == "last" and ch == " ":
                # 末 slot 词首字符 (协议空格后) → 类型首字符头
                t = types[r_pred]
                W = sim.W_first_t[t]
                b = sim.b_first_t[t]
                slot_in = wsrc.clone()
            else:
                slot_in = _cont_input(sim, wsrc, slot_chars, cls)
                W = sim.W_cont
                b = sim.b_cont
            sc = torch.mv(W, slot_in) + b
            sc[0] = -1e9
        else:
            if cur_role != 0:
                tmpl_chars = torch.zeros(256, dtype=torch.float32, device=dev)
            cur_role = 0
            slot_chars = torch.zeros(256, dtype=torch.float32, device=dev)
            x = torch.zeros(512 + n_roles + 1, dtype=torch.float32, device=dev)
            x[_ci(ch)] = 1.0
            x[256:512] = tmpl_chars
            x[512 + slots_done] = 1.0
            sc = torch.mv(sim.W_tmpl, x) + sim.b_tmpl
            sc[0] = -1e9
        if prefix is not None and step < len(prefix):
            next_code = ord(prefix[step]) if 0 <= ord(prefix[step]) <= 255 \
                else 63
        else:
            next_code = int(sc.argmax().item())
        if next_code == 0:
            break
        result.append(chr(next_code) if 0 <= next_code <= 255 else '?')
        # ★ '!' 由 tmpl 读出 (done 触发后 slots_done=n_roles)
        if next_code == ord(end_char) and slots_done >= n_roles:
            break
        ch = result[-1]
        prev_state = state.clone()
    return ''.join(result)


# ---------------- 封装类 ----------------

class NSlotGenerator:
    """v3.9 N-slot 多块路由生成器.

    参数:
        sim: 已加载的 RecurrentLIFSimulator (R36NoPos 实例)
        corpus: 训练对列表 [(inp, tgt), ...]
        held: 留出评估对 (仅存储, 不参与训练)
        lr, max_steps, proj_scale
    """

    BASE_CONFIG = {
        "quota_from_content": True,
        "slot_fatigue_lr": 0.0,
        "state_kwt_k": 48,
        "dir_quota_k": 16,
    }

    def __init__(self, sim, corpus, held=None, lr=0.5, max_steps=120,
                 proj_scale=0.125):
        self.sim = sim
        self.corpus = list(corpus)
        self.held = list(held) if held else []
        self.lr = lr
        self.max_steps = max_steps
        self.proj_scale = proj_scale
        self.sigma = fit_sigma(self.corpus)
        self.n_roles = max(max(align_roles_u(i, t)) for i, t in self.corpus)
        self.types = fit_slot_types(self.corpus, self.sigma)
        # 无条件重建全部读出头 (与 P54d 一致; _p37 基模型可能带旧 head 属性,
        # 惰性 hasattr 判断不可靠)
        init_unified_heads(sim, self.n_roles, proj_scale)

    # ---- 构造 ----

    @classmethod
    def from_base(cls, path, corpus, held=None, lr=0.5, max_steps=None,
                  proj_scale=0.125):
        sim = _load_sim(path)
        for k, v in cls.BASE_CONFIG.items():
            setattr(sim, k, v)
        if max_steps is None:
            max_steps = max(len(t) for _, t in corpus) + 12
        return cls(sim, corpus, held, lr, max_steps, proj_scale)

    @classmethod
    def from_pretrained(cls, path, corpus, held=None, lr=0.5, max_steps=None,
                        proj_scale=0.125):
        sim = _load_sim(path)
        if max_steps is None:
            max_steps = max(len(t) for _, t in corpus) + 12
        return cls(sim, corpus, held, lr, max_steps, proj_scale)

    # ---- 训练 ----

    def init_heads(self, force=False):
        if force or not hasattr(self.sim, "W_done"):
            init_unified_heads(self.sim, self.n_roles, self.proj_scale)

    def train(self, iters=60, log_every=10, log_fn=None):
        """统一训练全部头 (teacher 轨迹 margin 斜坡 Hebbian)."""
        self.init_heads()
        if log_fn is None:
            def log_fn(*a, **k):
                pass
        keys = ("wfirst", "role", "first", "cont", "tmpl", "done")
        t0 = time.perf_counter()
        for it in range(iters):
            agg = {k: [0, 0] for k in keys}
            for inp, tgt in self.corpus:
                cnt = teacher_pass_u(self.sim, inp, tgt, self.sigma,
                                     self.types, learn=True, lr=self.lr)
                for k in keys:
                    agg[k][0] += cnt[k][0]
                    agg[k][1] += cnt[k][1]
            if it == 0 or (it + 1) % log_every == 0:
                parts = " ".join(f"{k}={agg[k][0] / max(1, agg[k][1]):.3f}"
                                 for k in keys)
                log_fn(f"[v39 it={it + 1}/{iters}] {parts} "
                       f"time={time.perf_counter() - t0:.0f}s")
        return self

    # ---- 生成 / 评估 ----

    def generate(self, inp, trace=None, prefix=None):
        return generate_u(self.sim, inp, self.sigma, self.types,
                          max_steps=self.max_steps,
                          end_role=self.n_roles, trace=trace, prefix=prefix)

    def evaluate(self, pairs, tag=None, show=4, log_fn=None):
        """返回 (n_full, fails), fails=[(inp, got, tgt), ...]."""
        full = 0
        fails = []
        for inp, tgt in pairs:
            o = self.generate(inp)
            if o == tgt:
                full += 1
            else:
                fails.append((inp, o, tgt))
        if log_fn and tag:
            log_fn(f"   {tag}: FULL {full}/{len(pairs)}")
            for inp, o, tgt in fails[:show]:
                d = next((i for i, (a, b) in enumerate(zip(o, tgt)) if a != b),
                         min(len(o), len(tgt)))
                log_fn(f"   MISS {inp!r}")
                log_fn(f"        got  {o!r} (len {len(o)})")
                log_fn(f"        want {tgt!r} (len {len(tgt)}) 首分歧@{d}")
        return full, fails

    # ---- 持久化 ----

    def save(self, path):
        torch.save(self.sim, path)


# ---------------- 跨协议拼写单元测试 ----------------

def spell_word(sim, word, cls, n_roles):
    """给定词与协议类, 首字符头 + 统一 cont/done 头贪心拼写."""
    dev = sim.W_cont.device
    wsrc = encode_words(sim, word)[0]
    t = "an" if word in sr.ANS10 else "adj"
    out = []
    if cls == "last":
        out.append(" ")
        raw = torch.zeros(256, dtype=torch.float32, device=dev)
        raw[ord(" ")] = 1.0
        sc = torch.mv(sim.W_first_t[t], wsrc) + sim.b_first_t[t]
        c = int(sc.argmax().item())
        out.append(chr(c))
        raw[c] += 1.0
    else:
        sc = torch.mv(sim.W_first_t[t], wsrc) + sim.b_first_t[t]
        c = int(sc.argmax().item())
        out.append(chr(c))
        raw = torch.zeros(256, dtype=torch.float32, device=dev)
        raw[c] = 1.0
    for _ in range(10):
        din = _done_input(sim, wsrc, raw, cls)
        ds = torch.mv(sim.W_done, din) + sim.b_done
        if int(ds.argmax().item()) == 1:
            if cls == "last":
                x = torch.zeros(512 + n_roles + 1, dtype=torch.float32,
                                device=dev)
                x[_ci(out[-1])] = 1.0
                x[512 + n_roles] = 1.0
                sc = torch.mv(sim.W_tmpl, x) + sim.b_tmpl
                c = int(sc.argmax().item())
                out.append(chr(c) if 0 < c <= 255 else '?')
            return "".join(out)
        x = _cont_input(sim, wsrc, raw, cls)
        sc = torch.mv(sim.W_cont, x) + sim.b_cont
        sc[0] = -1e9
        c = int(sc.argmax().item())
        ch = chr(c)
        out.append(ch)
        raw[c] += 1.0
        if cls == "lead" and ch == " ":
            return "".join(out)
    return "".join(out)


# ---------------- 默认语料 (10×10, 与 P54 可比) ----------------
# ADJS / ANS 已在文件头部语料区定义。