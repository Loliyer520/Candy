# -*- coding: utf-8 -*-
"""slot_route_v38 — v3.8 slot 角色路由生成器 (可复用模块)

v3.8 最终架构的完整封装, 整合 P48b→P48h→P49b→P49c→P49e 修复链:
  1. 词级分离编码 encode_words — 每词独立 reset 后编码取 _last_goal,
     slot 头只读对应词表征 (sigma 映射 slot 角色→输入词序)
  2. 角色协议"空格双侧归 slot" — slot1=词+尾空格 ('fox '),
     slot2=首空格+词+尾'!' (' big!'), 模板只剩 'the '+'is' 共享链
  3. 六个专用读出头 (全 margin 斜坡 Hebbian 学习):
     W_first        句首字符 (输入 cf+direction)
     W_tmpl         模板链 (输入=当前字符onehot+模板段计数)
     W_slot_first[r] slot 首字符 (纯词表征)
     W_slot[r]      slot 续写 (词表征+计数向量投影)
     W_done_slot[r] 段完成判定 (纯输入侧, 零漂移)
     W_role         模板态角色判定 (融合态+slot进度投影)
  4. 终止 = end_role 尾 end_char 输出即 break

用法:
    from slot_route_v38 import SlotRouteGenerator, TRAIN96, HELD4

    # 从基础模型 (P37 配方) 统一训练全部头
    gen = SlotRouteGenerator.from_base(base_path, TRAIN96, HELD4)
    gen.train(iters=60)
    gen.save(out_path)
    print(gen.generate("red dog"))          # 'the dog is red!'

    # 或加载已收敛模型 (P49e 及本模块保存的模型)
    gen = SlotRouteGenerator.from_pretrained(model_path, TRAIN96, HELD4)

注意: 模型 pickle 引用本模块的 R36NoPos 类, 加载本模块保存的模型时
需保证 slot_route_v38.py 在 sys.path 上 (同目录运行脚本即可)。
"""
import sys
import time

_DOC_PATH = r"c:\Users\loliyc\Documents\Code\test\bio_neural_net"
if _DOC_PATH not in sys.path:
    sys.path.insert(0, _DOC_PATH)

import torch

from core.lif_v36 import RecurrentLIFSimulator as R36

# ---------------- 语料 (10 adj × 10 animal 拉丁方) ----------------

ADJS10 = ["red", "blue", "green", "big", "small",
          "fast", "slow", "cute", "wild", "calm"]
ANS10 = ["dog", "cat", "fox", "bear", "deer",
         "wolf", "bird", "fish", "lion", "hare"]

LATIN100 = [(f"{a} {an}", f"the {an} is {a}!") for a in ADJS10 for an in ANS10]
HELD4 = [("red dog", "the dog is red!"), ("blue cat", "the cat is blue!"),
         ("big fox", "the fox is big!"), ("small bird", "the bird is small!")]
TRAIN96 = [p for p in LATIN100 if p not in HELD4]


# ---------------- R36NoPos (pos 头禁用包装, pickle 兼容) ----------------

class R36NoPos(R36):
    """禁用 pos-head 路径的 R36 — 与 _p* 实验脚本中的定义完全一致。

    历史 .pt 模型的 pickle 引用 __main__.R36NoPos; _load_sim 会把本类
    注入调用进程的 __main__ 以完成反序列化。
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.W_ctx_to_pos = []
        self.b_ctx_to_pos = []
        self.expert_pos = []
        self.expert_pos_b = []
        self.stab_cum = []
        self.protect_mode = "off"

    def train_pos_heads(self, *a, **k):
        return None

    def train_pos_heads_experts(self, *a, **k):
        return 0

    def pos_head_recall(self, state, k):
        return None, 0.0

    def _append_pos_head(self):
        pass

    def _ensure_expert_pos(self, e, k):
        pass


def _load_sim(path):
    """加载 .pt 模型, 兼容 __main__.R36NoPos pickle 引用。"""
    import __main__ as m
    if not hasattr(m, "R36NoPos"):
        m.R36NoPos = R36NoPos
    return torch.load(path, weights_only=False)


# ---------------- 编码 ----------------

def encode_text(sim, inp):
    sim.reset_state()
    sim.reset_memory()
    sim._dmd_reset()
    sim.dmd_record_trace = True
    sim.encode_text_lif(inp, update_memory=True, record_dmd=True)
    sim.input_end()


def encode_words(sim, inp):
    """词级分离编码: 每词独立 reset 后编码, 取纯词表征。"""
    feats = []
    for w in inp.split():
        if not w:
            continue
        sim.reset_state()
        sim.reset_memory()
        sim._dmd_reset()
        sim.dmd_record_trace = True
        sim.encode_text_lif(w, update_memory=True, record_dmd=True)
        sim.input_end()
        g = (sim._last_goal.clone()
             if getattr(sim, "_last_goal", None) is not None
             else sim.dmd_committed_context.clone())
        feats.append(g)
    return feats


# ---------------- 角色协议 (空格双侧归 slot) ----------------

def align_roles(inp, tgt):
    """slot1=词+尾空格, slot2=首空格+词+尾'!'。训练期监督对齐用。"""
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
            if e < len(tgt) and tgt[e] == '!':
                roles[e] = r
    return roles


def fit_sigma(dialogues):
    """统计 slot 角色→输入词序映射 (训练语料投票)。"""
    votes = {}
    for inp, tgt in dialogues:
        words = inp.split()
        spans = []
        for w in words:
            i = tgt.find(w)
            if i >= 0:
                spans.append((i, i + len(w)))
        spans.sort()
        for r, (s, e) in enumerate(spans, start=1):
            w = tgt[s:e]
            if w in words:
                idx = words.index(w)
                votes.setdefault(r, {}).setdefault(idx, 0)
                votes[r][idx] += 1
    sigma = {}
    for r, d in votes.items():
        sigma[r] = max(d, key=d.get)
    return sigma


# ---------------- 头初始化 ----------------

def init_slot_heads(sim, n_roles):
    """W_role / W_slot[r] / W_slot_first[r] / W_slot_proj。"""
    dev = sim.W_first.device
    sim.W_role = (torch.randn(n_roles + 1, sim.feat_dim,
                              dtype=torch.float32, device=dev) * 0.05)
    sim.b_role = torch.zeros(n_roles + 1, dtype=torch.float32, device=dev)
    sim.W_slot = {}
    sim.b_slot = {}
    sim.W_slot_first = {}
    sim.b_slot_first = {}
    for r in range(1, n_roles + 1):
        sim.W_slot[r] = (torch.randn(256, sim.feat_dim,
                                     dtype=torch.float32, device=dev) * 0.05)
        sim.b_slot[r] = torch.zeros(256, dtype=torch.float32, device=dev)
        sim.W_slot_first[r] = (torch.randn(256, sim.feat_dim,
                                           dtype=torch.float32, device=dev) * 0.05)
        sim.b_slot_first[r] = torch.zeros(256, dtype=torch.float32,
                                          device=dev)
    sim.W_slot_proj = (torch.randn(sim.feat_dim, 256,
                                   dtype=torch.float32, device=dev)
                       * (1.0 / 16.0))


def init_tmpl_head(sim):
    """W_tmpl: 输入=[当前字符onehot, 模板段计数] (512维) → 下一字符。"""
    dev = sim.W_first.device
    sim.W_tmpl = torch.randn(256, 512, dtype=torch.float32, device=dev) * 0.05
    sim.b_tmpl = torch.zeros(256, dtype=torch.float32, device=dev)


def init_done_heads(sim, n_roles):
    """W_done_slot[r]: (词表征+计数投影) → {继续, 完成}。"""
    dev = sim.W_first.device
    sim.W_done_slot = {}
    sim.b_done_slot = {}
    for r in range(1, n_roles + 1):
        sim.W_done_slot[r] = (torch.randn(2, sim.feat_dim,
                                          dtype=torch.float32, device=dev) * 0.05)
        sim.b_done_slot[r] = torch.zeros(2, dtype=torch.float32, device=dev)


# ---------------- 学习规则 ----------------

def margin_update(W, b, row_t, row_p, scores, lr, feat):
    """margin 斜坡 Hebbian: err = clip(0.5*(s_p - s_t + 1), 0, 1)。"""
    s_t = scores[row_t].item()
    s_p = scores[row_p].item()
    err = min(1.0, max(0.0, 0.5 * (s_p - s_t + 1.0)))
    W[row_t] += lr * err * feat
    b[row_t] += 0.5 * lr * err
    W[row_p] -= lr * err * feat
    b[row_p] -= 0.5 * lr * err


# ---------------- teacher 轨迹 (全头统一训练) ----------------

def teacher_pass(sim, inp, tgt, sigma, learn=True, lr=0.5):
    """teacher 轨迹 pass: 一次遍历训练全部六个头。

    返回 {head: [ok, tot}] 计数字典。
    """
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

    roles = align_roles(inp, tgt)
    tgt_codes = [ord(c) for c in tgt]

    cnt = {k: [0, 0] for k in ("role", "first", "cont", "tmpl", "done")}
    cur_role = roles[0]
    slot_chars = torch.zeros(256, dtype=torch.float32,
                             device=sim.W_tmpl.device)
    tmpl_chars = torch.zeros(256, dtype=torch.float32,
                             device=sim.W_tmpl.device)
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
                slot_chars[ord(ch) if 0 <= ord(ch) <= 255 else 0] += 1.0
            else:
                tmpl_chars[ord(ch) if 0 <= ord(ch) <= 255 else 0] += 1.0
        else:
            cur_role = r_prev
            slot_chars = torch.zeros_like(slot_chars)
            tmpl_chars = torch.zeros_like(tmpl_chars)
            if r_prev > 0:
                slot_chars[ord(ch) if 0 <= ord(ch) <= 255 else 0] += 1.0
            else:
                tmpl_chars[ord(ch) if 0 <= ord(ch) <= 255 else 0] += 1.0
        recall = sim.recall_from_memassoc(sparse_feat, sparse_hint=True)
        seqctx = sim._seqctx_contrib(sparse_feat, step)
        state = sim._fuse_state_kwt(
            sparse_feat, recall,
            sim._chain_transition(prev_state) if prev_state is not None else None,
            direction, seqctx=seqctx, content=sim._content_quota_src(cf))
        sim.MemWork = state
        sim._dmd_step(state)
        feat = state
        feat_role = feat + torch.mv(sim.W_slot_proj, slot_chars)
        r_tgt = roles[step]
        # 段内判定路由: slot 段用 done 头, 模板段用角色头
        if cur_role > 0 and cur_role in sim.W_done_slot:
            wsrc_d = word_feats[sigma[cur_role]]
            din = wsrc_d.clone() + torch.mv(sim.W_slot_proj, slot_chars)
            ds = torch.mv(sim.W_done_slot[cur_role], din) + \
                sim.b_done_slot[cur_role]
            t_done = 1 if r_tgt != cur_role else 0
            d_pred = int(ds.argmax().item())
            cnt["done"][1] += 1
            cnt["done"][0] += (d_pred == t_done)
            if learn and d_pred != t_done:
                margin_update(sim.W_done_slot[cur_role],
                              sim.b_done_slot[cur_role],
                              t_done, d_pred, ds, lr, din)
            r_pred = 0 if d_pred == 1 else cur_role
        else:
            rs = torch.mv(sim.W_role, feat_role) + sim.b_role
            r_pred = int(rs.argmax().item())
            cnt["role"][1] += 1
            cnt["role"][0] += (r_pred == r_tgt)
            if learn and r_pred != r_tgt:
                margin_update(sim.W_role, sim.b_role, r_tgt, r_pred,
                              rs, lr, feat_role)
        t_code = tgt_codes[step]
        if r_tgt > 0:
            is_first = (r_tgt != r_prev)
            wsrc = word_feats[sigma[r_tgt]]
            if is_first:
                W = sim.W_slot_first[r_tgt]
                b = sim.b_slot_first[r_tgt]
                slot_in = wsrc.clone()
                cnt["first"][1] += 1
            else:
                W = sim.W_slot[r_tgt]
                b = sim.b_slot[r_tgt]
                slot_in = wsrc.clone() + torch.mv(
                    sim.W_slot_proj, slot_chars)
                cnt["cont"][1] += 1
            sc = torch.mv(W, slot_in) + b
            pred = int(sc.argmax().item())
            if is_first:
                cnt["first"][0] += (pred == t_code)
            else:
                cnt["cont"][0] += (pred == t_code)
            if learn and pred != t_code:
                margin_update(W, b, t_code, pred, sc, lr, slot_in)
        else:
            x = torch.zeros(512, dtype=torch.float32,
                            device=sim.W_tmpl.device)
            ci = ord(ch) if 0 <= ord(ch) <= 255 else 0
            x[ci] = 1.0
            x[256:] = tmpl_chars
            sc = torch.mv(sim.W_tmpl, x) + sim.b_tmpl
            pred = int(sc.argmax().item())
            cnt["tmpl"][1] += 1
            cnt["tmpl"][0] += (pred == t_code)
            if learn and pred != t_code:
                margin_update(sim.W_tmpl, sim.b_tmpl, t_code, pred,
                              sc, lr, x)
        ch = tgt[step]
        prev_state = state.clone()
    return cnt


# ---------------- 生成 (free-run) ----------------

def generate(sim, inp, sigma, max_steps=40, end_role=None, end_char='!',
             trace=None):
    """slot 角色路由生成: done 头段内判定 + W_role 模板态判定。

    trace: 可选 list, 逐步追加 {"step","ch","r_pred","cur_role"}。
    """
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

    if end_role is None:
        end_role = max(sigma) if sigma else 0

    result = []
    cf_feat = sim._mem_feature(cf)
    first_in = torch.max(cf_feat, direction * 1.0)
    f_scores = torch.mv(sim.W_first, first_in) + sim.b_first
    ch = chr(int(f_scores.argmax().item()))
    result.append(ch)

    cur_role = 0
    slot_chars = torch.zeros(256, dtype=torch.float32,
                             device=sim.W_tmpl.device)
    tmpl_chars = torch.zeros(256, dtype=torch.float32,
                             device=sim.W_tmpl.device)
    state = cf.clone()
    prev_state = None
    for step in range(1, max_steps):
        vec = sim._char_to_8bit_bias(ch)
        sim._multi_layer_forward(vec, n_loops=1)
        v_curr = sim.V_deep[-1] if sim.num_layers > 1 else sim.V
        sparse_feat = sim._dg_separate(v_curr)
        sim.update_coactivation(sparse_feat)
        if cur_role > 0:
            slot_chars[ord(ch) if 0 <= ord(ch) <= 255 else 0] += 1.0
        else:
            tmpl_chars[ord(ch) if 0 <= ord(ch) <= 255 else 0] += 1.0
        recall = sim.recall_from_memassoc(sparse_feat, sparse_hint=True)
        seqctx = sim._seqctx_contrib(sparse_feat, step)
        state = sim._fuse_state_kwt(
            sparse_feat, recall,
            sim._chain_transition(prev_state) if prev_state is not None else None,
            direction, seqctx=seqctx, content=sim._content_quota_src(cf))
        sim.MemWork = state
        sim._dmd_step(state)
        feat = state
        feat_role = feat + torch.mv(sim.W_slot_proj, slot_chars)
        # 段内判定路由
        if cur_role > 0 and cur_role in sim.W_done_slot:
            wsrc_d = word_feats[sigma[cur_role]]
            din = wsrc_d.clone() + torch.mv(sim.W_slot_proj, slot_chars)
            ds = torch.mv(sim.W_done_slot[cur_role], din) + \
                sim.b_done_slot[cur_role]
            d_pred = int(ds.argmax().item())
            r_pred = 0 if d_pred == 1 else cur_role
        else:
            rs = torch.mv(sim.W_role, feat_role) + sim.b_role
            r_pred = int(rs.argmax().item())
        if trace is not None:
            trace.append({"step": step, "ch": ch,
                          "r_pred": r_pred, "cur_role": cur_role})
        if r_pred > 0 and r_pred in sim.W_slot:
            wsrc = word_feats[sigma[r_pred]]
            if r_pred != cur_role:
                cur_role = r_pred
                slot_chars = torch.zeros_like(slot_chars)
                tmpl_chars = torch.zeros_like(tmpl_chars)
                W = sim.W_slot_first[r_pred]
                b = sim.b_slot_first[r_pred]
                slot_in = wsrc.clone()
            else:
                W = sim.W_slot[r_pred]
                b = sim.b_slot[r_pred]
                slot_in = wsrc.clone() + torch.mv(
                    sim.W_slot_proj, slot_chars)
            sc = torch.mv(W, slot_in) + b
            sc[0] = -1e9
        else:
            if cur_role != 0:
                tmpl_chars = torch.zeros_like(tmpl_chars)
            cur_role = 0
            slot_chars = torch.zeros_like(slot_chars)
            x = torch.zeros(512, dtype=torch.float32,
                            device=sim.W_tmpl.device)
            ci = ord(ch) if 0 <= ord(ch) <= 255 else 0
            x[ci] = 1.0
            x[256:] = tmpl_chars
            sc = torch.mv(sim.W_tmpl, x) + sim.b_tmpl
            sc[0] = -1e9
        next_code = int(sc.argmax().item())
        if next_code == 0:
            break
        result.append(chr(next_code) if 0 <= next_code <= 255 else '?')
        if cur_role == end_role and next_code == ord(end_char):
            break
        ch = result[-1]
        prev_state = state.clone()
    return ''.join(result)


# ---------------- 封装类 ----------------

class SlotRouteGenerator:
    """v3.8 slot 角色路由生成器。

    参数:
        sim: 已加载的 RecurrentLIFSimulator (R36NoPos 实例)
        corpus: 训练对话对列表 [(inp, tgt), ...]
        held: 留出评估对 (仅存储, 不参与训练)
        lr: 全头统一 margin 斜坡学习率 (默认 0.5)
        max_steps: 生成步数上限 (默认 40)
    """

    # P48h 协议基础配置 (from_base 时应用于 sim)
    BASE_CONFIG = {
        "quota_from_content": True,   # 成分 quota
        "slot_fatigue_lr": 0.0,       # 关闭疲劳 (v3.8 用架构解法替代)
        "state_kwt_k": 48,            # 融合态 top-k
        "dir_quota_k": 16,            # 方向保护通道
    }

    def __init__(self, sim, corpus, held=None, lr=0.5, max_steps=40):
        self.sim = sim
        self.corpus = list(corpus)
        self.held = list(held) if held else []
        self.lr = lr
        self.max_steps = max_steps
        self.sigma = fit_sigma(self.corpus)
        self.n_roles = max(max(align_roles(i, t)) for i, t in self.corpus)
        self._heads_ready = self._has_heads()

    # ---- 构造 ----

    @classmethod
    def from_base(cls, path, corpus, held=None, lr=0.5, max_steps=40,
                  base_config=None):
        """从基础模型 (P37 配方) 构造, 应用 P48h 协议配置。"""
        sim = _load_sim(path)
        cfg = dict(cls.BASE_CONFIG)
        if base_config:
            cfg.update(base_config)
        for k, v in cfg.items():
            setattr(sim, k, v)
        return cls(sim, corpus, held, lr, max_steps)

    @classmethod
    def from_pretrained(cls, path, corpus, held=None, lr=0.5, max_steps=40):
        """从已收敛模型 (P49e 或本模块保存) 构造, 不改配置。"""
        sim = _load_sim(path)
        return cls(sim, corpus, held, lr, max_steps)

    # ---- 头管理 ----

    def _has_heads(self):
        sim = self.sim
        if not all(hasattr(sim, a) for a in
                   ("W_role", "W_slot", "W_slot_first", "W_tmpl")):
            return False
        d = getattr(sim, "W_done_slot", None)
        return bool(d) and all(r in d for r in range(1, self.n_roles + 1))

    def init_heads(self, force=False):
        """初始化缺失的头 (force=True 全部重置)。"""
        sim = self.sim
        if force or not all(hasattr(sim, a) for a in
                            ("W_role", "W_slot", "W_slot_first")):
            init_slot_heads(sim, self.n_roles)
        if force or not hasattr(sim, "W_tmpl"):
            init_tmpl_head(sim)
        if force or not self._has_done():
            init_done_heads(sim, self.n_roles)
        self._heads_ready = True

    def _has_done(self):
        d = getattr(self.sim, "W_done_slot", None)
        return bool(d) and all(r in d for r in range(1, self.n_roles + 1))

    # ---- 训练 ----

    def train(self, iters=60, log_every=20, log_fn=None):
        """统一训练全部头 (teacher 轨迹 margin 斜坡 Hebbian)。"""
        if log_fn is None:
            def log_fn(*a, **k):
                pass
        if not self._heads_ready:
            self.init_heads()
        keys = ("role", "first", "cont", "tmpl", "done")
        t0 = time.perf_counter()
        for it in range(iters):
            agg = {k: [0, 0] for k in keys}
            for inp, tgt in self.corpus:
                cnt = teacher_pass(self.sim, inp, tgt, self.sigma,
                                   learn=True, lr=self.lr)
                for k in keys:
                    agg[k][0] += cnt[k][0]
                    agg[k][1] += cnt[k][1]
            if it == 0 or (it + 1) % log_every == 0:
                parts = " ".join(
                    f"{k}={agg[k][0] / max(1, agg[k][1]):.3f}" for k in keys)
                log_fn(f"[slot-route it={it + 1}/{iters}] {parts} "
                       f"time={time.perf_counter() - t0:.0f}s")
        return self

    # ---- 生成 / 评估 ----

    def generate(self, inp, trace=None):
        return generate(self.sim, inp, self.sigma,
                        max_steps=self.max_steps,
                        end_role=self.n_roles, trace=trace)

    def evaluate(self, pairs):
        """返回 (n_full, fails), fails=[(inp, got, tgt), ...]。"""
        full = 0
        fails = []
        for inp, tgt in pairs:
            o = self.generate(inp)
            if o == tgt:
                full += 1
            else:
                fails.append((inp, o, tgt))
        return full, fails

    # ---- 持久化 ----

    def save(self, path):
        torch.save(self.sim, path)
