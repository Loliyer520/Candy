"""
episodic_memory.py — P1 事件联合记忆层 v3 (v14, experiment18)

★ 同时解决"无限上下文"与"时间序列"两个死结 (文献驱动):

  v1 失败: LIF acc_state 投影 top-k 判别性崩塌 (diag_p1: 重叠 104-195/200)
  v2 失败: 字符位型 + 全局排列, 位置间串扰 — 长回复 (34 字符) 下
    其他位置位型反排列后残留 0.39×coeff 干扰, 信噪比 ~4:1 → 库14 86.6%
  v3 修复: 块绑定 (block binding) — 每位置独占一个维度块, 位置间
    串扰严格归零; 字符内重叠 = char_ones²/block_size (可控)

核心思想 (海马情景记忆 = 事件绑定 + 整体稀疏存储):
  - Horner 2015: 海马把事件各元素绑定成整体模式, 任意局部线索补全全部
  - Kanerva SDM: 稀疏高维分布式记忆, 内容寻址, 容量线性可扩展
  - Long Sequence Hopfield Memory (NeurIPS 2023): 序列整体存储而非逐项

机制 (全部在红线内: 无梯度/无BP/无连续学习信号/无批量优化):
  1. 位置-字符联合位型: 对每个 (位置 k, 字符 c) 生成块 k 内随机稀疏
     位型 p[k][c] (block_size 维, char_ones 个 1, 一次性生成, 不学习)
  2. 绑定: 序列 [c0..c_{L-1}] → event = Σ_k p[k][c_k]  (顺序编入块位置)
  3. 存储 (Hebbian 一次性写入, imprinting, 无 RPE):
     E += outer(bind(回复), bind(输入))     # 一个对话一条
  4. 回忆 (整体补全): raw = E @ bind(输入查询)
     → 位置 k: rk = raw[块k] → score[c] = p[k][c]·rk → argmax
     = 该位置字符 (无需逐字符预测)

★ 红线检查: 无梯度下降/反向传播; 学习 = Hebbian 外积共发放;
  解码 = 固定位型内积 argmax (联想匹配, 非学习); 无批量/epoch 平均
  (每对话一次性写入, 后到者叠加); 无偏置更新。
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpisodicEventMemory:
    """事件联合记忆 v3: 输入序列 → 整体补全回复序列 (块绑定, 纯联想记忆)"""

    def __init__(self, dim=8192, block_size=256, char_ones=8, max_pos=32,
                 seed=7):
        """
        Args:
            dim: 事件码总维度 (须 = max_pos × block_size)
            block_size: 每个位置独占的块维度 (决定字符位型分离度)
            char_ones: 每个 (位置,字符) 位型的活跃位数 (信号强度)
            max_pos: 最大序列位置 (块数, 超长回复末尾截断)
            seed: 固定随机种子 (位型一次性生成, 不学习)
        """
        assert dim % max_pos == 0, "dim 必须能被 max_pos 整除"
        self.dim = dim
        self.block_size = dim // max_pos
        self.char_ones = char_ones
        self.max_pos = max_pos
        if block_size is not None:
            assert block_size == self.block_size, \
                f"block_size {block_size} != dim/max_pos {self.block_size}"
        g = torch.Generator(device="cpu").manual_seed(seed)

        # 位置-字符联合位型: p[k][c] 是块 k 内 char_ones 个 1 (不学习)
        self.p = torch.zeros(max_pos, 256, self.block_size, device=DEVICE)
        for k in range(max_pos):
            for c in range(256):
                idx = torch.randperm(self.block_size, generator=g)[:char_ones]
                self.p[k, c, idx] = 1.0

        # 关联矩阵 E: dim × dim (Hebbian 叠加存储, 无偏置)
        self.E = torch.zeros(dim, dim, device=DEVICE)
        self._n_events = 0

    # ---------- 绑定: 序列 → 稀疏事件码 (顺序编入块位置) ----------

    def bind(self, codes):
        """字符序列 → 事件码 (输入/输出共用同一绑定机制)"""
        v = torch.zeros(self.dim, device=DEVICE)
        for k, c in enumerate(codes):
            if k >= self.max_pos:
                break
            lo = k * self.block_size
            v[lo:lo + self.block_size] += self.p[k, c]
        return v

    # ---------- 存储: Hebbian 一次性写入 (无 RPE, 无批量) ----------

    def store(self, in_codes, out_codes):
        """一个对话: E += outer(bind(回复), bind(输入)) — imprinting"""
        ctx = self.bind(in_codes)
        ev = self.bind(out_codes)
        self.E += torch.outer(ev, ctx)
        self._n_events += 1

    # ---------- 回忆: 整体补全 (无逐字符预测) ----------

    def recall(self, in_codes, length):
        """线索(输入序列) → 整体补全回复序列 (块内字符匹配)"""
        query = self.bind(in_codes)
        raw = self.E @ query
        codes = []
        for k in range(min(length, self.max_pos)):
            lo = k * self.block_size
            rk = raw[lo:lo + self.block_size]
            # 该位置全部候选字符与块内信号内积 → argmax (联想匹配, 非学习)
            scores = self.p[k] @ rk
            codes.append(int(scores.argmax()))
        return codes

    def recall_with_margin(self, in_codes, length):
        """整体补全 + 逐位置 margin (argmax 分数 − 第二高分)

        用于 experiment19 门控判据评估 (experiment14 表明位置头 margin
        无判别力, 本方法验证 P1 事件记忆的 margin 是否可用)。
        """
        query = self.bind(in_codes)
        raw = self.E @ query
        codes, margins = [], []
        for k in range(min(length, self.max_pos)):
            lo = k * self.block_size
            rk = raw[lo:lo + self.block_size]
            scores = self.p[k] @ rk
            top2 = torch.topk(scores, 2)
            codes.append(int(top2.indices[0]))
            margins.append(float(top2.values[0] - top2.values[1]))
        return codes, margins


class AutoRegressiveEventMemory:
    """② 自回归事件记忆 v2 (experiment20): 上下文消歧的前缀续写

    v1 证伪 (裸前缀索引): 回复前缀在语料中高度共享 (a/he/the...), E2
    同一列叠加多对话的下一步字符 → 共享前缀投票污染, 库内 12.1%,
    margin 判别力消失 (246 vs 217)。
    v2 修正 (文献: Drieu & Zugaro 2019 — theta 序列 = 外部输入 + 内在
    动力学整合): 查询 = bind(输入) + bind(前缀), 输入作情境消歧标签,
    前缀作进度指示。输入与回复分占不同位置块 (正交, 无冲突)。

    机制:
      E2 关联 (Hebbian 一次性写入, 无 RPE): 每对话每前缀 t:
        E2 += outer(p[t][c_t], bind(输入,0) + bind(回复前缀, max_pos_in))
      生成: query = bind(输入,0) + bind(已生成前缀, max_pos_in)
        → raw = E2 @ query → 块 (max_pos_in + t) 解码下一字符
      (t=0 首字符由外部输入链路提供, 不存自回归关联)

    ★ 红线检查: 学习 = Hebbian 外积; 解码 = 固定位型内积 argmax;
      无梯度/无BP/无连续信号/无批量/无偏置。
    """

    def __init__(self, dim=12288, char_ones=8, max_pos_in=32,
                 max_pos_out=32, seed=7):
        n_blocks = max_pos_in + max_pos_out
        assert dim % n_blocks == 0
        self.dim = dim
        self.block_size = dim // n_blocks
        self.char_ones = char_ones
        self.max_pos_in = max_pos_in
        self.max_pos_out = max_pos_out
        g = torch.Generator(device="cpu").manual_seed(seed)

        # 位置-字符联合位型: p[k][c] 是"回复位置 k"块的位型 (不学习)
        self.p = torch.zeros(max_pos_out, 256, self.block_size, device=DEVICE)
        for k in range(max_pos_out):
            for c in range(256):
                idx = torch.randperm(self.block_size, generator=g)[:char_ones]
                self.p[k, c, idx] = 1.0

        # 情境(输入)+进度(前缀) → 下一字符 关联矩阵 (Hebbian 叠加)
        self.E2 = torch.zeros(dim, dim, device=DEVICE)
        self._n_prefixes = 0

    def bind(self, codes, offset):
        """字符序列 → 事件码, 占 offset 起的连续位置块
        输入与回复共用字符位型 p[j][c], 但落在不同块空间 (正交)。"""
        v = torch.zeros(self.dim, device=DEVICE)
        for j, c in enumerate(codes):
            k = offset + j
            if k >= (self.max_pos_in + self.max_pos_out):
                break
            lo = k * self.block_size
            v[lo:lo + self.block_size] += self.p[j % self.max_pos_out, c]
        return v

    def store_dialogue(self, in_codes, out_codes):
        """一个对话: 每个回复前缀 t 存 E2 += outer(p[t][c_t], 情境+进度)"""
        ctx_base = self.bind(in_codes, 0)
        for t, c in enumerate(out_codes[:self.max_pos_out]):
            if t == 0:
                continue  # 首字符由输入链路提供
            ctx = ctx_base + self.bind(out_codes[:t], self.max_pos_in)
            nxt = torch.zeros(self.dim, device=DEVICE)
            lo = (self.max_pos_in + t) * self.block_size
            nxt[lo:lo + self.block_size] = self.p[t, c]
            self.E2 += torch.outer(nxt, ctx)
            self._n_prefixes += 1

    def next_char(self, in_codes, prefix_codes):
        """情境(输入) + 进度(前缀) → 下一字符 (code, margin)"""
        t = len(prefix_codes)
        if t <= 0 or t >= self.max_pos_out:
            return None, 0.0
        query = self.bind(in_codes, 0) + self.bind(prefix_codes, self.max_pos_in)
        raw = self.E2 @ query
        lo = (self.max_pos_in + t) * self.block_size
        rk = raw[lo:lo + self.block_size]
        scores = self.p[t] @ rk
        top2 = torch.topk(scores, 2)
        return int(top2.indices[0]), float(top2.values[0] - top2.values[1])
