"""
PyTorch GPU 加速 0-1 膜电位脉冲神经网络 — 生物学架构 (v14)

======================================================================
★ 核心架构哲学 — 四层隐藏层 + 双记忆层 + 选择性连接 + 侧抑制 + 关联学习
======================================================================

0. ★ 四层隐藏层 (v10) — 皮层分层处理:
   - L1→L2→L3→L4 前馈结构 (对应皮层 L2/3, L4, L5, L6)
   - L1 权重 W_ih 随机固定; L2-L4 层间权重 W_deep 奖赏调制 Hebbian 训练
   - 每层内部: 0-1 膜电位积分 + 侧抑制 (WTA)
   - 层间无反馈 / 无 W_hh 循环 / 无时间步迭代

1. ★ 0-1 膜电位神经元 — 电位强度 [0, 1] + 二值发放:
   - 每个神经元具有膜电位 V ∈ [0, 1]，表示电位强度
   - 漏电积分: V = V × (1 - leak) + input_current，钳位到 [0, 1]
   - 发放: output = 1 if V > threshold else 0 (二值发放)
   - 发放后部分重置: V = V × (1 - reset_factor) (保留残余电位)
   - 生物学依据: 神经元膜电位为连续值，发放是"全或无"事件
   - ★ 膜电位 V 携带 0-1 强度信息，输出仍为二值 {0, 1}

2. ★ 选择性连接 (稀疏突触):
   - 每个神经元只连接随机 ~50% 的输入神经元
   - 连接掩码: connection_mask = random > 0.5 (固定，不训练)
   - 生物学依据: 生物神经元不会连接到所有其他神经元
   - 减少冗余连接，提高计算效率

3. ★ 侧抑制 (Lateral Inhibition):
   - 发放的神经元抑制未发放的邻近神经元 (降低其膜电位)
   - 强度: ΔV = -inhibition_strength × (fired_ratio) × (1 - fired)
   - 生物学依据: 抑制性中间神经元实现 Winner-Take-All 机制
   - 防止神经元群体过度活跃，维持稀疏编码

4. ★ 奖赏预测误差 (RPE) 调制 Hebbian 学习 — 纯生物学规则，无梯度下降:
   - Δw = lr × RPE_j × pre_activity_i
   - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差 (期望发放 − 实际发放)
     +1: 应发未发 (该位期望发放但未发放 → 强化活跃突触)
     −1: 误发 (该位不应发放但发放了 → 削弱活跃突触)
      0: 预测正确 (无更新，奖赏已完全被预测)
   - pre_activity_i = 突触前神经元 i 的二值活动 {0, 1}
   - ★ 无偏置更新: 偏置是连续数值运算，已移除
   - 生物学依据: 多巴胺能神经元编码奖赏预测误差 (Schultz 1997)，
     投射到皮层门控突触可塑性 (dopamine-gated Hebbian plasticity, Izhikevich 2007)
   - 无梯度下降 / 反向传播 / 自动微分 / 批量优化 / 目标误差

5. ★ 关联记忆层 (Associative Memory, v10) — 共激活追踪 + 回忆提取:
   - 共激活矩阵 W_coact: 追踪神经元对的共发放频率
   - "Fire together, wire together": 共发放的神经元之间强化连接
   - 回忆: recall = clamp(W_coact·cue / (H/2), 0, 1) (分级 Hebbian 回忆, v12.4)
   - ★ v12.4: 由二值化 (raw > 0.5) 改为分级回忆强度。二值化丢失强度信息
     → 状态判别性崩塌 (5/14); 分级 → 14/14 (diag_ctx_discrim)
   - 生物学依据: 海马体关联记忆 → 皮层回忆

6. ★ 工作记忆层 MemWork (v12.2) — "now 模式", 无跨字符累积:
   - MemWork = max(v_peak, recall); v_peak = V_deep[-1] (V 跨字符持续累积, 不 reset)
   - recall = 分级回忆强度 (v12.4, 非二值; 先除期望活跃数再钳位, 保留分布形状)
   - v11 移除 forget_mask 与跨字符 max 累积 (无逐字符 reset, 无 OR 累积)
   - 状态携带 0-1 分级信息，非纯二值
   - ★ 实验验证 (experiment9 / diag_cmp3): 顺序 = update 先于 recall
     + v_peak 取 V 累积 + 不 reset 是已验证最优行为 (9-11/14);
     任何变更 (recall 先、v_peak 取输出、逐字符 reset) → 判别性崩塌 (0-2/14)
   - 生物学依据: 工作记忆通过神经元群体持续发放维持 (前额叶)

6b. ★ 渐进式深度训练 (v12) — 逐层加深, 课程式学习:
   - 阶段 depth=1..4: 先初始化新中间层为恒等映射 (scale=1.0, noise=0.0),
     再训练 W_h2o (RPE) + 李层 (仅当 mean_rpe>0 时 ΔW=lr×mean_rpe×outer(后层,前层))
   - 恒等初始化 = 纯中继: 无错误 → mean_rpe=0 → 不更新 → 表示不被破坏
   - 实验验证: noise=0.05 累积 σ≈0.57 → 6/72; noise=0.01 → 训练崩溃;
     scale=1.0/noise=0.0 完美中继, 各阶段解码 72/72

6c. ★ 确定性 W_coact 快照 (v12.3) — 训练/评估状态逐位一致:
   - 收集: 每个对话编码前记录 W_coact.clone() → _coact_snapshots (连续累积保留)
   - 评估: 恢复对应快照 + update_memory=True 重放 → 与训练状态逐位一致
   - W_seq 同样方案: train_sequence 记录 _seq_snapshots (逐对话快照)
   - 反例: 单一冻结快照 → 14 状态余弦 1.000 (判别性崩塌); 评估时继续累积
     → 状态漂移 (重编码 2/14); 快照恢复 → 同状态集评估一致 (11/14)

6d. ★ 分级关联回忆 (v12.4) — 回忆强度保留 0-1 分级, 不二值化:
   - 回忆: recall = clamp(W_coact·cue / (H/2), 0, 1) (除以期望活跃数再钳位)
   - 反例1 二值化 (raw > 0.5): 丢失强度信息 → 状态判别性崩塌
     (独立 W_ctx_to_first 5/14 vs 分级 14/14, diag_ctx_discrim)
   - 反例2 clamp(raw, 0, 1): raw 量级常 > 1, 直接钳位大量饱和 → 4-5/14
   - 先除期望最大共激活数再钳位才保留分布形状 (14/14)

6e. ★ 位置记忆头 (v13) — 记忆层修正非首字输出:
   - W_ctx_to_pos[k] (8×256): 上下文状态 → 回复第 k 字符, 与 W_ctx_to_first
     同机制 (纯二值阈值 + RPE 调制 Hebbian, 无偏置更新, W clamp ±10)
   - 生成时每步用位置记忆头对上下文状态 cf 回忆"回复第 step 字符"并
     直接覆盖 W_seq 候选 (记忆优先); 超出已训练位置回退 W_seq
   - ★ 实验验证 (experiment14): 位置头回忆 96.6% (14 对话, 快照一致);
     端到端修正 (θ=0 全量覆盖) 字符级 96.6% / 完整 6/14
   - ★ 反例: margin 门控无判别力 — 正确回忆 margin med=0.09 vs 错误
     med=0.05, 重叠严重; 门控阈值 θ>0 回退 W_seq 烂输出 (θ=0.5 → 28.6%,
     低于无门控 96.6%)。margin = min_j|raw_j| 不是可靠置信度
   - ★ 快照一致性: train_pos_heads 必须复用 _coact_snapshots 恢复状态,
     否则训练状态漂移 → 回忆 77.4% → 96.6%

7. ★ 随机稠密编码:
   - 每个字符 → 固定的随机 256-dim 二值向量 (~50% 活跃)
   - 确保不同字符的输入模式几乎正交，无需学习输入编码

======================================================================
★ 禁止走错的方向 (DON'T) — 阅读并遵守，否则架构会崩:
======================================================================
  ✗ 禁止使用 sigmoid / tanh / softmax 等连续数值映射函数
  ✗ 禁止使用梯度下降 / 反向传播 / autograd / 批量优化 / 目标误差
  ✗ 禁止使用余弦相似度 / 向量检索 / 列表匹配 / KNN / 最近邻
  ✗ 禁止使用时间步迭代 (timesteps) — 神经元是瞬时计算
  ✗ 禁止使用 W_hh 循环连接 (状态累积用 V 电位 + 随机遗忘)
  ✗ 禁止将目标值直接用于权重复新 (目标值仅用于判断正确/错误)
  ✗ 禁止将 W_ih 设为可训练 (随机固定，永不训练)
  ✗ 禁止使用任何形式的误差信号 / 损失函数 / 梯度计算
  ✗ 禁止更新偏置 (b_o, b_seq) — 偏置是连续数值运算
  ✗ 禁止使用加权平均更新状态 (state×α + output×β)

======================================================================
核心组件:
  1. 随机稠密编码: 字符 → 256-dim 二值向量
  2. W_ih (256×256): 随机固定权重 + 选择性连接掩码
  3. 0-1 膜电位神经元: V = clamp(V×(1-leak) + input, 0, 1), out = V>threshold
  4. 侧抑制: 发放神经元抑制未发放神经元
  5. 共激活矩阵 W_coact: 追踪关联学习
  6. W_h2o (8×256): 纯二值阈值解码 + 奖赏调制 Hebbian
  7. W_ctx_to_first (8×256): 纯二值阈值解码 + 奖赏调制 Hebbian
  7b. W_ctx_to_pos[k] (8×256): 位置记忆头 (v13), 上下文状态 → 回复第 k 字符,
      生成时逐位置修正 W_seq 的非首字输出 (margin 门控)
  8. W_seq (8×256): 纯二值阈值解码 + 奖赏调制 Hebbian
  9. 生成循环: 字符→随机编码→膜电位神经元→W_seq→下一字符→循环
  10. 状态累积: 0-1 V 电位 + 随机遗忘 (无加权平均)
  11. 学习规则: Δw = lr × RPE_j × pre_i (RPE = 期望发放 − 实际发放)
======================================================================
"""

import math
import random
import time
import numpy as np
import torch

# ============================================================
# ★ 二值阈值神经元参数 (非 LIF，无连续数值)
# ============================================================
# 无膜电位、无漏电流、无不应期、无时间步
# 神经元只做一件事: 加权和 > 0 则激活，否则不激活

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 1024              # 默认隐藏层神经元数 (v14.5 扩大规模: 256→1024)


class TorchLIFSimulator:
    """PyTorch GPU 加速 0-1 膜电位脉冲神经网络 — 纯前向计算，无 autograd。

    ★ 重要: 类名保留 LIF 仅向后兼容，实际使用 0-1 膜电位神经元 (非经典 LIF)。

    关键设计决策 (v9):
      - 神经元膜电位 V ∈ [0, 1]，携带电位强度信息
      - 漏电积分: V = V × (1 - leak) + input_current, clamp to [0, 1]
      - 发放: output = 1 if V > threshold else 0 (二值输出)
      - 发放后部分重置: V = V × (1 - reset_factor)
      - 侧抑制: 发放神经元抑制未发放神经元 (WTA)
      - 选择性连接: 每个神经元只连接 ~50% 的输入神经元
      - 解码: 纯二值阈值 (W·x + b > 0).float(), 无 sigmoid
      - 所有学习规则为奖赏预测误差调制 Hebbian (Δw = lr×RPE×pre)
      - RPE_j = target_j − out_j ∈ {−1, 0, +1} (多巴胺奖赏预测误差, Schultz 1997)
      - 无梯度下降 / 反向传播 / 批量优化 / 目标误差
    """

    def __init__(self, hidden_size=HIDDEN_SIZE, output_size=8, input_bias=1.0,
                 leak=0.1, threshold=0.5, reset_factor=0.3, inhibition_strength=0.2,
                 num_layers=1,
                 sfa_inc=0.0, sfa_decay=0.05, prospective=False):
        """★ v2.1: sfa_inc/sfa_decay — 神经元慢变量 (频率适应 SFA, Opt-in 默认关)

        生物学依据 (Subramoney 2024): 近期信息可先从神经元有效发放阈值等
        慢变量中即时解码 — 快速学习不依赖突触塑性。发放后阈值偏移升高
        (频率适应, 该神经元更难再次发放), 每步按 sfa_decay 缓慢衰减 —
        慢变量跨字符持续, 把"最近发放历史"留在神经元状态里, 对抗
        LIF 无快照的状态漂移 (开放对话乱码根因之一)。

        红线审核: 阈值偏移是神经元固有动力学 (非突触修改), 无梯度/
        无连续学习信号, 已获用户确认放宽。sfa_inc=0.0 时完全等价旧行为。
        """
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_bias = input_bias
        self.num_layers = num_layers  # ★ 隐藏层层数 (v10: 支持多层堆叠)

        # ============================================================
        # ★ 0-1 膜电位参数 (v9)
        # ============================================================
        self.leak = leak                    # 漏电率: 每步电位衰减比例
        self.threshold = threshold          # 发放阈值: V > threshold 则发放
        self.reset_factor = reset_factor    # 发放后重置比例: V *= (1 - reset_factor)
        self.V = torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)  # L1 膜电位 [0, 1]

        # ============================================================
        # ★ 慢变量 (SFA, v2.1): 发放后阈值偏移 (Opt-in, sfa_inc=0 关闭)
        # ============================================================
        self.sfa_inc = sfa_inc                    # 每次发放后的阈值偏移增量
        self.sfa_decay = sfa_decay                # 慢变量衰减率 (须比 leak 慢)
        self.thr_shift = torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)   # L1 阈值偏移
        self.thr_shifts_deep = [torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)
                                for _ in range(max(num_layers - 1, 0))]  # L2..LN 阈值偏移

        # ============================================================
        # ★ 侧抑制参数 (v9)
        # ============================================================
        self.inhibition_strength = inhibition_strength  # 侧抑制强度

        # ============================================================
        # ★ 选择性连接掩码 — 每个神经元只连接随机 ~50% 的输入 (v9)
        # ============================================================
        self.connection_mask = None  # 由 init_random_weights 初始化

        # ============================================================
        # ★ W_ih: 随机固定输入权重 (256×256)，永不训练
        # ============================================================
        self.W_ih = torch.zeros(hidden_size, hidden_size, dtype=torch.float32, device=DEVICE)

        # ★ v10 多层隐藏层: 层间权重 L_k→L_{k+1} (256×256)，奖赏调制 Hebbian 训练
        #   Δw = lr × global_RPE × out_post ⊗ out_pre (仅 RPE>0 时强化)
        #   初始化延迟到 init_random_weights (需要 scale/sparsity 参数)
        self.W_deep = []       # 层间权重列表 [W_l1l2, W_l2l3, ...]
        self.deep_masks = []   # 层间选择性连接掩码列表
        self.V_deep = [torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)
                       for _ in range(max(num_layers - 1, 0))]  # L2..LN 膜电位

        # ★ W_h2o: 解码权重 (8×256)，随机初始化小值，奖赏调制 Hebbian 训练
        self.W_h2o = torch.empty(output_size, hidden_size, dtype=torch.float32, device=DEVICE)
        self.W_h2o.uniform_(-0.1, 0.1)
        # b_o: 解码偏置 (8,)，随机初始化小值
        self.b_o = torch.empty(output_size, dtype=torch.float32, device=DEVICE)
        self.b_o.uniform_(-0.1, 0.1)

        # ============================================================
        # ★ v2.6 前瞻编码 (Brea 2016, Opt-in 默认关): W_prosp 预测
        #   神经元"自身下一步发放" — 内部预测回路 (无外部答案依赖)
        # ============================================================
        # 生物学依据 (Brea 2016, PLoS Comp Biol "Prospective Coding by
        # Spiking Neurons"): STDP 式局部规则让神经元匹配**自己预期的
        # 未来折扣发放率** (等价 TD(λ)) — 预测目标是自身未来发放,
        # 不是外部答案 → 内部自生成驱动, 摆脱"刺激→联想→反应"闭式链。
        #
        # 本实现: 学习规则与 W_seq 同族 (Δw = lr×RPE×pre), 但目标为
        #   target = 下一时刻工作记忆发放位型 (二值, 离散)
        #   RPE = target − pred ∈ {−1,0,+1} (离散奖赏, 红线合规)
        # 生成时 (自检回路, Ororbia 2019 SNPC "猜-查"): 候选字符经
        #   前瞻预测"若生成该字符, 下一步状态是否符合内部预期"择优。
        self.prospective = prospective
        # Note: In RecurrentLIFSimulator this will be re-initialized if feat_dim != hidden_size
        self.W_prosp = torch.zeros(hidden_size, hidden_size,
                                   dtype=torch.float32, device=DEVICE)
        self.W_prosp.uniform_(-0.1, 0.1)

    def init_random_weights(self, scale=0.8, connection_sparsity=0.5):
        """初始化随机固定 W_ih + 选择性连接掩码 (+ v10 层间权重)

        Args:
            scale: 权重均匀分布范围 [-scale, scale]
            connection_sparsity: 连接稀疏度 (0=全连接, 1=全断开)
                                 默认 0.5 表示每个神经元只连接 ~50% 的输入
        """
        self.W_ih = torch.empty(self.hidden_size, self.hidden_size,
                                dtype=torch.float32, device=DEVICE)
        self.W_ih.uniform_(-scale, scale)

        # ★ 选择性连接掩码 — 每个神经元只连接随机子集的输入 (v9)
        if connection_sparsity > 0:
            mask = torch.rand(self.hidden_size, self.hidden_size,
                              device=DEVICE) > connection_sparsity
            self.connection_mask = mask.float()
            self.W_ih = self.W_ih * self.connection_mask
        else:
            self.connection_mask = None

        # ★ v10: 初始化层间权重 L_k→L_{k+1} (随机小值 + 选择性连接掩码)
        self.W_deep = []
        self.deep_masks = []
        for _ in range(max(self.num_layers - 1, 0)):
            W_l = torch.empty(self.hidden_size, self.hidden_size,
                              dtype=torch.float32, device=DEVICE)
            W_l.uniform_(-scale, scale)
            if connection_sparsity > 0:
                m_l = (torch.rand(self.hidden_size, self.hidden_size,
                                  device=DEVICE) > connection_sparsity).float()
                W_l = W_l * m_l
            else:
                m_l = None
            self.W_deep.append(W_l)
            self.deep_masks.append(m_l)

    # ============================================================
    # 字符 → 结构化随机稠密编码映射表 (v14.5 方案 B)
    #
    # 前 output_size 位 = ASCII 码的 bit 位型 (双极性 {-1, +1}, 保证 W_h2o 线性可分)
    # 后 hidden_size - output_size 位 = 随机二值 (隐藏层特征多样性)
    #
    # 理论依据:
    #   单层感知器 (W_h2o) 的决策边界是超平面。纯随机编码中 ASCII 码 bit
    #   与编码向量无结构关系 → N > d 时线性不可分概率剧增 (Cover 1965)。
    #   结构化前缀将目标位显式编码 → 每个 bit j 的分类器直接在位置 j
    #   找到对应位 → 感知器收敛定理保证有限步内收敛。
    #
    #   双极性 {-1, +1}: target=1 → +1, target=0 → -1。
    #   单一权重 W_h2o[j,j] 承担两类判别, 信号不被随机噪声淹没。
    # ============================================================
    CHAR_CODEBOOK = {}  # 延迟初始化, 键 (ch, dim, output_size)

    def _get_char_code(self, ch):
        """获取字符的结构化随机稠密编码 (二值 {0,1})

        编码结构:
          [0..output_size-1]:           ASCII 码 bit 位型, 双极性 {-1, +1}
          [output_size..hidden_size-1]: 随机二值 {0, 1} (RNG seed=ord(ch))
        """
        out_size = self.output_size
        orth = getattr(self, "use_orthogonal_char", False)
        key = (ch, self.hidden_size, out_size, orth)
        if key not in TorchLIFSimulator.CHAR_CODEBOOK:
            code = ord(ch) if len(ch) == 1 else 0
            rng = np.random.RandomState(code)
            if orth:
                # ★ v3.6 正交字符编码: 全 hidden_size 维独立随机二值
                #   (双极性 {-1,+1}), 去除 ASCII 位型重叠, 让不同字符
                #   在 hidden 层即近似正交 (高维随机向量 cos≈0)。
                vec = torch.from_numpy(
                    (rng.rand(self.hidden_size) > 0.5).astype(np.float32)).to(DEVICE)
                vec = 2.0 * vec - 1.0
            else:
                struct_bits = torch.tensor(
                    [float((code >> j) & 1) for j in range(out_size)],
                    dtype=torch.float32, device=DEVICE)
                struct_bits = 2.0 * struct_bits - 1.0  # 双极性 {0,1}→{-1,+1}

                rand_size = self.hidden_size - out_size
                if rand_size > 0:
                    rand_part = torch.from_numpy(
                        (rng.rand(rand_size) > 0.5).astype(np.float32)).to(DEVICE)
                    vec = torch.cat([struct_bits, rand_part])
                else:
                    vec = struct_bits

            TorchLIFSimulator.CHAR_CODEBOOK[key] = vec
        return TorchLIFSimulator.CHAR_CODEBOOK[key]

    def _char_to_8bit(self, ch):
        """字符 → 结构化随机稠密编码 (兼容旧接口名)

        ★ v3.3 性能: 编码为只读缓存张量, 不再逐字符 clone
        (所有调用方只读: _layer_forward 仅做 mv, 不原地修改输入)。
        """
        return self._get_char_code(ch)

    def _char_to_8bit_bias(self, ch):
        """字符编码 + input_bias 的缓存合成向量 (逐字符热路径免分配)"""
        key = (ch, self.input_bias)
        cb = getattr(self, "_char_bias_codebook", None)
        if cb is None:
            cb = self._char_bias_codebook = {}
        v = cb.get(key)
        if v is None:
            v = self._get_char_code(ch) + self.input_bias
            cb[key] = v
        return v

    def _target_bits(self, code):
        """字节 → 8-bit 目标位型 缓存 (逐字符热路径免 Python 位运算+分配)"""
        c = getattr(self, "_target_cache", None)
        if c is None:
            c = self._target_cache = {}
        v = c.get(code)
        if v is None:
            v = torch.tensor(
                [float((code >> j) & 1) for j in range(self.output_size)],
                dtype=torch.float32, device=DEVICE)
            c[code] = v
        return v

    def _text_to_codes(self, text):
        """文本 → 字节码列表 (0-255, 接受全部值, 用于中文 2 字节编码)"""
        return [ord(c) for c in text]

    # ============================================================
    # ★ 0-1 膜电位神经元 (替代纯二值阈值神经元 v8)
    #
    # 生物学原理:
    #   神经元膜电位 V ∈ [0, 1] 表示电位强度。
    #   输入信号使膜电位升高 (积分)，漏电使膜电位衰减。
    #   当 V > threshold 时，神经元发放 (output=1)。
    #   发放后部分重置: V = V × (1 - reset_factor)，保留残余电位。
    #
    # 计算 (v9):
    #   1. 漏电: V = V × (1 - leak)
    #   2. 积分: V = clamp(V + sum(W_ih × input), 0, 1)
    #   3. 发放: output = 1 if V > threshold else 0
    #   4. 重置: V = V × (1 - reset_factor) if fired
    #   5. 侧抑制: 未发放的神经元 V 被抑制
    #
    # 与旧版 (v8) 的关键区别:
    #   - v8: 无膜电位，直接 output = (sum(W×input) > 0)
    #   - v9: 有膜电位 V ∈ [0, 1]，携带电位强度信息
    #   - v8: 无侧抑制
    #   - v9: 侧抑制实现 WTA 机制
    #   - v8: 状态累积使用纯二值 output
    #   - v9: 状态累积使用 V (0-1 分级电位)
    # ============================================================
    def reset_state(self):
        """重置所有层膜电位为 0 — 处理新序列前调用 (v14.5 原地清零, 免显存分配)"""
        self.V.zero_()
        for i in range(len(self.V_deep)):
            self.V_deep[i].zero_()
        # ★ v2.1: 重置慢变量 (每个新序列从头累积)
        self.thr_shift.zero_()
        for i in range(len(getattr(self, "thr_shifts_deep", []))):
            self.thr_shifts_deep[i].zero_()
        # ★ v14.12: 重置循环思考储层 (每个新序列从头累积)
        if hasattr(self, "reservoir"):
            self.reservoir.zero_()

    def _lateral_inhibition(self, fired_neurons, V=None):
        """★ 侧抑制 — 发放的神经元抑制未发放的神经元

        生物学原理:
          抑制性中间神经元接收兴奋性输入后，释放抑制性递质
          降低邻近神经元的膜电位，实现 WTA (Winner-Take-All)。

        计算:
          抑制量 = inhibition_strength × (发放比例)
          未发放神经元: V -= 抑制量
          已发放神经元: 不受影响 (已重置)

        Args:
            fired_neurons: 二值张量, 1=发放, 0=未发放
            V: 被抑制的膜电位张量 (默认 self.V)
        """
        if V is None:
            V = self.V
        # 发放比例 (张量级标量, 无 .item() 同步 — 消除 GPU 流水线阻塞)
        # 无发放时 inhibition=0, 与旧逻辑 (early return) 完全等价
        fired_ratio = fired_neurons.mean()

        # 抑制量: 与发放比例成正比
        inhibition = self.inhibition_strength * fired_ratio

        # 只有未发放的神经元被抑制
        inhibition_mask = 1.0 - fired_neurons  # 未发放 = 1, 已发放 = 0
        V.copy_(torch.clamp(V - inhibition * inhibition_mask, 0.0, 1.0))

    def _layer_forward(self, layer_idx, input_vec):
        """★ 第 layer_idx 层 0-1 膜电位神经元前向计算 — V ∈ [0, 1], 输出二值 {0, 1}

        ★ v10: 多层版本，层间权重可被奖赏调制 Hebbian 训练。
        每层动力学与单层一致: 漏电 → 积分 → 发放 → 重置 → 侧抑制。

        Args:
            layer_idx: 层索引 (0 = 输入层后的第一隐藏层)
            input_vec: 上一层输出 (256-dim, 二值 {0,1})

        Returns:
            output: 该层二值发放输出 (256-dim, 每个元素 0 或 1)
        """
        # 该层权重/掩码/膜电位
        if layer_idx == 0:
            W = self.W_ih
            mask = self.connection_mask
            V = self.V
            shift = self.thr_shift
        else:
            W = self.W_deep[layer_idx - 1]
            mask = self.deep_masks[layer_idx - 1]
            V = self.V_deep[layer_idx - 1]
            shift = self.thr_shifts_deep[layer_idx - 1]

        # 1. 漏电: 膜电位按比例衰减
        V = V * (1.0 - self.leak)

        # 2. 积分: 加权输入 (带选择性连接掩码)
        W_m = W * mask if mask is not None else W
        activation = torch.mv(W_m, input_vec)
        V = torch.clamp(V + activation, 0.0, 1.0)

        # 3. 发放: V > threshold 则输出 1
        #    ★ v2.1 慢变量: 启用时有效阈值 = threshold + 阈值偏移 (频率适应)
        if self.sfa_inc > 0:
            output = (V > (self.threshold + shift)).float()
        else:
            output = (V > self.threshold).float()

        # 4. 发放后部分重置: 保留残余电位
        V = torch.where(output > 0, V * (1.0 - self.reset_factor), V)

        # 5. 侧抑制: 发放神经元抑制未发放神经元
        if self.inhibition_strength > 0:
            self._lateral_inhibition(output, V)

        # ★ v2.1 慢变量更新: 发放后阈值偏移升高 (频率适应), 每步缓慢衰减。
        #   慢变量跨字符持续 (reset_state 才清零), 编码"最近发放历史"。
        if self.sfa_inc > 0:
            shift = shift * (1.0 - self.sfa_decay) + output * self.sfa_inc
            shift.clamp_(0.0, 1.0)

        # 写回该层膜电位
        if layer_idx == 0:
            self.V = V
            self.thr_shift = shift
        else:
            self.V_deep[layer_idx - 1] = V
            self.thr_shifts_deep[layer_idx - 1] = shift

        return output

    def _multi_layer_forward(self, input_vec, active_depth=None, n_loops=1):
        """★ v10: 多层前向传播 — 输入 → L1 → L2 → ... → LN, 返回最深层输出

        层间为前馈连接 (无 W_hh 循环, 无时间步迭代)，
        每层内部完成 0-1 膜电位积分 + WTA 侧抑制。

        ★ v12 渐进式深度训练: active_depth 限制只前向经过前 depth 层
          (用于逐层加深的课程训练, 见 train_multi_layer_stdp)。

        ★ v13.1 自回归循环 (用户指定方案): n_loops 次完整前向 —
          输入 → 8 层 → 输出1 → (输出1 作为输入) → 8 层 → 输出2 → ...
          "每次输入后自回归循环一次再传入输入"。轮次间 V 膜电位继续
          累积 (不 reset, 短时记忆延续); 每轮仍为纯前馈瞬时计算。
          n_loops=1 时与旧行为完全一致 (向后兼容)。
        """
        if active_depth is None:
            active_depth = self.num_layers
        out = input_vec
        for _ in range(n_loops):
            for k in range(min(active_depth, self.num_layers)):
                out = self._layer_forward(k, out)
        return out

    def _memory_to_lif_input(self, memory_state):
        """短期记忆状态投影回 LIF 输入空间。"""
        if memory_state.numel() == self.hidden_size:
            feedback = memory_state
        else:
            if (self.W_mem2lif is None or
                    self.W_mem2lif.shape != (self.hidden_size, memory_state.numel())):
                generator = torch.Generator(device=DEVICE)
                generator.manual_seed(20260814)
                self.W_mem2lif = torch.randn(
                    self.hidden_size, memory_state.numel(), dtype=torch.float32,
                    device=DEVICE, generator=generator) / (memory_state.numel() ** 0.5)
            feedback = torch.mv(self.W_mem2lif, memory_state)
        feedback = feedback - feedback.mean()
        scale = feedback.abs().max().clamp_min(1e-6)
        return feedback / scale

    def _think_character(self, char_input, update_memory=True, n_loops=1):
        """单字符触发 LIF 与短期记忆闭环, 状态稳定后返回。"""
        previous = self.MemWork.clone()
        stable_streak = 0
        output = torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)
        state = previous
        sparse_feat = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.thinking_last_steps = 0
        self.thinking_last_delta = 0.0

        for step in range(self.thinking_max_steps):
            if step == 0:
                loop_input = char_input
            else:
                feedback = self._memory_to_lif_input(state)
                strength = self.thinking_feedback_strength
                loop_input = torch.clamp(
                    char_input * (1.0 - strength) + feedback * strength,
                    -1.0, 1.0)

            output = self._multi_layer_forward(loop_input, n_loops=n_loops)
            v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
            sparse_feat = self._dg_separate(v_curr)
            recall = self.recall_from_memassoc(sparse_feat, sparse_hint=True)
            state = torch.max(sparse_feat, recall)
            self.MemWork = state

            delta = (state - previous).abs().mean().item()
            self.thinking_last_steps = step + 1
            self.thinking_last_delta = delta
            if delta <= self.thinking_delta_threshold:
                stable_streak += 1
            else:
                stable_streak = 0
            if stable_streak >= self.thinking_stable_steps:
                break
            previous = state.clone()

        if update_memory:
            self.update_coactivation(sparse_feat)

        return output, sparse_feat, state

    def _multi_layer_forward_all(self, input_vec, active_depth=None, n_loops=1):
        """★ v10: 多层前向传播，返回所有层的输出列表 [L1, L2, ..., LN]

        用于层间奖赏调制 Hebbian 训练 (需要每一层的突触前/后活动)。
        ★ v12: active_depth 限制只前向经过前 depth 层。
        ★ v13.1: n_loops>1 时返回最后一轮各层输出 (层间训练用)。
        """
        if active_depth is None:
            active_depth = self.num_layers
        last_round = []
        out = input_vec
        for _ in range(n_loops):
            round_outs = []
            for k in range(min(active_depth, self.num_layers)):
                out = self._layer_forward(k, out)
                round_outs.append(out)
            last_round = round_outs
        return last_round

    def _neuron_forward(self, input_vec):
        """★ 兼容旧接口: 调用第 0 层使用 0-1 膜电位动力学 (v9)

        命名保留 _neuron_forward 仅向后兼容，实际使用 0-1 膜电位。
        """
        return self._layer_forward(0, input_vec)

    def _binary_forward(self, input_vec):
        """★ 兼容旧接口: 调用 _neuron_forward 使用 0-1 膜电位动力学 (v9)

        命名保留 _binary_forward 仅向后兼容，实际使用 0-1 膜电位。
        """
        return self._neuron_forward(input_vec)

    # ============================================================
    # ★ 纯二值阈值解码: 二值神经元输出 → ASCII 字符
    #
    # 所有解码路径使用纯二值阈值:
    #   out = (W·x + b > 0).float()  ← 输出为 {0, 1}
    #   无 sigmoid, 无 tanh, 无连续数值映射
    #
    # 为什么不用 sigmoid:
    #   - sigmoid 将一个连续值映射到 (0, 1) 区间
    #   - 这是连续数值运算，不是生物学的二值机制
    #   - 纯二值阈值更简单: 加权和 > 0 则激活，否则不激活
    #   - 生物学中神经元没有 sigmoid 这样的全局映射函数
    # ============================================================
    @staticmethod
    def _binary_decode(W, x, b=None):
        """★ 纯二值阈值解码 — 无 sigmoid, 无连续数值输出

        out = (W·x + b > 0).float()  ← 纯二值输出 {0, 1}

        ★ 关键设计:
          - 输出只有二值 {0, 1}, 无连续数值 (raw/中心化信号)
          - 学习规则用奖赏预测误差 RPE_j = target_j − out_j 调制:
            Δw = lr × RPE × pre
          - 生物学依据: 多巴胺能神经元编码奖赏预测误差 (Schultz 1997),
            门控活跃突触的 Hebbian 可塑性 (dopamine-gated plasticity)
          - ★ 无 raw 返回值: 过去版本返回 raw 用于学习信号,
            这是"传入数值运算"的残留, 已移除。
            学习信号只使用二值 out {0, 1}。

        Args:
            W: 权重矩阵 (output_size, input_size)
            x: 输入向量 (input_size,)
            b: 偏置向量 (output_size,), 可选

        Returns:
            out: 纯二值输出向量 (output_size,), 每个元素 0 或 1
                 无 raw 返回值 — 学习规则只使用二值信号
        """
        raw = torch.mv(W, x)
        if b is not None:
            raw = raw + b
        # ★ 纯二值阈值: 无 sigmoid, 无连续数值映射
        out = (raw > 0).float()
        return out

    def fr_to_code(self, fr):
        """二值输出 → ASCII 码 (纯二值阈值解码)

        fr 是二值 {0,1} 向量，W_h2o 权重通过奖赏调制 Hebbian 训练得到
        解码使用纯二值阈值: out = (W_h2o·fr + b_o > 0).float()
        ★ 无 sigmoid, 无连续数值, 输出为纯二值 {0, 1}
        """
        fr_gpu = torch.tensor(fr, dtype=torch.float32, device=DEVICE) if not torch.is_tensor(fr) else fr
        out = self._binary_decode(self.W_h2o, fr_gpu, self.b_o)
        # out 是纯二值 {0, 1}, 直接解码为 ASCII 码
        code = 0
        for j in range(self.output_size):
            if out[j] >= 0.5:  # out[j] 是 0.0 或 1.0
                code |= (1 << j)
        return code

    def fr_to_char(self, fr):
        code = self.fr_to_code(fr)
        return chr(code) if 0 <= code <= 255 else '?'

    def check_decode(self, fr, expected_code):
        return self.fr_to_code(fr) == expected_code


# ============================================================
# ★ W_h2o 训练 — 奖赏预测误差调制 Hebbian 学习 (纯生物学规则)
#
# 替代了之前的批量梯度下降、LIF 动力学、连续数值学习信号。
# 这是纯生物学奖赏学习规则:
#
#   奖赏预测误差调制 Hebbian 规则 (v11):
#     Δw_ji = lr × RPE_j × pre_activity_i
#
#   - pre_activity_i: 突触前神经元 i 的二值活动 (0 或 1)
#   - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差
#     +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
#   - ★ 无 center = clamp(raw, -1, 1): 连续数值运算, 已移除
#   - ★ 无偏置更新: 连续数值运算, 已移除
#   - 生物学依据: 多巴胺能神经元编码奖赏预测误差 (Schultz 1997),
#     门控活跃突触的可塑性 (dopamine-gated Hebbian plasticity)
#   - 预测正确时 RPE = 0, 无多巴胺信号, 无可塑性 (奖赏完全被预测)
#   - 只有活跃的突触 (pre_activity_i = 1) 才被修改 ← 生物学事实
#   - 无批量处理，每个样本独立学习 ← 生物学事实
#   - 无梯度下降 / 反向传播 / 自动微分 / 目标误差
#   - ★ 解码使用纯二值阈值: out = (W·x + b > 0).float(), 无 sigmoid
#
# 为什么不是梯度下降:
#   - 梯度下降需要计算全局误差对每个权重的偏导
#   - 生物学中不存在反向传播误差的机制
#   - 奖赏调制 Hebbian 学习是神经科学已验证的突触可塑性模型
#   - 目标值仅用于生成奖赏预测误差信号 (期望发放)，不参与梯度计算
# ============================================================

def train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=200, verbose=True):
    """★ 奖赏调制 Hebbian 学习训练 W_h2o — 纯生物学规则

    ★ v7.2 关键变更: W_h2o 解码使用随机稠密编码 (而非隐藏层输出)
      原因: 随机 W_ih 产生的隐藏层输出与目标编码不相关,
            奖赏调制 Hebbian 学习无法从随机模式中提取有效信号。
      随机稠密编码是唯一的、与目标直接相关的, 学习效率更高。
      隐藏层输出仍用于上下文状态累积和 W_seq 序列预测。

    每个字符的随机稠密编码 (256-dim, ~50%活跃) 直接输入 W_h2o。
    W_h2o (8×256) 解码随机稠密编码 → 8-bit 字符编码。

    # ★ 学习规则: Δw = lr × RPE_j × pre_activity_i
    #   - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差 (期望发放 − 实际发放)
    #     +1: 该位应发未发 → 强化活跃突触
    #     −1: 该位误发     → 削弱活跃突触
    #      0: 预测正确     → 无更新 (多巴胺在奖赏被完全预测时静默)
    #   - pre_activity_i = 随机稠密编码的 i 位 (二值 {0, 1})
    #   - 无梯度下降，无目标误差，无批量处理，无时间步迭代
    #   - ★ 无 center = clamp(raw, -1, 1): 这是连续数值运算, 已移除
    #   - ★ 无 b_o 偏置更新: 偏置是连续数值运算, 已移除
    #
    # 生物学依据:
    #   多巴胺能神经元编码奖赏预测误差 (Schultz 1997)。
    #   当期望发放与实际发放不一致 (RPE ≠ 0) 时，多巴胺信号
    #   门控活跃突触的 Hebbian 可塑性 (dopamine-gated plasticity)。
    #   预测正确时 RPE = 0，无多巴胺信号，无可塑性 (奖赏完全被预测)。
    #
    # v11 关键变更: 由"逐位对错奖赏 ±1"改为"奖赏预测误差"。
    #   旧规则 Δw = lr×dopamine×out×pre 存在"死神经元"缺陷:
    #   只有当 out=1 (已发放) 时才更新，"应发未发"的位永远学不会。
    #   新规则 Δw = lr×RPE×pre 中 RPE 在应发未发时为 +1，
    #   使缺失位获得强化路径，与感知器学习等价 (决策边界可收敛)。

    Args:
        sim: TorchLIFSimulator 实例
        train_codes: ASCII 码列表
        num_epochs: 训练轮数
        verbose: 是否打印进度
    """
    if verbose:
        print(f"\n--- [奖赏调制 Hebbian] 训练 W_h2o (随机稠密编码直接输入) ---")
    t0 = time.perf_counter()

    n_vocab = len(train_codes)
    output_size = sim.output_size  # 8-bit ASCII
    hidden_size = sim.hidden_size

    # 目标: output_size-bit 编码 (仅用于计算奖赏信号)
    targets_gpu = torch.zeros(n_vocab, output_size, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(train_codes):
        for j in range(output_size):
            targets_gpu[i, j] = float((c >> j) & 1)

    # 输入: 结构化随机稠密编码 (二值, 含 ASCII bit 前缀)
    input_vecs_gpu = torch.zeros(n_vocab, hidden_size, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(train_codes):
        ch = chr(c) if 0 <= c <= 255 else '?'
        input_vecs_gpu[i] = sim._get_char_code(ch)

    # 初始评估 — 直接解码随机稠密编码
    init_correct = sum(1 for i, c in enumerate(train_codes) if sim.check_decode(input_vecs_gpu[i], c))
    if verbose:
        print(f"  初始解码准确率: {init_correct}/{n_vocab} ({init_correct/n_vocab:.1%})", flush=True)

    # 学习率 (Hebbian 学习率，非梯度下降步长)
    lr = 0.5

    for epoch in range(num_epochs):
        # ★ 随机打乱训练顺序 — 模拟生物学学习的不确定性
        indices = list(range(n_vocab))
        random.shuffle(indices)

        correct_count = 0
        for idx in indices:
            # ★ 随机稠密编码直接输入 W_h2o (跳过隐藏层)
            vec = input_vecs_gpu[idx]

            # ★ 纯二值阈值解码: out = (W_h2o·vec + b_o > 0).float()
            #    ★ 学习规则只使用二值 out, 无 raw 返回值
            out = sim._binary_decode(sim.W_h2o, vec, sim.b_o)

            # 目标编码 (用于计算奖赏预测误差)
            target = targets_gpu[idx]

            # ★ 奖赏预测误差 (RPE): RPE_j = target_j − out_j ∈ {−1, 0, +1}
            #   +1: 应发未发 (该位期望发放但未发放) → 强化
            #   −1: 误发 (该位不应发放但发放了)     → 削弱
            #    0: 预测正确                         → 无更新
            #   生物学依据: 多巴胺编码奖赏预测误差 (Schultz 1997)
            pred_bits = (out > 0.5).float()
            target_bits = (target > 0.5).float()
            rpe = target_bits - pred_bits

            # ★ 奖赏预测误差调制 Hebbian 更新 (v11):
            #    Δw_ji = lr × RPE_j × pre_activity_i
            #
            #    生物学原理:
            #    - 多巴胺信号 (RPE) 直接门控活跃突触的可塑性
            #    - 应发未发 (RPE=+1): 强化当前活跃的突触前输入
            #    - 误发 (RPE=-1): 削弱当前活跃的突触前输入
            #    - 预测正确 (RPE=0): 奖赏完全被预测, 无可塑性
            #    - ★ v11 移除 out 门控: 旧规则仅当 out=1 时更新,
            #      "应发未发"的位永远无学习路径 (死神经元)
            #    - ★ 无 center = clamp(raw, -1, 1): 连续数值运算, 已移除
            #    - ★ 无 b_o 偏置更新: 连续数值运算, 已移除
            #    - v14.5 性能优化: 13 次逐行更新合并为一次外积
            #      (ΔW = lr × outer(RPE, pre)), 数学完全等价,
            #      仍是逐样本更新 (非批量), 学习规则不变
            sim.W_h2o += lr * torch.outer(rpe, vec)
            sim.W_h2o.clamp_(-10.0, 10.0)

            # 统计全部正确的样本数
            if (pred_bits == target_bits).all().item():
                correct_count += 1

        acc = correct_count / n_vocab

        if (epoch + 1) % 100 == 0 or acc == 1.0:
            if verbose:
                print(f"    epoch {epoch+1}: acc={correct_count}/{n_vocab} ({acc:.1%}), lr={lr:.4f}", flush=True)
            lr *= 0.95
        if acc == 1.0:
            if verbose:
                print(f"  → 达到完美解码!", flush=True)
            break

    if verbose:
        final_correct = sum(1 for i, c in enumerate(train_codes) if sim.check_decode(input_vecs_gpu[i], c))
        print(f"  最终: acc={final_correct}/{n_vocab}, avg_active_rate=0.5 (固定)", flush=True)
        print(f"  训练完成: {time.perf_counter() - t0:.1f}s", flush=True)


# ============================================================
# RecurrentLIFSimulator — 8-bit 编码 + 真循环纯二值神经元生成
#
# 核心架构 (v7 — 纯二值阈值神经元 + 纯二值阈值解码):
#   1. 随机稠密编码: 每个字符 → 固定的随机 256-dim 二值向量
#   2. W_h2o (8×256): 纯二值阈值解码 + 奖赏调制 Hebbian
#   3. W_ctx_to_first (8×256): 纯二值阈值解码 + 奖赏调制 Hebbian
#   4. W_seq (8×256): 纯二值阈值解码 + 奖赏调制 Hebbian
#   5. 生成循环: 字符→随机编码→二值神经元→W_seq→下一字符→循环
#   6. 状态累积: 二值 OR + 指数衰减 (无 W_hh, 无连续数值运算)
#
# ★ v7 与 v6 的关键区别:
#   - 移除所有 sigmoid 连续数值映射 → 替换为纯二值阈值 (W·x + b > 0).float()
#   - 所有解码路径输出为纯二值 {0, 1}，无连续中间值
#   - 学习规则中的 out 现在是纯二值 0 或 1
#
# 无:
#   - sigmoid / tanh / softmax 等连续数值映射函数
#   - 余弦相似度/向量检索/列表匹配
#   - W_hh 循环连接
#   - 梯度下降/反向传播/批量优化/目标误差
#   - LIF 动力学 (膜电位/漏电/不应期/时间步)
#   - 连续发放率 (神经元输出为 0 或 1)
# ============================================================

def _seq_step_body(W_seq_h, W_seq_out, b_seq, feat, xn, tgt,
                   lr_h, lr_out, thr):
    """W_seq 深度读出口单样本在线更新体 (★ v3.3 供 torch.compile 捕获)

    与 train_sequence SUPER FAST PATH 逐样本计算逐位等价:
      h = (W_seq_h·feat > thr) → rpe_h = xn − h → W_seq_h += lr_h·rpe_h⊗feat
      out = (W_seq_out·h + b > 0) → rpe = tgt − out → W_seq_out += lr_out·rpe⊗h
    Returns:
        (out == tgt).all() 的浮点标量 (0/1)
    """
    h = (torch.mv(W_seq_h, feat) > thr).float()
    rpe_h = xn - h
    W_seq_h.addr_(rpe_h, feat, alpha=lr_h)
    out = (torch.mv(W_seq_out, h) + b_seq > 0).float()
    rpe = tgt - out
    W_seq_out.addr_(rpe, h, alpha=lr_out)
    return (out == tgt).all().float()


class RecurrentLIFSimulator(TorchLIFSimulator):
    """真循环 0-1 膜电位神经元仿真器 — 0-1 膜电位 + 选择性连接 + 侧抑制 + 关联学习

    与 TorchLIFSimulator 的区别:
      - 增加了 W_ctx_to_first (上下文→首字符预测)
      - 增加了 W_seq (序列预测)
      - 增加了共激活矩阵 W_coact (关联学习, "Fire together, wire together")
      - 0-1 分级状态累积 (V 电位 + 随机遗忘)
      - 所有学习使用奖赏预测误差调制 Hebbian (Δw = lr × RPE × pre)
      - 无偏置更新 (b_o, b_seq 不更新)
      - 所有解码使用纯二值阈值: (W·x + b > 0).float(), 无 sigmoid
      - 无 W_hh (状态累积改用 V 电位 + 随机遗忘)
    """

    def __init__(self, hidden_size=HIDDEN_SIZE, output_size=8, input_bias=1.0,
                 leak=0.1, threshold=0.5, reset_factor=0.3, inhibition_strength=0.2,
                 num_layers=4,
                 use_eligibility_trace=False, eligibility_lambda=0.9,
                 use_dg_separation=False, dg_k=32, dg_size_factor=4,
                 sfa_inc=0.0, sfa_decay=0.05, prospective=False,
                 protect_mode="off", protect_strength=0.5,
                 stab_beta=5.0, stab_decay=0.9, freq_thr=0.3,
                 use_reservoir=False, res_dim=1024, res_leak=0.3, res_thr=0.3,
                 goal_strength=0.5, use_experts=False, expert_thr=0.5,
                 expert_heads_cpu=True,
                 enable_long_memory_expansion=False,
                 long_memory_occupancy_threshold=0.5,
                 long_memory_write_gain_threshold=0.05,
                 long_memory_conflict_threshold=0.25,
                 long_memory_pressure_patience=3,
                 use_conjunctive_context=False, conj_k=None,
                 use_hybrid_pos_context=False, pos_context_alpha=0.5,
                 enable_synaptic_capture=False,
                 capture_salience_threshold=0.1,
                 capture_tag_threshold=0.05,
                 use_memory_thinking=False,
                 thinking_max_steps=4,
                 thinking_stable_steps=2,
                 thinking_delta_threshold=0.01,
                 thinking_feedback_strength=0.5,
                 use_orthogonal_char=False):
        """★ v14 (experiment17): 可选三因子资格迹 + DG 稀疏分离

        - use_eligibility_trace: 资格迹三因子学习 (文献: Gerstner & Lehmann
          2018; E-prop)。突触共激活设置资格迹 e_ji ← λ·e_ji + pre×post,
          权重变化由神经调质 RPE 门控: Δw = lr × M_j × e_ji。
          迹保留时间历史 → 学习可在行为时间尺度发生, 权重更新更平滑。
          默认 False = 原即时 RPE 调制 Hebbian (向后兼容)。
        - use_dg_separation: DG 稀疏分离 (文献: CLS Schapiro 2017;
          HiCL 2025 top-k sparsity)。记忆头输入 top-k 二值稀疏化,
          降低串扰 (crosstalk), 提升记忆库容量。默认 False 向后兼容。
        - dg_k: top-k 稀疏度 (每状态保留的 1 的数量, 默认 32)。
        - dg_size_factor: DG 扩容倍数 (dg_size = hidden_size * factor)。
          真正的模式分离需要先投射到高维空间再稀疏化，防止状态崩塌。
        - sfa_inc/sfa_decay: 神经元慢变量 (频率适应, v2.1, Opt-in 默认关),
          见 TorchLIFSimulator.__init__ 注释。
        - protect_mode/strength/stab_beta/stab_decay/freq_thr: ★ v2.4 突触
          保护 (ISI-CV 本地版, Opt-in 默认 off)。稳定性掩码保护已巩固突触,
          位置头在线更新时跳过/降速 → 防旧知识被覆盖 (灾难性遗忘)。
        """
        super().__init__(hidden_size, output_size, input_bias,
                         leak=leak, threshold=threshold,
                         reset_factor=reset_factor,
                         inhibition_strength=inhibition_strength,
                         num_layers=num_layers,
                         sfa_inc=sfa_inc, sfa_decay=sfa_decay,
                         prospective=prospective)

        # ★ v14: 三因子资格迹参数
        self.use_eligibility_trace = use_eligibility_trace
        self.eligibility_lambda = eligibility_lambda

        # ★ v3.6 上游正交字符编码: 让不同字符在 hidden 层即近似正交
        #   (去除 ASCII 位型重叠, 解决短词句间方向不可区分的根因)
        self.use_orthogonal_char = bool(use_orthogonal_char)

        # ★ v14: DG 稀疏分离参数
        self.use_dg_separation = use_dg_separation
        self.dg_k = dg_k
        self.dg_size = hidden_size * dg_size_factor
        self.feat_dim = self.dg_size if use_dg_separation else hidden_size
        
        # DG 随机稀疏投影 (EC -> DG) - 模拟海马齿状回的 Mossy Fiber 投影
        if use_dg_separation:
            self.W_ec2dg = torch.randn(self.dg_size, hidden_size, dtype=torch.float32, device=DEVICE) * 0.1
        else:
            self.W_ec2dg = None

        self.use_memory_thinking = bool(use_memory_thinking)
        self.thinking_max_steps = max(1, int(thinking_max_steps))
        self.thinking_stable_steps = max(1, int(thinking_stable_steps))
        self.thinking_delta_threshold = max(0.0, float(thinking_delta_threshold))
        self.thinking_feedback_strength = max(0.0, min(1.0, float(thinking_feedback_strength)))
        self.thinking_last_steps = 0
        self.thinking_last_delta = 0.0
        if self.feat_dim == hidden_size:
            self.W_mem2lif = None
        else:
            _thinking_gen = torch.Generator(device=DEVICE)
            _thinking_gen.manual_seed(20260814)
            self.W_mem2lif = torch.randn(
                hidden_size, self.feat_dim, dtype=torch.float32,
                device=DEVICE, generator=_thinking_gen) / (self.feat_dim ** 0.5)

        # ★ v2.4 突触保护 (ISI-CV 本地版, Opt-in 默认 off):
        #   稳定性掩码保护已巩固突触 → 位置头在线更新时跳过/降速
        #   protect_mode: off / sign (更新方向一致性) / freq (高发放列)
        #   / both; 掩码构造基于更新方向统计与发放频率 (慢变量, 同 SFA
        #   thr_shift 族), 不产生连续数值学习信号 (红线合规)。
        self.protect_mode = protect_mode          # off / sign / freq / both
        self.protect_strength = protect_strength  # α ∈ [0,1] 软保护强度
        self.stab_beta = stab_beta                # 稳定判定阈值 (|cum| > β)
        self.stab_decay = stab_decay              # 一致性累积衰减
        self.freq_thr = freq_thr                  # 高发放列保护阈值
        self.stab_cum = []                        # 每位置头同形状稳定性累积
        self._freq_count = torch.zeros(hidden_size, dtype=torch.float32,
                                       device=DEVICE)
        self._freq_seen = 0                       # 累计编码字符数 (频率统计)

        # 上下文→首字符预测: 8×256, 奖赏调制 Hebbian 训练
        self.W_ctx_to_first = torch.empty(output_size, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_ctx_to_first.uniform_(-0.1, 0.1)

        # ★ v13 位置记忆头: 上下文状态 → 回复第 k 字符 (8×256 × max_pos)
        #   对每个回复位置 k 一个独立 Hebbian 分类器, 与 W_ctx_to_first 同机制。
        #   用于修正 W_seq 循环生成的"非首字"输出 (experiment14: 字符级
        #   5.4% → 77.4%)。延迟初始化: 首用前检查容器是否为空 (保持向后兼容)。
        self.W_ctx_to_pos = []   # 每位置: (8, feat_dim)
        self.b_ctx_to_pos = []   # 每位置: (8,)

        # ============================================================
        # ★ v3.2 语义分槽专家 (Goal-Gated Experts) — 动态自动扩增
        # 问题: 固定 max_pos 个位置头被所有对话共享, 开放域 2000 对时
        # 特征冲突 → 位置头互相抵消 → 乱码 (容量不足)。
        # 方案: 按输入语义动态分配"专家", 每个专家拥有独立的位置头组。
        #   - 路由: 门向量 G_e · goal 的突触整合 (线性加权, 与 W·feat
        #     同族) + WTA 竞争 (侧抑制, 红线允许) → 选出发放最强的专家
        #   - 生长: 所有专家门激活 < expert_thr (新颖主题) → 分配新专家,
        #     新专家绑定该主题 DG 特征 (神经发生式, 非检索/模板匹配)
        #   - 记忆: 各专家位置头组独立 Hebbian 训练, 组间零冲突
        # use_experts=False 时完全走旧路径 (向后兼容)。
        # ============================================================
        self.use_experts = use_experts
        self.expert_thr = expert_thr          # 路由阈值 (门激活需超过才命中)
        self.expert_heads_cpu = bool(expert_heads_cpu)
        self._expert_active_idx = -1
        self.expert_gates = []                # 每专家门向量 (feat_dim, 二值稀疏 DG 特征)
        self.expert_first = []                # 每专家首字符头 (8, feat_dim)
        self.expert_first_b = []              # 每专家首字符偏置 (8,)
        self.expert_pos = []                  # 每专家: [位置头 (8, feat_dim), ...]
        self.expert_pos_b = []                # 每专家: [位置偏置 (8,), ...]

        # 序列预测: 8×256, 二值输出 → 下一字符编码
        self.W_seq = torch.empty(output_size, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_seq.uniform_(-0.1, 0.1)
        self.b_seq = torch.empty(output_size, dtype=torch.float32, device=DEVICE)
        self.b_seq.uniform_(-0.1, 0.1)

        # ★ v14.12 深度读出口 (方案 C): W_seq 前加可学习隐藏层
        #   隐藏层用预测编码信号训练 (Rao & Ballard 1999; Whittington &
        #   Bogacz 2017): 预测"下一步输入特征", 输出层从预测状态读字符。
        #   红线合规: 全部为 RPE 调制 Hebbian (Δw = lr×RPE×pre), 无 BP。
        self.W_seq_h = torch.empty(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_seq_h.uniform_(-0.1, 0.1)
        self.seq_h_thr = 0.2
        # 深度读出口输出层: 隐藏层状态 → 8-bit 字符
        self.W_seq_out = torch.empty(output_size, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_seq_out.uniform_(-0.1, 0.1)
        # 独立"结束"读出器: 状态 → 是否该终止 (与字符读出分离, 避免竞争)
        self.W_done = torch.zeros(1, self.feat_dim, dtype=torch.float32, device=DEVICE)
        # ★ v3.6 非对称链式状态流 (MPN Liu 2019 / 海马相位进动 O'Keefe &
        #   Recce 1993): 编码"上一步状态 → 当前状态"的顺序转移。
        #   非对称 (STDP pre→post 有向), 初始为 0, 由 STDP 塑造顺序敏感
        #   连接, 替代对称 max 并集 (13.26 序列身份丢失根因)。
        self.W_chain = torch.zeros(self.feat_dim, self.feat_dim,
                                   dtype=torch.float32, device=DEVICE)
        
        # 修正 TorchLIFSimulator 中 W_prosp 的形状
        self.W_prosp = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_prosp.uniform_(-0.1, 0.1)
        self.W_dmd_input = torch.empty(output_size, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_dmd_input.uniform_(-0.1, 0.1)
        self.b_dmd_input = torch.zeros(output_size, dtype=torch.float32, device=DEVICE)

        # ============================================================
        # ★ 共激活矩阵 (Associative Memory, v10 / v3.0 三级架构)
        # W_coact_temp: 临时记忆，单轮内追踪短期思考，新轮清空。
        # W_coact_long: 长期记忆，跨轮累积。
        # ============================================================
        self.W_coact_temp = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_coact_long = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.enable_long_memory_expansion = bool(enable_long_memory_expansion)
        self.long_memory_occupancy_threshold = min(1.0, max(0.0, float(long_memory_occupancy_threshold)))
        self.long_memory_write_gain_threshold = min(1.0, max(0.0, float(long_memory_write_gain_threshold)))
        self.long_memory_conflict_threshold = min(1.0, max(0.0, float(long_memory_conflict_threshold)))
        self.long_memory_pressure_patience = max(1, int(long_memory_pressure_patience))
        self.long_memory_pressure_streak = 0
        # ★ v3.3 P0b 突触标记与捕获 (STC, Frey & Morris 1997; Redondo &
        #   Morris 2011) — 临时记忆只在新颖/显著(捕获)且局部共激活足够强
        #   (标记)时选择性写入长期记忆, 拒绝冗余重放稀释长期块。
        self.enable_synaptic_capture = bool(enable_synaptic_capture)
        self.capture_salience_threshold = min(1.0, max(0.0, float(capture_salience_threshold)))
        self.capture_tag_threshold = min(1.0, max(0.0, float(capture_tag_threshold)))
        self.W_coact_blocks = [self.W_coact_long]
        self.long_memory_block_writes = [0]
        self.long_memory_active_block = 0
        self.coact_lr = 0.1  # 共激活学习率
        self.coact_decay = 0.99  # 共激活衰减率 (遗忘旧关联)
        # ★ v3.3 延迟衰减标量: W_coact_temp 以"缩放表示"存储 (实际值 = 存储×s),
        #   衰减只更新 s, 每 ~230 字符才做一次全矩阵重缩放 — 与逐字符
        #   mul_(decay) 数学完全等价, 但把每字符的 O(N²) 衰减降为 O(1)。
        self._coact_s = 1.0
        self.MemWork = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.mem_forget_ratio = 0.3

        # ============================================================
        # ★ v3.1 两阶段记忆模式开关 (CLS 互补学习系统):
        #   _long_term_read_enabled:  长期记忆读出 (W_coact_long) 是否允许
        #   _long_term_write_enabled: 短期→长期巩固 (consolidate) 是否允许
        #   _goal_guidance_enabled:   目标意图注入 (自上而下引导) 是否允许
        # 默认全开 = 旧行为 (向后兼容); 两阶段训练由 set_memory_mode 关闭。
        # ============================================================
        self._long_term_read_enabled = True
        self._long_term_write_enabled = True
        self._goal_guidance_enabled = True

        # ============================================================
        # ★ v14.12 循环思考储层 (Liquid State Machine, Maass 2002)
        # 生成时给 W_seq 提供"持续思考"的循环状态空间, 打破 1 阶马尔可夫。
        # 固定随机稀疏循环连接 (Mossy fiber 式) + LIF 0-1 膜电位动力学,
        # 不训练 (无学习信号), 只提供状态变换 → 红线合规。
        # reservoir:   循环状态 (跨字符持续, 序列开始清零)
        # W_res:       储层内循环连接 (随机稀疏, 固定)
        # W_res_in:    输入特征 → 储层 (固定)
        # W_res_out:   储层发放 → 思考特征 (固定投影, 融合进生成状态)
        # goal_strength: 目标意图注入强度 (自上而下信号, 方案 A)
        # ============================================================
        self.use_reservoir = use_reservoir
        self.res_dim = res_dim
        self.res_leak = res_leak
        self.res_thr = res_thr
        self.goal_strength = goal_strength
        self.reservoir = torch.zeros(res_dim, dtype=torch.float32, device=DEVICE)
        if use_reservoir:
            self.W_res = torch.randn(res_dim, res_dim, dtype=torch.float32, device=DEVICE) * 0.08
            sp_mask = (torch.rand(res_dim, res_dim, device=DEVICE) > 0.95).float()
            self.W_res *= sp_mask
            self.W_res_in = torch.randn(res_dim, self.feat_dim, dtype=torch.float32, device=DEVICE) * 0.1
            self.W_res_out = torch.randn(self.feat_dim, res_dim, dtype=torch.float32, device=DEVICE) * 0.1
        else:
            self.W_res = None
            self.W_res_in = None
            self.W_res_out = None
        self._last_goal = None
        self._last_pos_goal = None
        self._last_think = None
        self._target_cache = {}
        self._char_bias_codebook = {}

        # ★ 海马 barcode 式稀疏句索引 (Fang 2024): 给不同输入分配正交
        #   稀疏向量, 解决短词输入方向表征区分度不足 (13.32 根因)。
        #   默认关闭, 向后兼容; 启用后 input_end 时注入 committed 方向。
        self.use_input_barcode = False
        self._barcode_cache = {}
        self._barcode_next_start = 0
        self._last_input_text = None

        # ★ v3.4 DMD 动态意义方向 — D0 状态容器 (只记录, 不改变正式生成)
        #   semantic_state:    逐字累积的持续理解 (漏积分, 非输入快照)
        #   response_direction: 回答意义方向 (残差门控漏积分, 输入结束时定型)
        #   prediction_residual: 当前输入 vs 平滑预期轨迹的"惊讶/新颖性"向量
        self.dmd_semantic = None
        self.dmd_direction = None
        self.dmd_freq = None
        self.dmd_residual = None
        self.dmd_residual_norm = 0.0
        self.dmd_trace = []
        self.dmd_record_trace = False
        self.dmd_input_end = False
        self.dmd_committed_direction = None
        self.use_dmd_prospective = False
        self.use_dmd_input_prediction = False
        self.use_dmd_selective_consolidation = False
        self.dmd_consolidate_threshold = 0.05
        self.dmd_residual_sum = 0.0
        self.dmd_step_count = 0
        self.dmd_input_residual_sum = 0.0
        self.dmd_input_step_count = 0
        self.dmd_input_trace = []
        self._dmd_input_states = []
        self._dmd_prev_input_state = None
        self.dmd_sem_decay = 0.9      # semantic 漏积分衰减
        self.dmd_dir_decay = 0.95     # direction 基础保持率 (低残差→稳定)
        self.dmd_res_gain = 2.0       # 残差对方向更新速率的增益 (高残差→快修正)

        # ★ v3.3 P0a 时序合取上下文 (TCM/VSA 绑定, Howard & Kahana 2002;
        #   Plate 1995) — 把当前 DG 特征与已发生顺序绑定为单一稀疏码,
        #   替代 input_max/input_sum 的"字符袋"摘要, 使不同输入可分。
        #   _conj_perm 是固定随机置换 (只存索引, 顺序敏感); 绑定 = XOR +
        #   top-k (均为离散算子, 无连续学习信号, 红线内)。
        # ★ 独立 RNG: _conj_perm 用固定种子单独生成, 不消耗主 RNG 状态,
        #   保证"是否开启 P0"不改变网络初始化 → 与对照组严格同初始化对照。
        self.use_conjunctive_context = bool(use_conjunctive_context)
        self.conj_k = self.dg_k if conj_k is None else max(1, int(conj_k))
        self.use_hybrid_pos_context = bool(use_hybrid_pos_context)
        self.pos_context_alpha = max(0.0, float(pos_context_alpha))
        self._conj_ctx = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        if self.use_conjunctive_context:
            _gen = torch.Generator(device=DEVICE)
            _gen.manual_seed(20260812)
            self._conj_perm = torch.randperm(
                self.feat_dim, device=DEVICE, generator=_gen)
        else:
            self._conj_perm = None

    @property
    def W_coact(self):
        """兼容旧版接口: 联合关联记忆 (临时 + 长期)"""
        if not hasattr(self, "W_coact_temp") or self.W_coact_temp.shape[0] != self.feat_dim:
            self.W_coact_temp = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        if not hasattr(self, "W_coact_long") or self.W_coact_long.shape[0] != self.feat_dim:
            # 从旧字典中恢复真实的张量数据
            old_tensor = self.__dict__.get("W_coact_long")
            if old_tensor is None:
                old_tensor = self.__dict__.get("W_coact")
            if old_tensor is not None:
                if old_tensor.shape[0] == self.feat_dim:
                    self.W_coact_long = old_tensor
                else:
                    new_long = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
                    s0, s1 = old_tensor.shape
                    new_long[:s0, :s1] = old_tensor
                    self.W_coact_long = new_long
                if "W_coact" in self.__dict__:
                    del self.__dict__["W_coact"]
            else:
                self.W_coact_long = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self._ensure_long_memory_blocks()
        combined = torch.zeros_like(self.W_coact_temp)
        for block in self.W_coact_blocks:
            combined.add_(block)
        s = getattr(self, "_coact_s", 1.0)
        return torch.clamp(combined + self.W_coact_temp * s, 0.0, 1.0)

    @W_coact.setter
    def W_coact(self, value):
        """兼容旧版快照恢复 (直接覆盖长期记忆)"""
        if not hasattr(self, "W_coact_temp") or self.W_coact_temp.shape[0] != self.feat_dim:
            self.W_coact_temp = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.W_coact_long = value
        self.W_coact_blocks = [value]
        self.long_memory_block_writes = [0]
        self.long_memory_active_block = 0
        self.W_coact_temp.zero_()
        self._coact_s = 1.0

    def _ensure_long_memory_blocks(self):
        if not hasattr(self, "W_coact_long"):
            legacy_tensor = self.__dict__.get("W_coact")
            if legacy_tensor is None:
                legacy_tensor = torch.zeros(
                    self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
            self.W_coact_long = legacy_tensor
            self.__dict__.pop("W_coact", None)
        if not hasattr(self, "W_coact_temp"):
            self.W_coact_temp = torch.zeros_like(self.W_coact_long)
        if not hasattr(self, "enable_long_memory_expansion"):
            self.enable_long_memory_expansion = False
        if not hasattr(self, "long_memory_occupancy_threshold"):
            self.long_memory_occupancy_threshold = 0.5
        if not hasattr(self, "long_memory_write_gain_threshold"):
            self.long_memory_write_gain_threshold = 0.05
        if not hasattr(self, "long_memory_conflict_threshold"):
            self.long_memory_conflict_threshold = 0.25
        if not hasattr(self, "long_memory_pressure_patience"):
            self.long_memory_pressure_patience = 3
        if not hasattr(self, "long_memory_pressure_streak"):
            self.long_memory_pressure_streak = 0
        if not hasattr(self, "_coact_s"):
            self._coact_s = 1.0
        # ★ v3.3 旧模型迁移: 新开关与状态缺失时补默认 (向后兼容)
        if not hasattr(self, "enable_synaptic_capture"):
            self.enable_synaptic_capture = False
        if not hasattr(self, "capture_salience_threshold"):
            self.capture_salience_threshold = 0.1
        if not hasattr(self, "capture_tag_threshold"):
            self.capture_tag_threshold = 0.05
        if not hasattr(self, "use_conjunctive_context"):
            self.use_conjunctive_context = False
        if not hasattr(self, "conj_k"):
            self.conj_k = 32
        if not hasattr(self, "use_hybrid_pos_context"):
            self.use_hybrid_pos_context = False
        if not hasattr(self, "pos_context_alpha"):
            self.pos_context_alpha = 0.5
        if not hasattr(self, "_last_pos_goal"):
            self._last_pos_goal = None
        if not hasattr(self, "_conj_ctx") or self._conj_ctx.shape[0] != self.feat_dim:
            self._conj_ctx = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        if not hasattr(self, "_conj_perm"):
            self._conj_perm = None
        if not hasattr(self, "W_coact_blocks") or not self.W_coact_blocks:
            self.W_coact_blocks = [self.W_coact_long]
        if not hasattr(self, "long_memory_block_writes"):
            self.long_memory_block_writes = [0] * len(self.W_coact_blocks)
        if len(self.long_memory_block_writes) < len(self.W_coact_blocks):
            self.long_memory_block_writes.extend(
                [0] * (len(self.W_coact_blocks) - len(self.long_memory_block_writes)))
        if not hasattr(self, "long_memory_active_block"):
            self.long_memory_active_block = len(self.W_coact_blocks) - 1
        self.long_memory_active_block = min(
            max(0, self.long_memory_active_block), len(self.W_coact_blocks) - 1)
        self.W_coact_long = self.W_coact_blocks[0]

    def _long_memory_pressure(self, incoming):
        self._ensure_long_memory_blocks()
        block = self.W_coact_blocks[self.long_memory_active_block]
        occupied = block > 0
        incoming_active = incoming > 0
        occupancy = occupied.float().mean().item()
        incoming_count = int(incoming_active.sum().item())
        if incoming_count == 0:
            write_gain = 0.0
            conflict = 0.0
        else:
            write_gain = ((~occupied) & incoming_active).sum().item() / incoming_count
            conflict = (occupied & incoming_active).sum().item() / incoming_count
        pressured = (
            incoming_count > 0
            and occupancy >= self.long_memory_occupancy_threshold
            and write_gain <= self.long_memory_write_gain_threshold
            and conflict >= self.long_memory_conflict_threshold
        )
        self.long_memory_pressure_streak = self.long_memory_pressure_streak + 1 if pressured else 0
        return pressured and self.long_memory_pressure_streak >= self.long_memory_pressure_patience

    def _expand_long_memory(self):
        self._ensure_long_memory_blocks()
        self.W_coact_blocks.append(torch.zeros_like(self.W_coact_temp))
        self.long_memory_block_writes.append(0)
        self.long_memory_active_block = len(self.W_coact_blocks) - 1
        self.long_memory_pressure_streak = 0

    # ==================== 关联记忆回忆 ====================

    def _sparse_mv(self, W, cue):
        """稀疏线索的精确矩阵乘法: 仅读取 cue 非零列再求和, 与 torch.mv 数学等价"""
        cols = cue.nonzero(as_tuple=False).squeeze(1)
        if cols.numel() == 0:
            return torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)
        return W.index_select(1, cols).sum(dim=1)

    def recall_from_memassoc(self, cue, sparse_hint=None):
        """★ 关联记忆回忆 — 给定线索模式，提取关联的神经元群体 (分级强度, v12.4)

        生物学依据: 海马体关联记忆 → 皮层回忆。除法归一化 (Divisive Normalization)
        由抑制性中间神经元 (如 Basket cells) 介导，随全网总活跃度动态缩放兴奋性，
        防止癫痫样饱和发放 (Carandini & Heeger, 2012)。

        ★ v14.6 大规模优化: 固定 scale=hidden_size/2 在大语料长序列下依然会饱和。
        采用基于局部最大的动态侧抑制: scale = max(raw) (下限为期望活跃度)。
        这保证了 recall 永远不会全面饱和 (clamp 到 1.0 的神经元过多)，
        从而维持了状态特征 (state) 在 DG 稀疏化前的连续分布形状。

        Args:
            cue: 线索模式 (feat_dim-dim 稀疏特征)
            sparse_hint: True = cue 为 DG top-k 二值稀疏向量 (仅热路径调用方
                显式传入, 用 _sparse_mv 精确等价替代全量 mv, 减少 O(N²) 读取)

        Returns:
            recall: 分级回忆强度 (feat_dim-dim, 每个元素 ∈ [0, 1])
        ★ v14.10 内存优化: 直接对 long/temp 分别做 mv 再相加, 避免
        W_coact property 物化 64MB 级联合矩阵 (clamp(long+temp))。
        ★ v3.3 缩放临时记忆: W_coact_temp 存储为 实际值×_coact_s (延迟衰减),
        读取时乘以 _coact_s 还原 — 与旧逐字符衰减完全等价。
        """
        def mv_eff(W, c):
            return self._sparse_mv(W, c) if sparse_hint else torch.mv(W, c)
        s = getattr(self, "_coact_s", 1.0)
        # 联合记忆 = clamp(long + temp, 0, 1), 拆分为多次 mv 省 64MB 临时矩阵
        # ★ v3.1: _long_term_read_enabled=False (阶段一) 时跳过长期读出,
        # recall 只由临时记忆 (W_coact_temp) 驱动。
        if getattr(self, "_long_term_read_enabled", True):
            self._ensure_long_memory_blocks()
            block_raw = [mv_eff(block, cue) for block in self.W_coact_blocks]
            if block_raw:
                strengths = torch.stack([response.max() for response in block_raw])
                raw = block_raw[int(strengths.argmax().item())]
            else:
                raw = torch.zeros_like(cue)
            raw = raw + mv_eff(self.W_coact_temp, cue) * s
        else:
            raw = mv_eff(self.W_coact_temp, cue) * s
        raw = torch.clamp(raw, 0.0, 1.0)
        
        # ★ v14.6: 引入 k-WTA (Winner-Take-All) 侧抑制，防止长序列下全面饱和
        # 即使 W_coact 因长序列过度密集，也只允许激活最强的前 25% 神经元
        k = self.feat_dim // 4
        if raw.numel() > 0:
            threshold_val = torch.kthvalue(raw, self.feat_dim - k).values
            # 抑制低于阈值的激活
            raw = torch.where(raw >= threshold_val, raw, torch.zeros_like(raw))
            
        scale = self.feat_dim / 2.0  # 期望最大共激活数 (~50% 活跃)
        recall = torch.clamp(raw / scale, 0.0, 1.0)
        return recall

    def reset_memory(self):
        """重置工作记忆层 (关联记忆 W_coact_temp 会在新轮清空，W_coact_long 保留)"""
        self.MemWork = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        if hasattr(self, "_coact_trace"):
            if self._coact_trace.shape[0] != self.feat_dim:
                self._coact_trace = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
            else:
                self._coact_trace.zero_()
        if not hasattr(self, "W_coact_temp") or self.W_coact_temp.shape[0] != self.feat_dim:
            self.W_coact_temp = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        else:
            self.W_coact_temp.zero_()
        self._coact_s = 1.0
        if not hasattr(self, "W_coact_long") or self.W_coact_long.shape[0] != self.feat_dim:
            _ = self.W_coact
        # ★ v3.3 P0a: 每个新序列从头重建合取上下文
        if hasattr(self, "_conj_ctx"):
            if self._conj_ctx.shape[0] != self.feat_dim:
                self._conj_ctx = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
            else:
                self._conj_ctx.zero_()

    # ==================== 二值阈值神经元前向计算 ====================

    def _binary_forward_char(self, input_vec):
        """字符二值神经元前向计算 — 返回二值 {0, 1}"""
        return self._binary_forward(input_vec)

    def update_coactivation(self, output):
        """★ 更新共激活矩阵 — 关联学习 (Associative Learning)

        ★ v14.10 内存优化: 使用原地 mul_/addr_ 更新, 避免外积临时张量
        (feat_dim=4096 时每字符节约 64MB 分配, 防 GPU OOM)。
        """
        if not hasattr(self, "W_coact_temp"):
            _ = self.W_coact  # 触发 property 初始化

        # ★ v14.9 序列化关联: 引入不对称更新 (STDP) 替代纯词袋
        # 为了让 W_coact 编码时序转移 (上下文)，不仅关联 output 与 output，
        # 还要关联 output 与历史轨迹。我们用 self._coact_trace 模拟短时历史。
        # 这样 W_coact 就包含了 n-gram 转移统计，打破了长句子的状态对称性。
        if not hasattr(self, "_coact_trace"):
            self._coact_trace = torch.zeros_like(output)
        
        self._coact_trace = self._coact_trace * 0.5 + output
        # ★ v3.3 延迟衰减: 只更新标量 s (O(1)), 全矩阵 mul_(decay) 推迟到
        # s < 0.1 时一次性结算 — 与旧逐字符衰减数学完全等价。
        self._coact_s = getattr(self, "_coact_s", 1.0) * self.coact_decay
        # ★ v3.3 稀疏行写入: output 为二值稀疏向量, 只在非零行执行
        #   W[rows] += (lr/s)·trace 并 clamp, 零行无变化 — 与 addr_ 等价。
        rows = output.nonzero(as_tuple=False).squeeze(1)
        if rows.numel() > 0:
            s = self._coact_s
            row_slice = self.W_coact_temp.index_select(0, rows)
            row_slice.add_(self._coact_trace, alpha=self.coact_lr / s)
            row_slice.clamp_(0.0, 1.0 / s)
            self.W_coact_temp.index_copy_(0, rows, row_slice)
        if self._coact_s < 0.1:
            self.W_coact_temp.mul_(self._coact_s)
            self._coact_s = 1.0

    def consolidate_coactivation(self):
        """★ 临时记忆 -> 长期记忆巩固 (v3.0)

        ★ v3.1: _long_term_write_enabled=False (阶段一) 时巩固被屏蔽,
        临时记忆轨迹不会写入长期记忆 — 满足"训练时屏蔽长期记忆禁止记录"。
        """
        if not getattr(self, "_long_term_write_enabled", True):
            return
        if not hasattr(self, "W_coact_temp"):
            _ = self.W_coact
        self._ensure_long_memory_blocks()
        s = getattr(self, "_coact_s", 1.0)
        incoming = torch.clamp(self.W_coact_temp * (0.5 * s), 0.0, 1.0)
        if not torch.any(incoming > 0):
            self.long_memory_pressure_streak = 0
            self.W_coact_temp.zero_()
            self._coact_s = 1.0
            return
        # ★ v3.3 P0b 突触标记与捕获 (STC):
        #   - 捕获(全局门) = 相对活动长期块的新颖率; 低于阈值视为冗余重放
        #     → 丢弃, 不写入 (防冗余/冲突痕迹稀释长期记忆)。
        #   - 标记(逐突触) = incoming 超过 capture_tag_threshold 的强共激活
        #     才被捕获; 弱共激活视为噪声, 不写入。
        # ★ v3.4 DMD 选择性巩固: 输入期间的平均预测残差是"新颖性"信号 —
        #   残差低 (可预测/冗余) 的对话不写长期记忆, 只强化快系统;
        #   残差高 (新颖/惊讶) 才允许巩固进慢系统 (CLS 快速/慢速分工)。
        if getattr(self, "use_dmd_selective_consolidation", False):
            if self.dmd_step_count > 0:
                if self.dmd_input_step_count > 0:
                    mean_res = (self.dmd_input_residual_sum /
                                self.dmd_input_step_count)
                else:
                    mean_res = self.dmd_residual_sum / self.dmd_step_count
                if mean_res < getattr(self, "dmd_consolidate_threshold", 0.05):
                    self.long_memory_pressure_streak = 0
                    self.W_coact_temp.zero_()
                    self._coact_s = 1.0
                    return
        if self.enable_synaptic_capture:
            block = self.W_coact_blocks[self.long_memory_active_block]
            incoming_active = incoming > 0
            denom = int(incoming_active.sum().item())
            if denom > 0:
                novel = int((incoming_active & (block <= 0.0)).sum().item())
                if (novel / denom) < self.capture_salience_threshold:
                    self.long_memory_pressure_streak = 0
                    self.W_coact_temp.zero_()
                    self._coact_s = 1.0
                    return
            incoming = torch.where(
                incoming > self.capture_tag_threshold,
                incoming, torch.zeros_like(incoming))
            if not torch.any(incoming > 0):
                self.long_memory_pressure_streak = 0
                self.W_coact_temp.zero_()
                self._coact_s = 1.0
                return
        if self.enable_long_memory_expansion and self._long_memory_pressure(incoming):
            self._expand_long_memory()
        active = self.long_memory_active_block
        self.W_coact_blocks[active] = torch.clamp(
            self.W_coact_blocks[active] + incoming, 0.0, 1.0)
        self.long_memory_block_writes[active] += 1
        self.W_coact_long = self.W_coact_blocks[0]
        self.W_coact_temp.zero_()
        self._coact_s = 1.0

    def clear_long_term_memory(self):
        """★ v3.0 彻底清空长期记忆"""
        if not hasattr(self, "W_coact_temp"):
            _ = self.W_coact
        self._ensure_long_memory_blocks()
        self.W_coact_blocks = [torch.zeros_like(self.W_coact_temp)]
        self.long_memory_block_writes = [0]
        self.long_memory_active_block = 0
        self.long_memory_pressure_streak = 0
        self.W_coact_long = self.W_coact_blocks[0]
        self.W_coact_temp.zero_()
        self._coact_s = 1.0

    def clear_temporary_memory(self):
        """★ v3.0 仅清空临时记忆 (W_coact_temp + 迹), 保留长期记忆"""
        if not hasattr(self, "W_coact_temp") or self.W_coact_temp.shape[0] != self.feat_dim:
            self.W_coact_temp = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float32, device=DEVICE)
        else:
            self.W_coact_temp.zero_()
        self._coact_s = 1.0
        if hasattr(self, "_coact_trace"):
            if self._coact_trace.shape[0] != self.feat_dim:
                self._coact_trace = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
            else:
                self._coact_trace.zero_()
        self.MemWork = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)

    # ==================== v3.4 DMD 动态意义方向 (D0: 只记录) ====================

    def _dmd_reset(self):
        """重置 DMD 状态容器 — 每个新输入序列开始时调用"""
        self.dmd_semantic = torch.zeros(
            self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.dmd_direction = torch.zeros(
            self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.dmd_freq = torch.zeros(
            self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.dmd_residual = torch.zeros(
            self.feat_dim, dtype=torch.float32, device=DEVICE)
        self.dmd_residual_norm = 0.0
        self.dmd_trace = []
        self.dmd_input_end = False
        self.dmd_committed_direction = None
        self._dmd_predicted_next = None
        self.dmd_residual_sum = 0.0
        self.dmd_step_count = 0
        self.dmd_input_residual_sum = 0.0
        self.dmd_input_step_count = 0
        self.dmd_input_trace = []
        self._dmd_input_states = []
        self._dmd_input_encodings = []
        self._dmd_prev_input_state = None

    def _dmd_step(self, semantic_state):
        """逐字更新三类 DMD 状态 (纯局部运算, 不改变生成路径)

        - prediction_residual: 真正的预测误差 — 若 W_prosp 前瞻预测可用,
          残差 = |实际状态 − 上一步的前瞻预测| (预测加工); 否则回退
          泄漏代理 (当前状态与语义漏积分的差异)。
        - semantic_state: 漏积分累积的持续理解
        - response_direction: 主题方向 = 频次 top-k 索引 (主题稳定)
          × 泄漏积分权重 (顺序敏感)。反复出现的主题特征入选方向集合,
          但每个特征的强度保留其在语义轨迹中的相对位置 → 同义接近、
          重排敏感, 避免字符袋退化。
        """
        if (getattr(self, "use_dmd_prospective", False)
                and self._dmd_predicted_next is not None):
            residual = (semantic_state - self._dmd_predicted_next).abs().clamp(0.0, 1.0)
        else:
            residual = (semantic_state - self.dmd_semantic).abs().clamp(0.0, 1.0)
        res_norm = residual.mean().item()
        self.dmd_residual = residual
        self.dmd_residual_norm = res_norm
        self.dmd_residual_sum += res_norm
        self.dmd_step_count += 1

        self.dmd_semantic = (self.dmd_sem_decay * self.dmd_semantic
                             + (1.0 - self.dmd_sem_decay) * semantic_state)

        # 前瞻预测: 用 W_prosp 预测"自身下一步状态" (Brea 2016) —
        # 下个字符到达时与 _dmd_predicted_next 比较得真正预测误差。
        self._dmd_predicted_next = self._prospective_raw(
            self.dmd_semantic).detach().clone()

        self.dmd_freq = self.dmd_freq + semantic_state
        k = max(getattr(self, "dg_k", 8), self.feat_dim // 16)
        flat_freq = self.dmd_freq.flatten()
        if flat_freq.numel() > k:
            _, idx = flat_freq.topk(k)
            direction = torch.zeros_like(flat_freq)
            direction[idx] = self.dmd_semantic.flatten()[idx]
            imax = direction.max()
            self.dmd_direction = (direction / imax).reshape_as(self.dmd_freq)
        else:
            imax = self.dmd_semantic.max()
            self.dmd_direction = (self.dmd_semantic / imax
                                  if imax > 0 else self.dmd_semantic.clone())

        if self.dmd_record_trace:
            self.dmd_trace.append({
                "residual_norm": res_norm,
                "semantic": self.dmd_semantic.clone(),
                "direction": self.dmd_direction.clone(),
                "predicted_next": (self._dmd_predicted_next.clone()
                                   if self._dmd_predicted_next is not None else None),
            })

    def _dmd_input_predict(self, semantic_state):
        """输入侧 next-char 预测 (只读二值输出)"""
        feat = self._mem_feature(semantic_state)
        return self._binary_decode(self.W_dmd_input, feat, self.b_dmd_input)

    def train_dmd_input_predictor(self, texts, lr=0.1, n_iter=5):
        """只用输入文本自身训练 next-char 预测头, 不读取回复或位置。"""
        samples = []
        for text in texts:
            chars = [c for c in text if 0 <= ord(c) <= 255]
            if len(chars) < 2:
                continue
            self.reset_state()
            self.reset_memory()
            prev_state = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
            for i in range(len(chars) - 1):
                vec = self._char_to_8bit_bias(chars[i])
                self._multi_layer_forward(vec, n_loops=1)
                v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                current = self._dg_separate(v_curr)
                state = torch.max(current, self.recall_from_memassoc(current, sparse_hint=True))
                samples.append((state.clone(), self._target_bits(ord(chars[i + 1]))))
                prev_state = state
        for _ in range(max(1, int(n_iter))):
            for feat, target in samples:
                pred = self._dmd_input_predict(feat)
                rpe = target - pred
                self.W_dmd_input.addr_(rpe, self._mem_feature(feat), alpha=lr)
            self.W_dmd_input.clamp_(-10.0, 10.0)
        return self.dmd_input_accuracy(texts)

    def dmd_input_accuracy(self, texts):
        """评估输入侧 next-char 预测的逐位准确率"""
        total = correct = 0
        for text in texts:
            chars = [c for c in text if 0 <= ord(c) <= 255]
            if len(chars) < 2:
                continue
            self.reset_state()
            self.reset_memory()
            state = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
            for i in range(len(chars) - 1):
                self._multi_layer_forward(self._char_to_8bit_bias(chars[i]), n_loops=1)
                v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                state = torch.max(self._dg_separate(v_curr), state)
                pred = self._dmd_input_predict(state)
                target = self._target_bits(ord(chars[i + 1]))
                correct += int((pred == target).sum().item())
                total += self.output_size
        return correct / total if total else 0.0

    def _get_barcode(self, text):
        """海马 barcode 式稀疏句索引: 每个不同输入分配正交稀疏向量。

        用计数器顺序分配不重叠的位置段, 保证不同输入的 barcode 完全
        正交 (稀疏位置零重叠), 直到 feat_dim 空间耗尽后回绕。
        同一文本始终返回同一 barcode (缓存)。
        """
        b = self._barcode_cache.get(text)
        if b is not None:
            return b
        k = max(8, self.feat_dim // 16)
        start = self._barcode_next_start
        end = min(start + k, self.feat_dim)
        indices = list(range(start, end))
        if len(indices) < k:
            indices = list(range(k))
        barcode = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        barcode[indices] = 1.0
        self._barcode_cache[text] = barcode
        self._barcode_next_start = (end % self.feat_dim) if end >= self.feat_dim else end
        return barcode

    def input_end(self):
        """★ v3.4 DMD 输入结束事件 — 显式标记输入边界, 定型回答方向。"""
        self.dmd_input_end = True
        if self.dmd_direction is None:
            self._dmd_reset()
        self.dmd_committed_direction = self.dmd_direction.clone()
        context = getattr(self, "_last_goal", None)
        if context is None:
            context = getattr(self, "MemWork", None)
        self.dmd_committed_context = (
            context.detach().clone() if context is not None else
            self.dmd_committed_direction.clone())
        # 海马 barcode 注入: 正交稀疏句索引增强方向/上下文区分度
        if getattr(self, "use_input_barcode", False) and self._last_input_text is not None:
            barcode = self._get_barcode(self._last_input_text)
            self.dmd_committed_direction = torch.max(
                self.dmd_committed_direction, barcode)
            self.dmd_committed_context = torch.max(
                self.dmd_committed_context, barcode)
        return self.dmd_committed_direction.clone()

    def get_dmd_states(self):
        """D0 观测接口: 返回当前三类状态与残差标量的只读快照"""
        return {
            "semantic": self.dmd_semantic.clone() if self.dmd_semantic is not None else None,
            "direction": self.dmd_direction.clone() if self.dmd_direction is not None else None,
            "residual": self.dmd_residual.clone() if self.dmd_residual is not None else None,
            "residual_norm": self.dmd_residual_norm,
            "input_residual_mean": (self.dmd_input_residual_sum /
                                    self.dmd_input_step_count
                                    if self.dmd_input_step_count else 0.0),
            "input_step_count": self.dmd_input_step_count,
            "input_end": self.dmd_input_end,
            "committed_direction": (self.dmd_committed_direction.clone()
                                    if self.dmd_committed_direction is not None else None),
            "trace_len": len(self.dmd_trace),
        }

    def set_memory_mode(self, *, long_term_read=None, long_term_write=None,
                        goal_guidance=None):
        """★ v3.1 两阶段记忆训练模式开关 (CLS 互补学习系统)

        阶段一 (短时续写): long_term_read=False, long_term_write=False,
          goal_guidance=False → 长期记忆读写被屏蔽, 只允许 W_coact_temp /
          _coact_trace / 循环储层驱动的"短时文本续写", 无答案引导。
        阶段二 (长期巩固): 全部 True → 开放长期记忆读写, 通过逐对话
          巩固 (consolidate) 与目标引导学习泛化能力。

        Args:
            long_term_read: W_coact_long 读出开关 (None = 不变)
            long_term_write: 短期→长期巩固写入开关 (None = 不变)
            goal_guidance: 目标意图注入开关 (None = 不变)
        """
        if long_term_read is not None:
            self._long_term_read_enabled = bool(long_term_read)
        if long_term_write is not None:
            self._long_term_write_enabled = bool(long_term_write)
        if goal_guidance is not None:
            self._goal_guidance_enabled = bool(goal_guidance)

    # ==================== 序列编码 ====================

    def encode_text_lif(self, text, update_memory=True, n_loops=1,
                        context_coord=None, round_number=0, record_dmd=False):
        """逐字符处理文本，多层前向 + 工作记忆/关联记忆累积上下文状态

        ★ v10 关键变化:
          - 输入经 4 层隐藏层前向传播，输出为最深层 (L4) 二值发放
          - 独立工作记忆层 MemWork: 0-1 分级电位累积 + 随机遗忘
          - 独立关联记忆层 W_coact: 共发放追踪 + 回忆提取
          - 状态更新: state = max(V_LN, recall, state × forget_mask)

        ★ v13.1: n_loops 自回归循环 (每次输入后循环一次再传入输入),
          透传给 _multi_layer_forward; 默认 1 与旧行为一致。

        ★ v14.5: context_coord 路径积分注入 — 可选上下文坐标,
          将对话轮数信息叠加到最终状态, 使网络感知"第几轮对话"。

        ★ v3.4 D0: record_dmd=True 时逐字更新 DMD 动态意义方向状态
          (semantic / direction / residual), 只记录不改变生成。

        Args:
            text: 输入文本
            update_memory: 是否更新关联记忆 W_coact (学习/写入)。
            n_loops: 每次输入自回归前向轮数 (v13.1, 默认 1)
            context_coord: ContextCoord 实例 (可选, 多轮对话需要)
            round_number: 当前对话轮数 (默认 0)
            record_dmd: 是否记录 DMD 逐字状态 (v3.4 D0, 默认 False)

        Returns:
            outputs_list: 每个字符的最深层二值输出列表 (用于 W_seq 训练)
            state: 工作记忆累积状态 (hidden_size-dim, 每个元素 ∈ [0, 1])
        """
        chars = [c for c in text if 0 <= ord(c) <= 255]
        if not chars:
            return [], torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)

        # 记录输入文本, 供 input_end 的海马 barcode 索引注入使用
        self._last_input_text = text

        # ★ v10: 重置所有层膜电位 + 工作记忆层
        self.reset_state()
        self.reset_memory()

        outputs_list = []
        state = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        input_max = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        # ★ v3.2: 专家模式频次累积 — max 丢失频次导致 goal 全是公共词特征
        # (所有输入共享空格/停用词 → 不可分); sum 保留主题词频次 → top-k
        # 抓取各输入特有的高频主题特征 (区分度来源)
        input_sum = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        # ★ v3.4 D0: 记录开关 + 逐字状态容器重置
        self.dmd_record_trace = bool(record_dmd)
        self._dmd_reset()
        for ch in chars:
            vec = self._char_to_8bit_bias(ch)
            if self.use_memory_thinking:
                output, sparse_feat, state = self._think_character(
                    vec, update_memory=update_memory, n_loops=n_loops)
            else:
                # ★ v10: 4 层前向传播 → 最深层 (L4) 输出 (★ v13.1: n_loops 循环)
                output = self._multi_layer_forward(vec, n_loops=n_loops)
                v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                sparse_feat = self._dg_separate(v_curr)
            outputs_list.append(output)

            # ★ v2.4: 发放频率统计 (freq 突触保护先验, Opt-in)
            if getattr(self, "protect_mode", "off") in ("freq", "both"):
                self._freq_count += output
                self._freq_seen += 1
            # ★ v3.3 P0a: 时序合取上下文 — 把当前特征与已发生顺序绑定,
            # 替代跨字符 max/sum 的"字符袋"摘要 (顺序敏感, 输入可分)
            if getattr(self, "use_conjunctive_context", False):
                self._conj_ctx = self._conj_bind(sparse_feat, self._conj_ctx)
            # ★ v14.12: 输入语义摘要 = 全输入 DG 特征的 max 并集 (主题向量)
            input_max = torch.max(input_max, sparse_feat)
            if getattr(self, "use_experts", False):
                input_sum += sparse_feat
                # ★ v3.2: 专家模式下输入编码同步驱动储层 — 储层对输入
                # 产生独特内部状态 (LSM 输入指纹), 供专家路由使用
                # (普通模式不驱动, 保持向后兼容)
                if getattr(self, "use_reservoir", False):
                    self._encode_reservoir(sparse_feat)

            # ★ v10: 关联记忆层更新 (共发放追踪) — 先学习(写)再回忆(读)
            if not self.use_memory_thinking:
                recall = self.recall_from_memassoc(sparse_feat, sparse_hint=True)

                # ★ State is the union of current sparse feature and recalled long-term context
                # (No max accumulation across sequence, sparse representation prevents collapse)
                state = torch.max(sparse_feat, recall)
                if update_memory:
                    self.update_coactivation(sparse_feat)
                self.MemWork = state
            # ★ v3.4: 输入侧 next-char 预测残差
            #   预测当前字符, 只使用前一时刻语义状态; 首字符无预测历史。
            if (self.dmd_record_trace and
                    getattr(self, "use_dmd_input_prediction", False)):
                if self._dmd_prev_input_state is not None:
                    pred_bits = self._dmd_input_predict(
                        self._dmd_prev_input_state)
                    actual_bits = self._target_bits(ord(ch))
                    input_residual = (actual_bits - pred_bits).abs()
                    self.dmd_input_residual_sum += input_residual.mean().item()
                    self.dmd_input_step_count += 1
                    self.dmd_residual = input_residual
                    self.dmd_residual_norm = input_residual.mean().item()
                    self.dmd_input_trace.append({
                        "char": ch,
                        "residual_norm": self.dmd_residual_norm,
                        "predicted": pred_bits.clone(),
                    })
                self._dmd_prev_input_state = state.clone()
            # ★ v3.4 D0: 逐字更新 DMD 动态意义方向状态 (仅记录模式, 不影响默认路径)
            if self.dmd_record_trace:
                self._dmd_step(state)
                if not hasattr(self, "_dmd_input_states"):
                    self._dmd_input_states = []
                self._dmd_input_states.append(output.clone())
                if not hasattr(self, "_dmd_input_encodings"):
                    self._dmd_input_encodings = []
                self._dmd_input_encodings.append(vec.clone())

        # ★ v14.5: 路径积分坐标注入 (可选)
        if context_coord is not None:
            state = context_coord.inject(state, round_number)

        # ★ v14.12: 目标意图 = 输入语义摘要 (归一化, 供记忆头/生成自上而下引导)
        if hasattr(self, "_last_goal"):
            if getattr(self, "use_conjunctive_context", False):
                goal = self._conj_ctx.clone()
                pos_goal = goal
                if (getattr(self, "use_hybrid_pos_context", False)
                        and input_sum.sum().item() > 0):
                    continuous = input_sum / input_sum.max().clamp_min(1.0)
                    pos_goal = torch.clamp(
                        goal + self.pos_context_alpha * continuous, 0.0, 1.0)
                self._last_pos_goal = pos_goal
            else:
                imax = input_max.max()
                goal = input_max / imax if imax > 0 else input_max.clone()
                # ★ v3.2: 专家模式下改用频次累积 top-k — max 并集含全部公共词
                # 特征 (低维饱和/不可分), sum 按频次保留各输入的主题核心特征
                if getattr(self, "use_experts", False) and input_sum.sum().item() > 0:
                    k = max(32, self.feat_dim // 8)
                    flat = input_sum.flatten()
                    if flat.numel() > k:
                        _, idx = flat.topk(k)
                        goal = torch.zeros_like(flat)
                        goal[idx] = 1.0
                        goal = goal.reshape_as(input_sum)
                self._last_pos_goal = goal.clone()
            self._last_goal = goal
            # ★ v3.2: 保存储层输入指纹 (专家路由/位置头输入)
            if getattr(self, "use_experts", False) and getattr(self, "use_reservoir", False):
                res_spike = (self.reservoir > self.res_thr).float()
                self._last_think = res_spike.detach().clone()

        return outputs_list, state

    def encode_text_lif_states(self, text, update_memory=True, n_loops=1):
        """逐字符处理文本，返回所有中间状态

        ★ v10: 4 层前向 + 工作记忆/关联记忆，状态为分级值
        ★ v12.3: update_memory=False 时冻结 W_coact (评估/生成一致性)
        ★ v13.1: n_loops 自回归循环, 透传给 _multi_layer_forward

        Returns:
            outputs_list: 每个字符的最深层二值输出列表
            states: 每个字符处理后的工作记忆累积状态列表 (0-1 分级)
            final_state: 最终累积状态 (0-1 分级)
        """
        chars = [c for c in text if 0 <= ord(c) <= 255]
        if not chars:
            return [], [], torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)

        # ★ v10: 重置所有层膜电位 + 工作记忆层
        self.reset_state()
        self.reset_memory()

        outputs_list = []
        states = []
        state = torch.zeros(self.feat_dim, dtype=torch.float32, device=DEVICE)
        
        for ch in chars:
            vec = self._char_to_8bit_bias(ch)
            output = self._multi_layer_forward(vec, n_loops=n_loops)
            outputs_list.append(output)

            # ★ DG Separate the V to get sparse high-dim feature BEFORE associative memory
            v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
            sparse_feat = self._dg_separate(v_curr)

            # ★ v10: 关联记忆层更新 (共发放追踪) — 先学习(写)再回忆(读)
            # ★ v12.3: update_memory=False 时冻结 W_coact (不写入)
            if update_memory:
                self.update_coactivation(sparse_feat)
            recall = self.recall_from_memassoc(sparse_feat, sparse_hint=True)

            state = torch.max(sparse_feat, recall)
            states.append(state)
            self.MemWork = state

        return outputs_list, states, state

    # ==================== W_ctx_to_first 训练 ====================

    def train_context_to_first(self, dialogues, lr=0.05, n_iter=500, n_loops=1,
                               expected_gate=0.0):
        """训练 W_ctx_to_first — 二值累积状态 → 首字符 8-bit 编码

        使用奖赏预测误差调制 Hebbian 规则:
          Δw = lr × RPE_j × pre_state

        - RPE_j = target_j − pred_j ∈ {−1, 0, +1} — 奖赏预测误差
          +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
        - pre_state = 二值 {0, 1} 累积上下文状态
        - ★ 解码使用纯二值阈值: pred = (W_ctx_to_first·acc_state > 0).float(), 无 sigmoid
        - ★ 无 center = clamp(pred_raw, -1, 1): 连续数值运算, 已移除
        - 无偏置更新 (连续数值运算)

        ★ v13.1: n_loops 自回归循环 (透传给 encode_text_lif), 默认 1。
        ★ v2.2: expected_gate 预期门控 (预测性 Hebbian, 默认 0=关) —
          启用时只有"被预期活跃"的突触前参与强化 (Saponati 2023)。
        """
        # ★ v3.0/v14.12 确定性状态: 每对话从清空的临时记忆开始编码输入,
        #   记忆头特征 = 输入语义摘要 (max DG 特征并集, 含动词等语义特征)。
        #   → 库内复述 (同输入同摘要) 与未见泛化 (动词类别→回复模板) 统一支持。
        #   (旧方案: 回忆态 = 输入+回复 续接编码的时序迹, 只支持库内记忆)
        ctx_data = []
        self._coact_snapshots = None
        self._replay_states = [None] * len(dialogues)
        self._pos_feats = []
        for dialogue_idx, (inp, resp) in enumerate(dialogues):
            self.clear_temporary_memory()

            resp_codes = self._text_to_codes(resp)
            if not resp_codes:
                continue
            # 1) 编码输入 → 目标意图 feat = 输入语义摘要 (归一化 max DG 特征)
            _, acc_state = self.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
            if acc_state.sum().item() == 0:
                continue
            feat = self._last_goal.clone() if self._last_goal is not None else acc_state
            # ★ 仅记录真正进入训练数据的对话, 评估时按同序复用
            self._replay_states[dialogue_idx] = feat.detach().to("cpu", dtype=torch.float16)
            gate = self._expected_gate(acc_state, expected_gate) if expected_gate > 0 else None
            ctx_data.append((feat, resp_codes[0], gate))
            self._pos_feats.append((feat.detach().clone(), resp_codes))

        if not ctx_data:
            return

        # ★ v14: 资格迹模式需维护迹矩阵 (与 W 同形状)
        E = torch.zeros_like(self.W_ctx_to_first) if self.use_eligibility_trace else None

        has_gate = expected_gate > 0
        has_E = self.use_eligibility_trace
        N = len(ctx_data)

        feats_all = torch.stack([s[0] for s in ctx_data])
        tgts_all = torch.tensor(
            [[float((code >> j) & 1) for j in range(self.output_size)]
             for _, code, _ in ctx_data], dtype=torch.float32, device=DEVICE)
        gates_all = torch.stack([s[2] for s in ctx_data]) if has_gate else None

        for _ in range(n_iter):
            idx = torch.randperm(N, device=DEVICE).tolist()

            if not has_E and not has_gate:
                # SUPER FAST PATH
                for m in idx:
                    feat = feats_all[m]
                    tgt = tgts_all[m]
                    out = (torch.mv(self.W_ctx_to_first, feat) > 0).float()
                    rpe = tgt - out
                    self.W_ctx_to_first.addr_(rpe, feat, alpha=lr)
                    self.W_ctx_to_first.clamp_(-10.0, 10.0)
            else:
                # NORMAL PATH
                for m in idx:
                    feat = feats_all[m]
                    tgt = tgts_all[m]
                    gate = gates_all[m] if has_gate else None

                    pred = self._binary_decode(self.W_ctx_to_first, feat)
                    pred_bits = (pred > 0.5).float()
                    rpe = tgt - pred_bits
                    self._hebbian_step(self.W_ctx_to_first, rpe, feat, pred_bits, lr, E, gate=gate)

    # ==================== 位置记忆头训练 (v13) ====================

    def train_pos_heads(self, dialogues, max_pos=256, lr=0.05, n_iter=500, n_loops=1,
                        expected_gate=0.0):
        """训练位置记忆头 — 上下文状态 → 回复第 k 字符 (k = 0, 1, 2, ...)

        ★ 目标: "增加记忆层, 对非首字结果进行修正"。
          对回复的每个位置 k 训练一个独立 Hebbian 分类器 W_ctx_to_pos[k]
          (与 W_ctx_to_first 逐行同机制: 纯二值阈值解码, RPE 调制 Hebbian,
          无偏置更新, W clamp ±10)。

        ★ 状态获取与 train_context_to_first 完全一致 (v3.0):
          每对话前清空长期记忆 → 编码输入得到可复现状态,
          _replay_states 保存训练时的状态, 评估/生成时复用 → 训练/评估一致。

        ★ v13.1: n_loops 自回归循环 (透传给 encode_text_lif), 默认 1。
        ★ v2.2: expected_gate 预期门控 (预测性 Hebbian, 默认 0=关)。

        Args:
            dialogues: 对话对列表 [(inp, resp), ...]
            max_pos: 位置记忆头上限 (超出部分生成时回退 W_seq)
            lr: 学习率
            n_iter: 每位置训练迭代次数
            n_loops: 每次输入自回归前向轮数 (v13.1, 默认 1)
            expected_gate: 预期门阈值 (0 = 无门控, 向后兼容)
        """
        # 收集 (feat, 回复字符码列表) — 复用 train_context_to_first 的回忆态
        # ★ v3.0/v14.11: 用存储的 _pos_feats (情节绑定回忆态), 不再重新编码
        #   (重新编码会让状态依赖累积的长期记忆而漂移; 存储态与推理一致)
        data = []
        pos_feats = getattr(self, "_pos_feats", None)
        if pos_feats:
            for feat, resp_codes in pos_feats:
                if not resp_codes:
                    continue
                gate = self._expected_gate(feat, expected_gate) if expected_gate > 0 else None
                data.append((feat, resp_codes, gate))
        else:
            # 兼容旧路径: 重新编码 (每对话清空记忆, 无情节绑定)
            for i, (inp, resp) in enumerate(dialogues):
                resp_codes = self._text_to_codes(resp)
                if not resp_codes:
                    continue
                self.clear_long_term_memory()
                _, acc_state = self.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
                if acc_state.sum().item() == 0:
                    continue
                gate = self._expected_gate(acc_state, expected_gate) if expected_gate > 0 else None
                feat = self._mem_feature(acc_state)
                data.append((feat, resp_codes, gate))

        self.W_ctx_to_pos = []
        self.b_ctx_to_pos = []
        max_len = max((len(rc) for _, rc, _ in data), default=0)
        P = min(max_pos, max_len)
        if P == 0:
            return

        has_gate = expected_gate > 0
        has_E = self.use_eligibility_trace
        N = len(data)

        if not has_E and not has_gate:
            # ============ 并行路径 (v14.5): 位置维度 GPU 并行 ============
            # ★ 核心洞察: 不同位置头 W_ctx_to_pos[k] 完全独立 (各自独立的
            #   数据子集与权重), 可打包为 (P, 8, hidden) 3D 张量, 每步迭代
            #   同时更新所有位置头 → 内循环次数从 P×n_iter×N 降到 n_iter×N。
            # ★ 红线审查: 每个位置头内部仍是逐样本在线更新 (Δw=lr×RPE×feat,
            #   顺序不变), 只是把相互独立的位置头并行执行 — 等价于 8 个独立
            #   分类器并行训练, 非历史被回退的"批量更新" (组内 RPE 抵消)。
            # 特征与目标位型 (与串行路径同一数据源)
            feats_all = torch.stack([s[0] for s in data])          # (N, H)
            codes_arr = torch.tensor(
                [list(rc) + [0] * (max_len - len(rc)) for _, rc, _ in data],
                dtype=torch.long, device=DEVICE)
            tgt_bits = ((codes_arr.unsqueeze(-1) >> torch.arange(
                self.output_size, device=DEVICE)) & 1).float()    # (N, max_len, 8)
            # 每个位置的有效样本索引 (按原始对话序)
            # ★ max_n 是"每位置样本数上限" = N (非回复长度, 避免无效空转)
            max_n = N
            valid_idx = torch.zeros(P, max_n, dtype=torch.long, device=DEVICE)
            valid_mask = torch.zeros(P, max_n, dtype=torch.bool, device=DEVICE)
            for k in range(P):
                v = [i for i, (_, rc, _) in enumerate(data) if len(rc) > k]
                if v:
                    valid_idx[k, :len(v)] = torch.tensor(
                        v, dtype=torch.long, device=DEVICE)
                    valid_mask[k, :len(v)] = True

            W_all = torch.empty(P, self.output_size, self.feat_dim,
                                dtype=torch.float32, device=DEVICE)
            W_all.uniform_(-0.1, 0.1)
            b_all = torch.empty(P, self.output_size,
                                dtype=torch.float32, device=DEVICE)
            b_all.uniform_(-0.1, 0.1)

            for _ in range(n_iter):
                # 每位置独立随机打乱 (无效位 rnd=1 排最后, 保持语义一致)
                rnd = torch.rand(P, max_n, device=DEVICE)
                rnd.masked_fill_(~valid_mask, 1.0)
                _, lp = torch.sort(rnd, dim=1)              # (P, max_n)
                idx_m = torch.gather(valid_idx, 1, lp)      # (P, max_n) 全局样本索引
                vm = torch.gather(valid_mask, 1, lp)        # (P, max_n)
                k_idx = torch.arange(P, device=DEVICE)
                for m in range(max_n):
                    ids = idx_m[:, m]                       # (P,)
                    v = vm[:, m]                            # (P,)
                    fm = feats_all[ids]                     # (P, H)
                    # ★ 每个位置头 k 的目标必须是"自己位置"的字符 (tgt_bits[ids[k], k])
                    tg = tgt_bits[ids, k_idx]               # (P, 8)
                    out = (torch.bmm(W_all, fm.unsqueeze(-1)).squeeze(-1)
                           + b_all > 0).float()
                    rpe = (tg - out) * v.float().unsqueeze(1)
                    # 批量外积更新 (位置维度并行, 单头语义不变)
                    W_all = torch.baddbmm(W_all, rpe.unsqueeze(2),
                                          fm.unsqueeze(1), alpha=lr)
                    W_all.clamp_(-10.0, 10.0)

            self.W_ctx_to_pos = [W_all[k].clone() for k in range(P)]
            self.b_ctx_to_pos = [b_all[k].clone() for k in range(P)]
            return

        # ============ 串行路径 (启用资格迹/预期门时回退) ============
        for k in range(P):
            samples = [(feat, rc[k], gate) for feat, rc, gate in data if len(rc) > k]
            if not samples:
                break

            # ★ v14.5: 预先堆叠张量以利用连续内存和避免 Python 列表访问开销
            N_k = len(samples)
            feats_k = torch.stack([s[0] for s in samples])
            tgts_k = torch.tensor(
                [[float((code >> j) & 1) for j in range(self.output_size)]
                 for _, code, _ in samples], dtype=torch.float32, device=DEVICE)
            if has_gate:
                gates_k = torch.stack([s[2] for s in samples])
            else:
                gates_k = None

            W = torch.empty(self.output_size, self.feat_dim,
                            dtype=torch.float32, device=DEVICE)
            W.uniform_(-0.1, 0.1)
            b = torch.empty(self.output_size, dtype=torch.float32, device=DEVICE)
            b.uniform_(-0.1, 0.1)

            E = torch.zeros_like(W) if has_E else None

            for _ in range(n_iter):
                idx = torch.randperm(N_k, device=DEVICE).tolist()
                for m in idx:
                    feat = feats_k[m]
                    tgt = tgts_k[m]
                    gate = gates_k[m] if has_gate else None
                    out = (torch.mv(W, feat) + b > 0).float()
                    rpe = tgt - out
                    self._hebbian_step(W, rpe, feat, out, lr, E, gate=gate)

            self.W_ctx_to_pos.append(W)
            self.b_ctx_to_pos.append(b)

    # ==================== v3.2 专家化位置头训练 (动态自动扩增) ====================

    def train_pos_heads_experts(self, dialogues, lr=0.05, n_iter=500,
                                n_loops=1, expected_gate=0.0):
        """★ v3.2 语义分槽专家训练 — 动态自动扩增 (Goal-Gated Experts)

        流程:
          1. 每个对话编码输入 → goal (输入语义摘要)
          2. WTA 路由: 命中已有专家则复用; 所有专家门激活 < expert_thr
             (新颖主题) → 生长新专家 (绑定该主题 DG 特征)
          3. 每个专家独立训练: 首字符头 + 位置头组 (逐字符生长)
        组间零冲突 → 开放域容量随数据自动增长, 不再共享固定槽位。

        红线合规: 路由 = 突触整合 + WTA 竞争 + 阈值; 学习 = RPE 调制
        Hebbian; 生长 = 新颖性触发神经发生。无检索/模板/聚类索引。

        Returns:
            n_experts: 自动生长的专家数
        """
        self.expert_gates = []
        self.expert_first = []
        self.expert_first_b = []
        self.expert_pos = []
        self.expert_pos_b = []
        self._expert_active_idx = -1
        data = []  # (goal, resp_codes, expert_idx)
        for inp, resp in dialogues:
            resp_codes = self._text_to_codes(resp)
            if not resp_codes:
                continue
            self.clear_temporary_memory()
            _, acc_state = self.encode_text_lif(inp, update_memory=True,
                                                n_loops=n_loops)
            if acc_state.sum().item() == 0:
                continue
            goal = self.compute_goal_from_trace()
            if goal is None:
                goal = self._mem_feature(acc_state)
            pos_goal = getattr(self, "_last_pos_goal", None)
            if pos_goal is None:
                pos_goal = goal
            e = self._expert_route(goal)
            if e < 0:
                e = self._append_expert(goal)
            data.append((pos_goal.clone(), resp_codes, e))

        for goal, resp_codes, e in data:
            self._ensure_expert_pos(e, len(resp_codes) - 1)
            self._activate_expert_heads(e)
            # 首字符头 (RPE 调制 Hebbian, 与 W_ctx_to_first 同机制)
            tgt0 = torch.tensor(
                [float((resp_codes[0] >> j) & 1) for j in range(self.output_size)],
                dtype=torch.float32, device=DEVICE)
            Wf = self.expert_first[e]
            bf = self.expert_first_b[e]
            for _ in range(n_iter):
                pred = self._binary_decode(Wf, goal, bf)
                pred_bits = (pred > 0.5).float()
                rpe = tgt0 - pred_bits
                self._hebbian_step(Wf, rpe, goal, pred_bits, lr)
            # 位置头组 (逐位置独立, 与 train_pos_heads 同机制)
            for k, code in enumerate(resp_codes):
                target = torch.tensor(
                    [float((code >> j) & 1) for j in range(self.output_size)],
                    dtype=torch.float32, device=DEVICE)
                W = self.expert_pos[e][k]
                b = self.expert_pos_b[e][k]
                for _ in range(n_iter):
                    pred = self._binary_decode(W, goal, b)
                    pred_bits = (pred > 0.5).float()
                    rpe = target - pred_bits
                    self._hebbian_step(W, rpe, goal, pred_bits, lr)
        self._offload_active_expert()
        return len(self.expert_gates)

    # ==================== v14: DG 稀疏分离 + 三因子资格迹 ====================

    def _dg_separate(self, x):
        """★ DG 稀疏分离 (pattern separation) — 高维投射 + top-k 二值稀疏化

        生物学依据: 齿状回 (DG) 将皮层输入的稠密重叠表征转换为稀疏、
        非重叠编码, 降低记忆间干扰 (Complementary Learning Systems;
        Schapiro 2017)。通过高维随机投影 (Mossy fibers) 加上强侧抑制 (top-k)。

        作用: 解决 W_coact 状态崩塌问题。由于 W_coact 会在长序列后饱和，
        导致 x 趋于全 1 向量。直接 top-k 无法区分不同的全 1 向量。
        通过随机投影 W_ec2dg，即使 x 是高度饱和的，投影后也会在 DG 空间
        产生独特的连续值分布，再通过 top-k 就能提取出独特的稀疏正交特征。

        约束合规: 输出纯二值 {0,1}; 随机投影是固定权重不学习，top-k 是
        离散选择运算, 不产生连续数值学习信号, 不违反"无连续数值信号"红线。

        Args:
            x: 输入状态 (hidden_size-dim, 分级 [0,1])

        Returns:
            二值稀疏向量 (feat_dim, 恰好 k 个 1; 未启用时原样返回)
        """
        if not self.use_dg_separation:
            return x
        
        # 1. 抑制性中间神经元 (Feedforward Inhibition): 去除直流分量，突出特征差异
        x_centered = x - x.mean()
        
        # 2. 投射到高维 DG 空间
        dg_raw = torch.mv(self.W_ec2dg, x_centered)
        
        # 2. 强侧抑制 (WTA) -> 极度稀疏化
        k = min(self.dg_k, dg_raw.numel())
        if k <= 0:
            return torch.zeros_like(dg_raw)
            
        flat = dg_raw.reshape(-1)
        _, idx = flat.topk(k)
        out = torch.zeros_like(dg_raw)
        out.reshape(-1)[idx] = 1.0
        return out

    # ==================== v3.7 时序上下文 (TCM 漂移 + theta 相位) ====================
    # 文献: Howard & Kahana 2002 TCM (上下文逐项漂移绑定); Levy 1996 context
    #   codes (歧义序列自产上下文细胞消歧); CPG-PE NeurIPS 2024 (振荡位置码)。
    # 目的: 破 v3.6 内容轴瓶颈。语境/位置信息作为融合状态竞争评分注入,
    #   W_char 读 fused feat 时学用这些位。训练/生成逐位一致更新。
    # 固定随机投影 (无学习权重); 连续值评分 (已解禁)。

    def _ensure_seqctx(self):
        """懒初始化时序上下文参数 (固定随机投影, 无学习权重)."""
        if getattr(self, "W_ctx_in", None) is None:
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(20260821)
            scale = 1.0 / (self.feat_dim ** 0.5)
            self.W_ctx_in = torch.randn(self.feat_dim, self.feat_dim,
                                        device=DEVICE, generator=gen) * scale
            self.W_phase = torch.randn(self.feat_dim, 4, device=DEVICE,
                                       generator=gen) * scale
            self.ctx_rho = 0.9
            self.ctx_beta = 0.4
            self.ctx_w = 0.5
            self.theta_w = 0.5
        if getattr(self, "_seqctx", None) is None:
            self._seqctx = torch.zeros(self.feat_dim, dtype=torch.float32,
                                       device=DEVICE)

    def _seqctx_reset(self):
        """对话/生成起点清零时序上下文."""
        if getattr(self, "_seqctx", None) is not None:
            self._seqctx = torch.zeros(self.feat_dim, dtype=torch.float32,
                                       device=DEVICE)

    def _seqctx_contrib(self, sparse_feat, step):
        """更新 TCM 漂移上下文, 返回供融合的上下文 + theta 相位贡献.

        1) ctx <- rho*ctx + beta*(W_ctx_in * item)   (TCM 前缀漂移)
        2) phase = 多频振荡 + 位置斜坡, theta = W_phase * phase
        返回 ctx*ctx_w + theta*theta_w (正比竞争评分, 进 top-k)。
        """
        self._ensure_seqctx()
        self._seqctx.mul_(self.ctx_rho).add_(
            torch.mv(self.W_ctx_in, sparse_feat.float()), alpha=self.ctx_beta)
        pc = torch.zeros(4, dtype=torch.float32, device=DEVICE)
        pc[0] = math.sin(0.3 * step)
        pc[1] = math.sin(0.6 * step)
        pc[2] = math.sin(1.1 * step)
        pc[3] = 0.5 * step / 16.0
        theta = torch.mv(self.W_phase, pc)
        return self._seqctx * self.ctx_w + theta * self.theta_w

    def _chain_transition(self, prev_state):
        """★ v3.6 非对称链式状态流 — 上一步状态 → 当前状态的顺序贡献

        生物学依据: 海马 CA3 循环连接的非对称性 + 相位进动 (O'Keefe &
        Recce 1993) 编码序列顺序; MPN (Liu 2019) 证明局部 STDP 依赖
        "非对称连接链式状态流"才能鲁棒记忆多序列。W_chain[post, pre]
        表示 pre→post 有向转移强度, 非对称 (W_chain[i,j] ≠ W_chain[j,i])。

        红线合规: 纯矩阵乘 + 阈值 (离散), 无连续数值学习信号。
        W_chain 初始全 0 → 未训练时贡献恒 0, 单句行为完全不变。
        """
        if prev_state is None:
            return None
        return torch.clamp(torch.mv(self.W_chain, prev_state), 0.0, 1.0)

    # ----------------------------------------------------------
    # ★ v3.6 句级 WTA 竞争读出 (海马 CA3 WTA / 基底神经节竞争选择)
    # ----------------------------------------------------------
    def _ensure_sentence_templates(self, n):
        """初始化句模板 W_sent (n, feat_dim), 每句一个离散句方向。"""
        if (getattr(self, "W_sent", None) is None
                or self.W_sent.shape[0] != n):
            self.W_sent = torch.randn(
                n, self.feat_dim, dtype=torch.float32, device=DEVICE) * 0.1
        # ★ v3.8 固定随机稀疏句码: 每句 sent_code_k 个固定随机位 (近正交)。
        #   W_sent 行与输入 goal 相关 (同后缀句行 top-16 重叠 0.56, 判别弱);
        #   固定随机码两轴重叠均 ~k^2/feat_dim。固定投影, 无学习权重,
        #   红线合规 (Marr SDS / CA3 随机循环投影)。
        if (getattr(self, "sentence_codes", None) is None
                or self.sentence_codes.shape[0] != n
                or self.sentence_codes.shape[1] != self.feat_dim):
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(20260822)
            k_code = int(getattr(self, "sent_code_k", 24))
            codes = torch.zeros(n, self.feat_dim, dtype=torch.float32,
                                device=DEVICE)
            for r in range(n):
                idx = torch.randperm(self.feat_dim, device=DEVICE,
                                     generator=gen)[:k_code]
                codes[r, idx] = 1.0
            self.sentence_codes = codes

    def _select_sentence(self, goal):
        """句级 WTA 竞争: 输入语义摘要 → argmax 选句模板 (离散)。

        与 v3.5 字符竞争头 W_char 同源的竞争读出, 但作用于句级方向:
        raw = W_sent · goal → argmax 选出唯一句模板 (离散, 不漂移)。
        红线合规: 纯竞争选择 (WTA), 无 sigmoid/softmax/梯度。
        """
        raw = torch.mv(self.W_sent, goal)
        winner = int(raw.argmax().item())
        # ★ v3.8: 返回固定随机稀疏句码 (近正交身份通道) 而非 W_sent 行;
        #   旧缓存无句码时回退旧行为。
        if getattr(self, "sentence_codes", None) is not None:
            return self.sentence_codes[winner].clone(), winner
        return self.W_sent[winner].clone(), winner

    def _conj_bind(self, x_t, ctx):
        """★ v3.3 P0a 时序合取绑定 — XOR + 固定置换 + top-k (VSA 绑定)

        把当前 DG 稀疏特征 x_t 与"已发生顺序"的上下文 ctx 绑定为一个稀疏码:
            ctx ← top_k( XOR( x_t, permute(ctx) ), conj_k )
        - permute: 固定随机置换 (顺序敏感: 相同字符不同顺序 → 不同上下文)
        - XOR: 离散合取绑定 (Plate 1995), 非连续数值映射 (红线内)
        - top-k: 离散选择, 与 _dg_separate 同族 (红线内)
        不引入任何可学习权重 / 连续学习信号。绑定后仍为二值稀疏向量。

        Args:
            x_t: 当前字符 DG 特征 (feat_dim, 二值稀疏)
            ctx: 上一时刻合取上下文 (feat_dim, 二值稀疏)

        Returns:
            新合取上下文 (feat_dim, 二值稀疏, 至多 conj_k 个 1)
        """
        if self._conj_perm is None or self._conj_perm.numel() != self.feat_dim:
            _gen = torch.Generator(device=DEVICE)
            _gen.manual_seed(20260812)
            self._conj_perm = torch.randperm(
                self.feat_dim, device=DEVICE, generator=_gen)
        ctx_shifted = ctx[self._conj_perm]
        bound = torch.logical_xor(x_t.bool(), ctx_shifted.bool()).float()
        flat = bound.reshape(-1)
        k = min(int(self.conj_k), flat.numel())
        if k <= 0:
            return torch.zeros_like(bound)
        _, idx = flat.topk(k)
        out = torch.zeros_like(flat)
        out[idx] = 1.0
        # 只保留 top-k 中真正为 1 的位置 (bound 不足 k 个 1 时不补假位)
        out = out * (flat > 0.0).float()
        return out.reshape_as(bound)

    def _mem_feature(self, state):
        """记忆头统一输入特征 — state 已经是 DG 分离后的 feat_dim 向量，直接返回"""
        return state

    # ==================== v14.12 循环思考储层 + 目标注入 ====================

    def _encode_reservoir(self, x_t):
        """★ 循环思考储层一步更新 (Liquid State Machine, Maass 2002)

        生物学依据: 皮层/海马 CA3 的循环网络把"已说的话"保持在神经活动中,
        提供跨字符的持续状态 → W_seq 不再只是 1 阶马尔可夫。
        储层为固定随机稀疏连接 (Mossy fiber 式), 无学习信号 (红线合规),
        LIF 膜电位动力学: V ← (1-leak)·V + W_res·spike(V) + W_res_in·x。

        Args:
            x_t: 当前字符输入特征 (feat_dim 稀疏二值)

        Returns:
            think: 储层思考特征 (feat_dim, 随机投影, 供生成状态融合)
        """
        if not getattr(self, "use_reservoir", False):
            return None
        res = self.reservoir
        res_spike = (res > self.res_thr).float()
        res = (res * (1.0 - self.res_leak) +
               torch.mv(self.W_res, res_spike) +
               torch.mv(self.W_res_in, x_t))
        self.reservoir = torch.clamp(res, 0.0, 1.0)
        think = torch.mv(self.W_res_out, (self.reservoir > self.res_thr).float())
        return think

    def _guided_state(self, state, goal=None, think=None):
        """★ 生成状态融合: 当前状态 + 储层思考 + 目标意图注入 (方案 A/B)

        - think: 储层"思考"特征 (循环记忆, 方案 B)
        - goal:  输入意图 (自上而下目标信号, 方案 A)
        生物学: 前额叶→语言区的自上而下意图信号持续引导逐词生成;
        max 融合 = 神经元并集 (维持稀疏, 无连续数值学习信号)。
        """
        if think is not None and getattr(self, "use_reservoir", False):
            state = torch.max(state, think)
        # ★ v3.1: _goal_guidance_enabled=False (阶段一) 时禁止目标意图注入,
        # 防止"答案引导"泄漏进短时续写训练。
        if goal is not None and getattr(self, "_goal_guidance_enabled", True):
            # ★ P0: 方向二值化持久锚定 (Wang 2021 工作记忆多吸引子)。
            #   方向是 [0,1] 连续值, state 是二值 {0,1}。旧版 goal*0.5
            #   注入后 <=0.5, 永压不过已有位型, 方向失活。
            #   改为 (goal>0)->1 的持久位型, 每步以满强度 1 参与 max 并集,
            #   把状态持续拉回该句吸引子 — 防自由漫游吸引子合并。
            #   红线合规: 离散二值选择, 非连续数值学习信号。
            binary_goal = (goal > 1e-6).float()
            state = torch.max(state, binary_goal)
        return state

    def _slot_fatigue_step(self, sparse_feat, cf=None, code=None):
        """★ v3.8 P44/P46 读出贡献位疲劳 (重复抑制, 精准版)。

        输出/喂入字符 code 后, 疲劳 W_char[code] 行在 cf 活跃位上
        权重最高的 top-8 位 — 即 cf 中该成分的表征位 (读出权重反馈)。
        P44 版用生成上下文 sparse_feat 位, 与输入 cf 位空间不匹配
        (DG 位上下文依赖) → 扣错位损伤模板字符 ('blue dog'→'dodo')。
        (Grill-Spector 2006 repetition suppression; P21 EOS 脉冲同族)"""
        flr = float(getattr(self, "slot_fatigue_lr", 0.0) or 0.0)
        if flr <= 0 or cf is None or code is None:
            return
        W = getattr(self, "W_char", None)
        if W is None or not (0 <= int(code) < W.shape[0]):
            return
        cfa = (cf > 1e-6).float()
        contrib = W[int(code)] * cfa
        if contrib.max().item() <= 0:
            return
        _, idx = contrib.topk(min(8, contrib.numel()))
        fat = getattr(self, "_slot_fatigue", None)
        if fat is None or fat.numel() != contrib.numel():
            self._slot_fatigue = torch.zeros(
                self.feat_dim, dtype=torch.float32, device=DEVICE)
        self._slot_fatigue[idx] += flr

    def _slot_fatigue_reset(self):
        """句首重置成分疲劳 (每句独立)。"""
        if getattr(self, "_slot_fatigue", None) is not None:
            self._slot_fatigue = torch.zeros_like(self._slot_fatigue)

    def _content_quota_src(self, cf):
        """★ v3.8 P41 成分 quota 源: 优先 _last_goal (输入语义 max 摘要)。

        TCM cf = ρ·ctx + β·new 时序衰减偏向近期词 — 'red bear' 中 'red'
        (远期) 衰减多, top-k 被近期词占据 → 形容词位判别信号丢失
        (P40 拉丁方: 留出动物位 3/4 对, 形容词位仅 1/4)。_last_goal 为
        max DG 摘要, 无时序偏置, 两成分平等。"""
        g = getattr(self, "_last_goal", None)
        if g is not None and g.numel() == self.feat_dim:
            return g
        return cf

    def _fuse_state_kwt(self, sparse_feat, recall=None, chain=None,
                        goal=None, k=None, seqctx=None, content=None):
        """★ P6 分级状态融合 + k-WTA 侧抑制 (防 OR 并集单调饱和)

        诊断: 旧融合 max(sparse, recall, chain, direction, dmd_direction)
        为单调不减并集, 叠加 prev_state→W_chain→state 自反馈, 2 步内
        饱和至 fpop 244/256 → feat/h 逐步恒定 (h_ham=0) → 读出恒定
        → 生成重复锁定 (hellll...); 训练精度全靠逐步在线更新维持,
        冻结推理崩溃。

        修复: 树突分级求和 (连续值, 各来源加权) + top-k 侧抑制 —
        各来源按强度竞争固定 k 个槽位, 状态保持稀疏且随输入变化。
        生物学: 树突分级电位积分 + 中间神经元 k-WTA 侧抑制
        (Carandini & Heeger 2012; 与 recall_from_memassoc v14.6 同族)。
        """
        if k is None:
            k = getattr(self, "state_kwt_k", 0) or max(8, 2 * self.dg_k)
        fused = torch.zeros(self.feat_dim, dtype=torch.float32,
                            device=DEVICE)
        if sparse_feat is not None:
            fused = fused + sparse_feat.float()
        if chain is not None:
            fused = fused + 0.9 * chain
        if recall is not None:
            fused = fused + 0.8 * recall
        if goal is not None and getattr(self, "_goal_guidance_enabled", True):
            fused = fused + 0.6 * (goal > 1e-6).float()
        if seqctx is not None:
            fused = fused + seqctx  # v3.7 时序上下文
        k = min(int(k), fused.numel())
        if k <= 0 or fused.max().item() <= 0:
            return torch.zeros_like(fused)
        _, idx = fused.topk(k)
        out = torch.zeros_like(fused)
        out[idx] = 1.0
        # ★ v3.8 方向保护通道 (dir_quota_k > 0 启用): CA3 吸引子持续性 —
        #   句身份 (goal/direction) 的强位不受 top-k 竞争稀释, 直接并入
        #   融合状态 (P32/P33 实测存活率 0.29 且存活偏向共模位 → slot
        #   字符不可分)。训练/生成共用本函数, 逐位一致; 默认 0 兼容。
        # ★ v3.8 P38 成分化 quota (quota_from_content=True): 保护内容从
        #   随机句码改为输入成分表征 (content=cf) — P37 证明随机句码
        #   与 slot 内容无结构对应 (96 句线性不可分容量悬崖); cf 的成分
        #   位跨句共享 (同动物 10 句一致) → slot 判别线性可分 + 组合泛化。
        dq = int(getattr(self, "dir_quota_k", 0) or 0)
        if dq > 0 and getattr(self, "_goal_guidance_enabled", True):
            src_q = None
            if (content is not None
                    and getattr(self, "quota_from_content", False)):
                src_q = content.float()
                # ★ P44 重复抑制: 已输出成分位疲劳压低, 未输出成分凸显
                flr = float(getattr(self, "slot_fatigue_lr", 0.0) or 0.0)
                if flr > 0:
                    fat = getattr(self, "_slot_fatigue", None)
                    if fat is not None and fat.numel() == src_q.numel():
                        src_q = src_q - fat  # P46: fat 已含幅度
            elif goal is not None:
                src_q = goal
            if src_q is not None:
                g = src_q.float()
                if g.numel() == out.numel():
                    _, didx = g.topk(min(dq, g.numel()))
                    out[didx] = 1.0
        return out

    def compute_goal_from_trace(self):
        """返回当前目标意图 = 输入语义摘要 (max DG 特征, 归一化)

        ★ v14.12: 优先用 encode_text_lif 记录的 _last_goal (输入语义 max 摘要);
        未编码时回退到时序迹。
        """
        if getattr(self, "_last_goal", None) is not None:
            return self._last_goal.clone()
        if not hasattr(self, "_coact_trace"):
            return None
        goal = self._coact_trace.clone()
        fmax = goal.max()
        if fmax > 0:
            goal = goal / fmax
        return goal

    def _hebbian_step(self, W, rpe, pre, post, lr, E=None, gate=None,
                      protect=None):
        """★ 统一权重更新: 即时 RPE 调制 Hebbian 或三因子资格迹

        即时模式 (原规则, E=None):
            Δw_ji = lr × RPE_j × pre_i
        ★ v2.2 预期门控 (gate 非 None, 预测性 Hebbian):
            Δw_ji = lr × RPE_j × pre_i × gate_i
            gate ∈ {0,1} — 只有"被预期活跃"的突触前神经元参与强化
            (Saponati & Vinck 2023: 放大最能预测其他输入的突触)。
            gate 是布尔门控, 不产生连续数值学习信号 (红线内)。
        ★ v2.4 突触保护 (protect 非 None, ISI-CV 本地版):
            Δw_ji = lr × RPE_j × pre_i × (1 − protect_ji)
            protect ∈ [0,1] — 已巩固突触跳过/降速更新 (防在线学习
            覆盖旧知识)。掩码构造基于更新方向统计与发放频率 (慢变量,
            同 SFA thr_shift 族), 不产生连续数值学习信号 (红线内)。
        资格迹模式 (三因子, E 为与 W 同形状的迹矩阵):
            e_ji ← λ·e_ji + pre_i × post_j   (Hebbian 共激活设置迹)
            Δw_ji = lr × M_j × e_ji           (神经调质门控实际变化)
        文献: Gerstner & Lehmann 2018 (Eligibility Traces and
        Plasticity on Behavioral Time Scales); E-prop (Bellec 2020)。

        M_j = RPE_j ∈ {−1,0,+1} 仍是纯离散奖赏信号, 无连续梯度;
        资格迹是突触内部状态 (非学习信号), 不违反红线。

        Args:
            W: 权重矩阵 (output_size, input_size), 原地更新
            rpe: 奖赏预测误差 (output_size,), 元素 ∈ {−1,0,+1}
            pre: 突触前活动向量 (input_size,)
            post: 突触后二值输出 (output_size,), 资格迹模式需要
            lr: 学习率
            E: 资格迹矩阵 (与 W 同形状); None = 即时模式
            gate: 预期门 (input_size,), 元素 ∈ {0,1}; None = 无门控
            protect: 突触保护掩码 (与 W 同形状), 元素 ∈ [0,1];
                     None = 不保护 (v2.4)
        """
        # ★ 性能优化: 将 Python 级 for j 循环替换为张量并行运算 (数学等价，且仍为逐样本)
        if E is None:
            gpre = pre
            if gate is not None:
                gpre = gpre * gate  # 只有预期活跃的突触前参与更新

            if protect is None:
                W.addr_(rpe, gpre, alpha=lr)
            else:
                # rpe[:, None] * gpre[None, :] = outer(rpe, gpre)
                delta = lr * torch.outer(rpe, gpre) * (1.0 - protect)
                W.add_(delta)
        else:
            # E.shape: (output_size, input_size)
            E.mul_(self.eligibility_lambda).add_(torch.outer(post, pre))
            delta = lr * rpe.unsqueeze(1) * E
            if protect is not None:
                delta = delta * (1.0 - protect)
            W.add_(delta)

        W.clamp_(-10.0, 10.0)

    def _expected_gate(self, state, theta):
        """★ v2.2 预期门 — 预测性 Hebbian (Saponati & Vinck 2023)

        预期信号 = W_coact·state 共激活回忆 (与该状态共发放过的
        神经元群体, 即"当前状态预期出现的活动模式")。
        gate = (recall > theta) — 只有被预期活跃的突触前神经元
        参与奖赏强化 → 放大"最能预测其他输入"的突触, 复现 STDP
        时序不对称 (Keysers & Gazzola 2014)。

        红线合规: recall 是固定联想矩阵 W_coact 的线性回忆 (非学习);
        θ=0 时 gate 恒 1 (recall ≥ 0), 完全等价无门控 (向后兼容)。

        Args:
            state: 工作记忆累积状态 (hidden_size-dim, ∈ [0,1])
            theta: 预期门阈值 (0 = 恒 1 无门控)

        Returns:
            gate: 布尔门向量 (hidden_size-dim, {0,1})
        """
        if theta <= 0:
            # θ=0: 严格无门控 (即使 recall=0 的神经元也全开)
            return torch.ones(self.hidden_size, dtype=torch.float32, device=DEVICE)
        recall = self.recall_from_memassoc(state)
        return (recall > theta).float()

    def _expert_route(self, goal):
        """★ v3.2 专家路由 — 神经元竞争 (突触整合 + WTA + 阈值生长)

        每个专家门向量 G_e 对 goal 做线性突触加权求和 (与 W·feat 同族),
        再经 WTA 竞争 (侧抑制) 选出最强专家。若最强激活也低于 expert_thr
        (新颖主题, 尚无专家记忆) 返回 -1 → 触发新专家分配。
        红线合规: 无余弦相似度/最近邻/字符串/字典, 只有加权和 + 阈值 +
        竞争性发放 (生物基底节动作选择的 WTA)。
        """
        if not getattr(self, "use_experts", False) or not self.expert_gates:
            return -1
        best = -1
        best_act = self.expert_thr
        for e, gate in enumerate(self.expert_gates):
            overlap = float(torch.dot(gate, goal).item())
            norm = max(float(gate.sum().item()), 1.0)
            act = overlap / norm  # 输入覆盖该门神经元群体的比例 ∈ [0,1]
            if act > best_act:
                best_act = act
                best = e
        return best

    def _move_expert_heads(self, e, device):
        self.expert_first[e] = self.expert_first[e].to(device)
        self.expert_first_b[e] = self.expert_first_b[e].to(device)
        self.expert_pos[e] = [W.to(device) for W in self.expert_pos[e]]
        self.expert_pos_b[e] = [b.to(device) for b in self.expert_pos_b[e]]

    def _activate_expert_heads(self, e):
        if not getattr(self, "expert_heads_cpu", False):
            return
        active = getattr(self, "_expert_active_idx", -1)
        if active == e:
            return
        if active >= 0:
            self._move_expert_heads(active, "cpu")
        self._move_expert_heads(e, DEVICE)
        self._expert_active_idx = e

    def _offload_active_expert(self):
        active = getattr(self, "_expert_active_idx", -1)
        if getattr(self, "expert_heads_cpu", False) and active >= 0:
            self._move_expert_heads(active, "cpu")
            self._expert_active_idx = -1

    def _append_expert(self, goal):
        e = len(self.expert_gates)
        self.expert_gates.append((goal > 0).float().detach().clone())
        head_device = "cpu" if self.expert_heads_cpu else DEVICE
        W = torch.empty(self.output_size, self.feat_dim,
                        dtype=torch.float32, device=head_device)
        W.uniform_(-0.1, 0.1)
        b = torch.empty(self.output_size, dtype=torch.float32,
                        device=head_device)
        b.uniform_(-0.1, 0.1)
        self.expert_first.append(W)
        self.expert_first_b.append(b)
        self.expert_pos.append([])
        self.expert_pos_b.append([])
        return e

    def _ensure_expert_pos(self, e, k):
        # ★ v3.3: 新位置头跟随专家当前所在设备 (expert_first[e].device) —
        #   激活态专家 (CUDA) 新增位置头也在 CUDA, 未激活 (CPU) 则在 CPU,
        #   避免 _activate_expert_heads 短路时 CPU/CUDA 混设备。
        head_device = self.expert_first[e].device
        while len(self.expert_pos[e]) <= k:
            W = torch.empty(self.output_size, self.feat_dim,
                            dtype=torch.float32, device=head_device)
            W.uniform_(-0.1, 0.1)
            b = torch.empty(self.output_size, dtype=torch.float32,
                            device=head_device)
            b.uniform_(-0.1, 0.1)
            self.expert_pos[e].append(W)
            self.expert_pos_b[e].append(b)

    def pos_head_recall(self, state, k):
        """位置记忆头回忆: 状态 → 第 k 字符 (纯二值阈值解码 + margin 诊断)

        Returns:
            (code, margin): code = 回忆的 ASCII 码; margin = min_j |raw_j|
            是 8 个 bit 中解码最接近阈值 0 的 margin。
            ★ experiment14: margin 无判别力 (正确 med=0.09 vs 错误 med=0.05
            重叠) → 仅保留用于诊断, 不作为修正门控依据。
            超出已训练位置范围返回 (None, 0.0)。
        """
        # ★ v3.2: 专家模式 — 先用 WTA 竞争选专家, 再从该专家位置头组读
        # 路由/输入统一用 _last_goal (与训练一致), 缺失时回退 state
        if getattr(self, "use_experts", False) and self.expert_pos:
            goal = getattr(self, "_last_goal", None)
            route_in = goal if goal is not None else state
            e = self._expert_route(route_in)
            if e < 0 or k >= len(self.expert_pos[e]):
                return None, 0.0
            self._activate_expert_heads(e)
            pos_goal = getattr(self, "_last_pos_goal", None)
            feat = self._mem_feature(
                pos_goal if pos_goal is not None else route_in)
            raw = (torch.mv(self.expert_pos[e][k], feat) +
                   self.expert_pos_b[e][k])
            code = 0
            for j in range(self.output_size):
                if raw[j] > 0:
                    code |= (1 << j)
            margin = raw.abs().min().item()
            return code, margin
        if k >= len(self.W_ctx_to_pos):
            return None, 0.0
        # ★ v14: DG 稀疏分离 (与训练一致, 训练/回忆状态逐位相同)
        feat = self._mem_feature(state)
        raw = torch.mv(self.W_ctx_to_pos[k], feat) + self.b_ctx_to_pos[k]
        code = 0
        for j in range(self.output_size):
            if raw[j] > 0:
                code |= (1 << j)
        margin = raw.abs().min().item()
        return code, margin

    # ==================== W_seq 序列训练 — 奖赏调制 Hebbian ====================

    def _seq_predict(self, feat, gate=None, use_char_wta=False):
        """深度读出口预测下一字符 (方案 C)

        隐藏层 W_seq_h 用预测编码训练 (预测下一步输入特征), 输出层
        W_seq_out 从隐藏预测状态读字符。无 W_seq_h 时回退单层 W_seq。
        纯二值阈值解码, 无连续数值学习信号 (红线合规)。

        ★ v3.6 稀疏路由: gate (direction) 稀疏门控 h, 让不同句子的
          读出走不同的 W_seq_out 列组合 (共享权重 + 稀疏激活, 避免多句
          竞争 — 海马 CA3 稀疏编码 / 前额叶门控同源)。
        """
        if hasattr(self, "W_seq_h"):
            h = (torch.mv(self.W_seq_h, feat) > self.seq_h_thr).float()
            if gate is not None:
                h = h * (gate > 0).float()
            if use_char_wta and getattr(self, "W_char", None) is not None:
                # ★ P1 字符级 WTA 竞争读出 (McKinstry & Edelman 2013;
                #   VOWEL 2020): 256 字符神经元对 h 连续打分, argmax
                #   唯一胜者 — 替代独立 8bit 阈值 (可拼出不可能字节码,
                #   且无字符间竞争, 是“字节回放”退化根因)。
                # ★ P7: W_char 直接从融合状态 feat 读出, 绕过 W_seq_h
                #   瓶颈 — W_seq_h 被预测编码训练成逼近 sparse_feat
                #   (纯字符身份), 阈值化后 chain 位置位被洗掉, 双写字母
                #   (ll) 两步 h 不可分 → l→o 转移失败。feat 含
                #   chain (0.9)/recall (0.8)/direction (0.6) 融合位。
                scores = torch.mv(self.W_char, feat) + self.b_char
                # ★ P10-B: 暴露连续得分 (EOS margin 门控/诊断用)
                self._last_char_scores = scores
                code = int(scores.argmax().item())
                bits = torch.zeros(self.output_size, dtype=torch.float32,
                                   device=feat.device)
                for j in range(self.output_size):
                    if (code >> j) & 1:
                        bits[j] = 1.0
                return bits
            return (torch.mv(self.W_seq_out, h) + self.b_seq > 0).float()
        return self._binary_decode(self.W_seq, feat, self.b_seq)

    def train_sequence(self, dialogues, lr=0.5, n_iter=1000, n_loops=1,
                       expected_gate=0.0, use_goal=None, read_long_term=None,
                       write_long_term=None, early_stop_patience=None,
                       use_compile=True, scheduled_sampling_prob=0.0):
        """训练 W_seq — 奖赏预测误差调制 Hebbian 学习

        ★ 学习规则: Δw = lr × RPE_j × pre_activity
          - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差
            +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
          - pre_activity = 二值 {0, 1} 神经元输出 (来自 encode_text_lif)
          - ★ 无 center = clamp(raw, -1, 1): 连续数值运算, 已移除
          - ★ 无 b_seq 偏置更新: 连续数值运算, 已移除
          - 无 autograd，无反向传播，无批量处理，无目标误差

        ★ v13.1: n_loops 自回归循环 (透传给 encode_text_lif_states), 默认 1。
        ★ v2.2: expected_gate 预期门控 (预测性 Hebbian, 默认 0=关) —
          W_seq 是最自然的"预期下一字符"层: 门控 = 当前状态预期活跃的
          突触前参与强化 → 放大"最能预测其他输入"的突触 (Saponati 2023)。

        ★ v3.1 两阶段记忆训练 (CLS 互补学习系统):
          use_goal/read_long_term/write_long_term = None 时跟随
          set_memory_mode 的全局开关; 显式 True/False 可覆盖。
          - use_goal=False: 禁止目标意图注入 (阶段一: 纯短时续写)
          - read_long_term=False: 禁止长期记忆读出 (recall 只用
            W_coact_temp, 阶段一: 短时/工作记忆)
          - write_long_term=True: 每条对话结束后 consolidate_coactivation
            (短期轨迹 → W_coact_long 长期巩固, 阶段二: 开放长期记忆)

        Args:
            dialogues: 对话对列表 [(inp, resp), ...]
            lr: 学习率
            n_iter: 训练迭代次数
            n_loops: 每次输入自回归前向轮数 (v13.1, 默认 1)
            expected_gate: 预期门阈值 (0 = 无门控, 向后兼容)

        Returns:
            best_acc: 最佳预测准确率
        """
        # ★ v3.1: 显式参数覆盖全局模式, 训练期间临时生效, 结束后恢复。
        goal_on = getattr(self, "_goal_guidance_enabled", True) if use_goal is None else bool(use_goal)
        read_long = getattr(self, "_long_term_read_enabled", True) if read_long_term is None else bool(read_long_term)
        write_long = getattr(self, "_long_term_write_enabled", True) if write_long_term is None else bool(write_long_term)
        _saved_mode = (getattr(self, "_long_term_read_enabled", True),
                       getattr(self, "_long_term_write_enabled", True),
                       getattr(self, "_goal_guidance_enabled", True))
        self._long_term_read_enabled = read_long
        self._long_term_write_enabled = write_long
        self._goal_guidance_enabled = goal_on
        # ★ v3.0/v14.12: 每对话从清空的临时记忆开始; 先编码输入得目标意图
        #   goal, 再手动编码回复 (续接), 每步状态注入 储层思考 + goal —
        #   与 generate_recurrent 完全一致 (训练/推理状态逐位相同)。
        seq_data = []
        self._seq_snapshots = None
        try:
            for inp, resp in dialogues:
                self.clear_temporary_memory()

                resp_codes = self._text_to_codes(resp)
                if len(resp_codes) < 2:
                    continue
                # 1) 编码输入 → 目标意图 goal (仅输入语义, 自上而下引导)
                #    ★ v3.1: 阶段一 (goal_on=False) 不编码输入, 纯短时续写
                #    ★ v3.4: DMD 选择性巩固时记录输入期间残差 (新颖性门控)
                goal = None
                if (goal_on and
                        (getattr(self, "use_reservoir", False) or getattr(self, "goal_strength", 0) > 0)):
                    self.encode_text_lif(
                        inp, update_memory=True, n_loops=n_loops,
                        record_dmd=getattr(self, "use_dmd_selective_consolidation", False))
                    goal = self.compute_goal_from_trace()
                # ★ v3.1: 回复编码前一律重置状态 (输入编码后重置, 防跨对话泄漏)
                self.reset_state()
                self.reset_memory()
                # 2) 手动编码回复 (续接情节), 每步注入储层思考 + 目标意图
                #    样本: (prev_guided, x_next, resp[i], gate) — 训练/推理逐位一致
                prev_guided = None
                for i, ch in enumerate(resp):
                    feed_ch = ch
                    if (i > 0 and prev_guided is not None
                            and scheduled_sampling_prob > 0
                            and random.random() < scheduled_sampling_prob):
                        predicted_bits = self._seq_predict(prev_guided)
                        predicted_code = 0
                        for bit_idx in range(self.output_size):
                            if predicted_bits[bit_idx] >= 0.5:
                                predicted_code |= 1 << bit_idx
                        feed_ch = chr(predicted_code)
                    vec = self._char_to_8bit_bias(feed_ch)
                    if self.use_memory_thinking:
                        _, x_t, state = self._think_character(
                            vec, update_memory=True, n_loops=n_loops)
                    else:
                        self._multi_layer_forward(vec, n_loops=n_loops)
                        v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                        x_t = self._dg_separate(v_curr)
                        self.update_coactivation(x_t)
                        recall = self.recall_from_memassoc(x_t, sparse_hint=True)
                        state = torch.max(x_t, recall)
                    think = self._encode_reservoir(x_t)
                    guided = self._guided_state(state, goal, think)
                    self.MemWork = guided
                    if prev_guided is not None:
                        target = self._target_bits(resp_codes[i])
                        gate = self._expected_gate(guided, expected_gate) if expected_gate > 0 else None
                        seq_data.append((prev_guided, x_t, target, gate))
                    prev_guided = guided
                # ★ v3.1: 短期→长期巩固 (阶段二 write_long=True 时逐对话写入)
                if write_long:
                    self.consolidate_coactivation()
        finally:
            (self._long_term_read_enabled, self._long_term_write_enabled,
             self._goal_guidance_enabled) = _saved_mode

        if not seq_data:
            return 0.0

        n_data = len(seq_data)

        lr_current = lr
        # ★ v14: 资格迹模式需维护迹矩阵
        E = torch.zeros_like(self.W_seq) if self.use_eligibility_trace else None

        has_gate = expected_gate > 0
        has_E = self.use_eligibility_trace

        # 预先堆叠张量以利用连续内存
        feats_all = torch.stack([s[0] for s in seq_data])
        xnext_all = torch.stack([s[1] for s in seq_data])
        tgts_all = torch.stack([s[2] for s in seq_data])
        gates_all = torch.stack([s[3] for s in seq_data]) if has_gate else None

        # ★ v3.3 性能: unbind 成 Python 列表, 逐样本索引不再触发 GPU 内核
        feats_list = feats_all.unbind(0)
        xnext_list = xnext_all.unbind(0)
        tgts_list = tgts_all.unbind(0)

        # 初始评估 — 纯二值阈值解码 (张量累积, 末次单次 .item() 同步)
        correct_tensor = torch.zeros(1, dtype=torch.float32, device=DEVICE)
        for m in range(n_data):
            out = self._seq_predict(feats_list[m])
            if (out == tgts_list[m]).all().item():
                correct_tensor += 1.0
        best_acc = float(correct_tensor.item()) / n_data

        # ★ v3.3 编译路径: 仅在 CUDA + Triton 可用时尝试 torch.compile
        # (reduce-overhead = CUDA graph, 消除逐样本内核启动开销); 失败/缺
        # 依赖时自动回退到下方普通循环 — 两者逐位等价 (同一 _seq_step_body)。
        compiled_step = None
        if use_compile and DEVICE.type == "cuda" and not has_E and not has_gate:
            try:
                import importlib.util
                if importlib.util.find_spec("triton") is not None:
                    compiled_step = torch.compile(_seq_step_body, mode="reduce-overhead")
            except Exception:
                compiled_step = None

        no_improve = 0
        for epoch in range(n_iter):
            # 统计用张量累积 (无 .item() 同步, epoch 末一次性转 int)
            correct_tensor = torch.zeros(1, dtype=torch.float32, device=DEVICE)
            idx = torch.randperm(n_data, device=DEVICE).tolist()

            if not has_E and not has_gate:
                # SUPER FAST PATH — ★ v14.12 深度读出口 (方案 C):
                #   隐藏层 W_seq_h: 预测编码 (预测下一步输入特征, Rao & Ballard 1999)
                #   输出层 W_seq_out: 从隐藏预测状态读下一字符
                #   ★ v3.3 性能: 稀疏列 index_add 换成数学等价的 addr_ 外积
                #   (GPU 上 nonzero 索引开销 ≫ 稠密外积写入); clamp 留在 epoch 末尾
                if compiled_step is not None:
                    # CUDA graph 重放: 样本先拷贝进固定 buffer 再重放图
                    fb = torch.zeros_like(feats_all[0])
                    xb = torch.zeros_like(xnext_all[0])
                    tb = torch.zeros_like(tgts_all[0])
                    for m in idx:
                        fb.copy_(feats_list[m])
                        xb.copy_(xnext_list[m])
                        tb.copy_(tgts_list[m])
                        correct_tensor += compiled_step(
                            self.W_seq_h, self.W_seq_out, self.b_seq,
                            fb, xb, tb, lr_current * 0.5, lr_current,
                            self.seq_h_thr)
                else:
                    for m in idx:
                        feat = feats_list[m]
                        xn = xnext_list[m]
                        tgt = tgts_list[m]

                        h = (torch.mv(self.W_seq_h, feat) > self.seq_h_thr).float()
                        xn_bits = xn
                        rpe_h = xn_bits - h
                        # W_seq_h += lr*0.5 · outer(rpe_h, feat) — feat 零列增量为 0,
                        # 与旧 index_add_ 稀疏列更新逐元素数学等价
                        self.W_seq_h.addr_(rpe_h, feat, alpha=lr_current * 0.5)

                        out = (torch.mv(self.W_seq_out, h) + self.b_seq > 0).float()
                        rpe = tgt - out
                        self.W_seq_out.addr_(rpe, h, alpha=lr_current)

                        correct_tensor += (out == tgt).all().float()
                self.W_seq_h.clamp_(-10.0, 10.0)
                self.W_seq_out.clamp_(-10.0, 10.0)
            else:
                # NORMAL PATH
                for m in idx:
                    feat = feats_all[m]
                    tgt = tgts_all[m]
                    gate = gates_all[m] if has_gate else None

                    out = self._binary_decode(self.W_seq, feat, self.b_seq)
                    pred_bits = (out > 0.5).float()
                    rpe = tgt - pred_bits

                    self._hebbian_step(self.W_seq, rpe, feat, pred_bits, lr_current, E, gate=gate)

                    correct_tensor += (pred_bits == tgt).all().float()

            correct_count = int(correct_tensor.item())
            acc = correct_count / n_data
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % 200 == 0:
                lr_current *= 0.9
            # ★ v3.3 平台期提前停止: best_acc 连续 patience 个 epoch 无提升即停
            if early_stop_patience is not None and no_improve >= early_stop_patience:
                break

        return best_acc

    # ==================== v3.5.1 DMD on-policy rollout 训练 ====================
    # 目标: 消除 teacher-forcing 与 DMD free-running 的状态域不一致。
    # 思路: 训练态 = 推理态。每轮从 input_end() 的 context/direction 出发,
    #       后续字符反馈模型自身预测 (与 generate_recurrent_dmd 同链:
    #       _char_to_8bit_bias → _multi_layer_forward/_think_character →
    #       _dg_separate → recall → state → _guided_state → _dmd_step),
    #       对每个实际经过的状态, 以真实回复字符为目标做局部 Hebbian 更新。
    # 难度随 n_iter 与 rollout 增长; 定位为"单句/小语料闭环过拟合"验证工具。

    def train_dmd_rollout(self, dialogues, lr=0.5, n_iter=200, n_loops=1,
                          train_first=True, scheduled_sampling_prob=1.0,
                          max_steps=64, early_stop_patience=None,
                          dir_strength=1.0, lr_decay_step=50,
                          lr_decay_factor=0.9, eos_code=None,
                          train_done=False, consolidate_long=False,
                          train_chain=False, chain_lr=0.05,
                          confidence_gate=False,
                          use_dir_gated_readout=False,
                          dir_supervised_h=False,
                          dir_supervised_early_steps=2,
                          encode_retrieve_sep=False,
                          use_sentence_wta=False,
                          use_char_wta=False, char_wta_lr=0.5,
                          char_row_protect=False):
        """on-policy DMD rollout 训练 W_seq (与推理完全同域)

        Args:
            dialogues: [(inp, resp), ...]
            lr: W_seq 深读出口学习率
            n_iter: rollout 训练轮数
            n_loops: 每步自回归前向轮数
            train_first: 是否同时训练首字符头
            scheduled_sampling_prob: 字符反馈中"模型预测"的比例
              (1.0 = 纯 free-running 反馈, 0.0 = 纯真值反馈)
            max_steps: 单次 rollout 最大字符数
            dir_strength: DMD 方向注入强度
            lr_decay_step: lr 衰减间隔轮数 (默认 50, 与旧行为一致)
            lr_decay_factor: lr 衰减系数 (默认 0.9, 与旧行为一致)
            eos_code: 结束符字节码 (默认 None 不训练; 指定如 0 时在
              目标序列末尾追加 EOS, 训练模型在序列结束后输出该标记)
            train_done: 是否同时训练独立结束读出器 W_done
              (最后一个字符状态 → done=1, 其余 → done=0)
            consolidate_long: 每个对话结束后是否巩固短期→长期记忆
              (CLS 互补学习系统, 防多句灾难性遗忘)
            train_chain: 是否训练非对称链式状态流 W_chain (MPN Liu 2019)
            chain_lr: W_chain STDP 学习率
        Returns:
            best_acc: 每个 rollouted 状态处的整字符预测准确率
        """
        lr_h = lr * 0.5
        no_improve = 0
        best_acc = 0.0
        correct_all = 0.0
        total_all = 0
        # ★ v3.6 句级 WTA: 初始化句模板矩阵 (每句一个离散句方向)
        if use_sentence_wta:
            self._ensure_sentence_templates(len(dialogues))
        # ★ P1 字符级 WTA 读出头: 256 字符神经元 × feat_dim 连续权重
        if use_char_wta and getattr(self, "W_char", None) is None:
            self.W_char = (torch.randn(256, self.feat_dim,
                                       dtype=torch.float32, device=DEVICE) * 0.05)
            self.b_char = torch.zeros(256, dtype=torch.float32, device=DEVICE)
        # ★ P10-B 首字符 WTA 头 + P10-A 行级巩固置信度 (突触保护)
        if use_char_wta and getattr(self, "W_first", None) is None:
            self.W_first = (torch.randn(256, self.feat_dim,
                                        dtype=torch.float32, device=DEVICE) * 0.05)
            self.b_first = torch.zeros(256, dtype=torch.float32, device=DEVICE)
        if use_char_wta and getattr(self, "char_row_conf", None) is None:
            self.char_row_conf = torch.zeros(256, dtype=torch.float32,
                                             device=DEVICE)
        if use_char_wta and getattr(self, "first_row_conf", None) is None:
            self.first_row_conf = torch.zeros(256, dtype=torch.float32,
                                              device=DEVICE)
        for epoch in range(n_iter):
            correct = 0
            total = 0
            for di, (inp, resp) in enumerate(dialogues):
                resp_codes = self._text_to_codes(resp)
                if not resp_codes:
                    continue
                # EOS: 目标序列 = 真实回复 + 结束符 (0 表示终止)
                tgt_codes = resp_codes if eos_code is None else resp_codes + [eos_code]
                self.clear_temporary_memory()
                self.reset_state()
                self.reset_memory()
                self._slot_fatigue_reset()  # ★ P44 句首成分疲劳重置
                self.dmd_record_trace = True
                self._dmd_reset()
                self.encode_text_lif(inp, update_memory=True, n_loops=n_loops,
                                     record_dmd=True)
                self.input_end()
                cf = self.dmd_committed_context
                if use_sentence_wta:
                    # ★ v3.6 句级 WTA 竞争读出: 离散句方向 (WTA 选句模板,
                    #   不漂移 — 海马 CA3 WTA / 基底神经节竞争选择)。
                    goal = (self._last_goal.clone()
                            if getattr(self, "_last_goal", None) is not None
                            else cf)
                    direction, winner = self._select_sentence(goal)
                    # 训练 W_sent: 目标句行强化, 误选行削弱 (RPE-Hebbian)
                    if winner != di:
                        self.W_sent[di] += lr * goal
                        self.W_sent[winner] -= lr * goal
                else:
                    direction = self.dmd_committed_direction.clone()
                self.reset_state()
                self.reset_memory()
                self._seqctx_reset()
                self.dmd_direction = direction.clone()

                # 首字符
                cf_feat = self._mem_feature(cf)
                first_in = torch.max(cf_feat, direction * dir_strength)
                # ★ P10-B 首字符 WTA 读出: 256 神经元 argmax (与 W_char
                #   同族)。OOD 输入方向异常时必落最近已学字符 (模式补全),
                #   杜绝 8bit 阈值拼出的无效字节 ('je'/'jre' 退化根因)。
                if (use_char_wta
                        and getattr(self, "W_first", None) is not None):
                    f_scores = torch.mv(self.W_first, first_in) + self.b_first
                    self._last_first_scores = f_scores
                    first_code = int(f_scores.argmax().item())
                else:
                    first_bits = self._binary_decode(self.W_ctx_to_first, first_in)
                    first_code = 0
                    for j in range(self.output_size):
                        if first_bits[j] >= 0.5:
                            first_code |= (1 << j)
                tgt_first = resp_codes[0]
                if train_first:
                    out = (torch.mv(self.W_ctx_to_first, first_in) > 0).float()
                    rpe = self._target_bits(tgt_first) - out
                    self.W_ctx_to_first.addr_(rpe, first_in, alpha=lr)
                    self.W_ctx_to_first.clamp_(-10.0, 10.0)
                    # ★ P10-B: W_first margin 斜坡 Hebbian (与 W_char 同
                    #   规则) + P10-A 行级巩固保护
                    if (use_char_wta
                            and getattr(self, "W_first", None) is not None):
                        f_t = f_scores[tgt_first].item()
                        f_p = f_scores[first_code].item()
                        f_err = min(1.0, max(0.0, 0.5 * (f_p - f_t + 1.0)))
                        if getattr(self, "first_row_conf", None) is not None:
                            if first_code == tgt_first:
                                f2m = f_scores.clone()
                                f2m[tgt_first] = -1e9
                                f_mgn = (f_scores[tgt_first] - f2m.max()).item()
                                self.first_row_conf[tgt_first] = (
                                    0.98 * self.first_row_conf[tgt_first]
                                    + 0.02 * min(1.0, max(0.0, f_mgn * 0.5)))
                            else:
                                self.first_row_conf[tgt_first] *= 0.995
                        self.W_first[tgt_first] += char_wta_lr * f_err * first_in
                        self.b_first[tgt_first] += 0.5 * char_wta_lr * f_err
                        if first_code != tgt_first:
                            f_damp = 1.0
                            if (char_row_protect
                                    and getattr(self, "first_row_conf", None) is not None):
                                f_damp = (1.0 - self.protect_strength
                                          * self.first_row_conf[first_code].item())
                            self.W_first[first_code] -= char_wta_lr * f_err * first_in * f_damp
                            self.b_first[first_code] -= 0.5 * char_wta_lr * f_err * f_damp
                if first_code == tgt_first:
                    correct += 1
                total += 1
                ch = chr(first_code) if 0 <= first_code <= 255 else '?'
                prev_char = ch

                state = cf.clone()
                prev_state = None
                for step in range(1, min(max_steps, len(tgt_codes))):
                    # 字符反馈 (teacher/self 混合, scheduled_sampling_prob)
                    # ★ P3 修复 (off-by-one): teacher 分支此前喂 tgt_codes[step]
                    #   (当前步目标字符) 而读出目标也是 tgt_codes[step] —
                    #   任务退化为"回显刚喂入的字符" (echo), 与 self 分支
                    #   (喂上一步输出, 预测当前步) 语义冲突。echo 偏置使
                    #   生成陷入重复吸引子 (hellll.../空格循环)。修正为
                    #   喂 tgt_codes[step-1] (真实上一字符) — 真正的
                    #   teacher forcing: 喂真 char(k-1) → 预测真 char(k)。
                    feed_code = tgt_codes[step - 1]
                    is_retrieval = False
                    if (prev_char != '' and
                            __import__("random").random() < scheduled_sampling_prob):
                        # 若上一步输出即可靠反馈 (防首字符错误过早污染)
                        feed_code = ord(prev_char) if 0 <= ord(prev_char) <= 255 else tgt_codes[step - 1]
                        # ★ v3.6 编码/检索分离 (Hasselmo 2005): self 反馈
                        #   = 检索相位, 抑制学习 (只读权重, 不写)。
                        is_retrieval = True
                    vec = self._char_to_8bit_bias(chr(feed_code))
                    recall = None
                    if self.use_memory_thinking:
                        output, sparse_feat, state = self._think_character(
                            vec, update_memory=True, n_loops=n_loops)
                    else:
                        output = self._multi_layer_forward(vec, n_loops=n_loops)
                        v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                        sparse_feat = self._dg_separate(v_curr)
                        self.update_coactivation(sparse_feat)
                        self._slot_fatigue_step(
                            sparse_feat, cf=cf, code=feed_code)  # ★ P46
                        recall = self.recall_from_memassoc(sparse_feat, sparse_hint=True)
                    think = self._encode_reservoir(sparse_feat)
                    # ★ P6 分级融合 + k-WTA: 旧 max-OR 并集单调增 → 2 步
                    #   饱和 (fpop 244/256, feat/h 冻结, 读出恒定 → 重复
                    #   锁定)。加权求和 + top-k 侧抑制, 各来源竞争固定
                    #   稀疏预算; v3.6 链式状态流语义保留 (chain 以 0.9
                    #   权重参与, W_chain 未训练时贡献恒 0, 不变式保持)。
                    #   与 generate_recurrent_dmd 保持逐位一致。
                    seqctx = self._seqctx_contrib(sparse_feat, step)
                    state = self._fuse_state_kwt(
                        sparse_feat, recall,
                        self._chain_transition(prev_state)
                        if prev_state is not None else None,
                        direction, seqctx=seqctx,
                        content=self._content_quota_src(cf))
                    self.MemWork = state

                    prev_dir = self.dmd_direction.clone()
                    self._dmd_step(state)
                    dir_delta = (self.dmd_direction - prev_dir).abs().mean().item()

                    # 深读出口 Hebbian 更新 (目标 = 真实下一字符/EOS)
                    feat = self._mem_feature(state)
                    tgt = self._target_bits(tgt_codes[step])
                    h = (torch.mv(self.W_seq_h, feat) > self.seq_h_thr).float()
                    # ★ v3.6 稀疏路由: direction 门控 h, 不同句子走不同的
                    #   W_seq_out 列组合 (共享权重 + 稀疏激活, 避免多句竞争)
                    if use_dir_gated_readout and direction is not None:
                        h = h * (direction > 0).float()
                    # ★ P0 方案 1: direction 监督的 W_seq_h target
                    #   方向二值化后并聗进掐字符得联合target,
                    #   破开 W_seq 退化为“字节回放”: 向字符和方向同时预测下一步
                    # ★ P0 方案 1 v2: direction 监督仅早期融合 (Wang 2021)
                    #   起始意图引导开头, 中后期字符反馈主导。
                    #   2026减免: 恒定 direction 位淹没后续字符差异
                    #   → W_seq_h 学成“方向→固定前缀”回放。
                    if dir_supervised_h and direction is not None \
                            and step <= dir_supervised_early_steps:
                        xn = torch.max(sparse_feat.float(),
                                       (direction > 1e-6).float())
                    else:
                        xn = sparse_feat.float()
                    rpe_h = xn - h
                    if use_char_wta:
                        # ★ P1: argmax 字符竞争替代独立 bit 阈值读出
                        # ★ P7: 与 _seq_predict 一致, 直接从 feat 读出,
                        #   保留 chain/direction 位置位 (双写字母可分)
                        scores = torch.mv(self.W_char, feat) + self.b_char
                        wta_code = int(scores.argmax().item())
                        out = torch.zeros(self.output_size, dtype=torch.float32,
                                          device=feat.device)
                        for j in range(self.output_size):
                            if (wta_code >> j) & 1:
                                out[j] = 1.0
                    else:
                        out = (torch.mv(self.W_seq_out, h) + self.b_seq > 0).float()
                    rpe = tgt - out
                    # ★ v3.6 编码/检索分离 (Hasselmo 2005): 检索相位
                    #   (self 反馈) 抑制学习, 只读权重不写 — 防 free-run
                    #   漂移状态污染权重 (死亡螺旋的生物解法)。
                    if not (encode_retrieve_sep and is_retrieval):
                        self.W_seq_h.addr_(rpe_h, feat, alpha=lr_h)
                        if confidence_gate and direction is not None:
                            d_act = (direction > 0).float()
                            d_norm = d_act.sum().clamp_min(1.0)
                            conf = ((state * d_act).sum() / d_norm).clamp(0.0, 1.0)
                            self.W_seq_out.addr_(rpe, h, alpha=lr * conf)
                        else:
                            self.W_seq_out.addr_(rpe, h, alpha=lr)
                    out_code = 0
                    for j in range(self.output_size):
                        if out[j] >= 0.5:
                            out_code |= (1 << j)
                    if out_code == tgt_codes[step]:
                        correct += 1
                    total += 1
                    # ★ P1 字符级 WTA 学习: margin 斜坡误差驱动的局部
                    #   Hebbian (VOWEL 2020 — 局部 pre/post + 误差调制,
                    #   无梯度/BP)。误判时强化目标字符行、削弱误判行;
                    #   误差随得分差连续变化 (胜者压制越深 → 误差越大)。
                    if (use_char_wta
                            and not (encode_retrieve_sep and is_retrieval)):
                        # ★ P10-A 行级巩固置信度: 目标行胜出时按 margin
                        #   EMA 累积 (巩固 = 高置信), 落败时缓慢遗忘。
                        #   char_row_protect=True 时抑制更新对高置信行
                        #   衰减 — v2.4 突触保护思想移植到字符竞争层,
                        #   防 Phase B 统计训练覆盖 Phase A 已巩固响应
                        #   (P9 实证的灾难性干扰)。
                        t_row = tgt_codes[step]
                        if getattr(self, "char_row_conf", None) is not None:
                            if out_code == t_row:
                                s2m = scores.clone()
                                s2m[t_row] = -1e9
                                mgn = (scores[t_row] - s2m.max()).item()
                                self.char_row_conf[t_row] = (
                                    0.98 * self.char_row_conf[t_row]
                                    + 0.02 * min(1.0, max(0.0, mgn * 0.5)))
                            else:
                                self.char_row_conf[t_row] *= 0.995
                        if out_code != t_row:
                            s_t = scores[t_row].item()
                            s_p = scores[out_code].item()
                            err = min(1.0, max(0.0, 0.5 * (s_p - s_t + 1.0)))
                            self.W_char[t_row] += char_wta_lr * err * feat
                            self.b_char[t_row] += 0.5 * char_wta_lr * err
                            damp = 1.0
                            if (char_row_protect
                                    and getattr(self, "char_row_conf", None) is not None):
                                damp = (1.0 - self.protect_strength
                                        * self.char_row_conf[out_code].item())
                            self.W_char[out_code] -= char_wta_lr * err * feat * damp
                            self.b_char[out_code] -= 0.5 * char_wta_lr * err * damp
                    next_ch = chr(out_code) if 0 <= out_code <= 255 else '?'
                    prev_char = next_ch

                    # 独立结束读出器训练: 最后一个真实字符状态 → done=1
                    if train_done and not (encode_retrieve_sep and is_retrieval):
                        done_out = (torch.mv(self.W_done, feat) > 0).float()
                        done_target = 1.0 if step == len(resp_codes) - 1 else 0.0
                        rpe_done = done_target - done_out
                        self.W_done.addr_(rpe_done, feat, alpha=lr)

                    if len(tgt_codes) > 2 and step >= 2 and dir_delta <= 0.01:
                        # DMD 方向收敛但句未生成完整, 仍继续 (不提前 break)
                        pass

                    # ★ v3.6 非对称链式状态流训练: STDP pre=prev_state →
                    #   post=state (有向, 非对称)。W_chain[post, pre] 累积
                    #   "上一步→当前步" 的共激活, 塑造顺序敏感连接 (MPN)。
                    if (train_chain and prev_state is not None
                            and not (encode_retrieve_sep and is_retrieval)):
                        self.W_chain.addr_(state.float(), prev_state.float(),
                                           alpha=chain_lr)
                    prev_state = state.clone()

                self.W_seq_h.clamp_(-10.0, 10.0)
                self.W_seq_out.clamp_(-10.0, 10.0)
                self.W_done.clamp_(-10.0, 10.0)
                self.W_chain.clamp_(-10.0, 10.0)
                if use_char_wta and getattr(self, "W_char", None) is not None:
                    self.W_char.clamp_(-10.0, 10.0)
                    self.b_char.clamp_(-10.0, 10.0)
                if use_sentence_wta and getattr(self, "W_sent", None) is not None:
                    self.W_sent.clamp_(-10.0, 10.0)

                # CLS 互补学习系统: 每句结束后短期→长期巩固, 防多句遗忘
                if consolidate_long:
                    self.consolidate_coactivation()

            acc = correct / max(1, total)
            self._last_rollout_acc = acc  # ★ P5: 末轮 (非 best) 准确率
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % lr_decay_step == 0:
                lr *= lr_decay_factor
                lr_h = lr * 0.5
            if early_stop_patience is not None and no_improve >= early_stop_patience:
                break

        return best_acc

    # ==================== v2.6 前瞻编码 (Brea 2016) ====================
    # 内部预测回路: W_prosp 预测"自身下一步发放位型"。
    # 学习规则与 W_seq 同族 (Δw = lr×RPE×pre), 但目标不是外部字符
    # 答案, 而是下一时刻的工作记忆发放位型 (自监督, 无外部依赖) →
    # 生成时可用来做"猜-查"自检 (Ororbia 2019 SNPC)。

    def _ensure_prospective_state(self):
        """防御式初始化 — 旧模型 (v2.5 及更早) 缺 v2.6 属性时补齐"""
        if not hasattr(self, "prospective"):
            self.prospective = False
            self.W_prosp = torch.zeros(self.feat_dim, self.feat_dim,
                                       dtype=torch.float32, device=DEVICE)
            self.W_prosp.uniform_(-0.1, 0.1)

    def _prospective_predict(self, state):
        """前瞻预测: 当前状态 → 预期下一步发放位型 (二值)

        纯二值阈值解码 (与 W_seq/W_ctx_to_pos 同族), 无连续数值。
        """
        self._ensure_prospective_state()
        feat = self._mem_feature(state)
        return self._binary_decode(self.W_prosp, feat)

    def _prospective_raw(self, state):
        """前瞻预测实值投影 (只读, 供 DMD 预测残差使用)

        保留 W_prosp 的连续预测强度 (O(N²) 只读矩阵乘), 不产生学习
        信号 — 仅供状态残差计算, 不参与任何 Hebbian 更新。
        """
        self._ensure_prospective_state()
        feat = self._mem_feature(state)
        return torch.clamp(torch.mv(self.W_prosp, feat), 0.0, 1.0)

    def train_prospective(self, dialogues, lr=0.1, n_iter=5, n_loops=1):
        """★ v2.6 前瞻编码训练 — 神经元预测自身未来发放 (Brea 2016)

        学习规则: Δw = lr × RPE_j × pre_i
          - target = 下一时刻工作记忆状态位型 (states[i+1] > 0.5)
          - pred = W_prosp 对当前状态的预测 (二值阈值解码)
          - RPE = target − pred ∈ {−1,0,+1} (离散奖赏, 红线合规)

        与 W_seq 训练的关键区别:
          - W_seq 目标是"外部字符答案" (teacher forcing)
          - W_prosp 目标是"自身下一步状态" (内部自监督) — 神经元匹配
            自己预期的未来发放率, 不依赖外部答案 → 自生成驱动。

        Args:
            dialogues: 对话对列表 [(inp, resp), ...]
            lr: 学习率
            n_iter: 训练迭代次数
            n_loops: 每次输入自回归前向轮数

        Returns:
            best_acc: 最佳预测准确率 (状态位型逐位一致比例)
        """
        seq_data = []
        for inp, resp in dialogues:
            # ★ v3.0: 彻底清空长期记忆，防止无关的独立训练语料污染当前对话上下文 (状态崩塌)
            self.clear_long_term_memory()

            resp_codes = self._text_to_codes(resp)
            if len(resp_codes) < 2:
                continue
            _, states, _ = self.encode_text_lif_states(
                resp, update_memory=True, n_loops=n_loops)
            if len(states) < 2:
                continue
            for i in range(len(states) - 1):
                # 下一时刻状态位型 (二值化)
                target = (states[i + 1] > 0.5).float()
                seq_data.append((states[i], target))

        if not seq_data:
            return 0.0

        n_data = len(seq_data)
        # 度量: 逐位一致率 (per-bit) — 256 维位型全一致概率 ≈ 0,
        # 与 train_sequence (8 位) 不同, 全位一致无区分度; per-bit 一致
        # 反映"预测位型接近实际位型"的程度 (自检 Jaccard 同族度量)。
        def _per_bit_acc(data):
            total = 0.0
            for fr, target in data:
                pred = self._prospective_predict(fr)
                total += (pred == target).float().mean().item()
            return total / len(data)

        best_acc = _per_bit_acc(seq_data)

        for _ in range(n_iter):
            random.shuffle(seq_data)
            for fr, target in seq_data:
                feat = self._mem_feature(fr)
                pred = self._binary_decode(self.W_prosp, feat)
                rpe = target - pred  # {−1,0,+1}
                self._hebbian_step(self.W_prosp, rpe, feat, pred, lr)
            acc = _per_bit_acc(seq_data)
            if acc > best_acc:
                best_acc = acc
        return best_acc

    # ==================== v2.6 自检回路 (Ororbia 2019 SNPC 猜-查) ====================

    def generate_selfchecked(self, context_state, n_steps=30, max_repeat=3,
                             update_memory=True, use_pos_memory=True,
                             n_loops=1, event_guide=None,
                             n_candidates=4, prospect_w=1.0,
                             proto_fn=None, gate_threshold=1.0):
        """生成 + 自检回路 — 每步多候选 + 评估择优 (Ororbia 猜-查)

        流程 (每步):
          1. 候选字符集: W_seq 候选 (含位置头/事件修正) + 对候选码
             翻转单个 bit 的 n_candidates−1 个扰动邻居 → "思考备选"
          2. 对每个候选字符评估, 择优:
             a) proto_fn 给定 (统计原型引导, 推荐): 候选与海马统计
                原型在该位置的期望字符一致性 (Spens & Burgess 2024
                "海马重放 = 生成模型") — 有判别力 (实测 1.0 vs 0.0)
             b) 否则前瞻预测 (Brea 2016): 候选生成的下一步状态与
                W_prosp 预期位型的共发放 Jaccard
          3. 选择分数最高的候选字符 → 生成

        红线合规:
          - W_prosp 是固定联想矩阵的线性预测 (非学习), 无连续数值
          - 候选扰动 = 二值 bit 翻转 (离散)
          - 一致性评分 = 共发放比例 (布尔运算)
          - 全部 Opt-in (prospective=False / 未训练时行为不变)

        Args:
            context_state: 工作记忆累积上下文状态 (256-dim, ∈ [0, 1])
            n_steps: 最大生成字符数
            max_repeat: 最大连续重复数
            update_memory: 是否更新关联记忆 W_coact
            use_pos_memory: 是否启用位置记忆头修正
            n_loops: 自回归循环轮数
            event_guide: 可选外部事件记忆引导 (同 generate_recurrent)
            n_candidates: 每步候选数 (1 = 无扰动, 等价 generate_recurrent)
            prospect_w: 前瞻一致性权重 (>=1 时强制优先前瞻; 保留扩展)
            proto_fn: ★ v2.6 统计原型引导函数 (推荐) — 调用
                proto_fn(step) → 该位置统计期望字符码或 None。
                传入海马 recall_prototype 时, 自检 = 统计生成模型
                择优 (Spens & Burgess 2024), 判别力远优于前瞻预测。
                ★ v2.8 置信门控: proto_fn 也可返回 (code, margin) 元组
                (margin = argmax 分数 − 第二高分, 来自 recall_prototype_margin)。
                margin ≥ gate_threshold → 原型 1.0 主导 (强共识);
                margin < gate_threshold → 原型分 < 主候选 0.5, 让位给
                W_seq 组合主候选 (弱共识 = 统计噪声, 不强制模板化)。
            gate_threshold: ★ v2.8 原型置信门控阈值 (margin 归一化除
                数)。默认 1.0 (实测: 编码结构位 margin 1.3-2.5 强,
                内容噪声位 0.01-0.42 弱, 分离明显)。仅 proto_fn 返回
                margin 时生效; 返回纯 code 时原型恒 1.0 (向后兼容)。

        Returns:
            result: 生成文本
        """
        cf = (torch.tensor(context_state, dtype=torch.float32, device=DEVICE)
              if not torch.is_tensor(context_state) else context_state)

        self.reset_state()
        self.reset_memory()

        result = []
        repeat_count = 0
        prev_char = ''
        seen_patterns = set()

        # 首字符 (与 generate_recurrent 一致)
        cf_feat = self._mem_feature(cf)
        first_bits = self._binary_decode(self.W_ctx_to_first, cf_feat)
        first_code = 0
        for j in range(self.output_size):
            if first_bits[j] >= 0.5:
                first_code |= (1 << j)
        ch = chr(first_code) if 0 <= first_code <= 255 else '?'
        result.append(ch)
        prev_char = ch

        if n_steps <= 1:
            return ''.join(result)

        state = cf.clone()
        for step in range(1, n_steps):
            vec = self._char_to_8bit_bias(ch)
            output = self._multi_layer_forward(vec, n_loops=n_loops)

            if update_memory:
                self.update_coactivation(output)
            recall = self.recall_from_memassoc(output)

            v_peak = self.V_deep[-1] if self.num_layers > 1 else self.V
            state = v_peak
            self.MemWork = state

            # 主候选: W_seq + 位置头/事件修正 (原逻辑)
            next_bits = self._binary_decode(self.W_seq, self._mem_feature(state), self.b_seq)
            main_code = 0
            for j in range(self.output_size):
                if next_bits[j] >= 0.5:
                    main_code |= (1 << j)
            mem_override = False  # ★ v2.6: 位置头/事件是记忆修正, 记忆优先
            if event_guide is not None:
                g_code = event_guide(step)
                if g_code is not None:
                    main_code = g_code
                    mem_override = True
            elif use_pos_memory and self.W_ctx_to_pos:
                mem_code, margin = self.pos_head_recall(cf, step)
                if mem_code is not None and margin > 0.0:
                    main_code = mem_code
                    mem_override = True

            if mem_override:
                # 记忆优先: 位置头/事件修正直接采用, 不进入评估竞争
                # (防止候选把精确记忆拉偏 — 自检只服务开放生成)
                best_code = main_code
            else:
                # 自检回路: 候选集 = W_seq 主候选 + bit 翻转扰动
                candidates = {main_code}
                for k in range(max(n_candidates - 1, 0)):
                    flip = 1 << ((step + k) % self.output_size)
                    candidates.add(main_code ^ flip)

                # ★ 评估器: 优先统计原型 (proto_fn), 否则前瞻预测
                proto_code = proto_fn(step) if proto_fn is not None else None
                proto_margin = None
                if isinstance(proto_code, tuple):
                    # ★ v2.8 置信门控: proto_fn 返回 (code, margin)
                    proto_code, proto_margin = proto_code
                if proto_code is not None:
                    # 原型字符是"统计期望"候选 — 加入候选集使其可被选中
                    candidates.add(proto_code)

                best_code, best_score = main_code, -1.0
                for cand in candidates:
                    if not (0 <= cand <= 255):
                        continue
                    if proto_code is not None:
                        # 统计原型引导: 原型字符最高优先, 主候选次之,
                        # 扰动最低 (保留联想变化, 避免完全模板化)
                        if cand == proto_code:
                            if proto_margin is None:
                                # 向后兼容: 无 margin 时原型恒 1.0 主导
                                score = 1.0
                            else:
                                # ★ v2.8 置信门控: margin 归一化 —
                                # 强共识 (≥ gate_threshold) 原型 1.0,
                                # 弱共识原型分 < 主候选 0.5 → 让位
                                # 给 W_seq 组合主候选 (弱共识 = 统计
                                # 噪声, 不强制模板化)
                                score = min(1.0, proto_margin / gate_threshold)
                        elif cand == main_code:
                            score = 0.5
                        else:
                            score = 0.0
                    else:
                        # 前瞻预测 (Brea 2016): 候选生成状态的 Jaccard
                        save_V = self.V.clone()
                        save_Vd = [v.clone() for v in self.V_deep]
                        save_mem = self.MemWork.clone()
                        cand_vec = self._char_to_8bit_bias(chr(cand))
                        cand_out = self._multi_layer_forward(cand_vec, n_loops=n_loops)
                        cand_recall = self.recall_from_memassoc(cand_out)
                        cand_peak = self.V_deep[-1] if self.num_layers > 1 else self.V
                        s_next = torch.max(cand_peak, cand_recall)
                        # 恢复状态
                        self.V = save_V
                        self.V_deep = save_Vd
                        self.MemWork = save_mem

                        pred = self._prospective_predict(state)
                        sn_bits = (s_next > 0.5).float()
                        inter = (pred * sn_bits).sum().item()
                        union = ((pred + sn_bits) > 0).sum().item()
                        score = inter / union if union > 0 else 0.0
                    if score > best_score:
                        best_score, best_code = score, cand

                if proto_code is not None:
                    # ★ 统计原型引导下: 若无候选命中原型, 且主候选分数
                    #   全 0, 则保持主候选 (不强制替换成原型 — 保留
                    #   生成多样性, 避免退化为统计平均模板)
                    pass

            next_ch = chr(best_code) if 0 <= best_code <= 255 else '?'

            if next_ch == prev_char:
                repeat_count += 1
            else:
                repeat_count = 0
            if repeat_count >= max_repeat:
                break

            if len(result) >= 3:
                pattern = ''.join(result[-2:]) + next_ch
                if pattern in seen_patterns:
                    break
                seen_patterns.add(''.join(result[-3:]) + next_ch)

            result.append(next_ch)
            prev_char = next_ch
            ch = next_ch

        return ''.join(result)

    # ==================== 层间联合训练 (v10) ====================

    def _init_identity_layer(self, layer_idx, scale=1.0, noise=0.0):
        """★ v12: 将第 layer_idx 个层间权重初始化为接近恒等映射 (中继)

        恒等初始化: W ≈ scale × I + 小噪声
          - out = threshold(W·in + V) ≈ in → 信息不丢失 (纯中继)
          - 渐进式深度训练中, 新加入的层以"中继"身份开始,
            多巴胺奖赏调制再逐步塑造其连接模式。
          - 该层掩码置为全 1 (全连接), 保证恒等不被稀疏掩码破坏。

        ★ v12.1 噪声修正: noise 必须很小 (≤0.01)。
          非对角噪声项在 ~128 个活跃输入上累积 (std ≈ noise×√128),
          noise=0.05 时累积 std≈0.57, 超过阈值 0.5 → 随机发放,
          恒等中继失效 (实测阶段 2 初始解码仅 6/72)。noise=0.01 时
          累积 std≈0.11, 不越阈值, 恒等保真。
        """
        W = torch.eye(self.hidden_size, dtype=torch.float32, device=DEVICE) * scale
        if noise > 0:
            W = W + torch.randn_like(W) * noise
        self.W_deep[layer_idx] = W
        self.deep_masks[layer_idx] = torch.ones(
            self.hidden_size, self.hidden_size, dtype=torch.float32, device=DEVICE)

    def train_multi_layer_stdp(self, train_codes, num_epochs=200,
                               lr_layer=0.3, lr_out=0.5, verbose=True,
                               n_loops=1):
        """★ v12: 渐进式深度训练 — 逐层加深的课程训练 (用户指定方案)

        ★ v13.1: n_loops 自回归循环 (透传给 _multi_layer_forward_all),
          训练与推理/编码保持一致的循环次数, 默认 1。

        ★ 核心思想: 不一次性端到端训练四层 (实测层间无监督 Hebbian 会
          破坏信息保留: L4 解码从 2/72 退化到 1/72), 而是从浅层开始
          逐层加深, 每阶段打好基础:

          阶段 1: 输入→L1→输出    (训练 W_h2o 从 L1 输出解码)
          阶段 2: 输入→L1→L2→输出  (新层 W_deep[0] 恒等初始化, 训练 + W_h2o)
          阶段 3: 输入→L1→L2→L3→输出
          阶段 4: 输入→L1→L2→L3→L4→输出 (完整底模)

        ★ 每阶段新层间权重以"恒等映射"初始化 (W ≈ I, _init_identity_layer):
          - 新层输出 ≈ 输入 (纯中继) → 信息不丢失
          - 每阶段从"上一阶段已学好的浅层解"继续, 深度增长不破坏已学表示
          - 生物学依据: 新突触从中性连接开始, 多巴胺奖赏调制逐步塑造

        ★ 学习规则 (与 v11 一致, 奖赏预测误差调制 Hebbian):
          - W_h2o (从汇聚层输出解码): Δw = lr_out × RPE_j × out_peak
          - 层间: ΔW_lk = lr_layer × max(mean_RPE, 0) × out_{k+1} ⊗ out_k
          - 无梯度 / 反向传播 / 批量优化 / 目标误差

        Args:
            train_codes: ASCII 码列表 (字符集)
            num_epochs: 总训练轮数 (均分到每个深度阶段)
            lr_layer: 层间权重学习率
            lr_out: 解码层 (W_h2o) 学习率
            verbose: 是否打印进度
        """
        if verbose:
            print(f"\n--- [v12 渐进式深度训练] 逐层加深 L1→...→L{self.num_layers} + W_h2o ---")
        t0 = time.perf_counter()

        n_vocab = len(train_codes)
        output_size = self.output_size
        n_depth = self.num_layers

        # 目标: 8-bit 编码 (仅用于计算奖赏信号)
        targets_gpu = torch.zeros(n_vocab, output_size, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(train_codes):
            for j in range(output_size):
                targets_gpu[i, j] = float((c >> j) & 1)

        # 输入: 结构化随机稠密编码 (二值)
        input_vecs_gpu = torch.zeros(n_vocab, self.hidden_size, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(train_codes):
            ch = chr(c) if 0 <= c <= 255 else '?'
            input_vecs_gpu[i] = self._get_char_code(ch)

        def eval_depth(depth):
            """评估当前深度 (前 depth 层) 下汇聚层输出的解码准确率"""
            ok = 0
            for i, c in enumerate(train_codes):
                self.reset_state()
                out = self._multi_layer_forward(input_vecs_gpu[i], active_depth=depth,
                                                n_loops=n_loops)
                if self.check_decode(out, c):
                    ok += 1
            return ok

        # 每阶段轮数: 总轮数均分到各深度
        epochs_per_depth = max(30, num_epochs // n_depth)

        for depth in range(1, n_depth + 1):
            # 新加入的层间权重: 恒等初始化 (纯中继, 不破坏已学信息)
            if depth > 1:
                self._init_identity_layer(depth - 2)

            init_acc = eval_depth(depth)
            if verbose:
                print(f"  [阶段 {depth}/{n_depth}] 输入→L1→...→L{depth}→输出: "
                      f"初始解码 {init_acc}/{n_vocab}", flush=True)

            for epoch in range(epochs_per_depth):
                # ★ 随机打乱训练顺序 — 模拟生物学学习的不确定性
                indices = list(range(n_vocab))
                random.shuffle(indices)

                correct_tensor = torch.zeros(1, dtype=torch.float32, device=DEVICE)
                for idx in indices:
                    vec = input_vecs_gpu[idx]
                    target = targets_gpu[idx]

                    # 1. 重置所有层膜电位 (每个样本独立)
                    self.reset_state()

                    # 2. 前 depth 层前向 → 汇聚层输出 = outs[-1]
                    outs = self._multi_layer_forward_all(vec, active_depth=depth,
                                                         n_loops=n_loops)
                    out_peak = outs[-1]

                    # 3. W_h2o 解码汇聚层输出 (纯二值阈值)
                    pred = self._binary_decode(self.W_h2o, out_peak, self.b_o)

                    # 逐位奖赏预测误差 (v11): RPE_j = target_j − pred_j ∈ {−1, 0, +1}
                    pred_bits = (pred > 0.5).float()
                    target_bits = (target > 0.5).float()
                    rpe = target_bits - pred_bits
                    # 统计张量累积 (无 .item() 同步, epoch 末一次性转 int)
                    correct_tensor += (pred_bits == target_bits).all().float()

                    # 标量奖赏预测误差 (层间全局投射): 张量级, 无同步
                    mean_rpe = rpe.mean()

                    # 4a. 更新 W_h2o — 逐位奖赏预测误差调制 Hebbian (从汇聚层输出学习)
                    #     Δw = lr_out × RPE_j × out_peak (addr_ 单内核等价 8 次循环)
                    #     ★ v11: RPE 直接门控可塑性, "应发未发"获得强化路径
                    self.W_h2o.addr_(rpe, out_peak, alpha=lr_out)
                    self.W_h2o.clamp_(-10.0, 10.0)

                    # 4b. 更新已激活的层间权重 — 标量奖赏预测误差调制 Hebbian
                    #     ΔW_lk = lr_layer × max(mean_RPE, 0) × out_{k+1} ⊗ out_k
                    #     ★ v11 奖赏门控 (只强化不惩罚): mean_rpe<=0 时更新量 0,
                    #       等价旧 `if mean_rpe > 0` 分支且无 .item() 同步
                    mean_rpe_pos = torch.clamp(mean_rpe, min=0.0)
                    for m in range(depth - 1):
                        self.W_deep[m] += lr_layer * mean_rpe_pos * torch.outer(outs[m + 1], outs[m])
                correct_count = int(correct_tensor.item())
                for m in range(depth - 1):
                    self.W_deep[m].clamp_(-10.0, 10.0)

                if (epoch + 1) % 50 == 0 or correct_count == n_vocab:
                    if verbose:
                        print(f"    epoch {epoch+1}: acc={correct_count}/{n_vocab} "
                              f"({correct_count/n_vocab:.1%})", flush=True)
                    if correct_count == n_vocab:
                        break

            final_acc = eval_depth(depth)
            if verbose:
                print(f"  [阶段 {depth}] 完成: 解码 {final_acc}/{n_vocab}", flush=True)

        if verbose:
            print(f"  渐进式深度训练完成: {time.perf_counter() - t0:.1f}s "
                  f"(每阶段 {epochs_per_depth} epochs)", flush=True)

    # ==================== 循环生成 ====================

    def generate_recurrent(self, context_state, n_steps=30, max_repeat=3,
                           update_memory=True, use_pos_memory=True,
                           pos_margin_thresh=0.0, n_loops=1, event_guide=None,
                           goal=None):
        """多层 0-1 膜电位循环生成，工作记忆层 MemWork + W_seq 字符预测

        ★ 生成流程 (v10 — 4 层隐藏层 + 工作记忆层 + 关联记忆层):
          1. W_ctx_to_first(MemWork) → 首字符 (纯二值阈值解码)
          2. 重置所有层膜电位 + 工作记忆层
          3. 首字符 → 随机稠密编码 → 4 层前向 → L4 输出
          4. 关联记忆更新/回忆 + MemWork 累积: max(V_LN, recall, 旧×遗忘)
          5. W_seq(MemWork) → 下一字符 (纯二值阈值解码)
          6. 重复 3-5

        ★ v13 位置记忆头修正 (对非首字结果进行修正):
          每步生成时, 位置记忆头 W_ctx_to_pos[step] 对上下文状态 cf
          回忆"回复第 step 字符"并直接覆盖 W_seq 候选 (记忆优先)。
          ★ experiment14: 位置头回忆 96.6% vs W_seq ~5-40%;
            margin 门控无判别力 (正确回忆 med=0.09 vs 错误 med=0.05,
            重叠严重) → 默认 pos_margin_thresh=0.0 (无门控, 全量覆盖)。
            门控阈值 > 0 反而回退到 W_seq 烂输出 (θ=0.5 → 28.6%)。
            超出已训练位置时回退 W_seq (保留对未见输入的生成能力)。

        Args:
            context_state: 工作记忆累积上下文状态 (256-dim, ∈ [0, 1])
            n_steps: 最大生成字符数
            max_repeat: 最大连续重复数
            use_pos_memory: 是否启用位置记忆头修正
            pos_margin_thresh: 修正门控阈值 (默认 0 = 记忆优先全量覆盖;
              实验表明 margin 无判别力, 不建议调高)
            event_guide: 可选外部事件记忆引导 (v14.1 P1 集成) —
              callable(step) → 外部记忆候选字符码或 None (None 回退
              位置头/W_seq)。事件记忆优先于位置头, 用于大库逐字对话。

        Returns:
            result: 生成文本
        """
        cf = (torch.tensor(context_state, dtype=torch.float32, device=DEVICE)
              if not torch.is_tensor(context_state) else context_state)

        # ★ v14.12: 目标意图注入 — 未显式传入时回退到 _last_goal (仅输入语义)
        # ★ v3.2: 提前解析, 供首字符专家路由使用
        if goal is None:
            goal = getattr(self, "_last_goal", None)

        # ★ v10: 重置所有层膜电位 + 工作记忆层 — 新生成从头开始
        self.reset_state()
        self.reset_memory()

        result = []
        repeat_count = 0
        prev_char = ''
        seen_patterns = set()

        # Step 1: W_ctx_to_first → 首字符 (纯二值阈值解码)
        # ★ v14: DG 稀疏分离 (与训练一致)
        cf_feat = self._mem_feature(cf)
        if getattr(self, "use_experts", False) and self.expert_first:
            e = self._expert_route(goal if goal is not None else cf)
            if e >= 0:
                self._activate_expert_heads(e)
                pos_goal = getattr(self, "_last_pos_goal", None)
                expert_feat = self._mem_feature(
                    pos_goal if pos_goal is not None else cf)
                first_bits = self._binary_decode(
                    self.expert_first[e], expert_feat, self.expert_first_b[e])
            else:
                first_bits = self._binary_decode(self.W_ctx_to_first, cf_feat)
        else:
            first_bits = self._binary_decode(self.W_ctx_to_first, cf_feat)
        first_code = 0
        for j in range(self.output_size):
            if first_bits[j] >= 0.5:  # first_bits ∈ {0.0, 1.0}
                first_code |= (1 << j)

        ch = chr(first_code) if 0 <= first_code <= 255 else '?'
        result.append(ch)
        prev_char = ch

        if n_steps <= 1:
            return ''.join(result)

        # Step 2+: 循环生成
        # ★ v10: 工作记忆层累积 (0-1 分级电位)
        state = cf.clone()
        for step in range(1, n_steps):
            # ★ v10: 首字符 → 随机稠密编码 → 4 层前向 → L4 输出
            #   (★ v13.1: n_loops 自回归循环一次再传入输入)
            vec = self._char_to_8bit_bias(ch)
            if self.use_memory_thinking:
                output, sparse_feat, state = self._think_character(
                    vec, update_memory=update_memory, n_loops=n_loops)
            else:
                output = self._multi_layer_forward(vec, n_loops=n_loops)  # ★ v10: 4 层前向

                # ★ v10: 关联记忆层更新 (共发放追踪) — 先学习(写)再回忆(读)
                # ★ v12.2 顺序回退: 与 encode_text_lif 一致 (实验9 now 模式)
                # ★ v12.3: update_memory=False 时冻结 W_coact (评估/生成一致性)
                v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                sparse_feat = self._dg_separate(v_curr)

                if update_memory:
                    self.update_coactivation(sparse_feat)
                recall = self.recall_from_memassoc(sparse_feat, sparse_hint=True)
                state = torch.max(sparse_feat, recall)
            # ★ v14.12: 融合 循环储层思考 (B) + 目标意图注入 (A)
            think = self._encode_reservoir(sparse_feat)
            state = self._guided_state(state, goal, think)
            self.MemWork = state

            # ★ W_seq(工作记忆) → 下一字符 (纯二值阈值解码)
            # ★ v14: DG 稀疏分离 (与 train_sequence 一致)
            # ★ v14.12: 深度读出口 (预测编码隐藏层 + 输出层)
            next_bits = self._seq_predict(self._mem_feature(state))
            next_code = 0
            for j in range(self.output_size):
                if next_bits[j] >= 0.5:  # next_bits ∈ {0.0, 1.0}
                    next_code |= (1 << j)

            # ★ v13 位置记忆头修正 (非首字): 记忆确信时覆盖 W_seq 候选
            #   step 从 1 起 → 对应"回复第 step 字符" (位置 0 首字符由
            #   W_ctx_to_first 处理, 不在此修正)。margin 门控保留泛化:
            #   记忆头对未见输入 margin 低 → 回退 W_seq。
            # ★ v14.1 事件记忆引导 (P1 集成): event_guide 优先于位置头,
            #   再回退 W_seq — 大库记忆 (experiment18) 逐字对话引导。
            if event_guide is not None:
                g_code = event_guide(step)
                if g_code is not None:
                    next_code = g_code
            elif use_pos_memory and (
                    self.W_ctx_to_pos
                    or getattr(self, "expert_pos", None)):
                # ★ v3.3 修复: 专家模式下位置头在 expert_pos 中训练,
                # 共享 W_ctx_to_pos 为空 → 旧条件恒 False → 修正从不触发。
                # pos_head_recall 内部已按 use_experts 分支取 expert_pos。
                mem_code, margin = self.pos_head_recall(cf, step)
                if mem_code is not None and margin > pos_margin_thresh:
                    next_code = mem_code

            next_ch = chr(next_code) if 0 <= next_code <= 255 else '?'

            # 防重复
            if next_ch == prev_char:
                repeat_count += 1
            else:
                repeat_count = 0
            if repeat_count >= max_repeat:
                break

            # 防循环
            if len(result) >= 3:
                pattern = ''.join(result[-2:]) + next_ch
                if pattern in seen_patterns:
                    break
                seen_patterns.add(''.join(result[-3:]) + next_ch)

            result.append(next_ch)
            prev_char = next_ch
            ch = next_ch

        return ''.join(result)

    # ==================== v3.4 DMD 动态意义方向 (D1/D2) ====================

    def generate_recurrent_dmd(self, context_state, n_steps=30, max_repeat=3,
                               update_memory=True, n_loops=1,
                               direction=None, dir_strength=1.0,
                               min_steps=3, done_stable_steps=3,
                               done_delta_thresh=0.01,
                               disable_early_stop=False, eos_code=None,
                               use_done_signal=False, dir_anchor=False,
                               use_dir_gated_readout=False,
                               use_char_wta=False, eos_margin=0.0,
                               char_sfa_eos=0.0, sfa_eos_char=0):
        """★ v3.4 D2: 方向状态驱动的闭环字符生成原型

        与 generate_recurrent 的关键区别 (对应 3.4 文档 §4.4):
          - 首字符只读取 response_direction + 当前动态状态,
            不使用 expert_pos[e][k] / W_ctx_to_pos 位置头;
          - 每个输出字符重新反馈到网络 (逐字状态演化);
          - 终止由 response_done 信号触发 (方向状态收敛),
            不读取目标答案长度/位置。

        Args:
            context_state: 工作记忆累积上下文状态
            n_steps: 最大生成字符数 (硬上限, 防御性)
            max_repeat: 最大连续重复数
            update_memory: 是否更新关联记忆
            n_loops: 每步自回归前向轮数
            direction: 回答意义方向 (默认取 self.dmd_direction;
               缺失时回退 _last_goal; 仍缺失则回退 context_state)
            dir_strength: 方向注入强度 (0-1 乘到 state 上做 max 融合)
            min_steps: response_done 最早触发步数 (防过短回复)
            done_stable_steps: 方向持续稳定步数 (收敛判定)
            done_delta_thresh: 方向变化量阈值 (低于即视为稳定)

        Returns:
            result: 生成文本 (未显式依赖位置/答案长度)
        """
        cf = (torch.tensor(context_state, dtype=torch.float32, device=DEVICE)
              if not torch.is_tensor(context_state) else context_state)

        if direction is None:
            direction = getattr(self, "dmd_committed_direction", None)
        if direction is None:
            direction = getattr(self, "dmd_direction", None)
        if direction is None:
            direction = getattr(self, "_last_goal", None)
        if direction is None:
            direction = cf
        direction = direction.detach().clone()

        self.reset_state()
        # ★ P2 训练/生成状态一致性修复: train_dmd_rollout 在 encode 后执行
        #   第二次 reset_memory() (清 W_coact_temp/_coact_trace/_conj_ctx/
        #   MemWork), 生成路径此前只 reset_state() — 输入编码
        #   (encode_text_lif update_memory=True) 写入的共激活痕迹残留,
        #   生成第 1 步 recall_from_memassoc 检索到输入模式关联,
        #   state 被污染后经 update_coactivation 自我强化 → 饱和退化
        #   (hellll.../空格循环)。此处与训练循环对齐。
        self.reset_memory()
        self._slot_fatigue_reset()  # ★ P44 句首成分疲劳重置
        if getattr(self, "dmd_direction", None) is None:
            self._dmd_reset()
        self.dmd_direction = direction.clone()
        self._seqctx_reset()

        result = []
        repeat_count = 0
        prev_char = ''
        seen_patterns = set()
        stable_streak = 0
        self._dmd_done = False
        recent_ex = 0.0  # ★ v3.7 短记忆: 终结符连续活动计数 (EOS 脉冲)

        # 首字符: 只读取方向状态 + 当前上下文, 不经位置头
        cf_feat = self._mem_feature(cf)
        first_in = torch.max(cf_feat, direction * dir_strength)
        # ★ P10-B 首字符 WTA: 与训练读出一致, OOD 输入必落最近已学
        #   字符 (模式补全), 杜绝阈值拼出的无效字节
        if (use_char_wta
                and getattr(self, "W_first", None) is not None):
            f_scores = torch.mv(self.W_first, first_in) + self.b_first
            self._last_first_scores = f_scores
            first_code = int(f_scores.argmax().item())
        else:
            first_bits = self._binary_decode(self.W_ctx_to_first, first_in)
            first_code = 0
            for j in range(self.output_size):
                if first_bits[j] >= 0.5:
                    first_code |= (1 << j)
        ch = chr(first_code) if 0 <= first_code <= 255 else '?'
        result.append(ch)
        prev_char = ch

        if n_steps <= 1:
            self._dmd_done = True
            return ''.join(result)

        state = cf.clone()
        prev_state = None
        for step in range(1, n_steps):
            vec = self._char_to_8bit_bias(ch)
            recall = None
            if self.use_memory_thinking:
                output, sparse_feat, state = self._think_character(
                    vec, update_memory=update_memory, n_loops=n_loops)
            else:
                output = self._multi_layer_forward(vec, n_loops=n_loops)
                v_curr = self.V_deep[-1] if self.num_layers > 1 else self.V
                sparse_feat = self._dg_separate(v_curr)
                if update_memory:
                    self.update_coactivation(sparse_feat)
                self._slot_fatigue_step(
                    sparse_feat, cf=cf,
                    code=ord(ch) if ch and 0 <= ord(ch) <= 255 else None)  # ★ P46
                recall = self.recall_from_memassoc(sparse_feat, sparse_hint=True)
            think = self._encode_reservoir(sparse_feat)
            # ★ P6 分级融合 + k-WTA (与 train_dmd_rollout 逐位一致):
            #   防 max-OR 并集单调饱和 (feat 冻结 → 读出恒定 → 重复锁定)。
            #   方向 (committed direction, 0.6 权重) 在融合内竞争锚定,
            #   取代旧二次 OR 重导向 (dir_anchor 参数保留兼容, 不再使用)。
            seqctx = self._seqctx_contrib(sparse_feat, step)
            state = self._fuse_state_kwt(
                sparse_feat, recall,
                self._chain_transition(prev_state)
                if prev_state is not None else None,
                direction, seqctx=seqctx,
                content=self._content_quota_src(cf))
            self.MemWork = state

            # 输出反馈: 已生成字符的状态重新驱动方向演化 (早停/收敛判定)
            prev_dir = self.dmd_direction.clone()
            self._dmd_step(state)
            dir_delta = (self.dmd_direction - prev_dir).abs().mean().item()
            # ★ v3.6 稀疏路由: direction 门控读出 h
            gate = direction if use_dir_gated_readout else None
            next_bits = self._seq_predict(self._mem_feature(state), gate=gate,
                                          use_char_wta=use_char_wta)
            next_code = 0
            for j in range(self.output_size):
                if next_bits[j] >= 0.5:
                    next_code |= (1 << j)
            next_ch = chr(next_code) if 0 <= next_code <= 255 else '?'

            if (not self._dmd_done
                    and char_sfa_eos > 0.0
                    and recent_ex >= 1
                    and use_char_wta
                    and eos_code is not None):
                # ★ v3.7 短记忆 EOS 脉冲: 终结符连续活动抬升 EOS 槽,
                #   同时压制终结符本身 (Firing-rate adaptation) —
                #   打破重复字符尾部状态塌缩导致的 EOS 不触发。
                sc = getattr(self, "_last_char_scores", None)
                if sc is not None:
                    sc[eos_code] += char_sfa_eos * recent_ex
                    sc[sfa_eos_char] -= char_sfa_eos * recent_ex
                    next_code = int(sc.argmax().item())
                    next_ch = (chr(next_code)
                               if 0 <= next_code <= 255 else '?')
                # 累积终结符活动计数 (短记忆, 衰减由每次重置清零)
                if next_code == sfa_eos_char:
                    recent_ex += 1.0
                else:
                    recent_ex = 0.0

            # EOS: 显式结束符终止 (独立于早停启发式, 不受 disable_early_stop 影响)
            # ★ P10-B EOS margin 门控: 弱胜不终止 — OOD/早期状态 EOS
            #   噪声性胜出 (得分差小) 时改取次优字符续写; 库内末步
            #   EOS 经 margin 训练胜出充分 (得分差大) 正常终止。
            if eos_code is not None and next_code == eos_code:
                sc = getattr(self, "_last_char_scores", None)
                margin_ok = True
                if sc is not None:
                    sc2 = sc.clone()
                    sc2[eos_code] = -1e9
                    eos_m = (sc[eos_code] - sc2.max()).item()
                    self._last_eos_margin = eos_m
                    if eos_margin > 0.0 and eos_m < eos_margin:
                        margin_ok = False
                        alt = int(sc2.argmax().item())
                        if 0 <= alt <= 255 and alt != eos_code:
                            next_code = alt
                            next_ch = chr(next_code)
                if margin_ok:
                    self._dmd_done = True
                    break

            # response_done: 方向状态收敛 → 自主终止 (不依赖目标长度)
            # disable_early_stop=True 时显式禁用早停, 只受 n_steps 上限约束
            if not disable_early_stop and step >= min_steps:
                if dir_delta <= done_delta_thresh:
                    stable_streak += 1
                else:
                    stable_streak = 0
                if stable_streak >= done_stable_steps:
                    self._dmd_done = True
                    break

            if not disable_early_stop and next_ch == prev_char:
                repeat_count += 1
            else:
                repeat_count = 0
            if not disable_early_stop and repeat_count >= max_repeat:
                self._dmd_done = True
                break

            if not disable_early_stop and len(result) >= 3:
                pattern = ''.join(result[-2:]) + next_ch
                if pattern in seen_patterns:
                    self._dmd_done = True
                    break
                seen_patterns.add(''.join(result[-3:]) + next_ch)

            result.append(next_ch)
            prev_char = next_ch
            ch = next_ch
            prev_state = state.clone()

            # done head: 学习到的"该结束"信号 (在 append 后检测, 保留末字符)
            if use_done_signal and step >= min_steps:
                done_score = torch.mv(self.W_done, self._mem_feature(state))
                if done_score.item() > 0:
                    self._dmd_done = True
                    break

        return ''.join(result)

    def generate_from_dmd(self, n_steps=30, max_repeat=3,
                          update_memory=True, n_loops=1,
                          dir_strength=1.0, min_steps=3,
                          done_stable_steps=3, done_delta_thresh=0.01,
                          replay_input=False, disable_early_stop=False,
                          eos_code=None, use_done_signal=False,
                          dir_anchor=False, use_dir_gated_readout=False,
                          use_sentence_wta=False,
                          use_char_wta=False, eos_margin=0.0):
        """从 input_end 提交的 DMD 方向生成。"""
        if replay_input:
            return self.generate_replay_input_codes(n_steps=n_steps)
        direction = getattr(self, "dmd_committed_direction", None)
        context = getattr(self, "dmd_committed_context", None)
        if direction is None:
            raise RuntimeError("call input_end() after encoding input before DMD generation")
        if context is None:
            context = direction
        # ★ v3.6 句级 WTA: 推理时用 WTA 选离散句模板 (不漂移)
        if use_sentence_wta and getattr(self, "W_sent", None) is not None:
            goal = (getattr(self, "_last_goal", None)
                    if getattr(self, "_last_goal", None) is not None
                    else context)
            direction, _ = self._select_sentence(goal)
        return self.generate_recurrent_dmd(
            context, n_steps=n_steps, max_repeat=max_repeat,
            update_memory=update_memory, n_loops=n_loops,
            direction=direction, dir_strength=dir_strength,
            min_steps=min_steps, done_stable_steps=done_stable_steps,
            done_delta_thresh=done_delta_thresh,
            disable_early_stop=disable_early_stop,
            eos_code=eos_code, use_done_signal=use_done_signal,
            dir_anchor=dir_anchor,
            use_dir_gated_readout=use_dir_gated_readout,
            use_char_wta=use_char_wta, eos_margin=eos_margin)

    # ==================== v3.6 短期记忆回放 ====================
    # 目标: 检验短期记忆是否足以支撑句级复述 (记忆回放模式)。
    # 思路: 输入编码时记录每个字符的 DG 稀疏特征快照 (输入记忆),
    #       回放时逐字符从快照取特征 → W_h2o 解码 → 输出。
    #       绕过 W_seq 自回归, 隔离"记忆存储/提取"能力与"生成预测"能力。
    # 红线合规: 纯记忆存取 + 二值竞争解码, 无连续数值学习信号。

    def record_input_snapshot(self):
        """记录当前输入字符序列快照 (从 _coact_trace 反推不可行, 直接存)"""
        if not hasattr(self, "_input_snapshot"):
            self._input_snapshot = []
        self._input_snapshot = []

    def generate_replay_from_snapshot(self, n_steps=None, snapshot=None):
        """从输入快照回放 — 记忆回放模式 (绕过 W_seq 自回归)

        输入编码期间若调用 record_dmd=True 且 use_dmd_input_prediction 或
        dmd_record_trace, 已记录的 DG 特征快照可用于回放。此处从
        _dmd_input_states (逐字符状态) 提取, 用 W_h2o 解码回放。

        Args:
            n_steps: 回放最大字符数 (默认与快照等长)
            snapshot: 外部传入字符序列 (调试用, 默认用内部快照)

        Returns:
            replay_text: 回放文本
        """
        states = getattr(self, "_dmd_input_states", None)
        if snapshot is None and not states:
            raise RuntimeError(
                "no input snapshot: encode with record_dmd=True first")
        if snapshot is not None:
            codes = [ord(c) for c in snapshot if 0 <= ord(c) <= 255]
            chars = [chr(c) for c in codes]
            out = ''.join(chars[:n_steps]) if n_steps else ''.join(chars)
            return out
        codes = []
        for st in states:
            bits = self._binary_decode(self.W_h2o, st, self.b_o)
            code = 0
            for j in range(self.output_size):
                if bits[j] >= 0.5:
                    code |= (1 << j)
            if 0 <= code <= 255:
                codes.append(code)
        if n_steps:
            codes = codes[:n_steps]
        return ''.join(chr(c) for c in codes)

    def generate_replay_input_codes(self, n_steps=None):
        """从输入侧编码快照回放 — 对照: 验证输入侧编码可逆。

        记录 _dmd_input_encodings (输入字符编码, 含 ASCII 位型),
        回放时直接从编码提取 ASCII — 仅用于对照证明"输入可逆",
        区分"输入编码可逆"与"状态可回放"两种记忆能力。
        """
        encs = getattr(self, "_dmd_input_encodings", None)
        if not encs:
            raise RuntimeError("no input encodings: encode with record_dmd=True first")
        codes = []
        for enc in encs:
            code = 0
            for j in range(self.output_size):
                if enc[j] > 0:
                    code |= (1 << j)
            codes.append(code)
        if n_steps:
            codes = codes[:n_steps]
        return ''.join(chr(c) for c in codes)


    # ==================== v2.4 突触保护 (ISI-CV 本地版) ====================
    # 稳定性掩码保护已巩固突触: 位置头在线更新时跳过/降速, 防旧知识
    # 被覆盖。掩码构造基于更新方向一致性 + 发放频率 (慢变量, 同 SFA
    # thr_shift 族), 不产生连续数值学习信号 (红线合规)。

    def _ensure_protect_state(self):
        """防御式初始化 — 旧模型 (v2.3 及更早) 缺 v2.4 属性时补齐"""
        if not hasattr(self, "protect_mode"):
            self.protect_mode = "off"
            self.protect_strength = 0.5
            self.stab_beta = 5.0
            self.stab_decay = 0.9
            self.freq_thr = 0.3
            self.stab_cum = []
            self._freq_count = torch.zeros(self.hidden_size,
                                           dtype=torch.float32, device=DEVICE)
            self._freq_seen = 0
        if not hasattr(self, "stab_cum"):
            self.stab_cum = []
        # 位置头追加后同步稳定性累积 (防御: 旧模型/新 append 缺口)
        while len(self.stab_cum) < len(self.W_ctx_to_pos):
            self.stab_cum.append(
                torch.zeros_like(self.W_ctx_to_pos[len(self.stab_cum)]))

    def _freq_norm(self):
        """归一化发放频率向量 (None = 尚无统计)"""
        if getattr(self, "_freq_seen", 0) <= 0:
            return None
        return self._freq_count / self._freq_seen

    def _pos_protect_mask(self, k):
        """位置头 k 的突触保护掩码 (0=自由, α=软保护强度)

        - sign 模式: |stab_cum| > β — 更新方向长期一致的突触 = 已巩固
        - freq 模式: 归一化发放频率 > thr 的列 = 核心编码神经元
        - both 模式: 取两者并集 (max)
        返回与 W_ctx_to_pos[k] 同形状的 [0,α] 掩码; None = 不保护。
        """
        if self.protect_mode == "off" or k >= len(self.stab_cum):
            return None
        mask = torch.zeros_like(self.W_ctx_to_pos[k])
        if self.protect_mode in ("sign", "both"):
            mask = (self.stab_cum[k].abs() > self.stab_beta).float()
        if self.protect_mode in ("freq", "both"):
            fn = self._freq_norm()
            if fn is not None:
                col = (fn > self.freq_thr).float().unsqueeze(0)
                mask = torch.maximum(mask, col.expand_as(mask))
        return mask * self.protect_strength

    def _update_stab(self, k, rpe, pre):
        """更新位置头 k 的稳定性累积 — 同方向更新 → |stab|↑ (保护),
        方向翻转 → |stab|↓ (释放)。O(8×256)/头, 与权重更新并行。
        """
        if self.protect_mode in ("off", "freq") or k >= len(self.stab_cum):
            return
        update_sign = torch.outer(rpe, pre)   # ∈ {−1,0,1} × pre
        self.stab_cum[k] = self.stab_decay * self.stab_cum[k] + update_sign

    def _append_pos_head(self):
        """追加一个新位置记忆头 (随机初始化, 与稳定性累积同步)"""
        W = torch.empty(self.output_size, self.feat_dim,
                        dtype=torch.float32, device=DEVICE)
        W.uniform_(-0.1, 0.1)
        b = torch.empty(self.output_size, dtype=torch.float32,
                        device=DEVICE)
        b.uniform_(-0.1, 0.1)
        self.W_ctx_to_pos.append(W)
        self.b_ctx_to_pos.append(b)
        if hasattr(self, "stab_cum"):
            self.stab_cum.append(torch.zeros_like(W))

    # ==================== 在线学习 ====================

    def train_on_dialogue(self, inp, resp, lr=0.05, n_iter=30, train_pos=True,
                          sdm=None, semantic_align=True, novelty_thr=0.5,
                          novelty_hint=None):
        """在线学习单个对话对 — 奖赏预测误差调制 Hebbian 增量更新

        ★ 学习规则: Δw = lr × RPE_j × pre_activity
          - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差
            +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
          - pre_activity = 二值 {0, 1} 突触前活动
          - ★ 无 center = clamp(raw, -1, 1): 连续数值运算, 已移除
          - ★ 无偏置更新: 连续数值运算, 已移除
          - 可在推理过程中随时调用，不需要重新训练整个数据集

        ★ v2.3 位置头在线扩展 (train_pos=True, 方案 3):
          - 新对话自动 append 位置记忆头 (W_ctx_to_pos[k]/b_ctx_to_pos[k])
            并训练 → 绕过"位置头不随在线学习更新"限制, 新学对话
            也能被位置头修正 (与 train_pos_heads 同机制: acc_state +
            _mem_feature + RPE Hebbian)。
          - 已有位置头 (k < len) 也增量更新 (对齐该对话的状态)。

        ★ v2.4 语义对齐 + 突触保护 (CLS 双系统分工):
          - semantic_align + sdm: 按 novelty 分流 — 新语义类逐字 append
            新槽位; 熟悉类用 SDM 原型多数投票码做统计 target (不 append,
            共享位置头只向类共识收敛, 不再被逐字 RPE 抵消 → 防混合模板)。
          - novelty_hint 须由调用方在 SDM 写入前计算 (写入后 novelty 归零
            会自污染判定); 未传则此处按"尚未写入"计算。
          - protect_mode != off 时 (见 __init__): 更新挂稳定性掩码, 已巩固
            突触跳过/降速 → 防旧知识被覆盖。
        """
        resp_codes = self._text_to_codes(resp)
        if not resp_codes:
            return

        # 1) 训练 W_ctx_to_first: 二值累积状态 → 首字符
        _, acc_state = self.encode_text_lif(inp)
        if acc_state.sum().item() == 0:
            return

        first_target = torch.tensor(
            [float((resp_codes[0] >> j) & 1) for j in range(self.output_size)],
            dtype=torch.float32, device=DEVICE
        )
        for _ in range(n_iter):
            # ★ 纯二值阈值解码: 无 sigmoid, 无连续数值, 无 raw
            pred = self._binary_decode(self.W_ctx_to_first, acc_state)

            # ★ 奖赏预测误差 (RPE): RPE_j = target_j − pred_j ∈ {−1, 0, +1}
            pred_bits = (pred > 0.5).float()
            target_bits = (first_target > 0.5).float()
            rpe = target_bits - pred_bits

            # ★ 奖赏预测误差调制 Hebbian (v11):
            #    Δw_ji = lr × RPE_j × acc_state
            #    ★ 无 center = clamp(pred_raw, -1, 1): 已移除
            for j in range(self.output_size):
                self.W_ctx_to_first[j] += lr * rpe[j] * acc_state
            self.W_ctx_to_first.clamp_(-10.0, 10.0)

        # ★ v2.3/v2.4 位置头在线扩展/语义对齐 (双模式):
        #   - novelty ≥ thr (新语义类) 或未启用语义对齐/无 SDM:
        #     逐字 append 新位置头训练 (新槽位零污染, v2.3 行为)
        #   - novelty < thr (熟悉类): 语义对齐 — target 用 SDM 原型
        #     多数投票码 (统计结构), 不 append → 共享头只被统计拉向
        #     类共识, 不被逐字 RPE 抵消 (防混合模板/灾难性遗忘)
        #   训练输入均用 feat = _mem_feature(acc_state) (与
        #   train_pos_heads/generate_recurrent 的 pos_head_recall 一致)。
        #   更新一律挂 protect 掩码 (v2.4, 突触保护)。
        if train_pos:
            self._ensure_protect_state()
            # 语义类判定: 优先用调用方 novelty_hint (须存储前计算,
            # 避免 SDM 写入后自污染); 未传则此处按"尚未写入"计算。
            use_verbatim = True
            if semantic_align and sdm is not None:
                if novelty_hint is None:
                    novelty_hint = sdm.novelty([ord(c) for c in inp])
                use_verbatim = novelty_hint >= novelty_thr

            if use_verbatim:
                # —— 新语义类: 逐字训练, 缺失位置头自动 append ——
                for k, code in enumerate(resp_codes):
                    while len(self.W_ctx_to_pos) <= k:
                        self._append_pos_head()
                    target = torch.tensor(
                        [float((code >> j) & 1) for j in range(self.output_size)],
                        dtype=torch.float32, device=DEVICE
                    )
                    W = self.W_ctx_to_pos[k]
                    protect = self._pos_protect_mask(k)
                    for _ in range(n_iter):
                        feat = self._mem_feature(acc_state)
                        pred = self._binary_decode(W, feat, self.b_ctx_to_pos[k])
                        pred_bits = (pred > 0.5).float()
                        target_bits = (target > 0.5).float()
                        rpe = target_bits - pred_bits
                        self._hebbian_step(W, rpe, feat, pred_bits, lr,
                                           protect=protect)
                        self._update_stab(k, rpe, feat)
            else:
                # —— 熟悉类: 语义对齐 (统计 target, 不 append) ——
                proto = sdm.recall_prototype(
                    [ord(c) for c in inp], length=len(self.W_ctx_to_pos))
                for k in range(len(self.W_ctx_to_pos)):
                    if k >= len(proto) or proto[k] == 0:
                        continue    # 原型该位置无内容 → 不更新 (防推空)
                    target = torch.tensor(
                        [float((proto[k] >> j) & 1) for j in range(self.output_size)],
                        dtype=torch.float32, device=DEVICE
                    )
                    W = self.W_ctx_to_pos[k]
                    protect = self._pos_protect_mask(k)
                    for _ in range(n_iter):
                        feat = self._mem_feature(acc_state)
                        pred = self._binary_decode(W, feat, self.b_ctx_to_pos[k])
                        pred_bits = (pred > 0.5).float()
                        target_bits = (target > 0.5).float()
                        rpe = target_bits - pred_bits
                        self._hebbian_step(W, rpe, feat, pred_bits, lr,
                                           protect=protect)
                        self._update_stab(k, rpe, feat)

        # 2) 训练 W_seq: 工作记忆状态 → 下一字符编码 (纯二值奖赏调制 Hebbian)
        if len(resp_codes) >= 2:
            # ★ v10: 取每步工作记忆状态 (而非隐藏层输出)
            _, states, _ = self.encode_text_lif_states(resp)
            if len(states) < 2:
                return

            seq_data = []
            for i in range(len(states) - 1):
                target = torch.tensor(
                    [float((resp_codes[i + 1] >> j) & 1) for j in range(self.output_size)],
                    dtype=torch.float32, device=DEVICE
                )
                seq_data.append((states[i], target))

            for _ in range(n_iter):
                # ★ 随机打乱
                random.shuffle(seq_data)
                for fr, target in seq_data:
                    # ★ 纯二值阈值解码: 无 sigmoid, 无连续数值, 无 raw
                    out = self._binary_decode(self.W_seq, fr, self.b_seq)

                    # ★ 奖赏预测误差 (RPE): RPE_j = target_j − out_j ∈ {−1, 0, +1}
                    pred_bits = (out > 0.5).float()
                    target_bits = (target > 0.5).float()
                    rpe = target_bits - pred_bits

                    # ★ 奖赏预测误差调制 Hebbian (v11):
                    #    Δw_ji = lr × RPE_j × fr
                    #    ★ 无 center = clamp(raw, -1, 1): 已移除
                    #    ★ 无 b_seq 偏置更新: 已移除
                    for j in range(self.output_size):
                        self.W_seq[j] += lr * rpe[j] * fr
                    self.W_seq.clamp_(-10.0, 10.0)


if __name__ == "__main__":
    # 快速测试 — 验证 0-1 膜电位神经元 + 选择性连接 + 侧抑制 + 纯二值解码
    print("=" * 60)
    print("  PyTorch GPU 加速 0-1 膜电位脉冲神经网络测试 (v9)")
    print("  神经元: 0-1 膜电位 V (leak+threshold+reset)")
    print("  选择性连接: 每个神经元只连接 ~50% 输入")
    print("  侧抑制: 发放神经元抑制未发放神经元")
    print("  解码: 纯二值阈值 (out = (W·x + b > 0).float(), 无 sigmoid)")
    print("  学习规则: 奖赏预测误差调制 Hebbian (Δw = lr×RPE×pre, RPE=期望−实际)")
    print("  状态累积: 0-1 V 电位 + 随机遗忘 (无加权平均)")
    print("  关联学习: 共激活矩阵 W_coact (Fire together, wire together)")
    print("=" * 60)

    sim = TorchLIFSimulator()
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)

    test_chars = "Hello GPU!"
    print(f"\n测试编码: {test_chars}")
    print(f"(每个字符经 0-1 膜电位神经元处理，输出二值 {{0,1}})\n")
    for ch in test_chars:
        vec = sim._char_to_8bit(ch) + sim.input_bias
        t0 = time.perf_counter()
        output = sim._neuron_forward(vec)  # ★ v9: 0-1 膜电位神经元
        dt = (time.perf_counter() - t0) * 1000
        decoded = sim.fr_to_char(output)
        match = "✓" if decoded == ch else "✗"
        active = torch.sum(output > 0.5).item()  # 活跃神经元数
        # 膜电位统计
        V_mean = sim.V.mean().item()
        V_max = sim.V.max().item()
        print(f"  '{ch}' → '{decoded}' {match} (active={active}, V_mean={V_mean:.3f}, V_max={V_max:.3f}, {dt:.1f}ms)")

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"  PyTorch CUDA 加速模块就绪!")
    print(f"  ★ 重要: 膜电位 V ∈ [0, 1], 携带电位强度信息")
    print(f"  ★ 重要: 选择性连接: connection_sparsity=0.5")
    print(f"  ★ 重要: 侧抑制: inhibition_strength={sim.inhibition_strength}")
    print(f"  ★ 重要: 解码使用纯二值阈值 (W·x + b > 0)，无 sigmoid")
    print(f"  ★ 重要: 学习规则为奖赏预测误差调制 Hebbian — Δw = lr×RPE×pre (RPE=期望−实际)")
    print(f"  ★ 重要: 共激活矩阵 W_coact 实现关联学习")
    print(f"  ★ 重要: 状态累积使用 0-1 V 电位 + 随机遗忘 (无加权平均)")