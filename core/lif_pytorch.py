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
HIDDEN_SIZE = 256               # 隐藏层神经元数 (固定为 256)


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
                 num_layers=1):
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
    # 字符 → 随机稠密编码映射表
    # 每个字符绑定一个固定的随机 256-dim 二值向量 (~50% 活跃)
    # 生物学类比: 不同感官输入激活不同的神经元群体
    # 此编码是二值 {0, 1}，无连续数值
    # ============================================================
    CHAR_CODEBOOK = {}  # 延迟初始化

    @staticmethod
    def _get_char_code(ch):
        """获取字符的随机稠密编码 (256-dim, ~50%活跃, 二值 {0,1})"""
        if ch not in TorchLIFSimulator.CHAR_CODEBOOK:
            code = ord(ch)
            rng = np.random.RandomState(code)
            vec = torch.from_numpy((rng.rand(HIDDEN_SIZE) > 0.5).astype(np.float32)).to(DEVICE)
            TorchLIFSimulator.CHAR_CODEBOOK[ch] = vec
        return TorchLIFSimulator.CHAR_CODEBOOK[ch]

    @staticmethod
    def _char_to_8bit(ch):
        """字符 → 随机稠密编码 (兼容旧接口名)"""
        return TorchLIFSimulator._get_char_code(ch).clone()

    @staticmethod
    def _text_to_codes(text):
        """文本 → ASCII 码列表 (仅可打印字符)"""
        return [ord(c) for c in text if 32 <= ord(c) <= 126]

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
        """重置所有层膜电位为 0 — 处理新序列前调用"""
        self.V = torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)
        for i in range(len(self.V_deep)):
            self.V_deep[i] = torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)

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
        # 发放比例 (有多少神经元被激活了)
        fired_ratio = fired_neurons.sum().item() / self.hidden_size
        if fired_ratio == 0:
            return

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
        else:
            W = self.W_deep[layer_idx - 1]
            mask = self.deep_masks[layer_idx - 1]
            V = self.V_deep[layer_idx - 1]

        # 1. 漏电: 膜电位按比例衰减
        V = V * (1.0 - self.leak)

        # 2. 积分: 加权输入 (带选择性连接掩码)
        W_m = W * mask if mask is not None else W
        activation = torch.mv(W_m, input_vec)
        V = torch.clamp(V + activation, 0.0, 1.0)

        # 3. 发放: V > threshold 则输出 1
        output = (V > self.threshold).float()

        # 4. 发放后部分重置: 保留残余电位
        V = torch.where(output > 0, V * (1.0 - self.reset_factor), V)

        # 5. 侧抑制: 发放神经元抑制未发放神经元
        if self.inhibition_strength > 0:
            self._lateral_inhibition(output, V)

        # 写回该层膜电位
        if layer_idx == 0:
            self.V = V
        else:
            self.V_deep[layer_idx - 1] = V

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
        return chr(code) if 32 <= code <= 126 else '?'

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
    output_size = 8
    hidden_size = HIDDEN_SIZE

    # 目标: 8-bit 编码 (仅用于计算奖赏信号)
    targets_gpu = torch.zeros(n_vocab, output_size, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(train_codes):
        for j in range(output_size):
            targets_gpu[i, j] = float((c >> j) & 1)

    # 输入: 随机稠密编码 (二值) — 直接用于 W_h2o 解码
    input_vecs_gpu = torch.zeros(n_vocab, hidden_size, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(train_codes):
        ch = chr(c) if 32 <= c <= 126 else '?'
        input_vecs_gpu[i] = TorchLIFSimulator._get_char_code(ch)

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
            for j in range(output_size):
                sim.W_h2o[j] += lr * rpe[j] * vec
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
                 use_dg_separation=False, dg_k=32):
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
        """
        super().__init__(hidden_size, output_size, input_bias,
                         leak=leak, threshold=threshold,
                         reset_factor=reset_factor,
                         inhibition_strength=inhibition_strength,
                         num_layers=num_layers)

        # ★ v14: 三因子资格迹参数
        self.use_eligibility_trace = use_eligibility_trace
        self.eligibility_lambda = eligibility_lambda

        # ★ v14: DG 稀疏分离参数
        self.use_dg_separation = use_dg_separation
        self.dg_k = dg_k

        # 上下文→首字符预测: 8×256, 奖赏调制 Hebbian 训练
        self.W_ctx_to_first = torch.empty(output_size, hidden_size, dtype=torch.float32, device=DEVICE)
        self.W_ctx_to_first.uniform_(-0.1, 0.1)

        # ★ v13 位置记忆头: 上下文状态 → 回复第 k 字符 (8×256 × max_pos)
        #   对每个回复位置 k 一个独立 Hebbian 分类器, 与 W_ctx_to_first 同机制。
        #   用于修正 W_seq 循环生成的"非首字"输出 (experiment14: 字符级
        #   5.4% → 77.4%)。延迟初始化: 首用前检查容器是否为空 (保持向后兼容)。
        self.W_ctx_to_pos = []   # 每位置: (8, hidden_size)
        self.b_ctx_to_pos = []   # 每位置: (8,)

        # 序列预测: 8×256, 二值输出 → 下一字符编码
        self.W_seq = torch.empty(output_size, hidden_size, dtype=torch.float32, device=DEVICE)
        self.W_seq.uniform_(-0.1, 0.1)
        self.b_seq = torch.empty(output_size, dtype=torch.float32, device=DEVICE)
        self.b_seq.uniform_(-0.1, 0.1)

        # ============================================================
        # ★ 共激活矩阵 W_coact — 关联记忆层 (Associative Memory, v10)
        #
        # 追踪神经元对的共发放频率，实现"Fire together, wire together"。
        # W_coact[i][j] = 神经元 i 和 j 的共激活次数 (归一化到 [0, 1])
        # v10: 提升为显式关联记忆层，可被 recall_from_memassoc 读取回忆。
        # ============================================================
        self.W_coact = torch.zeros(hidden_size, hidden_size, dtype=torch.float32, device=DEVICE)
        self.coact_lr = 0.1  # 共激活学习率
        self.coact_decay = 0.99  # 共激活衰减率 (遗忘旧关联)

        # ============================================================
        # ★ 工作记忆层 MemWork — 0-1 分级电位累积 + 随机遗忘 (v10)
        #
        # 独立于隐藏层的显式工作记忆结构 (生物学: 前额叶持续发放)。
        # 更新规则 (v9 状态累积的独立成层):
        #   MemWork = max(V_LN, recall, MemWork × forget_mask)
        #   - V_LN: 最深层 (L4) 膜电位 (0-1 分级)
        #   - recall: 关联记忆回忆 (W_coact 提取的关联模式)
        #   - forget_mask: 每步 30% 神经元随机遗忘
        # 解码层 W_ctx_to_first / W_seq 均从 MemWork 取信号。
        # ============================================================
        self.MemWork = torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)
        self.mem_forget_ratio = 0.3  # 每步随机遗忘比例

        # ★ W_hh 已移除! 记忆由 MemWork 工作记忆层承担
        # ★ LIF 动力学已移除! 使用 _binary_forward 替代

    # ==================== 关联记忆回忆 ====================

    def recall_from_memassoc(self, cue):
        """★ 关联记忆回忆 — 给定线索模式，提取关联的神经元群体 (分级强度, v12.4)

        生物学依据: 海马体关联记忆 → 皮层回忆。
        给定当前最深层输出 (cue)，W_coact·cue 得到与之共发放过的
        神经元群体加权和。除以期望最大共激活数 (hidden_size/2, ~50%
        活跃) 归一化到 [0,1] 并饱和钳位 → **分级回忆强度** (非二值)。

        ★ v12.4 关键变更: 由二值化 (raw > threshold) 改为分级回忆:
          recall = min(raw / (hidden_size/2), 1)
          实验验证 (diag_ctx_discrim): 二值化丢失回忆强度信息 → 经 max
          注入状态后判别性崩塌 (独立 W_ctx_to_first 5/14); 分级回忆 →
          14/14。状态判别性是 W_ctx_to_first 的瓶颈 (原 6/14)。
          ★ 反例: clamp(raw, 0, 1) 也失败 (4-5/14) — raw 量级常 > 1,
          直接钳位大量饱和; 先除期望活跃数再钳位才保留分布形状。

        ★ 注意: 这是突触矩阵的线性回忆 (Hebbian 回忆)，
              不是余弦相似度/向量检索 (禁止项)。

        Args:
            cue: 线索模式 (256-dim 二值输出)

        Returns:
            recall: 分级回忆强度 (256-dim, 每个元素 ∈ [0, 1])
        """
        raw = torch.mv(self.W_coact, cue)
        scale = self.hidden_size / 2.0  # 期望最大共激活数 (~50% 活跃 = 128)
        recall = torch.clamp(raw / scale, 0.0, 1.0)
        return recall

    def reset_memory(self):
        """重置工作记忆层 (关联记忆 W_coact 不重置，保留长期记忆)"""
        self.MemWork = torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)

    # ==================== 二值阈值神经元前向计算 ====================

    def _binary_forward_char(self, input_vec):
        """字符二值神经元前向计算 — 返回二值 {0, 1}"""
        return self._binary_forward(input_vec)

    def update_coactivation(self, output):
        """★ 更新共激活矩阵 — 关联学习 (Associative Learning)

        "Fire together, wire together":
        如果神经元 i 和 j 同时发放，则 W_coact[i][j] 增加。
        共激活强度反映神经元之间的关联程度。

        Args:
            output: 当前步的二值发放输出 (256-dim, {0, 1})
        """
        # 外积: output[i] × output[j] = 1 当且仅当两者都发放
        coact_update = torch.outer(output, output)

        # 更新共激活矩阵 (带衰减)
        self.W_coact = self.coact_decay * self.W_coact + self.coact_lr * coact_update

        # 钳位到 [0, 1]
        self.W_coact.clamp_(0.0, 1.0)

    # ==================== 序列编码 ====================

    def encode_text_lif(self, text, update_memory=True, n_loops=1):
        """逐字符处理文本，多层前向 + 工作记忆/关联记忆累积上下文状态

        ★ v10 关键变化:
          - 输入经 4 层隐藏层前向传播，输出为最深层 (L4) 二值发放
          - 独立工作记忆层 MemWork: 0-1 分级电位累积 + 随机遗忘
          - 独立关联记忆层 W_coact: 共发放追踪 + 回忆提取
          - 状态更新: state = max(V_LN, recall, state × forget_mask)

        ★ v13.1: n_loops 自回归循环 (每次输入后循环一次再传入输入),
          透传给 _multi_layer_forward; 默认 1 与旧行为一致。

        Args:
            text: 输入文本
            update_memory: 是否更新关联记忆 W_coact (学习/写入)。
              ★ v12.3: 评估/生成时应为 False (冻结), 保持与训练时
              状态一致; 训练收集前由训练函数预热并冻结快照。
            n_loops: 每次输入自回归前向轮数 (v13.1, 默认 1)

        Returns:
            outputs_list: 每个字符的最深层二值输出列表 (用于 W_seq 训练)
            state: 工作记忆累积状态 (256-dim, 每个元素 ∈ [0, 1])
        """
        chars = [c for c in text if 32 <= ord(c) <= 126]
        if not chars:
            return [], torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)

        # ★ v10: 重置所有层膜电位 + 工作记忆层
        self.reset_state()
        self.reset_memory()

        outputs_list = []
        state = torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)
        for ch in chars:
            vec = self._char_to_8bit(ch) + self.input_bias
            # ★ v10: 4 层前向传播 → 最深层 (L4) 输出 (★ v13.1: n_loops 循环)
            output = self._multi_layer_forward(vec, n_loops=n_loops)
            outputs_list.append(output)

            # ★ v10: 关联记忆层更新 (共发放追踪) — 先学习(写)再回忆(读)
            # ★ v12.2 顺序回退: 恢复 update 先于 recall (实验9验证的 now 模式
            #   = 8~10/14)。recall 先于 update 会破坏判别性 (实测 0~2/14):
            #   recall 用旧 W_coact 提取"历史"模式, state 与当前字符活动
            #   脱节; update 先行使 recall 含当前共发放, 状态自洽。
            # ★ v12.3: update_memory=False 时冻结 W_coact (不写入),
            #   用于评估/生成与训练状态一致 (见 train_context_to_first)。
            if update_memory:
                self.update_coactivation(output)
            recall = self.recall_from_memassoc(output)

            # ★ v10: 工作记忆层更新 — 当前活动 + 关联记忆回忆
            #
            # ★ v11 关键变更: 移除跨字符 max 累积 (state × forget_mask)
            #   now 模式: state = max(V_LN, recall), 不跨字符 OR 累积。
            #   实验9 验证: now 模式首字符预测 9~11/14, decay/forget_mask
            #   模式仅 2/14。
            #
            # ★ v12.2 回退: v_peak 恢复为累积膜电位 V (而非当前字符输出),
            #   并移除每字符 reset_state。实验验证: v_peak=V 累积 + 不 reset
            #   = 8~10/14; v_peak=output + reset 仅 0~2/14 (state 丢失跨字符
            #   上下文, 只反映最后一个字符)。跨字符信息由膜电位累积承载。
            v_peak = self.V_deep[-1] if self.num_layers > 1 else self.V
            state = torch.max(v_peak, recall)
            self.MemWork = state

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
        chars = [c for c in text if 32 <= ord(c) <= 126]
        if not chars:
            return [], [], torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)

        # ★ v10: 重置所有层膜电位 + 工作记忆层
        self.reset_state()
        self.reset_memory()

        outputs_list = []
        states = []
        state = torch.zeros(self.hidden_size, dtype=torch.float32, device=DEVICE)
        for ch in chars:
            vec = self._char_to_8bit(ch) + self.input_bias
            output = self._multi_layer_forward(vec, n_loops=n_loops)
            outputs_list.append(output)

            # ★ v10: 关联记忆层更新 (共发放追踪) — 先学习(写)再回忆(读)
            # ★ v12.2 顺序回退: 与 encode_text_lif 一致 (实验9 now 模式)
            # ★ v12.3: update_memory=False 时冻结 W_coact (不写入)
            if update_memory:
                self.update_coactivation(output)
            recall = self.recall_from_memassoc(output)

            # ★ v10: 工作记忆层更新 — 当前活动 + 关联记忆回忆
            #
            # ★ v11 关键变更: 移除跨字符 max 累积 (state × forget_mask)
            #   与 encode_text_lif 一致 (now 模式):
            #     - state = max(V_LN, recall), 不跨字符 OR 累积
            #     - 实验9 验证: now 模式首字符预测 9~11/14
            #
            # ★ v12.2 回退: v_peak 恢复为累积膜电位 V, 不 reset
            v_peak = self.V_deep[-1] if self.num_layers > 1 else self.V
            state = torch.max(v_peak, recall)
            states.append(state)
            self.MemWork = state

        return outputs_list, states, state

    # ==================== W_ctx_to_first 训练 ====================

    def train_context_to_first(self, dialogues, lr=0.05, n_iter=500, n_loops=1):
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
        """
        # ★ v12.3 连续累积 + 确定性快照 (训练/评估状态一致性)
        #   路径1 冻结快照: 所有对话用同一 W_coact → recall 相同 →
        #   状态无判别性 (实测 14 状态余弦 1.000) ✗
        #   路径2 连续累积: 每个对话状态含独特历史 (判别性好,
        #   同一状态集训练+评估 11/14), 但评估时继续累积 → 漂移
        #   (重编码评估 2/14) ✗
        #   修复: 记录每个对话 encode 前的 W_coact 快照, 评估时
        #   恢复对应快照 → 训练/评估状态完全一致, 且保留累积判别性 ✓
        ctx_data = []
        self._coact_snapshots = []
        for inp, resp in dialogues:
            resp_codes = self._text_to_codes(resp)
            if not resp_codes:
                continue
            # 记录"处理该对话前"的 W_coact (连续累积到当前)
            snap = self.W_coact.clone()
            _, acc_state = self.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
            # 检查状态是否有活跃神经元 (二值状态中 1 的数量)
            if acc_state.sum().item() == 0:
                continue
            # ★ 快照与 ctx_data 对齐: _coact_snapshots[i] ↔ ctx_data[i]
            #   (仅记录真正进入训练数据的对话, 评估时按同序恢复)
            self._coact_snapshots.append(snap)
            ctx_data.append((acc_state, resp_codes[0]))

        if not ctx_data:
            return

        # ★ v14: 资格迹模式需维护迹矩阵 (与 W 同形状)
        E = torch.zeros_like(self.W_ctx_to_first) if self.use_eligibility_trace else None

        for _ in range(n_iter):
            # ★ 随机打乱 — 模拟生物学学习的不确定性
            random.shuffle(ctx_data)
            for acc_state, first_code in ctx_data:
                # ★ v14: DG 稀疏分离 (启用时 top-k 二值稀疏化)
                feat = self._mem_feature(acc_state)
                target = torch.tensor(
                    [float((first_code >> j) & 1) for j in range(self.output_size)],
                    dtype=torch.float32, device=DEVICE
                )
                # ★ 纯二值阈值解码: 无 sigmoid, 无连续数值, 无 raw 返回值
                pred = self._binary_decode(self.W_ctx_to_first, feat)

                # ★ 奖赏预测误差 (RPE): RPE_j = target_j − pred_j ∈ {−1, 0, +1}
                #   +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
                pred_bits = (pred > 0.5).float()
                target_bits = (target > 0.5).float()
                rpe = target_bits - pred_bits

                # ★ v14: 统一更新入口 (即时 Hebbian 或三因子资格迹)
                self._hebbian_step(self.W_ctx_to_first, rpe, feat, pred_bits, lr, E)

    # ==================== 位置记忆头训练 (v13) ====================

    def train_pos_heads(self, dialogues, max_pos=64, lr=0.05, n_iter=500,
                        n_loops=1, batch_size=1):
        """训练位置记忆头 — 上下文状态 → 回复第 k 字符 (k = 0, 1, 2, ...)

        ★ 目标: "增加记忆层, 对非首字结果进行修正"。
          对回复的每个位置 k 训练一个独立 Hebbian 分类器 W_ctx_to_pos[k]
          (与 W_ctx_to_first 逐行同机制: 纯二值阈值解码, RPE 调制 Hebbian,
          无偏置更新, W clamp ±10)。

        ★ 快照方案与 train_context_to_first 完全一致:
          逐对话记录 encode 前 W_coact 快照 (连续累积保留判别性),
          _coact_snapshots[i] ↔ 训练样本 i, 评估时恢复对应快照。

        ★ v13.1: n_loops 自回归循环 (透传给 encode_text_lif), 默认 1。

        ★ v14.2 批量更新 (可选): batch_size > 1 时 mini-batch Hebbian,
          更新 = ΔW = lr × Σ(rpe ⊗ pre) 外积和, 规则公式不变,
          特征仅计算一次 (确定性 top-k); 资格迹不兼容 (自动回退)。

        Args:
            dialogues: 对话对列表 [(inp, resp), ...]
            max_pos: 位置记忆头上限 (超出部分生成时回退 W_seq)
            lr: 学习率
            n_iter: 每位置训练迭代次数
            n_loops: 每次输入自回归前向轮数 (v13.1, 默认 1)
            batch_size: 批量更新大小 (默认 1 = 原逐样本即时 Hebbian)
        """
        if batch_size > 1 and self.use_eligibility_trace:
            print("  [warn] 批量更新与资格迹不兼容 → 回退逐样本 (batch_size=1)")
            batch_size = 1
        # 收集 (acc_state, 回复字符码列表) — 复用 _coact_snapshots (v12.3)
        # ★ 必须恢复对应快照再编码: train_context_to_first 后 W_coact 已是
        #   最后一个对话的累积状态, 不恢复则位置头训练状态与生成/评估漂移。
        data = []
        for i, (inp, resp) in enumerate(dialogues):
            resp_codes = self._text_to_codes(resp)
            if not resp_codes:
                continue
            if i < len(self._coact_snapshots):
                self.W_coact = self._coact_snapshots[i].clone()
            _, acc_state = self.encode_text_lif(inp, update_memory=True, n_loops=n_loops)
            if acc_state.sum().item() == 0:
                continue
            data.append((acc_state, resp_codes))

        self.W_ctx_to_pos = []
        self.b_ctx_to_pos = []
        max_len = max((len(rc) for _, rc in data), default=0)
        for k in range(min(max_pos, max_len)):
            samples = [(st, rc[k]) for st, rc in data if len(rc) > k]
            if not samples:
                break
            W = torch.empty(self.output_size, self.hidden_size,
                            dtype=torch.float32, device=DEVICE)
            W.uniform_(-0.1, 0.1)
            b = torch.empty(self.output_size, dtype=torch.float32, device=DEVICE)
            b.uniform_(-0.1, 0.1)
            tgts = torch.tensor(
                [[float((code >> j) & 1) for j in range(self.output_size)]
                 for _, code in samples], dtype=torch.float32, device=DEVICE)
            # ★ v14: 每位置独立资格迹矩阵
            E = torch.zeros_like(W) if self.use_eligibility_trace else None
            idx = list(range(len(samples)))
            if batch_size > 1:
                # ★ v14.2 批量更新: 特征确定性, 预计算一次;
                #   ΔW = lr × Σ(rpe ⊗ pre) 外积和矩阵运算
                feats_all = self._mem_feature_batch(
                    torch.stack([samples[i][0] for i in idx]))
                for _ in range(n_iter):
                    random.shuffle(idx)
                    for start in range(0, len(samples), batch_size):
                        bi = idx[start:start + batch_size]
                        fb = feats_all[bi]
                        outs = (fb @ W.t() + b > 0).float()
                        rpes = tgts[bi] - outs
                        W += lr * (rpes.t() @ fb)
                        W.clamp_(-10.0, 10.0)
            else:
                # ★ 原逐样本即时 Hebbian (默认, batch_size=1)
                for _ in range(n_iter):
                    random.shuffle(idx)
                    for m in idx:
                        st, _ = samples[m]
                        # ★ v14: DG 稀疏分离 (训练/回忆一致)
                        feat = self._mem_feature(st)
                        out = (W @ feat + b > 0).float()
                        rpe = tgts[m] - out
                        # ★ v14: 统一更新入口 (即时 Hebbian 或三因子资格迹)
                        self._hebbian_step(W, rpe, feat, out, lr, E)
            self.W_ctx_to_pos.append(W)
            self.b_ctx_to_pos.append(b)

    # ==================== v14: DG 稀疏分离 + 三因子资格迹 ====================

    def _dg_separate(self, x):
        """★ DG 稀疏分离 (pattern separation) — top-k 二值稀疏化

        生物学依据: 齿状回 (DG) 将皮层输入的稠密重叠表征转换为稀疏、
        非重叠编码, 降低记忆间干扰 (Complementary Learning Systems;
        Schapiro 2017)。HiCL (2025) 在 CA3 前用 top-k 稀疏分离。

        作用: 两个不同状态经 top-k 后支持集重叠度期望降至 k²/N,
        串扰 (crosstalk) 大幅下降 → 提高关联记忆库容量 (直击
        experiment16 记忆库扩容崩溃)。

        约束合规: 输出纯二值 {0,1}; top-k 是离散选择运算, 不产生
        连续数值学习信号, 不违反"无连续数值信号"红线。

        Args:
            x: 输入状态 (256-dim, 分级 [0,1])

        Returns:
            二值稀疏向量 (256-dim, 恰好 k 个 1; 未启用时原样返回)
        """
        if not self.use_dg_separation:
            return x
        k = min(self.dg_k, x.numel())
        if k <= 0:
            return torch.zeros_like(x)
        flat = x.reshape(-1)
        _, idx = flat.topk(k)
        out = torch.zeros_like(x)
        out.reshape(-1)[idx] = 1.0
        return out

    def _mem_feature(self, state):
        """记忆头统一输入特征 — 训练/推理一致的 DG 分离入口

        所有记忆头 (W_ctx_to_first / W_ctx_to_pos) 的训练与回忆
        都必须经此入口, 保证启用 DG 分离时训练/推理状态逐位一致。
        """
        return self._dg_separate(state)

    def _mem_feature_batch(self, states):
        """批量版 _mem_feature — 按行 top-k 稀疏化, 与逐样本版逐位一致

        仅用于批量更新 (batch_size > 1) 的向量化实现: torch.topk(dim=1)
        每行独立, 数学结果与逐样本 _dg_separate 完全相同, 仅合并
        矩阵运算减少 GPU kernel 启动开销。
        """
        if not self.use_dg_separation:
            return states
        k = min(self.dg_k, states.shape[1])
        if k <= 0:
            return torch.zeros_like(states)
        out = torch.zeros_like(states)
        _, idx = states.topk(k, dim=1)
        out.scatter_(1, idx, 1.0)
        return out

    def _hebbian_step(self, W, rpe, pre, post, lr, E=None):
        """★ 统一权重更新: 即时 RPE 调制 Hebbian 或三因子资格迹

        即时模式 (原规则, E=None):
            Δw_ji = lr × RPE_j × pre_i
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
        """
        if E is None:
            for j in range(self.output_size):
                W[j] += lr * rpe[j] * pre
        else:
            for j in range(self.output_size):
                E[j] = self.eligibility_lambda * E[j] + pre * post[j]
                W[j] += lr * rpe[j] * E[j]
        W.clamp_(-10.0, 10.0)

    def pos_head_recall(self, state, k):
        """位置记忆头回忆: 状态 → 第 k 字符 (纯二值阈值解码 + margin 诊断)

        Returns:
            (code, margin): code = 回忆的 ASCII 码; margin = min_j |raw_j|
            是 8 个 bit 中解码最接近阈值 0 的 margin。
            ★ experiment14: margin 无判别力 (正确 med=0.09 vs 错误 med=0.05
            重叠) → 仅保留用于诊断, 不作为修正门控依据。
            超出已训练位置范围返回 (None, 0.0)。
        """
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

    def train_sequence(self, dialogues, lr=0.5, n_iter=1000, n_loops=1,
                       batch_size=1):
        """训练 W_seq — 奖赏预测误差调制 Hebbian 学习

        ★ 学习规则: Δw = lr × RPE_j × pre_activity
          - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差
            +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
          - pre_activity = 二值 {0, 1} 神经元输出 (来自 encode_text_lif)
          - ★ 无 center = clamp(raw, -1, 1): 连续数值运算, 已移除
          - ★ 无 b_seq 偏置更新: 连续数值运算, 已移除
          - 无 autograd，无反向传播，无批量处理，无目标误差

        ★ v14.2 批量更新 (可选, 默认 batch_size=1 完全等价原逻辑):
          batch_size > 1 时把逐样本循环向量化为 mini-batch Hebbian —
          batch 内共享当前 W 计算 RPE, 更新 = ΔW = lr × Σ_batch(rpe ⊗ pre)
          外积和 (矩阵运算), 学习规则公式逐样本不变, 无梯度/无损失。
          仅减少 GPU kernel 启动开销 (小张量逐样本调用是耗时主因)。
          与逐样本在线更新 (每个样本看到更新后的 W) 有 mini-batch
          语义差异; 资格迹模式不兼容 (自动回退 batch_size=1)。

        Args:
            dialogues: 对话对列表 [(inp, resp), ...]
            lr: 学习率
            n_iter: 训练迭代次数
            n_loops: 每次输入自回归前向轮数 (v13.1, 默认 1)
            batch_size: 批量更新大小 (默认 1 = 原逐样本即时 Hebbian)

        Returns:
            best_acc: 最佳预测准确率
        """
        # ★ v12.3 确定性快照: 与 train_context_to_first 相同方案 —
        #   逐对话记录 encode 前的 W_coact 快照 (连续累积保留),
        #   评估时恢复对应快照 + update_memory=True 重放 → 状态逐位一致。
        #   (旧方案: 预热后单冻结快照 → 所有对话共用同一 W_coact,
        #   状态判别性崩塌, 见 train_context_to_first 注释)
        seq_data = []
        self._seq_snapshots = []
        for inp, resp in dialogues:
            resp_codes = self._text_to_codes(resp)
            if len(resp_codes) < 2:
                continue
            # 记录"处理该回复前"的 W_coact (连续累积到当前)
            snap = self.W_coact.clone()
            # ★ v10: 取每步的工作记忆状态 (而非隐藏层输出)
            _, states, _ = self.encode_text_lif_states(resp, update_memory=True, n_loops=n_loops)
            if len(states) < 2:
                continue
            # ★ 快照与对话对齐: _seq_snapshots[i] ↔ 第 i 个有效对话
            #   (评估时按同序恢复, 与 Step 2 的 _coact_snapshots 一致)
            self._seq_snapshots.append(snap)
            for i in range(len(states) - 1):
                target = torch.tensor(
                    [float((resp_codes[i + 1] >> j) & 1) for j in range(self.output_size)],
                    dtype=torch.float32, device=DEVICE
                )
                seq_data.append((states[i], target))

        if not seq_data:
            return 0.0

        n_data = len(seq_data)

        # 初始评估 — 纯二值阈值解码
        correct = 0
        for fr, target in seq_data:
            out = self._binary_decode(self.W_seq, fr, self.b_seq)
            pred_bits = (out > 0.5).int()
            target_bits = (target > 0.5).int()
            if (pred_bits == target_bits).all().item():
                correct += 1
        best_acc = correct / n_data

        lr_current = lr
        # ★ v14: 资格迹模式需维护迹矩阵
        E = torch.zeros_like(self.W_seq) if self.use_eligibility_trace else None
        if batch_size > 1 and self.use_eligibility_trace:
            print("  [warn] 批量更新与资格迹不兼容 → 回退逐样本 (batch_size=1)")
            batch_size = 1
        for epoch in range(n_iter):
            # ★ 随机打乱训练顺序 — 模拟生物学学习的不确定性
            random.shuffle(seq_data)

            correct_count = 0
            if batch_size > 1:
                # ★ v14.2 批量更新 (可选): batch 内共享当前 W 计算 RPE,
                #   更新 = ΔW = lr × Σ_batch(rpe ⊗ pre) (外积和矩阵运算,
                #   规则公式逐样本不变, 仅合并 kernel 减少启动开销)
                for start in range(0, n_data, batch_size):
                    batch = seq_data[start:start + batch_size]
                    feats = self._mem_feature_batch(
                        torch.stack([fr for fr, _ in batch]))
                    tgt = torch.stack([t for _, t in batch])
                    # 与 _binary_decode 逐位一致: (W·x + b > 0).float()
                    outs = (feats @ self.W_seq.t() + self.b_seq > 0).float()
                    rpes = tgt - outs
                    self.W_seq += lr_current * (rpes.t() @ feats)
                    self.W_seq.clamp_(-10.0, 10.0)
                    correct_count += (outs == tgt).all(dim=1).sum().item()
            else:
                # ★ 原逐样本即时 Hebbian (默认, batch_size=1)
                for fr, target in seq_data:
                    # ★ v14: DG 稀疏分离 (启用时 top-k 二值稀疏化)
                    feat = self._mem_feature(fr)
                    # fr 是二值 {0, 1} (来自 encode_text_lif)
                    # ★ 纯二值阈值解码: 无 sigmoid, 无连续数值, 无 raw
                    out = self._binary_decode(self.W_seq, feat, self.b_seq)

                    # ★ 奖赏预测误差 (RPE): RPE_j = target_j − out_j ∈ {−1, 0, +1}
                    pred_bits = (out > 0.5).float()
                    target_bits = (target > 0.5).float()
                    rpe = target_bits - pred_bits

                    # ★ v14: 统一更新入口 (即时 Hebbian 或三因子资格迹)
                    #   Δw_ji = lr × RPE_j × pre_activity (即时, v11)
                    #   或 三因子: 迹 e_ji ← λe_ji + pre×post, Δw = lr × M_j × e_ji
                    self._hebbian_step(self.W_seq, rpe, feat, pred_bits, lr_current, E)

                    # 统计全部正确的样本数
                    if (pred_bits == target_bits).all().item():
                        correct_count += 1

            acc = correct_count / n_data
            if acc > best_acc:
                best_acc = acc
            if (epoch + 1) % 200 == 0:
                lr_current *= 0.9

        return best_acc

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

        # 输入: 随机稠密编码 (二值)
        input_vecs_gpu = torch.zeros(n_vocab, self.hidden_size, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(train_codes):
            ch = chr(c) if 32 <= c <= 126 else '?'
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

                correct_count = 0
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
                    n_correct = int((pred_bits == target_bits).sum().item())

                    # 标量奖赏预测误差 (层间全局投射): 位级 RPE 的平均
                    mean_rpe = float(rpe.mean().item())

                    # 4a. 更新 W_h2o — 逐位奖赏预测误差调制 Hebbian (从汇聚层输出学习)
                    #     Δw = lr_out × RPE_j × out_peak
                    #     ★ v11: RPE 直接门控可塑性, "应发未发"获得强化路径
                    for j in range(output_size):
                        self.W_h2o[j] += lr_out * rpe[j] * out_peak
                    self.W_h2o.clamp_(-10.0, 10.0)

                    # 4b. 更新已激活的层间权重 — 标量奖赏预测误差调制 Hebbian
                    #     ΔW_lk = lr_layer × max(mean_RPE, 0) × out_{k+1} ⊗ out_k
                    #     ★ v11 奖赏门控 (只强化不惩罚): 全局惩罚时若削弱
                    #       所有活跃突触, 层间权重会单调收缩到 0 (网络死亡)
                    if mean_rpe > 0:
                        for m in range(depth - 1):
                            self.W_deep[m] += lr_layer * mean_rpe * torch.outer(outs[m + 1], outs[m])
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
                           pos_margin_thresh=0.0, n_loops=1, event_guide=None):
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
        first_bits = self._binary_decode(self.W_ctx_to_first, cf_feat)
        first_code = 0
        for j in range(self.output_size):
            if first_bits[j] >= 0.5:  # first_bits ∈ {0.0, 1.0}
                first_code |= (1 << j)

        ch = chr(first_code) if 32 <= first_code <= 126 else '?'
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
            vec = self._char_to_8bit(ch) + self.input_bias
            output = self._multi_layer_forward(vec, n_loops=n_loops)  # ★ v10: 4 层前向

            # ★ v10: 关联记忆层更新 (共发放追踪) — 先学习(写)再回忆(读)
            # ★ v12.2 顺序回退: 与 encode_text_lif 一致 (实验9 now 模式)
            # ★ v12.3: update_memory=False 时冻结 W_coact (评估/生成一致性)
            if update_memory:
                self.update_coactivation(output)
            recall = self.recall_from_memassoc(output)

            # ★ v10: 工作记忆层更新 — 当前活动 + 关联记忆回忆
            # ★ v11: 移除跨字符 max 累积, 与 encode_text_lif 一致 (now 模式)
            # ★ v12.2 回退: v_peak 恢复为累积膜电位 V, 不 reset
            v_peak = self.V_deep[-1] if self.num_layers > 1 else self.V
            state = torch.max(v_peak, recall)
            self.MemWork = state

            # ★ W_seq(工作记忆) → 下一字符 (纯二值阈值解码)
            # ★ v14: DG 稀疏分离 (与 train_sequence 一致)
            next_bits = self._binary_decode(self.W_seq, self._mem_feature(state), self.b_seq)
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
            elif use_pos_memory and self.W_ctx_to_pos:
                mem_code, margin = self.pos_head_recall(cf, step)
                if mem_code is not None and margin > pos_margin_thresh:
                    next_code = mem_code

            next_ch = chr(next_code) if 32 <= next_code <= 126 else '?'

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

    # ==================== 在线学习 ====================

    def train_on_dialogue(self, inp, resp, lr=0.05, n_iter=30):
        """在线学习单个对话对 — 奖赏预测误差调制 Hebbian 增量更新

        ★ 学习规则: Δw = lr × RPE_j × pre_activity
          - RPE_j = target_j − out_j ∈ {−1, 0, +1} — 奖赏预测误差
            +1: 应发未发 → 强化; −1: 误发 → 削弱; 0: 预测正确 → 无更新
          - pre_activity = 二值 {0, 1} 突触前活动
          - ★ 无 center = clamp(raw, -1, 1): 连续数值运算, 已移除
          - ★ 无偏置更新: 连续数值运算, 已移除
          - 可在推理过程中随时调用，不需要重新训练整个数据集
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