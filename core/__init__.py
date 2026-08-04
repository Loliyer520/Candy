"""
core — 生物脉冲神经网络 (Bio Neural Net) 成品内核

纯生物脉冲神经网络 (SNN) 核心:
  - lif_pytorch.py    : 0-1 膜电位神经元引擎 (LIF + 四层前馈 + 双记忆层 + 循环生成)
  - episodic_memory.py: 事件记忆层 (P1 整体事件 + v2 自回归事件记忆, 纯联想)
  - trainer.py        : 训练流程 + 对话推理 + 模型持久化

学习规则 (项目红线):
  仅允许奖赏预测误差调制 Hebbian (Δw = lr × RPE × pre, RPE ∈ {−1,0,+1});
  禁止梯度/BP/损失函数/批量优化/sigmoid/softmax/余弦检索。
"""

from .lif_pytorch import (
    TorchLIFSimulator,
    RecurrentLIFSimulator,
    train_w_h2o_stdp_gpu,
    DEVICE,
)
from .episodic_memory import (
    EpisodicEventMemory,
    AutoRegressiveEventMemory,
)
from .trainer import (
    DIALOGUES,
    train_full,
    RecurrentTrainer,
    save_model,
    load_model,
)

__all__ = [
    "TorchLIFSimulator",
    "RecurrentLIFSimulator",
    "train_w_h2o_stdp_gpu",
    "DEVICE",
    "EpisodicEventMemory",
    "AutoRegressiveEventMemory",
    "DIALOGUES",
    "train_full",
    "RecurrentTrainer",
    "save_model",
    "load_model",
]
