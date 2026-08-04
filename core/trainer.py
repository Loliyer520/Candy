"""
core/trainer.py — 生物脉冲神经网络成品训练器封装

提供:
  - DIALOGUES: 内置对话训练集 (14 对, 训练验证过的规模)
  - train_full(): 完整训练流程 (字符解码 → 四层渐进 → 记忆头 → W_seq)
  - RecurrentTrainer: 对话推理封装 (生成回复)
  - save_model() / load_model(): 模型持久化 (models/*.spt)

学习规则全程为奖赏预测误差调制 Hebbian (Δw = lr × RPE × pre),
RPE ∈ {−1, 0, +1}, 无梯度/BP/损失函数/批量优化 (项目红线)。
"""

import os
import torch

from .lif_pytorch import (
    RecurrentLIFSimulator,
    train_w_h2o_stdp_gpu,
    DEVICE,
)

__all__ = [
    "DIALOGUES",
    "train_full",
    "RecurrentTrainer",
    "save_model",
    "load_model",
]

# ============================================================
# 内置对话数据 (14 对 — 训练验证过的最佳规模)
# ============================================================
DIALOGUES = [
    # 问候
    ("Hi!", "Hello!"),
    ("Hello!", "Hi there! How are you?"),
    ("How are you?", "I am doing well, thanks!"),
    ("What is your name?", "My name is Candy, nice to meet you!"),
    ("Who you are?", "I'm Candy, your AI assistant!"),
    # 知识
    ("What is 7 times 8?", "7 times 8 equals 56."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    # 情绪
    ("I feel sad today.", "I'm sorry to hear that. Would you like to talk?"),
    ("I am so happy!", "That's wonderful! I'm glad you're feeling great today!"),
    # 告别/感谢
    ("Goodbye my friend.", "See you later! Take good care!"),
    ("Thank you so much.", "You're welcome friend. Happy to help!"),
    # 闲聊
    ("What's up?", "Not much, just relaxing. How about you?"),
    ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side!"),
]


# ============================================================
# 完整训练流程 (对应 test_recurrent_learning.py Step 1→3)
# ============================================================
def train_full(dialogues=None, hidden_size=256, num_layers=4,
               decode_epochs=200, seq_iters=200, verbose=True):
    """训练完整生物脉冲网络, 返回训练好的 RecurrentLIFSimulator。

    流程:
      Step 1   : W_h2o 字符解码 (奖赏调制 Hebbian)
      Step 1.5 : 四层渐进式层间训练 (L1→L4, 新层恒等初始化)
      Step 2   : W_ctx_to_first 上下文→首字符
      Step 2.5 : W_ctx_to_pos 位置记忆头 (修正非首字)
      Step 3   : W_seq 序列转移 (奖赏调制 Hebbian)

    注:
      - W_seq 结构性不可解 (experiment11, README v13), 默认 200 iters
        仅保留学习形态, 端到端质量依赖位置头修正 (Step 2.5)。
      - 训练后保留 _coact_snapshots/_seq_snapshots — 库内记忆场景
        (快照恢复) 字符级 96.5% 依赖它们 (README v13 Step 4)。
    """
    dialogues = DIALOGUES if dialogues is None else dialogues

    sim = RecurrentLIFSimulator(
        hidden_size=hidden_size, output_size=8, input_bias=1.0,
        leak=0.1, threshold=0.5, reset_factor=0.3, inhibition_strength=0.2,
        num_layers=num_layers,
        use_dg_separation=True, dg_k=64,   # v14 推荐配置 (稀疏分离)
    )
    sim.init_random_weights(scale=0.8, connection_sparsity=0.5)

    basic_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'\":;-"
    train_codes = [ord(c) for c in basic_chars]

    if verbose:
        print(f"[Step 1] 训练字符解码 (奖赏调制 Hebbian, {decode_epochs} epochs)...", flush=True)
    train_w_h2o_stdp_gpu(sim, train_codes, num_epochs=decode_epochs, verbose=verbose)

    if verbose:
        print(f"[Step 1.5] 四层渐进式层间训练 ...", flush=True)
    sim.train_multi_layer_stdp(train_codes, num_epochs=decode_epochs,
                               lr_layer=0.3, lr_out=0.5, verbose=verbose)

    if verbose:
        print(f"[Step 2] 训练 W_ctx_to_first ({len(dialogues)} 对话)...", flush=True)
    sim.train_context_to_first(dialogues, lr=0.05, n_iter=500)

    if verbose:
        print(f"[Step 2.5] 训练位置记忆头 (修正非首字)...", flush=True)
    sim.train_pos_heads(dialogues, lr=0.05, n_iter=500)

    if verbose:
        print(f"[Step 3] 训练 W_seq ({seq_iters} iters, 结构性不可解仅保留形态)...", flush=True)
    best_acc = sim.train_sequence(dialogues, lr=0.5, n_iter=seq_iters)
    if verbose:
        print(f"  W_seq best_acc = {best_acc:.1%} (已知限制 ~5-22%)", flush=True)
    return sim


# ============================================================
# 对话推理封装
# ============================================================
class RecurrentTrainer:
    """循环网络训练/推理器 — 0-1 膜电位神经元 + 记忆层循环生成。

    所有学习使用奖赏预测误差调制 Hebbian (RPE ∈ {−1,0,+1}),
    解码使用纯二值阈值 (W·x + b > 0).float(), 无 sigmoid/softmax。
    """

    def __init__(self, sim, dialogues=None):
        self.sim = sim
        self.dialogues = [] if dialogues is None else list(dialogues)

    def train_on_dialogue(self, inp, resp, n_iter=20):
        """在线学习单个对话对 — 存储 + 奖赏调制 Hebbian 增量更新"""
        self.dialogues.append((inp, resp))
        self.sim.train_on_dialogue(inp, resp, lr=0.05, n_iter=n_iter)

    def generate_response(self, input_text, max_steps=30, use_pos_memory=True):
        """生成回复 (开放链路) — 纯二值神经元循环生成, 无匹配, 无 sigmoid。

        注: 开放链路受 v13 已知限制 (无快照 → 状态漂移/记忆幻觉),
        库内记忆场景请用 memory_replay_response。
        """
        _, context_state = self.sim.encode_text_lif(input_text, update_memory=True)
        if context_state.sum().item() == 0:
            return "", 0.0
        result = self.sim.generate_recurrent(
            context_state, max_steps, max_repeat=3,
            update_memory=True, use_pos_memory=use_pos_memory)
        confidence = min(1.0, len(result) / max_steps) if result else 0.0
        return result, confidence

    def memory_replay_response(self, inp, resp, max_steps=30,
                               use_pos_memory=True):
        """库内记忆场景: 恢复该对话的训练快照 (W_coact) 后重放。

        ★ README v13 Step 4: 快照一致后字符级 96.5% (位置头修正)。
        快照 = 处理该回复前的 W_coact, 排除后续对话污染。
        返回 (生成文本, 置信度); 对话不在训练库时返回 (None, 0.0)。
        建议调用方传 max_steps=len(resp), 避免超目标长度生成。
        """
        sim = self.sim
        snapshots = getattr(sim, "_coact_snapshots", None)
        if snapshots is None or len(snapshots) != len(self.dialogues):
            return None, 0.0
        try:
            idx = next(i for i, (a, b) in enumerate(self.dialogues)
                       if a == inp and b == resp)
        except StopIteration:
            return None, 0.0
        # 与 train_pos_heads/train_context_to_first 完全一致:
        # 恢复该对话快照 → encode(inp) → 上下文状态 (无需重放回复)
        sim.W_coact = snapshots[idx].clone()
        _, context_state = sim.encode_text_lif(inp, update_memory=True)
        if context_state.sum().item() == 0:
            return "", 0.0
        result = sim.generate_recurrent(
            context_state, max_steps, max_repeat=3,
            update_memory=True, use_pos_memory=use_pos_memory)
        confidence = min(1.0, len(result) / max_steps) if result else 0.0
        return result, confidence


# ============================================================
# 模型持久化 (models/*.spt)
# ============================================================
def save_model(sim, path):
    """保存训练好的仿真器 (整对象 pickle, 含全部权重)"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(sim, path)
    return path


def load_model(path, device=None):
    """加载保存的仿真器到目标设备"""
    device = device or DEVICE
    sim = torch.load(path, map_location=device, weights_only=False)
    return sim
