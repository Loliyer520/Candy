"""
experiment15b.py — 0-1 膜电位自回归循环的饱和机制验证 (v13.1)

假说: n_loops=2 解码崩溃 (1/72) 的根因是 V 膜电位跨轮次累积 → 迅速
饱和到 clamp 上限 1 → 几乎所有神经元持续发放 → 输出失去判别性
(信息坍缩到全 1), 与"输入内容"无关。

验证: 随机初始化的 8 层网络, 对随机输入跑多轮前向, 统计每轮后
隐藏层输出/膜电位的活跃率 (V>0.5 比例)。若活跃率单调逼近 100%,
则证实饱和坍缩机制。
"""

import sys, os, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lif_pytorch import RecurrentLIFSimulator, DEVICE

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

H = 256


def main():
    for num_layers in (4, 8):
        sim = RecurrentLIFSimulator(hidden_size=H, output_size=8,
                                    input_bias=1.0, leak=0.1, threshold=0.5,
                                    reset_factor=0.3, inhibition_strength=0.2,
                                    num_layers=num_layers)
        sim.init_random_weights(scale=0.8, connection_sparsity=0.5)
        print(f"\nnum_layers={num_layers}:", flush=True)
        for trial in range(3):
            sim.reset_state()
            vec = torch.rand(H, dtype=torch.float32, device=DEVICE)
            vec = (vec > 0.5).float()  # 随机二值输入
            acts = []
            out = vec
            for loop in range(4):
                for k in range(num_layers):
                    out = sim._layer_forward(k, out)
                act = float((out > 0.5).float().mean().item())
                v_mean = float(sim.V_deep[-1].flatten().mean().item())
                acts.append(f"loop{loop+1}:out_act={act:.3f} v_mean={v_mean:.3f}")
            print(f"  trial{trial}: " + " | ".join(acts), flush=True)


if __name__ == "__main__":
    main()
