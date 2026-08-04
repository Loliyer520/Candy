"""
experiment15c.py — 循环趋同假说验证 (v13.1)

假说修正: L8+循环解码崩溃 (1/72) 不是"V 饱和到全发放" (experiment15b 已
证伪: 活跃率稳定 ~0.45), 而是"不同输入经循环后输出趋同到吸引子" —
信息被逐轮抹平, 判别性丧失 → 72 字符全部解码为同一字符。

验证: 随机初始网络, 对两个不同随机输入分别跑多轮前向, 统计每轮后
两输出的汉明距离。若距离随循环轮数单调下降 → 趋同假说成立。
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


def forward_loops(sim, vec, n_loops, num_layers):
    out = vec
    for _ in range(n_loops):
        for k in range(num_layers):
            out = sim._layer_forward(k, out)
    return out


def main():
    for num_layers in (4, 8):
        print(f"\nnum_layers={num_layers}:", flush=True)
        sim_a = RecurrentLIFSimulator(hidden_size=H, output_size=8, input_bias=1.0,
                                      leak=0.1, threshold=0.5, reset_factor=0.3,
                                      inhibition_strength=0.2, num_layers=num_layers)
        sim_b = RecurrentLIFSimulator(hidden_size=H, output_size=8, input_bias=1.0,
                                      leak=0.1, threshold=0.5, reset_factor=0.3,
                                      inhibition_strength=0.2, num_layers=num_layers)
        sim_a.init_random_weights(scale=0.8, connection_sparsity=0.5)
        sim_b.init_random_weights(scale=0.8, connection_sparsity=0.5)
        # 同一权重 (拷贝), 不同输入
        sim_b.W_deep = [w.clone() for w in sim_a.W_deep]
        sim_b.W_h2o = sim_a.W_h2o.clone()
        sim_b.b_o = sim_a.b_o.clone()

        for trial in range(3):
            a = (torch.rand(H, device=DEVICE) > 0.5).float()
            b = (torch.rand(H, device=DEVICE) > 0.5).float()
            hd0 = int((a != b).sum().item())
            sim_a.reset_state(); sim_b.reset_state()
            out_a, out_b = a, b
            dists = [f"loop0:hd={hd0}"]
            for loop in range(4):
                out_a = forward_loops(sim_a, out_a, 1, num_layers)
                out_b = forward_loops(sim_b, out_b, 1, num_layers)
                hd = int((out_a != out_b).sum().item())
                dists.append(f"loop{loop+1}:hd={hd}")
            print(f"  trial{trial}: " + " | ".join(dists), flush=True)


if __name__ == "__main__":
    main()
