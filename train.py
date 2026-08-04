#!/usr/bin/env python
"""
train.py — 生物脉冲神经网络训练入口

用法:
  python train.py                      # 训练内置 14 对话, 保存到 models/candy.spt
  python train.py --name mybot         # 自定义模型名
  python train.py --epochs 300         # 调整解码训练轮数

训练流程 (全部为奖赏预测误差调制 Hebbian, 无梯度):
  Step 1 字符解码 → Step 1.5 四层渐进 → Step 2 首字符记忆头
  → Step 2.5 位置记忆头 → Step 3 序列转移 → 保存 models/*.spt
"""

import argparse
import os
import time

from core import train_full, save_model, DIALOGUES, RecurrentTrainer


def main():
    ap = argparse.ArgumentParser(description="生物脉冲神经网络训练入口")
    ap.add_argument("--name", default="candy", help="模型名 (保存为 models/<name>.spt)")
    ap.add_argument("--models-dir", default="models", help="模型输出目录")
    ap.add_argument("--epochs", type=int, default=200, help="字符解码训练轮数")
    ap.add_argument("--hidden-size", type=int, default=256, help="隐藏层神经元数")
    ap.add_argument("--layers", type=int, default=4, help="隐藏层层数")
    ap.add_argument("--quiet", action="store_true", help="关闭详细输出")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 60, flush=True)
    print("  生物脉冲神经网络训练 — 奖赏调制 Hebbian (无梯度)", flush=True)
    print("=" * 60, flush=True)

    sim = train_full(
        dialogues=DIALOGUES,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        decode_epochs=args.epochs,
        verbose=not args.quiet,
    )

    # 训练后快速自检: 库内记忆场景 (快照恢复, 复现 README v13 96.5%)
    trainer = RecurrentTrainer(sim, dialogues=DIALOGUES)
    print("\n[自检] 库内记忆场景 (位置头修正, 快照恢复):", flush=True)
    n_ok = 0
    for inp, resp in DIALOGUES:
        result, conf = trainer.memory_replay_response(inp, resp, max_steps=len(resp))
        mark = "OK" if result == resp else "miss"
        n_ok += (result == resp)
        print(f"  用户: {inp}\n  Bot : {result[:60]}  [{mark}, conf={conf:.2f}]",
              flush=True)
    print(f"  完整复述: {n_ok}/{len(DIALOGUES)}", flush=True)

    path = save_model(sim, os.path.join(args.models_dir, args.name + ".spt"))
    print(f"\n[完成] 模型已保存: {path}  (总耗时 {time.perf_counter()-t0:.0f}s)", flush=True)
    print("运行 python chat.py --model %s 开始对话" % path.replace("\\", "/"), flush=True)


if __name__ == "__main__":
    main()
