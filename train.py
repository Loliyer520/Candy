#!/usr/bin/env python
"""
train.py — 生物脉冲神经网络训练入口

用法:
  python train.py                              # 训练内置英文 14 对话 (hidden 256)
  python train.py --zh --n-dialogues 200       # 训练中文预料 (hidden 1024, external codec)

v14.5 中文模式 (--zh):
  - 中文通过 zh_codec.encode() 编码为纯 ASCII 序列后训练网络
  - 网络核心 (lif_pytorch.py) 保持纯 8-bit ASCII 不变
  - 参数 hidden 1024、结构化编码 (ASCII bit 前缀 + 随机部分)
"""

import argparse
import os
import time

from core import train_full, save_model, DIALOGUES, RecurrentTrainer
from core.trainer import load_zh_dialogues
from core.zh_codec import encode, decode


def main():
    ap = argparse.ArgumentParser(description="生物脉冲神经网络训练入口")
    ap.add_argument("--name", default=None, help="模型名 (保存为 models/<name>.spt)")
    ap.add_argument("--models-dir", default="models", help="模型输出目录")
    ap.add_argument("--epochs", type=int, default=200, help="字符解码训练轮数")
    ap.add_argument("--hidden-size", type=int, default=None, help="隐藏层神经元数")
    ap.add_argument("--layers", type=int, default=4, help="隐藏层层数")
    ap.add_argument("--quiet", action="store_true", help="关闭详细输出")
    ap.add_argument("--zh", action="store_true", help="中文模式 (zh_codec 编码为 ASCII)")
    ap.add_argument("--corpus", default=r"D:\Doc\AI\Candy4\res\基础预料.txt",
                    help="中文预料 txt 路径")
    ap.add_argument("--n-dialogues", type=int, default=200, help="抽样中文对话对数")
    ap.add_argument("--seq-iters", type=int, default=200, help="W_seq 训练迭代")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 60, flush=True)
    print("  生物脉冲神经网络训练 — 奖赏调制 Hebbian (无梯度)", flush=True)
    print("=" * 60, flush=True)

    if args.zh:
        print(f"[语料] 加载中文对话 (抽样 {args.n_dialogues} 对)...", flush=True)
        raw_dialogues = load_zh_dialogues(
            args.corpus, n=args.n_dialogues, user_max=120, resp_min=4, resp_max=200)
        print(f"[语料] 实际加载 {len(raw_dialogues)} 对", flush=True)
        dialogues = [(encode(inp), encode(resp)) for inp, resp in raw_dialogues]
        print(f"[编码] 示例: {raw_dialogues[0][0][:20]} → {dialogues[0][0][:30]}", flush=True)
        hidden_size = args.hidden_size if args.hidden_size else 1024
        name = args.name if args.name else "candy_zh"
    else:
        dialogues = DIALOGUES
        hidden_size = args.hidden_size if args.hidden_size else 256
        name = args.name if args.name else "candy"

    sim = train_full(
        dialogues=dialogues,
        hidden_size=hidden_size,
        num_layers=args.layers,
        decode_epochs=args.epochs,
        seq_iters=args.seq_iters,
        verbose=not args.quiet,
    )

    # 训练后快速自检
    trainer = RecurrentTrainer(sim, dialogues=dialogues)
    print("\n[自检] 库内记忆场景 (位置头修正, 快照恢复):", flush=True)
    n_ok = 0
    for inp, resp in dialogues[:min(10, len(dialogues))]:
        result, conf = trainer.memory_replay_response(inp, resp, max_steps=len(resp))
        decoded = decode(result) if result and args.zh else result
        orig_resp = decode(resp) if args.zh else resp
        mark = "OK" if result == resp else "miss"
        n_ok += (result == resp)
        print(f"  Orig: {orig_resp[:30]}\n  Bot : {decoded[:30]}  [{mark}]", flush=True)
    print(f"  完整复述: {n_ok}/{min(10, len(dialogues))}", flush=True)

    path = save_model(sim, os.path.join(args.models_dir, name + ".spt"))
    print(f"\n[完成] 模型已保存: {path}  (总耗时 {time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"运行: python chat.py --model {path.replace(chr(92), '/')}", flush=True)


if __name__ == "__main__":
    main()