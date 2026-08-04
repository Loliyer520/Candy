#!/usr/bin/env python
"""
chat.py — 生物脉冲神经网络对话入口

用法:
  python chat.py --model models/candy.spt                 # 交互对话 (开放链路)
  python chat.py --model models/candy.spt --once "Hello"  # 单次查询
  python chat.py --model models/candy.spt --replay        # 库内记忆场景演示

生成机制: 0-1 膜电位神经元循环生成 + 位置记忆头修正 (无 LLM/检索)。
已知限制 (README v13): 开放链路无快照 → 状态漂移/记忆幻觉; 库内记忆场景
(快照恢复) 字符级 96.5% 是当前可靠能力 (--replay 演示)。
"""

import argparse

from core import load_model, RecurrentTrainer, DIALOGUES, DEVICE


def main():
    ap = argparse.ArgumentParser(description="生物脉冲神经网络对话入口")
    ap.add_argument("--model", default="models/candy.spt", help="模型文件路径")
    ap.add_argument("--once", default=None, help="单次输入 (不进入交互模式)")
    ap.add_argument("--replay", action="store_true",
                    help="库内记忆场景演示 (快照恢复, 复述训练对话)")
    ap.add_argument("--max-steps", type=int, default=30, help="最大生成步数")
    ap.add_argument("--no-pos-memory", action="store_true", help="关闭位置记忆头修正")
    args = ap.parse_args()

    print(f"[加载] {args.model} → {DEVICE} ...", flush=True)
    sim = load_model(args.model)
    trainer = RecurrentTrainer(sim, dialogues=DIALOGUES)
    use_pos = not args.no_pos_memory
    print("[就绪] 生物脉冲神经网络 (0-1 膜电位 + 奖赏调制 Hebbian)", flush=True)

    if args.replay:
        print("\n[库内记忆场景] 快照恢复 + 位置头修正:", flush=True)
        n_ok = 0
        for inp, resp in DIALOGUES:
            result, conf = trainer.memory_replay_response(
                inp, resp, max_steps=len(resp), use_pos_memory=use_pos)
            mark = "OK" if result == resp else "miss"
            n_ok += (result == resp)
            print(f"  用户: {inp}\n  Bot : {result[:60]}  [{mark}, conf={conf:.2f}]",
                  flush=True)
        print(f"  完整复述: {n_ok}/{len(DIALOGUES)}", flush=True)
        return

    if args.once is not None:
        result, conf = trainer.generate_response(
            args.once, max_steps=args.max_steps, use_pos_memory=use_pos)
        print(f"用户: {args.once}", flush=True)
        print(f"Bot [conf={conf:.2f}]: {result}", flush=True)
        return

    print("\n[开放链路] 输入 'quit' 退出对话 (已知限制: 无快照 → 状态漂移)",
          flush=True)
    while True:
        try:
            user_input = input("你: ").strip()
            if user_input.lower() in ("quit", "exit", "q", "退出"):
                break
            if not user_input:
                continue
            result, conf = trainer.generate_response(
                user_input, max_steps=args.max_steps, use_pos_memory=use_pos)
            print(f"Bot [conf={conf:.2f}]: {result[:120]}", flush=True)
        except (EOFError, KeyboardInterrupt):
            break
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
