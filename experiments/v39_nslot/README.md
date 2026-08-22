# v3.9 — N-slot 多块路由长文本续写 (2026-08-22)

在纯 spiking LIF 框架 (core/lif_v36.py) 上, 以 **块 (slot) 路由** 实现
60 字长文本续写复述与组合泛化。核心交付模块为项目根 `nslot_v39.py`.

## 分块可行性结论 (P52/P53 证伪 + 弱意义落地)
- **强意义自主分块 (无监督涌现块边界)** 被系统性证伪: 预测误差 margin /
  单尺度新颖度 / 双时间尺度差分 / TCM 漂移转向 4 类信号全灭。
  根本原因: chunk 是语法/语义抽象, 字符级动力学不含块级表征。
- **弱意义 (学习块边界头决策)** 可行: 块边界 = slot 转移 = `W_done` 判定。
- 因此 v3.9 采用 **N-slot 路由多块架构**: 每个 slot 一个语义块,
  块边界 = slot 转移 = `W_done` 判定, 块内 v3.7 TCM+theta, 块间 slot 路由。

## 读出头体系 (全 margin 斜坡 Hebbian 学习, 无梯度/无查表/无规则匹配)
- `W_first` 句首 / `W_first_t[an/adj]` 类型分离词首 / `W_first_last` 末slot首空格
- `W_cont` 统一续写头 `[wsrc+proj(剥离计数)]⊕onehot3(协议类)`
- `W_done` 统一段完成头 `[wsrc+proj(原始计数)]⊕onehot3(协议类)`
- `W_tmpl` 模板链 / `W_role` 模板态判定 / `W_slot_proj` 计数投影(scale放大)

## 修复链
- P54 基线: 分离头 + proj scale 1/16 → A 25/30+3/6
- P54b: 计数通道放大 + 协议类共享头 → A 30/30+5/6
- P54c: 统一 cont 头 (跨协议拼写迁移) → A 30/30+6/6
- P54d: 统一 done 头 + '!' 重分配模板 (终止轴跨协议迁移)
  → C 相 (8-slot, 76-82字) **留出 6/6 + 拼写 10/10 + 前缀 9/9**

## 验证结果
| 实验 | 块/槽 | 目标长度 | 训练内 | 留出组合 |
|---|---|---|---|---|
| P54d A | 2块/4slot | 35-39字 | 30/30 | 6/6 |
| P54d B | 3块/6slot | 55-60字 | 30/30 | 4/6 |
| P54d C | 4块/8slot | 76-82字 | 32/32 | **6/6** |
| P55 (200条) | 4块/8slot | 76-80字 | 200/200 | **23/24** |

## P55 (200 条) 关键结果
- 扩充词表 15 形容词 × 15 动物, 200 训练 + 24 留出, 4 块 (8slot) 76-80 字
- 训练内 FULL 200/200; 留出 23/24 (95.8% 组合泛化); 前缀续写 18/18
- 所有 slot 15/15 词全覆盖 → 避免 P54 覆盖缺口, it=10 六头全收敛 1.000
- 唯一留出失败 'fast cat ... smart bear': done@step71 提前, free-run 漂移跳过
  ' is ' 模板段 (前缀探针全对, 拼写 bear@mid OK, 纯 free-run 边界

## 复现配方
1. 训练基础模型 (P37 配方): state_kwt_k=48, dir_quota_k=16, 句WTA
2. `from nslot_v39 import NSlotGenerator, build_blocks_corpus`
   - `train, held = build_blocks_corpus(n_blocks=4, n_train=200, n_held=24, seed=13)`
   - `NSlotGenerator.from_base(_p37.pt, train, held).train(iters=60).save(out.pt)`
3. `gen.generate("big deer green hare calm fox fast wolf")`
   -> `'the deer is big and the hare is green and the fox is calm and the wolf is fast!'`

注: 模型 pickle 引用 slot_route_v38.R36NoPos, 需同目录有 slot_route_v38.py。


---

## r1 对外接口 (candyfish.py)

v3.9 封装为顶层 API `CandyFish` (项目根 `candyfish.py`), 可从外部直接调用。
r1 相对裸用 nslot_v39 的改进: 路由元数据 (sigma/types/n_roles) 内嵌模型文件,
加载不再需要重建同 seed 语料。

```python
from candyfish import CandyFish

# 训练新模型 (从 P37 基模型)
train, held = CandyFish.build_corpus(n_blocks=4, n_train=200, n_held=24, seed=13)
cf = CandyFish.train_new(train, held=held, iters=60)
cf.save("model.pt")

# 加载 (无需 corpus)
cf = CandyFish.load("model.pt")

# 续写 / 前缀续写
cf.continue_text("big deer green hare calm fox fast wolf")
cf.continue_text("big deer green hare calm fox fast wolf", prefix="the deer is ")

# 继续训练 / 评估
cf.train_more(train, iters=5)
cf.evaluate(held)
```

冒烟验收 (_p57): 无 corpus 加载 + 续写与裸调用完全一致 (留出 23/24);
train_new/train_more/save/reload/输入校验全通过。
