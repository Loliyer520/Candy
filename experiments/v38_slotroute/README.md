# v3.8 — slot 角色路由组合泛化 (2026-08-22)

在纯 spiking LIF 框架 (core/lif_v36.py) 上解决组合泛化的 binding problem:
位置 × 内容的解耦. 核心交付模块为项目根 `slot_route_v38.py`.

## 架构 (Greff 2020: segregation-representation-composition 落地)
- **词级分离编码 encode_words**: 每词独立 reset 编码取 `_last_goal`, slot 头只读对应词表征
- **空格双侧归 slot 协议**: slot1=词+尾空格('fox '), slot2=首空格+词+尾'!'(' big!')
- **六个专用读出头** (全 margin 斜坡 Hebbian 学习):
  - `W_first` 句首 / `W_tmpl` 模板链(字符onehot+段计数) /
    `W_slot_first[r]` slot首字符 / `W_slot[r]` slot续写(词表征+计数multiset) /
    `W_done_slot[r]` 段完成判定(纯输入侧, 零漂移) / `W_role` 模板态判定

## 验证结果
| 实验 | 语料 | 训练内 | 留出组合 |
|---|---|---|---|
| P50 (96 拉丁方) | 10adj×10animal 训练96 | 96/96 | 4/4 |
| P51 (60 条验收) | 扩充 15×15=225, 训练60 | 60/60 | 5/6 |

## 复现配方
1. 训练基础模型 (P37 配方): state_kwt_k=48, dir_quota_k=16, p=0/0.5 两相, 句WTA
2. `from slot_route_v38 import SlotRouteGenerator, TRAIN96, HELD4`
   - `SlotRouteGenerator.from_base(_p37.pt, TRAIN96, HELD4).train(iters=60).save(out.pt)`
   - 或 `from_pretrained(已收敛.pt)` 直接生成
3. `gen.generate("red dog")` -> `'the dog is red!'`

注: 历史 .pt 模型 pickle 引用 `__main__.R36NoPos`, 模块 `_load_sim` 自动注入。
