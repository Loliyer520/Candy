# experiments/ — 历史实验版本归档

本目录存放开发过程中全部实验性脚本 (experiment2 ~ experiment36, diag_*, verify_*),
每个实验对应 README.md 中 v11~v14.4 的归档记录。

**注意**:
- 这些是历史研究代码, 仅供考古/复现, 不作为成品使用。
- 数据文件位于 `../data/` (原为项目根目录的 english_pairs_1000.txt 等),
  脚本内相对路径 `english_pairs_1000.txt` 在此目录下不再有效。
- 依赖核心模块位于 `../core/` (原为根目录的 episodic_memory.py 等),
  如需运行, 请将 `..` 加入 sys.path, 例如:
  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from core.episodic_memory import ...
  ```

## 关键结论速查 (详见根 README.md)

| 实验 | 结论 |
|------|------|
| 11-13 | W_seq 逐字符预测结构性不可解; 异联想链失败 |
| 17 | 资格迹证伪; DG 稀疏分离 (k=64) 小库根除串扰 |
| 18 | P1 事件联合记忆 v3 — 同时解无限上下文 + 时间序列 |
| 20 | 自回归事件记忆 v2 上下文消歧 (库内 +41.8pp) |
| 29 | SDM 非叠加存储突破 (库50 93.2%) |
| 30-34 | 动态容量四败 — 有偏投影破坏均匀聚桶 |
| 35-36 | 叠表/分区/遗忘全败 — 固定池缩放定论 |
