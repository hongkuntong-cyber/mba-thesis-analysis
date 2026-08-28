# 特征字典（协议版本 1.0）

所有特征均按单个 SKU 的按周序列计算。`N_observed` 为实际存在的周记录数，`N_positive` 为销量严格大于 0 的周数。

| 特征 | 冻结定义 | 使用范围 | 状态 |
|---|---|---|---|
| mean_sales | 全部观察周销量算术平均值 | 确认性候选 | 与 ADI、nonzero_mean 存在精确关系，不与 nonzero_mean 同时使用 |
| median_sales | 全部观察周销量中位数 | 确认性候选 | 如低变异，按事实报告，不事后改用正销量中位数 |
| std_sales | 全部观察周销量样本标准差，ddof=1 | 确认性候选 | 不与 CV²机械等同 |
| nonzero_mean | 正销量周的算术平均值 | 确认性候选 | 与 mean_sales 二选一 |
| CV² | 正销量周样本标准差除以正销量均值后平方，ddof=1 | 理论锚点 | 主样本至少5个正销量周，因此可计算 |
| ADI | `N_observed / N_positive` | 理论锚点 | 所有正式组合必须包含 |
| zero_ratio | `N_zero / N_observed` | 画像与核对 | 因 `ADI=1/(1-zero_ratio)` 不进入正式组合 |
| acf1 | 全周序列与其滞后1周序列的 Pearson 相关；任一侧零方差时按 0 记录并加标记 | 确认性候选 | 不删除零销量周 |
| trend_coef | 未恢复原冻结公式 | 探索性 | 不进入确认性特征—K选择 |
| seasonality_idx | 未恢复原冻结公式 | 探索性 | 不进入确认性特征—K选择 |
| peak_ratio | 未恢复原冻结公式 | 探索性 | 不进入确认性特征—K选择 |
| promo_weight_ratio | 既有月份权重属于探索设定，缺乏独立验证 | 探索性 | 不进入确认性特征—K选择 |

精确核对关系：

`mean_sales = nonzero_mean / ADI`

`ADI = 1 / (1-zero_ratio)`

允许的正式组合：ADI、CV²固定，附加特征从 mean_sales、median_sales、std_sales、nonzero_mean、acf1 中确定性枚举，但 mean_sales 与 nonzero_mean 不共存。
