# 特征字典 V2：理论准入后的正式候选池

冻结日期：2026-08-31  
证据登记表：`protocol/feature_evidence_registry.csv`

## 1. 候选池结论

正式候选池不是 12 项，也不预设数量。本轮事前文献与逻辑审查后，共有 4 项通过准入：

1. 理论锚点：`ADI`、`CV2`；
2. 补充候选：`approx_entropy`、`trailing_zero_share`。

正式组合必须包含两个锚点，因此只枚举以下 4 个组合：

- `ADI + CV2`；
- `ADI + CV2 + approx_entropy`；
- `ADI + CV2 + trailing_zero_share`；
- `ADI + CV2 + approx_entropy + trailing_zero_share`。

这 4 个组合都是候选，尚未指定哪一个是主聚类特征集。主特征集只能在候选池冻结后按预定聚类指标选择，预测结果不得参与。

## 2. 统一数据口径

- 输入为某个分析时点可见的完整 SKU 周序列；
- 完整期间描述使用当期 Raw 或 V2 序列，Rolling-Origin 只使用预测起点前序列；
- 需求为非负数；正需求定义为 `y_t > 0`，零需求定义为 `y_t = 0`；
- 缺失周必须在数据审计阶段处理，特征函数不对缺失值做静默填补；
- 所有候选至少要求 5 个正需求周进入主聚类样本；`CV2` 至少需要 2 个正需求周才可计算；
- Yeo–Johnson 与标准化只在当前样本或当前训练窗口内拟合。

## 3. 正式特征公式

### 3.1 ADI：平均需求间隔

设完整观察周数为 `N`，正需求周数为 `N_positive`：

`ADI = N / N_positive`

- 使用全部完整观察周；
- `N_positive = 0` 时记为正无穷，不进入聚类；
- 构念：间歇程度；
- 依据：Syntetos、Boylan 与 Croston（2005），Kostenko 与 Hyndman（2006）。

### 3.2 CV2：正需求量变异系数平方

设正需求量均值为 `mean_positive`，样本标准差为 `sd_positive`：

`CV2 = (sd_positive / mean_positive)^2`

- 仅使用正需求周；
- 标准差自由度固定为 `ddof=1`；
- 正需求周少于 2 或正需求均值不大于 0 时不可计算；
- 构念：正需求事件的规模波动；
- 依据同上。

### 3.3 approx_entropy：近似熵

对完整周序列 `y_1,...,y_N` 固定：嵌入维度 `m=2`，容差 `r=0.5*s_y`，其中 `s_y` 为全序列样本标准差。对长度为 `m` 的子序列，使用最大绝对距离并包含自身匹配：

`C_i^m(r) = count_j(max_k |y_(i+k)-y_(j+k)| <= r) / (N-m+1)`

`Phi_m(r) = mean_i(log(C_i^m(r)))`

`approx_entropy = Phi_m(r) - Phi_(m+1)(r)`

- 使用全部周，包括零需求周；
- 最低完整观察长度固定为 5 周；
- 常数序列返回 0，表示完全规则；
- 不允许负值、缺失值或无穷值；
- 构念：序列规律性与不可预测性，值越大通常表示越不规则；
- 依据：Li 等（2023）及其作者复现代码；证据等级 B。

### 3.4 trailing_zero_share：末端连续零值占比

设序列末端连续零需求周长度为 `L_end`：

`trailing_zero_share = L_end / N`

- 使用全部周；
- 如果最后一周为正需求，则 `L_end=0`；
- 不删除序列中间的正需求或零需求；
- 最低完整观察长度固定为 1 周，但正式样本仍服从正需求周门槛；
- 构念：近期停滞或潜在淘汰状态；
- 依据：Li 等（2023）的 `Percent.zero.end`；证据等级 B。

## 4. 冲突与排除规则

- `zero_ratio` 与 `ADI` 精确等价，不进入正式池；
- `percent_beyond_sigma`、`linear_chunk_variance`、`mean_abs_change` 均由原研究归入波动构念；`CV2` 已预先冻结为该构念代表，因此不并列进入 Ward；
- `ratio_last_chunk` 与 `trailing_zero_share` 同属近期停滞构念，后者公式更直接、便于企业解释，前者仅可作未来敏感性分析；
- `mean_sales`、`median_sales`、`nonzero_mean`、`acf1`、趋势、季节性和峰值指标未在本次冻结检索中同时满足原始文献、精确公式和构念独立性，不进入正式池；
- `promo_weight_ratio` 缺少真实 SKU—周促销字段，不进入正式池。

任何排除特征若进入确认性组合，程序必须失败，而不是自动忽略。

## 5. 文献

- Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). *On the categorization of demand patterns*. Journal of the Operational Research Society, 56(5), 495–503. https://doi.org/10.1057/palgrave.jors.2601841
- Kostenko, A. V., & Hyndman, R. J. (2006). *A note on the categorization of demand patterns*. Journal of the Operational Research Society, 57(10), 1256–1257. https://doi.org/10.1057/palgrave.jors.2602211
- Li, L., Kang, Y., Petropoulos, F., & Li, F. (2023). *Feature-based intermittent demand forecast combinations: accuracy and inventory implications*. International Journal of Production Research, 61(22), 7557–7572. https://doi.org/10.1080/00207543.2022.2153941
