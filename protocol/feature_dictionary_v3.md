# 特征字典 V3：业务导向聚类候选池

冻结日期：2026-09-01。所有变量按 SKU 的连续完整周序列计算；需求值必须有限、非负。

## 正式主池

### ADI

`ADI = N_observed / N_positive`。使用全部观察周；`N_positive` 为严格大于 0 的周数。

### CV2

`CV2 = (sd_positive / mean_positive)^2`。仅使用正需求周，样本标准差 `ddof=1`。

### nonzero_mean

`nonzero_mean = mean(y_t | y_t > 0)`。表示一次需求事件的平均规模。因 `mean_sales=nonzero_mean/ADI`，`mean_sales` 不进入 Ward。

### acf1

仅使用相邻且日期相差 7 天的完整周对，计算全周需求 `y_t` 与 `y_(t-1)` 的 Pearson 相关。任一侧零方差或有效周对少于 2 时记为 0，并保留 `acf1_zero_variance` 标记。

### trend_coef

将当前 SKU 的观察位置线性缩放至 `[0,1]`，以全部周需求对标准化时间执行 OLS：

`y_t = alpha + beta*x_t + error_t`

`trend_coef = beta / nonzero_mean`

正值表示观察窗口内需求上升，负值表示下降；除以正需求均值以减少销量尺度影响。不解释为因果趋势。

### peak_ratio

`peak_ratio = max(y_t | y_t>0) / nonzero_mean`

表示最大单次需求相对典型需求事件的倍数。它与 CV2 可能相关但不存在精确恒等，相关性与敏感性完整报告。

### trailing_zero_share

`trailing_zero_share = L_end / N_observed`，其中 `L_end` 为序列末端连续零需求周数。表示近期休眠而非总体间歇程度。

### promo_response_index

月度 W=0/1/2/3 权重及跨月周日历平均规则见 V3.0 修订。先计算：

`promo_weight_ratio = sum(W_t*y_t)/sum(y_t)`

再计算：

`promo_response_index = promo_weight_ratio / mean(W_t | observed weeks)`

正式聚类使用归一化指标；直接权重只用于画像。该变量是业务日历代理，不能用于促销因果推断。

## 独立敏感性特征

### seasonality_idx

仅在 `N_observed>=104` 的长历史样本计算。先用 OLS 去除线性趋势，再计算残差序列与其 52 周滞后序列的 Pearson 相关，负相关截断为 0，范围 `[0,1]`。它是年度季节依赖诊断，不进入完整样本的主组合。

### approximate_entropy

沿用 V2 冻结公式：`m=2`，`r=0.5*全序列样本标准差`，包含自身匹配。仅做技术敏感性。

## 画像或排除变量

- `promo_weight_ratio`：促销原始暴露画像；
- `zero_ratio`：因与 ADI 精确对应，仅作核对；
- `mean_sales`：因可由 ADI 与 nonzero_mean 推导，仅作画像；
- `std_sales`：与规模和 CV2 重叠，仅作画像；
- `median_sales`：间歇序列中常为 0，仅作画像。
