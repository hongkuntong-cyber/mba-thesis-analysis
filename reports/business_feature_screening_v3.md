# 业务导向聚类特征筛选 V3.0

> 证据标签：回顾性方法开发，不是独立确认性结果。协议提交发生在本轮聚类运行之前。

## 冻结设计

- 正式主池：ADI, CV2, acf1, nonzero_mean, peak_ratio, promo_response_index, trailing_zero_share, trend_coef
- 理论锚点：ADI、CV2
- 正式组合：64 个
- 主样本：197 个 SKU
- Ward K=2；80% 无放回子样本；重复 500 次
- 选择规则：结构门后使用 Silhouette 与稳定性 ARI 的 Pareto 短名单，不建立综合分

## Pareto 短名单

| feature_set | n_features | silhouette | stability_ari_median | minimum_cluster_jaccard_median | cluster_sizes | raw_v2_ari | main_strict_ari | ward_kmeans_ari |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADI+CV2+peak_ratio | 3 | 0.4005 | 0.8039 | 0.8584 | 1:138;2:59 | 1.0000 | 0.5118 | 0.5685 |
| ADI+CV2+acf1+nonzero_mean | 4 | 0.3692 | 0.8757 | 0.9219 | 1:120;2:77 | 0.9199 | 0.6018 | 0.9004 |

当前状态：**存在 2 个 Pareto 候选，需要稳健性与业务画像审阅**。如果短名单不止一个，报告多个候选，不利用预测结果强行反选聚类特征。

## 高相关诊断

| pair | pearson | spearman |
| --- | --- | --- |
| CV2 / peak_ratio | 0.8224 | 0.8795 |
| peak_ratio / CV2 | 0.8224 | 0.8795 |

高相关只作为重复赋权风险提示；除精确恒等关系外，不根据当前结果临时删除候选。

## 解释边界

`promo_response_index` 是用户提供 Amazon 促销月份分布形成的业务日历代理，不是 SKU 级折扣或广告投入，也不能用于促销因果推断。年度季节性因 32 个 SKU 只有 30 周记录，另在不少于 104 周的长历史样本中检验，不参与本表主特征重选。
