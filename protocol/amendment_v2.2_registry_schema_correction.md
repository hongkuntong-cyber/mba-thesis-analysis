# 研究协议修订 2.2：证据登记表布尔字段校正

修订日期：2026-08-31  
状态：代码准入门实现期间发现，尚未运行任何新版特征—K 网格

`protocol/feature_evidence_registry.csv` 中 `mean_sales`、`median_sales`、`nonzero_mean` 三个已被标记为 `rejected_formula_unfrozen` 的特征，其 `formula_frozen` 单元格误留为空白。

本修订只把三个空白值显式改为 `false`，以满足失败关闭校验器的布尔字段约束。以下内容均未改变：

- 特征状态；
- 证据等级；
- 准入特征数量；
- 正式候选组合；
- 公式；
- K=2 选择规则；
- 清洗、样本、稳定性或预测协议。

校正发生在任何新版 Silhouette、ARI、K 值或预测结果计算之前，不构成结果驱动的方法变更。旧登记表继续保留在 Git 历史和 `feature_evidence_registry_v2.0_initial.csv` 中。
