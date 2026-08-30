# 图表映射与口径

| 报告部分 | 分析问题 | 图形 | 字段与样本 | 支持的结论 | 调色与非颜色区分 | 输出 |
|---|---|---|---|---|---|---|
| 特征—K选择 | 两个主要指标是否同时支持候选 | 散点图 | 60 个确认性候选；Silhouette、稳定性 ARI | 冻结四维 K=2 不是单指标最高，但按预设基准规则保留 | K 用颜色、最终候选用星形 | `outputs/clustering/figures/confirmatory_silhouette_stability.png` |
| K比较 | 四维特征下 K=2–6 如何变化 | 分组柱图 | 5 个 K；两个 0–1 指标 | K=2 同时高于 K=3–6 | 蓝/金并列、直接 K 标签 | `outputs/clustering/figures/selected_feature_k_comparison.png` |
| 层次结构 | Ward 合并距离是否支持大分裂 | 树状图 | V2 主样本 N=197 | 展示而不替代数值选择 | 单蓝线、无叶标签 | `outputs/clustering/figures/ward_dendrogram.png` |
| 归属稳定性 | SKU 是否反复回到原簇 | 分组箱线图 | 197 个 SKU，500 次子样本 | 多数稳定，但存在边界 SKU | 蓝/金箱体、0.75 虚线 | `outputs/clustering/figures/sku_assignment_stability.png` |
| 留出模型 | 单模型在最终窗口的量级误差 | 排序横条 | 147 个 SKU；销量加权 WAPE | MA4_proxy 最低，ADIDA2 次之 | 模型直接标签、数值标签 | `outputs/forecast/figures/holdout_model_wape.png` |
| 管理路径 | 冻结路由如何分配 | 横条 | 留出共同 147 个 SKU | 仅 8 个进入预测管理 | 三类直接标签 | `outputs/forecast/figures/holdout_management_paths.png` |
| 起点稳定性 | 模型排名是否跨起点稳定 | 分组柱图 | 6 个离散起点 × 4 模型 | 排名频繁变化 | 固定模型颜色与图例 | `outputs/forecast/figures/rolling_origin_core_wape.png` |
| ADIDA敏感性 | 聚合尺度是否改善量级误差 | 柱图 | 790 个共同 SKU-起点 | 2周的销量加权 WAPE 最低 | 四尺度直接标签 | `outputs/forecast/figures/adida_aggregation_sensitivity.png` |
| 机制贡献 | 哪条路由造成净变化 | 有正负零线横条 | 留出 147 个 SKU | Naive 增益被 Zero 规则部分抵消 | 正负位置与数值标签，不依赖红绿 | `outputs/forecast/figures/holdout_path_contribution.png` |
