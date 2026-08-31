# 理论先行特征门：回顾性方法开发结果

> 证据标签：**回顾性方法开发，不是新的确认性检验**。同一工作簿的旧结果已经被查看；本文件不能被表述为事前注册结果。

## 冻结输入

- 协议版本：2.1
- 证据登记表 SHA256：`28e4f18e9d1168b2862763f313ab73d96db1d555f9d9c5a8ac532cf834369c39`
- 理论锚点：ADI, CV2
- 补充候选：approx_entropy, trailing_zero_share
- 合规候选组合数：4
- 主聚类 K：2；K=3–6 只作敏感性分析
- 稳定性重复次数：500

## K=2 主特征选择

选择结果：`ADI + CV2`

选择理由：No admitted supplementary feature set dominated the frozen ADI+CV2 theory benchmark on both primary metrics; the benchmark was retained.

| feature_set | n_features | silhouette | stability_ari_median | min_cluster_size | cluster_sizes | k2_pareto |
| --- | --- | --- | --- | --- | --- | --- |
| ADI+CV2 | 2 | 0.3885 | 0.6505 | 74 | 1:123;2:74 | yes |
| ADI+CV2+approx_entropy | 3 | 0.3331 | 0.4039 | 44 | 1:153;2:44 | no |
| ADI+CV2+trailing_zero_share | 3 | 0.3458 | 0.5525 | 85 | 1:112;2:85 | no |
| ADI+CV2+approx_entropy+trailing_zero_share | 4 | 0.2887 | 0.5718 | 83 | 1:83;2:114 | no |

## 稳健性

- Ward 与 K-means ARI：0.6822
- Raw 与 V2 共同 SKU 的 ARI：0.4027（N=197）
- 正需求周 ≥5 与 ≥10 的共同 SKU ARI：0.7410（N=148）
- SKU 归属稳定性中位数：0.9976

## 解释边界

本轮只验证“文献准入门能否被代码严格执行”以及冻结候选在既有数据上的回顾性表现。预测算法池和簇—模型路由不能反向改变候选特征或主特征集。真正的确认性判断要等待新增、未查看的完整周数据。
