# V4.4 固定120个SKU的公平比较

打开 `fair_ma4_fixed120_v4_4.ipynb` 阅读已经执行的结果。

本轮是补充探索性分析。原始数据截至2026-08-02，主样本名单固定120个SKU，
另保留77个低频SKU作分层对照。新SKU尾部滚动与旧统一日历结果分开报告。

## 复现

在仓库根目录安装 `requirements-notebook.txt`，然后执行：

```bash
python -m src.pxq_fair_v4_4
python -m src.validate_pxq_fair_v4_4
python -m unittest tests.test_pxq_fair_v4_4 -v
```

第一条命令发现完整准备结果后只复算汇总，不重复清洗或旧模型。
需要从原始数据完整重建时，请在新工作副本中移走 `outputs/pxq_fair_v4_4/`，再执行第一条命令。
不要删除历史V4.0—V4.3文件。

笔记本只读取可复核结果并检验文件指纹，不重新运行旧实验。可用 Jupyter 打开并运行所有单元。

## 来源与结果

- 冻结协议：`protocol/amendment_v4.4_fixed120_fair_ma4.md`
- 固定配置：`config/pxq_fair_v4_4.yaml`
- 计算脚本：`src/pxq_fair_v4_4.py`
- 独立校验：`src/validate_pxq_fair_v4_4.py`
- 完整V4.3来源包：`inputs/prior_v4_3/pxq_v4_3_results.zip`（内含逐文件SHA256）
- 样本身份：`outputs/pxq_fair_v4_4/cohort_membership.csv`
- 计划和不可评价原因：`outputs/pxq_fair_v4_4/universe_coverage.csv`
- 逐SKU覆盖：`outputs/pxq_fair_v4_4/sku_coverage.csv`
- 同样本数量比较：`outputs/pxq_fair_v4_4/quantity_comparisons.csv`
- 概率与MA4排序诊断：`outputs/pxq_fair_v4_4/probability_summary.csv`
- 分层适用性：`outputs/pxq_fair_v4_4/quantity_by_layer.csv`

GitHub基线中的旧 `outputs/pxq_validation_v4/rolling_origin_predictions.csv` 已损坏，
本轮不读取它，改用结果包中通过全部58项指纹校验的完整旧预测副本。新输出单独保存。

数量主候选是未平滑的周期发生频率×条件周期均量，不是早期周发生率p×q。
同历史范围内，它等于历史周期均值；这一恒等关系已逐条验证，不应包装成数量算法创新。
