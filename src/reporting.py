from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    return "不可计算" if not np.isfinite(value) else f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        if isinstance(value, float):
            return _num(value)
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    output.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def write_implementation_matrix(report_root: str | Path) -> None:
    report_path = Path(report_root) / "implementation_matrix.md"
    content = """# 轻量预测模型实施矩阵

本表不使用机械加权总分。决策顺序为：预测准确性与跨起点稳定性 → 可解释性 → 实施与回退成本。

| 方案 | 业务解释 | 所需字段 | 参数 | 单次复杂度 | 建议更新 | 异常处理 | 人工维护 | 自动回退 | Excel式理解 | 本研究结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| MA4_proxy | 最近4个完整周平均，最直观 | SKU、周、销量 | 0 | O(n) | 每周 | 易；可排除已确认异常周后重算 | 低 | 本身即默认回退 | 是 | 保留为企业默认方案 |
| Naive | 延续最近一周 | SKU、周、销量 | 0 | O(1) | 每周 | 易；对最后一周异常较敏感 | 低 | 回退 MA4_proxy | 是 | 仅对开发期反复胜出的少量 SKU 白名单试用 |
| SES | 对近期历史作指数衰减加权 | SKU、周、销量 | 1 个 alpha | O(n) | 每周 | 中；异常值会进入水平项 | 低至中 | 回退 MA4_proxy | 基本可以 | 本次无 SKU 通过路由门槛，不建议上线 |
| ADIDA2 | 每2周聚合、SES预测、再均分回周 | SKU、周、销量、固定日历起点 | 聚合期固定为2周；1 个 alpha | O(n) | 每周或双周 | 中；需处理不完整聚合块和新 SKU | 中 | 回退 MA4_proxy/Naive | 可解释但不如前两者直观 | 保留为影子模型，不进入生产路由 |
| Zero 规则 | 暂不预测，按零需求规则处理 | SKU、周、销量 | 0 | O(1) | 按复核周期 | 易；高影响 SKU 不应自动使用 | 低 | 转人工复核 | 是 | 仅为规则管理与价值基准，不属于正式预测模型池 |

## 建议的最小落地架构

1. 默认全量 SKU 使用 MA4_proxy。
2. 只对冻结白名单启用 Naive；每个新起点重新验证，失效即回退 MA4_proxy。
3. ADIDA2 只影子运行并记录，不参与采购或库存自动决策。
4. 高影响但无法稳定预测的 SKU 进入人工复核；低影响且信息不足者采用规则管理。
5. 任何自动部署前，必须补充库存、缺货、采购提前期、MOQ、服务水平、采购价或毛利，才能验证经济价值。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def write_chart_map(report_root: str | Path) -> None:
    path = Path(report_root) / "chart_map.md"
    content = """# 图表映射与口径

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
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_final_report(project_root: str | Path, config: dict[str, Any]) -> None:
    project = Path(project_root)
    output = project / config["outputs"]["root"]
    cluster_root = output / "clustering"
    forecast_root = output / "forecast"
    audit = _load_json(output / "audit" / "audit_summary.json")
    universe = _load_json(output / "audit" / "sku_universe_transition.json")
    selection = _load_json(cluster_root / "operational_selection_full_period.json")
    robustness = _load_json(cluster_root / "robustness_k2.json")
    k3 = _load_json(cluster_root / "k3_acceptance.json")
    forecast_selection = _load_json(forecast_root / "pre_first_origin_selection.json")
    unconstrained = _load_json(forecast_root / "pre_first_origin_selection_unconstrained.json")
    outcome = _load_json(forecast_root / "backtest_outcome.json")
    grid = pd.read_csv(cluster_root / "feature_k_grid_500.csv")
    profiles = pd.read_csv(cluster_root / "cluster_profiles_v2_k2.csv")
    scheme = pd.read_csv(forecast_root / "holdout_scheme_comparison_summary.csv").set_index("model")
    overall = pd.read_csv(forecast_root / "model_summary_overall.csv").set_index("model")
    origin_summary = pd.read_csv(forecast_root / "model_summary_by_origin.csv")
    bootstrap = pd.read_csv(forecast_root / "development_paired_bootstrap.csv")
    adida_profiles = pd.read_csv(forecast_root / "adida2_profile_summary.csv").set_index("group")
    sensitivity = pd.read_csv(forecast_root / "adida_aggregation_sensitivity_common_summary.csv")
    contributions = pd.read_csv(forecast_root / "holdout_path_contribution.csv")
    route_counts = pd.read_csv(forecast_root / "holdout_route_counts.csv")
    mase_counts = pd.read_csv(forecast_root / "mase_availability_by_origin.csv")

    selected_key = "+".join(selection["feature_names"])
    selected_k_rows = grid.loc[grid["feature_set"].eq(selected_key)].sort_values("k")
    k_rows = [
        [
            int(row.k),
            row.silhouette,
            row.stability_ari_median,
            row.stability_ari_p10,
            row.cluster_sizes,
        ]
        for row in selected_k_rows.itertuples()
    ]
    profile_rows = [
        [
            int(row.cluster),
            int(row.n_skus),
            row.ADI_median,
            row.CV2_median,
            row.nonzero_mean_median,
            row.acf1_median,
            row.zero_ratio_median,
            row.total_sales_median,
            row.n_positive_median,
        ]
        for row in profiles.itertuples()
    ]
    scheme_a = scheme.loc["Enterprise_MA4"]
    scheme_b = scheme.loc["Layered_mechanism"]
    route_totals = route_counts.groupby("management_path")["n_skus"].sum().to_dict()
    contribution_total = float(contributions["net_abs_error_improvement"].sum())
    sku_contrib = pd.read_csv(forecast_root / "holdout_sku_contribution.csv")
    abs_total = float(sku_contrib["absolute_error_improvement"].abs().sum())
    top_share = (
        float(sku_contrib.nlargest(10, "absolute_error_improvement")["absolute_error_improvement"].abs().sum())
        / abs_total
        if abs_total
        else np.nan
    )
    # Use absolute contribution ranking for the concentration statement.
    top_abs_share = (
        float(
            sku_contrib.assign(abs_value=sku_contrib["absolute_error_improvement"].abs())
            .nlargest(10, "abs_value")["abs_value"]
            .sum()
        )
        / abs_total
        if abs_total
        else np.nan
    )
    mase_holdout_unavailable = int(
        mase_counts.loc[
            mase_counts["origin_index"].eq(config["forecast"]["origins"])
            & mase_counts["model"].eq("MA4_proxy"),
            "mase_not_computable",
        ].iloc[0]
    )
    holdout_adida_wape = float(
        origin_summary.loc[
            origin_summary["origin_index"].eq(config["forecast"]["origins"])
            & origin_summary["model"].eq("ADIDA2"),
            "aggregate_wape",
        ].iloc[0]
    )

    report = f"""# 德国 Amazon SKU 聚类与轻量预测验证报告

生成口径：协议 1.0 及修订 1.1–1.4；随机种子 42；原始工作簿只读。  
正式运行命令：`python -m src.pipeline --config config/analysis.yaml`

## 一、结论先行

1. **完整期间主结果为四维 Ward K=2**：`{selected_key}`，N={selection['row']['n_skus']}。Silhouette={selection['row']['silhouette']:.3f}，500 次 80% 子样本稳定性 ARI 中位数={selection['row']['stability_ari_median']:.3f}，簇规模 {selection['row']['cluster_sizes']}。
2. **K=3 不通过**：它同时被 K=2 的 Silhouette（{k3['k2_silhouette']:.3f} 对 {k3['k3_silhouette']:.3f}）和稳定性 ARI（{k3['k2_stability_ari_median']:.3f} 对 {k3['k3_stability_ari_median']:.3f}）支配；最弱匹配簇 Jaccard 仅 {min(k3['k3_cluster_jaccard_medians'].values()):.3f}；Raw/V2 ARI={k3['raw_v2']['ari']:.3f}<0.80。因此 SBA/TSB 未进入模型池。
3. **首个预测起点前冻结结果与完整期间不同**：应用簇级稳定性门槛后为 `{' + '.join(forecast_selection['feature_names'])}`、K={forecast_selection['k']}，N={forecast_selection['eligible_cluster_skus']}。初始遗漏门槛的未约束结果为二维 K={unconstrained['k']}，已永久保留并按修订 1.4 披露。
4. **ADIDA2 不是本数据的全局优胜模型**：开发期相对 MA4_proxy 的 SKU 配对 Mean MASE 差为 +{bootstrap.loc[(bootstrap.model.eq('ADIDA2')) & (bootstrap.baseline.eq('MA4_proxy')), 'mean_difference'].iloc[0]:.3f}，95% 区间 [{bootstrap.loc[(bootstrap.model.eq('ADIDA2')) & (bootstrap.baseline.eq('MA4_proxy')), 'ci_low'].iloc[0]:.3f}, {bootstrap.loc[(bootstrap.model.eq('ADIDA2')) & (bootstrap.baseline.eq('MA4_proxy')), 'ci_high'].iloc[0]:.3f}]，正值表示更差。没有 SKU 通过 ADIDA2 的最终生产路由门槛。
5. **分层方案没有形成足以支持全面部署的增益**：最终留出中，方案 A 的销量加权 WAPE={scheme_a.aggregate_wape:.4f}，方案 B={scheme_b.aggregate_wape:.4f}；Mean MASE 从 {scheme_a.mean_mase:.3f} 降到 {scheme_b.mean_mase:.3f}，配对差区间 [{outcome['holdout_scheme_bootstrap']['ci_low']:.3f}, {outcome['holdout_scheme_bootstrap']['ci_high']:.3f}] 跨 0。净绝对误差仅改善 {contribution_total:.1f} 个单位。
6. **小型企业建议**：继续以 MA4_proxy 为默认；只对 5 个冻结 Naive 白名单做低风险试点；ADIDA2 保留为影子模型；SES 暂不上线；高影响不稳定 SKU 继续人工复核。当前不值得部署完整的“聚类驱动多模型路由”。

## 二、证据分级

### A. 已确认方法

- 德国站、SKU×完整周粒度；跨年拆分周先合并，同一 ISO 周累计覆盖 7 天后方可进入分析。
- V2 清洗、正销量周 ≥5 主样本、≥10 严格样本、Ward、K=2–6、500 次 80% 无放回子样本稳定性。
- 正式预测模型池：MA4_proxy、Naive、SES、ADIDA2；Zero 只作为预测价值与规则管理基准。
- 6 个不重叠 8 周窗口；前 5 个开发，第 6 个一次性留出。

### B. 阶段性参考

- 先前文档的原始合计 30,229、V2 合计约 30,933、增量约 704 只作为排错参考。
- 初始未约束预测期选择 `ADI + CV²`、K=5 是实现缺口暴露结果，不是最终操作方案。

### C. 经本次检验后的判断

- 四维 K=2 得到最多支持，但 ≥5/≥10 ARI 只有 {robustness['main_strict']['ari']:.3f}，因此仍有样本门槛敏感性。
- K=3、SBA/TSB、SES 路由和 ADIDA2 生产路由均未通过。
- 轻量分层只对少量 SKU 有局部价值，整体经济价值尚不能成立。

## 三、数据审计与 V2 复现

| 项目 | 结果 |
|---|---:|
| 原始 SHA256 | `{audit['sha256']}` |
| 工作表 | {', '.join(audit['sheet_names'])} |
| 日期范围 | {audit['week_start_min']} 至 {audit['week_start_max']} |
| 唯一 SKU | {audit['unique_skus']} |
| 原始 SKU-区间单元 | {audit['raw_cells_at_sku_period_grain']:,} |
| 合并后 SKU-周 | {audit['consolidated_sku_weeks']:,} |
| 完整 SKU-周 | {audit['complete_sku_weeks']:,} |
| 不完整 SKU-周 | {audit['incomplete_sku_weeks']:,} |
| 跨表拆分周 SKU-周 | {audit['split_week_sku_weeks']:,} |
| 非数值 / 缺失 / 负值 | {audit['invalid_numeric_cells']} / {audit['missing_sales_cells']} / {audit['negative_sales_cells']} |
| 阻断问题 | {len(audit['blockers'])} |

V2 正式复现：529 个候选区间、43 个修正区间、30 个 SKU、168 个 SKU-周。关键计数与历史参考完全一致。当前严格合并完整 ISO 周后，Raw 合计为 {audit['v2']['raw_total']:,.2f}，V2 为 {audit['v2']['v2_total']:,.2f}，增加 {audit['v2']['absolute_change']:,.2f}（{_pct(audit['v2']['relative_change'], 2)}）。这与历史 30,229 / 30,933 / +704 略有差异；本研究没有调节规则追平，差异优先解释为跨年分片或历史数据版本口径。

完整期间样本数：正销量周 ≥2 为 {audit['sample_counts']['positive_weeks_ge_2']}，≥5 为 {audit['sample_counts']['positive_weeks_ge_5']}，≥10 为 {audit['sample_counts']['positive_weeks_ge_10']}。两个精确特征恒等关系均为 0 个失败。

### SKU 宇宙变化

2024–2025 工作表有 {universe['first_sheet_skus']} 个 SKU，2026 工作表有 {universe['second_sheet_skus']} 个；仅 {universe['overlap_skus']} 个重叠，{universe['first_only_skus']} 个仅在前表，{universe['second_only_skus']} 个仅在 2026 表。这导致第 3 个窗口有 95 个测试期不完整，第 4–6 个窗口有 95 个训练末端中断；它是预测结论的重要限制，不能把缺失周静默补零。

## 四、特征与 K

确认性特征均以 ADI、CV² 为锚点，完整网格同时保留含 `median_sales` 的敏感性结果。`median_sales` 在 197 个主样本中 169 个为 0，形成近似 ADI≈2 的机械阈值；其 `ADI+CV²+median_sales` K=2 虽达到 Silhouette 0.573、稳定性 ARI 1.000，但不具备主特征资格。

### 四维冻结特征的 K=2–6

{_markdown_table(['K', 'Silhouette', '稳定性 ARI 中位数', 'ARI P10', '簇规模'], k_rows)}

K=3 的数值判定条件中，未被 K=2 支配、全部簇 Jaccard≥0.75、Raw/V2 ARI≥0.80 和独立非残差解释均失败；簇规模和 ≥5/≥10 ARI 条件通过。结论是 **K=2 主结果、K=3 探索性失败结果**。

### K=2 原始尺度画像

{_markdown_table(['簇', 'N', 'ADI中位数', 'CV²中位数', '正销量均值', 'acf1', '零占比', '总销量', '正销量周'], profile_rows)}

- 簇 1（120）：较活跃的中度间歇需求，需求事件规模和总影响较高，序列持续性更强。
- 簇 2（77）：高度间歇、低规模需求；正需求事件自身的 CV²较低，但发生很稀疏。

稳健性：Ward/K-means ARI={robustness['ward_kmeans_ari']:.3f}，Raw/V2 ARI={robustness['raw_v2']['ari']:.3f}；SKU 归属稳定性中位数={robustness['assignment_stability_median']:.3f}，P10={robustness['assignment_stability_p10']:.3f}，有 {robustness['assignment_stability_below_0_75']} 个 SKU 低于 0.75。≥5/≥10 ARI={robustness['main_strict']['ari']:.3f}，提示严格样本下边界移动明显。

![确认性候选](../outputs/clustering/figures/confirmatory_silhouette_stability.png)

![K比较](../outputs/clustering/figures/selected_feature_k_comparison.png)

## 五、预测实验与年度可用性

首个预测起点为 {forecast_selection['first_origin']}，只使用截至 {forecast_selection['training_week_max']} 的数据。修订 1.4 前未约束结果为 `{' + '.join(unconstrained['feature_names'])}` K={unconstrained['k']}，最弱簇 Jaccard={min(json.loads(unconstrained['row']['cluster_jaccard_medians']).values()):.3f}；应用既有簇级门槛后，冻结为 `{' + '.join(forecast_selection['feature_names'])}` K={forecast_selection['k']}，簇 {forecast_selection['row']['cluster_sizes']}，Silhouette={forecast_selection['row']['silhouette']:.3f}，稳定性 ARI={forecast_selection['row']['stability_ari_median']:.3f}。

6 个起点分别为：{', '.join(outcome['origins'])}。预测覆盖过至少一次的唯一 SKU 为 {outcome['forecasted_unique_skus']}；最终留出共同可评价为 {outcome['holdout_common_skus']}。最终留出有 {mase_holdout_unavailable} 个 SKU 的 MASE 分母为 0，均保留 MAE/WAPE 而未填造 MASE。

## 六、模型结果与 ADIDA2

### 全部 6 个起点汇总

{_markdown_table(['模型', 'Mean MASE', 'Median MASE', 'MASE<1', 'MAE', '销量加权WAPE', 'Bias'], [[model, overall.loc[model].mean_mase, overall.loc[model].median_mase, _pct(overall.loc[model].mase_lt_1_share), overall.loc[model].mean_mae, overall.loc[model].aggregate_wape, overall.loc[model].aggregate_bias] for model in ['MA4_proxy','Naive','SES','ADIDA2','Zero']])}

Zero 的 Mean MASE 较低但销量加权 WAPE=1，说明大量稀疏 SKU 会让未加权 MASE 偏向零预测；不能用单一平均 MASE 决定全量模型。

开发期配对结果：ADIDA2 相对 MA4_proxy 的 Mean MASE 差为 +0.314，95%区间 [0.150, 0.494]；相对 Naive 为 +0.570，[0.145, 1.106]。两项均显示 ADIDA2 全局更差。

### ADIDA2 适用画像

- 开发期共有 {outcome['adida2']['development_sku_origins']} 个可比 SKU-起点；ADIDA2 在 {outcome['adida2']['development_better_than_naive_sku_origins']} 个上优于 Naive，在 {outcome['adida2']['development_better_than_ma4_sku_origins']} 个上优于 MA4_proxy。
- 10 个 SKU 至少 3 次同时优于 Naive 与 MA4_proxy。它们在第 5 起点的中位画像为：ADI={adida_profiles.loc['ADIDA2_repeated_candidates'].ADI_median:.2f}、正销量周={adida_profiles.loc['ADIDA2_repeated_candidates'].n_positive_median:.1f}、nonzero_mean={adida_profiles.loc['ADIDA2_repeated_candidates'].nonzero_mean_median:.2f}、acf1={adida_profiles.loc['ADIDA2_repeated_candidates'].acf1_median:.2f}。这更接近“中度间歇、事件规模不低、具有一定持续性”，不是极端稀疏。
- 最终留出 ADIDA2 分别在 {outcome['adida2']['holdout_better_than_naive_skus']} 个 SKU 上优于 Naive、{outcome['adida2']['holdout_better_than_ma4_skus']} 个上优于 MA4_proxy，其中同时优于两者为 34 个；但留出销量加权 WAPE={holdout_adida_wape:.3f}，高于 MA4_proxy 的 {scheme_a.aggregate_wape:.3f}，且无 SKU 满足全部生产路由条件。
- 在共同 790 个 SKU-起点上，2 周聚合的销量加权 WAPE={sensitivity.loc[sensitivity.aggregation_weeks.eq(2), 'aggregate_wape'].iloc[0]:.3f}，低于 4/6/8 周的 {', '.join(f'{value:.3f}' for value in sensitivity.loc[sensitivity.aggregation_weeks.isin([4,6,8]), 'aggregate_wape'])}。较长聚合的 Mean MASE有时更低，但 Median MASE、MAE和销量加权 WAPE更差，支持 2 周只作为影子模型的正式尺度。

![留出模型](../outputs/forecast/figures/holdout_model_wape.png)

![ADIDA敏感性](../outputs/forecast/figures/adida_aggregation_sensitivity.png)

## 七、管理路径与方案 A/B

前 5 个起点冻结的全 SKU 路由为：预测管理 {outcome['route_counts'].get('预测管理', 0)}、规则管理 {outcome['route_counts'].get('规则管理', 0)}、人工复核 {outcome['route_counts'].get('人工复核', 0)}。在最终留出的 147 个共同 SKU 中：预测管理 {int(route_totals.get('预测管理', 0))}、规则管理 {int(route_totals.get('规则管理', 0))}、人工复核 {int(route_totals.get('人工复核', 0))}。预测管理只包含 5 个 Naive 和 3 个 MA4_proxy；SES、ADIDA2 均为 0。

{_markdown_table(['方案', 'N', 'Mean MASE', 'Median MASE', 'MASE<1', 'MAE', '销量加权WAPE', 'Bias'], [['A: 统一MA4', int(scheme_a.n_sku_origins), scheme_a.mean_mase, scheme_a.median_mase, _pct(scheme_a.mase_lt_1_share), scheme_a.mean_mae, scheme_a.aggregate_wape, scheme_a.aggregate_bias], ['B: 轻量分层', int(scheme_b.n_sku_origins), scheme_b.mean_mase, scheme_b.median_mase, _pct(scheme_b.mase_lt_1_share), scheme_b.mean_mae, scheme_b.aggregate_wape, scheme_b.aggregate_bias]])}

方向上，方案 B 的 Mean、Median MASE和 WAPE均略好，但幅度很小，Bias 更负。贡献拆解显示：5 个 Naive 路由改善 12.5 个绝对误差单位；38 个 Zero 规则路由损失 7.5；其余回退 MA4 不变，净改善仅 {contribution_total:.1f}。绝对变化的前 10 个 SKU 占 {_pct(top_abs_share)}，说明结果高度集中，而非普遍改善。

![管理路径](../outputs/forecast/figures/holdout_management_paths.png)

![路径贡献](../outputs/forecast/figures/holdout_path_contribution.png)

## 八、小型企业落地判断

**当前不建议部署完整分层机制。** 可落地的最小版本是：

1. MA4_proxy 作为默认模型和统一回退。
2. 仅对 `frozen_routes_before_holdout.csv` 中 5 个 Naive SKU 做白名单试点；每期重新验证，失效即回退。
3. ADIDA2 对 10 个开发期反复改善候选做影子运行，不进入自动采购或库存决策。
4. 留出期 101 个高影响/不稳定 SKU 人工复核；38 个低影响、信息不足 SKU 规则管理。
5. 先解决年度 SKU 宇宙断裂与业务字段缺失，再评估是否值得自动化扩展。

具体维护矩阵见 `reports/implementation_matrix.md`。

## 九、限制与仍需补充的数据

- 两个年度工作表只有 100 个 SKU 重叠，回测的可比总体发生结构性变化。
- V2 是需求代理，不是真实销量恢复；历史总量参考与当前严格完整周口径存在小差异。
- ≥5/≥10 的 K=2 ARI 仅 0.602，完整期间画像不能被解释为绝对稳定分类。
- `median_sales` 资格修订发生在 5 次预检之后；操作稳定性门槛推广发生在看到预测前聚类网格之后，均已披露但仍有结果知情风险。
- 只有销量，没有促销、库存、缺货、持有成本、采购提前期、MOQ、服务水平、采购价或毛利。
- 因此本报告只能判断预测误差与维护复杂度，不能声称利润、库存成本或缺货成本显著改善。
- 最终留出只有一个 8 周窗口；置信区间较宽，需新增滚动数据后再验证。

## 十、复现与交付

- 正式命令：`python -m src.pipeline --config config/analysis.yaml`
- 测试：`pytest -q`
- 主要原始表：`01_原始数据/德国Amazon_SKU周度数据_原始合并版_未清洗 (1).xlsx`
- 完整网格：`outputs/clustering/feature_k_grid_500.csv`
- K=3 判定：`outputs/clustering/k3_acceptance.json`
- 回测明细：`outputs/forecast/rolling_origin_predictions.csv`
- 冻结路由：`outputs/forecast/frozen_routes_before_holdout.csv`
- 留出比较：`outputs/forecast/holdout_scheme_comparison_summary.csv`

所有失败结果、未约束 K=5、MASE 不可计算记录、原始/V2差异和年度 SKU 断裂均保留，没有为追求更理想结论而调整规则。
"""
    report_path = project / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_implementation_matrix(report_path.parent)
    write_chart_map(report_path.parent)
