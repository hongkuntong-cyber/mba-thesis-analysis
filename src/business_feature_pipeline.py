from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

from .cleaning_v2 import apply_v2_cleaning
from .clustering import (
    algorithm_robustness,
    closest_cluster_profile_effects,
    cluster_profiles,
    compare_solution_labels,
    fit_solution,
    mark_pareto,
)
from .config import load_config
from .data_audit import load_workbook_long
from .feature_gate import enumerate_admitted_feature_sets, load_feature_gate
from .features import compute_features, verify_feature_identities
from .stability import (
    _matched_jaccards,
    canonicalize_labels,
    fit_ward_matrix,
    sku_assignment_stability,
    transform_features,
)


PROFILE_COLUMNS = [
    "ADI",
    "CV2",
    "nonzero_mean",
    "acf1",
    "trend_coef",
    "peak_ratio",
    "trailing_zero_share",
    "promo_response_index",
    "promo_weight_ratio",
    "mean_sales",
    "zero_ratio",
    "total_sales",
    "n_positive",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _feature_key(feature_names: Iterable[str]) -> str:
    return "+".join(feature_names)


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    table = frame[list(columns)].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    lines = [
        "| " + " | ".join(str(value) for value in table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def evaluate_shared_transform_k2(
    features: pd.DataFrame,
    feature_sets: Iterable[Iterable[str]],
    *,
    repetitions: int,
    sample_fraction: float,
    seed: int,
    k: int = 2,
) -> pd.DataFrame:
    """Evaluate every frozen subset using paired subsamples and shared transforms.

    PowerTransformer estimates each column independently. Fitting all admitted
    columns once per subsample and then slicing the requested subset is therefore
    numerically equivalent to fitting each subset separately, while making the
    64-combination protocol tractable.
    """
    frozen_sets = [tuple(values) for values in feature_sets]
    all_names = tuple(dict.fromkeys(name for values in frozen_sets for name in values))
    full_all, _ = transform_features(features, all_names)
    indices = {
        key: [all_names.index(name) for name in names]
        for names in frozen_sets
        for key in [_feature_key(names)]
    }
    full_labels: dict[str, np.ndarray] = {}
    rows: dict[str, dict[str, Any]] = {}
    aris: dict[str, list[float]] = {}
    jaccards: dict[str, dict[int, list[float]]] = {}

    for names in frozen_sets:
        key = _feature_key(names)
        matrix = full_all[:, indices[key]]
        labels = fit_ward_matrix(matrix, k)
        full_labels[key] = labels
        canonical = canonicalize_labels(features, labels)
        counts = pd.Series(canonical).value_counts().sort_index()
        rows[key] = {
            "feature_set": key,
            "n_features": len(names),
            "k": k,
            "n_skus": len(features),
            "valid": True,
            "error": "",
            "silhouette": float(silhouette_score(matrix, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "cluster_sizes": ";".join(
                f"{int(cluster)}:{int(size)}" for cluster, size in counts.items()
            ),
        }
        aris[key] = []
        jaccards[key] = {cluster: [] for cluster in range(k)}

    n_rows = len(features)
    sample_size = max(k + 1, int(np.floor(n_rows * sample_fraction)))
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sample_indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        sample_all, _ = transform_features(features.iloc[sample_indices], all_names)
        for names in frozen_sets:
            key = _feature_key(names)
            sample_labels = fit_ward_matrix(sample_all[:, indices[key]], k)
            restricted_full = full_labels[key][sample_indices]
            aris[key].append(adjusted_rand_score(restricted_full, sample_labels))
            matched = _matched_jaccards(restricted_full, sample_labels, k)
            for cluster, score in matched.items():
                jaccards[key][cluster].append(score)

    for key in rows:
        values = np.asarray(aris[key], dtype=float)
        medians = {
            str(cluster + 1): float(np.median(scores))
            for cluster, scores in jaccards[key].items()
        }
        rows[key].update(
            {
                "stability_ari_mean": float(np.mean(values)),
                "stability_ari_median": float(np.median(values)),
                "stability_ari_p10": float(np.quantile(values, 0.10)),
                "stability_ari_p90": float(np.quantile(values, 0.90)),
                "cluster_jaccard_medians": json.dumps(medians, ensure_ascii=False),
                "minimum_cluster_jaccard_median": float(min(medians.values())),
            }
        )
    return pd.DataFrame(rows.values())


def mark_structural_pareto(
    grid: pd.DataFrame,
    *,
    minimum_cluster_jaccard_median: float,
    minimum_cluster_size: int,
    minimum_cluster_share: float,
) -> pd.DataFrame:
    output = grid.copy()
    output["structural_eligible"] = (
        output["valid"]
        & output["minimum_cluster_jaccard_median"].ge(minimum_cluster_jaccard_median)
        & output["min_cluster_size"].ge(minimum_cluster_size)
        & output["min_cluster_size"].div(output["n_skus"]).ge(minimum_cluster_share)
    )
    marked = mark_pareto(output, eligibility=output["structural_eligible"])
    output["pareto"] = marked["pareto"]
    return output


def _load_inputs(
    config: dict[str, Any], project_root: Path
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame]:
    loaded = load_workbook_long(
        project_root / config["input"]["workbook"],
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook SHA256 differs from the frozen V3.0 configuration")
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")
    cleaned = apply_v2_cleaning(loaded.weekly_complete, **config["cleaning_v2"])
    return (
        loaded,
        cleaned,
        compute_features(cleaned.weekly, "sales_v2"),
        compute_features(cleaned.weekly, "sales_raw"),
    )


def _candidate_diagnostics(
    review_keys: list[str],
    *,
    v2_main: pd.DataFrame,
    raw_features: pd.DataFrame,
    all_v2_features: pd.DataFrame,
    minimum_positive: int,
    strict_minimum: int,
    repetitions: int,
    sample_fraction: float,
    n_init: int,
    seed: int,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    profile_rows: list[pd.DataFrame] = []
    effect_rows: list[pd.DataFrame] = []
    assignment_rows: list[pd.DataFrame] = []
    for key in review_keys:
        names = key.split("+")
        raw = raw_features.loc[
            raw_features["n_positive"].ge(minimum_positive)
            & np.isfinite(raw_features[names]).all(axis=1)
        ].reset_index(drop=True)
        strict = all_v2_features.loc[
            all_v2_features["n_positive"].ge(strict_minimum)
            & np.isfinite(all_v2_features[names]).all(axis=1)
        ].reset_index(drop=True)
        labeled_v2 = fit_solution(v2_main, names, k)
        labeled_raw = fit_solution(raw, names, k)
        labeled_strict = fit_solution(strict, names, k)
        assignment = sku_assignment_stability(
            v2_main,
            names,
            k,
            repetitions=repetitions,
            sample_fraction=sample_fraction,
            seed=seed,
        )
        assignment.insert(0, "feature_set", key)
        assignment_rows.append(assignment)
        profiles = cluster_profiles(labeled_v2, profile_columns=PROFILE_COLUMNS)
        profiles.insert(0, "feature_set", key)
        profile_rows.append(profiles)
        effects = closest_cluster_profile_effects(
            labeled_v2,
            columns=[
                "ADI",
                "CV2",
                "nonzero_mean",
                "acf1",
                "trend_coef",
                "peak_ratio",
                "trailing_zero_share",
                "promo_response_index",
            ],
        )
        effects.insert(0, "feature_set", key)
        effect_rows.append(effects)
        rows.append(
            {
                "feature_set": key,
                "n_features": len(names),
                **algorithm_robustness(
                    labeled_v2, names, k, n_init=n_init, seed=seed
                ),
                "raw_v2_ari": compare_solution_labels(labeled_raw, labeled_v2)["ari"],
                "main_strict_ari": compare_solution_labels(labeled_v2, labeled_strict)["ari"],
                "assignment_stability_median": float(
                    assignment["assignment_stability"].median()
                ),
                "assignment_stability_p10": float(
                    assignment["assignment_stability"].quantile(0.10)
                ),
                "assignment_stability_lt_075": int(
                    assignment["assignment_stability"].lt(0.75).sum()
                ),
                "n_raw": len(raw),
                "n_v2": len(v2_main),
                "n_strict": len(strict),
            }
        )
    return (
        pd.DataFrame(rows),
        pd.concat(profile_rows, ignore_index=True),
        pd.concat(effect_rows, ignore_index=True),
        pd.concat(assignment_rows, ignore_index=True),
    )


def _render_report(
    path: Path,
    *,
    summary: dict[str, Any],
    grid: pd.DataFrame,
    diagnostics: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    shortlist = grid.loc[grid["pareto"]].merge(
        diagnostics, on=["feature_set", "n_features"], how="left"
    )
    shortlist_table = _markdown_table(
        shortlist.sort_values(["n_features", "feature_set"]),
        [
            "feature_set",
            "n_features",
            "silhouette",
            "stability_ari_median",
            "minimum_cluster_jaccard_median",
            "cluster_sizes",
            "raw_v2_ari",
            "main_strict_ari",
            "ward_kmeans_ari",
        ],
    )
    corr_pairs = correlations.loc[
        correlations["feature_left"].ne(correlations["feature_right"])
    ].copy()
    corr_pairs["pair"] = corr_pairs["feature_left"] + " / " + corr_pairs["feature_right"]
    corr_pairs = corr_pairs.loc[corr_pairs["abs_spearman"].ge(0.80)].sort_values(
        "abs_spearman", ascending=False
    )
    corr_table = (
        _markdown_table(corr_pairs.head(12), ["pair", "pearson", "spearman"])
        if len(corr_pairs)
        else "没有绝对 Spearman 相关达到 0.80 的特征对。"
    )
    text = f"""# 业务导向聚类特征筛选 V3.0

> 证据标签：回顾性方法开发，不是独立确认性结果。协议提交发生在本轮聚类运行之前。

## 冻结设计

- 正式主池：{', '.join(summary['feature_gate']['admitted_features'])}
- 理论锚点：ADI、CV2
- 正式组合：{summary['feature_gate']['n_feature_sets']} 个
- 主样本：{summary['sample_counts']['main']} 个 SKU
- Ward K=2；80% 无放回子样本；重复 {summary['stability_repetitions']} 次
- 选择规则：结构门后使用 Silhouette 与稳定性 ARI 的 Pareto 短名单，不建立综合分

## Pareto 短名单

{shortlist_table}

当前状态：**{summary['decision_status']}**。如果短名单不止一个，报告多个候选，不利用预测结果强行反选聚类特征。

## 高相关诊断

{corr_table}

高相关只作为重复赋权风险提示；除精确恒等关系外，不根据当前结果临时删除候选。

## 解释边界

`promo_response_index` 是用户提供 Amazon 促销月份分布形成的业务日历代理，不是 SKU 级折扣或广告投入，也不能用于促销因果推断。年度季节性因 32 个 SKU 只有 30 周记录，另在不少于 104 周的长历史样本中检验，不参与本表主特征重选。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_business_feature_pipeline(
    config_path: str | Path, *, debug: bool = False
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    if config["project"].get("analysis_mode") != "retrospective_method_development":
        raise RuntimeError("V3.0 is explicitly a retrospective method-development run")
    gate = load_feature_gate(
        project_root / config["features"]["evidence_registry"],
        required_anchors=config["features"]["required_anchors"],
        expected_sha256=config["features"]["evidence_registry_sha256"],
        require_finalized=True,
    )
    feature_sets = enumerate_admitted_feature_sets(gate)
    if len(feature_sets) != 64:
        raise RuntimeError(f"V3.0 requires exactly 64 formal combinations, got {len(feature_sets)}")

    output_relative = config["outputs"]["root"]
    output = project_root / (f"{output_relative}_debug" if debug else output_relative)
    audit_root = output / "audit"
    cluster_root = output / "clustering"
    audit_root.mkdir(parents=True, exist_ok=True)
    cluster_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    loaded, cleaned, v2_features, raw_features = _load_inputs(config, project_root)
    missing = sorted(set(gate.admitted_features).difference(v2_features.columns))
    if missing:
        raise RuntimeError(f"Admitted V3.0 features are not implemented: {missing}")
    minimum_positive = config["samples"]["main_min_positive_weeks"]
    finite = np.isfinite(v2_features[list(gate.admitted_features)]).all(axis=1)
    v2_main = v2_features.loc[
        v2_features["n_positive"].ge(minimum_positive) & finite
    ].reset_index(drop=True)
    expected_main = int(v2_features["n_positive"].ge(minimum_positive).sum())
    if len(v2_main) != expected_main:
        raise RuntimeError(
            "Formal business features changed the main sample; all 64 combinations "
            f"must use the same SKUs ({len(v2_main)} != {expected_main})"
        )

    repetitions = (
        config["clustering"]["stability"]["debug_repetitions"]
        if debug
        else config["clustering"]["stability"]["repetitions"]
    )
    grid = evaluate_shared_transform_k2(
        v2_main,
        feature_sets,
        repetitions=repetitions,
        sample_fraction=config["clustering"]["stability"]["sample_fraction"],
        seed=config["project"]["seed"],
        k=config["clustering"]["operational_k"],
    )
    gates = config["clustering"]["structural_gates"]
    grid = mark_structural_pareto(
        grid,
        minimum_cluster_jaccard_median=gates["minimum_cluster_jaccard_median"],
        minimum_cluster_size=gates["minimum_cluster_size"],
        minimum_cluster_share=gates["minimum_cluster_share"],
    )
    grid["analysis_mode"] = config["project"]["analysis_mode"]
    grid["registry_sha256"] = gate.registry_sha256
    grid.to_csv(cluster_root / f"feature_grid_k2_{repetitions}.csv", index=False)

    shortlist = sorted(grid.loc[grid["pareto"], "feature_set"].tolist())
    benchmark = _feature_key(config["clustering"]["feature_selection"]["benchmark_features"])
    review_keys = sorted(set(shortlist + [benchmark]))
    diagnostics, profiles, effects, assignments = _candidate_diagnostics(
        review_keys,
        v2_main=v2_main,
        raw_features=raw_features,
        all_v2_features=v2_features,
        minimum_positive=minimum_positive,
        strict_minimum=config["samples"]["robustness_min_positive_weeks"],
        repetitions=repetitions,
        sample_fraction=config["clustering"]["stability"]["sample_fraction"],
        n_init=config["clustering"]["kmeans_robustness"]["n_init"],
        seed=config["project"]["seed"],
        k=config["clustering"]["operational_k"],
    )
    diagnostics.to_csv(cluster_root / "pareto_candidate_robustness.csv", index=False)
    profiles.to_csv(cluster_root / "pareto_candidate_profiles.csv", index=False)
    effects.to_csv(cluster_root / "pareto_candidate_profile_effects.csv", index=False)
    assignments.to_csv(cluster_root / "pareto_candidate_assignment_stability.csv", index=False)

    names = list(gate.admitted_features)
    pearson = v2_main[names].corr(method="pearson")
    spearman = v2_main[names].corr(method="spearman")
    correlation_rows = []
    for left in names:
        for right in names:
            correlation_rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "pearson": float(pearson.loc[left, right]),
                    "spearman": float(spearman.loc[left, right]),
                    "abs_spearman": float(abs(spearman.loc[left, right])),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(cluster_root / "feature_correlations.csv", index=False)
    v2_features.to_csv(cluster_root / "features_v2_all_skus.csv", index=False)
    raw_features.to_csv(cluster_root / "features_raw_all_skus.csv", index=False)
    v2_main.to_csv(cluster_root / "features_v2_main_sample.csv", index=False)
    cleaned.intervals.to_csv(audit_root / "v2_intervals.csv", index=False)

    long_history_n = int(
        (
            v2_features["n_positive"].ge(minimum_positive)
            & v2_features["n_observed"].ge(
                config["samples"]["seasonality_min_observed_weeks"]
            )
        ).sum()
    )
    summary = {
        "analysis_mode": config["project"]["analysis_mode"],
        "confirmatory": False,
        "protocol_version": config["project"]["protocol_version"],
        "feature_gate": {
            "registry_sha256": gate.registry_sha256,
            "anchors": list(gate.anchors),
            "supplementaries": list(gate.supplementaries),
            "admitted_features": list(gate.admitted_features),
            "n_feature_sets": len(feature_sets),
        },
        "stability_repetitions": repetitions,
        "operational_k": config["clustering"]["operational_k"],
        "pareto_shortlist": shortlist,
        "decision_status": (
            "唯一 Pareto 候选，可进入后续稳健性解释"
            if len(shortlist) == 1
            else f"存在 {len(shortlist)} 个 Pareto 候选，需要稳健性与业务画像审阅"
        ),
        "sample_counts": {
            "all_skus": len(v2_features),
            "main": len(v2_main),
            "strict": int(v2_features["n_positive"].ge(10).sum()),
            "long_history": long_history_n,
            "short_history_excluded_from_seasonality": int(len(v2_main) - long_history_n),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(cluster_root / "business_feature_screening_summary.json", summary)
    audit = {
        **loaded.audit,
        "expected_sha256_match": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "protocol_version": config["project"]["protocol_version"],
        "random_seed": config["project"]["seed"],
        "v2": cleaned.summary,
        "sample_counts": summary["sample_counts"],
        "feature_identities": verify_feature_identities(v2_features),
    }
    _write_json(audit_root / "audit_summary.json", audit)
    report_path = project_root / config["outputs"]["report"]
    if debug:
        report_path = report_path.with_name(f"{report_path.stem}_debug{report_path.suffix}")
    _render_report(
        report_path,
        summary=summary,
        grid=grid,
        diagnostics=diagnostics,
        correlations=correlations,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the V3.0 business-aware K=2 feature screening pipeline."
    )
    parser.add_argument("--config", default="config/analysis_business_features.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_business_feature_pipeline(args.config, debug=args.debug),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
