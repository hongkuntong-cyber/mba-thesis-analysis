from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from .cleaning_v2 import apply_v2_cleaning
from .clustering import (
    algorithm_robustness,
    cluster_profile_intervals,
    cluster_profiles,
    compare_solution_labels,
    evaluate_feature_grid,
    fit_solution,
    mark_pareto,
    select_theory_benchmark_solution,
)
from .config import load_config
from .data_audit import load_workbook_long
from .feature_gate import enumerate_admitted_feature_sets, load_feature_gate
from .features import compute_features, verify_feature_identities
from .stability import sku_assignment_stability


PROFILE_COLUMNS = [
    "ADI",
    "CV2",
    "approx_entropy",
    "trailing_zero_share",
    "nonzero_mean",
    "mean_sales",
    "zero_ratio",
    "total_sales",
    "n_positive",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_theory_first_inputs(
    config: dict[str, Any], project_root: Path
) -> tuple[Any, Any, pd.DataFrame]:
    input_path = project_root / config["input"]["workbook"]
    loaded = load_workbook_long(
        input_path,
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    expected_sha = config["input"]["expected_sha256"]
    if loaded.audit["sha256"] != expected_sha:
        raise RuntimeError(
            "Raw workbook SHA256 does not match the frozen V2.1 configuration"
        )
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")
    cleaned = apply_v2_cleaning(loaded.weekly_complete, **config["cleaning_v2"])
    return loaded, cleaned, compute_features(cleaned.weekly)


def _render_report(
    path: Path,
    *,
    summary: dict[str, Any],
    k2_rows: pd.DataFrame,
) -> None:
    table = k2_rows[
        [
            "feature_set",
            "n_features",
            "silhouette",
            "stability_ari_median",
            "min_cluster_size",
            "cluster_sizes",
            "k2_pareto",
        ]
    ].copy()
    for column in ["silhouette", "stability_ari_median"]:
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    table["k2_pareto"] = table["k2_pareto"].map(lambda value: "yes" if value else "no")
    headers = [str(column) for column in table.columns]
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    markdown_table = "\n".join(markdown_lines)
    selected = " + ".join(summary["selection"]["feature_names"])
    robustness = summary["robustness"]
    text = f"""# 理论先行特征门：回顾性方法开发结果

> 证据标签：**回顾性方法开发，不是新的确认性检验**。同一工作簿的旧结果已经被查看；本文件不能被表述为事前注册结果。

## 冻结输入

- 协议版本：{summary['protocol_version']}
- 证据登记表 SHA256：`{summary['feature_gate']['registry_sha256']}`
- 理论锚点：{', '.join(summary['feature_gate']['anchors'])}
- 补充候选：{', '.join(summary['feature_gate']['supplementaries'])}
- 合规候选组合数：{summary['feature_gate']['n_feature_sets']}
- 主聚类 K：2；K=3–6 只作敏感性分析
- 稳定性重复次数：{summary['stability_repetitions']}

## K=2 主特征选择

选择结果：`{selected}`

选择理由：{summary['selection']['reason']}

{markdown_table}

## 稳健性

- Ward 与 K-means ARI：{robustness['ward_kmeans_ari']:.4f}
- Raw 与 V2 共同 SKU 的 ARI：{robustness['raw_v2']['ari']:.4f}（N={robustness['raw_v2']['n_common_skus']}）
- 正需求周 ≥5 与 ≥10 的共同 SKU ARI：{robustness['main_strict']['ari']:.4f}（N={robustness['main_strict']['n_common_skus']}）
- SKU 归属稳定性中位数：{robustness['assignment_stability_median']:.4f}

## 解释边界

本轮只验证“文献准入门能否被代码严格执行”以及冻结候选在既有数据上的回顾性表现。预测算法池和簇—模型路由不能反向改变候选特征或主特征集。真正的确认性判断要等待新增、未查看的完整周数据。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_theory_first_pipeline(
    config_path: str | Path,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    if config["project"].get("analysis_mode") != "retrospective_method_development":
        raise RuntimeError(
            "This entrypoint requires analysis_mode=retrospective_method_development; "
            "a future confirmatory run needs a separately frozen configuration."
        )

    registry_path = project_root / config["features"]["evidence_registry"]
    gate = load_feature_gate(
        registry_path,
        required_anchors=config["features"]["required_anchors"],
        expected_sha256=config["features"]["evidence_registry_sha256"],
        require_finalized=True,
    )
    feature_sets = enumerate_admitted_feature_sets(gate)
    if config["features"].get("selection_source") != "evidence_registry_only":
        raise RuntimeError("Feature selection must be sourced only from the evidence registry")

    output_relative = config["outputs"]["root"]
    output = project_root / (f"{output_relative}_debug" if debug else output_relative)
    audit_root = output / "audit"
    cluster_root = output / "clustering"
    audit_root.mkdir(parents=True, exist_ok=True)
    cluster_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    loaded, cleaned, features = _load_theory_first_inputs(config, project_root)
    missing_computed = sorted(set(gate.admitted_features).difference(features.columns))
    if missing_computed:
        raise RuntimeError(f"Admitted features are not implemented: {missing_computed}")
    finite = np.isfinite(features[list(gate.admitted_features)]).all(axis=1)
    minimum_positive = config["samples"]["main_min_positive_weeks"]
    main_features = features.loc[
        features["n_positive"].ge(minimum_positive) & finite
    ].reset_index(drop=True)
    if main_features.empty:
        raise RuntimeError("No SKU passes the frozen main-sample and feature-validity gates")

    features.to_csv(cluster_root / "features_v2_all_skus.csv", index=False)
    main_features.to_csv(cluster_root / "features_v2_main_sample.csv", index=False)
    cleaned.intervals.to_csv(audit_root / "v2_intervals.csv", index=False)

    repetitions = (
        config["clustering"]["stability"]["debug_repetitions"]
        if debug
        else config["clustering"]["stability"]["repetitions"]
    )
    grid = evaluate_feature_grid(
        main_features,
        anchors=gate.anchors,
        optional=gate.supplementaries,
        k_values=config["clustering"]["sensitivity_k_values"],
        repetitions=repetitions,
        sample_fraction=config["clustering"]["stability"]["sample_fraction"],
        seed=config["project"]["seed"],
    ).rename(columns={"pareto": "all_k_pareto"})
    expected_keys = {"+".join(values) for values in feature_sets}
    observed_keys = set(grid["feature_set"].unique())
    if observed_keys != expected_keys:
        raise RuntimeError(
            "Evaluated feature sets do not equal the frozen registry-derived sets: "
            f"expected {sorted(expected_keys)}, observed {sorted(observed_keys)}"
        )
    k2_marked = mark_pareto(
        grid.rename(columns={"all_k_pareto": "pareto"}),
        eligibility=grid["k"].eq(config["clustering"]["operational_k"]),
    )
    grid["k2_pareto"] = k2_marked["pareto"]
    grid["analysis_mode"] = config["project"]["analysis_mode"]
    grid["registry_sha256"] = gate.registry_sha256
    grid.to_csv(cluster_root / f"feature_k_grid_{repetitions}.csv", index=False)

    rule = config["clustering"]["feature_selection"]
    selection = select_theory_benchmark_solution(
        grid,
        benchmark_features=rule["benchmark_features"],
        benchmark_k=rule["benchmark_k"],
        minimum_cluster_jaccard_median=rule["minimum_cluster_jaccard_median"],
        minimum_cluster_size=rule["minimum_cluster_size"],
        minimum_cluster_share=rule["minimum_cluster_share"],
        epsilon=rule["epsilon"],
    )

    selected_features = list(selection.feature_names)
    selected_k = int(selection.k)
    labeled_v2 = fit_solution(main_features, selected_features, selected_k)
    raw_features = compute_features(cleaned.weekly, "sales_raw")
    raw_finite = np.isfinite(raw_features[selected_features]).all(axis=1)
    raw_main = raw_features.loc[
        raw_features["n_positive"].ge(minimum_positive) & raw_finite
    ].reset_index(drop=True)
    strict_minimum = config["samples"]["robustness_min_positive_weeks"]
    strict = features.loc[
        features["n_positive"].ge(strict_minimum)
        & np.isfinite(features[selected_features]).all(axis=1)
    ].reset_index(drop=True)
    labeled_raw = fit_solution(raw_main, selected_features, selected_k)
    labeled_strict = fit_solution(strict, selected_features, selected_k)

    assignment = sku_assignment_stability(
        main_features,
        selected_features,
        selected_k,
        repetitions=repetitions,
        sample_fraction=config["clustering"]["stability"]["sample_fraction"],
        seed=config["project"]["seed"],
    )
    robustness = {
        **algorithm_robustness(
            labeled_v2,
            selected_features,
            selected_k,
            n_init=config["clustering"]["kmeans_robustness"]["n_init"],
            seed=config["project"]["seed"],
        ),
        "raw_v2": compare_solution_labels(labeled_raw, labeled_v2),
        "main_strict": compare_solution_labels(labeled_v2, labeled_strict),
        "assignment_stability_median": float(assignment["assignment_stability"].median()),
        "assignment_stability_p10": float(assignment["assignment_stability"].quantile(0.10)),
    }

    profiles = cluster_profiles(labeled_v2, profile_columns=PROFILE_COLUMNS)
    intervals = cluster_profile_intervals(
        labeled_v2,
        repetitions=1000 if not debug else 50,
        seed=config["project"]["seed"],
        profile_columns=PROFILE_COLUMNS,
    )
    labeled_v2.to_csv(cluster_root / "cluster_assignments_v2_k2.csv", index=False)
    profiles.to_csv(cluster_root / "cluster_profiles_v2_k2.csv", index=False)
    intervals.to_csv(cluster_root / "cluster_profile_intervals_v2_k2.csv", index=False)
    assignment.to_csv(cluster_root / "sku_assignment_stability_k2.csv", index=False)

    audit_payload = {
        **loaded.audit,
        "expected_sha256_match": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "random_seed": config["project"]["seed"],
        "analysis_mode": config["project"]["analysis_mode"],
        "protocol_version": config["project"]["protocol_version"],
        "feature_gate": {
            "registry_path": str(gate.registry_path.relative_to(project_root)),
            "registry_sha256": gate.registry_sha256,
            "anchors": list(gate.anchors),
            "supplementaries": list(gate.supplementaries),
            "feature_sets": [list(values) for values in feature_sets],
        },
        "v2": cleaned.summary,
        "sample_counts": {
            "all_skus": int(len(features)),
            "positive_weeks_ge_2": int(features["n_positive"].ge(2).sum()),
            "positive_weeks_ge_5": int(features["n_positive"].ge(5).sum()),
            "positive_weeks_ge_10": int(features["n_positive"].ge(10).sum()),
            "main_feature_complete": int(len(main_features)),
        },
        "feature_identities": verify_feature_identities(features),
    }
    _write_json(audit_root / "audit_summary.json", audit_payload)

    summary = {
        "analysis_mode": config["project"]["analysis_mode"],
        "confirmatory": False,
        "protocol_version": config["project"]["protocol_version"],
        "feature_gate": {
            "registry_sha256": gate.registry_sha256,
            "anchors": list(gate.anchors),
            "supplementaries": list(gate.supplementaries),
            "n_feature_sets": len(feature_sets),
        },
        "stability_repetitions": repetitions,
        "selection": {
            "feature_names": selected_features,
            "k": selected_k,
            "reason": selection.reason,
            "row": selection.row,
        },
        "robustness": robustness,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(cluster_root / "theory_first_selection.json", summary)
    report_path = project_root / config["outputs"]["report"]
    if debug:
        report_path = report_path.with_name(f"{report_path.stem}_debug{report_path.suffix}")
    _render_report(
        report_path,
        summary=summary,
        k2_rows=grid.loc[grid["k"].eq(selected_k)].copy(),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the V2.1 theory-first, fail-closed feature gate and K=2 analysis."
    )
    parser.add_argument("--config", default="config/analysis_theory_first.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use the frozen debug repetition count and separate debug outputs.",
    )
    args = parser.parse_args()
    summary = run_theory_first_pipeline(args.config, debug=args.debug)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
