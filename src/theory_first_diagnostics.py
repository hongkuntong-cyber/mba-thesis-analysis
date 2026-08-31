from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cleaning_v2 import apply_v2_cleaning
from .clustering import compare_solution_labels, fit_solution
from .config import load_config
from .data_audit import load_workbook_long
from .feature_gate import enumerate_admitted_feature_sets, load_feature_gate
from .features import compute_features


def _cluster_sizes(frame: pd.DataFrame) -> str:
    counts = frame["cluster"].value_counts().sort_index()
    return ";".join(f"{int(cluster)}:{int(count)}" for cluster, count in counts.items())


def _agreement_details(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    corrected_skus: set[str] | None = None,
) -> dict[str, float | int]:
    merged = left[["sku", "cluster"]].merge(
        right[["sku", "cluster"]], on="sku", suffixes=("_left", "_right")
    )
    changed = merged["cluster_left"].ne(merged["cluster_right"])
    details: dict[str, float | int] = {
        "n_common": int(len(merged)),
        "same_label_share": float((~changed).mean()) if len(merged) else np.nan,
        "n_changed": int(changed.sum()),
    }
    if corrected_skus is not None:
        corrected = merged["sku"].astype(str).isin(corrected_skus)
        details.update(
            {
                "n_changed_corrected_skus": int((changed & corrected).sum()),
                "n_changed_uncorrected_skus": int((changed & ~corrected).sum()),
            }
        )
    return details


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    table = frame[list(columns)].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def run_candidate_diagnostics(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output = project_root / config["outputs"]["root"]
    selection_path = output / "clustering" / "theory_first_selection.json"
    if not selection_path.exists():
        raise RuntimeError("Run the formal theory-first pipeline before diagnostics")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("stability_repetitions") != config["clustering"]["stability"]["repetitions"]:
        raise RuntimeError("Diagnostics require the formal 500-repeat output, not debug output")

    gate = load_feature_gate(
        project_root / config["features"]["evidence_registry"],
        required_anchors=config["features"]["required_anchors"],
        expected_sha256=config["features"]["evidence_registry_sha256"],
    )
    if selection["feature_gate"]["registry_sha256"] != gate.registry_sha256:
        raise RuntimeError("Formal output and current feature registry hashes differ")

    v2_path = output / "clustering" / "features_v2_all_skus.csv"
    v2_features = pd.read_csv(v2_path)
    loaded = load_workbook_long(
        project_root / config["input"]["workbook"],
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook hash changed after the formal run")
    cleaned = apply_v2_cleaning(loaded.weekly_complete, **config["cleaning_v2"])
    raw_features = compute_features(cleaned.weekly, "sales_raw")
    raw_features.to_csv(output / "clustering" / "features_raw_all_skus.csv", index=False)
    corrected_skus = set(
        cleaned.weekly.loc[cleaned.weekly["v2_corrected"], "sku"].astype(str).unique()
    )

    minimum_positive = config["samples"]["main_min_positive_weeks"]
    strict_minimum = config["samples"]["robustness_min_positive_weeks"]
    k = config["clustering"]["operational_k"]
    rows: list[dict[str, Any]] = []
    for feature_tuple in enumerate_admitted_feature_sets(gate):
        names = list(feature_tuple)
        v2_main = v2_features.loc[
            v2_features["n_positive"].ge(minimum_positive)
            & np.isfinite(v2_features[names]).all(axis=1)
        ].reset_index(drop=True)
        raw_main = raw_features.loc[
            raw_features["n_positive"].ge(minimum_positive)
            & np.isfinite(raw_features[names]).all(axis=1)
        ].reset_index(drop=True)
        strict = v2_features.loc[
            v2_features["n_positive"].ge(strict_minimum)
            & np.isfinite(v2_features[names]).all(axis=1)
        ].reset_index(drop=True)
        labeled_v2 = fit_solution(v2_main, names, k)
        labeled_raw = fit_solution(raw_main, names, k)
        labeled_strict = fit_solution(strict, names, k)
        raw_v2 = compare_solution_labels(labeled_raw, labeled_v2)
        main_strict = compare_solution_labels(labeled_v2, labeled_strict)
        raw_v2_agreement = _agreement_details(
            labeled_raw, labeled_v2, corrected_skus=corrected_skus
        )
        strict_agreement = _agreement_details(labeled_v2, labeled_strict)
        rows.append(
            {
                "feature_set": "+".join(names),
                "n_features": len(names),
                "raw_v2_ari": raw_v2["ari"],
                "raw_v2_same_label_share": raw_v2_agreement["same_label_share"],
                "raw_v2_changed_skus": raw_v2_agreement["n_changed"],
                "changed_corrected_skus": raw_v2_agreement[
                    "n_changed_corrected_skus"
                ],
                "changed_uncorrected_skus": raw_v2_agreement[
                    "n_changed_uncorrected_skus"
                ],
                "main_strict_ari": main_strict["ari"],
                "main_strict_same_label_share": strict_agreement["same_label_share"],
                "main_strict_changed_skus": strict_agreement["n_changed"],
                "v2_cluster_sizes": _cluster_sizes(labeled_v2),
                "raw_cluster_sizes": _cluster_sizes(labeled_raw),
                "strict_cluster_sizes": _cluster_sizes(labeled_strict),
                "n_v2": len(v2_main),
                "n_raw": len(raw_main),
                "n_strict": len(strict),
            }
        )

    diagnostics = pd.DataFrame(rows)
    csv_path = output / "clustering" / "candidate_robustness_k2.csv"
    diagnostics.to_csv(csv_path, index=False)
    selected_key = "+".join(selection["selection"]["feature_names"])
    selected_row = diagnostics.loc[diagnostics["feature_set"].eq(selected_key)]
    if len(selected_row) != 1:
        raise RuntimeError("Selected feature set is missing from candidate diagnostics")
    selected_payload = selected_row.iloc[0].to_dict()
    payload = {
        "analysis_mode": config["project"]["analysis_mode"],
        "confirmatory": False,
        "selection_changed": False,
        "selected_feature_set": selected_key,
        "selected_diagnostics": selected_payload,
        "corrected_sku_count": len(corrected_skus),
        "interpretation": (
            "The frozen primary-metric selection is unchanged. Raw/V2 and sample-threshold "
            "results are robustness diagnostics and cannot be used to reselect features."
        ),
    }
    json_path = output / "clustering" / "candidate_robustness_k2.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    report = project_root / "reports" / "theory_first_robustness_diagnostics.md"
    table = _markdown_table(
        diagnostics,
        [
            "feature_set",
            "raw_v2_ari",
            "raw_v2_same_label_share",
            "raw_v2_changed_skus",
            "main_strict_ari",
            "main_strict_same_label_share",
        ],
    )
    selected_raw_v2 = float(selected_payload["raw_v2_ari"])
    selected_strict = float(selected_payload["main_strict_ari"])
    report.write_text(
        f"""# 理论先行 K=2 稳健性诊断

> 本文件是正式 500 次主网格之后的诊断输出，不改变已冻结的特征选择规则，也不属于新的确认性检验。

## 候选组合诊断

{table}

## 入选方案解释

冻结规则入选 `{' + '.join(selection['selection']['feature_names'])}`。其 Raw/V2 ARI 为 {selected_raw_v2:.4f}，正需求周 ≥5 与 ≥10 样本的 ARI 为 {selected_strict:.4f}。

Raw/V2 ARI 明显远离 1，说明 V2 代理修正虽然只直接影响少量 SKU，也会通过全样本变换与 Ward 合并路径改变其他 SKU 的边界归属。因此，当前结果可以表述为“在 V2 口径和 80% 子样本条件下达到预定簇级稳定门槛”，但不能表述为“对清洗口径高度稳健”。

其他候选组合的诊断结果不得反向用于改写主特征集。未来确认性检验应继续报告该敏感性，并重点观察新增数据后 Raw/V2 一致性是否改善。
""",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Raw/V2 and sample-threshold robustness without reselection."
    )
    parser.add_argument("--config", default="config/analysis_theory_first.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            run_candidate_diagnostics(args.config),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
