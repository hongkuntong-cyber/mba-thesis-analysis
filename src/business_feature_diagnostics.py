from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .business_feature_pipeline import (
    PROFILE_COLUMNS,
    _load_inputs,
    evaluate_shared_transform_k2,
)
from .clustering import (
    cluster_profile_intervals,
    compare_solution_labels,
    fit_solution,
)
from .config import load_config
from .feature_gate import enumerate_admitted_feature_sets, load_feature_gate
from .stability import (
    canonicalize_labels,
    fit_ward_matrix,
    subsample_stability_multi_k,
    transform_features,
)


def _key(names: Iterable[str]) -> str:
    return "+".join(names)


def feature_increment_diagnostics(
    grid: pd.DataFrame,
    *,
    anchors: Iterable[str],
    supplementaries: Iterable[str],
) -> pd.DataFrame:
    """Describe paired metric changes when each supplementary feature is added.

    This is a post-screening diagnostic, not an alternative selection score.
    Each comparison holds the other supplementary constructs fixed.
    """
    anchor_names = tuple(anchors)
    optional_names = tuple(supplementaries)
    indexed = grid.set_index("feature_set", drop=False)
    rows: list[dict[str, Any]] = []
    for feature in optional_names:
        others = tuple(name for name in optional_names if name != feature)
        deltas: list[dict[str, float]] = []
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                base_names = (*anchor_names, *subset)
                added_names = (
                    *anchor_names,
                    *(name for name in optional_names if name in {*subset, feature}),
                )
                base = indexed.loc[_key(base_names)]
                added = indexed.loc[_key(added_names)]
                deltas.append(
                    {
                        "silhouette": float(added["silhouette"] - base["silhouette"]),
                        "stability": float(
                            added["stability_ari_median"]
                            - base["stability_ari_median"]
                        ),
                        "jaccard": float(
                            added["minimum_cluster_jaccard_median"]
                            - base["minimum_cluster_jaccard_median"]
                        ),
                    }
                )
        delta = pd.DataFrame(deltas)
        rows.append(
            {
                "feature": feature,
                "paired_comparisons": len(delta),
                "median_delta_silhouette": float(delta["silhouette"].median()),
                "mean_delta_silhouette": float(delta["silhouette"].mean()),
                "share_silhouette_improved": float(delta["silhouette"].gt(0).mean()),
                "median_delta_stability_ari": float(delta["stability"].median()),
                "mean_delta_stability_ari": float(delta["stability"].mean()),
                "share_stability_improved": float(delta["stability"].gt(0).mean()),
                "share_both_improved": float(
                    (delta["silhouette"].gt(0) & delta["stability"].gt(0)).mean()
                ),
                "median_delta_minimum_jaccard": float(delta["jaccard"].median()),
                "pareto_candidate_count": int(
                    indexed.loc[
                        indexed["pareto"]
                        & indexed["feature_set"].map(
                            lambda value: feature in str(value).split("+")
                        )
                    ].shape[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _k_sensitivity(
    features: pd.DataFrame,
    review_keys: Iterable[str],
    *,
    k_values: Iterable[int],
    repetitions: int,
    sample_fraction: float,
    seed: int,
    structural_gates: dict[str, float | int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    requested_k = [int(value) for value in k_values]
    for feature_set in review_keys:
        names = feature_set.split("+")
        transformed, _ = transform_features(features, names)
        stability = subsample_stability_multi_k(
            features,
            names,
            requested_k,
            repetitions=repetitions,
            sample_fraction=sample_fraction,
            seed=seed,
        )
        for k in requested_k:
            labels = fit_ward_matrix(transformed, k)
            canonical = canonicalize_labels(features, labels)
            counts = pd.Series(canonical).value_counts().sort_index()
            jaccards = {
                int(cluster): float(value)
                for cluster, value in json.loads(
                    str(stability[k]["cluster_jaccard_medians"])
                ).items()
            }
            minimum_jaccard = min(jaccards.values())
            minimum_size = int(counts.min())
            rows.append(
                {
                    "feature_set": feature_set,
                    "k": k,
                    "n_skus": len(features),
                    "silhouette": float(silhouette_score(transformed, labels)),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(transformed, labels)
                    ),
                    "cluster_sizes": ";".join(
                        f"{int(cluster)}:{int(size)}"
                        for cluster, size in counts.items()
                    ),
                    "min_cluster_size": minimum_size,
                    "minimum_cluster_jaccard_median": minimum_jaccard,
                    "structural_eligible": bool(
                        minimum_jaccard
                        >= float(structural_gates["minimum_cluster_jaccard_median"])
                        and minimum_size
                        >= int(structural_gates["minimum_cluster_size"])
                        and minimum_size / len(features)
                        >= float(structural_gates["minimum_cluster_share"])
                    ),
                    **stability[k],
                }
            )
    return pd.DataFrame(rows)


def _added_feature_sensitivity(
    features: pd.DataFrame,
    review_keys: Iterable[str],
    *,
    extra_feature: str,
    repetitions: int,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    requested: list[tuple[str, ...]] = []
    mapping: list[dict[str, str]] = []
    for feature_set in review_keys:
        base = tuple(feature_set.split("+"))
        added = (*base, extra_feature)
        requested.extend([base, added])
        mapping.extend(
            [
                {
                    "feature_set": _key(base),
                    "base_feature_set": feature_set,
                    "variant": "base",
                },
                {
                    "feature_set": _key(added),
                    "base_feature_set": feature_set,
                    "variant": f"plus_{extra_feature}",
                },
            ]
        )
    unique_requested = list(dict.fromkeys(requested))
    evaluated = evaluate_shared_transform_k2(
        features,
        unique_requested,
        repetitions=repetitions,
        sample_fraction=sample_fraction,
        seed=seed,
        k=2,
    )
    output = pd.DataFrame(mapping).merge(evaluated, on="feature_set", how="left")
    output.insert(0, "sensitivity_feature", extra_feature)
    return output


def _pairwise_solution_ari(
    features: pd.DataFrame, review_keys: Iterable[str]
) -> pd.DataFrame:
    keys = list(review_keys)
    solutions = {
        key: fit_solution(features, key.split("+"), 2) for key in keys
    }
    rows = []
    for left, right in combinations(keys, 2):
        comparison = compare_solution_labels(solutions[left], solutions[right])
        rows.append(
            {
                "left_feature_set": left,
                "right_feature_set": right,
                **comparison,
            }
        )
    return pd.DataFrame(rows)


def run_diagnostics(
    config_path: str | Path, *, profile_bootstrap_repetitions: int = 1000
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output_root = project_root / config["outputs"]["root"] / "clustering"
    grid_path = output_root / "feature_grid_k2_500.csv"
    if not grid_path.exists():
        raise RuntimeError("Run the formal 500-repetition V3.0 screen first")
    grid = pd.read_csv(grid_path)
    if not grid["analysis_mode"].eq("retrospective_method_development").all():
        raise RuntimeError("Diagnostics require the retrospective V3.0 result grid")

    gate = load_feature_gate(
        project_root / config["features"]["evidence_registry"],
        required_anchors=config["features"]["required_anchors"],
        expected_sha256=config["features"]["evidence_registry_sha256"],
        require_finalized=True,
    )
    expected_sets = {_key(values) for values in enumerate_admitted_feature_sets(gate)}
    if set(grid["feature_set"]) != expected_sets:
        raise RuntimeError("Formal result grid does not match the 64 frozen feature sets")
    benchmark = _key(config["clustering"]["feature_selection"]["benchmark_features"])
    pareto_keys = sorted(grid.loc[grid["pareto"], "feature_set"].tolist())
    review_keys = sorted(set([benchmark, *pareto_keys]))

    _, _, v2_features, _ = _load_inputs(config, project_root)
    admitted = list(gate.admitted_features)
    main_mask = v2_features["n_positive"].ge(
        config["samples"]["main_min_positive_weeks"]
    ) & np.isfinite(v2_features[admitted]).all(axis=1)
    main = v2_features.loc[main_mask].reset_index(drop=True)
    long_names = [*admitted, "seasonality_idx"]
    long_mask = (
        v2_features["n_positive"].ge(config["samples"]["main_min_positive_weeks"])
        & v2_features["n_observed"].ge(
            config["samples"]["seasonality_min_observed_weeks"]
        )
        & np.isfinite(v2_features[long_names]).all(axis=1)
    )
    long_history = v2_features.loc[long_mask].reset_index(drop=True)

    repetitions = int(config["clustering"]["stability"]["repetitions"])
    sample_fraction = float(config["clustering"]["stability"]["sample_fraction"])
    seed = int(config["project"]["seed"])
    incremental = feature_increment_diagnostics(
        grid,
        anchors=gate.anchors,
        supplementaries=gate.supplementaries,
    )
    k_grid = _k_sensitivity(
        main,
        review_keys,
        k_values=config["clustering"]["post_selection_sensitivity_k_values"],
        repetitions=repetitions,
        sample_fraction=sample_fraction,
        seed=seed,
        structural_gates=config["clustering"]["structural_gates"],
    )
    seasonality = _added_feature_sensitivity(
        long_history,
        review_keys,
        extra_feature="seasonality_idx",
        repetitions=repetitions,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    entropy = _added_feature_sensitivity(
        main,
        review_keys,
        extra_feature="approx_entropy",
        repetitions=repetitions,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    pairwise = _pairwise_solution_ari(main, review_keys)

    interval_rows = []
    for feature_set in review_keys:
        labeled = fit_solution(main, feature_set.split("+"), 2)
        intervals = cluster_profile_intervals(
            labeled,
            repetitions=profile_bootstrap_repetitions,
            seed=seed,
            profile_columns=PROFILE_COLUMNS,
        )
        intervals.insert(0, "feature_set", feature_set)
        interval_rows.append(intervals)
    profile_intervals = pd.concat(interval_rows, ignore_index=True)

    incremental.to_csv(output_root / "feature_increment_diagnostics.csv", index=False)
    k_grid.to_csv(output_root / "post_selection_k2_to_k6.csv", index=False)
    seasonality.to_csv(output_root / "long_history_seasonality_sensitivity.csv", index=False)
    entropy.to_csv(output_root / "approx_entropy_sensitivity.csv", index=False)
    pairwise.to_csv(output_root / "candidate_pairwise_ari.csv", index=False)
    profile_intervals.to_csv(output_root / "candidate_profile_intervals.csv", index=False)

    summary = {
        "analysis_mode": "retrospective_method_development",
        "confirmatory": False,
        "review_feature_sets": review_keys,
        "pareto_feature_sets": pareto_keys,
        "main_sample_n": len(main),
        "long_history_sample_n": len(long_history),
        "stability_repetitions": repetitions,
        "profile_bootstrap_repetitions": profile_bootstrap_repetitions,
        "k_values": list(config["clustering"]["post_selection_sensitivity_k_values"]),
    }
    (output_root / "business_feature_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen post-selection diagnostics for V3.0 clustering."
    )
    parser.add_argument("--config", default="config/analysis_business_features.yaml")
    parser.add_argument("--profile-bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            run_diagnostics(
                args.config,
                profile_bootstrap_repetitions=args.profile_bootstrap_repetitions,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
