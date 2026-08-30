from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

from .features import enumerate_confirmatory_feature_sets
from .stability import (
    canonicalize_labels,
    fit_ward,
    fit_ward_matrix,
    subsample_stability_multi_k,
    transform_features,
)


@dataclass(frozen=True)
class SelectedSolution:
    feature_names: tuple[str, ...]
    k: int
    reason: str
    row: dict[str, object]


def _feature_key(feature_names: Iterable[str]) -> str:
    return "+".join(feature_names)


def evaluate_feature_grid(
    features: pd.DataFrame,
    *,
    anchors: Iterable[str],
    optional: Iterable[str],
    k_values: Iterable[int],
    repetitions: int,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_sets = enumerate_confirmatory_feature_sets(anchors, optional)
    requested_k = [int(k) for k in k_values]
    for feature_names in feature_sets:
        try:
            transformed, _ = transform_features(features, feature_names)
            stability_by_k = subsample_stability_multi_k(
                features,
                feature_names,
                requested_k,
                repetitions=repetitions,
                sample_fraction=sample_fraction,
                seed=seed,
            )
            setup_error = ""
        except Exception as exc:
            transformed = np.empty((0, 0))
            stability_by_k = {}
            setup_error = f"{type(exc).__name__}: {exc}"
        for k in requested_k:
            record: dict[str, object] = {
                "feature_set": _feature_key(feature_names),
                "n_features": len(feature_names),
                "k": int(k),
                "n_skus": int(len(features)),
                "valid": False,
                "error": "",
            }
            try:
                if setup_error:
                    raise ValueError(setup_error)
                labels = fit_ward_matrix(transformed, int(k))
                canonical = canonicalize_labels(features, labels)
                counts = pd.Series(canonical).value_counts().sort_index()
                record.update(
                    {
                        "silhouette": float(silhouette_score(transformed, labels)),
                        "calinski_harabasz": float(calinski_harabasz_score(transformed, labels)),
                        "min_cluster_size": int(counts.min()),
                        "max_cluster_size": int(counts.max()),
                        "cluster_sizes": ";".join(f"{idx}:{value}" for idx, value in counts.items()),
                    }
                )
                record.update(stability_by_k[int(k)])
                record["valid"] = True
            except Exception as exc:  # retain failed combinations in the audit table
                record["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)

    results = pd.DataFrame(rows)
    return mark_pareto(results)


def mark_pareto(results: pd.DataFrame, eligibility: pd.Series | None = None) -> pd.DataFrame:
    """Mark non-dominated rows on Silhouette and median stability ARI."""
    output = results.copy()
    output["pareto"] = False
    if eligibility is None:
        eligibility = pd.Series(True, index=output.index)
    valid = output.loc[output["valid"] & eligibility.reindex(output.index, fill_value=False)].copy()
    for idx, row in valid.iterrows():
        dominated = (
            (valid["silhouette"] >= row["silhouette"])
            & (valid["stability_ari_median"] >= row["stability_ari_median"])
            & (
                (valid["silhouette"] > row["silhouette"])
                | (valid["stability_ari_median"] > row["stability_ari_median"])
            )
        ).any()
        output.loc[idx, "pareto"] = not bool(dominated)
    return output


def select_operational_solution(
    results: pd.DataFrame,
    *,
    benchmark_features: Iterable[str],
    benchmark_k: int,
    minimum_cluster_jaccard_median: float | None = None,
    minimum_cluster_size: int | None = None,
    minimum_cluster_share: float | None = None,
) -> SelectedSolution:
    benchmark_key = _feature_key(benchmark_features)
    valid = results.loc[results["valid"]].copy()
    if minimum_cluster_jaccard_median is not None:
        valid = valid.loc[
            valid["cluster_jaccard_medians"].map(
                lambda value: min(float(item) for item in json.loads(value).values())
                >= minimum_cluster_jaccard_median
            )
        ].copy()
    if minimum_cluster_size is not None:
        valid = valid.loc[valid["min_cluster_size"].ge(minimum_cluster_size)].copy()
    if minimum_cluster_share is not None:
        valid = valid.loc[
            valid["min_cluster_size"].div(valid["n_skus"]).ge(minimum_cluster_share)
        ].copy()
    benchmark = valid.loc[
        valid["feature_set"].eq(benchmark_key) & valid["k"].eq(benchmark_k)
    ]
    if len(benchmark) != 1:
        raise ValueError(f"Expected one benchmark row for {benchmark_key}, K={benchmark_k}")
    benchmark_row = benchmark.iloc[0]
    epsilon = 1e-12
    challengers = valid.loc[
        (valid["silhouette"] >= benchmark_row["silhouette"] - epsilon)
        & (valid["stability_ari_median"] >= benchmark_row["stability_ari_median"] - epsilon)
        & (
            (valid["silhouette"] > benchmark_row["silhouette"] + epsilon)
            | (valid["stability_ari_median"] > benchmark_row["stability_ari_median"] + epsilon)
        )
    ].copy()
    pool = pd.concat([benchmark, challengers], ignore_index=True).drop_duplicates(
        ["feature_set", "k"]
    )
    selected = pool.sort_values(
        ["n_features", "stability_ari_median", "silhouette", "k"],
        ascending=[True, False, False, True],
    ).iloc[0]
    is_benchmark = selected["feature_set"] == benchmark_key and int(selected["k"]) == benchmark_k
    reason = (
        "No challenger dominated the pre-specified four-feature K=2 benchmark; benchmark retained."
        if is_benchmark
        else "Selected after the frozen cluster-level stability gate, then by benchmark dominance and parsimony."
    )
    return SelectedSolution(
        feature_names=tuple(str(selected["feature_set"]).split("+")),
        k=int(selected["k"]),
        reason=reason,
        row=selected.to_dict(),
    )


def fit_solution(features: pd.DataFrame, feature_names: Iterable[str], k: int) -> pd.DataFrame:
    output = features.copy().reset_index(drop=True)
    fit = fit_ward(output, feature_names, k)
    output["cluster"] = canonicalize_labels(output, fit.labels)
    return output


def cluster_profiles(labeled_features: pd.DataFrame) -> pd.DataFrame:
    profile_columns = [
        "ADI",
        "CV2",
        "nonzero_mean",
        "mean_sales",
        "median_sales",
        "std_sales",
        "acf1",
        "zero_ratio",
        "total_sales",
        "n_positive",
    ]
    aggregations: dict[str, tuple[str, str]] = {"n_skus": ("sku", "count")}
    for column in profile_columns:
        aggregations[f"{column}_median"] = (column, "median")
        aggregations[f"{column}_mean"] = (column, "mean")
    return labeled_features.groupby("cluster", as_index=False).agg(**aggregations)


def cluster_profile_intervals(
    labeled_features: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """SKU-level percentile intervals for raw-scale cluster medians."""
    profile_columns = [
        "ADI",
        "CV2",
        "nonzero_mean",
        "mean_sales",
        "median_sales",
        "std_sales",
        "acf1",
        "zero_ratio",
        "total_sales",
        "n_positive",
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for cluster, frame in labeled_features.groupby("cluster", sort=True):
        for column in profile_columns:
            values = frame[column].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            boot = np.empty(repetitions, dtype=float)
            for idx in range(repetitions):
                boot[idx] = float(np.median(rng.choice(values, size=len(values), replace=True)))
            rows.append(
                {
                    "cluster": int(cluster),
                    "feature": column,
                    "n_skus": int(len(values)),
                    "median": float(np.median(values)),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def closest_cluster_profile_effects(
    labeled_features: pd.DataFrame,
    columns: Iterable[str] = ("ADI", "CV2", "nonzero_mean", "acf1"),
) -> pd.DataFrame:
    """Compare each cluster with its nearest profile using pooled-SD median effects."""
    names = list(columns)
    cluster_ids = sorted(int(value) for value in labeled_features["cluster"].unique())
    pair_rows: list[dict[str, object]] = []
    for left_idx, left in enumerate(cluster_ids):
        left_frame = labeled_features.loc[labeled_features["cluster"].eq(left)]
        for right in cluster_ids[left_idx + 1 :]:
            right_frame = labeled_features.loc[labeled_features["cluster"].eq(right)]
            effects: dict[str, float] = {}
            for column in names:
                left_values = left_frame[column].dropna().to_numpy(dtype=float)
                right_values = right_frame[column].dropna().to_numpy(dtype=float)
                left_var = float(np.var(left_values, ddof=1)) if len(left_values) > 1 else 0.0
                right_var = float(np.var(right_values, ddof=1)) if len(right_values) > 1 else 0.0
                denominator = max(len(left_values) + len(right_values) - 2, 1)
                pooled = np.sqrt(
                    max(
                        ((len(left_values) - 1) * left_var + (len(right_values) - 1) * right_var)
                        / denominator,
                        0.0,
                    )
                )
                difference = abs(float(np.median(left_values)) - float(np.median(right_values)))
                effects[column] = difference / pooled if pooled > 0 else (np.inf if difference > 0 else 0.0)
            finite_vector = np.asarray([effects[name] for name in names], dtype=float)
            finite_vector = np.where(np.isfinite(finite_vector), finite_vector, 1e6)
            pair_rows.append(
                {
                    "left_cluster": left,
                    "right_cluster": right,
                    "profile_distance": float(np.sqrt(np.sum(finite_vector**2))),
                    "max_effect": float(max(effects.values())),
                    "max_effect_feature": max(effects, key=effects.get),
                    **{f"effect_{name}": value for name, value in effects.items()},
                }
            )
    pairs = pd.DataFrame(pair_rows)
    rows: list[dict[str, object]] = []
    for cluster in cluster_ids:
        candidates = pairs.loc[
            pairs["left_cluster"].eq(cluster) | pairs["right_cluster"].eq(cluster)
        ].sort_values(["profile_distance", "left_cluster", "right_cluster"])
        nearest = candidates.iloc[0]
        other = int(
            nearest["right_cluster"]
            if int(nearest["left_cluster"]) == cluster
            else nearest["left_cluster"]
        )
        rows.append(
            {
                "cluster": cluster,
                "nearest_cluster": other,
                "profile_distance": float(nearest["profile_distance"]),
                "max_effect": float(nearest["max_effect"]),
                "max_effect_feature": str(nearest["max_effect_feature"]),
                **{f"effect_{name}": float(nearest[f"effect_{name}"]) for name in names},
            }
        )
    return pd.DataFrame(rows)


def algorithm_robustness(
    labeled_features: pd.DataFrame,
    feature_names: Iterable[str],
    k: int,
    *,
    n_init: int,
    seed: int,
) -> dict[str, float]:
    fit = fit_ward(labeled_features, feature_names, k)
    ward_labels = fit.labels
    kmeans_labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(
        fit.transformed
    )
    return {"ward_kmeans_ari": float(adjusted_rand_score(ward_labels, kmeans_labels))}


def compare_solution_labels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label_column: str = "cluster",
) -> dict[str, float | int]:
    merged = left[["sku", label_column]].merge(
        right[["sku", label_column]], on="sku", suffixes=("_left", "_right")
    )
    return {
        "n_common_skus": int(len(merged)),
        "ari": float(
            adjusted_rand_score(merged[f"{label_column}_left"], merged[f"{label_column}_right"])
        )
        if len(merged)
        else np.nan,
    }
