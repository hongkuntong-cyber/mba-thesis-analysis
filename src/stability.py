from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import PowerTransformer


@dataclass(frozen=True)
class ClusterFit:
    labels: np.ndarray
    transformed: np.ndarray
    transformer: PowerTransformer


def fit_ward_matrix(matrix: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matrix)


def transform_features(frame: pd.DataFrame, feature_names: Iterable[str]) -> tuple[np.ndarray, PowerTransformer]:
    names = list(feature_names)
    matrix = frame[names].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in feature set: {names}")
    variances = np.var(matrix, axis=0)
    if np.any(variances == 0):
        constant = [name for name, variance in zip(names, variances) if variance == 0]
        raise ValueError(f"Constant features: {constant}")
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformed = transformer.fit_transform(matrix)
    if not np.isfinite(transformed).all():
        raise ValueError(f"Non-finite transformed values: {names}")
    return transformed, transformer


def fit_ward(frame: pd.DataFrame, feature_names: Iterable[str], k: int) -> ClusterFit:
    transformed, transformer = transform_features(frame, feature_names)
    labels = fit_ward_matrix(transformed, k)
    return ClusterFit(labels=labels, transformed=transformed, transformer=transformer)


def canonicalize_labels(features: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    working = features[["ADI", "CV2", "nonzero_mean", "sku"]].copy()
    working["label"] = labels
    profile = (
        working.groupby("label", as_index=False)
        .agg(ADI=("ADI", "median"), CV2=("CV2", "median"), nonzero_mean=("nonzero_mean", "median"))
        .sort_values(["ADI", "CV2", "nonzero_mean", "label"], ascending=[True, True, False, True])
    )
    mapping = {int(old): idx + 1 for idx, old in enumerate(profile["label"].tolist())}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def _matched_jaccards(full_labels: np.ndarray, sample_labels: np.ndarray, k: int) -> dict[int, float]:
    scores = np.zeros((k, k), dtype=float)
    for full in range(k):
        full_mask = full_labels == full
        for sample in range(k):
            sample_mask = sample_labels == sample
            union = np.count_nonzero(full_mask | sample_mask)
            intersection = np.count_nonzero(full_mask & sample_mask)
            scores[full, sample] = intersection / union if union else 0.0
    row_ind, col_ind = linear_sum_assignment(-scores)
    return {int(row): float(scores[row, col]) for row, col in zip(row_ind, col_ind)}


def _align_sample_labels(
    full_labels: np.ndarray, sample_labels: np.ndarray, k: int
) -> np.ndarray:
    """Align arbitrary sample cluster IDs to the restricted full-sample IDs."""
    counts = np.zeros((k, k), dtype=int)
    for full in range(k):
        for sample in range(k):
            counts[full, sample] = int(
                np.count_nonzero((full_labels == full) & (sample_labels == sample))
            )
    full_ids, sample_ids = linear_sum_assignment(-counts)
    mapping = {int(sample): int(full) for full, sample in zip(full_ids, sample_ids)}
    return np.asarray([mapping[int(label)] for label in sample_labels], dtype=int)


def subsample_stability(
    features: pd.DataFrame,
    feature_names: Iterable[str],
    k: int,
    *,
    repetitions: int,
    sample_fraction: float,
    seed: int,
) -> dict[str, float | str]:
    names = list(feature_names)
    full_fit = fit_ward(features, names, k)
    full_labels = full_fit.labels
    n_rows = len(features)
    sample_size = max(k + 1, int(np.floor(n_rows * sample_fraction)))
    rng = np.random.default_rng(seed)
    aris: list[float] = []
    jaccards: dict[int, list[float]] = {cluster: [] for cluster in range(k)}

    for _ in range(repetitions):
        sample_indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        sample = features.iloc[sample_indices]
        sample_fit = fit_ward(sample, names, k)
        restricted_full = full_labels[sample_indices]
        aris.append(adjusted_rand_score(restricted_full, sample_fit.labels))
        matched = _matched_jaccards(restricted_full, sample_fit.labels, k)
        for cluster, score in matched.items():
            jaccards[cluster].append(score)

    ari_array = np.asarray(aris, dtype=float)
    cluster_medians = {
        str(cluster + 1): float(np.median(values)) if values else np.nan
        for cluster, values in jaccards.items()
    }
    return {
        "stability_ari_mean": float(np.mean(ari_array)),
        "stability_ari_median": float(np.median(ari_array)),
        "stability_ari_p10": float(np.quantile(ari_array, 0.10)),
        "stability_ari_p90": float(np.quantile(ari_array, 0.90)),
        "cluster_jaccard_medians": json.dumps(cluster_medians, ensure_ascii=False),
    }


def subsample_stability_multi_k(
    features: pd.DataFrame,
    feature_names: Iterable[str],
    k_values: Iterable[int],
    *,
    repetitions: int,
    sample_fraction: float,
    seed: int,
) -> dict[int, dict[str, float | str]]:
    """Reuse each subsample transformation across every requested K."""
    names = list(feature_names)
    requested_k = [int(k) for k in k_values]
    full_transformed, _ = transform_features(features, names)
    full_labels = {k: fit_ward_matrix(full_transformed, k) for k in requested_k}
    n_rows = len(features)
    sample_size = max(max(requested_k) + 1, int(np.floor(n_rows * sample_fraction)))
    rng = np.random.default_rng(seed)
    aris: dict[int, list[float]] = {k: [] for k in requested_k}
    jaccards: dict[int, dict[int, list[float]]] = {
        k: {cluster: [] for cluster in range(k)} for k in requested_k
    }

    for _ in range(repetitions):
        sample_indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        sample = features.iloc[sample_indices]
        sample_transformed, _ = transform_features(sample, names)
        for k in requested_k:
            sample_labels = fit_ward_matrix(sample_transformed, k)
            restricted_full = full_labels[k][sample_indices]
            aris[k].append(adjusted_rand_score(restricted_full, sample_labels))
            matched = _matched_jaccards(restricted_full, sample_labels, k)
            for cluster, score in matched.items():
                jaccards[k][cluster].append(score)

    output: dict[int, dict[str, float | str]] = {}
    for k in requested_k:
        ari_array = np.asarray(aris[k], dtype=float)
        cluster_medians = {
            str(cluster + 1): float(np.median(values)) if values else np.nan
            for cluster, values in jaccards[k].items()
        }
        output[k] = {
            "stability_ari_mean": float(np.mean(ari_array)),
            "stability_ari_median": float(np.median(ari_array)),
            "stability_ari_p10": float(np.quantile(ari_array, 0.10)),
            "stability_ari_p90": float(np.quantile(ari_array, 0.90)),
            "cluster_jaccard_medians": json.dumps(cluster_medians, ensure_ascii=False),
        }
    return output


def sku_assignment_stability(
    features: pd.DataFrame,
    feature_names: Iterable[str],
    k: int,
    *,
    repetitions: int,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Estimate each SKU's probability of returning to its full-sample cluster.

    This is an 80% no-replacement subsample diagnostic, not a traditional
    bootstrap probability. Transformations and Ward are refit inside each draw.
    """
    names = list(feature_names)
    full_fit = fit_ward(features, names, k)
    full_labels = full_fit.labels
    canonical = canonicalize_labels(features, full_labels)
    n_rows = len(features)
    sample_size = max(k + 1, int(np.floor(n_rows * sample_fraction)))
    rng = np.random.default_rng(seed)
    appearances = np.zeros(n_rows, dtype=int)
    matches = np.zeros(n_rows, dtype=int)

    for _ in range(repetitions):
        sample_indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        sample = features.iloc[sample_indices]
        sample_fit = fit_ward(sample, names, k)
        restricted_full = full_labels[sample_indices]
        aligned = _align_sample_labels(restricted_full, sample_fit.labels, k)
        appearances[sample_indices] += 1
        matches[sample_indices] += aligned == restricted_full

    probability = np.divide(
        matches,
        appearances,
        out=np.full(n_rows, np.nan, dtype=float),
        where=appearances > 0,
    )
    return pd.DataFrame(
        {
            "sku": features["sku"].astype(str).to_numpy(),
            "full_cluster": canonical,
            "subsample_appearances": appearances,
            "same_cluster_count": matches,
            "assignment_stability": probability,
        }
    ).sort_values(["assignment_stability", "sku"], ascending=[True, True])
