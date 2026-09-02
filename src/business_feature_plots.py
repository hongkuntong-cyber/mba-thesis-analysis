from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from .stability import transform_features
from .visualization import BLUE, GOLD, GRID, INK, OLIVE, ORANGE, _header, _save, _style


def _short_label(feature_set: str) -> str:
    if feature_set == "ADI+CV2":
        return "ADI + CV²"
    suffix = feature_set.removeprefix("ADI+CV2+")
    return "ADI + CV² + " + suffix.replace("+", " + ")


def render_business_feature_charts(
    output_root: str | Path,
    *,
    recommended_feature_set: str = "ADI+CV2+acf1+nonzero_mean",
) -> dict[str, str]:
    root = Path(output_root) / "clustering"
    figures = root / "figures"
    grid = pd.read_csv(root / "feature_grid_k2_500.csv")
    if recommended_feature_set not in set(grid.loc[grid["pareto"], "feature_set"]):
        raise ValueError("The plotted recommendation must be on the formal Pareto frontier")

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ineligible = grid.loc[~grid["structural_eligible"]]
    eligible = grid.loc[grid["structural_eligible"] & ~grid["pareto"]]
    pareto = grid.loc[grid["pareto"]]
    ax.scatter(
        ineligible["silhouette"],
        ineligible["stability_ari_median"],
        s=34,
        color="#C5CBD1",
        alpha=0.70,
        label="Fails structural gate",
    )
    ax.scatter(
        eligible["silhouette"],
        eligible["stability_ari_median"],
        s=42,
        color=BLUE,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.4,
        label="Passes structural gate",
    )
    colors = {"ADI+CV2+acf1+nonzero_mean": GOLD, "ADI+CV2+peak_ratio": ORANGE}
    for _, row in pareto.iterrows():
        key = str(row["feature_set"])
        ax.scatter(
            [row["silhouette"]],
            [row["stability_ari_median"]],
            s=105,
            color=colors.get(key, GOLD),
            edgecolor=INK,
            linewidth=0.7,
            zorder=5,
        )
        label = "4D: acf1 + event size" if "acf1" in key else "3D: peak ratio"
        offset = (-7, 11) if "acf1" in key else (-7, -20)
        ax.annotate(
            label,
            (row["silhouette"], row["stability_ari_median"]),
            xytext=offset,
            textcoords="offset points",
            ha="right",
            color=INK,
            fontsize=9,
        )
    benchmark = grid.loc[grid["feature_set"].eq("ADI+CV2")].iloc[0]
    ax.scatter(
        [benchmark["silhouette"]],
        [benchmark["stability_ari_median"]],
        marker="D",
        s=58,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        zorder=4,
        label="ADI + CV² benchmark",
    )
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Median 80% subsample stability ARI")
    _style(ax)
    ax.legend(frameon=False, loc="lower left")
    _header(
        fig,
        "Frozen K=2 feature combinations",
        "64 combinations; V2 main sample N=197; 500 paired 80% samples without replacement",
    )
    _save(fig, figures / "v3_feature_pareto_frontier.png")

    k_grid = pd.read_csv(root / "post_selection_k2_to_k6.csv")
    selected_k = k_grid.loc[k_grid["feature_set"].eq(recommended_feature_set)].sort_values("k")
    x = np.arange(len(selected_k))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.bar(
        x - width / 2,
        selected_k["silhouette"],
        width,
        color=BLUE,
        label="Silhouette",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        selected_k["stability_ari_median"],
        width,
        color=GOLD,
        label="Median stability ARI",
        edgecolor="white",
    )
    ax.set_xticks(x, [f"K={int(value)}" for value in selected_k["k"]])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    _style(ax)
    ax.legend(frameon=False, loc="upper right")
    _header(
        fig,
        "K=2–6 structural sensitivity",
        f"{_short_label(recommended_feature_set)}; only K=2 passes every frozen structural gate",
    )
    _save(fig, figures / "v3_recommended_k_sensitivity.png")

    increments = pd.read_csv(root / "feature_increment_diagnostics.csv").sort_values(
        "median_delta_stability_ari"
    )
    labels = {
        "acf1": "Lag-1 autocorrelation",
        "nonzero_mean": "Positive-event mean",
        "peak_ratio": "Peak ratio",
        "promo_response_index": "Promotion response",
        "trailing_zero_share": "Trailing-zero share",
        "trend_coef": "Trend coefficient",
    }
    y = np.arange(len(increments))
    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    ax.barh(
        y - 0.18,
        increments["median_delta_silhouette"],
        0.34,
        color=BLUE,
        label="Median Δ Silhouette",
        edgecolor="white",
    )
    ax.barh(
        y + 0.18,
        increments["median_delta_stability_ari"],
        0.34,
        color=GOLD,
        label="Median Δ stability ARI",
        edgecolor="white",
    )
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y, [labels[value] for value in increments["feature"]])
    ax.set_xlabel("Paired change after adding the feature")
    _style(ax, "x")
    ax.legend(frameon=False, loc="lower right")
    _header(
        fig,
        "Paired incremental feature diagnostics",
        "32 matched additions per feature; descriptive only, not a replacement selection score",
    )
    _save(fig, figures / "v3_incremental_feature_effects.png")

    assignments = pd.read_csv(root / "pareto_candidate_assignment_stability.csv")
    selected_assignments = assignments.loc[
        assignments["feature_set"].eq(recommended_feature_set)
    ]
    clusters = sorted(selected_assignments["full_cluster"].unique())
    arrays = [
        selected_assignments.loc[
            selected_assignments["full_cluster"].eq(cluster), "assignment_stability"
        ].to_numpy()
        for cluster in clusters
    ]
    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    boxes = ax.boxplot(arrays, patch_artist=True, widths=0.55, showfliers=True)
    for box, color in zip(boxes["boxes"], [BLUE, GOLD]):
        box.set_facecolor(color)
        box.set_alpha(0.78)
    for median in boxes["medians"]:
        median.set_color(INK)
        median.set_linewidth(1.5)
    ax.axhline(0.75, color=ORANGE, linestyle="--", linewidth=1.2, label="0.75 reference")
    ax.set_xticks(range(1, len(clusters) + 1), [f"Cluster {int(value)}" for value in clusters])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("SKU assignment stability")
    _style(ax)
    ax.legend(frameon=False, loc="lower right")
    _header(
        fig,
        "SKU assignment stability under the four-feature K=2 solution",
        "Share of appearances returning each SKU to its full-sample cluster; 500 paired subsamples",
    )
    _save(fig, figures / "v3_recommended_assignment_stability.png")

    features = pd.read_csv(root / "features_v2_main_sample.csv")
    assignment_export = selected_assignments.merge(
        features,
        on="sku",
        how="left",
        validate="one_to_one",
    )
    assignment_export.insert(0, "analysis_mode", "retrospective_method_development")
    assignment_export.insert(
        1, "selection_status", "provisional_downstream_recommendation"
    )
    assignment_export.to_csv(
        root / "recommended_main_cluster_assignments.csv", index=False
    )
    transformed, _ = transform_features(features, recommended_feature_set.split("+"))
    hierarchy = linkage(transformed, method="ward")
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    dendrogram(
        hierarchy,
        no_labels=True,
        color_threshold=0,
        above_threshold_color=BLUE,
        ax=ax,
    )
    ax.set_xlabel("SKU leaves")
    ax.set_ylabel("Ward merge distance")
    _style(ax)
    _header(
        fig,
        "Ward hierarchy under the four-feature solution",
        "Yeo–Johnson transformation and standardization; V2 main sample N=197",
    )
    _save(fig, figures / "v3_recommended_ward_dendrogram.png")

    return {
        "recommended_feature_set": recommended_feature_set,
        "figures": str(figures),
        "n_figures": 5,
        "assignment_table": str(root / "recommended_main_cluster_assignments.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render V3.0 business feature figures")
    parser.add_argument("--output-root", default="outputs/business_features_v3")
    parser.add_argument(
        "--recommended-feature-set",
        default="ADI+CV2+acf1+nonzero_mean",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            render_business_feature_charts(
                args.output_root,
                recommended_feature_set=args.recommended_feature_set,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
