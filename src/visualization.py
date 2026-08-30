from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from .stability import transform_features


BLUE = "#2F6B9A"
GOLD = "#D6A53A"
ORANGE = "#D97836"
OLIVE = "#7A8B49"
PINK = "#B95F78"
INK = "#25313C"
GRID = "#D9DEE3"


def _style(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A949E")
    ax.grid(axis=axis, color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.125, y=0.98, ha="left", color=INK, fontsize=14, weight="bold")
    fig.text(0.125, 0.93, subtitle, ha="left", color="#58636E", fontsize=10)
    fig.subplots_adjust(top=0.84)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_clustering_charts(output_root: str | Path) -> None:
    root = Path(output_root) / "clustering"
    figures = root / "figures"
    grid = pd.read_csv(root / "feature_k_grid_500.csv")
    selection = json.loads(
        (root / "operational_selection_full_period.json").read_text(encoding="utf-8")
    )
    key = "+".join(selection["feature_names"])
    selected_rows = grid.loc[grid["feature_set"].eq(key)].set_index("k")
    eligible = grid.loc[grid["confirmatory_eligible"]]
    palette = {2: BLUE, 3: GOLD, 4: ORANGE, 5: OLIVE, 6: PINK}

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for current_k, frame in eligible.groupby("k"):
        ax.scatter(
            frame["silhouette"],
            frame["stability_ari_median"],
            s=46,
            alpha=0.72,
            color=palette[int(current_k)],
            edgecolor="white",
            linewidth=0.5,
            label=f"K={int(current_k)}",
        )
    selected = selected_rows.loc[int(selection["k"])]
    ax.scatter(
        [selected["silhouette"]],
        [selected["stability_ari_median"]],
        marker="*",
        s=260,
        color=INK,
        edgecolor="white",
        linewidth=0.8,
        label="Frozen operational candidate",
        zorder=5,
    )
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Median stability ARI")
    _style(ax)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    _header(
        fig,
        "Confirmatory feature–K candidates",
        "V2 main sample N=197; 500 × 80% sampling without replacement",
    )
    _save(fig, figures / "confirmatory_silhouette_stability.png")

    comparison = selected_rows.loc[[2, 3, 4, 5, 6]]
    x, width = np.arange(5), 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    ax.bar(x - width / 2, comparison["silhouette"], width, color=BLUE, label="Silhouette")
    ax.bar(
        x + width / 2,
        comparison["stability_ari_median"],
        width,
        color=GOLD,
        label="Median stability ARI",
    )
    ax.set_xticks(x, [f"K={value}" for value in [2, 3, 4, 5, 6]])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    _style(ax)
    ax.legend(frameon=False, loc="upper right")
    _header(
        fig,
        "K=2–6 under the frozen four-feature set",
        "ADI + CV² + nonzero_mean + acf1; V2 main sample N=197",
    )
    _save(fig, figures / "selected_feature_k_comparison.png")

    features = pd.read_csv(root / "features_v2_main_sample.csv")
    transformed, _ = transform_features(features, selection["feature_names"])
    hierarchy = linkage(transformed, method="ward")
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
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
        "Ward hierarchy for the frozen four-feature set",
        "Yeo–Johnson transformation and standardization; V2 main sample N=197",
    )
    _save(fig, figures / "ward_dendrogram.png")

    assignment = pd.read_csv(root / "sku_assignment_stability_k2.csv")
    clusters = sorted(assignment["full_cluster"].unique())
    arrays = [
        assignment.loc[
            assignment["full_cluster"].eq(cluster), "assignment_stability"
        ].to_numpy()
        for cluster in clusters
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    boxes = ax.boxplot(arrays, patch_artist=True, showfliers=True, widths=0.55)
    for box, color in zip(boxes["boxes"], [BLUE, GOLD]):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    for median in boxes["medians"]:
        median.set_color(INK)
        median.set_linewidth(1.5)
    ax.axhline(0.75, color=ORANGE, linestyle="--", linewidth=1.2, label="0.75 reference")
    ax.set_xticks(range(1, len(clusters) + 1), [f"Cluster {int(value)}" for value in clusters])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Assignment stability")
    _style(ax)
    ax.legend(frameon=False, loc="lower right")
    _header(
        fig,
        "SKU assignment stability for final K=2",
        "Share of sampled runs returning each SKU to its full-sample cluster",
    )
    _save(fig, figures / "sku_assignment_stability.png")


def render_forecast_charts(output_root: str | Path, config: dict[str, Any]) -> None:
    root = Path(output_root) / "forecast"
    figures = root / "figures"
    predictions = pd.read_csv(root / "rolling_origin_predictions.csv")
    holdout = predictions.loc[predictions["origin_index"].eq(config["forecast"]["origins"])]
    holdout_summary = _summarize(holdout, ["model"]).sort_values("aggregate_wape")
    colors = {"MA4_proxy": BLUE, "ADIDA2": GOLD, "SES": ORANGE, "Naive": OLIVE, "Zero": PINK}

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    bars = ax.barh(
        holdout_summary["model"],
        holdout_summary["aggregate_wape"],
        color=[colors[value] for value in holdout_summary["model"]],
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    ax.set_xlim(0, max(1.05, float(holdout_summary["aggregate_wape"].max()) * 1.12))
    ax.set_xlabel("Volume-weighted WAPE")
    _style(ax, "x")
    _header(fig, "Holdout model comparison", "Origin 6; 147 SKUs; lower is better")
    _save(fig, figures / "holdout_model_wape.png")

    routes = pd.read_csv(root / "frozen_routes_before_holdout.csv")
    holdout_skus = set(holdout["sku"].unique())
    counts = (
        routes.loc[routes["sku"].isin(holdout_skus), "management_path"]
        .value_counts()
        .reindex(["预测管理", "规则管理", "人工复核"])
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bars = ax.barh(
        ["Forecast management", "Rule management", "Manual review"],
        counts.to_numpy(),
        color=[BLUE, GOLD, ORANGE],
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.0f", padding=4, color=INK)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.set_xlabel("SKUs")
    _style(ax, "x")
    _header(
        fig,
        "Frozen management paths in the holdout",
        "147 common evaluable SKUs; routes fixed before origin 6",
    )
    _save(fig, figures / "holdout_management_paths.png")

    origin_summary = pd.read_csv(root / "model_summary_by_origin.csv")
    core = origin_summary.loc[origin_summary["model"].isin(config["forecast"]["models"])]
    model_order = config["forecast"]["models"]
    x, width = np.arange(1, 7), 0.19
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    for offset, model in enumerate(model_order):
        values = core.loc[core["model"].eq(model)].sort_values("origin_index")["aggregate_wape"]
        ax.bar(
            x + (offset - 1.5) * width,
            values,
            width,
            label=model,
            color=colors[model],
        )
    ax.set_xticks(x, [f"O{value}" for value in x])
    ax.set_ylabel("Volume-weighted WAPE")
    ax.set_xlabel("Rolling origin")
    _style(ax)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    _header(
        fig,
        "Core-model performance by rolling origin",
        "Six non-overlapping 8-week windows; lower is better",
    )
    _save(fig, figures / "rolling_origin_core_wape.png")

    sensitivity = pd.read_csv(root / "adida_aggregation_sensitivity_common_summary.csv").sort_values(
        "aggregation_weeks"
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(
        sensitivity["model"],
        sensitivity["aggregate_wape"],
        color=[BLUE, GOLD, ORANGE, OLIVE],
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK)
    ax.set_ylim(0, max(1.08, float(sensitivity["aggregate_wape"].max()) * 1.10))
    ax.set_ylabel("Volume-weighted WAPE")
    _style(ax)
    _header(
        fig,
        "ADIDA aggregation sensitivity",
        "Common sample: 790 SKU-origins; 2 weeks is the formal model",
    )
    _save(fig, figures / "adida_aggregation_sensitivity.png")

    contribution = pd.read_csv(root / "holdout_path_contribution.csv")
    contribution["label"] = contribution["management_path"].map(
        {"预测管理": "Forecast", "规则管理": "Rule", "人工复核": "Manual review"}
    ) + " / " + contribution["routed_model"]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    plot_colors = [
        BLUE if value >= 0 else GOLD for value in contribution["net_abs_error_improvement"]
    ]
    bars = ax.barh(
        contribution["label"],
        contribution["net_abs_error_improvement"],
        color=plot_colors,
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%+.1f", padding=4, color=INK)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_xlabel("Reduction in absolute error units vs enterprise MA4")
    _style(ax, "x")
    _header(
        fig,
        "Holdout contribution by frozen route",
        "Positive values improve on enterprise MA4; total net gain = 5 units",
    )
    _save(fig, figures / "holdout_path_contribution.png")


def _summarize(predictions: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    # Local import avoids a visualization/evaluation import cycle in lightweight use.
    from .evaluation import summarize_predictions

    return summarize_predictions(predictions, groups)
