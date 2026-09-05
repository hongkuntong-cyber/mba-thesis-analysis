from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZON_TITLES = {
    "30_day_proxy": "28-day proxy (4 weeks)",
    "60_day_proxy": "63-day proxy (9 weeks)",
    "90_day_proxy": "91-day proxy (13 weeks)",
}


def plot_pxq_reliability(output_root: str | Path) -> Path:
    """Plot fixed-bin reliability for the V4.1 primary probability method."""
    root = Path(output_root)
    source = pd.read_csv(root / "reliability_bins.csv")
    source = source.loc[source["method"].eq("PXQ_independence")].copy()
    if source.empty:
        raise RuntimeError("No PXQ_independence reliability rows were found")

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharex=True, sharey=True)
    blue = "#2F6B9A"
    dark = "#2B3137"
    grid = "#D9DEE3"
    for axis, (label, frame) in zip(
        axes, source.groupby("horizon_label", sort=False), strict=True
    ):
        frame = frame.sort_values("mean_probability")
        axis.plot([0, 1], [0, 1], color=dark, linestyle="--", linewidth=1.2, label="Ideal")
        sizes = 40 + 0.35 * frame["n_sku_origins"].to_numpy(dtype=float)
        axis.plot(
            frame["mean_probability"],
            frame["observed_event_rate"],
            color=blue,
            linewidth=2.0,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=blue,
            markeredgewidth=1.8,
            zorder=3,
        )
        axis.scatter(
            frame["mean_probability"],
            frame["observed_event_rate"],
            s=sizes,
            facecolors="white",
            edgecolors=blue,
            linewidths=1.6,
            zorder=4,
        )
        for row in frame.itertuples(index=False):
            axis.annotate(
                f"n={int(row.n_sku_origins)}",
                (row.mean_probability, row.observed_event_rate),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=8,
                color=dark,
            )
        axis.set_title(HORIZON_TITLES[str(label)], fontsize=11, color=dark, pad=9)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(True, color=grid, linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(dark)
        axis.set_xlabel("Mean predicted probability", fontsize=9, color=dark)
    axes[0].set_ylabel("Observed demand-occurrence rate", fontsize=9, color=dark)
    figure.suptitle(
        "PXQ independent-week probability reliability",
        fontsize=15,
        color=dark,
        x=0.05,
        ha="left",
        y=1.02,
    )
    figure.text(
        0.05,
        0.94,
        "Five frozen probability bins; marker labels show common SKU-origin counts. Dashed line is perfect calibration.",
        fontsize=9,
        color="#59636E",
        ha="left",
    )
    figure.tight_layout(rect=[0.03, 0.02, 1.0, 0.90])
    destination = root / "figures" / "pxq_independence_reliability.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render V4.1 probability calibration figures.")
    parser.add_argument("--output-root", default="outputs/pxq_probability_v4_1")
    args = parser.parse_args()
    print(plot_pxq_reliability(args.output_root))


if __name__ == "__main__":
    main()
