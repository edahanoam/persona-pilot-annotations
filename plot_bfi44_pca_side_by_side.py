"""Create a side-by-side PCA plot from BFI-44 evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_ORDER = ("full", "full_no_bfi", "d", "dm")
COLORS = {
    "human": "#111827",
    "full": "#2563eb",
    "full_no_bfi": "#9333ea",
    "d": "#dc2626",
    "dm": "#16a34a",
}
LABELS = {
    "human": "Human",
    "full": "Full profile",
    "full_no_bfi": "Full summary without BFI scores",
    "d": "Demographics",
    "dm": "Demographics + qualitative",
}


def make_side_by_side_plot(output_dir: Path) -> Path:
    coordinates = pd.read_csv(output_dir / "pca_coordinates.csv", dtype={"pid": str})
    variance = pd.read_csv(output_dir / "pca_explained_variance.csv")
    required = {"pid", "source", "PC1", "PC2"}
    missing = required - set(coordinates.columns)
    if missing:
        raise ValueError(f"pca_coordinates.csv is missing columns: {sorted(missing)}")

    sources = [source for source in DISPLAY_ORDER if source in set(coordinates["source"])]
    if not sources:
        raise ValueError("No supported model sources were found in pca_coordinates.csv")

    human = coordinates.loc[
        coordinates["source"] == "human", ["pid", "PC1", "PC2"]
    ].set_index("pid")
    if human.empty:
        raise ValueError("pca_coordinates.csv contains no human coordinates")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_xy = coordinates.loc[:, ["PC1", "PC2"]].to_numpy(dtype=float)
    x_padding = max(float(np.ptp(all_xy[:, 0])) * 0.05, 0.5)
    y_padding = max(float(np.ptp(all_xy[:, 1])) * 0.05, 0.5)
    x_limits = (all_xy[:, 0].min() - x_padding, all_xy[:, 0].max() + x_padding)
    y_limits = (all_xy[:, 1].min() - y_padding, all_xy[:, 1].max() + y_padding)

    figure, axes = plt.subplots(
        1, len(sources), figsize=(6 * len(sources), 6), sharex=True, sharey=True,
        squeeze=False,
    )
    for axis, source in zip(axes.ravel(), sources):
        model = coordinates.loc[
            coordinates["source"] == source, ["pid", "PC1", "PC2"]
        ].set_index("pid")
        common_pids = human.index.intersection(model.index, sort=False)
        human_xy = human.loc[common_pids].to_numpy(dtype=float)
        model_xy = model.loc[common_pids].to_numpy(dtype=float)

        for human_point, model_point in zip(human_xy, model_xy):
            axis.plot(
                [human_point[0], model_point[0]],
                [human_point[1], model_point[1]],
                color=COLORS[source], alpha=0.10, linewidth=0.7,
            )
        axis.scatter(
            human_xy[:, 0], human_xy[:, 1], s=34, alpha=0.82,
            color=COLORS["human"], label=LABELS["human"],
        )
        axis.scatter(
            model_xy[:, 0], model_xy[:, 1], s=25, alpha=0.82,
            color=COLORS[source], label=LABELS[source],
        )
        axis.set_title(f"Human vs {LABELS[source]}")
        axis.axhline(0, color="#9ca3af", linewidth=0.6)
        axis.axvline(0, color="#9ca3af", linewidth=0.6)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.legend(frameon=False)

    pc1 = float(variance.iloc[0]["explained_variance_ratio"]) * 100
    pc2 = float(variance.iloc[1]["explained_variance_ratio"]) * 100
    figure.supxlabel(f"Human PCA component 1 ({pc1:.1f}% variance)")
    figure.supylabel(f"Human PCA component 2 ({pc2:.1f}% variance)")
    figure.suptitle("BFI-44 model answers projected into human PCA space")
    figure.tight_layout()

    destination = output_dir / "pca_projection_by_condition.png"
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot each BFI-44 model condition beside the others."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluations/bfi44")
    )
    args = parser.parse_args()
    destination = make_side_by_side_plot(args.output_dir)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
