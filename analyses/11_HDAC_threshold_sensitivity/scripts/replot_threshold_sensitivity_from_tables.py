#!/usr/bin/env python3
"""Replot Supplementary Fig. 12a from the deposited threshold-effect table."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


RMST_TAU_MIN = 30.0
SENS_COMPARISONS = ("TSA", "RGFP966")
THRESHOLD_SWEEPS = {
    "mtetr": {
        "values": [0.85, 1.00, 1.10, 1.30, 1.50],
        "paper_value": 1.10,
        "snapshot_value": 1.30,
    },
    "mcp": {
        "values": [0.85, 1.00, 1.15, 1.25, 1.50],
        "paper_value": 1.25,
        "snapshot_value": 1.15,
    },
}


def resolve_module_root(data_root: Path) -> Path:
    for candidate in (
        data_root / "revision" / "11_HDAC_threshold_sensitivity",
        data_root / "11_HDAC_threshold_sensitivity",
        data_root,
    ):
        if (candidate / "threshold_sweeps" / "threshold_sensitivity_effects.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find threshold-sensitivity tables below {data_root}")


def plot_threshold_sensitivity_summary_revised(
    effects: pd.DataFrame,
    panel_label_fontsize: float = 11,
    font_size: float = 8,
) -> plt.Figure:
    colors = {"TSA - DMSO": "#E69F00", "RGFP966 - DMSO": "#009E73"}
    metric_specs = [
        ("active_fraction", "Delta Active fraction"),
        ("active_rmst", f"Delta Active RMST (min; tau={RMST_TAU_MIN:g})"),
        ("inactive_rmst", f"Delta Inactive RMST (min; tau={RMST_TAU_MIN:g})"),
    ]
    row_specs = [
        ("mtetr", "Nanog"),
        ("mtetr", "Sox2"),
        ("mcp", "Nanog"),
        ("mcp", "Sox2"),
    ]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.4, 8.2))
    grid = figure.add_gridspec(
        nrows=5,
        ncols=3,
        height_ratios=[1.0, 1.0, 0.18, 1.0, 1.0],
        left=0.11,
        right=0.98,
        bottom=0.13,
        top=0.89,
        hspace=0.46,
        wspace=0.32,
    )
    axes = np.empty((4, 3), dtype=object)
    for column_index in range(3):
        axes[0, column_index] = figure.add_subplot(grid[0, column_index])
        axes[1, column_index] = figure.add_subplot(
            grid[1, column_index], sharex=axes[0, column_index], sharey=axes[0, column_index]
        )
        axes[2, column_index] = figure.add_subplot(
            grid[3, column_index], sharey=axes[0, column_index]
        )
        axes[3, column_index] = figure.add_subplot(
            grid[4, column_index], sharex=axes[2, column_index], sharey=axes[0, column_index]
        )

    for row_index, (sweep_type, gene) in enumerate(row_specs):
        config = THRESHOLD_SWEEPS[sweep_type]
        threshold_values = np.asarray(config["values"], dtype=float)
        threshold_name = "mTetR" if sweep_type == "mtetr" else "MCP"
        for column_index, (metric, column_title) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            axis.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
            axis.axvline(config["paper_value"], color="#D62728", linewidth=1.0, zorder=2)
            axis.axvline(
                config["snapshot_value"], color="#7F7F7F", linewidth=1.0, linestyle=":", zorder=2
            )
            for comparison in SENS_COMPARISONS:
                comparison_label = f"{comparison} - DMSO"
                subset = effects.loc[
                    effects["sweep_type"].eq(sweep_type)
                    & effects["gene"].eq(gene)
                    & effects["metric"].eq(metric)
                    & effects["comparison"].eq(comparison_label)
                ].sort_values("threshold")
                if subset.empty:
                    continue
                x = subset["threshold"].to_numpy(float)
                estimate = subset["estimate"].to_numpy(float)
                low = subset["ci_low"].to_numpy(float)
                high = subset["ci_high"].to_numpy(float)
                axis.fill_between(
                    x, low, high, color=colors[comparison_label], alpha=0.18, linewidth=0, zorder=3
                )
                axis.plot(
                    x,
                    estimate,
                    color=colors[comparison_label],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.25,
                    zorder=4,
                )
            axis.set_xticks(threshold_values)
            axis.set_xticklabels([f"{value:g}" for value in threshold_values])
            axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.margins(x=0.05)
            if row_index in (0, 2):
                axis.set_title(column_title, pad=5)
                axis.tick_params(axis="x", labelbottom=False)
            if row_index in (1, 3):
                axis.set_xlabel(f"{threshold_name} normalized threshold")
            if column_index == 0:
                axis.set_ylabel(rf"$\mathit{{{gene}}}$" "\nDrug - DMSO", labelpad=6)

    top_block_y = axes[0, 0].get_position().y1 + 0.045
    lower_block_y = axes[2, 0].get_position().y1 + 0.045
    figure.text(0.02, top_block_y, "a", fontsize=panel_label_fontsize, fontweight="bold")
    figure.text(0.5, top_block_y, "mTetR threshold sensitivity", ha="center", fontsize=font_size + 0.5)
    figure.text(0.02, lower_block_y, "b", fontsize=panel_label_fontsize, fontweight="bold")
    figure.text(0.5, lower_block_y, "MCP threshold sensitivity", ha="center", fontsize=font_size + 0.5)

    legend_handles = [
        Line2D([0], [0], color="#D62728", linewidth=1.0),
        Line2D([0], [0], color="#7F7F7F", linewidth=1.0, linestyle=":"),
        Line2D([0], [0], color=colors["TSA - DMSO"], marker="o", linewidth=1.25),
        Line2D([0], [0], color=colors["RGFP966 - DMSO"], marker="o", linewidth=1.25),
    ]
    figure.legend(
        legend_handles,
        ["Main-analysis threshold", "Snapshot threshold", "TSA - DMSO", "RGFP966 - DMSO"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.5,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GAO2026_DATA_ROOT", "data")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("GAO2026_OUTPUT_ROOT", "outputs")),
    )
    args = parser.parse_args()
    module_root = resolve_module_root(args.data_root.resolve())
    output_root = args.output_root.resolve() / "11_HDAC_threshold_sensitivity"
    output_root.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(module_root / "threshold_sweeps" / "threshold_sensitivity_effects.csv")

    expected_rows = 2 * 5 * 2 * 2 * 3
    if len(effects) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} threshold-effect rows, found {len(effects)}")
    combinations = effects.groupby(["sweep_type", "gene", "comparison", "metric"]).size()
    if not combinations.eq(5).all():
        raise AssertionError("Each threshold-effect series must contain exactly five thresholds")

    figure = plot_threshold_sensitivity_summary_revised(effects)
    output_pdf = output_root / "R3_m11_threshold_sensitivity_summary_revised.pdf"
    output_png = output_root / "R3_m11_threshold_sensitivity_summary_revised.png"
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    effects.assign(table_validation="PASS").to_csv(
        output_root / "threshold_sensitivity_table_validation.csv", index=False
    )
    print(output_pdf)


if __name__ == "__main__":
    main()
