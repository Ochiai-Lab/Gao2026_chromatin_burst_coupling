#!/usr/bin/env python3
"""Replot the six Supplementary Fig. 13 panel groups from final tables."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_MIN = (0.0, 15.0, 30.0, 45.0, 60.0)
GROUP_ORDER = ("DMSO", "TSA", "RGFP966", "THZ1")
COLORS = {"DMSO": "#377EB8", "TSA": "#FF7F00", "RGFP966": "#4DAF4A", "THZ1": "#984EA3"}
PANEL_SPECS = (
    ("a", "Nanog", "H3K27ac", "HDACi"),
    ("b", "Sox2", "H3K27ac", "HDACi"),
    ("c", "Nanog", "H3K27ac", "THZ1"),
    ("d", "Nanog", "p300", "THZ1"),
    ("e", "Nanog", "HDAC1", "THZ1"),
    ("f", "Nanog", "HDAC3", "THZ1"),
)


def resolve_module_root(data_root: Path) -> Path:
    for candidate in (
        data_root / "revision" / "14_acute_inhibitor",
        data_root / "14_acute_inhibitor",
        data_root,
    ):
        if (candidate / "fig_s13_signal_summary.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Supplementary Fig. 13 tables below {data_root}")


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3, width=0.8)


def p_symbol(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def signal_limits(summary: pd.DataFrame, comparison: str, metric: str) -> tuple[float, float]:
    subset = summary.loc[summary["metric"].eq(metric)]
    low = min(-0.25, subset["q25_log2_fc"].min())
    high = max(0.25, subset["q75_log2_fc"].max())
    span = max(1.0, high - low)
    annotation_rows = 2 if comparison == "HDACi" else 1
    low = math.floor((low - 0.08 * span) * 2) / 2
    high = math.ceil((high + (0.16 + 0.10 * annotation_rows) * span) * 2) / 2
    return low, high


def annotate_vs_dmso(axis: plt.Axes, tests: pd.DataFrame, treatments: list[str]) -> None:
    low, high = axis.get_ylim()
    span = high - low
    for row_index, treatment in enumerate(treatments):
        y_position = high - (0.10 + 0.09 * row_index) * span
        for time_min in TIME_MIN[1:]:
            match = tests.loc[
                tests["time_min"].eq(time_min)
                & tests["group_1"].eq("DMSO")
                & tests["group_2"].eq(treatment)
            ]
            if not match.empty and match["holm_p"].iloc[0] < 0.05:
                axis.text(
                    time_min,
                    y_position,
                    p_symbol(float(match["holm_p"].iloc[0])),
                    color=COLORS[treatment],
                    ha="center",
                    va="center",
                    fontsize=5.5,
                )


def plot_active_panel(axis: plt.Axes, subset: pd.DataFrame, groups: list[str], gene: str) -> None:
    for group in groups:
        values = subset.loc[subset["group"].eq(group)].sort_values("time_min")
        x = values["time_min"].to_numpy(float)
        y = values["active_fraction"].to_numpy(float)
        axis.fill_between(
            x,
            values["ci_low"].to_numpy(float),
            values["ci_high"].to_numpy(float),
            color=COLORS[group],
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(x, y, color=COLORS[group], marker="o", label=group)
    axis.set(
        xticks=TIME_MIN,
        ylim=(-0.04, 1.04),
        xlabel="Time (min)",
        ylabel="Active fraction",
        title=rf"$\it{{{gene}}}$ transcription",
    )
    style_axis(axis)


def plot_signal_panel(
    axis: plt.Axes,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    groups: list[str],
    gene: str,
    target: str,
    comparison: str,
    metric: str,
    location: str,
) -> None:
    metric_summary = summary.loc[summary["metric"].eq(metric)]
    metric_tests = tests.loc[tests["metric"].eq(metric)]
    for group in groups:
        values = metric_summary.loc[metric_summary["group"].eq(group)].sort_values("time_min")
        x = values["time_min"].to_numpy(float)
        y = values["median_log2_fc"].to_numpy(float)
        axis.fill_between(
            x,
            values["q25_log2_fc"].to_numpy(float),
            values["q75_log2_fc"].to_numpy(float),
            color=COLORS[group],
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(x, y, color=COLORS[group], marker="o", label=group)
    axis.axhline(0, color="0.70", linewidth=0.6, zorder=0)
    axis.set(
        xticks=TIME_MIN,
        ylim=signal_limits(summary, comparison, metric),
        xlabel="Time (min)",
        ylabel=r"Relative SNAP intensity (log$_2$[$I_t$/$I_0$])",
        title=rf"$\it{{{gene}}}$ {location} {target}",
    )
    style_axis(axis)
    annotate_vs_dmso(axis, metric_tests, [group for group in groups if group != "DMSO"])


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
    output_root = args.output_root.resolve() / "14_acute_inhibitor"
    output_root.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(module_root / "fig_s13_signal_summary.csv")
    tests = pd.read_csv(module_root / "fig_s13_signal_tests.csv")
    active = pd.read_csv(module_root / "fig_s13_active_fraction.csv")
    manifest = pd.read_csv(module_root / "fig_s13_experiment_manifest.csv")

    required_metrics = set(summary["metric"])
    if required_metrics != {"local_bg", "nuclear_bg"}:
        raise AssertionError(
            "The current manuscript requires both nuclear_bg and local_bg tables; "
            f"found {sorted(required_metrics)}"
        )
    expected_panels = {spec[0] for spec in PANEL_SPECS}
    if set(summary["panel"]) != expected_panels:
        raise AssertionError(f"Expected panels a-f, found {sorted(set(summary['panel']))}")
    experiment_counts = dict(zip(manifest["panel"], manifest["n_experiments"]))
    if experiment_counts != {"a": 2, "b": 3, "c": 2, "d": 2, "e": 3, "f": 3}:
        raise AssertionError(f"Experiment counts disagree with the legend: {experiment_counts}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for panel, gene, target, comparison in PANEL_SPECS:
        panel_summary = summary.loc[summary["panel"].eq(panel)].copy()
        panel_tests = tests.loc[tests["panel"].eq(panel)].copy()
        groups = [group for group in GROUP_ORDER if group in set(panel_summary["group"])]
        if panel in ("a", "b"):
            figure, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 60 / 25.4))
            plot_signal_panel(
                axes[0], panel_summary, panel_tests, groups, gene, target, comparison, "nuclear_bg", "nuclear"
            )
            plot_signal_panel(
                axes[1], panel_summary, panel_tests, groups, gene, target, comparison, "local_bg", "local"
            )
        else:
            panel_active = active.loc[active["panel"].eq(panel)].copy()
            figure, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 60 / 25.4))
            plot_active_panel(axes[0], panel_active, groups, gene)
            plot_signal_panel(
                axes[1], panel_summary, panel_tests, groups, gene, target, comparison, "nuclear_bg", "nuclear"
            )
            plot_signal_panel(
                axes[2], panel_summary, panel_tests, groups, gene, target, comparison, "local_bg", "local"
            )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(groups),
            handlelength=1.6,
            columnspacing=1.0,
        )
        figure.subplots_adjust(
            left=0.07 if panel not in ("a", "b") else 0.105,
            right=0.985,
            bottom=0.26,
            top=0.78,
            wspace=0.52 if panel not in ("a", "b") else 0.42,
        )
        output_pdf = output_root / f"Supplementary_Fig_13{panel}_{gene}_{target}_{comparison}.pdf"
        figure.savefig(output_pdf)
        plt.close(figure)
        print(output_pdf)

    manifest.assign(table_validation="PASS").to_csv(
        output_root / "Supplementary_Fig_13_table_validation.csv", index=False
    )


if __name__ == "__main__":
    main()
