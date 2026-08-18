#!/usr/bin/env python3
"""Replot Supplementary Fig. 12b-c from deposited cell-level tables.

The plotting implementation is copied from the final time-window notebook. The
Mann-Whitney tests and within-window Holm correction are recalculated to verify
that the deposited cell-level values reproduce the deposited statistics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


WINDOWS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0))
GROUPS = ("DMSO", "TSA", "RGFP966")
COMPARISONS = ("TSA", "RGFP966")
GROUP_COLOR = {"DMSO": "C0", "TSA": "C1", "RGFP966": "C2"}
PLOT_FONT_SIZE = 10


def resolve_module_root(data_root: Path) -> Path:
    for candidate in (
        data_root / "revision" / "10_HDAC_time_windows",
        data_root / "10_HDAC_time_windows",
        data_root,
    ):
        if (candidate / "Fig5_duty_cycle_by_window_stats.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find HDAC time-window tables below {data_root}")


def holm_adjust_array(p_values: list[float]) -> np.ndarray:
    p_values_array = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values_array)
    adjusted = np.empty(len(p_values_array), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted_value = (len(p_values_array) - rank) * p_values_array[index]
        running_max = max(running_max, adjusted_value)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def p_to_stars(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def add_sig_bracket(
    axis: plt.Axes, x1: float, x2: float, y: float, text: str, h: float = 0.02
) -> None:
    axis.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.8, clip_on=False)
    axis.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=PLOT_FONT_SIZE)


def recalculate_statistics(cell_values: pd.DataFrame, gene: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window_index, (low, high) in enumerate(WINDOWS):
        label = f"{int(low)}-{int(high)} min"
        baseline = cell_values.loc[
            cell_values["win_idx"].eq(window_index) & cell_values["group"].eq("DMSO"),
            "active_fraction",
        ].dropna().to_numpy()
        raw_p_values: list[float] = []
        for group in COMPARISONS:
            treatment = cell_values.loc[
                cell_values["win_idx"].eq(window_index) & cell_values["group"].eq(group),
                "active_fraction",
            ].dropna().to_numpy()
            raw_p_values.append(
                float(mannwhitneyu(baseline, treatment, alternative="two-sided").pvalue)
            )
        adjusted = holm_adjust_array(raw_p_values)
        for group, raw_p, adjusted_p in zip(COMPARISONS, raw_p_values, adjusted):
            rows.append(
                {
                    "gene": gene,
                    "win_idx": window_index,
                    "window": label,
                    "compare": f"DMSO vs {group}",
                    "p_raw": raw_p,
                    "p_holm": float(adjusted_p),
                }
            )
    return pd.DataFrame(rows)


def plot_time_windows(cell_values: pd.DataFrame, statistics: pd.DataFrame, gene: str) -> plt.Figure:
    slot = len(GROUPS) + 1
    figure, axis = plt.subplots(figsize=(1.15 * len(WINDOWS) * len(GROUPS), 2.6))
    jitter = np.random.default_rng(0)
    positions: dict[tuple[int, str], float] = {}

    for window_index, _ in enumerate(WINDOWS):
        for group_index, group in enumerate(GROUPS):
            position = window_index * slot + group_index
            positions[(window_index, group)] = position
            values = cell_values.loc[
                cell_values["win_idx"].eq(window_index) & cell_values["group"].eq(group),
                "active_fraction",
            ].dropna().to_numpy()
            axis.boxplot([values], positions=[position], widths=0.6, showfliers=False)
            axis.plot(
                jitter.normal(position, 0.06, size=len(values)),
                values,
                "o",
                ms=3,
                alpha=0.6,
                color=GROUP_COLOR[group],
            )

    y_top = np.nanmax(cell_values["active_fraction"].to_numpy(float))
    bracket_start = (y_top if np.isfinite(y_top) else 1.0) + 0.03
    for window_index in range(len(WINDOWS)):
        subset = statistics.loc[statistics["win_idx"].eq(window_index)]
        for comparison_index, row in enumerate(subset.itertuples(index=False)):
            treatment = row.compare.split(" vs ")[1]
            add_sig_bracket(
                axis,
                positions[(window_index, "DMSO")],
                positions[(window_index, treatment)],
                bracket_start + comparison_index * 0.12,
                p_to_stars(row.p_holm),
                h=0.015,
            )

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for window_index, (low, high) in enumerate(WINDOWS):
        for group in GROUPS:
            tick_positions.append(positions[(window_index, group)])
            tick_labels.append(group)
        center = float(np.mean([positions[(window_index, group)] for group in GROUPS]))
        axis.text(
            center,
            -0.30,
            f"{int(low)}-{int(high)} min",
            ha="center",
            va="top",
            transform=axis.get_xaxis_transform(),
            fontsize=8,
        )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
    axis.set_ylim(0, 1.0 + 0.12 * len(COMPARISONS))
    axis.set_ylabel("Fraction of Active frames")
    axis.set_title(rf"$\it{{{gene}}}$: Active fraction by time window")
    axis.grid(axis="y", alpha=0.3)
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
    output_root = args.output_root.resolve() / "10_HDAC_time_windows"
    output_root.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    deposited_statistics = pd.read_csv(module_root / "Fig5_duty_cycle_by_window_stats.csv")
    validation_frames: list[pd.DataFrame] = []
    for gene in ("Nanog", "Sox2"):
        cell_values = pd.read_csv(module_root / f"Fig5_{gene}_duty_cycle_by_window_cell.csv")
        recalculated = recalculate_statistics(cell_values, gene)
        expected = deposited_statistics.loc[deposited_statistics["gene"].eq(gene)].reset_index(drop=True)
        recalculated = recalculated.reset_index(drop=True)
        if not (
            expected[["gene", "win_idx", "window", "compare"]].equals(
                recalculated[["gene", "win_idx", "window", "compare"]]
            )
            and np.allclose(expected["p_raw"], recalculated["p_raw"], rtol=1e-12, atol=1e-15)
            and np.allclose(expected["p_holm"], recalculated["p_holm"], rtol=1e-12, atol=1e-15)
        ):
            raise AssertionError(f"Time-window statistics do not reproduce for {gene}")

        figure = plot_time_windows(cell_values, expected, gene)
        for text_object in figure.findobj(match=Text):
            text_object.set_fontsize(PLOT_FONT_SIZE)
        output_pdf = output_root / f"Fig5_{gene}_duty_cycle_by_window.pdf"
        figure.savefig(output_pdf, bbox_inches="tight", dpi=300)
        plt.close(figure)
        recalculated["validation"] = "PASS"
        validation_frames.append(recalculated)
        print(output_pdf)

    pd.concat(validation_frames, ignore_index=True).to_csv(
        output_root / "time_window_statistics_validation.csv", index=False
    )


if __name__ == "__main__":
    main()
