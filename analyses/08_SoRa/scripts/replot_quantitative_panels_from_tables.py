#!/usr/bin/env python3
"""Replot the quantitative portion of Supplementary Fig. 4f.

The statistical inputs are the exact tables exported by the manuscript SoRa
pipeline. This script deliberately does not reconstruct the representative
mean-image panels, which require the per-condition ``gao_fig1c_scaled_images``
NPZ files that are not in the release candidate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


CONDITIONS = (
    ("nanog_h3k27ac", r"$\it{Nanog}$ H3K27ac"),
    ("nanog_ser5ph", r"$\it{Nanog}$ RNAPII Ser5ph"),
    ("sox2_h3k27ac", r"$\it{Sox2}$ H3K27ac"),
    ("sox2_ser5ph", r"$\it{Sox2}$ RNAPII Ser5ph"),
)
REPLICATES = ("rep1",)
STATE_COLORS = {
    "Active": "tab:orange",
    "Inactive": "tab:blue",
    "Random": "tab:green",
}
FONT_SIZE = 10


def resolve_module_root(data_root: Path) -> Path:
    candidates = (
        data_root / "revision" / "08_SoRa",
        data_root / "08_SoRa",
        data_root,
    )
    for candidate in candidates:
        if (candidate / "biological_replicate_results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find revision/08_SoRa/biological_replicate_results below "
        f"{data_root}"
    )


def format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = n/a"
    return f"p = {p_value:.1e}"


def load_group(module_root: Path, condition: str, replicate: str) -> dict[str, pd.DataFrame | pd.Series]:
    table_root = (
        module_root
        / "biological_replicate_results"
        / f"{condition}_{replicate}"
        / "tables"
    )
    radial = pd.read_csv(table_root / "gao_fig1c_radial_profiles.csv")
    values = pd.read_csv(table_root / "gao_fig1c_core_annulus_values.csv")
    statistics = pd.read_csv(table_root / "gao_fig1c_core_annulus_statistics.csv")

    snap_mask = radial["channel"].astype(str).str.startswith("SNAPtag")
    radial = radial.loc[snap_mask].copy()
    snap_channel = radial["channel"].iloc[0]
    values = values.loc[values["channel"].eq(snap_channel)].copy()
    statistics = statistics.loc[statistics["channel"].eq(snap_channel)].iloc[0]

    active = values.loc[values["state"].eq("Active"), "core_minus_annulus"].dropna()
    inactive = values.loc[values["state"].eq("Inactive"), "core_minus_annulus"].dropna()
    recalculated_p = mannwhitneyu(active, inactive, alternative="two-sided").pvalue
    checks = {
        "n_active_match": len(active) == int(statistics["n_active"]),
        "n_inactive_match": len(inactive) == int(statistics["n_inactive"]),
        "active_median_match": np.isclose(
            np.median(active), statistics["active_median_core_minus_annulus"], rtol=0, atol=1e-12
        ),
        "inactive_median_match": np.isclose(
            np.median(inactive), statistics["inactive_median_core_minus_annulus"], rtol=0, atol=1e-12
        ),
        "pvalue_match": np.isclose(
            recalculated_p, statistics["pvalue_two_sided"], rtol=1e-12, atol=0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"SoRa table validation failed for {condition}_{replicate}: {checks}")
    return {"radial": radial, "values": values, "statistics": statistics}


def plot_violin(axis: plt.Axes, values: pd.DataFrame, statistics: pd.Series) -> None:
    active = values.loc[
        values["state"].eq("Active"), "core_minus_annulus"
    ].dropna().to_numpy(float)
    inactive = values.loc[
        values["state"].eq("Inactive"), "core_minus_annulus"
    ].dropna().to_numpy(float)

    violins = axis.violinplot(
        [active, inactive],
        positions=[0, 1],
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=100,
    )
    for body, state in zip(violins["bodies"], ("Active", "Inactive")):
        body.set_facecolor(STATE_COLORS[state])
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
        body.set_alpha(0.72)

    axis.scatter(
        [0, 1],
        [float(np.median(active)), float(np.median(inactive))],
        marker="_",
        s=120,
        linewidths=1.5,
        color="black",
        zorder=3,
    )
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    axis.set_xticks([0, 1], ["Act.", "Inac."])
    axis.set_xlim(-0.55, 1.55)
    axis.margins(y=0.08)

    y_min, y_max = axis.get_ylim()
    y_range = max(y_max - y_min, 1.0)
    bracket_y = y_max + 0.035 * y_range
    bracket_tick = 0.025 * y_range
    text_y = bracket_y + 0.025 * y_range
    axis.plot(
        [0, 0, 1, 1],
        [bracket_y - bracket_tick, bracket_y, bracket_y, bracket_y - bracket_tick],
        color="black",
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        0.5,
        text_y,
        format_pvalue(float(statistics["pvalue_two_sided"])),
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )
    axis.set_ylim(y_min, text_y + 0.055 * y_range)
    axis.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


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
    output_root = args.output_root.resolve() / "08_SoRa"
    output_root.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        len(CONDITIONS),
        2 * len(REPLICATES),
        figsize=(6.0 * len(REPLICATES), 10.0),
        constrained_layout=True,
    )
    validation_rows: list[dict[str, object]] = []

    for row_index, (condition, row_label) in enumerate(CONDITIONS):
        for replicate_index, replicate in enumerate(REPLICATES):
            data = load_group(module_root, condition, replicate)
            radial_axis = axes[row_index, 2 * replicate_index]
            violin_axis = axes[row_index, 2 * replicate_index + 1]
            radial = data["radial"]

            for state in ("Active", "Inactive", "Random"):
                profile = radial.loc[radial["state"].eq(state)].sort_values("radius_um")
                if profile.empty:
                    continue
                radius = profile["radius_um"].to_numpy(float)
                mean = profile["mean_scaled_intensity"].to_numpy(float)
                sem = profile["sem_scaled_intensity"].to_numpy(float)
                radial_axis.plot(radius, mean, color=STATE_COLORS[state], linewidth=1.8, label=state)
                radial_axis.fill_between(
                    radius,
                    mean - sem,
                    mean + sem,
                    color=STATE_COLORS[state],
                    alpha=0.20,
                    linewidth=0,
                )
            radial_axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
            radial_axis.set_xlabel("Radius (µm)")
            radial_axis.set_ylabel("Scaled intensity (max = 100)")
            radial_axis.grid(color="#D0D0D0", linestyle=":", linewidth=0.7)
            radial_axis.spines[["top", "right"]].set_visible(False)
            radial_axis.set_title("Displayed experiment")
            plot_violin(violin_axis, data["values"], data["statistics"])
            violin_axis.set_ylabel("Core minus annulus (a.u.)")

            stats = data["statistics"]
            validation_rows.append(
                {
                    "condition": condition,
                    "public_replicate": replicate,
                    "biological_replicate": replicate_index + 1,
                    "n_active": int(stats["n_active"]),
                    "n_inactive": int(stats["n_inactive"]),
                    "pvalue_two_sided": float(stats["pvalue_two_sided"]),
                    "table_validation": "PASS",
                }
            )

        axes[row_index, 0].text(
            -0.42,
            0.5,
            row_label,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=FONT_SIZE,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Supplementary Fig. 4f quantitative replot from deposited tables\n"
        "Representative mean-image panels require non-deposited NPZ image arrays",
        y=1.075,
        fontsize=12,
    )
    output_pdf = output_root / "Supplementary_Fig_4f_quantitative_replot_no_images.pdf"
    figure.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(figure)

    validation_path = output_root / "Supplementary_Fig_4f_table_validation.csv"
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    print(output_pdf)
    print(validation_path)


if __name__ == "__main__":
    main()
