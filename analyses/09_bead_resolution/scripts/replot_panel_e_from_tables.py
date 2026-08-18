#!/usr/bin/env python3
"""Replot Supplementary Fig. 4e from the deposited paired-bead table.

The plotting block is the panel-e portion of the final manuscript notebook.
Panel d cannot be regenerated without the two bead ND2 stacks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CHANNEL_LABELS = {0: "640 nm", 1: "515 nm", 2: "445 nm"}
CHANNEL_COLORS = {0: "#C83E4D", 1: "#3A9D5D", 2: "#2D79B7"}


def resolve_module_root(data_root: Path) -> Path:
    for candidate in (
        data_root / "revision" / "09_bead_resolution",
        data_root / "09_bead_resolution",
        data_root,
    ):
        if (candidate / "paired_bead_source_data.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find the bead source-data table below {data_root}")


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
    output_root = args.output_root.resolve() / "09_bead_resolution"
    output_root.mkdir(parents=True, exist_ok=True)

    paired_beads = pd.read_csv(module_root / "paired_bead_source_data.csv")
    if "excluded_as_aggregate" in paired_beads:
        paired_beads = paired_beads.loc[~paired_beads["excluded_as_aggregate"].astype(bool)].copy()
    paired_summary = pd.read_csv(module_root / "paired_bead_summary.csv")

    expected_counts = dict(zip(paired_summary["channel_index"], paired_summary["paired_beads"]))
    observed_counts = paired_beads.groupby("channel_index").size().to_dict()
    if observed_counts != {int(key): int(value) for key, value in expected_counts.items()}:
        raise AssertionError(
            f"Paired-bead counts do not match the manuscript summary: {observed_counts} vs {expected_counts}"
        )
    for row in paired_summary.itertuples(index=False):
        pairs = paired_beads.loc[paired_beads["channel_index"].eq(row.channel_index)]
        checks = {
            "normal_median": np.isclose(
                pairs["normal_fwhm_xy_nm"].median(), row.normal_median_fwhm_xy_nm, rtol=0, atol=1e-12
            ),
            "sora_median": np.isclose(
                pairs["sora_fwhm_xy_nm"].median(), row.sora_median_fwhm_xy_nm, rtol=0, atol=1e-12
            ),
        }
        if not all(checks.values()):
            raise AssertionError(f"Bead summary mismatch for channel {row.channel_label}: {checks}")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    channel_indices = list(CHANNEL_LABELS)
    figure, paired_axis = plt.subplots(figsize=(5.6, 4.3))
    group_centers = np.arange(len(channel_indices), dtype=float) * 2.3

    for channel_index, group_center in zip(channel_indices, group_centers):
        pairs = paired_beads.query("channel_index == @channel_index")
        x_normal = group_center - 0.28
        x_sora = group_center + 0.28
        for pair in pairs.itertuples(index=False):
            paired_axis.plot(
                [x_normal, x_sora],
                [pair.normal_fwhm_xy_nm, pair.sora_fwhm_xy_nm],
                color="#B8B8B8",
                linewidth=0.65,
                alpha=0.70,
                zorder=1,
            )
        paired_axis.scatter(
            np.full(len(pairs), x_normal),
            pairs["normal_fwhm_xy_nm"],
            s=14,
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.6,
            zorder=2,
        )
        paired_axis.scatter(
            np.full(len(pairs), x_sora),
            pairs["sora_fwhm_xy_nm"],
            s=14,
            facecolor=CHANNEL_COLORS[channel_index],
            edgecolor=CHANNEL_COLORS[channel_index],
            linewidth=0.6,
            zorder=3,
        )
        paired_axis.plot(
            [x_normal - 0.12, x_normal + 0.12],
            [pairs["normal_fwhm_xy_nm"].median()] * 2,
            color="black",
            linewidth=2.0,
            zorder=4,
        )
        paired_axis.plot(
            [x_sora - 0.12, x_sora + 0.12],
            [pairs["sora_fwhm_xy_nm"].median()] * 2,
            color="black",
            linewidth=2.0,
            zorder=4,
        )

    paired_axis.set_xticks(group_centers)
    paired_axis.set_xticklabels([CHANNEL_LABELS[index] for index in channel_indices])
    paired_axis.set_xlabel("Detection channel")
    paired_axis.set_ylabel("Observed lateral FWHM (nm)")
    paired_axis.grid(axis="y", linestyle=":", color="#D0D0D0", linewidth=0.7)
    paired_axis.spines[["top", "right"]].set_visible(False)
    paired_axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white",
                markeredgecolor="#333333", label="CSU-W1"
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="#666666",
                markeredgecolor="#666666", label="SoRa"
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    paired_axis.set_ylim(top=300)
    figure.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.18)

    figure_base = output_root / "Supplementary_Fig_4e_bead_width_replot"
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(figure_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    validation = paired_summary.copy()
    validation["table_validation"] = "PASS"
    validation.to_csv(output_root / "Supplementary_Fig_4e_table_validation.csv", index=False)
    print(figure_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
