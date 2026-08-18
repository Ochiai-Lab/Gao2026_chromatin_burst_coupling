#!/usr/bin/env python3
"""Replot Fig. 3c from the deposited emission and signal matrices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


STATE_NAME_MAP = {
    1: "Heterochromatin",
    2: "Intergenic_Region",
    3: "Weakly_Repressed_Polycomb",
    4: "Repressed_Polycomb",
    5: "Bivalent_Promoter",
    6: "Weak_Enhancer",
    7: "Strong_Enhancer",
    8: "Enhancer",
    9: "Transcription_Transition",
    10: "Transcription_Elongation",
    11: "Insulator",
    12: "Active_Promoter",
}


def to_long(
    frame: pd.DataFrame,
    value_name: str,
    column_name: str,
) -> pd.DataFrame:
    result = (
        frame.rename_axis("chromatin_state")
        .reset_index()
        .melt(
            id_vars="chromatin_state",
            var_name=column_name,
            value_name=value_name,
        )
    )
    result.insert(
        0,
        "plot_state_order",
        result["chromatin_state"].map(
            {name.replace("_", " "): order for order, name in enumerate(frame.index, 1)}
        ),
    )
    state_id_by_label = {
        state_name.replace("_", " "): state_id
        for state_id, state_name in STATE_NAME_MAP.items()
    }
    result.insert(
        1,
        "chromhmm_state_id",
        result["chromatin_state"].map(state_id_by_label),
    )
    return result


def main() -> int:
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

    data_root = args.data_root.expanduser().resolve()
    module_root = data_root / "16_ChromHMM_Fig3c"
    if not module_root.is_dir():
        module_root = data_root / "revision" / "16_ChromHMM_Fig3c"
    output_root = args.output_root.expanduser().resolve() / "16_ChromHMM_Fig3c"
    output_root.mkdir(parents=True, exist_ok=True)

    emissions_file = module_root / "emissions_12.txt"
    signal_matrix_file = module_root / "chromState_signal_matrix.tsv"
    output_pdf = output_root / "Fig.3c.pdf"

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    sns.set(style="white")

    fs_title = 16
    fs_xtick = 16
    fs_ytick = 16
    fs_cbar_label = 16
    fs_cbar_tick = 16

    bottom = pd.read_csv(signal_matrix_file, sep="\t", index_col=0)
    bottom = bottom.drop(index=["RNAPII"], errors="ignore")
    state_order = bottom.columns.tolist()

    emissions_raw = pd.read_csv(emissions_file, sep="\t")
    emissions_raw["state_name"] = emissions_raw["State (Emission order)"].map(
        STATE_NAME_MAP
    )
    emissions_raw = emissions_raw.set_index("state_name").drop(
        columns=["State (Emission order)"]
    )

    states_common = [state for state in state_order if state in emissions_raw.index]
    state_labels = [state.replace("_", " ") for state in states_common]

    emissions_left = emissions_raw.loc[states_common].iloc[:, ::-1]
    emissions_left.index = state_labels
    marks_labels = emissions_left.columns.tolist()

    state_factor = bottom[states_common].T.iloc[:, ::-1]
    right_zscore = state_factor.sub(state_factor.mean(axis=0), axis=1)
    right_zscore = right_zscore.div(state_factor.std(axis=0), axis=1).fillna(0)
    right_zscore.index = state_labels
    factor_labels = right_zscore.columns.tolist()

    emission_long = to_long(emissions_left, "emission_probability", "mark")
    mean_signal = state_factor.copy()
    mean_signal.index = state_labels
    mean_signal_long = to_long(mean_signal, "length_weighted_mean_signal", "factor")
    zscore_long = to_long(right_zscore, "z_score", "factor")
    emission_long.to_csv(
        output_root / "fig3c_emission_probabilities_long.csv", index=False
    )
    mean_signal_long.to_csv(
        output_root / "fig3c_factor_state_mean_signal_long.csv", index=False
    )
    zscore_long.to_csv(
        output_root / "fig3c_factor_state_zscores_long.csv", index=False
    )

    figure, (axis_left, axis_right) = plt.subplots(
        1,
        2,
        figsize=(11, 7),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1.7]},
    )

    heatmap_left = sns.heatmap(
        emissions_left,
        ax=axis_left,
        cmap="Blues",
        cbar=True,
        cbar_kws={
            "orientation": "horizontal",
            "pad": 0.08,
            "fraction": 0.05,
            "label": "ChromHMM\nemission probabilities",
        },
    )
    axis_left.set_title("Probability of mark/state", pad=70, fontsize=fs_title)
    axis_left.set_xlabel("")
    axis_left.xaxis.tick_top()
    axis_left.tick_params(axis="x", labeltop=True, labelbottom=False)
    axis_left.set_xticklabels(
        marks_labels, rotation=-90, ha="center", fontsize=fs_xtick
    )
    axis_left.tick_params(axis="y", rotation=0, labelsize=fs_ytick)
    colorbar_left = heatmap_left.collections[0].colorbar
    colorbar_left.ax.xaxis.label.set_ha("center")
    colorbar_left.ax.xaxis.label.set_size(fs_cbar_label)
    colorbar_left.ax.tick_params(labelsize=fs_cbar_tick)
    colorbar_left.formatter = FormatStrFormatter("%.1f")
    colorbar_left.update_ticks()

    heatmap_right = sns.heatmap(
        right_zscore,
        ax=axis_right,
        cmap="RdBu_r",
        center=0.0,
        cbar=True,
        cbar_kws={
            "orientation": "horizontal",
            "pad": 0.08,
            "fraction": 0.05,
            "label": "Z-score",
        },
    )
    axis_right.set_title("Factor Enrichment", pad=40, fontsize=fs_title)
    axis_right.set_xlabel("")
    axis_right.set_ylabel("")
    axis_right.xaxis.tick_top()
    axis_right.tick_params(axis="x", labeltop=True, labelbottom=False)
    axis_right.set_xticklabels(
        factor_labels, rotation=-90, ha="center", fontsize=fs_xtick
    )
    axis_right.tick_params(axis="y", left=False, labelleft=False)
    colorbar_right = heatmap_right.collections[0].colorbar
    colorbar_right.ax.xaxis.label.set_size(fs_cbar_label)
    colorbar_right.ax.tick_params(labelsize=fs_cbar_tick)
    colorbar_right.formatter = FormatStrFormatter("%.1f")
    colorbar_right.update_ticks()

    figure.tight_layout()
    figure.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.close(figure)

    expected_path = module_root / "fig3c_factor_state_zscores_long.csv"
    validation_rows: list[dict[str, object]] = []
    if expected_path.exists():
        expected = pd.read_csv(expected_path)
        key_columns = [
            "plot_state_order",
            "chromhmm_state_id",
            "chromatin_state",
            "factor",
        ]
        merged = zscore_long.merge(
            expected,
            on=key_columns,
            how="outer",
            suffixes=("_observed", "_expected"),
            indicator=True,
        )
        numeric_difference = (
            merged["z_score_observed"] - merged["z_score_expected"]
        ).abs()
        max_difference = float(numeric_difference.max())
        passed = bool(
            merged["_merge"].eq("both").all()
            and np.allclose(
                merged["z_score_observed"],
                merged["z_score_expected"],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )
        validation_rows.append(
            {
                "comparison": "Fig. 3c deposited versus regenerated z-scores",
                "observed_rows": len(zscore_long),
                "expected_rows": len(expected),
                "max_abs_difference": max_difference,
                "validation_status": "PASS" if passed else "FAIL",
            }
        )
    else:
        validation_rows.append(
            {
                "comparison": "Fig. 3c deposited versus regenerated z-scores",
                "observed_rows": len(zscore_long),
                "expected_rows": "",
                "max_abs_difference": "",
                "validation_status": "REFERENCE_NOT_AVAILABLE",
            }
        )
    pd.DataFrame(validation_rows).to_csv(
        output_root / "Fig_3c_table_validation.csv", index=False
    )
    print(f"Saved {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
