#!/usr/bin/env python3
"""Validate Gao2026 revision-derived processed tables without microscopy raw data."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def revision_root(data_root: Path) -> Path:
    candidate = data_root / "revision"
    return candidate if candidate.is_dir() else data_root


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GAO2026_DATA_ROOT", "data")),
        help="Root containing the extracted revision/ directory, or revision itself.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    root = revision_root(data_root)
    require(root.is_dir(), f"revision data root exists: {root}")

    cross_root = data_root / "02_Cross_correlation" / "csv"
    if not cross_root.is_dir():
        cross_root = root / "02_Cross_correlation_publication_inputs" / "csv"
    required_cross_files = {
        "all_cells_spot_intensity_seg_cell_BRD4.csv",
        "all_cells_spot_intensity_seg_cell_CHD4.csv",
        "all_cells_spot_intensity_seg_cell_H3K27ac.csv",
        "all_cells_spot_intensity_seg_cell_HDAC1.csv",
        "all_cells_spot_intensity_seg_cell_HDAC3.csv",
        "all_cells_spot_intensity_seg_cell_HDAC3_del_detrend.csv",
        "all_cells_spot_intensity_P300.csv",
        "all_cells_spot_intensity_seg_cell_SIN3A.csv",
        "all_cells_spot_intensity_seg_cell_SOX2.csv",
        "all_cells_spot_intensity_seg_cell_SOX2_del_detrend.csv",
        "all_cells_spot_intensity_seg_cell_SOX2_1min.csv",
    }
    observed_cross_files = {path.name for path in cross_root.glob("*.csv")}
    require(
        required_cross_files.issubset(observed_cross_files),
        "all 11 cross-correlation inputs used for multiple-testing correction are present",
    )

    temporal = read_rows(
        root / "06_SOX2_temporal_resolution" / "SOX2_cross_correlation_per_cell.csv"
    )
    temporal_cells: dict[str, set[str]] = defaultdict(set)
    for row in temporal:
        temporal_cells[row["condition"]].add(row["cell_id"])
    observed_temporal_n = {key: len(value) for key, value in temporal_cells.items()}
    require(
        observed_temporal_n == {"SOX2_30s": 97, "SOX2_1min": 80, "SOX2_2min": 32},
        f"SOX2 temporal-resolution cell counts: {observed_temporal_n}",
    )

    msd_rows = read_rows(root / "07_MSD" / "all_replicates_msd_per_trajectory.csv")
    msd_summary = read_rows(root / "07_MSD" / "replicate_msd_summary.csv")
    require(len(msd_rows) == 15442, "MSD per-trajectory rows = 15442")
    require(len(msd_summary) == 528, "MSD replicate-summary rows = 528")

    sora_profiles = list((root / "08_SoRa").rglob("gao_fig1c_radial_profiles.csv"))
    require(len(sora_profiles) == 8, "SoRa biological-replicate radial-profile files = 8")

    bead_rows = read_rows(root / "09_bead_resolution" / "paired_bead_source_data.csv")
    bead_counts: dict[str, int] = defaultdict(int)
    for row in bead_rows:
        if row["excluded_as_aggregate"].strip().lower() == "false":
            bead_counts[row["channel_label"]] += 1
    require(
        dict(bead_counts) == {"640 nm": 22, "515 nm": 17, "445 nm": 19},
        f"retained paired-bead counts: {dict(bead_counts)}",
    )

    time_stats = read_rows(root / "10_HDAC_time_windows" / "Fig5_duty_cycle_by_window_stats.csv")
    require(len(time_stats) == 12, "HDAC time-window statistical rows = 12")

    threshold_rows = read_rows(
        root
        / "11_HDAC_threshold_sensitivity"
        / "threshold_sweeps"
        / "threshold_sensitivity_effects.csv"
    )
    require(len(threshold_rows) == 120, "HDAC threshold-sensitivity effect rows = 120")

    fig5_root = root / "12_Fig5_integrated"
    fig5_expected_rows = {
        "Nanog": {"perframe": 9028, "runs": 1444},
        "Sox2": {"perframe": 6954, "runs": 1284},
    }
    for gene, expected in fig5_expected_rows.items():
        aggregate_root = fig5_root / gene / "tetr_mcp_hybrid_analysis"
        perframe_rows = read_rows(aggregate_root / "aggregated_perframe.csv")
        dwell_rows = read_rows(aggregate_root / "aggregated_runs_by_cell.csv")
        require(
            len(perframe_rows) == expected["perframe"],
            f"Fig. 5 {gene} aggregate per-frame rows = {expected['perframe']}",
        )
        require(
            len(dwell_rows) == expected["runs"],
            f"Fig. 5 {gene} aggregate dwell rows = {expected['runs']}",
        )
        required_columns = {"length_min", "left_censored", "right_censored"}
        require(
            required_columns.issubset(dwell_rows[0]),
            f"Fig. 5 {gene} dwell censoring columns are present",
        )
        km_rows = [
            row
            for row in dwell_rows
            if row["left_censored"].strip().lower() == "false"
        ]
        max_duration = max(float(row["length_min"]) for row in km_rows)
        require(
            max_duration <= 60.0,
            f"KM-included Fig. 5 {gene} dwell duration <= 60 min "
            f"(max={max_duration})",
        )

    fig5_integrated = fig5_root / "integrated_tables"
    require(
        len(read_rows(fig5_integrated / "Combined_duty_cycle_celllevel_MAIN.csv"))
        == 196,
        "Fig. 5 combined cell-level duty-cycle rows = 196",
    )

    marcs_rows = read_rows(root / "13_MARCS" / "complexes_volcano_all_complexes.csv")
    marcs_counts: dict[str, int] = defaultdict(int)
    for row in marcs_rows:
        marcs_counts[row["feature"]] += 1
    require(
        dict(marcs_counts) == {"H3ac": 129, "H4ac": 130, "H3K4me3": 130},
        f"MARCS plotted complex counts: {dict(marcs_counts)}",
    )

    acute_root = root / "14_acute_inhibitor"
    acute_counts = {
        "per_cell": len(read_rows(acute_root / "fig_s13_signal_per_cell.csv")),
        "summary": len(read_rows(acute_root / "fig_s13_signal_summary.csv")),
        "tests": len(read_rows(acute_root / "fig_s13_signal_tests.csv")),
        "active_fraction": len(read_rows(acute_root / "fig_s13_active_fraction.csv")),
        "manifest": len(read_rows(acute_root / "fig_s13_experiment_manifest.csv")),
    }
    require(
        acute_counts
        == {
            "per_cell": 13470,
            "summary": 140,
            "tests": 80,
            "active_fraction": 40,
            "manifest": 6,
        },
        f"Supplementary Fig. 13 table rows: {acute_counts}",
    )
    acute_per_cell = read_rows(acute_root / "fig_s13_signal_per_cell.csv")
    require(
        {row["metric"] for row in acute_per_cell} == {"local_bg", "nuclear_bg"},
        "Supplementary Fig. 13 includes both locus-centered and nucleus-wide metrics",
    )
    acute_manifest = read_rows(acute_root / "fig_s13_experiment_manifest.csv")
    experiment_counts = {
        row["panel"]: int(row["n_experiments"]) for row in acute_manifest
    }
    require(
        experiment_counts == {"a": 2, "b": 3, "c": 2, "d": 2, "e": 3, "f": 3},
        f"Supplementary Fig. 13 independent-experiment counts: {experiment_counts}",
    )

    chipseq_root = root / "15_chipseq_enrichment"
    chipseq_counts = {
        "genome_bins": len(read_rows(chipseq_root / "fig_s5e_genome_bins_joint_zscore.csv")),
        "gene_windows": len(read_rows(chipseq_root / "fig_s5e_gene_windows_joint_zscore.csv")),
        "target_summary": len(read_rows(chipseq_root / "fig_s5e_target_summary.csv")),
        "accession_map": len(read_rows(chipseq_root / "fig_s5e_accession_map.csv")),
    }
    require(
        chipseq_counts
        == {"genome_bins": 98316, "gene_windows": 72, "target_summary": 18, "accession_map": 18},
        f"Supplementary Fig. 5e table rows: {chipseq_counts}",
    )

    chromhmm_root = root / "16_ChromHMM_Fig3c"
    chromhmm_counts = {
        "emissions": len(read_rows(chromhmm_root / "fig3c_emission_probabilities_long.csv")),
        "mean_signals": len(read_rows(chromhmm_root / "fig3c_factor_state_mean_signal_long.csv")),
        "zscores": len(read_rows(chromhmm_root / "fig3c_factor_state_zscores_long.csv")),
        "states": len(read_rows(chromhmm_root / "fig3c_chromhmm_state_order.csv")),
        "accessions": len(read_rows(chromhmm_root / "fig3c_bigwig_accession_map.csv")),
        "state_beds": len(list((chromhmm_root / "state_beds").glob("*.bed"))),
    }
    require(
        chromhmm_counts
        == {
            "emissions": 120,
            "mean_signals": 216,
            "zscores": 216,
            "states": 12,
            "accessions": 18,
            "state_beds": 12,
        },
        f"Fig. 3c ChromHMM source rows and BED files: {chromhmm_counts}",
    )

    print("Processed-table validation completed successfully; no raw images were accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
