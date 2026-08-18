#!/usr/bin/env python3
"""Compare regenerated Gao2026 tables with deposited reference tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(*paths: Path) -> Path:
    """Return the first existing layout, preserving the first path for errors."""
    return next((path for path in paths if path.exists()), paths[0])


def compare_csv(
    label: str,
    observed: Path,
    expected: Path,
    ignored_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    row: dict[str, object] = {
        "comparison": label,
        "observed": str(observed),
        "expected": str(expected),
        "status": "missing",
        "observed_rows": None,
        "expected_rows": None,
        "max_abs_numeric_difference": None,
    }
    if not observed.exists() or not expected.exists():
        return row

    actual = pd.read_csv(observed).drop(
        columns=list(ignored_columns), errors="ignore"
    )
    reference = pd.read_csv(expected).drop(
        columns=list(ignored_columns), errors="ignore"
    )
    row["observed_rows"] = len(actual)
    row["expected_rows"] = len(reference)
    if actual.shape != reference.shape or list(actual.columns) != list(reference.columns):
        row["status"] = "shape_or_column_mismatch"
        return row

    numeric_columns = [
        column
        for column in actual.columns
        if pd.api.types.is_numeric_dtype(actual[column])
        and pd.api.types.is_numeric_dtype(reference[column])
    ]
    non_numeric_columns = [
        column for column in actual.columns if column not in numeric_columns
    ]
    non_numeric_equal = all(
        actual[column].fillna("<NA>").astype(str).equals(
            reference[column].fillna("<NA>").astype(str)
        )
        for column in non_numeric_columns
    )

    maximum_difference = 0.0
    numeric_equal = True
    for column in numeric_columns:
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(reference[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            maximum_difference = max(
                maximum_difference,
                float(np.max(np.abs(left[finite] - right[finite]))),
            )
        numeric_equal = numeric_equal and np.allclose(
            left,
            right,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    row["max_abs_numeric_difference"] = maximum_difference
    row["status"] = "pass" if numeric_equal and non_numeric_equal else "value_mismatch"
    return row


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
    output_root = args.output_root.expanduser().resolve()
    revision_root = data_root / "revision"
    if not revision_root.is_dir():
        revision_root = data_root
    seqfish_plot_root = first_existing(
        data_root / "04_seqFISH/plot_tables",
        revision_root / "04_seqFISH_plot_tables",
    )
    mtchil_plot_root = first_existing(
        data_root / "05_sci_mtChIL_seq/plot_tables",
        revision_root / "05_sci_mtChIL_plot_tables",
    )

    comparisons: list[tuple[str, Path, Path, tuple[str, ...]]] = [
        (
            "02 cross-correlation LLI summary",
            output_root / "02_Cross_correlation/output_LLI/LLI_summary_mod.csv",
            data_root
            / "02_Cross_correlation/reference_outputs/LLI_summary_mod.csv",
            ("csv_path",),
        ),
        (
            "04 seqFISH gene activity ratio",
            output_root
            / "04_seqFISH/results_saESC_delta/saESC_gene_on_ratio_by_RNAid.csv",
            seqfish_plot_root / "saESC_gene_on_ratio_by_RNAid.csv",
            (),
        ),
        (
            "04 seqFISH quartile statistics",
            output_root
            / "04_seqFISH/outputs/saESC_delta_wilcoxon_bh_onratio_quartiles.csv",
            seqfish_plot_root / "saESC_delta_wilcoxon_bh_onratio_quartiles.csv",
            (),
        ),
        (
            "04 seqFISH marker statistics",
            output_root / "04_seqFISH/results_saESC_delta/saESC_delta_stats.csv",
            seqfish_plot_root / "saESC_delta_stats.csv",
            (),
        ),
        (
            "05 sci-mtChIL active-cell values",
            output_root
            / "05_sci_mtChIL_seq/plot/nCell_posi_violin_box_u5d5n_posratio_bins500/nCell_posi_long_table.csv",
            mtchil_plot_root / "nCell_posi_long_table.csv",
            (),
        ),
        (
            "05 sci-mtChIL active-cell summary",
            output_root
            / "05_sci_mtChIL_seq/plot/nCell_posi_violin_box_u5d5n_posratio_bins500/nCell_posi_summary_by_target_bin.csv",
            mtchil_plot_root / "nCell_posi_summary_by_target_bin.csv",
            (),
        ),
        (
            "05 sci-mtChIL ratio values",
            output_root
            / "05_sci_mtChIL_seq/ratio_violin_boxplot_pairU5D5_chilRank_bins500/ratio_long_table_bins.csv",
            mtchil_plot_root / "ratio_long_table_bins.csv",
            (),
        ),
        (
            "05 sci-mtChIL ratio statistics",
            output_root
            / "05_sci_mtChIL_seq/ratio_violin_boxplot_pairU5D5_chilRank_bins500/stats_wilcoxon_bh_bins.csv",
            mtchil_plot_root / "stats_wilcoxon_bh_bins.csv",
            (),
        ),
        (
            "07 MSD replicate summary",
            output_root / "07_MSD/replicate_msd_summary.csv",
            data_root / "07_MSD/replicate_msd_summary.csv",
            (),
        ),
        (
            "07 MSD anomalous fits",
            output_root / "07_MSD/replicate_anomalous_fit.csv",
            data_root / "07_MSD/replicate_anomalous_fit.csv",
            (),
        ),
    ]

    expected_fig5 = data_root / "12_Fig5_integrated/integrated_tables"
    observed_fig5 = output_root / "12_Fig5_integrated/tables"
    for expected_path in sorted(expected_fig5.glob("*.csv")):
        comparisons.append(
            (
                f"12 Fig. 5 {expected_path.name}",
                observed_fig5 / expected_path.name,
                expected_path,
                (),
            )
        )

    rows = [compare_csv(*comparison) for comparison in comparisons]

    validation_files = (
        "08_SoRa/Supplementary_Fig_4f_table_validation.csv",
        "09_bead_resolution/Supplementary_Fig_4e_table_validation.csv",
        "10_HDAC_time_windows/time_window_statistics_validation.csv",
        "11_HDAC_threshold_sensitivity/threshold_sensitivity_table_validation.csv",
        "14_acute_inhibitor/Supplementary_Fig_13_table_validation.csv",
        "15_chipseq_enrichment/fig_s5e_table_validation.csv",
        "16_ChromHMM_Fig3c/Fig_3c_table_validation.csv",
    )
    for relative_path in validation_files:
        path = output_root / relative_path
        status = "missing"
        observed_rows = None
        if path.exists():
            validation = pd.read_csv(path)
            observed_rows = len(validation)
            status_columns = [
                column for column in validation.columns if "validation" in column
            ]
            status = (
                "pass"
                if status_columns
                and all(
                    validation[column].astype(str).eq("PASS").all()
                    for column in status_columns
                )
                else "value_mismatch"
            )
        rows.append(
            {
                "comparison": f"table validation {relative_path}",
                "observed": str(path),
                "expected": "all validation rows equal PASS",
                "status": status,
                "observed_rows": observed_rows,
                "expected_rows": observed_rows,
                "max_abs_numeric_difference": 0.0 if status == "pass" else None,
            }
        )

    report = pd.DataFrame(rows)
    report_path = output_root / "reproduction_numeric_validation.tsv"
    report.to_csv(report_path, sep="\t", index=False)
    print(
        report[
            [
                "comparison",
                "status",
                "observed_rows",
                "expected_rows",
                "max_abs_numeric_difference",
            ]
        ].to_string(index=False)
    )
    failures = report.loc[report["status"] != "pass"]
    print(f"\nPASS={len(report) - len(failures)} FAIL={len(failures)}")
    return 1 if not failures.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
