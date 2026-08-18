#!/usr/bin/env python3
"""Validate the date-stamped Normal and SoRa revision outputs."""

from __future__ import annotations


# Portable Gao2026 release paths. Override these with environment variables.
from pathlib import Path as _Gao2026Path
import os as _gao2026_os

_GAO2026_REPO_ROOT = _Gao2026Path(__file__).resolve().parents[3]
GAO2026_DATA_ROOT = _Gao2026Path(
    _gao2026_os.environ.get("GAO2026_DATA_ROOT", _GAO2026_REPO_ROOT / "data")
).expanduser().resolve()
GAO2026_RAW_ROOT = _Gao2026Path(
    _gao2026_os.environ.get("GAO2026_RAW_ROOT", GAO2026_DATA_ROOT / "external_raw")
).expanduser().resolve()
GAO2026_OUTPUT_ROOT = _Gao2026Path(
    _gao2026_os.environ.get("GAO2026_OUTPUT_ROOT", _GAO2026_REPO_ROOT / "outputs")
).expanduser().resolve()
GAO2026_EXTERNAL_ROOT = _Gao2026Path(
    _gao2026_os.environ.get("GAO2026_EXTERNAL_ROOT", GAO2026_DATA_ROOT / "external")
).expanduser().resolve()

import json
from pathlib import Path

import nbformat
import pandas as pd


BASE = Path(
    str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO')
)
REVISION_DIR = BASE / "revision_20260809"
NORMAL_AGGREGATE = (
    BASE / "normal_snapshot_outputs_all_datasets_gao3d_revision_20260809"
)
SORA_AGGREGATE = BASE / (
    "sora_snapshot_outputs_all_replicates_rep1_rep4_gao3d_"
    "rmass039um_r1035_mcp105_revision_20260809"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_errors(path: Path) -> list[dict]:
    notebook = nbformat.read(path, as_version=4)
    errors: list[dict] = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    {
                        "cell_index": cell_index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )
    return errors


def random_control_coverage(dataset_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for root in sorted(dataset_roots):
        tables = root / "tables"
        random_table = pd.read_csv(
            tables / "gao_random_control_population_by_fov.csv"
        )
        segmentation = pd.read_csv(
            tables / "segmentation_regions_all_fovs.csv"
        )
        selected = (
            segmentation["selected"]
            .astype(str)
            .str.lower()
            .eq("true")
            .sum()
        )
        controls = int(random_table["random_control_count"].sum())
        rows.append(
            {
                "dataset": root.name,
                "random_controls": controls,
                "qc_selected_nuclei": int(selected),
                "coverage": controls / selected if selected else float("nan"),
                "population": ";".join(
                    sorted(random_table["population"].dropna().unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_normal() -> dict:
    manifest = load_json(NORMAL_AGGREGATE / "batch_analysis_manifest.json")
    roots = list(
        BASE.glob(
            "normal_snapshot_outputs_*_gao3d_revision_20260809"
        )
    )
    roots = [root for root in roots if root != NORMAL_AGGREGATE]
    coverage = random_control_coverage(roots)
    notebook = REVISION_DIR / (
        "normal_fixed_snapshot_all_datasets_gao3d_revision_20260809.ipynb"
    )
    report = {
        "manifest_passed": bool(manifest["all_validations_passed"]),
        "dataset_count": len(roots),
        "fov_count": int(manifest["total_fov_count"]),
        "random_control_count": int(coverage["random_controls"].sum()),
        "qc_selected_nuclei": int(coverage["qc_selected_nuclei"].sum()),
        "minimum_random_coverage": float(coverage["coverage"].min()),
        "notebook_errors": notebook_errors(notebook),
    }
    report["passed"] = bool(
        report["manifest_passed"]
        and report["dataset_count"] == 8
        and report["fov_count"] == 1300
        and report["minimum_random_coverage"] >= 0.95
        and not report["notebook_errors"]
    )
    coverage.to_csv(
        REVISION_DIR / "normal_random_control_coverage_revision_20260809.csv",
        index=False,
    )
    return report


def validate_sora() -> dict:
    manifest = load_json(SORA_AGGREGATE / "batch_analysis_manifest.json")
    roots = list(
        BASE.glob("sora_snapshot_outputs_*_rmass039um_revision_20260809")
    )
    coverage_rows: list[dict] = []
    for root in sorted(roots):
        acquisition_manifest = load_json(root / "analysis_manifest.json")
        coverage_rows.append(
            {
                "dataset": root.name,
                "random_controls": int(
                    acquisition_manifest["n_all_qc_random_controls"]
                ),
                "qc_selected_nuclei": int(
                    acquisition_manifest["n_qc_selected_nuclei"]
                ),
                "coverage": float(
                    acquisition_manifest["random_control_coverage"]
                ),
                "population": (
                    "all_QC_passing_nuclei_independent_of_mTetR_detection"
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    notebook = REVISION_DIR / (
        "sora_fixed_snapshot_rep1_rep4_gao3d_revision_20260809.ipynb"
    )
    focus_root = SORA_AGGREGATE / "focus_qc_sensitivity"
    focus_outputs = [
        focus_root / "tables" / "excluded_acquisitions.csv",
        focus_root / "tables" / "focus_qc_sensitivity_comparison.csv",
        focus_root / "figures" / "focus_qc_passing_gao_summaries.png",
    ]
    distance_um = float(manifest["selected_thresholds"]["r_mass_radius_um"])
    report = {
        "manifest_passed": bool(manifest["all_validations_passed"]),
        "dataset_count": len(roots),
        "fov_count": int(manifest["total_fov_count"]),
        "random_control_count": int(coverage["random_controls"].sum()),
        "qc_selected_nuclei": int(coverage["qc_selected_nuclei"].sum()),
        "minimum_random_coverage": float(coverage["coverage"].min()),
        "r_mass_radius_px": int(
            manifest["selected_thresholds"]["r_mass_radius_px"]
        ),
        "r_mass_radius_um": distance_um,
        "focus_qc_outputs_complete": all(
            path.is_file() and path.stat().st_size > 0
            for path in focus_outputs
        ),
        "notebook_errors": notebook_errors(notebook),
    }
    report["passed"] = bool(
        report["manifest_passed"]
        and report["dataset_count"] == 24
        and report["fov_count"] == 8503
        and report["minimum_random_coverage"] >= 0.95
        and report["r_mass_radius_px"] == 17
        and abs(distance_um - 0.39) <= 0.01
        and report["focus_qc_outputs_complete"]
        and not report["notebook_errors"]
    )
    coverage.to_csv(
        REVISION_DIR / "sora_random_control_coverage_revision_20260809.csv",
        index=False,
    )
    return report


def main() -> None:
    report = {
        "normal": validate_normal(),
        "sora": validate_sora(),
    }
    report["passed"] = bool(
        report["normal"]["passed"] and report["sora"]["passed"]
    )
    output_path = REVISION_DIR / "revision_validation_20260809.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
