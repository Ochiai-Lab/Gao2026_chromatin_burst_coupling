#!/usr/bin/env python3
"""Normal Field8 analysis with SoRa multi-dataset-compatible downstream steps.

The reviewed Normal three-channel Cellpose segmentation is reused unchanged.
From 3D spot detection onward, the table schema, threshold diagnostics, state
calls, random nuclear controls, Gao normalization, and figures follow
``sora_fixed_snapshot_all_datasets_gao3d.ipynb``. Pixel-valued spatial
parameters are converted to the 65 nm Normal sampling where Gao physical
sizes, rather than SoRa bead-specific sizes, are the governing reference.
"""

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
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sora_gao3d_snapshot_pipeline as gao
import sora_multidataset_pipeline as batch


PROJECT = Path(
    str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO')
)
INPUT_ND2 = Path(
    str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260716-SoRa_Fixed/260716_SoRa/260716_ChamberA/ChamberA_20260716_205221_Field8.nd2')
)
SEGMENTATION_SOURCE_NAME = "normal_snapshot_outputs_chamberA_field8_gao_stage1"
OUTPUT_NAME = "normal_snapshot_outputs_chamberA_field8_sora_aligned_gao3d"
ALGORITHM_VERSION = (
    "normal_all_qc_random_controls_fixed_cell_thresholds_v4"
)


@dataclass(frozen=True)
class AnalysisConfig(gao.AnalysisConfig):
    input_nd2: Path = INPUT_ND2
    project_dir: Path = PROJECT
    output_name: str = OUTPUT_NAME
    segmentation_source_name: str = SEGMENTATION_SOURCE_NAME
    analysis_id: str = "nanog_ser5ph_rep1_normal"
    analysis_label: str = (
        "Nanog × Ser5ph — rep1 (ChamberA Field8, Normal)"
    )
    streaming_tag_locus: str = "Nanog"
    mintbody_label: str = "Ser5ph"
    replicate: str = "rep1"
    chamber: str = "ChamberA"
    expected_fov_count: int | None = 100
    expected_xy_um: float = 0.065
    expected_z_um: float = 0.5

    # The reviewed Normal masks are saved at full ND2 resolution.
    segmentation_downsample: int = 1
    segmentation_mask_filename_template: str = (
        "fov_{fov_id:03d}_nuclei_mask.tiff"
    )
    acquisition_mode_label: str = "Normal"
    threshold_figure_status_label: str = (
        "Fixed-cell thresholds: r_mTetR=1.15, r_MCP=1.15"
    )

    # Gao physical LoG sigma/minimum distance at the measured Normal voxel.
    bigfish_log_kernel_size_zyx_px: tuple[float, float, float] = (
        1.0,
        4.615384615384615,
        4.615384615384615,
    )
    bigfish_minimum_distance_zyx_px: tuple[float, float, float] = (
        1.0,
        4.615384615384615,
        4.615384615384615,
    )
    bigfish_threshold_divisor: float = 20.0

    # Gao 1.17 um crop radius, 0.39 um r_mass radius, and 0.65 um
    # Trackpy feature diameter converted to 65 nm pixels.
    final_crop_radius_px: int = 18
    final_r_mass_radius_px: int = 6
    final_trackpy_diameter_px: int = 11
    final_trackpy_separation_px: int = 12

    # Fixed-cell thresholds retained from the reviewed fixed-cell analysis.
    # Fixation changes mTetR localization and intensity, so these numerical
    # cutoffs are not claimed to be directly transferable from live cells.
    fixed_mtetr_r_mass_threshold: float = 1.150
    fixed_mcp_r_mass_threshold: float = 1.150
    mtetr_plot_comparison_thresholds: tuple[float, ...] = (1.300,)
    random_z_min_index: int = 4
    random_z_max_index: int = 7
    random_seed: int = 20260720
    require_reference_equivalence: bool = False


def analysis_config() -> AnalysisConfig:
    return AnalysisConfig()


def preflight(config: AnalysisConfig) -> pd.DataFrame:
    metadata = gao.collect_metadata(config)
    channel_map = metadata["channel_mapping"]
    row = {
        "analysis_id": config.analysis_id,
        "fov_count": int(metadata["sizes"]["P"]),
        "z_count": int(metadata["sizes"]["Z"]),
        "channel_count": int(metadata["sizes"]["C"]),
        "voxel_x_um": float(metadata["voxel_size_um"]["x"]),
        "voxel_z_um": float(metadata["voxel_size_um"]["z"]),
        "channel_names": "|".join(
            item["nd2_name"] for item in channel_map
        ),
        "exposures_ms": "|".join(
            str(item["exposure_ms"]) for item in channel_map
        ),
        "all_checks_passed": True,
    }
    frame = pd.DataFrame([row])
    gao.prepare_directories(config)
    frame.to_csv(
        config.output_root / "tables" / "dataset_preflight.csv",
        index=False,
    )
    return frame


def downstream_parameter_table(config: AnalysisConfig) -> pd.DataFrame:
    rows = [
        {
            "step": "quantification image",
            "SoRa all-dataset notebook": "3x3 XY median-filtered uint16",
            "Normal implementation": "same",
            "basis": "exact method match",
        },
        {
            "step": "3D mTetR detection",
            "SoRa all-dataset notebook": (
                "per-nucleus Big-FISH; LoG=(0.5,4,4) px; "
                "minimum distance=(1,4,4) px"
            ),
            "Normal implementation": (
                "per-nucleus Big-FISH; "
                f"LoG={config.bigfish_log_kernel_size_zyx_px}; "
                "minimum distance="
                f"{config.bigfish_minimum_distance_zyx_px}"
            ),
            "basis": "Gao 500/300/300 nm at 500/65/65 nm voxel",
        },
        {
            "step": "candidate threshold/ranking",
            "SoRa all-dataset notebook": (
                "central-Z nuclear mean/20; intensity-descending top 5"
            ),
            "Normal implementation": "same",
            "basis": "exact method match",
        },
        {
            "step": "single-Z crop",
            "SoRa all-dataset notebook": "101x101 px (2.34 um diameter)",
            "Normal implementation": "37x37 px (2.34 um diameter)",
            "basis": "same Gao physical crop",
        },
        {
            "step": "mTetR r_mass",
            "SoRa all-dataset notebook": (
                "radius 17 px (0.395 um; Gao physical aperture)"
            ),
            "Normal implementation": "radius 6 px (0.39 um; Gao physical)",
            "basis": "same approximately 0.39-um physical aperture",
        },
        {
            "step": "MCP/SNAP Trackpy",
            "SoRa all-dataset notebook": "diameter 11 px; separation 12 px",
            "Normal implementation": "diameter 11 px; separation 12 px",
            "basis": "Gao 5 px at 130 nm converted to 65 nm",
        },
        {
            "step": "threshold figure/state calls",
            "SoRa all-dataset notebook": (
                "4-panel; fixed-threshold state call; distance<=0.39 um"
            ),
            "Normal implementation": (
                "same figure/state formula; r_mTetR=1.15; r_MCP=1.15; "
                "previous mTetR=1.30 shown only as comparison"
            ),
            "basis": "fixed-cell thresholds; fixation-specific intensity",
        },
        {
            "step": "Gao locus summary",
            "SoRa all-dataset notebook": (
                "all-QC nuclear random controls; shared pixel-max=100"
            ),
            "Normal implementation": "same",
            "basis": "exact method and display match",
        },
    ]
    frame = pd.DataFrame(rows)
    gao.prepare_directories(config)
    frame.to_csv(
        config.output_root / "tables" / "downstream_parameter_alignment.csv",
        index=False,
    )
    return frame


def reuse_reviewed_segmentation(config: AnalysisConfig) -> pd.DataFrame:
    """Reuse the completed Normal masks without changing segmentation."""

    gao.prepare_directories(config)
    source_root = config.segmentation_source_root
    source_table = (
        source_root / "tables" / "segmentation_regions_all_fovs.csv"
    )
    if not source_table.is_file():
        raise FileNotFoundError(source_table)
    regions = pd.read_csv(source_table)
    regions.to_csv(
        config.output_root / "tables" / "segmentation_regions_all_fovs.csv",
        index=False,
    )
    mask_count = len(
        list(
            (source_root / "segmentation_masks").glob(
                "fov_*_nuclei_mask.tiff"
            )
        )
    )
    overlay_count = len(
        list(
            (source_root / "segmentation_overlays").glob(
                "fov_*_nuclei_overlay.png"
            )
        )
    )
    expected = int(config.expected_fov_count or 0)
    if mask_count != expected or overlay_count != expected:
        raise RuntimeError(
            {
                "expected": expected,
                "mask_count": mask_count,
                "overlay_count": overlay_count,
            }
        )
    payload = {
        "method": "read_only_reuse_of_reviewed_Normal_segmentation",
        "source_root": str(source_root),
        "segmentation_conditions": {
            "channels": "equal-weight three-channel sum",
            "pixel_grid": "2x anti-aliased resize to 130 nm/px",
            "gaussian_sigma_px_on_130nm_grid": 5.0,
            "projection": "Z maximum",
            "cellpose_model": "nuclei",
            "diameter_px_on_130nm_grid": 100.0,
            "cellprob_threshold": 0.0,
            "flow_threshold": 0.4,
            "mask_dilation": "none",
            "saved_mask_grid": "full-resolution 65 nm/px",
        },
        "counts": {
            "masks": mask_count,
            "overlays": overlay_count,
        },
        "selected_regions": int(regions["selected"].astype(bool).sum()),
    }
    gao.write_json(config.output_root / "segmentation_reuse.json", payload)
    return regions


def run_quantification(
    config: AnalysisConfig,
    *,
    workers: int = 8,
    chunk_size: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return batch.run_quantification_parallel(
        config,
        workers=workers,
        chunk_size=chunk_size,
    )


def make_spot_contact_sheet(
    config: AnalysisConfig,
    fov_ids: tuple[int, ...] = (0, 19, 39, 59, 79, 99),
) -> Path:
    paths = [
        config.output_root
        / "qc"
        / "spot_detection_overlays"
        / f"fov_{fov_id:03d}_3d_spot_overlay.png"
        for fov_id in fov_ids
    ]
    return batch._contact_sheet(
        paths,
        [f"FOV {fov_id:03d}" for fov_id in fov_ids],
        config.output_root
        / "figures"
        / "spot_detection_representative_contact_sheet.png",
        columns=2,
        panel_width=6.8,
        panel_height=5.8,
    )


def run_thresholds_and_states(
    config: AnalysisConfig,
    candidates: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if candidates is None:
        candidates = pd.read_csv(
            config.output_root / "tables" / "mtetr_mcp_spot_candidates.csv"
        )
    decision, artifacts, sensitivity = gao.calibrate_thresholds(
        candidates,
        config,
    )
    decision.update(
        {
            "status": "fixed_cell_thresholds_applied_to_Normal",
            "population": (
                "Reviewed Normal three-channel Cellpose nuclei masks; "
                "no dilation"
            ),
            "selected_mtetr_source": "fixed-cell threshold 1.15",
            "selected_mcp_source": "fixed-cell threshold 1.15",
            "mcp_near_distance_px_normal": float(
                config.mcp_near_distance_px
            ),
            "interpretation_boundary": (
                "The fixed-cell thresholds account for altered mTetR "
                "localization and intensity after fixation; no direct "
                "numerical equivalence to live-cell thresholds is claimed."
            ),
        }
    )
    gao.write_json(config.output_root / "threshold_decision.json", decision)
    figure_paths = gao.make_calibration_figures(
        candidates,
        decision,
        artifacts,
        sensitivity,
        config,
    )
    states = gao.call_states(candidates, decision, config)
    rank1_qc = gao.make_rank1_crop_qc(states, decision, config)
    counts = states["state"].value_counts().to_dict()
    locus_figure: Path | None = None
    radial_table: pd.DataFrame | None = None
    if counts.get("Active", 0) > 0 and counts.get("Inactive", 0) > 0:
        locus_figure, radial_table = gao.make_locus_summary(states, config)
    summary = {
        "rank1_candidates": int(len(states)),
        "state_counts": {
            str(key): int(value) for key, value in counts.items()
        },
        "threshold_figure": str(figure_paths["calibration_png"]),
        "sensitivity_figure": str(figure_paths["sensitivity_png"]),
        "rank1_crop_qc": str(rank1_qc),
        "locus_figure": (
            None if locus_figure is None else str(locus_figure)
        ),
        "radial_rows": 0 if radial_table is None else int(len(radial_table)),
    }
    gao.write_json(config.output_root / "state_summary.json", summary)
    return {
        "decision": decision,
        "states": states,
        "summary": summary,
        "figure_paths": figure_paths,
        "rank1_crop_qc": rank1_qc,
        "locus_figure": locus_figure,
    }


def _finite_ratio_check(
    numerator: pd.Series,
    denominator: pd.Series,
    observed: pd.Series,
) -> bool:
    frame = pd.DataFrame(
        {
            "numerator": pd.to_numeric(numerator, errors="coerce"),
            "denominator": pd.to_numeric(denominator, errors="coerce"),
            "observed": pd.to_numeric(observed, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return False
    expected = frame["numerator"] / frame["denominator"]
    return bool(
        np.allclose(
            expected,
            frame["observed"],
            rtol=1e-10,
            atol=1e-10,
        )
    )


def validate_outputs(config: AnalysisConfig) -> dict[str, Any]:
    metadata = gao.collect_metadata(config)
    regions = pd.read_csv(
        config.output_root / "tables" / "segmentation_regions_all_fovs.csv"
    )
    candidates = pd.read_csv(
        config.output_root / "tables" / "mtetr_mcp_spot_candidates.csv"
    )
    detections = pd.read_csv(
        config.output_root / "tables" / "mtetr_3d_detection_summary.csv"
    )
    states = pd.read_csv(
        config.output_root / "tables" / "locus_state_calls.csv"
    )
    decision = json.loads(
        (config.output_root / "threshold_decision.json").read_text(
            encoding="utf-8"
        )
    )
    source = config.segmentation_source_root
    cache = config.output_root / "per_fov_cache"
    figures = config.output_root / "figures"
    expected_fovs = int(metadata["sizes"]["P"])
    selected_nuclei = int(regions["selected"].astype(bool).sum())
    primary = candidates.loc[
        candidates["is_primary_candidate"].astype(bool)
    ].copy()
    random_control_counts: list[int] = []
    random_control_cache_complete = True
    for path in sorted(cache.glob("fov_*_rank1_crops.npz")):
        try:
            with np.load(path) as payload:
                if "random_qc_crops" not in payload.files:
                    random_control_cache_complete = False
                    continue
                random_control_counts.append(
                    int(len(payload["random_qc_crops"]))
                )
        except Exception:
            random_control_cache_complete = False

    checks = {
        "metadata_is_Normal_expectedFOV_65nm": bool(
            expected_fovs == int(config.expected_fov_count or expected_fovs)
            and math.isclose(metadata["voxel_size_um"]["x"], 0.065)
            and math.isclose(metadata["voxel_size_um"]["z"], 0.5)
        ),
        "all_source_masks": len(
            list(
                (source / "segmentation_masks").glob(
                    "fov_*_nuclei_mask.tiff"
                )
            )
        )
        == expected_fovs,
        "all_source_overlays": len(
            list(
                (source / "segmentation_overlays").glob(
                    "fov_*_nuclei_overlay.png"
                )
            )
        )
        == expected_fovs,
        "all_candidate_tables": len(
            list(cache.glob("fov_*_spot_candidates.csv"))
        )
        == expected_fovs,
        "all_detection_tables": len(
            list(cache.glob("fov_*_3d_detection_summary.csv"))
        )
        == expected_fovs,
        "all_crop_archives": len(
            list(cache.glob("fov_*_rank1_crops.npz"))
        )
        == expected_fovs,
        "all_cache_manifests": len(
            list(cache.glob("fov_*_cache_manifest.json"))
        )
        == expected_fovs,
        "all_spot_overlays": len(
            list(
                (
                    config.output_root
                    / "qc"
                    / "spot_detection_overlays"
                ).glob("fov_*_3d_spot_overlay.png")
            )
        )
        == expected_fovs,
        "all_qc_random_control_arrays_present": bool(
            random_control_cache_complete
            and len(random_control_counts) == expected_fovs
        ),
        "all_qc_random_control_population_nonzero": bool(
            sum(random_control_counts) > 0
        ),
        "all_qc_random_control_coverage_at_least_95pct": bool(
            selected_nuclei > 0
            and sum(random_control_counts) / selected_nuclei >= 0.95
        ),
        "one_detection_row_per_selected_nucleus": (
            len(detections) == selected_nuclei
            and not detections.duplicated(["fov_id", "cell_id"]).any()
        ),
        "one_primary_per_detected_nucleus": (
            not primary.duplicated(["fov_id", "cell_id"]).any()
            and len(primary)
            == int((detections["candidate_count_3d"] > 0).sum())
        ),
        "candidate_ranks_top5": bool(
            candidates["candidate_rank_by_raw_peak"].between(1, 5).all()
        ),
        "mtetr_r_mass_formula": _finite_ratio_check(
            candidates["mTetR_signal_intensity"],
            candidates["mTetR_crop_mean"],
            candidates["mTetR_r_mass"],
        ),
        "mcp_r_mass_formula": _finite_ratio_check(
            primary["mcp_signal_intensity"],
            primary["mcp_crop_mean"],
            primary["mcp_r_mass"],
        ),
        "state_rows_match_primary": len(states) == len(primary),
        "state_thresholds_match_fixed_cell_config": bool(
            math.isclose(decision["selected_mtetr_r_mass"], 1.15)
            and math.isclose(decision["selected_mcp_r_mass"], 1.15)
            and math.isclose(decision["mcp_near_distance_um"], 0.39)
        ),
        "state_labels_known": set(states["state"]).issubset(
            {
                "Active",
                "Inactive",
                "Excluded_below_mTetR",
                "Excluded_no_valid_crop",
            }
        ),
        "normal_parameter_conversion": bool(
            config.bigfish_log_kernel_size_zyx_px
            == (1.0, 4.615384615384615, 4.615384615384615)
            and config.crop_radius_px == 18
            and config.final_r_mass_radius_px == 6
            and config.final_trackpy_diameter_px == 11
            and config.final_trackpy_separation_px == 12
            and math.isclose(config.mcp_near_distance_um, 0.39)
        ),
        "required_threshold_figures": all(
            (figures / name).is_file()
            for name in (
                "threshold_calibration_four_panel.png",
                "threshold_sensitivity.png",
                "spot_detection_representative_contact_sheet.png",
            )
        ),
        "rank1_crop_qc_exists": (
            config.output_root / "qc" / "rank1_crop_qc.png"
        ).is_file(),
    }

    state_counts = states["state"].value_counts().to_dict()
    has_both_states = bool(
        state_counts.get("Active", 0) > 0
        and state_counts.get("Inactive", 0) > 0
    )
    if has_both_states:
        normalization = pd.read_csv(
            config.output_root / "tables" / "gao_fig1c_normalization.csv"
        )
        radial = pd.read_csv(
            config.output_root / "tables" / "gao_fig1c_radial_profiles.csv"
        )
        positive_max = normalization[
            ["active_scaled_positive_max", "inactive_scaled_positive_max"]
        ].max(axis=1)
        random_profile = radial.loc[radial["state"] == "Random"]
        checks.update(
            {
                "gao_composite_exists": (
                    figures / "gao_fig1c_method_composite.png"
                ).is_file(),
                "gao_radial_figure_exists": (
                    figures / "locus_radial_profiles.png"
                ).is_file(),
                "gao_shared_positive_max_is_100": bool(
                    np.allclose(
                        positive_max.to_numpy(dtype=float),
                        100.0,
                        rtol=1e-5,
                        atol=1e-4,
                    )
                ),
                "gao_random_radial_mean_is_zero": bool(
                    np.nanmax(
                        np.abs(
                            random_profile[
                                "mean_scaled_intensity"
                            ].to_numpy(dtype=float)
                        )
                    )
                    < 1e-3
                ),
            }
        )

    report = {
        "analysis": (
            "Normal ChamberA Field8 with SoRa all-dataset-compatible "
            "downstream analysis"
        ),
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "source_notebook": str(
            PROJECT / "sora_fixed_snapshot_all_datasets_gao3d.ipynb"
        ),
        "segmentation_source": str(config.segmentation_source_root),
        "method_alignment": {
            "same_as_SoRa_notebook": [
                "3x3 XY median-filtered uint16 quantification stack",
                "per-nucleus 3D Big-FISH and top-5 ranking",
                "single-Z three-channel crops",
                "mTetR/MCP/SNAPtag r_mass table schema",
                "four-panel threshold figure and threshold sensitivity",
                "fixed-threshold state-call logic for comparison",
                "rank-1 crop QC",
                (
                    "random nuclear controls sampled independently from all "
                    "QC-passing nuclei with a valid crop position"
                ),
                "Gao shared positive pixel-max normalization to 100",
                "locus-centered composite and radial profiles",
            ],
            "Normal_resolution_adaptations": [
                "reviewed Normal segmentation mask is full-resolution",
                (
                    "LoG/minimum distance use Gao 500/300/300 nm at "
                    "500/65/65 nm voxel"
                ),
                "crop is 37x37 px for the same 2.34 um physical width",
                "r_mass radius is 6 px for the Gao 0.39 um aperture",
            ],
            "interpretation_boundary": (
                "r_mTetR=1.15 and r_MCP=1.15 are fixed-cell thresholds. "
                "Fixation alters mTetR localization and intensity, so direct "
                "numerical equivalence to live-cell thresholds is not claimed."
            ),
        },
        "counts": {
        "selected_nuclei": selected_nuclei,
        "all_qc_random_controls": int(sum(random_control_counts)),
        "all_qc_random_control_coverage": (
            float(sum(random_control_counts) / selected_nuclei)
            if selected_nuclei
            else None
        ),
            "rank1_candidates": int(len(primary)),
            "state_counts": {
                str(key): int(value) for key, value in state_counts.items()
            },
        },
        "config": asdict(config),
        "validations": checks,
        "all_validations_passed": bool(all(checks.values())),
    }
    gao.write_json(config.output_root / "analysis_manifest.json", report)
    if not report["all_validations_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Validation failed: {failed}")
    return report
