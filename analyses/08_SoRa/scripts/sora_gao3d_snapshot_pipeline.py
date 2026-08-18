#!/usr/bin/env python3
"""Gao-compatible 3D snapshot analysis for fixed-cell SoRa data.

This module deliberately reuses the already-reviewed nuclear segmentation
from ``sora_snapshot_pipeline`` while replacing the old mTetR MIP candidate
step with Gao's 3D Big-FISH LoG/local-maximum/threshold workflow.  Detection,
single-Z crops, r_mass measurements, and random nuclear controls all use the
same 3x3 XY median-filtered uint16 data path as Gao's ``processed_1``.
Big-FISH is run once per nuclear bounding box with the exact final condition;
this preserves the candidate ordering used in the FOV 001-010 sweep while
avoiding the other 17 sensitivity conditions.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import bigfish
import bigfish.detection as bigfish_detection
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from scipy.ndimage import median_filter
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

import sora_snapshot_pipeline as legacy


tp.quiet(suppress=True)

ALGORITHM_VERSION = (
    "per_nucleus_bbox_bigfish_all_qc_random_controls_rmass6px_v4"
)


@dataclass(frozen=True)
class AnalysisConfig(legacy.AnalysisConfig):
    """Final parameters fixed by bead fitting and FOV 001-010 sensitivity QC."""

    output_name: str = "sora_snapshot_outputs_chamberA_field1_gao3d_final"
    segmentation_source_name: str = (
        "sora_snapshot_outputs_chamberA_field1_final_"
        "aa4x_sigma11_d110_cp05_dilate15_gao"
    )
    median_kernel_zyx: tuple[int, int, int] = (1, 3, 3)
    central_z_index: int = 5
    bigfish_log_kernel_size_zyx_px: tuple[float, float, float] = (0.5, 4.0, 4.0)
    bigfish_minimum_distance_zyx_px: tuple[float, float, float] = (1.0, 4.0, 4.0)
    bigfish_threshold_divisor: float = 20.0
    final_crop_radius_px: int = 50
    # The r_mass aperture is resolution-adapted independently of the
    # mTetR-MCP colocalization cutoff. Six SoRa pixels correspond to a
    # 0.139-um radius (0.279-um diameter).
    final_r_mass_radius_px: int = 6
    final_trackpy_diameter_px: int = 11
    final_trackpy_separation_px: int = 12
    fixed_mtetr_r_mass_threshold: float = 1.045
    fixed_mcp_r_mass_threshold: float = 1.100
    mtetr_plot_comparison_thresholds: tuple[float, ...] = (1.015, 1.030)
    top_n_mtetr_candidates: int = 5
    random_z_min_index: int = 4
    random_z_max_index: int = 7
    random_seed: int = 20260719
    require_reference_equivalence: bool = True

    @property
    def segmentation_source_root(self) -> Path:
        return self.project_dir / self.segmentation_source_name

    @property
    def crop_radius_px(self) -> int:
        return int(self.final_crop_radius_px)

    @property
    def crop_radius_um(self) -> float:
        return float(self.final_crop_radius_px * self.expected_xy_um)

    @property
    def r_mass_pad_px(self) -> int:
        return int(self.final_r_mass_radius_px)

    @property
    def r_mass_pad_um(self) -> float:
        return float(self.final_r_mass_radius_px * self.expected_xy_um)

    @property
    def spot_diameter_px(self) -> int:
        return int(self.final_trackpy_diameter_px)

    @property
    def spot_diameter_um(self) -> float:
        return float(self.final_trackpy_diameter_px * self.expected_xy_um)


prepare_directories = legacy.prepare_directories
collect_metadata = legacy.collect_metadata
config_manifest = legacy.config_manifest
write_json = legacy.write_json
robust_normalize = legacy.robust_normalize
centered_crop = legacy.centered_crop
gao_r_mass = legacy.gao_r_mass
_upsample_mask = legacy._upsample_mask
_choose_random_center = legacy._choose_random_center


CANDIDATE_COLUMNS = [
    "fov_id",
    "cell_id",
    "locus_z_index",
    "locus_y_px",
    "locus_x_px",
    "locus_y_um",
    "locus_x_um",
    "candidate_rank_by_raw_peak",
    "is_primary_candidate",
    "candidate_count_3d",
    "bigfish_threshold",
    "mTetR_local_peak_raw_intensity",
    "mTetR_r_mass",
    "mTetR_signal_intensity",
    "mTetR_crop_mean",
    "locus_z_edge",
    "SNAP_projection_region_mean",
    "mTetR_projection_region_mean",
    "MCP_projection_region_mean",
    "mcp_y_crop_px",
    "mcp_x_crop_px",
    "mcp_r_mass",
    "mcp_signal_intensity",
    "mcp_crop_mean",
    "mcp_distance_px",
    "mcp_distance_um",
    "mcp_refinement",
    "snap_y_crop_px",
    "snap_x_crop_px",
    "snap_r_mass",
    "snap_signal_intensity",
    "snap_crop_mean",
    "snap_distance_px",
    "snap_distance_um",
    "snap_refinement",
    "random_crop_available",
    "random_z_index",
    "random_y_px",
    "random_x_px",
]

EXCLUSION_COLUMNS = [
    "fov_id",
    "cell_id",
    "candidate_rank",
    "reason",
]

DETECTION_COLUMNS = [
    "fov_id",
    "cell_id",
    "bigfish_threshold",
    "candidate_count_3d",
    "primary_z",
    "primary_y",
    "primary_x",
    "primary_intensity",
    "primary_z_edge",
]


def run_segmentation(config: AnalysisConfig) -> pd.DataFrame:
    """Reuse the reviewed 4x Cellpose nuclei masks without re-running Cellpose."""

    prepare_directories(config)
    source_table = (
        config.segmentation_source_root
        / "tables"
        / "segmentation_regions_all_fovs.csv"
    )
    if not source_table.is_file():
        raise FileNotFoundError(f"Missing segmentation source table: {source_table}")
    regions = pd.read_csv(source_table)
    regions.to_csv(
        config.output_root / "tables" / "segmentation_regions_all_fovs.csv",
        index=False,
    )
    source_masks = config.segmentation_source_root / "segmentation_masks"
    source_pre_masks = (
        config.segmentation_source_root / "segmentation_masks_pre_dilation"
    )
    source_overlays = (
        config.segmentation_source_root / "segmentation_overlays"
    )
    counts = {
        "masks": len(list(source_masks.glob("fov_*_selected_mask.tiff"))),
        "pre_dilation_masks": len(
            list(source_pre_masks.glob("fov_*_selected_mask_pre_dilation.tiff"))
        ),
        "overlays": len(
            list(source_overlays.glob("fov_*_selected_regions.png"))
        ),
    }
    payload = {
        "method": "read_only_reuse",
        "source_root": str(config.segmentation_source_root),
        "segmentation_conditions": {
            "downsample": "4x anti-aliased bilinear",
            "background_subtraction_sigma_px_full": 30,
            "signal_gaussian_sigma_px_full": 11,
            "cellpose_model": "nuclei",
            "diameter_px_4x": 110,
            "cellprob_threshold": 0.5,
            "flow_threshold": 0.4,
            "label_aware_dilation_px_4x": 15,
        },
        "counts": counts,
        "selected_regions": int(regions["selected"].astype(bool).sum()),
    }
    write_json(config.output_root / "segmentation_reuse.json", payload)
    with nd2.ND2File(config.input_nd2) as handle:
        expected_fovs = int(handle.sizes["P"])
    if not all(value == expected_fovs for value in counts.values()):
        raise RuntimeError(f"Incomplete segmentation source: {counts}")
    return regions


def _load_channel(
    dask_array: Any,
    fov_id: int,
    channel_index: int,
    median_kernel_zyx: tuple[int, int, int],
) -> np.ndarray:
    volume = np.asarray(
        dask_array[int(fov_id), :, int(channel_index)].compute(),
        dtype=np.uint16,
    )
    return np.asarray(
        median_filter(volume, size=median_kernel_zyx),
        dtype=np.uint16,
    )


def _mask_path(config: AnalysisConfig, fov_id: int) -> Path:
    filename_template = getattr(
        config,
        "segmentation_mask_filename_template",
        "fov_{fov_id:03d}_selected_mask.tiff",
    )
    return (
        config.segmentation_source_root
        / "segmentation_masks"
        / filename_template.format(fov_id=int(fov_id))
    )


def _random_control_mask_path(
    config: AnalysisConfig,
    fov_id: int,
) -> Path:
    """Prefer the reviewed pre-dilation mask for random nuclear controls."""

    pre_dilation = (
        config.segmentation_source_root
        / "segmentation_masks_pre_dilation"
        / f"fov_{int(fov_id):03d}_selected_mask_pre_dilation.tiff"
    )
    return pre_dilation if pre_dilation.is_file() else _mask_path(config, fov_id)


def _cache_path(config: AnalysisConfig, fov_id: int, suffix: str) -> Path:
    return (
        config.output_root
        / "per_fov_cache"
        / f"fov_{fov_id:03d}_{suffix}"
    )


def _empty_crop_payload(config: AnalysisConfig) -> dict[str, np.ndarray]:
    crop_size = 2 * config.crop_radius_px + 1
    return {
        "cell_ids": np.empty(0, dtype=np.int32),
        "locus_crops": np.empty(
            (0, 3, crop_size, crop_size), dtype=np.uint16
        ),
        "random_crops": np.empty(
            (0, 3, crop_size, crop_size), dtype=np.uint16
        ),
        "random_qc_cell_ids": np.empty(0, dtype=np.int32),
        "random_qc_z_indices": np.empty(0, dtype=np.int16),
        "random_qc_y_px": np.empty(0, dtype=np.int32),
        "random_qc_x_px": np.empty(0, dtype=np.int32),
        "random_qc_crops": np.empty(
            (0, 3, crop_size, crop_size), dtype=np.uint16
        ),
    }


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fov_cache_ready(
    required: list[Path],
    crops_path: Path,
    manifest_path: Path,
) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("algorithm_version") != ALGORITHM_VERSION:
            return False
        with np.load(crops_path) as data:
            required_arrays = {
                "cell_ids",
                "locus_crops",
                "random_crops",
                "random_qc_cell_ids",
                "random_qc_z_indices",
                "random_qc_y_px",
                "random_qc_x_px",
                "random_qc_crops",
            }
            return required_arrays.issubset(set(data.files))
    except Exception:
        return False


def _channel_top1(
    image: np.ndarray,
    *,
    channel: str,
    diameter: int,
    separation: int,
    r_mass_radius: int,
    crop_radius: int,
    pixel_um: float,
) -> dict[str, float | bool | str]:
    """Gao-style trackpy detection with explicit SoRa feature scales."""

    prefix = "mcp" if channel == "MCP" else "snap"
    result: dict[str, float | bool | str] = {
        f"{prefix}_y_crop_px": np.nan,
        f"{prefix}_x_crop_px": np.nan,
        f"{prefix}_r_mass": np.nan,
        f"{prefix}_signal_intensity": np.nan,
        f"{prefix}_crop_mean": float(np.mean(image)),
        f"{prefix}_distance_px": np.nan,
        f"{prefix}_distance_um": np.nan,
        f"{prefix}_refinement": "not_run",
    }
    features = tp.locate(
        np.asarray(image, dtype=np.float32),
        diameter=int(diameter),
        separation=int(separation),
        topn=1 if channel == "MCP" else None,
    )
    if features is None or features.empty:
        return result
    if channel == "SNAP":
        distances = np.hypot(
            features["y"].to_numpy() - crop_radius,
            features["x"].to_numpy() - crop_radius,
        )
        selected = features.iloc[[int(np.argmin(distances))]].copy()
    else:
        selected = features.iloc[[0]].copy()
    try:
        refined = tp.refine_leastsq(
            selected,
            np.asarray(image, dtype=np.float32),
            int(diameter),
            fit_function="gauss",
        ).reset_index(drop=True)
        refinement = "gaussian"
    except Exception:
        refined = selected.reset_index(drop=True)
        refinement = "locate_only"
    y_px = float(refined.loc[0, "y"])
    x_px = float(refined.loc[0, "x"])
    intensity = gao_r_mass(
        np.asarray([[y_px, x_px]], dtype=float),
        image,
        int(r_mass_radius),
    )
    distance_px = float(np.hypot(y_px - crop_radius, x_px - crop_radius))
    result.update(
        {
            f"{prefix}_y_crop_px": y_px,
            f"{prefix}_x_crop_px": x_px,
            f"{prefix}_r_mass": (
                np.nan if intensity is None else float(intensity["r_mass"])
            ),
            f"{prefix}_signal_intensity": (
                np.nan
                if intensity is None
                else float(intensity["signal_intensity"])
            ),
            f"{prefix}_crop_mean": (
                float(np.mean(image))
                if intensity is None
                else float(intensity["crop_mean"])
            ),
            f"{prefix}_distance_px": distance_px,
            f"{prefix}_distance_um": distance_px * float(pixel_um),
            f"{prefix}_refinement": refinement,
        }
    )
    return result


def _detect_nuclear_spots(
    mtetr_volume: np.ndarray,
    full_mask: np.ndarray,
    mask_small: np.ndarray,
    fov_id: int,
    config: AnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the exact sweep/Gao Big-FISH path once per nuclear bounding box."""

    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    region_lookup = {int(region.label): region for region in regionprops(mask_small)}
    padding = int(
        math.ceil(4 * max(config.bigfish_log_kernel_size_zyx_px[1:]))
    )

    for cell_id in range(1, int(mask_small.max()) + 1):
        region_pixels = full_mask == cell_id
        threshold = float(
            mtetr_volume[config.central_z_index][region_pixels].mean()
            / config.bigfish_threshold_divisor
        )
        region = region_lookup[cell_id]
        small_min_y, small_min_x, small_max_y, small_max_x = region.bbox
        min_y = max(
            0,
            int(small_min_y * config.segmentation_downsample) - padding,
        )
        min_x = max(
            0,
            int(small_min_x * config.segmentation_downsample) - padding,
        )
        max_y = min(
            mtetr_volume.shape[-2],
            int(small_max_y * config.segmentation_downsample) + padding,
        )
        max_x = min(
            mtetr_volume.shape[-1],
            int(small_max_x * config.segmentation_downsample) + padding,
        )
        local_volume = mtetr_volume[:, min_y:max_y, min_x:max_x]
        spots = bigfish_detection.detect_spots(
            images=local_volume,
            threshold=threshold,
            return_threshold=False,
            remove_duplicate=True,
            log_kernel_size=config.bigfish_log_kernel_size_zyx_px,
            minimum_distance=config.bigfish_minimum_distance_zyx_px,
        )
        spots = np.asarray(spots, dtype=int).reshape((-1, 3))
        if len(spots):
            spots[:, 1] += min_y
            spots[:, 2] += min_x
            inside = (
                full_mask[
                    np.clip(spots[:, 1], 0, full_mask.shape[0] - 1),
                    np.clip(spots[:, 2], 0, full_mask.shape[1] - 1),
                ]
                == cell_id
            )
            spots = spots[inside]
        if len(spots) == 0:
            exclusions.append(
                {
                    "fov_id": int(fov_id),
                    "cell_id": int(cell_id),
                    "candidate_rank": np.nan,
                    "reason": "no_3d_mTetR_spot",
                }
            )
            detection_rows.append(
                {
                    "fov_id": int(fov_id),
                    "cell_id": int(cell_id),
                    "bigfish_threshold": threshold,
                    "candidate_count_3d": 0,
                    "primary_z": np.nan,
                    "primary_y": np.nan,
                    "primary_x": np.nan,
                    "primary_intensity": np.nan,
                    "primary_z_edge": False,
                }
            )
            continue

        intensities = mtetr_volume[spots[:, 0], spots[:, 1], spots[:, 2]]
        # Gao ranks spots by the median-filtered mTetR center intensity.
        # Big-FISH's connected-component ordering can differ across
        # Linux/macOS for equal-intensity maxima.  Preserve Gao's effective
        # first-coordinate behavior deterministically: intensity descending,
        # then z, y, x ascending.
        order = np.lexsort(
            (
                spots[:, 2],
                spots[:, 1],
                spots[:, 0],
                -intensities.astype(np.float64),
            )
        )
        spots = spots[order]
        intensities = intensities[order]
        primary_z, primary_y, primary_x = map(int, spots[0])
        detection_rows.append(
            {
                "fov_id": int(fov_id),
                "cell_id": int(cell_id),
                "bigfish_threshold": threshold,
                "candidate_count_3d": int(len(spots)),
                "primary_z": primary_z,
                "primary_y": primary_y,
                "primary_x": primary_x,
                "primary_intensity": float(intensities[0]),
                "primary_z_edge": bool(
                    primary_z == 0
                    or primary_z == mtetr_volume.shape[0] - 1
                ),
            }
        )
        for rank, (spot, intensity) in enumerate(
            zip(
                spots[: config.top_n_mtetr_candidates],
                intensities[: config.top_n_mtetr_candidates],
            ),
            start=1,
        ):
            z_index, locus_y, locus_x = map(int, spot)
            crop = centered_crop(
                mtetr_volume[z_index][None, ...],
                locus_y,
                locus_x,
                config.crop_radius_px,
            )
            mtetr_metrics = None
            if crop is None:
                exclusions.append(
                    {
                        "fov_id": int(fov_id),
                        "cell_id": int(cell_id),
                        "candidate_rank": int(rank),
                        "reason": "candidate_crop_out_of_frame",
                    }
                )
            else:
                try:
                    mtetr_metrics = gao_r_mass(
                        np.asarray(
                            [[config.crop_radius_px, config.crop_radius_px]],
                            dtype=float,
                        ),
                        crop[0],
                        config.r_mass_pad_px,
                    )
                except Exception as exc:
                    exclusions.append(
                        {
                            "fov_id": int(fov_id),
                            "cell_id": int(cell_id),
                            "candidate_rank": int(rank),
                            "reason": (
                                "mtetr_trackpy_error:"
                                f"{type(exc).__name__}"
                            ),
                        }
                    )
            row = {column: np.nan for column in CANDIDATE_COLUMNS}
            row.update(
                {
                    "fov_id": int(fov_id),
                    "cell_id": int(cell_id),
                    "locus_z_index": z_index,
                    "locus_y_px": locus_y,
                    "locus_x_px": locus_x,
                    "locus_y_um": locus_y * config.expected_xy_um,
                    "locus_x_um": locus_x * config.expected_xy_um,
                    "candidate_rank_by_raw_peak": int(rank),
                    "is_primary_candidate": bool(rank == 1),
                    "candidate_count_3d": int(len(spots)),
                    "bigfish_threshold": threshold,
                    "mTetR_local_peak_raw_intensity": float(intensity),
                    "mTetR_r_mass": (
                        np.nan
                        if mtetr_metrics is None
                        else float(mtetr_metrics["r_mass"])
                    ),
                    "mTetR_signal_intensity": (
                        np.nan
                        if mtetr_metrics is None
                        else float(mtetr_metrics["signal_intensity"])
                    ),
                    "mTetR_crop_mean": (
                        np.nan
                        if mtetr_metrics is None
                        else float(mtetr_metrics["crop_mean"])
                    ),
                    "locus_z_edge": bool(
                        z_index == 0
                        or z_index == mtetr_volume.shape[0] - 1
                    ),
                    "mcp_refinement": "not_run",
                    "snap_refinement": "not_run",
                    "random_crop_available": False,
                }
            )
            candidates.append(row)
    return candidates, exclusions, detection_rows


def _save_spot_overlay(
    config: AnalysisConfig,
    fov_id: int,
    mtetr_volume: np.ndarray,
    mask_small: np.ndarray,
    candidates: pd.DataFrame,
) -> Path:
    output_dir = config.output_root / "qc" / "spot_detection_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fov_{fov_id:03d}_3d_spot_overlay.png"
    factor = config.segmentation_downsample
    expected_shape = (
        mask_small.shape[0] * factor,
        mask_small.shape[1] * factor,
    )
    if mtetr_volume.shape[-2:] == expected_shape:
        mip = mtetr_volume.max(axis=0).astype(np.float32)
        low, high = np.percentile(mip, (1, 99.8))
        display_image = np.clip((mip - low) / max(high - low, 1), 0, 1)
        display_small = display_image[::factor, ::factor]
    else:
        display_small = np.zeros(mask_small.shape, dtype=np.float32)
    fig, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    axis.imshow(display_small, cmap="gray", origin="upper")
    boundary = find_boundaries(mask_small, mode="outer")
    axis.contour(boundary.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
    primary = candidates.loc[
        candidates["is_primary_candidate"].astype(bool)
    ]
    if len(primary):
        scatter = axis.scatter(
            primary["locus_x_px"] / factor,
            primary["locus_y_px"] / factor,
            c=primary["locus_z_index"],
            cmap="turbo",
            vmin=0,
            vmax=mtetr_volume.shape[0] - 1,
            s=42,
            edgecolors="white",
            linewidths=1.4,
            alpha=0.9,
        )
        colorbar = fig.colorbar(scatter, ax=axis, shrink=0.72)
        colorbar.set_label("primary mTetR Z index")
    axis.set_title(
        f"FOV {fov_id:03d}: median-filtered mTetR MIP\n"
        f"cyan=nuclear mask; circles=3D primary loci (n={len(primary)})"
    )
    axis.set_axis_off()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _process_one_fov(
    fov_id: int,
    dask_array: Any,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_path = _cache_path(config, fov_id, "spot_candidates.csv")
    exclusion_path = _cache_path(config, fov_id, "spot_exclusions.csv")
    detection_path = _cache_path(config, fov_id, "3d_detection_summary.csv")
    crops_path = _cache_path(config, fov_id, "rank1_crops.npz")
    manifest_path = _cache_path(config, fov_id, "cache_manifest.json")
    overlay_path = (
        config.output_root
        / "qc"
        / "spot_detection_overlays"
        / f"fov_{fov_id:03d}_3d_spot_overlay.png"
    )
    required = [
        candidate_path,
        exclusion_path,
        detection_path,
        crops_path,
        overlay_path,
        manifest_path,
    ]
    if (
        _fov_cache_ready(required, crops_path, manifest_path)
        and not config.overwrite_quantification
    ):
        return (
            pd.read_csv(candidate_path),
            pd.read_csv(exclusion_path),
            pd.read_csv(detection_path),
        )

    mask_small = tifffile.imread(_mask_path(config, fov_id))
    if int(mask_small.max()) == 0:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
        exclusions = pd.DataFrame(columns=EXCLUSION_COLUMNS)
        detections = pd.DataFrame(columns=DETECTION_COLUMNS)
        _write_csv_atomic(candidates, candidate_path)
        _write_csv_atomic(exclusions, exclusion_path)
        _write_csv_atomic(detections, detection_path)
        _write_npz_atomic(crops_path, _empty_crop_payload(config))
        blank = np.zeros((11, 16, 16), dtype=np.uint16)
        _save_spot_overlay(config, fov_id, blank, mask_small, candidates)
        _write_json_atomic(
            manifest_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "fov_id": int(fov_id),
                "n_nuclei": 0,
                "n_saved_candidates": 0,
                "complete": True,
            },
        )
        return candidates, exclusions, detections

    mtetr = _load_channel(
        dask_array,
        fov_id,
        config.mtetr_channel,
        config.median_kernel_zyx,
    )
    full_mask = _upsample_mask(
        mask_small,
        config.segmentation_downsample,
        (mtetr.shape[-2], mtetr.shape[-1]),
    )
    random_mask_small = tifffile.imread(
        _random_control_mask_path(config, fov_id)
    )
    random_full_mask = _upsample_mask(
        random_mask_small,
        config.segmentation_downsample,
        (mtetr.shape[-2], mtetr.shape[-1]),
    )
    candidate_rows, exclusion_rows, detection_rows = (
        _detect_nuclear_spots(
            mtetr,
            full_mask,
            mask_small,
            fov_id,
            config,
        )
    )
    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    exclusions = pd.DataFrame(exclusion_rows, columns=EXCLUSION_COLUMNS)
    detections = pd.DataFrame(detection_rows, columns=DETECTION_COLUMNS)
    _save_spot_overlay(config, fov_id, mtetr, mask_small, candidates)

    rng = np.random.default_rng(config.random_seed + int(fov_id))
    crop_size = 2 * config.crop_radius_px + 1

    # Generate one random control independently for every QC-passing nuclear
    # region. This population is deliberately independent of mTetR detection,
    # candidate ranking, and Active/Inactive state assignment.
    random_qc_cell_ids: list[int] = []
    random_qc_z_indices: list[int] = []
    random_qc_y_px: list[int] = []
    random_qc_x_px: list[int] = []
    random_qc_crops: list[np.ndarray] = []
    random_qc_population = [
        int(value)
        for value in np.unique(random_full_mask)
        if int(value) > 0
    ]
    for cell_id in random_qc_population:
        random_center = _choose_random_center(
            random_full_mask == cell_id,
            config.crop_radius_px,
            rng,
        )
        if random_center is None:
            exclusions.loc[len(exclusions)] = {
                "fov_id": int(fov_id),
                "cell_id": cell_id,
                "candidate_rank": 0,
                "reason": "all_qc_random_center_unavailable",
            }
            continue
        random_z = int(
            rng.integers(
                config.random_z_min_index,
                config.random_z_max_index + 1,
            )
        )
        random_mtetr = centered_crop(
            mtetr[random_z][None, ...],
            int(random_center[0]),
            int(random_center[1]),
            config.crop_radius_px,
        )
        if random_mtetr is None:
            exclusions.loc[len(exclusions)] = {
                "fov_id": int(fov_id),
                "cell_id": cell_id,
                "candidate_rank": 0,
                "reason": "all_qc_random_crop_out_of_frame",
            }
            continue
        random_crop = np.zeros(
            (3, crop_size, crop_size),
            dtype=np.uint16,
        )
        random_crop[config.mtetr_channel] = random_mtetr[0]
        random_qc_cell_ids.append(cell_id)
        random_qc_z_indices.append(random_z)
        random_qc_y_px.append(int(random_center[0]))
        random_qc_x_px.append(int(random_center[1]))
        random_qc_crops.append(random_crop)

    primary_rows = candidates.index[
        candidates["is_primary_candidate"].astype(bool)
    ].tolist()
    crop_cell_ids: list[int] = []
    locus_crops: list[np.ndarray] = []
    random_crops: list[np.ndarray] = []
    row_to_crop_index: dict[int, int] = {}

    for row_index in primary_rows:
        row = candidates.loc[row_index]
        if not np.isfinite(row["mTetR_crop_mean"]):
            continue
        z_index = int(row["locus_z_index"])
        locus_y = int(row["locus_y_px"])
        locus_x = int(row["locus_x_px"])
        mtetr_crop = centered_crop(
            mtetr[z_index][None, ...],
            locus_y,
            locus_x,
            config.crop_radius_px,
        )
        if mtetr_crop is None:
            continue
        cell_id = int(row["cell_id"])
        random_center = _choose_random_center(
            full_mask == cell_id,
            config.crop_radius_px,
            rng,
        )
        random_z: int | None = None
        random_mtetr: np.ndarray | None = None
        if random_center is None:
            exclusions.loc[len(exclusions)] = {
                "fov_id": int(fov_id),
                "cell_id": cell_id,
                "candidate_rank": 1,
                "reason": "random_center_unavailable",
            }
        else:
            random_z = int(
                rng.integers(
                    config.random_z_min_index,
                    config.random_z_max_index + 1,
                )
            )
            random_mtetr = centered_crop(
                mtetr[random_z][None, ...],
                int(random_center[0]),
                int(random_center[1]),
                config.crop_radius_px,
            )
            if random_mtetr is None:
                exclusions.loc[len(exclusions)] = {
                    "fov_id": int(fov_id),
                    "cell_id": cell_id,
                    "candidate_rank": 1,
                    "reason": "random_crop_out_of_frame",
                }
        crop_index = len(locus_crops)
        row_to_crop_index[int(row_index)] = crop_index
        crop_cell_ids.append(cell_id)
        locus = np.zeros((3, crop_size, crop_size), dtype=np.uint16)
        random = np.zeros_like(locus)
        locus[config.mtetr_channel] = mtetr_crop[0]
        if random_mtetr is not None:
            random[config.mtetr_channel] = random_mtetr[0]
        locus_crops.append(locus)
        random_crops.append(random)
        random_available = bool(
            random_center is not None
            and random_z is not None
            and random_mtetr is not None
        )
        candidates.loc[
            row_index, "random_crop_available"
        ] = random_available
        if random_available:
            candidates.loc[row_index, "random_z_index"] = int(random_z)
            candidates.loc[row_index, "random_y_px"] = int(random_center[0])
            candidates.loc[row_index, "random_x_px"] = int(random_center[1])

    for channel in (config.snap_channel, config.mcp_channel):
        volume = _load_channel(
            dask_array,
            fov_id,
            channel,
            config.median_kernel_zyx,
        )
        projection = volume.max(axis=0)
        for random_index, (random_z, random_y, random_x) in enumerate(
            zip(
                random_qc_z_indices,
                random_qc_y_px,
                random_qc_x_px,
            )
        ):
            random = centered_crop(
                volume[int(random_z)][None, ...],
                int(random_y),
                int(random_x),
                config.crop_radius_px,
            )
            if random is None:
                raise RuntimeError(
                    "Unexpected all-QC random crop failure "
                    f"FOV {fov_id} random index {random_index}"
                )
            random_qc_crops[random_index][channel] = random[0]
        for row_index, crop_index in row_to_crop_index.items():
            row = candidates.loc[row_index]
            z_index = int(row["locus_z_index"])
            locus_y = int(row["locus_y_px"])
            locus_x = int(row["locus_x_px"])
            locus = centered_crop(
                volume[z_index][None, ...],
                locus_y,
                locus_x,
                config.crop_radius_px,
            )
            if locus is None:
                raise RuntimeError(
                    f"Unexpected locus crop failure FOV {fov_id} row {row_index}"
                )
            locus_crops[crop_index][channel] = locus[0]
            if bool(row["random_crop_available"]):
                random_z = int(row["random_z_index"])
                random_y = int(row["random_y_px"])
                random_x = int(row["random_x_px"])
                random = centered_crop(
                    volume[random_z][None, ...],
                    random_y,
                    random_x,
                    config.crop_radius_px,
                )
                if random is None:
                    raise RuntimeError(
                        "Unexpected random crop failure "
                        f"FOV {fov_id} row {row_index}"
                    )
                random_crops[crop_index][channel] = random[0]
            region_values = projection[full_mask == int(row["cell_id"])]
            column = (
                "SNAP_projection_region_mean"
                if channel == config.snap_channel
                else "MCP_projection_region_mean"
            )
            candidates.loc[row_index, column] = float(region_values.mean())
        del volume

    mtetr_projection = mtetr.max(axis=0)
    for row_index, crop_index in row_to_crop_index.items():
        cell_id = int(candidates.loc[row_index, "cell_id"])
        candidates.loc[
            row_index, "mTetR_projection_region_mean"
        ] = float(mtetr_projection[full_mask == cell_id].mean())
        locus = locus_crops[crop_index]
        try:
            mcp_metrics = _channel_top1(
                locus[config.mcp_channel],
                channel="MCP",
                diameter=config.final_trackpy_diameter_px,
                separation=config.final_trackpy_separation_px,
                r_mass_radius=config.final_r_mass_radius_px,
                crop_radius=config.crop_radius_px,
                pixel_um=config.expected_xy_um,
            )
            for key, value in mcp_metrics.items():
                candidates.loc[row_index, key] = value
        except Exception as exc:
            exclusions.loc[len(exclusions)] = {
                "fov_id": int(fov_id),
                "cell_id": cell_id,
                "candidate_rank": 1,
                "reason": f"mcp_trackpy_error:{type(exc).__name__}",
            }
        try:
            snap_metrics = _channel_top1(
                locus[config.snap_channel],
                channel="SNAP",
                diameter=config.final_trackpy_diameter_px,
                separation=config.final_trackpy_separation_px,
                r_mass_radius=config.final_r_mass_radius_px,
                crop_radius=config.crop_radius_px,
                pixel_um=config.expected_xy_um,
            )
            for key, value in snap_metrics.items():
                candidates.loc[row_index, key] = value
        except Exception as exc:
            exclusions.loc[len(exclusions)] = {
                "fov_id": int(fov_id),
                "cell_id": cell_id,
                "candidate_rank": 1,
                "reason": f"snap_trackpy_error:{type(exc).__name__}",
            }

    random_qc_stack = (
        np.stack(random_qc_crops)
        if random_qc_crops
        else np.empty(
            (0, 3, crop_size, crop_size),
            dtype=np.uint16,
        )
    )
    if locus_crops:
        crop_payload = {
            "cell_ids": np.asarray(crop_cell_ids, dtype=np.int32),
            "locus_crops": np.stack(locus_crops),
            "random_crops": np.stack(random_crops),
            "random_qc_cell_ids": np.asarray(
                random_qc_cell_ids,
                dtype=np.int32,
            ),
            "random_qc_z_indices": np.asarray(
                random_qc_z_indices,
                dtype=np.int16,
            ),
            "random_qc_y_px": np.asarray(
                random_qc_y_px,
                dtype=np.int32,
            ),
            "random_qc_x_px": np.asarray(
                random_qc_x_px,
                dtype=np.int32,
            ),
            "random_qc_crops": random_qc_stack,
        }
    else:
        crop_payload = _empty_crop_payload(config)
        crop_payload.update(
            {
                "random_qc_cell_ids": np.asarray(
                    random_qc_cell_ids,
                    dtype=np.int32,
                ),
                "random_qc_z_indices": np.asarray(
                    random_qc_z_indices,
                    dtype=np.int16,
                ),
                "random_qc_y_px": np.asarray(
                    random_qc_y_px,
                    dtype=np.int32,
                ),
                "random_qc_x_px": np.asarray(
                    random_qc_x_px,
                    dtype=np.int32,
                ),
                "random_qc_crops": random_qc_stack,
            }
        )

    _write_csv_atomic(candidates, candidate_path)
    _write_csv_atomic(exclusions, exclusion_path)
    _write_csv_atomic(detections, detection_path)
    _write_npz_atomic(crops_path, crop_payload)
    _write_json_atomic(
        manifest_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "fov_id": int(fov_id),
            "n_nuclei": int(len(detections)),
            "n_saved_candidates": int(len(candidates)),
            "n_rank1_crops": int(len(crop_cell_ids)),
            "n_qc_nuclei_for_random_controls": int(
                len(random_qc_population)
            ),
            "n_all_qc_random_crops": int(len(random_qc_cell_ids)),
            "random_control_mask": str(
                _random_control_mask_path(config, fov_id)
            ),
            "complete": True,
        },
    )
    return candidates, exclusions, detections


def run_quantification(
    config: AnalysisConfig,
    fov_ids: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run restartable Gao 3D quantification for requested or all FOVs."""

    prepare_directories(config)
    metadata = collect_metadata(config)
    n_positions = int(metadata["sizes"]["P"])
    effective_fovs = (
        list(range(n_positions))
        if fov_ids is None
        else [int(value) for value in fov_ids]
    )
    candidate_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    detection_frames: list[pd.DataFrame] = []
    with nd2.ND2File(config.input_nd2) as handle:
        dask_array = handle.to_dask()
        for ordinal, fov_id in enumerate(effective_fovs, start=1):
            candidates, exclusions, detections = _process_one_fov(
                fov_id,
                dask_array,
                config,
            )
            if len(candidates):
                candidate_frames.append(candidates)
            if len(exclusions):
                exclusion_frames.append(exclusions)
            if len(detections):
                detection_frames.append(detections)
            if ordinal % 5 == 0 or ordinal == len(effective_fovs):
                print(
                    f"Gao 3D quantification: {ordinal}/{len(effective_fovs)} FOV; "
                    f"FOV {fov_id:03d}, nuclei={len(detections)}, "
                    f"saved candidates={len(candidates)}",
                    flush=True,
                )
    all_candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    all_exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames
        else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    )
    all_detections = (
        pd.concat(detection_frames, ignore_index=True)
        if detection_frames
        else pd.DataFrame(columns=DETECTION_COLUMNS)
    )
    if len(all_candidates):
        all_candidates = all_candidates.sort_values(
            ["fov_id", "cell_id", "candidate_rank_by_raw_peak"]
        ).reset_index(drop=True)
    tables = config.output_root / "tables"
    all_candidates.to_csv(
        tables / "mtetr_mcp_spot_candidates.csv",
        index=False,
    )
    all_exclusions.to_csv(
        tables / "spot_candidate_exclusions.csv",
        index=False,
    )
    all_detections.to_csv(
        tables / "mtetr_3d_detection_summary.csv",
        index=False,
    )
    return all_candidates, all_exclusions


def validate_fov001_010_equivalence(
    config: AnalysisConfig,
    reference_csv: Path | None = None,
) -> dict[str, Any]:
    """Require exact primary coordinates/counts versus the completed sweep."""

    if reference_csv is None:
        reference_csv = (
            config.project_dir
            / "sora_spot_parameter_sweep_fov001_010"
            / "tables"
            / "log_summary.csv"
        )
    reference = pd.read_csv(reference_csv)
    reference = reference.loc[
        reference["condition"] == "logz0p5_logxy4p0_mdxy4p0",
        [
            "fov_id",
            "cell_id",
            "candidate_count",
            "primary_z",
            "primary_y",
            "primary_x",
            "primary_intensity",
            "primary_r_mass_r6",
            "primary_crop_mean",
        ],
    ].copy()
    new_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    for fov_id in range(1, 11):
        new_frames.append(
            pd.read_csv(_cache_path(config, fov_id, "3d_detection_summary.csv"))
        )
        candidate_frames.append(
            pd.read_csv(_cache_path(config, fov_id, "spot_candidates.csv"))
        )
    new = pd.concat(new_frames, ignore_index=True)
    primary = pd.concat(candidate_frames, ignore_index=True)
    primary = primary.loc[
        primary["is_primary_candidate"].astype(bool),
        [
            "fov_id",
            "cell_id",
            "mTetR_r_mass",
            "mTetR_crop_mean",
        ],
    ]
    merged = reference.merge(
        new,
        on=["fov_id", "cell_id"],
        how="outer",
        suffixes=("_reference", "_new"),
        indicator=True,
    ).merge(
        primary,
        on=["fov_id", "cell_id"],
        how="left",
    )
    coordinate_equal = (
        (merged["primary_z_reference"] == merged["primary_z_new"])
        & (merged["primary_y_reference"] == merged["primary_y_new"])
        & (merged["primary_x_reference"] == merged["primary_x_new"])
    )
    candidate_count_difference = (
        merged["candidate_count_3d"].astype(float)
        - merged["candidate_count"].astype(float)
    )
    candidate_count_relative_difference = (
        candidate_count_difference.abs()
        / merged["candidate_count"].astype(float).clip(lower=1)
    )
    count_equal = candidate_count_difference == 0
    intensity_equal = np.isclose(
        merged["primary_intensity_reference"],
        merged["primary_intensity_new"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    rmass_equal = np.isclose(
        merged["primary_r_mass_r6"],
        merged["mTetR_r_mass"],
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    crop_mean_equal = np.isclose(
        merged["primary_crop_mean"],
        merged["mTetR_crop_mean"],
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    merged["coordinate_equal"] = coordinate_equal
    merged["candidate_count_equal"] = count_equal
    merged["candidate_count_difference"] = candidate_count_difference
    merged["candidate_count_relative_difference"] = (
        candidate_count_relative_difference
    )
    merged["primary_intensity_equal"] = intensity_equal
    merged["r_mass_equal"] = rmass_equal
    merged["crop_mean_equal"] = crop_mean_equal
    merged.to_csv(
        config.output_root
        / "tables"
        / "fov001_010_sweep_equivalence.csv",
        index=False,
    )
    checks = {
        "row_count": int(len(merged)),
        "all_rows_matched": bool((merged["_merge"] == "both").all()),
        "all_primary_coordinates_exact": bool(coordinate_equal.all()),
        "all_candidate_counts_exact": bool(count_equal.all()),
        "maximum_absolute_candidate_count_difference": int(
            candidate_count_difference.abs().max()
        ),
        "maximum_relative_candidate_count_difference": float(
            candidate_count_relative_difference.max()
        ),
        "all_primary_intensities_exact": bool(intensity_equal.all()),
        "all_r_mass_equal": bool(rmass_equal.all()),
        "all_crop_means_equal": bool(crop_mean_equal.all()),
    }
    checks["all_passed"] = bool(
        checks["all_rows_matched"]
        and checks["all_primary_coordinates_exact"]
        and checks["all_candidate_counts_exact"]
        and checks["all_primary_intensities_exact"]
        and checks["all_r_mass_equal"]
        and checks["all_crop_means_equal"]
    )
    write_json(
        config.output_root / "fov001_010_equivalence.json",
        checks,
    )
    if not checks["all_passed"]:
        raise RuntimeError(f"FOV 001-010 equivalence failed: {checks}")
    return checks


def calibrate_thresholds(
    candidates: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Retain diagnostic plots but fix the user-selected state thresholds."""

    primary = candidates.loc[
        candidates["is_primary_candidate"].astype(bool)
    ].copy()
    secondary = candidates.loc[
        ~candidates["is_primary_candidate"].astype(bool)
    ].copy()
    calibration_primary = primary.loc[
        primary["mTetR_crop_mean"] >= config.mtetr_display_min_crop_mean
    ].copy()
    calibration_secondary = secondary.loc[
        secondary["mTetR_crop_mean"] >= config.mtetr_display_min_crop_mean
    ].copy()
    low = calibration_primary.loc[
        calibration_primary["mTetR_crop_mean"]
        <= config.mtetr_lower_crop_mean_max,
        "mTetR_r_mass",
    ].dropna().to_numpy()
    high = calibration_primary.loc[
        calibration_primary["mTetR_crop_mean"]
        > config.mtetr_lower_crop_mean_max,
        "mTetR_r_mass",
    ].dropna().to_numpy()
    mtetr_raw, mtetr_artifact = legacy.kde_crosspoint(
        low,
        high,
        0.85,
        config.min_kde_relative_density,
    )
    rank_raw, rank_artifact = legacy.kde_crosspoint(
        calibration_primary["mTetR_r_mass"].dropna().to_numpy(),
        calibration_secondary["mTetR_r_mass"].dropna().to_numpy(),
        0.85,
        config.min_kde_relative_density,
    )
    selected_mtetr = float(config.fixed_mtetr_r_mass_threshold)
    selected_mcp = float(config.fixed_mcp_r_mass_threshold)
    mcp_population = primary.loc[
        np.isfinite(primary["mTetR_r_mass"])
        & (primary["mTetR_r_mass"] >= selected_mtetr)
        & np.isfinite(primary["mcp_r_mass"])
        & np.isfinite(primary["mcp_distance_um"])
    ].copy()
    near = mcp_population.loc[
        mcp_population["mcp_distance_um"] <= config.mcp_near_distance_um,
        "mcp_r_mass",
    ].to_numpy()
    far = mcp_population.loc[
        mcp_population["mcp_distance_um"] > config.mcp_near_distance_um,
        "mcp_r_mass",
    ].to_numpy()
    mcp_raw, mcp_artifact = legacy.kde_crosspoint(
        near,
        far,
        0.85,
        config.min_kde_relative_density,
    )
    sensitivity_rows: list[dict[str, Any]] = []
    mtetr_grid = np.unique(
        np.round(
            np.r_[
                np.linspace(0.98, 1.08, 21),
                selected_mtetr,
                1.030,
            ],
            4,
        )
    )
    mcp_grid = np.unique(np.round([selected_mcp, 1.05, 1.10, 1.15], 4))
    for mtetr_threshold in mtetr_grid:
        retained = primary.loc[
            primary["mTetR_r_mass"] >= float(mtetr_threshold)
        ]
        for mcp_threshold in mcp_grid:
            active = retained.loc[
                (retained["mcp_r_mass"] >= float(mcp_threshold))
                & (
                    retained["mcp_distance_um"]
                    <= config.mcp_near_distance_um
                )
            ]
            sensitivity_rows.append(
                {
                    "mTetR_r_mass_threshold": float(mtetr_threshold),
                    "mcp_r_mass_threshold": float(mcp_threshold),
                    "n_regions_retained": int(len(retained)),
                    "n_active": int(len(active)),
                    "active_fraction": (
                        float(len(active) / len(retained))
                        if len(retained)
                        else np.nan
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        config.output_root / "tables" / "threshold_sensitivity.csv",
        index=False,
    )
    decision = {
        "status": "final_fixed_after_user_threshold_update",
        "population": "Cellpose QC-selected final dilated nuclear masks",
        "rank1_candidate_count": int(len(primary)),
        "rank1_calibration_count_crop_mean_at_least_90": int(
            len(calibration_primary)
        ),
        "mtetr_display_min_crop_mean": float(
            config.mtetr_display_min_crop_mean
        ),
        "mtetr_crop_mean_split": float(config.mtetr_lower_crop_mean_max),
        "mtetr_crop_mean_split_method": (
            "fixed_lower_90_to_102_inclusive_higher_above_102"
        ),
        "mtetr_kde_crosspoint_raw_diagnostic": (
            float(mtetr_raw) if np.isfinite(mtetr_raw) else None
        ),
        "rank1_vs_secondary_kde_crosspoint_raw_diagnostic": (
            float(rank_raw) if np.isfinite(rank_raw) else None
        ),
        "selected_mtetr_r_mass": selected_mtetr,
        "selected_mtetr_source": f"user_fixed_config_{selected_mtetr:g}",
        "mcp_kde_crosspoint_raw_diagnostic": (
            float(mcp_raw) if np.isfinite(mcp_raw) else None
        ),
        "selected_mcp_r_mass": selected_mcp,
        "selected_mcp_source": f"user_fixed_config_{selected_mcp:g}",
        "mcp_near_distance_um": float(config.mcp_near_distance_um),
        "mcp_near_distance_px_sora": float(config.mcp_near_distance_px),
        "final_spot_parameters": {
            "median_kernel_zyx": config.median_kernel_zyx,
            "log_kernel_size_zyx_px": (
                config.bigfish_log_kernel_size_zyx_px
            ),
            "minimum_distance_zyx_px": (
                config.bigfish_minimum_distance_zyx_px
            ),
            "bigfish_threshold": "central-plane nuclear mTetR mean / 20",
            "r_mass_radius_px": config.final_r_mass_radius_px,
            "trackpy_diameter_px": config.final_trackpy_diameter_px,
            "trackpy_separation_px": config.final_trackpy_separation_px,
        },
    }
    write_json(config.output_root / "threshold_decision.json", decision)
    artifacts = {
        "mtetr": mtetr_artifact,
        "mtetr_rank_vs_secondary": rank_artifact,
        "mcp": mcp_artifact,
        "primary": primary,
        "calibration_primary": calibration_primary,
        "mcp_population": mcp_population,
    }
    return decision, artifacts, sensitivity


make_calibration_figures = legacy.make_calibration_figures
make_rank1_crop_qc = legacy.make_rank1_crop_qc
make_locus_summary = legacy.make_locus_summary


def call_states(
    candidates: pd.DataFrame,
    decision: dict[str, Any],
    config: AnalysisConfig,
) -> pd.DataFrame:
    primary = candidates.loc[
        candidates["is_primary_candidate"].astype(bool)
    ].copy()
    mtetr_threshold = float(decision["selected_mtetr_r_mass"])
    mcp_threshold = float(decision["selected_mcp_r_mass"])
    valid_crop = np.isfinite(primary["mTetR_r_mass"])
    primary["eligible_mTetR_locus"] = (
        valid_crop & (primary["mTetR_r_mass"] >= mtetr_threshold)
    )
    primary["state"] = "Excluded_below_mTetR"
    primary.loc[~valid_crop, "state"] = "Excluded_no_valid_crop"
    active = (
        primary["eligible_mTetR_locus"]
        & np.isfinite(primary["mcp_r_mass"])
        & (primary["mcp_r_mass"] >= mcp_threshold)
        & np.isfinite(primary["mcp_distance_um"])
        & (
            primary["mcp_distance_um"]
            <= config.mcp_near_distance_um
        )
    )
    inactive = primary["eligible_mTetR_locus"] & ~active
    primary.loc[active, "state"] = "Active"
    primary.loc[inactive, "state"] = "Inactive"
    primary["mTetR_r_mass_threshold"] = mtetr_threshold
    primary["mcp_r_mass_threshold"] = mcp_threshold
    primary["mcp_distance_threshold_um"] = config.mcp_near_distance_um
    primary["state_call_method"] = (
        "Gao 3D Big-FISH primary mTetR; same-Z MCP top1; "
        "fixed r_mTetR/r_MCP/physical-distance rule"
    )
    primary.to_csv(
        config.output_root / "tables" / "locus_state_calls.csv",
        index=False,
    )
    return primary


def validate_outputs(
    config: AnalysisConfig,
    metadata: dict[str, Any],
    regions: pd.DataFrame,
    candidates: pd.DataFrame,
    state_calls: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    expected_fovs = int(metadata["sizes"]["P"])
    source = config.segmentation_source_root
    cache = config.output_root / "per_fov_cache"
    figures = config.output_root / "figures"
    tables = config.output_root / "tables"
    counts = {
        "source_segmentation_masks": len(
            list((source / "segmentation_masks").glob("fov_*_selected_mask.tiff"))
        ),
        "source_pre_dilation_masks": len(
            list(
                (source / "segmentation_masks_pre_dilation").glob(
                    "fov_*_selected_mask_pre_dilation.tiff"
                )
            )
        ),
        "source_segmentation_overlays": len(
            list(
                (source / "segmentation_overlays").glob(
                    "fov_*_selected_regions.png"
                )
            )
        ),
        "spot_candidate_fov_tables": len(
            list(cache.glob("fov_*_spot_candidates.csv"))
        ),
        "spot_detection_fov_tables": len(
            list(cache.glob("fov_*_3d_detection_summary.csv"))
        ),
        "rank1_crop_archives": len(
            list(cache.glob("fov_*_rank1_crops.npz"))
        ),
        "completed_cache_manifests": len(
            list(cache.glob("fov_*_cache_manifest.json"))
        ),
        "spot_detection_overlays": len(
            list(
                (
                    config.output_root
                    / "qc"
                    / "spot_detection_overlays"
                ).glob("fov_*_3d_spot_overlay.png")
            )
        ),
    }
    equivalence_path = (
        config.output_root / "fov001_010_equivalence.json"
    )
    if config.require_reference_equivalence:
        equivalence_passed = bool(
            equivalence_path.is_file()
            and json.loads(
                equivalence_path.read_text(encoding="utf-8")
            ).get("all_passed")
        )
    else:
        equivalence_passed = True
    state_counts = state_calls["state"].value_counts().to_dict()
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
    has_both_states = bool(
        state_counts.get("Active", 0) > 0
        and state_counts.get("Inactive", 0) > 0
    )
    validations = {
        "all_source_masks": (
            counts["source_segmentation_masks"] == expected_fovs
        ),
        "all_source_pre_dilation_masks": (
            counts["source_pre_dilation_masks"] == expected_fovs
        ),
        "all_source_overlays": (
            counts["source_segmentation_overlays"] == expected_fovs
        ),
        "all_quantification_tables": (
            counts["spot_candidate_fov_tables"] == expected_fovs
        ),
        "all_detection_tables": (
            counts["spot_detection_fov_tables"] == expected_fovs
        ),
        "all_crop_archives": (
            counts["rank1_crop_archives"] == expected_fovs
        ),
        "all_cache_manifests": (
            counts["completed_cache_manifests"] == expected_fovs
        ),
        "all_spot_overlays": (
            counts["spot_detection_overlays"] == expected_fovs
        ),
        "all_qc_random_control_arrays_present": bool(
            random_control_cache_complete
            and len(random_control_counts) == expected_fovs
        ),
        "all_qc_random_control_population_nonzero": bool(
            sum(random_control_counts) > 0
        ),
        "selected_regions_nonzero": bool(
            regions["selected"].astype(bool).sum() > 0
        ),
        "rank1_candidates_nonzero": bool(
            candidates["is_primary_candidate"].astype(bool).sum() > 0
        ),
        "state_rows_match_rank1_candidates": bool(
            len(state_calls)
            == candidates["is_primary_candidate"].astype(bool).sum()
        ),
        "state_labels_are_known": bool(
            set(state_calls["state"]).issubset(
                {
                    "Active",
                    "Inactive",
                    "Excluded_below_mTetR",
                    "Excluded_no_valid_crop",
                }
            )
        ),
        "reference_equivalence_required_or_passed": equivalence_passed,
        "gao_fig1c_composite_exists_or_not_applicable": bool(
            not has_both_states
            or (figures / "gao_fig1c_method_composite.png").is_file()
        ),
        "gao_radial_profiles_exist_or_not_applicable": bool(
            not has_both_states
            or (figures / "locus_radial_profiles.png").is_file()
        ),
        "gao_normalization_table_exists_or_not_applicable": bool(
            not has_both_states
            or (tables / "gao_fig1c_normalization.csv").is_file()
        ),
    }
    report = {
        "analysis": (
            f"{config.analysis_label} fixed-cell SoRa "
            "Gao-compatible 3D snapshot"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {
            "nd2": getattr(nd2, "__version__", "unknown"),
            "trackpy": getattr(tp, "__version__", "unknown"),
            "bigfish": getattr(bigfish, "__version__", "unknown"),
        },
        "github_reference": {
            "repository": "Ochiai-Lab/gao2026_temporal",
            "commit": "e830aad3d1542f0c8cd8e555ae8cc06a16cef9ae",
            "notebook": (
                "analyses/01_Snapshot_analysis/notebooks/"
                "01_Snapshot_analysis.ipynb"
            ),
        },
        "method_adaptations": [
            "Reviewed Cellpose nuclei masks are reused read-only.",
            (
                "All spot detection and quantification uses the same "
                "3x3 XY median-filtered uint16 stack as Gao processed_1."
            ),
            (
                "Each nuclear bounding box uses 3D Big-FISH with LoG sigma "
                "(0.5,4,4) px, "
                "minimum distance (1,4,4) px, and nuclear central-plane "
                "mean/20 threshold, matching the FOV 001-010 sweep exactly."
            ),
            (
                f"The r_mass radius is {config.final_r_mass_radius_px} px "
                f"({config.r_mass_pad_um:.3f} um), matching the Gao "
                "0.39-um physical aperture; Trackpy diameter/separation "
                "remain resolution-adapted at 11/12 px."
            ),
            (
                "Primary locus crops are median-filtered single-Z "
                "101x101 px images; no MIP is used downstream."
            ),
            (
                f"State thresholds are fixed at "
                f"r_mTetR>={config.fixed_mtetr_r_mass_threshold:g}, "
                f"r_MCP>={config.fixed_mcp_r_mass_threshold:g}, "
                f"distance<={config.mcp_near_distance_um:g} um."
            ),
            (
                "One random crop is generated independently for every "
                "QC-passing nucleus, using the pre-dilation mask when "
                "available; Gao random-mean subtraction and shared "
                "Active/Inactive pixel-max scaling to 100 are retained."
            ),
        ],
        "counts": counts,
        "n_selected_regions": int(
            regions["selected"].astype(bool).sum()
        ),
        "n_saved_candidates": int(len(candidates)),
        "n_rank1_candidates": int(
            candidates["is_primary_candidate"].astype(bool).sum()
        ),
        "n_all_qc_random_controls": int(sum(random_control_counts)),
        "state_counts": state_counts,
        "gao_summary_applicable": has_both_states,
        "threshold_decision": decision,
        "validations": validations,
        "all_validations_passed": bool(all(validations.values())),
        "config": asdict(config),
    }
    write_json(config.output_root / "analysis_manifest.json", report)
    if not report["all_validations_passed"]:
        failed = [name for name, value in validations.items() if not value]
        raise RuntimeError(f"Output validation failed: {failed}")
    return report
