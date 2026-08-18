#!/usr/bin/env python3
"""Restartable fixed-cell SoRa snapshot analysis for ChamberA Field1.

The workflow follows the Gao 2026 snapshot analysis while adapting spatial
parameters to the SoRa pixel size and using all three fluorescence channels for
Cellpose segmentation.
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
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import cellpose
import nd2
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import torch
import trackpy as tp
from PIL import Image
from cellpose import models
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, mannwhitneyu
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import expand_labels, find_boundaries
from skimage.transform import resize


sns.set_theme(context="notebook", style="whitegrid")
tp.quiet(suppress=True)


@dataclass(frozen=True)
class AnalysisConfig:
    """All analysis choices needed to reproduce this first-pass run."""

    input_nd2: Path = Path(
        str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260716-SoRa_Fixed/260716_SoRa/260716_ChamberA/ChamberA_20260716_205221_Field1.nd2')
    )
    project_dir: Path = Path(
        str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO')
    )
    output_name: str = (
        "sora_snapshot_outputs_chamberA_field1_final_"
        "aa4x_sigma11_d110_cp05_dilate15_gao"
    )
    analysis_id: str = "nanog_ser5ph_rep1"
    analysis_label: str = "Nanog × Ser5ph — rep1 (ChamberA Field1)"
    streaming_tag_locus: str = "Nanog"
    mintbody_label: str = "Ser5ph"
    replicate: str = "rep1"
    chamber: str = "ChamberA"
    expected_fov_count: int | None = 400
    snap_channel: int = 0
    mtetr_channel: int = 1
    mcp_channel: int = 2
    expected_xy_um: float = 0.023214285714285715
    expected_z_um: float = 0.5
    segmentation_downsample: int = 4
    segmentation_downsample_method: str = "antialias_resize"
    segmentation_background_sigma_px_full: float = 30.0
    segmentation_signal_sigma_px_full: float = 11.0
    segmentation_dilation_px_small: int = 15
    cellpose_model: str = "nuclei"
    cellpose_diameter_px_small: float = 110.0
    cellpose_cellprob_threshold: float = 0.5
    cellpose_flow_threshold: float = 0.4
    cellpose_min_size_px_small: int = 100
    min_region_area_um2: float = 30.0
    max_region_area_um2: float = 700.0
    exclude_border_regions: bool = True
    reference_pixel_um: float = 0.13
    crop_radius_reference_px: int = 9
    r_mass_pad_reference_px: int = 3
    spot_diameter_reference_px: int = 5
    peak_min_distance_reference_px: int = 2
    mcp_near_distance_reference_px: int = 3
    top_n_mtetr_candidates: int = 5
    reference_mtetr_r_mass: float = 1.15
    reference_mcp_r_mass: float = 1.15
    fixed_mcp_r_mass_threshold: float = 1.025
    mtetr_plot_comparison_thresholds: tuple[float, ...] = (1.015, 1.030)
    min_kde_relative_density: float = 0.10
    mtetr_display_min_crop_mean: float = 90.0
    mtetr_lower_crop_mean_max: float = 102.0
    random_z_min_index: int = 3
    random_z_max_index: int = 7
    gao_metric_core_radius_reference_px: int = 2
    gao_metric_annulus_inner_reference_px: int = 6
    gao_metric_annulus_outer_reference_px: int = 8
    random_seed: int = 20260718
    overwrite_segmentation: bool = False
    overwrite_quantification: bool = False

    @property
    def output_root(self) -> Path:
        return self.project_dir / self.output_name

    @property
    def segmentation_pixel_um(self) -> float:
        return self.expected_xy_um * self.segmentation_downsample

    @property
    def segmentation_dilation_um(self) -> float:
        return self.segmentation_dilation_px_small * self.segmentation_pixel_um

    @property
    def cellpose_diameter_um(self) -> float:
        return self.cellpose_diameter_px_small * self.segmentation_pixel_um

    @property
    def crop_radius_um(self) -> float:
        return self.crop_radius_reference_px * self.reference_pixel_um

    @property
    def r_mass_pad_um(self) -> float:
        return self.r_mass_pad_reference_px * self.reference_pixel_um

    @property
    def spot_diameter_um(self) -> float:
        return self.spot_diameter_reference_px * self.reference_pixel_um

    @property
    def peak_min_distance_um(self) -> float:
        return self.peak_min_distance_reference_px * self.reference_pixel_um

    @property
    def mcp_near_distance_um(self) -> float:
        return self.mcp_near_distance_reference_px * self.reference_pixel_um

    def scaled_odd_pixels(self, distance_um: float) -> int:
        value = max(1, int(round(distance_um / self.expected_xy_um)))
        return value if value % 2 == 1 else value + 1

    @property
    def crop_radius_px(self) -> int:
        return max(1, int(round(self.crop_radius_um / self.expected_xy_um)))

    @property
    def r_mass_pad_px(self) -> int:
        return max(1, int(round(self.r_mass_pad_um / self.expected_xy_um)))

    @property
    def spot_diameter_px(self) -> int:
        return self.scaled_odd_pixels(self.spot_diameter_um)

    @property
    def peak_min_distance_px(self) -> int:
        return max(1, int(round(self.peak_min_distance_um / self.expected_xy_um)))

    @property
    def mcp_near_distance_px(self) -> float:
        return self.mcp_near_distance_um / self.expected_xy_um

    @property
    def gao_metric_core_radius_px(self) -> int:
        return max(
            1,
            int(
                round(
                    self.gao_metric_core_radius_reference_px
                    * self.reference_pixel_um
                    / self.expected_xy_um
                )
            ),
        )

    @property
    def gao_metric_annulus_px(self) -> tuple[int, int]:
        inner = max(
            1,
            int(
                round(
                    self.gao_metric_annulus_inner_reference_px
                    * self.reference_pixel_um
                    / self.expected_xy_um
                )
            ),
        )
        outer = max(
            inner + 1,
            int(
                round(
                    self.gao_metric_annulus_outer_reference_px
                    * self.reference_pixel_um
                    / self.expected_xy_um
                )
            ),
        )
        return inner, outer


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def prepare_directories(config: AnalysisConfig) -> dict[str, Path]:
    root = config.output_root
    directories = {
        "root": root,
        "inputs": root / "segmentation_inputs",
        "masks": root / "segmentation_masks",
        "masks_pre_dilation": root / "segmentation_masks_pre_dilation",
        "overlays": root / "segmentation_overlays",
        "tables": root / "tables",
        "cache": root / "per_fov_cache",
        "figures": root / "figures",
        "qc": root / "qc",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def collect_metadata(config: AnalysisConfig) -> dict[str, Any]:
    """Validate the ND2 dimensions and record trustworthy channel metadata."""

    if not config.input_nd2.is_file():
        raise FileNotFoundError(config.input_nd2)
    with nd2.ND2File(config.input_nd2) as handle:
        sizes = {name: int(size) for name, size in handle.sizes.items()}
        voxel = handle.voxel_size()
        channel_names = [channel.channel.name for channel in handle.metadata.channels]
        text_info = dict(handle.text_info)
        dtype = str(handle.dtype)
    voxel_um = {"x": float(voxel.x), "y": float(voxel.y), "z": float(voxel.z)}
    if sizes.get("C") != 3 or sizes.get("Z") != 11 or sizes.get("P", 0) <= 0:
        raise ValueError(f"Unexpected ND2 dimensions: {sizes}")
    if (
        config.expected_fov_count is not None
        and sizes.get("P") != int(config.expected_fov_count)
    ):
        raise ValueError(
            f"Unexpected FOV count: {sizes.get('P')} "
            f"(expected {config.expected_fov_count})"
        )
    if not np.allclose(
        [voxel_um["x"], voxel_um["y"], voxel_um["z"]],
        [config.expected_xy_um, config.expected_xy_um, config.expected_z_um],
    ):
        raise ValueError(f"Unexpected voxel size: {voxel_um}")

    # The generic excitationLambdaNm fields in this ND2 are not reliable.
    # The acquisition text and channel names identify the actual laser order.
    description = str(text_info.get("description", ""))
    plane_settings: list[dict[str, float | int | None]] = []
    for plane_number in range(1, 4):
        start = description.find(f"Plane #{plane_number}:")
        end = description.find(f"Plane #{plane_number + 1}:")
        block = description[start : end if end >= 0 else None]
        exposure_match = re.search(r"Exposure:\s*(\d+)\s*ms", block)
        power_match = re.search(r"Power:\s*([\d.]+)", block)
        plane_settings.append(
            {
                "exposure_ms": (
                    int(exposure_match.group(1)) if exposure_match else None
                ),
                "laser_power_percent": (
                    float(power_match.group(1)) if power_match else None
                ),
            }
        )
    channel_mapping = [
        {
            "index": 0,
            "nd2_name": channel_names[0],
            "laser_nm": 640,
            **plane_settings[0],
            "biological_label": (
                f"SNAPtag / {config.mintbody_label} mintbody"
            ),
        },
        {
            "index": 1,
            "nd2_name": channel_names[1],
            "laser_nm": 515,
            **plane_settings[1],
            "biological_label": "mTetR-YFP / mGold2s",
        },
        {
            "index": 2,
            "nd2_name": channel_names[2],
            "laser_nm": 445,
            **plane_settings[2],
            "biological_label": "MCP-CFP",
        },
    ]
    metadata = {
        "input_nd2": str(config.input_nd2),
        "input_bytes": config.input_nd2.stat().st_size,
        "sizes": sizes,
        "dtype": dtype,
        "voxel_size_um": voxel_um,
        "channel_mapping": channel_mapping,
        "channel_order_source": "ND2 channel names and acquisition text_info",
        "text_info": text_info,
        "scaled_parameters": {
            "crop_radius_px": config.crop_radius_px,
            "crop_size_px": 2 * config.crop_radius_px + 1,
            "crop_radius_um": config.crop_radius_um,
            "r_mass_pad_px": config.r_mass_pad_px,
            "r_mass_pad_um": config.r_mass_pad_um,
            "trackpy_spot_diameter_px": config.spot_diameter_px,
            "trackpy_spot_diameter_um": config.spot_diameter_um,
            "peak_min_distance_px": config.peak_min_distance_px,
            "peak_min_distance_um": config.peak_min_distance_um,
            "mcp_near_distance_px": config.mcp_near_distance_px,
            "mcp_near_distance_um": config.mcp_near_distance_um,
        },
    }
    write_json(config.output_root / "nd2_metadata.json", metadata)
    return metadata


def load_position(handle: nd2.ND2File, fov_id: int) -> np.ndarray:
    stack = np.asarray(handle.asarray(position=int(fov_id)))
    if stack.ndim == 5:
        if stack.shape[0] != 1:
            raise ValueError(f"Unexpected T dimension for FOV {fov_id}: {stack.shape}")
        stack = stack[0]
    if stack.ndim != 4:
        raise ValueError(f"Expected ZCYX for FOV {fov_id}, got {stack.shape}")
    return stack


def robust_normalize(
    image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.8
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, (low_percentile, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    height = image.shape[-2] // factor * factor
    width = image.shape[-1] // factor * factor
    cropped = image[..., :height, :width]
    shape = cropped.shape[:-2] + (height // factor, factor, width // factor, factor)
    return cropped.reshape(shape).mean(axis=(-3, -1))


def downsample_anti_aliased_bilinear(
    image: np.ndarray, factor: int
) -> np.ndarray:
    """Downsample one 2D image using bilinear interpolation and anti-aliasing."""

    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    height = int(image.shape[-2]) // factor
    width = int(image.shape[-1]) // factor
    return np.asarray(
        resize(
            np.asarray(image, dtype=np.float32),
            output_shape=(height, width),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ),
        dtype=np.float32,
    )


def downsample_for_segmentation(
    image: np.ndarray, factor: int, method: str
) -> np.ndarray:
    """Apply the configured downsampling method to one preprocessed channel."""

    if method == "antialias_resize":
        return downsample_anti_aliased_bilinear(image, factor)
    if method == "block_mean":
        return np.asarray(block_mean(image, factor), dtype=np.float32)
    raise ValueError(f"Unsupported segmentation downsample method: {method}")


def preprocess_mip_for_cellpose(
    mip: np.ndarray,
    background_sigma_px: float,
    signal_sigma_px: float,
) -> np.ndarray:
    """Apply the requested 16-bit high-pass and smoothing sequence.

    1. Estimate the background from the raw MIP with Gaussian sigma=30 px.
    2. Subtract it from the raw MIP, clip negative values to zero, and quantize
       to unsigned 16 bit.
    3. Smooth the clipped signal with the configured Gaussian sigma and return
       uint16.
    """

    raw = np.asarray(mip, dtype=np.float32)
    background = gaussian_filter(raw, sigma=float(background_sigma_px))
    background_subtracted = np.clip(raw - background, 0, np.iinfo(np.uint16).max)
    background_subtracted_16bit = np.rint(background_subtracted).astype(np.uint16)
    smoothed = gaussian_filter(
        background_subtracted_16bit.astype(np.float32),
        sigma=float(signal_sigma_px),
    )
    return np.rint(
        np.clip(smoothed, 0, np.iinfo(np.uint16).max)
    ).astype(np.uint16)


def _fov_path(directory: Path, fov_id: int, suffix: str) -> Path:
    return directory / f"fov_{int(fov_id):03d}_{suffix}"


def _read_csv_or_empty(path: Path, columns: Iterable[str] = ()) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def _region_qc(
    masks: np.ndarray, config: AnalysisConfig
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected_mask = np.zeros_like(masks, dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    pixel_um_small = config.expected_xy_um * config.segmentation_downsample
    next_label = 1
    height, width = masks.shape
    for region in regionprops(masks):
        min_row, min_col, max_row, max_col = region.bbox
        touches_border = min_row == 0 or min_col == 0 or max_row == height or max_col == width
        area_um2 = float(region.area * pixel_um_small**2)
        reasons: list[str] = []
        if area_um2 < config.min_region_area_um2:
            reasons.append("area_below_min")
        if area_um2 > config.max_region_area_um2:
            reasons.append("area_above_max")
        if config.exclude_border_regions and touches_border:
            reasons.append("touches_border")
        selected = len(reasons) == 0
        selected_label = next_label if selected else 0
        if selected:
            selected_mask[masks == region.label] = next_label
            next_label += 1
        rows.append(
            {
                "raw_label": int(region.label),
                "cell_id": int(selected_label),
                "selected": bool(selected),
                "exclusion_reason": ";".join(reasons),
                "area_px_downsampled": int(region.area),
                "area_um2": area_um2,
                "touches_border": bool(touches_border),
                "centroid_y_px_downsampled": float(region.centroid[0]),
                "centroid_x_px_downsampled": float(region.centroid[1]),
                "bbox_min_y_px_downsampled": int(min_row),
                "bbox_min_x_px_downsampled": int(min_col),
                "bbox_max_y_px_downsampled": int(max_row),
                "bbox_max_x_px_downsampled": int(max_col),
            }
        )
    return selected_mask, rows


def _save_segmentation_overlay(
    path: Path,
    rgb: np.ndarray,
    raw_masks: np.ndarray,
    selected_mask_pre_dilation: np.ndarray,
    rows: list[dict[str, Any]],
    fov_id: int,
    analysis_label: str,
    selected_mask_final: np.ndarray | None = None,
    dilation_px: int = 0,
    dilation_um: float = 0.0,
) -> None:
    if selected_mask_final is None:
        selected_mask_final = selected_mask_pre_dilation
    overlay = rgb.copy()
    final_pixels = selected_mask_final > 0
    overlay[final_pixels] = (
        0.78 * overlay[final_pixels]
        + 0.22 * np.array([1.0, 0.88, 0.0], dtype=np.float32)
    )
    excluded_boundary = find_boundaries(raw_masks, mode="outer")
    pre_dilation_boundary = find_boundaries(
        selected_mask_pre_dilation, mode="outer"
    )
    final_boundary = find_boundaries(selected_mask_final, mode="outer")
    overlay[excluded_boundary] = np.array([1.0, 0.15, 0.15])
    overlay[pre_dilation_boundary] = np.array([0.0, 1.0, 1.0])
    overlay[final_boundary] = np.array([1.0, 1.0, 0.0])
    selected_rows = [row for row in rows if row["selected"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Cellpose input context\nR=SNAP, G=mTetR, B=MCP")
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Selected regions: n={len(selected_rows)}\n"
        f"cyan=pre-dilation; yellow=final +{dilation_px}px "
        f"({dilation_um:.2f} µm); red=excluded"
    )
    for row in selected_rows:
        axes[1].text(
            row["centroid_x_px_downsampled"],
            row["centroid_y_px_downsampled"],
            str(row["cell_id"]),
            color="white",
            fontsize=6,
            ha="center",
            va="center",
        )
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{analysis_label} — FOV {fov_id:03d}")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_segmentation(
    config: AnalysisConfig,
    fov_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Run/reuse all-channel Cellpose segmentation and save one overlay per FOV."""

    directories = prepare_directories(config)
    metadata = collect_metadata(config)
    n_positions = int(metadata["sizes"]["P"])
    effective_fovs = list(range(n_positions)) if fov_ids is None else [int(x) for x in fov_ids]
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_gpu = True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        use_gpu = True
    else:
        device = torch.device("cpu")
        use_gpu = False
    if config.cellpose_model == "nuclei":
        model = models.CellposeModel(
            gpu=use_gpu,
            device=device,
            model_type="nuclei",
        )
        cellpose_api = "Cellpose 3 nuclei model"
    else:
        model = models.CellposeModel(
            gpu=use_gpu,
            device=device,
            pretrained_model=config.cellpose_model,
        )
        cellpose_api = "Cellpose 4 pretrained model"
    all_rows: list[pd.DataFrame] = []
    cellpose_evaluated = False

    with nd2.ND2File(config.input_nd2) as handle:
        for ordinal, fov_id in enumerate(effective_fovs, start=1):
            input_path = _fov_path(directories["inputs"], fov_id, "all_channels_mixed.png")
            mask_path = _fov_path(directories["masks"], fov_id, "selected_mask.tiff")
            pre_dilation_mask_path = _fov_path(
                directories["masks_pre_dilation"],
                fov_id,
                "selected_mask_pre_dilation.tiff",
            )
            overlay_path = _fov_path(directories["overlays"], fov_id, "selected_regions.png")
            table_path = _fov_path(directories["cache"], fov_id, "segmentation_regions.csv")

            cache_ready = all(
                path.is_file()
                for path in (
                    input_path,
                    mask_path,
                    pre_dilation_mask_path,
                    overlay_path,
                    table_path,
                )
            )
            if cache_ready and not config.overwrite_segmentation:
                all_rows.append(
                    _read_csv_or_empty(
                        table_path,
                        columns=[
                            "fov_id",
                            "raw_label",
                            "cell_id",
                            "selected",
                            "exclusion_reason",
                        ],
                    )
                )
                if ordinal % 25 == 0 or ordinal == len(effective_fovs):
                    print(f"Segmentation cache: {ordinal}/{len(effective_fovs)} FOV")
                continue

            stack = load_position(handle, fov_id)
            projections = stack.max(axis=0).astype(np.float32)
            processed = np.stack(
                [
                    preprocess_mip_for_cellpose(
                        channel,
                        config.segmentation_background_sigma_px_full,
                        config.segmentation_signal_sigma_px_full,
                    )
                    for channel in projections
                ]
            )
            small = np.stack(
                [
                    downsample_for_segmentation(
                        channel,
                        config.segmentation_downsample,
                        config.segmentation_downsample_method,
                    )
                    for channel in processed
                ]
            )
            normalized = np.stack(
                [robust_normalize(channel) for channel in small]
            )
            rgb = np.moveaxis(normalized, 0, -1)

            # Equal-weight mixing occurs only after every channel has undergone
            # the requested full-resolution 30-px background subtraction,
            # uint16 clipping, and configured signal smoothing.
            mixed = robust_normalize(normalized.mean(axis=0))
            mixed_16bit = np.rint(mixed * np.iinfo(np.uint16).max).astype(np.uint16)
            Image.fromarray(mixed_16bit, mode="I;16").save(input_path)

            raw_masks, _, _ = model.eval(
                mixed_16bit,
                channels=[0, 0],
                diameter=config.cellpose_diameter_px_small,
                normalize=True,
                flow_threshold=config.cellpose_flow_threshold,
                cellprob_threshold=config.cellpose_cellprob_threshold,
                min_size=config.cellpose_min_size_px_small,
                # Cellpose 3 otherwise reloads the same pretrained weights
                # on every per-FOV eval call.  Keep the already-loaded model
                # after the first uncached FOV; this does not change inference
                # parameters or numerical results.
                loop_run=cellpose_evaluated,
            )
            cellpose_evaluated = True
            raw_masks = np.asarray(raw_masks)
            selected_mask_pre_dilation, rows = _region_qc(raw_masks, config)
            selected_mask_final = expand_labels(
                selected_mask_pre_dilation,
                distance=float(config.segmentation_dilation_px_small),
            ).astype(np.uint16, copy=False)
            tifffile.imwrite(
                pre_dilation_mask_path,
                selected_mask_pre_dilation,
                compression="zlib",
            )
            tifffile.imwrite(mask_path, selected_mask_final, compression="zlib")
            final_counts = np.bincount(
                selected_mask_final.ravel(),
                minlength=int(selected_mask_pre_dilation.max()) + 1,
            )
            for row in rows:
                row["dilation_radius_px_downsampled"] = int(
                    config.segmentation_dilation_px_small
                )
                row["dilation_radius_um"] = float(
                    config.segmentation_dilation_um
                )
                if row["selected"]:
                    final_area_px = int(final_counts[int(row["cell_id"])])
                    row["area_px_downsampled_final"] = final_area_px
                    row["area_um2_final"] = float(
                        final_area_px * config.segmentation_pixel_um**2
                    )
                else:
                    row["area_px_downsampled_final"] = 0
                    row["area_um2_final"] = 0.0
            _save_segmentation_overlay(
                overlay_path,
                rgb,
                raw_masks,
                selected_mask_pre_dilation,
                rows,
                fov_id,
                config.analysis_label,
                selected_mask_final=selected_mask_final,
                dilation_px=config.segmentation_dilation_px_small,
                dilation_um=config.segmentation_dilation_um,
            )
            frame = pd.DataFrame(rows)
            frame.insert(0, "fov_id", int(fov_id))
            frame.to_csv(table_path, index=False)
            all_rows.append(frame)
            if ordinal % 10 == 0 or ordinal == len(effective_fovs):
                print(
                    f"Segmentation: {ordinal}/{len(effective_fovs)} FOV; "
                    f"selected in current FOV={int(selected_mask_final.max())}"
                )

    regions = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    regions.to_csv(directories["tables"] / "segmentation_regions_all_fovs.csv", index=False)
    summary = {
        "gpu": use_gpu,
        "device": str(device or "cpu"),
        "cellpose_version": getattr(cellpose, "version", "unknown"),
        "cellpose_model": config.cellpose_model,
        "cellpose_api": cellpose_api,
        "n_fovs": len(effective_fovs),
        "n_raw_regions": int(len(regions)),
        "n_selected_regions": int(regions["selected"].astype(bool).sum()) if not regions.empty else 0,
        "n_inputs": len(list(directories["inputs"].glob("fov_*_all_channels_mixed.png"))),
        "n_masks": len(list(directories["masks"].glob("fov_*_selected_mask.tiff"))),
        "n_masks_pre_dilation": len(
            list(
                directories["masks_pre_dilation"].glob(
                    "fov_*_selected_mask_pre_dilation.tiff"
                )
            )
        ),
        "n_overlays": len(list(directories["overlays"].glob("fov_*_selected_regions.png"))),
        "region_qc": {
            "min_area_um2": config.min_region_area_um2,
            "max_area_um2": config.max_region_area_um2,
            "exclude_border": config.exclude_border_regions,
        },
        "preprocessing": {
            "sequence": (
                "per-channel MIP -> Gaussian background -> subtract -> "
                "clip negative to zero -> uint16 -> Gaussian signal smoothing "
                "-> per-channel normalization -> equal-weight mix"
            ),
            "background_sigma_px_full_resolution": (
                config.segmentation_background_sigma_px_full
            ),
            "signal_sigma_px_full_resolution": (
                config.segmentation_signal_sigma_px_full
            ),
            "downsample_method": config.segmentation_downsample_method,
            "downsample_factor": config.segmentation_downsample,
            "mixed_png_dtype": "uint16",
        },
        "final_mask": {
            "method": (
                "skimage.segmentation.expand_labels; Euclidean label-aware "
                "dilation without label overlap"
            ),
            "dilation_radius_px_downsampled": (
                config.segmentation_dilation_px_small
            ),
            "dilation_radius_um": config.segmentation_dilation_um,
        },
    }
    write_json(config.output_root / "segmentation_summary.json", summary)
    return regions


def centered_crop(
    plane_cyx: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray | None:
    y0, y1 = int(center_y - radius), int(center_y + radius + 1)
    x0, x1 = int(center_x - radius), int(center_x + radius + 1)
    if y0 < 0 or x0 < 0 or y1 > plane_cyx.shape[-2] or x1 > plane_cyx.shape[-1]:
        return None
    crop = plane_cyx[:, y0:y1, x0:x1]
    expected = (plane_cyx.shape[0], 2 * radius + 1, 2 * radius + 1)
    return crop if crop.shape == expected else None


def gao_r_mass(
    coordinates_yx: np.ndarray,
    image: np.ndarray,
    pad: int,
) -> dict[str, float] | None:
    """Gao/STtag r_mass: local COM mass / area / whole-crop mean."""

    values = np.asarray(image, dtype=np.float32)
    crop_mean = float(np.mean(values))
    if not np.isfinite(crop_mean) or crop_mean <= 0:
        return None
    padded = np.pad(values, pad_width=pad, mode="constant")
    coordinates = np.asarray(coordinates_yx, dtype=float) + pad
    refined = tp.refine_com(padded, padded, radius=pad, coords=coordinates)
    if refined is None or refined.empty:
        return None
    mass_column = "raw_mass" if "raw_mass" in refined.columns else "mass"
    signal_intensity = float(refined[mass_column].iloc[0]) / (math.pi * pad**2)
    return {
        "r_mass": signal_intensity / crop_mean,
        "signal_intensity": signal_intensity,
        "crop_mean": crop_mean,
    }


def gao_mcp_top1(
    image: np.ndarray,
    diameter: int,
    pad: int,
    crop_radius: int,
    pixel_um: float,
) -> dict[str, float]:
    result = {
        "mcp_y_crop_px": np.nan,
        "mcp_x_crop_px": np.nan,
        "mcp_r_mass": np.nan,
        "mcp_signal_intensity": np.nan,
        "mcp_crop_mean": float(np.mean(image)),
        "mcp_distance_px": np.nan,
        "mcp_distance_um": np.nan,
    }
    detected = tp.locate(np.asarray(image, dtype=np.float32), diameter, topn=1)
    if detected.empty:
        return result
    try:
        refined = tp.refine_leastsq(
            detected, np.asarray(image, dtype=np.float32), diameter, fit_function="gauss"
        ).reset_index(drop=True)
    except Exception:
        refined = detected.reset_index(drop=True)
    y_px = float(refined.loc[0, "y"])
    x_px = float(refined.loc[0, "x"])
    intensity = gao_r_mass(np.array([[y_px, x_px]]), image, pad)
    distance_px = float(np.hypot(y_px - crop_radius, x_px - crop_radius))
    result.update(
        {
            "mcp_y_crop_px": y_px,
            "mcp_x_crop_px": x_px,
            "mcp_r_mass": np.nan if intensity is None else intensity["r_mass"],
            "mcp_signal_intensity": (
                np.nan if intensity is None else intensity["signal_intensity"]
            ),
            "mcp_crop_mean": np.nan if intensity is None else intensity["crop_mean"],
            "mcp_distance_px": distance_px,
            "mcp_distance_um": distance_px * pixel_um,
        }
    )
    return result


def _upsample_mask(mask_small: np.ndarray, factor: int, shape_yx: tuple[int, int]) -> np.ndarray:
    mask = np.repeat(np.repeat(mask_small, factor, axis=0), factor, axis=1)
    return mask[: shape_yx[0], : shape_yx[1]]


def _choose_random_center(
    region_mask: np.ndarray,
    radius: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    """Sample one point uniformly from a nuclear label, as in Gao controls."""

    height, width = region_mask.shape
    valid = region_mask.copy()
    valid[:radius, :] = False
    valid[-radius:, :] = False
    valid[:, :radius] = False
    valid[:, -radius:] = False
    coordinates = np.argwhere(valid)
    if len(coordinates) == 0:
        return None
    chosen = coordinates[int(rng.integers(0, len(coordinates)))]
    return int(chosen[0]), int(chosen[1])


def _analyze_single_fov(
    stack: np.ndarray,
    mask_small: np.ndarray,
    fov_id: int,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    if not (
        0
        <= config.random_z_min_index
        <= config.random_z_max_index
        < stack.shape[0]
    ):
        raise ValueError(
            "Random-control Z range is outside the input stack: "
            f"{config.random_z_min_index}–{config.random_z_max_index} "
            f"for {stack.shape[0]} planes"
        )
    projections = stack.max(axis=0).astype(np.float32)
    full_mask = _upsample_mask(
        mask_small,
        config.segmentation_downsample,
        (int(stack.shape[-2]), int(stack.shape[-1])),
    )
    mtetr_projection = projections[config.mtetr_channel]
    smoothed_mtetr = gaussian_filter(mtetr_projection, sigma=2.0)
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    crop_cell_ids: list[int] = []
    locus_crops: list[np.ndarray] = []
    random_crops: list[np.ndarray] = []
    rng = np.random.default_rng(config.random_seed + int(fov_id))

    for region in regionprops(mask_small):
        cell_id = int(region.label)
        small_min_y, small_min_x, small_max_y, small_max_x = region.bbox
        min_y = max(0, small_min_y * config.segmentation_downsample)
        min_x = max(0, small_min_x * config.segmentation_downsample)
        max_y = min(stack.shape[-2], small_max_y * config.segmentation_downsample)
        max_x = min(stack.shape[-1], small_max_x * config.segmentation_downsample)
        local_region = full_mask[min_y:max_y, min_x:max_x] == cell_id
        if not local_region.any():
            exclusions.append(
                {"fov_id": fov_id, "cell_id": cell_id, "reason": "upsampled_mask_missing"}
            )
            continue

        local_image = smoothed_mtetr[min_y:max_y, min_x:max_x]
        threshold_abs = float(np.percentile(local_image[local_region], 75))
        local_peaks = peak_local_max(
            local_image,
            labels=local_region.astype(np.uint8),
            min_distance=config.peak_min_distance_px,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        if len(local_peaks) == 0:
            masked_values = np.where(local_region, local_image, -np.inf)
            peak = np.unravel_index(np.argmax(masked_values), masked_values.shape)
            local_peaks = np.asarray([peak], dtype=int)
        peaks = local_peaks + np.array([min_y, min_x])
        peak_values = mtetr_projection[peaks[:, 0], peaks[:, 1]]
        rank_order = np.argsort(peak_values)[::-1][: config.top_n_mtetr_candidates]
        valid_for_cell: list[tuple[dict[str, Any], np.ndarray]] = []

        for candidate_rank, peak_index in enumerate(rank_order, start=1):
            locus_y, locus_x = peaks[int(peak_index)]
            z_index = int(np.argmax(stack[:, config.mtetr_channel, locus_y, locus_x]))
            plane = stack[z_index]
            crop = centered_crop(
                plane, int(locus_y), int(locus_x), config.crop_radius_px
            )
            if crop is None:
                exclusions.append(
                    {
                        "fov_id": fov_id,
                        "cell_id": cell_id,
                        "candidate_rank": candidate_rank,
                        "reason": "candidate_crop_out_of_frame",
                    }
                )
                continue
            try:
                mtetr_metrics = gao_r_mass(
                    np.array([[config.crop_radius_px, config.crop_radius_px]]),
                    crop[config.mtetr_channel],
                    config.r_mass_pad_px,
                )
            except Exception as exc:
                mtetr_metrics = None
                exclusions.append(
                    {
                        "fov_id": fov_id,
                        "cell_id": cell_id,
                        "candidate_rank": candidate_rank,
                        "reason": f"mtetr_trackpy_error:{type(exc).__name__}",
                    }
                )
            if mtetr_metrics is None:
                continue
            row = {
                "fov_id": int(fov_id),
                "cell_id": cell_id,
                "locus_z_index": z_index,
                "locus_y_px": int(locus_y),
                "locus_x_px": int(locus_x),
                "locus_y_um": float(locus_y * config.expected_xy_um),
                "locus_x_um": float(locus_x * config.expected_xy_um),
                "candidate_rank_by_raw_peak": int(candidate_rank),
                "is_primary_candidate": False,
                "mTetR_local_peak_raw_intensity": float(peak_values[int(peak_index)]),
                "mTetR_r_mass": float(mtetr_metrics["r_mass"]),
                "mTetR_signal_intensity": float(mtetr_metrics["signal_intensity"]),
                "mTetR_crop_mean": float(mtetr_metrics["crop_mean"]),
                "SNAP_projection_region_mean": float(
                    projections[config.snap_channel][full_mask == cell_id].mean()
                ),
                "mTetR_projection_region_mean": float(
                    projections[config.mtetr_channel][full_mask == cell_id].mean()
                ),
                "MCP_projection_region_mean": float(
                    projections[config.mcp_channel][full_mask == cell_id].mean()
                ),
            }
            valid_for_cell.append((row, crop))

        if not valid_for_cell:
            exclusions.append(
                {"fov_id": fov_id, "cell_id": cell_id, "reason": "no_valid_candidate"}
            )
            continue

        primary_row, primary_crop = valid_for_cell[0]
        primary_row["is_primary_candidate"] = True
        try:
            primary_row.update(
                gao_mcp_top1(
                    primary_crop[config.mcp_channel],
                    config.spot_diameter_px,
                    config.r_mass_pad_px,
                    config.crop_radius_px,
                    config.expected_xy_um,
                )
            )
        except Exception as exc:
            primary_row.update(
                {
                    "mcp_y_crop_px": np.nan,
                    "mcp_x_crop_px": np.nan,
                    "mcp_r_mass": np.nan,
                    "mcp_signal_intensity": np.nan,
                    "mcp_crop_mean": np.nan,
                    "mcp_distance_px": np.nan,
                    "mcp_distance_um": np.nan,
                }
            )
            exclusions.append(
                {
                    "fov_id": fov_id,
                    "cell_id": cell_id,
                    "candidate_rank": 1,
                    "reason": f"mcp_trackpy_error:{type(exc).__name__}",
                }
            )

        region_mask = full_mask == cell_id
        random_center = _choose_random_center(
            region_mask,
            config.crop_radius_px,
            rng,
        )
        if random_center is None:
            random_crop = np.zeros_like(primary_crop)
            primary_row["random_crop_available"] = False
        else:
            random_z_index = int(
                rng.integers(
                    config.random_z_min_index,
                    config.random_z_max_index + 1,
                )
            )
            random_crop = centered_crop(
                stack[random_z_index],
                random_center[0],
                random_center[1],
                config.crop_radius_px,
            )
            if random_crop is None:
                random_crop = np.zeros_like(primary_crop)
                primary_row["random_crop_available"] = False
            else:
                primary_row["random_crop_available"] = True
                primary_row["random_z_index"] = random_z_index
                primary_row["random_y_px"] = random_center[0]
                primary_row["random_x_px"] = random_center[1]

        crop_cell_ids.append(cell_id)
        locus_crops.append(np.asarray(primary_crop, dtype=np.uint16))
        random_crops.append(np.asarray(random_crop, dtype=np.uint16))
        candidates.extend(row for row, _ in valid_for_cell)

    crop_size = 2 * config.crop_radius_px + 1
    crop_payload = {
        "cell_ids": np.asarray(crop_cell_ids, dtype=np.int32),
        "locus_crops": (
            np.stack(locus_crops)
            if locus_crops
            else np.empty((0, 3, crop_size, crop_size), dtype=np.uint16)
        ),
        "random_crops": (
            np.stack(random_crops)
            if random_crops
            else np.empty((0, 3, crop_size, crop_size), dtype=np.uint16)
        ),
    }
    return pd.DataFrame(candidates), pd.DataFrame(exclusions), crop_payload


def run_quantification(
    config: AnalysisConfig,
    fov_ids: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify top-5 mTetR candidates and rank-1 MCP for each selected region."""

    directories = prepare_directories(config)
    metadata = collect_metadata(config)
    n_positions = int(metadata["sizes"]["P"])
    effective_fovs = list(range(n_positions)) if fov_ids is None else [int(x) for x in fov_ids]
    candidate_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []

    with nd2.ND2File(config.input_nd2) as handle:
        for ordinal, fov_id in enumerate(effective_fovs, start=1):
            mask_path = _fov_path(directories["masks"], fov_id, "selected_mask.tiff")
            candidate_path = _fov_path(directories["cache"], fov_id, "spot_candidates.csv")
            exclusion_path = _fov_path(directories["cache"], fov_id, "spot_exclusions.csv")
            crops_path = _fov_path(directories["cache"], fov_id, "rank1_crops.npz")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing segmentation mask: {mask_path}")
            cache_ready = all(
                path.is_file() for path in (candidate_path, exclusion_path, crops_path)
            )
            if cache_ready and not config.overwrite_quantification:
                candidates = _read_csv_or_empty(
                    candidate_path,
                    columns=[
                        "fov_id",
                        "cell_id",
                        "candidate_rank_by_raw_peak",
                        "is_primary_candidate",
                    ],
                )
                exclusions = _read_csv_or_empty(
                    exclusion_path,
                    columns=["fov_id", "cell_id", "candidate_rank", "reason"],
                )
            else:
                stack = load_position(handle, fov_id)
                mask_small = tifffile.imread(mask_path)
                candidates, exclusions, crop_payload = _analyze_single_fov(
                    stack, mask_small, fov_id, config
                )
                if candidates.empty and len(candidates.columns) == 0:
                    candidates = pd.DataFrame(
                        columns=[
                            "fov_id",
                            "cell_id",
                            "candidate_rank_by_raw_peak",
                            "is_primary_candidate",
                        ]
                    )
                if exclusions.empty and len(exclusions.columns) == 0:
                    exclusions = pd.DataFrame(
                        columns=["fov_id", "cell_id", "candidate_rank", "reason"]
                    )
                candidates.to_csv(candidate_path, index=False)
                exclusions.to_csv(exclusion_path, index=False)
                np.savez_compressed(crops_path, **crop_payload)
            if not candidates.empty:
                candidate_frames.append(candidates)
            if not exclusions.empty:
                exclusion_frames.append(exclusions)
            if ordinal % 10 == 0 or ordinal == len(effective_fovs):
                print(
                    f"Quantification: {ordinal}/{len(effective_fovs)} FOV; "
                    f"current candidates={len(candidates)}"
                )

    all_candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame()
    )
    all_exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames
        else pd.DataFrame(columns=["fov_id", "cell_id", "candidate_rank", "reason"])
    )
    if all_candidates.empty:
        raise RuntimeError("No spot candidates were generated.")
    all_candidates = all_candidates.sort_values(
        ["fov_id", "cell_id", "candidate_rank_by_raw_peak"]
    ).reset_index(drop=True)
    all_candidates.to_csv(
        directories["tables"] / "mtetr_mcp_spot_candidates.csv", index=False
    )
    all_exclusions.to_csv(
        directories["tables"] / "spot_candidate_exclusions.csv", index=False
    )
    return all_candidates, all_exclusions


def kde_crosspoint(
    values_a: np.ndarray,
    values_b: np.ndarray,
    minimum: float,
    min_relative_density: float,
    grid_size: int = 2000,
) -> tuple[float, dict[str, Any] | None]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if (
        values_a.size < 20
        or values_b.size < 20
        or np.nanstd(values_a) == 0
        or np.nanstd(values_b) == 0
    ):
        return np.nan, None
    try:
        kde_a, kde_b = gaussian_kde(values_a), gaussian_kde(values_b)
    except Exception:
        return np.nan, None
    low = max(float(minimum), float(min(values_a.min(), values_b.min())))
    high = float(max(values_a.max(), values_b.max()))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.nan, None
    grid = np.linspace(low, high, grid_size)
    density_a, density_b = kde_a(grid), kde_b(grid)
    differences = density_a - density_b
    changes = np.flatnonzero(np.diff(np.sign(differences)) != 0)
    crossings: list[float] = []
    for index in changes:
        x0, x1 = grid[index], grid[index + 1]
        y0, y1 = differences[index], differences[index + 1]
        value = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else (x0 + x1) / 2
        if np.isfinite(value) and value >= minimum:
            crossings.append(float(value))
    # Prefer the crossing between the group medians. This avoids adopting a
    # mathematically valid but biologically unhelpful far-tail crossing.
    median_midpoint = float((np.median(values_a) + np.median(values_b)) / 2)
    if crossings:
        median_low = float(min(np.median(values_a), np.median(values_b)))
        median_high = float(max(np.median(values_a), np.median(values_b)))
        central_crossings = [
            value for value in crossings if median_low <= value <= median_high
        ]
        candidate_crossings = central_crossings if central_crossings else crossings
        raw = min(candidate_crossings, key=lambda value: abs(value - median_midpoint))
    else:
        raw = np.nan
    if np.isfinite(raw):
        density_at_cross = float(kde_a([raw])[0])
        relative_density = density_at_cross / max(
            float(density_a.max()), float(density_b.max()), np.finfo(float).eps
        )
    else:
        density_at_cross = np.nan
        relative_density = np.nan
    artifact = {
        "grid": grid,
        "density_a": density_a,
        "density_b": density_b,
        "values_a": values_a,
        "values_b": values_b,
        "n_a": int(values_a.size),
        "n_b": int(values_b.size),
        "median_a": float(np.median(values_a)),
        "median_b": float(np.median(values_b)),
        "all_crosspoints": crossings,
        "raw_crosspoint": raw,
        "density_at_crosspoint": density_at_cross,
        "relative_density_at_crosspoint": relative_density,
        "is_reliable": bool(
            np.isfinite(relative_density) and relative_density >= min_relative_density
        ),
    }
    return raw, artifact


def calibrate_thresholds(
    candidates: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Estimate thresholds and compare them with the Oyama 1.15 reference."""

    primary = candidates.loc[candidates["is_primary_candidate"].astype(bool)].copy()
    secondary = candidates.loc[~candidates["is_primary_candidate"].astype(bool)].copy()
    calibration_primary = primary.loc[
        primary["mTetR_crop_mean"] >= config.mtetr_display_min_crop_mean
    ].copy()
    calibration_secondary = secondary.loc[
        secondary["mTetR_crop_mean"] >= config.mtetr_display_min_crop_mean
    ].copy()
    crop_split = float(config.mtetr_lower_crop_mean_max)
    split_method = "fixed_lower_90_to_102_inclusive_higher_above_102"
    low = calibration_primary.loc[
        calibration_primary["mTetR_crop_mean"] <= crop_split, "mTetR_r_mass"
    ].to_numpy()
    high = calibration_primary.loc[
        calibration_primary["mTetR_crop_mean"] > crop_split, "mTetR_r_mass"
    ].to_numpy()
    mtetr_raw, mtetr_artifact = kde_crosspoint(
        low, high, 0.85, config.min_kde_relative_density
    )
    mtetr_retention = (
        float((calibration_primary["mTetR_r_mass"] >= mtetr_raw).mean())
        if np.isfinite(mtetr_raw)
        else np.nan
    )
    mtetr_reliable = (
        mtetr_artifact is not None
        and mtetr_artifact["is_reliable"]
        and 0.85 <= mtetr_raw <= 1.60
        and 0.05 <= mtetr_retention <= 0.95
    )

    rank_secondary_raw, rank_secondary_artifact = kde_crosspoint(
        calibration_primary["mTetR_r_mass"].to_numpy(),
        calibration_secondary["mTetR_r_mass"].to_numpy(),
        0.85,
        config.min_kde_relative_density,
    )
    rank_secondary_retention = (
        float(
            (calibration_primary["mTetR_r_mass"] >= rank_secondary_raw).mean()
        )
        if np.isfinite(rank_secondary_raw)
        else np.nan
    )
    rank_secondary_reliable = (
        rank_secondary_artifact is not None
        and rank_secondary_artifact["is_reliable"]
        and rank_secondary_artifact["median_a"] > rank_secondary_artifact["median_b"]
        and 0.85 <= rank_secondary_raw <= 1.60
        and 0.05 <= rank_secondary_retention <= 0.95
    )
    reference_mtetr_retention = float(
        (
            calibration_primary["mTetR_r_mass"]
            >= config.reference_mtetr_r_mass
        ).mean()
    )
    if mtetr_reliable:
        selected_mtetr = float(mtetr_raw)
        mtetr_source = "rank1_crop_mean_group_KDE_crosspoint"
    elif rank_secondary_reliable:
        selected_mtetr = float(rank_secondary_raw)
        mtetr_source = "rank1_vs_ranks2to5_KDE_crosspoint"
    elif 0.05 <= reference_mtetr_retention <= 0.95:
        selected_mtetr = float(config.reference_mtetr_r_mass)
        mtetr_source = "Oyama_1.15_reference_plausible_retention"
    else:
        selected_mtetr = float(
            calibration_primary["mTetR_r_mass"].quantile(0.75)
        )
        mtetr_source = "rank1_empirical_75th_percentile_fallback"

    mcp_population = primary.loc[
        np.isfinite(primary["mTetR_r_mass"])
        & (primary["mTetR_r_mass"] >= selected_mtetr)
        & np.isfinite(primary["mcp_r_mass"])
        & np.isfinite(primary["mcp_distance_um"])
    ].copy()
    near = mcp_population.loc[
        mcp_population["mcp_distance_um"] <= config.mcp_near_distance_um, "mcp_r_mass"
    ].to_numpy()
    far = mcp_population.loc[
        mcp_population["mcp_distance_um"] > config.mcp_near_distance_um, "mcp_r_mass"
    ].to_numpy()
    mcp_raw, mcp_artifact = kde_crosspoint(
        near, far, 0.85, config.min_kde_relative_density
    )
    mcp_near_retention = (
        float(np.mean(near >= mcp_raw)) if len(near) and np.isfinite(mcp_raw) else np.nan
    )
    mcp_reliable = (
        mcp_artifact is not None
        and mcp_artifact["is_reliable"]
        and mcp_artifact["median_a"] > mcp_artifact["median_b"]
        and 0.85 <= mcp_raw <= 2.00
        and 0.05 <= mcp_near_retention <= 0.95
    )
    reference_mcp_near_retention = (
        float(np.mean(near >= config.reference_mcp_r_mass)) if len(near) else np.nan
    )
    if mcp_reliable:
        diagnostic_mcp = float(mcp_raw)
        diagnostic_mcp_source = "near_far_KDE_crosspoint"
    elif len(near) and 0.05 <= reference_mcp_near_retention <= 0.95:
        diagnostic_mcp = float(config.reference_mcp_r_mass)
        diagnostic_mcp_source = "Oyama_1.15_reference_plausible_near_retention"
    elif len(near):
        diagnostic_mcp = float(np.quantile(near, 0.25))
        diagnostic_mcp_source = "near_group_empirical_25th_percentile_fallback"
    else:
        finite_mcp = primary["mcp_r_mass"].dropna()
        diagnostic_mcp = (
            float(finite_mcp.quantile(0.75))
            if len(finite_mcp)
            else float(config.reference_mcp_r_mass)
        )
        diagnostic_mcp_source = "all_rank1_empirical_75th_percentile_fallback"

    # The diagnostic near/far calculation is retained for review, but the
    # state-calling threshold is explicitly fixed by the current analysis
    # specification.
    selected_mcp = float(config.fixed_mcp_r_mass_threshold)
    mcp_source = "user_fixed_config_1.025"
    fixed_mcp_near_retention = (
        float(np.mean(near >= selected_mcp)) if len(near) else np.nan
    )

    sensitivity_rows: list[dict[str, Any]] = []
    finite_mtetr = primary["mTetR_r_mass"].dropna()
    sensitivity_low = max(0.80, float(finite_mtetr.quantile(0.05)) - 0.02)
    sensitivity_high = max(
        float(finite_mtetr.quantile(0.95)) + 0.02,
        config.reference_mtetr_r_mass + 0.05,
    )
    sensitivity_grid = np.unique(
        np.round(
            np.r_[
                np.linspace(sensitivity_low, sensitivity_high, 12),
                selected_mtetr,
                config.reference_mtetr_r_mass,
            ],
            4,
        )
    )
    mcp_grid = np.unique(
        np.round(
            np.r_[selected_mcp, config.reference_mcp_r_mass, 1.10, 1.20],
            4,
        )
    )
    for mtetr_threshold in sensitivity_grid:
        retained = primary.loc[primary["mTetR_r_mass"] >= mtetr_threshold]
        for mcp_threshold in mcp_grid:
            active = retained.loc[
                (retained["mcp_r_mass"] >= float(mcp_threshold))
                & (retained["mcp_distance_um"] <= config.mcp_near_distance_um)
            ]
            sensitivity_rows.append(
                {
                    "mTetR_r_mass_threshold": float(mtetr_threshold),
                    "mcp_r_mass_threshold": float(mcp_threshold),
                    "n_regions_retained": int(len(retained)),
                    "n_active": int(len(active)),
                    "active_fraction": (
                        float(len(active) / len(retained)) if len(retained) else np.nan
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows).drop_duplicates()
    sensitivity.to_csv(
        config.output_root / "tables" / "threshold_sensitivity.csv", index=False
    )

    decision = {
        "status": "first_pass_provisional",
        "population": "Cellpose QC-selected regions; GFP/CFP expression gate not used",
        "rank1_candidate_count": int(len(primary)),
        "rank1_calibration_count_crop_mean_at_least_90": int(
            len(calibration_primary)
        ),
        "mtetr_display_min_crop_mean": float(config.mtetr_display_min_crop_mean),
        "mtetr_crop_mean_split": crop_split,
        "mtetr_crop_mean_split_method": split_method,
        "mtetr_kde_crosspoint_raw": float(mtetr_raw) if np.isfinite(mtetr_raw) else None,
        "mtetr_kde_relative_density": (
            None
            if mtetr_artifact is None
            else float(mtetr_artifact["relative_density_at_crosspoint"])
        ),
        "mtetr_kde_reliable_and_plausible": bool(mtetr_reliable),
        "mtetr_kde_rank1_retention": mtetr_retention,
        "rank1_vs_secondary_kde_crosspoint_raw": (
            float(rank_secondary_raw) if np.isfinite(rank_secondary_raw) else None
        ),
        "rank1_vs_secondary_kde_reliable_and_plausible": bool(
            rank_secondary_reliable
        ),
        "rank1_vs_secondary_rank1_retention": rank_secondary_retention,
        "oyama_reference_mtetr_rank1_retention": reference_mtetr_retention,
        "selected_mtetr_r_mass": selected_mtetr,
        "selected_mtetr_source": mtetr_source,
        "mcp_kde_crosspoint_raw": float(mcp_raw) if np.isfinite(mcp_raw) else None,
        "mcp_kde_relative_density": (
            None
            if mcp_artifact is None
            else float(mcp_artifact["relative_density_at_crosspoint"])
        ),
        "mcp_kde_reliable_and_plausible": bool(mcp_reliable),
        "mcp_kde_near_retention": mcp_near_retention,
        "oyama_reference_mcp_near_retention": reference_mcp_near_retention,
        "mcp_diagnostic_suggested_threshold": diagnostic_mcp,
        "mcp_diagnostic_suggested_source": diagnostic_mcp_source,
        "fixed_mcp_near_retention": fixed_mcp_near_retention,
        "selected_mcp_r_mass": selected_mcp,
        "selected_mcp_source": mcp_source,
        "mcp_near_distance_um": config.mcp_near_distance_um,
        "mcp_near_distance_px_sora": config.mcp_near_distance_px,
        "oyama_reference": {
            "r_mTetR": config.reference_mtetr_r_mass,
            "r_MCP": config.reference_mcp_r_mass,
            "distance_px_at_0.13_um_per_px": config.mcp_near_distance_reference_px,
            "note": "Reference only; crop-mean split 1150 is not transferred across microscopes.",
        },
    }
    write_json(config.output_root / "threshold_decision.json", decision)
    artifacts = {
        "mtetr": mtetr_artifact,
        "mtetr_rank_vs_secondary": rank_secondary_artifact,
        "mcp": mcp_artifact,
        "primary": primary,
        "calibration_primary": calibration_primary,
        "mcp_population": mcp_population,
    }
    return decision, artifacts, sensitivity


def make_calibration_figures(
    candidates: pd.DataFrame,
    decision: dict[str, Any],
    artifacts: dict[str, Any],
    sensitivity: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, Path]:
    directories = prepare_directories(config)
    top5 = candidates.loc[
        candidates["mTetR_crop_mean"] >= config.mtetr_display_min_crop_mean
    ].dropna(
        subset=["mTetR_crop_mean", "mTetR_signal_intensity", "mTetR_r_mass"]
    )
    primary = artifacts["primary"]
    calibration_primary = artifacts["calibration_primary"]
    mtetr_artifact = artifacts["mtetr"]
    mcp_artifact = artifacts["mcp"]
    selected_mtetr = float(decision["selected_mtetr_r_mass"])
    selected_mcp = float(decision["selected_mcp_r_mass"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    if mtetr_artifact is not None:
        ax_a.plot(
            mtetr_artifact["grid"],
            mtetr_artifact["density_a"],
            label=(
                f"{config.mtetr_display_min_crop_mean:.0f} ≤ crop mean ≤ "
                f"{config.mtetr_lower_crop_mean_max:.0f} "
                f"(n={mtetr_artifact['n_a']})"
            ),
        )
        ax_a.plot(
            mtetr_artifact["grid"],
            mtetr_artifact["density_b"],
            label=(
                f"crop mean > {config.mtetr_lower_crop_mean_max:.0f} "
                f"(n={mtetr_artifact['n_b']})"
            ),
        )
        raw = mtetr_artifact["raw_crosspoint"]
        if np.isfinite(raw):
            ax_a.axvline(raw, color="0.25", linestyle="-.", label=f"KDE={raw:.3f}")
    else:
        low_values = calibration_primary.loc[
            calibration_primary["mTetR_crop_mean"]
            <= config.mtetr_lower_crop_mean_max,
            "mTetR_r_mass",
        ].dropna()
        high_values = calibration_primary.loc[
            calibration_primary["mTetR_crop_mean"]
            > config.mtetr_lower_crop_mean_max,
            "mTetR_r_mass",
        ].dropna()
        for values, label, color in (
            (
                low_values,
                (
                    f"{config.mtetr_display_min_crop_mean:.0f} ≤ crop mean ≤ "
                    f"{config.mtetr_lower_crop_mean_max:.0f} (n={len(low_values)})"
                ),
                "tab:blue",
            ),
            (
                high_values,
                (
                    f"crop mean > {config.mtetr_lower_crop_mean_max:.0f} "
                    f"(n={len(high_values)})"
                ),
                "tab:orange",
            ),
        ):
            if len(values):
                ax_a.hist(
                    values,
                    bins=min(30, max(3, int(np.sqrt(len(values))))),
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    color=color,
                    label=label,
                )
        ax_a.text(
            0.02,
            0.98,
            "KDE not estimated (one group n<20); empirical histogram shown",
            transform=ax_a.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.25",
        )
    ax_a.axvline(
        selected_mtetr, color="crimson", linestyle="--", label=f"selected={selected_mtetr:.3f}"
    )
    ax_a.axvline(
        config.reference_mtetr_r_mass,
        color="tab:purple",
        linestyle=":",
        label="Oyama reference=1.15",
    )
    ax_a.set(xlabel="r_mTetR", ylabel="Density", title="Rank-1 mTetR KDE")
    ax_a.legend(frameon=False, fontsize=8)

    secondary = top5.loc[~top5["is_primary_candidate"].astype(bool)]
    rank1 = top5.loc[top5["is_primary_candidate"].astype(bool)]
    ax_b.scatter(
        secondary["mTetR_crop_mean"],
        secondary["mTetR_signal_intensity"],
        s=6,
        alpha=0.2,
        color="0.4",
        label="ranks 2–5",
    )
    ax_b.scatter(
        rank1["mTetR_crop_mean"],
        rank1["mTetR_signal_intensity"],
        s=10,
        alpha=0.55,
        color="tab:blue",
        label="rank 1",
    )
    if len(top5):
        x_line = np.linspace(0, float(top5["mTetR_crop_mean"].quantile(0.995)) * 1.1, 200)
        ax_b.plot(
            x_line,
            selected_mtetr * x_line,
            "--",
            color="crimson",
            label=f"r={selected_mtetr:.3f}",
        )
        comparison_colors = ("tab:orange", "tab:green", "tab:brown")
        for threshold_value, color in zip(
            config.mtetr_plot_comparison_thresholds,
            comparison_colors,
        ):
            ax_b.plot(
                x_line,
                float(threshold_value) * x_line,
                ":",
                linewidth=1.6,
                color=color,
                label=f"r={float(threshold_value):.3f}",
            )
    ax_b.set(
        xlabel="mTetR crop mean",
        ylabel="mTetR spot signal intensity",
        title="mTetR top-5 candidates / region",
    )
    ax_b.set_xlim(left=config.mtetr_display_min_crop_mean)
    ax_b.legend(frameon=False, fontsize=8)

    mcp_population = artifacts["mcp_population"]
    if mcp_artifact is not None:
        ax_c.plot(
            mcp_artifact["grid"],
            mcp_artifact["density_a"],
            label=f"near ≤{config.mcp_near_distance_um:.2f} µm (n={mcp_artifact['n_a']})",
        )
        ax_c.plot(
            mcp_artifact["grid"],
            mcp_artifact["density_b"],
            label=f"far (n={mcp_artifact['n_b']})",
        )
        raw = mcp_artifact["raw_crosspoint"]
        if np.isfinite(raw):
            ax_c.axvline(raw, color="0.25", linestyle="-.", label=f"KDE={raw:.3f}")
    else:
        near_values = mcp_population.loc[
            mcp_population["mcp_distance_um"]
            <= config.mcp_near_distance_um,
            "mcp_r_mass",
        ].dropna()
        far_values = mcp_population.loc[
            mcp_population["mcp_distance_um"]
            > config.mcp_near_distance_um,
            "mcp_r_mass",
        ].dropna()
        for values, label, color in (
            (
                near_values,
                (
                    f"near ≤{config.mcp_near_distance_um:.2f} µm "
                    f"(n={len(near_values)})"
                ),
                "tab:blue",
            ),
            (
                far_values,
                f"far (n={len(far_values)})",
                "tab:orange",
            ),
        ):
            if len(values):
                ax_c.hist(
                    values,
                    bins=min(20, max(3, int(np.sqrt(len(values))))),
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    color=color,
                    label=label,
                )
        ax_c.text(
            0.02,
            0.98,
            "KDE not estimated (one group n<20); empirical histogram shown",
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.25",
        )
    ax_c.axvline(
        selected_mcp, color="crimson", linestyle="--", label=f"selected={selected_mcp:.3f}"
    )
    ax_c.set(xlabel="r_MCP", ylabel="Density", title="MCP near/far KDE")
    ax_c.legend(frameon=False, fontsize=8)

    if len(mcp_population):
        near = mcp_population["mcp_distance_um"] <= config.mcp_near_distance_um
        ax_d.scatter(
            mcp_population.loc[near, "mcp_distance_um"],
            mcp_population.loc[near, "mcp_r_mass"],
            s=10,
            alpha=0.55,
            label="near",
        )
        ax_d.scatter(
            mcp_population.loc[~near, "mcp_distance_um"],
            mcp_population.loc[~near, "mcp_r_mass"],
            s=10,
            alpha=0.3,
            label="far",
        )
    ax_d.axvline(config.mcp_near_distance_um, color="0.4", linestyle=":")
    ax_d.axhline(selected_mcp, color="crimson", linestyle="--")
    ax_d.set(
        xlabel="MCP distance from mTetR (µm)",
        ylabel="r_MCP",
        title="MCP distance and intensity",
    )
    ax_d.legend(frameon=False, fontsize=8)
    calibration_status = getattr(
        config,
        "threshold_figure_status_label",
        (
            "final fixed thresholds"
            if hasattr(config, "fixed_mtetr_r_mass_threshold")
            else "first-pass"
        ),
    )
    acquisition_mode_label = getattr(config, "acquisition_mode_label", "SoRa")
    fig.suptitle(
        f"{config.analysis_label}\n"
        f"Fixed-cell {acquisition_mode_label} threshold calibration "
        f"(GFP-independent; {calibration_status})",
        fontsize=14,
    )
    calibration_png = directories["figures"] / "threshold_calibration_four_panel.png"
    calibration_pdf = directories["figures"] / "threshold_calibration_four_panel.pdf"
    fig.savefig(calibration_png, dpi=220, bbox_inches="tight")
    fig.savefig(calibration_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    selected_mcp_rows = sensitivity.loc[
        np.isclose(sensitivity["mcp_r_mass_threshold"], selected_mcp)
    ]
    axes[0].plot(
        selected_mcp_rows["mTetR_r_mass_threshold"],
        selected_mcp_rows["n_regions_retained"],
        marker="o",
    )
    axes[1].plot(
        selected_mcp_rows["mTetR_r_mass_threshold"],
        selected_mcp_rows["active_fraction"],
        marker="o",
    )
    for axis in axes:
        axis.axvline(selected_mtetr, color="crimson", linestyle="--")
        axis.axvline(config.reference_mtetr_r_mass, color="tab:purple", linestyle=":")
        axis.set_xlabel("r_mTetR threshold")
    axes[0].set_ylabel("Regions retained")
    axes[1].set_ylabel("Active fraction")
    axes[0].set_title("mTetR threshold sensitivity")
    axes[1].set_title(f"Active fraction at r_MCP={selected_mcp:.3f}")
    sensitivity_png = directories["figures"] / "threshold_sensitivity.png"
    sensitivity_pdf = directories["figures"] / "threshold_sensitivity.pdf"
    fig.savefig(sensitivity_png, dpi=220, bbox_inches="tight")
    fig.savefig(sensitivity_pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "calibration_png": calibration_png,
        "calibration_pdf": calibration_pdf,
        "sensitivity_png": sensitivity_png,
        "sensitivity_pdf": sensitivity_pdf,
    }


def call_states(
    candidates: pd.DataFrame,
    decision: dict[str, Any],
    config: AnalysisConfig,
) -> pd.DataFrame:
    primary = candidates.loc[candidates["is_primary_candidate"].astype(bool)].copy()
    mtetr_threshold = float(decision["selected_mtetr_r_mass"])
    mcp_threshold = float(decision["selected_mcp_r_mass"])
    primary["eligible_mTetR_locus"] = (
        np.isfinite(primary["mTetR_r_mass"])
        & (primary["mTetR_r_mass"] >= mtetr_threshold)
    )
    primary["state"] = "Excluded_below_mTetR"
    active = (
        primary["eligible_mTetR_locus"]
        & np.isfinite(primary["mcp_r_mass"])
        & (primary["mcp_r_mass"] >= mcp_threshold)
        & np.isfinite(primary["mcp_distance_um"])
        & (primary["mcp_distance_um"] <= config.mcp_near_distance_um)
    )
    inactive = primary["eligible_mTetR_locus"] & ~active
    primary.loc[active, "state"] = "Active"
    primary.loc[inactive, "state"] = "Inactive"
    primary["mTetR_r_mass_threshold"] = mtetr_threshold
    primary["mcp_r_mass_threshold"] = mcp_threshold
    primary["mcp_distance_threshold_um"] = config.mcp_near_distance_um
    primary["state_call_method"] = (
        "Gao global top1 MCP r_mass and physical-distance rule; per-locus best Z"
    )
    primary.to_csv(config.output_root / "tables" / "locus_state_calls.csv", index=False)
    return primary


def _load_crop_lookup(
    fov_ids: Iterable[int], config: AnalysisConfig
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    cache_dir = config.output_root / "per_fov_cache"
    for fov_id in sorted(set(int(x) for x in fov_ids)):
        path = _fov_path(cache_dir, fov_id, "rank1_crops.npz")
        if not path.is_file():
            continue
        with np.load(path) as data:
            cell_ids = data["cell_ids"]
            locus_crops = data["locus_crops"]
            random_crops = data["random_crops"]
            for cell_id, locus, random in zip(cell_ids, locus_crops, random_crops):
                lookup[(fov_id, int(cell_id))] = (locus, random)
    return lookup


def _load_all_qc_random_crops(
    config: AnalysisConfig,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Load controls sampled from all QC-passing nuclei in each cached FOV."""

    random_crops: list[np.ndarray] = []
    population_rows: list[dict[str, Any]] = []
    separate_random_cache = config.output_root / "random_control_cache"
    cache_dir = (
        separate_random_cache
        if separate_random_cache.is_dir()
        else config.output_root / "per_fov_cache"
    )
    for path in sorted(cache_dir.glob("fov_*_rank1_crops.npz")):
        match = re.search(r"fov_(\d+)_rank1_crops\.npz$", path.name)
        if match is None:
            continue
        fov_id = int(match.group(1))
        with np.load(path) as data:
            if "random_qc_crops" not in data.files:
                raise RuntimeError(
                    "Missing all-QC random controls in revision cache: "
                    f"{path}"
                )
            controls = np.asarray(data["random_qc_crops"])
            cell_ids = np.asarray(data["random_qc_cell_ids"])
            if len(controls) != len(cell_ids):
                raise RuntimeError(
                    "Random-control crop/cell-ID count mismatch: "
                    f"{path}"
                )
            random_crops.extend(
                control.astype(np.float32) for control in controls
            )
            population_rows.append(
                {
                    "fov_id": fov_id,
                    "random_control_count": int(len(controls)),
                    "population": (
                        "all_QC_passing_nuclei_independent_of_mTetR_detection"
                    ),
                }
            )
    return random_crops, pd.DataFrame(population_rows)


def make_rank1_crop_qc(
    state_calls: pd.DataFrame,
    decision: dict[str, Any],
    config: AnalysisConfig,
) -> Path:
    threshold = float(decision["selected_mtetr_r_mass"])
    frame = state_calls.copy()
    finite_r_mass = frame["mTetR_r_mass"].dropna()
    interquartile_range = float(
        finite_r_mass.quantile(0.75) - finite_r_mass.quantile(0.25)
    )
    qc_band = max(0.0015, min(0.010, 0.25 * interquartile_range))
    frame["qc_stratum"] = np.select(
        [
            frame["mTetR_r_mass"] < threshold - qc_band,
            frame["mTetR_r_mass"].between(
                threshold - qc_band, threshold + qc_band
            ),
            frame["mTetR_r_mass"] > threshold + qc_band,
        ],
        ["below_threshold", "near_threshold", "above_threshold"],
        default="other",
    )
    rng = np.random.default_rng(config.random_seed)
    selections: list[pd.DataFrame] = []
    for stratum in ("below_threshold", "near_threshold", "above_threshold"):
        subset = frame.loc[frame["qc_stratum"] == stratum]
        if len(subset):
            indices = rng.choice(len(subset), size=min(4, len(subset)), replace=False)
            selections.append(subset.iloc[indices])
    selected = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    selected.to_csv(config.output_root / "tables" / "rank1_crop_qc_selection.csv", index=False)
    output = config.output_root / "qc" / "rank1_crop_qc.png"
    if selected.empty:
        return output
    lookup = _load_crop_lookup(selected["fov_id"].astype(int), config)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    plotted = 0
    for row in selected.itertuples(index=False):
        key = (int(row.fov_id), int(row.cell_id))
        if key not in lookup:
            continue
        crop = lookup[key][0].astype(np.float32)
        mtetr = robust_normalize(crop[config.mtetr_channel])
        mcp = robust_normalize(crop[config.mcp_channel])
        rgb = np.dstack([mtetr, mcp, np.zeros_like(mtetr)])
        axis = axes[plotted]
        axis.imshow(rgb)
        axis.plot(
            config.crop_radius_px,
            config.crop_radius_px,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="white",
        )
        if np.isfinite(row.mcp_x_crop_px):
            axis.plot(row.mcp_x_crop_px, row.mcp_y_crop_px, "+", color="cyan")
        axis.set_title(
            f"{row.qc_stratum}\nF{int(row.fov_id):03d} C{int(row.cell_id)} "
            f"r={row.mTetR_r_mass:.3f}",
            fontsize=8,
        )
        axis.axis("off")
        plotted += 1
    for axis in axes[plotted:]:
        axis.axis("off")
    fig.suptitle(
        "Rank-1 crop QC: mTetR red, MCP green\n"
        f"selected={threshold:.4f}; near band=±{qc_band:.4f}"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def _radial_profile(
    images: np.ndarray,
    pixel_um: float,
    r_max_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Gao's integer-radius bincount profile within the crop radius."""

    images = np.asarray(images, dtype=np.float32)
    if len(images) == 0:
        return np.array([]), np.empty((0, 0))
    size = images.shape[-1]
    center = (size - 1) / 2
    yy, xx = np.indices((size, size))
    radius_bin = np.sqrt((xx - center) ** 2 + (yy - center) ** 2).astype(
        np.int64
    )
    valid = radius_bin <= int(r_max_px)
    counts = np.bincount(
        radius_bin[valid].ravel(), minlength=int(r_max_px) + 1
    )
    profiles = np.full(
        (len(images), int(r_max_px) + 1), np.nan, dtype=np.float64
    )
    for index, image in enumerate(images):
        sums = np.bincount(
            radius_bin[valid].ravel(),
            weights=image[valid].ravel(),
            minlength=int(r_max_px) + 1,
        )
        profiles[index] = sums / np.maximum(counts, 1)
    radii_um = np.arange(int(r_max_px) + 1) * float(pixel_um)
    return radii_um, profiles


def _profile_sem(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.zeros(values.shape[1], dtype=float)
    return np.nanstd(values, axis=0, ddof=1) / np.sqrt(len(values))


def make_locus_summary(
    state_calls: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[Path, pd.DataFrame]:
    """Apply the Gao Fig.1c pixel-max normalization and composite plotting.

    The implementation follows cells 21, 25, 45, 47, and 51 of the pinned
    Gao notebook. One random nuclear point is sampled per region, its crop is
    taken from an independently sampled central Z plane, and the mean random
    crop is subtracted from every locus and random crop. For each channel, the
    same Active/Inactive scale sets the largest positive pixel in either
    baseline-subtracted aggregate image to 100.
    """

    eligible = state_calls.loc[
        state_calls["state"].isin(["Active", "Inactive"])
    ].copy()
    lookup = _load_crop_lookup(state_calls["fov_id"].astype(int), config)
    locus_by_state: dict[str, list[np.ndarray]] = {"Active": [], "Inactive": []}
    random_all, random_population = _load_all_qc_random_crops(config)
    for row in state_calls.itertuples(index=False):
        key = (int(row.fov_id), int(row.cell_id))
        if key not in lookup:
            continue
        locus, _ = lookup[key]
        if str(row.state) in locus_by_state:
            locus_by_state[str(row.state)].append(locus.astype(np.float32))
    if (
        not random_all
        or not locus_by_state["Active"]
        or not locus_by_state["Inactive"]
    ):
        raise RuntimeError("Insufficient cached crops for locus-centered summary.")

    random_stack = np.stack(random_all).astype(np.float32)
    random_mean = np.mean(random_stack, axis=0)
    state_stacks = {
        state: np.stack(crops).astype(np.float32)
        for state, crops in locus_by_state.items()
    }
    channel_specs = (
        (
            config.snap_channel,
            f"SNAPtag / {config.mintbody_label} mintbody",
        ),
        (config.mcp_channel, "MCP"),
        (config.mtetr_channel, "mTetR"),
    )
    state_colors = {
        "Active": "tab:orange",
        "Inactive": "tab:blue",
        "Random": "tab:green",
    }
    core_radius_px = config.gao_metric_core_radius_px
    annulus_inner_px, annulus_outer_px = config.gao_metric_annulus_px
    crop_shape = random_mean.shape[-2:]
    center = ((crop_shape[0] - 1) / 2, (crop_shape[1] - 1) / 2)
    yy, xx = np.indices(crop_shape)
    distance_px = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    core_mask = distance_px <= core_radius_px
    annulus_mask = (
        (distance_px >= annulus_inner_px)
        & (distance_px <= annulus_outer_px)
    )

    normalization_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scaled_images: dict[str, np.ndarray] = {}
    profile_data: dict[int, dict[str, Any]] = {}

    for channel, channel_label in channel_specs:
        active_aggregate = np.mean(
            state_stacks["Active"][:, channel], axis=0
        )
        inactive_aggregate = np.mean(
            state_stacks["Inactive"][:, channel], axis=0
        )
        active_subtracted = active_aggregate - random_mean[channel]
        inactive_subtracted = inactive_aggregate - random_mean[channel]
        pixel_max = float(
            np.nanmax(
                [
                    np.nanmax(active_subtracted),
                    np.nanmax(inactive_subtracted),
                ]
            )
        )
        scale = 100.0 / pixel_max if np.isfinite(pixel_max) and pixel_max > 0 else 1.0
        active_display = active_subtracted * scale
        inactive_display = inactive_subtracted * scale
        scaled_images[f"channel_{channel}_active"] = active_display
        scaled_images[f"channel_{channel}_inactive"] = inactive_display
        scaled_images[f"channel_{channel}_random_mean"] = random_mean[channel]
        normalization_rows.append(
            {
                "channel_index": int(channel),
                "channel": channel_label,
                "normalization_mode": "Gao_pixel_max",
                "random_control_population": (
                    "all_QC_passing_nuclei_independent_of_mTetR_detection"
                ),
                "random_control_count": int(len(random_stack)),
                "active_count": int(len(state_stacks["Active"])),
                "inactive_count": int(len(state_stacks["Inactive"])),
                "baseline_subtracted_positive_pixel_max_raw": pixel_max,
                "scale_to_positive_max_100": float(scale),
                "active_scaled_positive_max": float(np.nanmax(active_display)),
                "inactive_scaled_positive_max": float(
                    np.nanmax(inactive_display)
                ),
            }
        )

        channel_profiles: dict[str, np.ndarray] = {}
        for state in ("Active", "Inactive"):
            scaled_crops = (
                state_stacks[state][:, channel] - random_mean[channel]
            ) * scale
            radii_um, profiles = _radial_profile(
                scaled_crops,
                config.expected_xy_um,
                config.crop_radius_px,
            )
            channel_profiles[state] = profiles
            mean_profile = np.nanmean(profiles, axis=0)
            sem_profile = _profile_sem(profiles)
            for radius_um, mean_value, sem_value in zip(
                radii_um, mean_profile, sem_profile
            ):
                radial_rows.append(
                    {
                        "channel": channel_label,
                        "state": state,
                        "radius_um": float(radius_um),
                        "mean_scaled_intensity": float(mean_value),
                        "sem_scaled_intensity": float(sem_value),
                        "n": int(len(profiles)),
                    }
                )
            for scaled_crop in scaled_crops:
                metric_rows.append(
                    {
                        "channel": channel_label,
                        "state": state,
                        "core_minus_annulus": float(
                            np.nanmean(scaled_crop[core_mask])
                            - np.nanmean(scaled_crop[annulus_mask])
                        ),
                    }
                )

        scaled_random = (
            random_stack[:, channel] - random_mean[channel]
        ) * scale
        _, random_profiles = _radial_profile(
            scaled_random,
            config.expected_xy_um,
            config.crop_radius_px,
        )
        channel_profiles["Random"] = random_profiles
        random_mean_profile = np.nanmean(random_profiles, axis=0)
        random_sem_profile = _profile_sem(random_profiles)
        for radius_um, mean_value, sem_value in zip(
            radii_um, random_mean_profile, random_sem_profile
        ):
            radial_rows.append(
                {
                    "channel": channel_label,
                    "state": "Random",
                    "radius_um": float(radius_um),
                    "mean_scaled_intensity": float(mean_value),
                    "sem_scaled_intensity": float(sem_value),
                    "n": int(len(random_profiles)),
                }
            )
        profile_data[channel] = {
            "label": channel_label,
            "radii_um": radii_um,
            "profiles": channel_profiles,
        }

    normalization = pd.DataFrame(normalization_rows)
    radial_summary = pd.DataFrame(radial_rows)
    enrichment = pd.DataFrame(metric_rows)
    statistics_rows: list[dict[str, Any]] = []
    for channel_label in enrichment["channel"].unique():
        active = enrichment.loc[
            (enrichment["channel"] == channel_label)
            & (enrichment["state"] == "Active"),
            "core_minus_annulus",
        ].dropna()
        inactive = enrichment.loc[
            (enrichment["channel"] == channel_label)
            & (enrichment["state"] == "Inactive"),
            "core_minus_annulus",
        ].dropna()
        if len(active) and len(inactive):
            statistic, pvalue = mannwhitneyu(active, inactive, alternative="two-sided")
        else:
            statistic, pvalue = np.nan, np.nan
        statistics_rows.append(
            {
                "channel": channel_label,
                "n_active": int(len(active)),
                "n_inactive": int(len(inactive)),
                "active_median_core_minus_annulus": (
                    float(active.median()) if len(active) else np.nan
                ),
                "inactive_median_core_minus_annulus": (
                    float(inactive.median()) if len(inactive) else np.nan
                ),
                "mannwhitney_u": float(statistic),
                "pvalue_two_sided": float(pvalue),
            }
        )
    statistics = pd.DataFrame(statistics_rows)
    tables_dir = config.output_root / "tables"
    figures_dir = config.output_root / "figures"
    normalization.to_csv(
        tables_dir / "gao_fig1c_normalization.csv", index=False
    )
    random_population.to_csv(
        tables_dir / "gao_random_control_population_by_fov.csv",
        index=False,
    )
    radial_summary.to_csv(
        tables_dir / "gao_fig1c_radial_profiles.csv", index=False
    )
    enrichment.to_csv(
        tables_dir / "gao_fig1c_core_annulus_values.csv", index=False
    )
    statistics.to_csv(
        tables_dir / "gao_fig1c_core_annulus_statistics.csv", index=False
    )
    enrichment.to_csv(
        tables_dir / "locus_core_annulus_values.csv", index=False
    )
    statistics.to_csv(
        tables_dir / "locus_core_annulus_statistics.csv", index=False
    )
    np.savez_compressed(
        tables_dir / "gao_fig1c_scaled_images.npz",
        random_mean=random_mean,
        **scaled_images,
    )
    write_json(
        tables_dir / "gao_fig1c_method.json",
        {
            "github_repository": "Ochiai-Lab/gao2026_temporal",
            "github_commit": "e830aad3d1542f0c8cd8e555ae8cc06a16cef9ae",
            "github_notebook": (
                "analyses/01_Snapshot_analysis/notebooks/"
                "01_Snapshot_analysis.ipynb"
            ),
            "source_cells": [21, 25, 45, 47, 51],
            "random_control": (
                "one uniformly sampled xy point per final dilated nuclear "
                "label; independently sampled central Z plane; all available "
                "nuclei contribute to the random mean"
            ),
            "random_z_indices_inclusive": [
                config.random_z_min_index,
                config.random_z_max_index,
            ],
            "normalization": (
                "per channel: subtract the all-nucleus random-crop mean; "
                "scale Active and Inactive with one shared factor so their "
                "largest positive aggregate-image pixel equals 100"
            ),
            "radial_profile": (
                "integer-radius bins; every locus/random crop is random-mean "
                "subtracted and multiplied by the same channel scale"
            ),
            "metric": {
                "name": "core_minus_annulus",
                "reference_pixels_at_0.13_um_per_px": {
                    "core_radius": (
                        config.gao_metric_core_radius_reference_px
                    ),
                    "annulus": [
                        config.gao_metric_annulus_inner_reference_px,
                        config.gao_metric_annulus_outer_reference_px,
                    ],
                },
                "sora_pixels": {
                    "core_radius": core_radius_px,
                    "annulus": [
                        annulus_inner_px,
                        annulus_outer_px,
                    ],
                },
            },
        },
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(15, 11.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35, 1.1]},
    )
    for row_index, (channel, channel_label) in enumerate(channel_specs):
        active_display = scaled_images[f"channel_{channel}_active"]
        inactive_display = scaled_images[f"channel_{channel}_inactive"]
        vabs = max(
            float(np.nanmax(np.abs(active_display))),
            float(np.nanmax(np.abs(inactive_display))),
            1.0,
        )
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        image_active = axes[row_index, 0].imshow(
            active_display,
            cmap="coolwarm",
            norm=norm,
            origin="lower",
        )
        axes[row_index, 1].imshow(
            inactive_display,
            cmap="coolwarm",
            norm=norm,
            origin="lower",
        )
        axes[row_index, 0].set_title(
            f"{channel_label} Active\n"
            f"(N = {len(state_stacks['Active']):,})",
            loc="left",
        )
        axes[row_index, 1].set_title(
            f"{channel_label} Inactive\n"
            f"(N = {len(state_stacks['Inactive']):,})",
            loc="left",
        )
        for image_axis in axes[row_index, :2]:
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            image_axis.set_aspect("equal")
        colorbar = fig.colorbar(
            image_active,
            ax=[axes[row_index, 0], axes[row_index, 1]],
            orientation="horizontal",
            fraction=0.045,
            pad=0.04,
        )
        colorbar.set_label(
            "Intensity (random nuclear locus-subtracted, max=100)"
        )

        radial_axis = axes[row_index, 2]
        channel_profile = profile_data[channel]
        radii_um = channel_profile["radii_um"]
        for state in ("Active", "Inactive", "Random"):
            profiles = channel_profile["profiles"][state]
            mean_profile = np.nanmean(profiles, axis=0)
            sem_profile = _profile_sem(profiles)
            radial_axis.plot(
                radii_um,
                mean_profile,
                linewidth=2,
                color=state_colors[state],
                label=f"{state} (N={len(profiles):,})",
            )
            radial_axis.fill_between(
                radii_um,
                mean_profile - sem_profile,
                mean_profile + sem_profile,
                color=state_colors[state],
                alpha=0.22,
            )
        radial_axis.set_xlabel("Radius (µm)")
        radial_axis.set_ylabel("Intensity (max=100)")
        radial_axis.set_xlim(0, config.crop_radius_um)
        radial_axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        radial_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        radial_axis.grid(True, alpha=0.3)
        radial_axis.set_title(f"{channel_label} radial intensity")
        radial_axis.legend(frameon=False, fontsize=8)

        violin_axis = axes[row_index, 3]
        active_metric = enrichment.loc[
            (enrichment["channel"] == channel_label)
            & (enrichment["state"] == "Active"),
            "core_minus_annulus",
        ].to_numpy()
        inactive_metric = enrichment.loc[
            (enrichment["channel"] == channel_label)
            & (enrichment["state"] == "Inactive"),
            "core_minus_annulus",
        ].to_numpy()
        violin = violin_axis.violinplot(
            [active_metric, inactive_metric],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
            widths=0.8,
        )
        for body, color in zip(
            violin["bodies"],
            (state_colors["Active"], state_colors["Inactive"]),
        ):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.4)
        violin_axis.set_xticks([1, 2], ["Active", "Inactive"])
        violin_axis.set_ylabel(
            "Core − annulus intensity (max=100)"
        )
        violin_axis.set_title(
            f"{channel_label} core ≤{core_radius_px}px − "
            f"annulus {annulus_inner_px}–{annulus_outer_px}px"
        )
        violin_axis.grid(axis="y", alpha=0.3)
        _, pvalue = mannwhitneyu(
            active_metric, inactive_metric, alternative="two-sided"
        )
        violin_axis.text(
            0.5,
            0.98,
            f"MWU p={pvalue:.2e}",
            transform=violin_axis.transAxes,
            ha="center",
            va="top",
        )

    fig.suptitle(
        f"Gao et al. 2026 Fig.1c method — {config.analysis_label}\n"
        "random nuclear locus normalization; shared Active/Inactive "
        "pixel-max scaling",
        fontsize=14,
    )
    output = figures_dir / "gao_fig1c_method_composite.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        figures_dir / "locus_centered_state_averages.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        figures_dir / "locus_centered_state_averages.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    radial_fig, radial_axes = plt.subplots(
        1, 3, figsize=(13.5, 4.2), constrained_layout=True
    )
    for axis, (channel, channel_label) in zip(radial_axes, channel_specs):
        channel_profile = profile_data[channel]
        radii_um = channel_profile["radii_um"]
        for state in ("Active", "Inactive", "Random"):
            profiles = channel_profile["profiles"][state]
            mean_profile = np.nanmean(profiles, axis=0)
            sem_profile = _profile_sem(profiles)
            axis.plot(
                radii_um,
                mean_profile,
                linewidth=2,
                color=state_colors[state],
                label=f"{state} (N={len(profiles):,})",
            )
            axis.fill_between(
                radii_um,
                mean_profile - sem_profile,
                mean_profile + sem_profile,
                color=state_colors[state],
                alpha=0.22,
            )
        axis.set_xlim(0, config.crop_radius_um)
        axis.set_xlabel("Radius (µm)")
        axis.set_ylabel("Intensity (max=100)")
        axis.set_title(channel_label)
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False, fontsize=8)
    radial_fig.suptitle(
        "Gao et al. 2026 radial profiles — random nuclear locus normalized"
    )
    radial_path = figures_dir / "locus_radial_profiles.png"
    radial_fig.savefig(radial_path, dpi=300, bbox_inches="tight")
    radial_fig.savefig(radial_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(radial_fig)

    return output, statistics


def validate_outputs(
    config: AnalysisConfig,
    metadata: dict[str, Any],
    regions: pd.DataFrame,
    candidates: pd.DataFrame,
    state_calls: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    directories = prepare_directories(config)
    expected_fovs = int(metadata["sizes"]["P"])
    counts = {
        "segmentation_inputs": len(list(directories["inputs"].glob("fov_*_all_channels_mixed.png"))),
        "segmentation_masks": len(list(directories["masks"].glob("fov_*_selected_mask.tiff"))),
        "segmentation_masks_pre_dilation": len(
            list(
                directories["masks_pre_dilation"].glob(
                    "fov_*_selected_mask_pre_dilation.tiff"
                )
            )
        ),
        "segmentation_overlays": len(list(directories["overlays"].glob("fov_*_selected_regions.png"))),
        "spot_candidate_fov_tables": len(list(directories["cache"].glob("fov_*_spot_candidates.csv"))),
        "rank1_crop_archives": len(list(directories["cache"].glob("fov_*_rank1_crops.npz"))),
    }
    validations = {
        "all_expected_fov_inputs": counts["segmentation_inputs"] == expected_fovs,
        "all_expected_fov_masks": counts["segmentation_masks"] == expected_fovs,
        "all_expected_fov_pre_dilation_masks": (
            counts["segmentation_masks_pre_dilation"] == expected_fovs
        ),
        "all_expected_fov_overlays": counts["segmentation_overlays"] == expected_fovs,
        "all_expected_fov_quant_tables": counts["spot_candidate_fov_tables"] == expected_fovs,
        "all_expected_fov_crop_archives": counts["rank1_crop_archives"] == expected_fovs,
        "selected_regions_nonzero": bool(regions["selected"].astype(bool).sum() > 0),
        "rank1_candidates_nonzero": bool(
            candidates["is_primary_candidate"].astype(bool).sum() > 0
        ),
        "eligible_loci_nonzero": bool(
            state_calls["state"].isin(["Active", "Inactive"]).sum() > 0
        ),
        "gao_fig1c_composite_exists": (
            directories["figures"] / "gao_fig1c_method_composite.png"
        ).is_file(),
        "gao_radial_profiles_exist": (
            directories["figures"] / "locus_radial_profiles.png"
        ).is_file(),
        "gao_normalization_table_exists": (
            directories["tables"] / "gao_fig1c_normalization.csv"
        ).is_file(),
    }
    report = {
        "analysis": f"{config.analysis_label} fixed-cell SoRa snapshot",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {
            "nd2": getattr(nd2, "__version__", "unknown"),
            "trackpy": getattr(tp, "__version__", "unknown"),
            "torch": torch.__version__,
            "cellpose": getattr(cellpose, "version", "unknown"),
        },
        "github_reference": {
            "repository": "Ochiai-Lab/gao2026_temporal",
            "commit": "e830aad3d1542f0c8cd8e555ae8cc06a16cef9ae",
            "notebook": "analyses/01_Snapshot_analysis/notebooks/01_Snapshot_analysis.ipynb",
        },
        "method_adaptations": [
            "Fixed-cell analysis; no temporal linkage.",
            (
                "Each channel MIP is background-subtracted with Gaussian sigma "
                f"30 px, clipped to uint16, smoothed with sigma "
                f"{config.segmentation_signal_sigma_px_full:g} px, normalized, "
                "then equally mixed and downsampled by 4x anti-aliased bilinear "
                "resize for Cellpose."
            ),
            "Cellpose 3 nucleus-specific pretrained model is used.",
            (
                "QC-selected labels are expanded by "
                f"{config.segmentation_dilation_px_small} px in the 4x image "
                "with skimage.expand_labels; the expanded, non-overlapping "
                "labels are used by all downstream quantification."
            ),
            "Gao/Oyama spatial parameters are converted from 0.13 µm/px to SoRa physical units.",
            "mTetR candidate Z is selected per locus from the 11-plane stack.",
            (
                "Gao Fig.1c uses one uniformly sampled random nuclear locus per "
                "available nucleus, an independently sampled central Z plane, "
                "random-mean subtraction, and per-channel shared "
                "Active/Inactive pixel-max scaling to 100."
            ),
            "The Oyama crop-mean split of 1150 is not transferred across microscopes.",
            "mTetR calibration displays crop mean >=90 and defines lower as 90-102 inclusive.",
            (
                "r_MCP is user-fixed at "
                f"{config.fixed_mcp_r_mass_threshold:.3f}; near/far KDE is "
                "retained as a diagnostic only."
            ),
            "Thresholds are provisional until calibration plots and crop QC are biologically reviewed.",
        ],
        "counts": counts,
        "n_selected_regions": int(regions["selected"].astype(bool).sum()),
        "n_spot_candidates": int(len(candidates)),
        "n_rank1_candidates": int(candidates["is_primary_candidate"].astype(bool).sum()),
        "state_counts": state_calls["state"].value_counts().to_dict(),
        "threshold_decision": decision,
        "validations": validations,
        "all_validations_passed": bool(all(validations.values())),
        "config": asdict(config),
    }
    write_json(config.output_root / "analysis_manifest.json", report)
    if not report["all_validations_passed"]:
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Output validation failed: {failed}")
    return report


def load_cached_tables(config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = config.output_root / "tables"
    return (
        pd.read_csv(tables / "segmentation_regions_all_fovs.csv"),
        pd.read_csv(tables / "mtetr_mcp_spot_candidates.csv"),
        pd.read_csv(tables / "locus_state_calls.csv"),
    )


def config_manifest(config: AnalysisConfig) -> dict[str, Any]:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "config": asdict(config),
        "environment": {
            "python": sys.executable,
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "user": os.environ.get("USER"),
        },
    }
    write_json(config.output_root / "config_manifest.json", payload)
    return payload
