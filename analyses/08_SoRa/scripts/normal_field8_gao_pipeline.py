#!/usr/bin/env python3
"""Gao-compatible stage-1 analysis for ChamberA Field8 Normal images.

Scope:
1. Three-channel Cellpose nuclei segmentation.
2. Central-plane nuclear mean intensities.
3. Per-nucleus 3D mTetR spot detection and Gao r_mass quantification.
4. QC figures needed before choosing Normal-data thresholds.

The implementation follows Gao et al. for Gaussian projection, 3x3 XY median
filtering, per-nucleus mean/20 Big-FISH threshold, physical spot radius,
spot-centered crop, and trackpy COM intensity. Pixel-valued spatial parameters
are scaled from Gao's 130 nm reference to the measured 65 nm Normal pixels.
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
from pathlib import Path
from typing import Any, Iterable

import bigfish
import bigfish.detection as bigfish_detection
import cellpose
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import torch
import trackpy as tp
from PIL import Image, ImageDraw, ImageFont
from cellpose import models
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.transform import resize


sns.set_theme(context="notebook", style="whitegrid")
tp.quiet(suppress=True)

ALGORITHM_VERSION = "normal_field8_gao_stage1_v1"


@dataclass(frozen=True)
class AnalysisConfig:
    input_nd2: Path = Path(
        str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260716-SoRa_Fixed/260716_SoRa/260716_ChamberA/ChamberA_20260716_205221_Field8.nd2')
    )
    project_dir: Path = Path(
        str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO')
    )
    output_name: str = "normal_snapshot_outputs_chamberA_field8_gao_stage1"
    analysis_id: str = "nanog_ser5ph_rep1_normal"
    analysis_label: str = "Nanog × Ser5ph — rep1 (ChamberA Field8, Normal)"
    expected_fov_count: int = 100
    snap_channel: int = 0
    mtetr_channel: int = 1
    mcp_channel: int = 2
    expected_xy_um: float = 0.065
    expected_z_um: float = 0.5
    central_z_index: int = 5

    # Gao reference acquisition and physical segmentation settings.
    gao_reference_xy_um: float = 0.130
    segmentation_downsample: int = 2
    gao_gaussian_sigma_px: float = 5.0
    gao_cellpose_diameter_px: float = 100.0
    cellpose_model: str = "nuclei"
    cellpose_cellprob_threshold: float = 0.0
    cellpose_flow_threshold: float = 0.4
    gao_cellpose_min_size_px: int = 15
    saturation_fraction: float = 0.0035

    # Gao processed_1 and 3D mTetR detection.
    median_kernel_zyx: tuple[int, int, int] = (1, 3, 3)
    bigfish_spot_radius_nm: tuple[float, float, float] = (
        500.0,
        300.0,
        300.0,
    )
    bigfish_threshold_divisor: float = 20.0
    top_n_mtetr_candidates: int = 5

    # Gao crop radius=9 px, r_mass radius=3 px at 130 nm/px.
    gao_crop_radius_px: int = 9
    gao_r_mass_radius_px: int = 3
    overwrite_segmentation: bool = False
    overwrite_quantification: bool = False

    @property
    def output_root(self) -> Path:
        return self.project_dir / self.output_name

    @property
    def segmentation_gaussian_sigma_px(self) -> float:
        return (
            self.gao_gaussian_sigma_px
            * self.gao_reference_xy_um
            / self.segmentation_pixel_um
        )

    @property
    def cellpose_diameter_px(self) -> float:
        return (
            self.gao_cellpose_diameter_px
            * self.gao_reference_xy_um
            / self.segmentation_pixel_um
        )

    @property
    def cellpose_diameter_um(self) -> float:
        return self.cellpose_diameter_px * self.segmentation_pixel_um

    @property
    def segmentation_pixel_um(self) -> float:
        return self.expected_xy_um * self.segmentation_downsample

    @property
    def cellpose_min_size_px(self) -> int:
        """Scale Gao's pixel-area cutoff to the same physical area."""

        scale = self.gao_reference_xy_um / self.segmentation_pixel_um
        return int(round(self.gao_cellpose_min_size_px * scale**2))

    @property
    def crop_radius_px(self) -> int:
        return int(
            round(
                self.gao_crop_radius_px
                * self.gao_reference_xy_um
                / self.expected_xy_um
            )
        )

    @property
    def crop_radius_um(self) -> float:
        return self.crop_radius_px * self.expected_xy_um

    @property
    def r_mass_radius_px(self) -> int:
        return int(
            round(
                self.gao_r_mass_radius_px
                * self.gao_reference_xy_um
                / self.expected_xy_um
            )
        )

    @property
    def r_mass_radius_um(self) -> float:
        return self.r_mass_radius_px * self.expected_xy_um

    @property
    def voxel_size_nm(self) -> tuple[float, float, float]:
        return (
            self.expected_z_um * 1000.0,
            self.expected_xy_um * 1000.0,
            self.expected_xy_um * 1000.0,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_directories(config: AnalysisConfig) -> dict[str, Path]:
    root = config.output_root
    directories = {
        "root": root,
        "inputs": root / "segmentation_inputs",
        "raw_masks": root / "segmentation_raw_masks",
        "masks": root / "segmentation_masks",
        "overlays": root / "segmentation_overlays",
        "segmentation_cache": root / "segmentation_cache",
        "quantification_cache": root / "quantification_cache",
        "spot_overlays": root / "spot_overlays",
        "tables": root / "tables",
        "figures": root / "figures",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _plane_settings(description: str) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for plane_number in range(1, 4):
        start = description.find(f"Plane #{plane_number}:")
        end = description.find(f"Plane #{plane_number + 1}:")
        block = description[start : end if end >= 0 else None]

        def capture(pattern: str, cast: type) -> Any:
            match = re.search(pattern, block)
            return cast(match.group(1)) if match else None

        settings.append(
            {
                "channel_index": plane_number - 1,
                "laser_nm": capture(r"ExW:(\d+)", int),
                "laser_power_percent": capture(r"Power:\s*([\d.]+)", float),
                "exposure_ms": capture(r"Exposure:\s*(\d+)\s*ms", int),
            }
        )
    return settings


def collect_metadata(config: AnalysisConfig) -> dict[str, Any]:
    if not config.input_nd2.is_file():
        raise FileNotFoundError(config.input_nd2)
    with nd2.ND2File(config.input_nd2) as handle:
        sizes = {key: int(value) for key, value in handle.sizes.items()}
        voxel = handle.voxel_size()
        microscope = handle.metadata.channels[0].microscope
        channel_names = [
            channel.channel.name for channel in handle.metadata.channels
        ]
        text_info = dict(handle.text_info)
        dtype = str(handle.dtype)
    checks = {
        "fov_count": sizes.get("P") == config.expected_fov_count,
        "zcyx": (
            sizes.get("Z") == 11
            and sizes.get("C") == 3
            and sizes.get("Y") == sizes.get("X") == 2304
        ),
        "uint16": dtype == "uint16",
        "voxel": np.allclose(
            [voxel.x, voxel.y, voxel.z],
            [config.expected_xy_um, config.expected_xy_um, config.expected_z_um],
        ),
        "objective": (
            float(microscope.objectiveMagnification) == 100.0
            and float(microscope.objectiveNumericalAperture) == 1.45
        ),
        "normal_mode": (
            float(microscope.zoomMagnification) == 1.0
            and "sora" not in list(microscope.modalityFlags or [])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"ND2 metadata checks failed: {checks}")
    metadata = {
        "path": str(config.input_nd2),
        "sizes": sizes,
        "dtype": dtype,
        "voxel_size_um": {
            "x": float(voxel.x),
            "y": float(voxel.y),
            "z": float(voxel.z),
        },
        "objective": {
            "name": microscope.objectiveName,
            "magnification": float(microscope.objectiveMagnification),
            "na": float(microscope.objectiveNumericalAperture),
            "zoom": float(microscope.zoomMagnification),
            "modality_flags": list(microscope.modalityFlags or []),
        },
        "channel_names": channel_names,
        "channel_mapping": [
            {
                "index": config.snap_channel,
                "biological_label": "SNAPtag / Ser5ph mintbody",
            },
            {
                "index": config.mtetr_channel,
                "biological_label": "mTetR-3xmGold2s",
            },
            {
                "index": config.mcp_channel,
                "biological_label": "MCP-3xCFP",
            },
        ],
        "acquisition_settings": _plane_settings(
            str(text_info.get("description", ""))
        ),
        "checks": checks,
    }
    write_json(config.output_root / "nd2_metadata.json", metadata)
    return metadata


def config_manifest(config: AnalysisConfig) -> dict[str, Any]:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "config": asdict(config),
        "derived_parameters": {
            "segmentation_gaussian_sigma_px": (
                config.segmentation_gaussian_sigma_px
            ),
            "segmentation_gaussian_sigma_um": (
                config.segmentation_gaussian_sigma_px
                * config.segmentation_pixel_um
            ),
            "segmentation_downsample": config.segmentation_downsample,
            "segmentation_pixel_um": config.segmentation_pixel_um,
            "cellpose_diameter_px": config.cellpose_diameter_px,
            "cellpose_diameter_um": config.cellpose_diameter_um,
            "cellpose_min_size_px": config.cellpose_min_size_px,
            "crop_radius_px": config.crop_radius_px,
            "crop_size_px": 2 * config.crop_radius_px + 1,
            "crop_radius_um": config.crop_radius_um,
            "r_mass_radius_px": config.r_mass_radius_px,
            "r_mass_radius_um": config.r_mass_radius_um,
            "voxel_size_nm": config.voxel_size_nm,
            "bigfish_spot_radius_nm": config.bigfish_spot_radius_nm,
        },
        "method_alignment": {
            "gao_preserved": [
                "Gaussian-smoothed Z maximum projection for Cellpose",
                "Cellpose nuclei model and 13 um physical diameter",
                "3x3 XY median-filtered uint16 stack for quantification",
                "central-plane nuclear mTetR mean / 20 Big-FISH threshold",
                "Big-FISH voxel_size and 500/300/300 nm spot radius",
                "brightest in-mask 3D candidate per nucleus",
                "2.34 um-radius crop and 0.39 um r_mass radius",
                "spot signal = refined raw_mass / (pi * radius^2)",
            ],
            "normal_specific_changes": [
                "Cellpose input combines all three channels instead of Gao's two",
                "Cellpose input is 2x anti-aliased to Gao's 130 nm pixel grid",
                "3D detection and intensity quantification retain raw 65 nm pixels",
                "no fixed raw-intensity prefilter is applied before distribution QC",
            ],
            "not_used": [
                "SoRa Gaussian30 background subtraction",
                "SoRa Gaussian11 smoothing",
                "SoRa 4x downsample",
                "SoRa 15 px mask dilation",
                "SoRa bead-fitted LoG=(0.5,4,4) px",
            ],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cellpose": getattr(cellpose, "version", "unknown"),
            "bigfish": getattr(bigfish, "__version__", "unknown"),
            "torch": torch.__version__,
        },
    }
    write_json(config.output_root / "analysis_config.json", payload)
    return payload


def robust_normalize(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values[np.isfinite(values)], (
        low_percentile,
        high_percentile,
    ))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def gao_stretch(image: np.ndarray, saturation_fraction: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(
        values,
        [saturation_fraction * 100.0, 100.0 - saturation_fraction * 100.0],
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def load_position(handle: nd2.ND2File, fov_id: int) -> np.ndarray:
    stack = np.asarray(handle.asarray(position=int(fov_id)))
    if stack.ndim == 5:
        if stack.shape[0] != 1:
            raise ValueError(f"Unexpected T dimension: {stack.shape}")
        stack = stack[0]
    if stack.ndim != 4:
        raise ValueError(f"Expected ZCYX, got {stack.shape}")
    return np.asarray(stack, dtype=np.uint16)


def make_cellpose_input(
    stack_zcyx: np.ndarray,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Gao Gaussian+Zmax input, modified only to include all three channels."""

    # Gaussian filtering is linear, so filtering the three-channel sum is
    # numerically equivalent to filtering each channel and then summing, while
    # avoiding three full-resolution convolutions.
    combined_raw = stack_zcyx.astype(np.float32).sum(axis=1)
    factor = int(config.segmentation_downsample)
    output_shape = (
        combined_raw.shape[0],
        combined_raw.shape[1] // factor,
        combined_raw.shape[2] // factor,
    )
    combined_resampled = resize(
        combined_raw,
        output_shape=output_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    sigma = float(config.segmentation_gaussian_sigma_px)
    combined = gaussian_filter(
        combined_resampled,
        sigma=(0.0, sigma, sigma),
    )
    channel_mips_full = [
        stack_zcyx[:, channel_index].max(axis=0).astype(np.float32)
        for channel_index in range(3)
    ]
    channel_mips = [
        resize(
            mip,
            output_shape=output_shape[1:],
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
        for mip in channel_mips_full
    ]
    projection = gao_stretch(
        combined.max(axis=0), config.saturation_fraction
    )
    mixed_uint16 = np.rint(projection * np.iinfo(np.uint16).max).astype(
        np.uint16
    )
    rgb = np.moveaxis(
        np.stack([robust_normalize(mip) for mip in channel_mips]),
        0,
        -1,
    )
    return mixed_uint16, rgb


def _save_segmentation_overlay(
    path: Path,
    mixed: np.ndarray,
    rgb: np.ndarray,
    raw_masks: np.ndarray,
    selected_masks: np.ndarray,
    excluded_raw_labels: set[int],
    fov_id: int,
    config: AnalysisConfig,
) -> None:
    boundaries = find_boundaries(selected_masks, mode="outer")
    overlay = np.clip(rgb.copy(), 0.0, 1.0)
    overlay[boundaries] = (1.0, 1.0, 0.0)
    if excluded_raw_labels:
        excluded = np.isin(raw_masks, list(excluded_raw_labels))
        overlay[find_boundaries(excluded, mode="outer")] = (1.0, 0.0, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mixed, cmap="gray")
    axes[0].set_title(
        "Cellpose input\n3-channel projection, 130 nm/px Gao grid"
    )
    axes[1].imshow(rgb)
    axes[1].set_title("Raw MIP context\nR=SNAP, G=mTetR, B=MCP")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Selected nuclei (n={int(selected_masks.max())}); "
        "red=below Gao-equivalent min area"
    )
    for region in regionprops(selected_masks):
        y, x = region.centroid
        axes[2].text(
            x,
            y,
            str(region.label),
            color="white",
            fontsize=5,
            ha="center",
            va="center",
        )
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{config.analysis_label} — FOV {fov_id:03d}")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_segmentation(
    config: AnalysisConfig,
    fov_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    directories = prepare_directories(config)
    metadata = collect_metadata(config)
    effective_fovs = (
        list(range(int(metadata["sizes"]["P"])))
        if fov_ids is None
        else [int(value) for value in fov_ids]
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_gpu = True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        use_gpu = True
    else:
        device = torch.device("cpu")
        use_gpu = False
    model = models.CellposeModel(
        gpu=use_gpu,
        device=device,
        model_type=config.cellpose_model,
    )
    all_rows: list[pd.DataFrame] = []
    cellpose_evaluated = False
    with nd2.ND2File(config.input_nd2) as handle:
        for ordinal, fov_id in enumerate(effective_fovs, start=1):
            input_path = (
                directories["inputs"]
                / f"fov_{fov_id:03d}_three_channel_gao_projection.png"
            )
            mask_path = (
                directories["masks"]
                / f"fov_{fov_id:03d}_nuclei_mask.tiff"
            )
            raw_mask_path = (
                directories["raw_masks"]
                / f"fov_{fov_id:03d}_raw_nuclei_mask.tiff"
            )
            overlay_path = (
                directories["overlays"]
                / f"fov_{fov_id:03d}_nuclei_overlay.png"
            )
            table_path = (
                directories["segmentation_cache"]
                / f"fov_{fov_id:03d}_regions.csv"
            )
            cache_ready = all(
                path.is_file() and path.stat().st_size > 0
                for path in (
                    input_path,
                    raw_mask_path,
                    mask_path,
                    overlay_path,
                    table_path,
                )
            )
            if cache_ready and not config.overwrite_segmentation:
                all_rows.append(pd.read_csv(table_path))
                continue

            stack = load_position(handle, fov_id)
            mixed, rgb = make_cellpose_input(stack, config)
            Image.fromarray(mixed, mode="I;16").save(input_path)
            masks, _, _ = model.eval(
                mixed,
                channels=[0, 0],
                diameter=config.cellpose_diameter_px,
                normalize=True,
                flow_threshold=config.cellpose_flow_threshold,
                cellprob_threshold=config.cellpose_cellprob_threshold,
                min_size=config.cellpose_min_size_px,
                loop_run=cellpose_evaluated,
            )
            cellpose_evaluated = True
            raw_masks_small = np.asarray(masks, dtype=np.uint16)
            factor = int(config.segmentation_downsample)
            raw_masks = np.repeat(
                np.repeat(raw_masks_small, factor, axis=0),
                factor,
                axis=1,
            )[: stack.shape[-2], : stack.shape[-1]]
            tifffile.imwrite(raw_mask_path, raw_masks, compression="zlib")
            rows: list[dict[str, Any]] = []
            height, width = raw_masks_small.shape
            selected_masks_small = np.zeros_like(
                raw_masks_small, dtype=np.uint16
            )
            excluded_raw_labels: set[int] = set()
            next_label = 1
            minimum_area_um2 = (
                config.gao_cellpose_min_size_px
                * config.gao_reference_xy_um**2
            )
            for region in regionprops(raw_masks_small):
                min_y, min_x, max_y, max_x = region.bbox
                area_um2 = float(
                    region.area * config.segmentation_pixel_um**2
                )
                selected = area_um2 >= minimum_area_um2
                cell_id = next_label if selected else 0
                if selected:
                    selected_masks_small[
                        raw_masks_small == region.label
                    ] = next_label
                    next_label += 1
                else:
                    excluded_raw_labels.add(int(region.label))
                rows.append(
                    {
                        "fov_id": int(fov_id),
                        "raw_label": int(region.label),
                        "cell_id": int(cell_id),
                        "selected": bool(selected),
                        "exclusion_reason": (
                            "" if selected else "area_below_gao_physical_min"
                        ),
                        "area_px": int(region.area * factor**2),
                        "area_um2": area_um2,
                        "centroid_y_px": float(
                            (region.centroid[0] + 0.5) * factor - 0.5
                        ),
                        "centroid_x_px": float(
                            (region.centroid[1] + 0.5) * factor - 0.5
                        ),
                        "touches_border": bool(
                            min_y == 0
                            or min_x == 0
                            or max_y == height
                            or max_x == width
                        ),
                    }
                )
            selected_masks = np.repeat(
                np.repeat(selected_masks_small, factor, axis=0),
                factor,
                axis=1,
            )[: stack.shape[-2], : stack.shape[-1]]
            tifffile.imwrite(mask_path, selected_masks, compression="zlib")
            _save_segmentation_overlay(
                overlay_path,
                mixed,
                rgb,
                raw_masks_small,
                selected_masks_small,
                excluded_raw_labels,
                fov_id,
                config,
            )
            frame = pd.DataFrame(rows)
            frame.to_csv(table_path, index=False)
            all_rows.append(frame)
            if ordinal % 10 == 0 or ordinal == len(effective_fovs):
                print(
                    f"Segmentation {ordinal}/{len(effective_fovs)}; "
                    f"current selected nuclei={int(selected_masks.max())}"
                )
    regions = pd.concat(all_rows, ignore_index=True)
    regions.to_csv(
        directories["tables"] / "segmentation_regions_all_fovs.csv",
        index=False,
    )
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "fov_count": len(effective_fovs),
        "raw_region_count": int(len(regions)),
        "nuclei_count": int(regions["selected"].astype(bool).sum()),
        "device": str(device),
        "cellpose_model": config.cellpose_model,
        "cellpose_diameter_px": config.cellpose_diameter_px,
        "cellpose_diameter_um": config.cellpose_diameter_um,
        "gaussian_sigma_px": config.segmentation_gaussian_sigma_px,
        "gaussian_sigma_um": (
            config.segmentation_gaussian_sigma_px
            * config.segmentation_pixel_um
        ),
        "segmentation_downsample": config.segmentation_downsample,
        "segmentation_pixel_um": config.segmentation_pixel_um,
        "three_channel_combination": (
            "three-channel sum -> 2x anti-aliased resize to 130 nm/px -> "
            "Gaussian -> Z max"
        ),
        "mask_dilation": "none (Gao method)",
        "minimum_region_area_um2": (
            config.gao_cellpose_min_size_px
            * config.gao_reference_xy_um**2
        ),
    }
    write_json(config.output_root / "segmentation_summary.json", summary)
    return regions


def centered_crop(
    plane: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray | None:
    y0, y1 = center_y - radius, center_y + radius + 1
    x0, x1 = center_x - radius, center_x + radius + 1
    if y0 < 0 or x0 < 0 or y1 > plane.shape[0] or x1 > plane.shape[1]:
        return None
    crop = plane[y0:y1, x0:x1]
    expected = 2 * radius + 1
    return crop if crop.shape == (expected, expected) else None


def gao_r_mass(
    image: np.ndarray,
    center_yx: tuple[float, float],
    radius: int,
) -> dict[str, float] | None:
    values = np.asarray(image, dtype=np.float32)
    crop_mean = float(values.mean())
    if not np.isfinite(crop_mean) or crop_mean <= 0:
        return None
    padded = np.pad(values, pad_width=radius, mode="constant")
    coordinates = np.asarray([center_yx], dtype=float) + radius
    refined = tp.refine_com(
        padded, padded, radius=radius, coords=coordinates
    )
    if refined is None or refined.empty:
        return None
    mass_column = "raw_mass" if "raw_mass" in refined.columns else "mass"
    signal = float(refined[mass_column].iloc[0]) / (
        math.pi * radius**2
    )
    return {
        "mTetR_signal_intensity": signal,
        "mTetR_crop_mean": crop_mean,
        "r_mTetR_crop": signal / crop_mean,
    }


QUANT_COLUMNS = [
    "fov_id",
    "cell_id",
    "candidate_rank",
    "is_primary",
    "candidate_count_3d",
    "locus_z_index",
    "locus_y_px",
    "locus_x_px",
    "mTetR_local_peak_intensity",
    "mTetR_nuclear_mean_central_z",
    "bigfish_threshold",
    "mTetR_signal_intensity",
    "mTetR_crop_mean",
    "r_mTetR_crop",
    "r_mTetR_nuclear",
    "valid_crop",
]


def _quantify_fov(
    handle: nd2.ND2File,
    fov_id: int,
    config: AnalysisConfig,
    directories: dict[str, Path],
) -> pd.DataFrame:
    table_path = (
        directories["quantification_cache"]
        / f"fov_{fov_id:03d}_mtetr_candidates.csv"
    )
    overlay_path = (
        directories["spot_overlays"]
        / f"fov_{fov_id:03d}_mtetr_spots.png"
    )
    if (
        table_path.is_file()
        and table_path.stat().st_size > 0
        and overlay_path.is_file()
        and not config.overwrite_quantification
    ):
        return pd.read_csv(table_path)

    mask_path = (
        directories["masks"] / f"fov_{fov_id:03d}_nuclei_mask.tiff"
    )
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = tifffile.imread(mask_path)
    dask_array = handle.to_dask()
    mtetr_raw = np.asarray(
        dask_array[fov_id, :, config.mtetr_channel].compute(),
        dtype=np.uint16,
    )
    mtetr = np.asarray(
        median_filter(mtetr_raw, size=config.median_kernel_zyx),
        dtype=np.uint16,
    )
    rows: list[dict[str, Any]] = []
    primary_points: list[tuple[int, int, int, int]] = []
    for region in regionprops(mask):
        cell_id = int(region.label)
        region_mask = mask == cell_id
        nuclear_mean = float(
            mtetr[config.central_z_index][region_mask].mean()
        )
        threshold = nuclear_mean / config.bigfish_threshold_divisor
        min_y, min_x, max_y, max_x = region.bbox
        local = mtetr[:, min_y:max_y, min_x:max_x]
        local_mask = region_mask[min_y:max_y, min_x:max_x]
        spots = bigfish_detection.detect_spots(
            images=local,
            threshold=threshold,
            return_threshold=False,
            voxel_size=config.voxel_size_nm,
            spot_radius=config.bigfish_spot_radius_nm,
        )
        spots = np.asarray(spots, dtype=int).reshape((-1, 3))
        if len(spots):
            spots[:, 1] += min_y
            spots[:, 2] += min_x
            inside = (
                mask[
                    np.clip(spots[:, 1], 0, mask.shape[0] - 1),
                    np.clip(spots[:, 2], 0, mask.shape[1] - 1),
                ]
                == cell_id
            )
            spots = spots[inside]
        if len(spots) == 0:
            row = {column: np.nan for column in QUANT_COLUMNS}
            row.update(
                {
                    "fov_id": fov_id,
                    "cell_id": cell_id,
                    "candidate_rank": 0,
                    "is_primary": False,
                    "candidate_count_3d": 0,
                    "mTetR_nuclear_mean_central_z": nuclear_mean,
                    "bigfish_threshold": threshold,
                    "valid_crop": False,
                }
            )
            rows.append(row)
            continue
        intensities = mtetr[spots[:, 0], spots[:, 1], spots[:, 2]]
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
        for rank, (spot, peak) in enumerate(
            zip(
                spots[: config.top_n_mtetr_candidates],
                intensities[: config.top_n_mtetr_candidates],
            ),
            start=1,
        ):
            z_index, y_px, x_px = map(int, spot)
            crop = centered_crop(
                mtetr[z_index],
                y_px,
                x_px,
                config.crop_radius_px,
            )
            metrics = None
            if crop is not None:
                try:
                    center = (
                        float(config.crop_radius_px),
                        float(config.crop_radius_px),
                    )
                    metrics = gao_r_mass(
                        crop, center, config.r_mass_radius_px
                    )
                except Exception:
                    metrics = None
            row = {column: np.nan for column in QUANT_COLUMNS}
            row.update(
                {
                    "fov_id": fov_id,
                    "cell_id": cell_id,
                    "candidate_rank": rank,
                    "is_primary": rank == 1,
                    "candidate_count_3d": len(spots),
                    "locus_z_index": z_index,
                    "locus_y_px": y_px,
                    "locus_x_px": x_px,
                    "mTetR_local_peak_intensity": float(peak),
                    "mTetR_nuclear_mean_central_z": nuclear_mean,
                    "bigfish_threshold": threshold,
                    "valid_crop": metrics is not None,
                }
            )
            if metrics is not None:
                row.update(metrics)
                row["r_mTetR_nuclear"] = (
                    metrics["mTetR_signal_intensity"] / nuclear_mean
                    if nuclear_mean > 0
                    else np.nan
                )
            rows.append(row)
            if rank == 1:
                primary_points.append((z_index, y_px, x_px, cell_id))

    frame = pd.DataFrame(rows, columns=QUANT_COLUMNS)
    frame.to_csv(table_path, index=False)
    _save_spot_overlay(
        overlay_path, mtetr, mask, primary_points, fov_id, config
    )
    return frame


def _save_spot_overlay(
    path: Path,
    mtetr: np.ndarray,
    mask: np.ndarray,
    points: list[tuple[int, int, int, int]],
    fov_id: int,
    config: AnalysisConfig,
) -> None:
    mip = robust_normalize(mtetr.max(axis=0))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mip, cmap="gray")
    ax.contour(mask > 0, levels=[0.5], colors="cyan", linewidths=0.35)
    if points:
        z = np.asarray([point[0] for point in points])
        y = np.asarray([point[1] for point in points])
        x = np.asarray([point[2] for point in points])
        scatter = ax.scatter(
            x,
            y,
            c=z,
            s=20,
            cmap="viridis",
            vmin=0,
            vmax=10,
            edgecolors="white",
            linewidths=0.3,
        )
        fig.colorbar(scatter, ax=ax, label="mTetR locus Z index")
    ax.set_title(
        f"{config.analysis_label}\nFOV {fov_id:03d}: primary 3D mTetR loci "
        f"(n={len(points)})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_mtetr_quantification(
    config: AnalysisConfig,
    fov_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    directories = prepare_directories(config)
    metadata = collect_metadata(config)
    effective_fovs = (
        list(range(int(metadata["sizes"]["P"])))
        if fov_ids is None
        else [int(value) for value in fov_ids]
    )
    frames: list[pd.DataFrame] = []
    with nd2.ND2File(config.input_nd2) as handle:
        for ordinal, fov_id in enumerate(effective_fovs, start=1):
            frames.append(
                _quantify_fov(handle, fov_id, config, directories)
            )
            if ordinal % 10 == 0 or ordinal == len(effective_fovs):
                print(
                    f"mTetR quantification {ordinal}/{len(effective_fovs)}"
                )
    candidates = pd.concat(frames, ignore_index=True)
    candidates.to_csv(
        directories["tables"] / "mtetr_candidates_all_fovs.csv",
        index=False,
    )
    rank1 = candidates[
        candidates["is_primary"].fillna(False).astype(bool)
    ].copy()
    rank1.to_csv(
        directories["tables"] / "mtetr_rank1_all_fovs.csv",
        index=False,
    )
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "fov_count": len(effective_fovs),
        "nuclear_rows": int(
            candidates[["fov_id", "cell_id"]].drop_duplicates().shape[0]
        ),
        "rank1_count": int(len(rank1)),
        "valid_rank1_count": int(rank1["valid_crop"].sum()),
        "no_spot_count": int(
            (
                candidates.groupby(["fov_id", "cell_id"])[
                    "candidate_count_3d"
                ].max()
                == 0
            ).sum()
        ),
        "median_filter": config.median_kernel_zyx,
        "voxel_size_nm": config.voxel_size_nm,
        "spot_radius_nm": config.bigfish_spot_radius_nm,
        "threshold": "central-plane nuclear mTetR mean / 20",
    }
    write_json(config.output_root / "mtetr_quantification_summary.json", summary)
    return candidates


def make_contact_sheet(
    paths: list[Path],
    output_path: Path,
    title: str,
    columns: int = 2,
    width_px: int = 2200,
) -> Path:
    images = [Image.open(path).convert("RGB") for path in paths if path.is_file()]
    if not images:
        raise RuntimeError(f"No images for contact sheet: {title}")
    cell_width = width_px // columns
    resized: list[Image.Image] = []
    for image in images:
        ratio = cell_width / image.width
        resized.append(
            image.resize(
                (cell_width, int(round(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        )
    rows = math.ceil(len(resized) / columns)
    row_heights = [
        max(
            resized[index].height
            for index in range(row * columns, min((row + 1) * columns, len(resized)))
        )
        for row in range(rows)
    ]
    title_height = 70
    canvas = Image.new(
        "RGB",
        (width_px, title_height + sum(row_heights)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), title, fill="black", font=ImageFont.load_default())
    y_offset = title_height
    for index, image in enumerate(resized):
        row = index // columns
        column = index % columns
        canvas.paste(image, (column * cell_width, y_offset))
        if column == columns - 1 or index == len(resized) - 1:
            y_offset += row_heights[row]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def make_segmentation_contact_sheet(
    config: AnalysisConfig,
    fov_ids: Iterable[int] = (0, 19, 39, 59, 79, 99),
) -> Path:
    directories = prepare_directories(config)
    paths = [
        directories["overlays"] / f"fov_{int(fov):03d}_nuclei_overlay.png"
        for fov in fov_ids
    ]
    return make_contact_sheet(
        paths,
        directories["figures"] / "segmentation_representative_contact_sheet.png",
        f"{config.analysis_label}: representative segmentation",
        columns=2,
    )


def make_spot_contact_sheet(
    config: AnalysisConfig,
    fov_ids: Iterable[int] = (0, 19, 39, 59, 79, 99),
) -> Path:
    directories = prepare_directories(config)
    paths = [
        directories["spot_overlays"] / f"fov_{int(fov):03d}_mtetr_spots.png"
        for fov in fov_ids
    ]
    return make_contact_sheet(
        paths,
        directories["figures"] / "mtetr_spot_representative_contact_sheet.png",
        f"{config.analysis_label}: representative 3D mTetR loci",
        columns=2,
    )


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def make_mtetr_distribution_figures(
    config: AnalysisConfig,
    candidates: pd.DataFrame | None = None,
) -> dict[str, Path]:
    directories = prepare_directories(config)
    if candidates is None:
        candidates = pd.read_csv(
            directories["tables"] / "mtetr_candidates_all_fovs.csv"
        )
    rank1 = candidates[
        candidates["is_primary"].fillna(False).astype(bool)
        & candidates["valid_crop"].fillna(False).astype(bool)
    ].copy()
    if rank1.empty:
        raise RuntimeError("No valid rank-1 mTetR crops")
    all_nuclei = (
        candidates.sort_values(
            ["fov_id", "cell_id", "candidate_rank"]
        )
        .drop_duplicates(["fov_id", "cell_id"])
        .copy()
    )
    all_nuclei["spot_detected"] = (
        pd.to_numeric(
            all_nuclei["candidate_count_3d"], errors="coerce"
        ).fillna(0)
        > 0
    )

    figure_path = directories["figures"] / "mtetr_intensity_distributions.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sns.histplot(
        data=all_nuclei,
        x="mTetR_nuclear_mean_central_z",
        hue="spot_detected",
        bins=50,
        kde=True,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        "Central-plane nuclear mTetR mean\nall nuclei, split by 3D detection"
    )
    sns.histplot(
        data=rank1,
        x="mTetR_signal_intensity",
        bins=50,
        kde=True,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("mTetR spot signal intensity")
    sns.histplot(
        data=rank1,
        x="r_mTetR_crop",
        bins=50,
        kde=True,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("r_mTetR = spot signal / crop mean")
    sns.histplot(
        data=rank1,
        x="r_mTetR_nuclear",
        bins=50,
        kde=True,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Spot signal / central-plane nuclear mean")
    fig.suptitle(
        f"{config.analysis_label}\nGao-compatible Normal mTetR distributions "
        f"(valid rank-1 n={len(rank1)})"
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    scatter_path = (
        directories["figures"] / "mtetr_signal_vs_background.png"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.scatterplot(
        data=rank1,
        x="mTetR_crop_mean",
        y="mTetR_signal_intensity",
        hue="r_mTetR_crop",
        palette="viridis",
        s=18,
        alpha=0.6,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Spot signal vs spot-centered crop mean")
    sns.scatterplot(
        data=rank1,
        x="mTetR_nuclear_mean_central_z",
        y="mTetR_signal_intensity",
        hue="r_mTetR_nuclear",
        palette="magma",
        s=18,
        alpha=0.6,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Spot signal vs nuclear mean")
    fig.suptitle(
        f"{config.analysis_label}\nNo fixed Normal r_mTetR threshold applied"
    )
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    top5_path = directories["figures"] / "mtetr_top5_candidate_qc.png"
    valid_candidates = candidates[
        candidates["valid_crop"].fillna(False).astype(bool)
        & (pd.to_numeric(candidates["candidate_rank"], errors="coerce") > 0)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.boxplot(
        data=valid_candidates,
        x="candidate_rank",
        y="r_mTetR_crop",
        showfliers=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=valid_candidates.sample(
            min(3000, len(valid_candidates)), random_state=20260720
        ),
        x="candidate_rank",
        y="r_mTetR_crop",
        color="black",
        size=1.5,
        alpha=0.25,
        ax=axes[0],
    )
    axes[0].set_title("r_mTetR by 3D candidate rank")
    sns.histplot(
        data=rank1,
        x="candidate_count_3d",
        bins=40,
        ax=axes[1],
    )
    axes[1].set_title("3D candidate count per nucleus")
    fig.suptitle(f"{config.analysis_label}: candidate QC")
    fig.tight_layout()
    fig.savefig(top5_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "nuclei_count": int(len(all_nuclei)),
        "spot_detected_nuclei": int(all_nuclei["spot_detected"].sum()),
        "spot_detection_fraction": float(all_nuclei["spot_detected"].mean()),
        "valid_rank1_count": int(len(rank1)),
        "nuclear_mean": {
            "population": "all segmented nuclei",
            "median": float(np.median(_finite(all_nuclei, "mTetR_nuclear_mean_central_z"))),
            "p05": float(np.percentile(_finite(all_nuclei, "mTetR_nuclear_mean_central_z"), 5)),
            "p95": float(np.percentile(_finite(all_nuclei, "mTetR_nuclear_mean_central_z"), 95)),
        },
        "spot_signal": {
            "median": float(np.median(_finite(rank1, "mTetR_signal_intensity"))),
            "p05": float(np.percentile(_finite(rank1, "mTetR_signal_intensity"), 5)),
            "p95": float(np.percentile(_finite(rank1, "mTetR_signal_intensity"), 95)),
        },
        "r_mTetR_crop": {
            "median": float(np.median(_finite(rank1, "r_mTetR_crop"))),
            "p05": float(np.percentile(_finite(rank1, "r_mTetR_crop"), 5)),
            "p95": float(np.percentile(_finite(rank1, "r_mTetR_crop"), 95)),
        },
        "r_mTetR_nuclear": {
            "median": float(np.median(_finite(rank1, "r_mTetR_nuclear"))),
            "p05": float(np.percentile(_finite(rank1, "r_mTetR_nuclear"), 5)),
            "p95": float(np.percentile(_finite(rank1, "r_mTetR_nuclear"), 95)),
        },
    }
    write_json(config.output_root / "mtetr_distribution_summary.json", summary)
    return {
        "distributions": figure_path,
        "signal_vs_background": scatter_path,
        "top5_qc": top5_path,
    }


def validate_stage1(config: AnalysisConfig) -> dict[str, Any]:
    directories = prepare_directories(config)
    metadata = collect_metadata(config)
    regions = pd.read_csv(
        directories["tables"] / "segmentation_regions_all_fovs.csv"
    )
    candidates = pd.read_csv(
        directories["tables"] / "mtetr_candidates_all_fovs.csv"
    )
    rank1 = pd.read_csv(
        directories["tables"] / "mtetr_rank1_all_fovs.csv"
    )
    expected_fovs = int(metadata["sizes"]["P"])
    checks = {
        "metadata_all_pass": all(metadata["checks"].values()),
        "segmentation_inputs_complete": (
            len(list(directories["inputs"].glob("fov_*.png")))
            == expected_fovs
        ),
        "segmentation_masks_complete": (
            len(list(directories["masks"].glob("fov_*.tiff")))
            == expected_fovs
        ),
        "segmentation_raw_masks_complete": (
            len(list(directories["raw_masks"].glob("fov_*.tiff")))
            == expected_fovs
        ),
        "segmentation_overlays_complete": (
            len(list(directories["overlays"].glob("fov_*.png")))
            == expected_fovs
        ),
        "segmentation_all_fovs_present": (
            regions["fov_id"].nunique() == expected_fovs
        ),
        "quantification_cache_complete": (
            len(
                list(
                    directories["quantification_cache"].glob(
                        "fov_*_mtetr_candidates.csv"
                    )
                )
            )
            == expected_fovs
        ),
        "spot_overlays_complete": (
            len(list(directories["spot_overlays"].glob("fov_*.png")))
            == expected_fovs
        ),
        "all_nuclei_have_quantification_row": (
            candidates[["fov_id", "cell_id"]].drop_duplicates().shape[0]
            == int(regions["selected"].astype(bool).sum())
        ),
        "rank1_unique_per_nucleus": (
            not rank1.duplicated(["fov_id", "cell_id"]).any()
        ),
        "valid_rank1_values_finite": bool(
            np.isfinite(
                rank1.loc[
                    rank1["valid_crop"].astype(bool),
                    [
                        "mTetR_signal_intensity",
                        "mTetR_crop_mean",
                        "mTetR_nuclear_mean_central_z",
                        "r_mTetR_crop",
                        "r_mTetR_nuclear",
                    ],
                ].to_numpy(float)
            ).all()
        ),
        "required_figures_exist": all(
            (
                directories["figures"] / name
            ).is_file()
            for name in (
                "segmentation_representative_contact_sheet.png",
                "mtetr_spot_representative_contact_sheet.png",
                "mtetr_intensity_distributions.png",
                "mtetr_signal_vs_background.png",
                "mtetr_top5_candidate_qc.png",
            )
        ),
    }
    payload = {
        "all_checks_passed": bool(all(checks.values())),
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "checks": checks,
        "counts": {
            "fovs": expected_fovs,
            "nuclei": int(regions["selected"].astype(bool).sum()),
            "rank1": int(len(rank1)),
            "valid_rank1": int(rank1["valid_crop"].sum()),
        },
    }
    write_json(config.output_root / "stage1_validation.json", payload)
    if not payload["all_checks_passed"]:
        raise RuntimeError(f"Stage-1 validation failed: {checks}")
    return payload
