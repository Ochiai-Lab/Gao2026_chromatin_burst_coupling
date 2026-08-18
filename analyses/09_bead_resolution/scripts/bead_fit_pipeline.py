#!/usr/bin/env python3
"""Fit 200 nm bead images acquired by conventional CSU-W1 or SoRa.

The primary estimates use the same 3x3 XY median filter as Gao et al. The raw
stack is fitted in parallel only to quantify how much that nonlinear filter
changes apparent spot width. The script writes fit-level data, diagnostic PNGs,
and parameter recommendations converted to the sample voxel size.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import least_squares
from skimage.feature import peak_local_max


FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
BEAD_RADIUS_NM = 100.0
BEAD_AXIS_SIGMA_NM = BEAD_RADIUS_NM / math.sqrt(5.0)
MIN_FIT_R_SQUARED = 0.80


@dataclass(frozen=True)
class OpticalMetadata:
    path: str
    objective_name: str
    objective_magnification: float
    objective_na: float
    zoom_magnification: float
    sora: bool
    voxel_z_nm: float
    voxel_y_nm: float
    voxel_x_nm: float
    channel_names: tuple[str, ...]


def odd_ceil(value: float, minimum: int = 3) -> int:
    """Smallest odd integer greater than or equal to value."""
    integer = max(minimum, int(math.ceil(value)))
    return integer if integer % 2 == 1 else integer + 1


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, np.finfo(np.float64).eps)


def read_optical_metadata(path: Path) -> OpticalMetadata:
    with nd2.ND2File(path) as image:
        channels = image.metadata.channels
        first = channels[0].microscope
        voxel = image.voxel_size()
        modality = {str(flag).lower() for flag in (first.modalityFlags or [])}
        return OpticalMetadata(
            path=str(path),
            objective_name=str(first.objectiveName),
            objective_magnification=float(first.objectiveMagnification),
            objective_na=float(first.objectiveNumericalAperture),
            zoom_magnification=float(first.zoomMagnification),
            sora=any("sora" in flag for flag in modality),
            voxel_z_nm=float(voxel.z) * 1000.0,
            voxel_y_nm=float(voxel.y) * 1000.0,
            voxel_x_nm=float(voxel.x) * 1000.0,
            channel_names=tuple(str(channel.channel.name) for channel in channels),
        )


def load_channel_zyx(path: Path, channel_index: int) -> np.ndarray:
    with nd2.ND2File(path) as image:
        sizes = tuple(image.sizes)
        if sizes != ("Z", "C", "Y", "X"):
            raise ValueError(f"Expected Z,C,Y,X bead data, got sizes={sizes}")
        array = np.asarray(image.to_dask()[:, channel_index].compute())
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D bead channel stack, got {array.shape}")
    return array


def gaussian3d(
    parameters: np.ndarray,
    z_grid: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    background, amplitude, z0, y0, x0, sigma_z, sigma_y, sigma_x = parameters
    exponent = (
        ((z_grid - z0) / sigma_z) ** 2
        + ((y_grid - y0) / sigma_y) ** 2
        + ((x_grid - x0) / sigma_x) ** 2
    )
    return background + amplitude * np.exp(-0.5 * exponent)


def fit_gaussian3d(patch: np.ndarray) -> dict[str, float] | None:
    data = np.asarray(patch, dtype=np.float64)
    z_size, y_size, x_size = data.shape
    z_grid, y_grid, x_grid = np.indices(data.shape, dtype=np.float64)
    maximum_index = np.unravel_index(int(np.argmax(data)), data.shape)
    background0 = float(np.percentile(data, 10.0))
    amplitude0 = max(float(data.max()) - background0, 1.0)
    initial = np.array(
        [
            background0,
            amplitude0,
            float(maximum_index[0]),
            float(maximum_index[1]),
            float(maximum_index[2]),
            1.2,
            3.0,
            3.0,
        ],
        dtype=np.float64,
    )
    lower = np.array(
        [
            0.0,
            0.0,
            max(0.0, maximum_index[0] - 2.5),
            max(0.0, maximum_index[1] - 3.0),
            max(0.0, maximum_index[2] - 3.0),
            0.25,
            0.45,
            0.45,
        ]
    )
    upper = np.array(
        [
            max(float(data.max()), 1.0),
            max(4.0 * amplitude0, 2.0),
            min(float(z_size - 1), maximum_index[0] + 2.5),
            min(float(y_size - 1), maximum_index[1] + 3.0),
            min(float(x_size - 1), maximum_index[2] + 3.0),
            min(6.0, z_size / 2.0),
            min(12.0, y_size / 2.0),
            min(12.0, x_size / 2.0),
        ]
    )

    scale = max(robust_sigma(data), math.sqrt(max(float(data.max()), 1.0)))

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = gaussian3d(parameters, z_grid, y_grid, x_grid)
        return ((model - data) / scale).ravel()

    try:
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=500,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    if not result.success:
        return None

    fitted = gaussian3d(result.x, z_grid, y_grid, x_grid)
    residual_sum = float(np.sum((data - fitted) ** 2))
    total_sum = float(np.sum((data - data.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    names = ("background", "amplitude", "z0", "y0", "x0", "sigma_z_px", "sigma_y_px", "sigma_x_px")
    values = {name: float(value) for name, value in zip(names, result.x)}
    values["r_squared"] = r_squared
    values["cost"] = float(result.cost)
    values["sigma_xy_px"] = math.sqrt(values["sigma_y_px"] * values["sigma_x_px"])
    values["fwhm_xy_px"] = FWHM_PER_SIGMA * values["sigma_xy_px"]
    return values


def central_z_for_xy(volume: np.ndarray, y: int, x: int, half_width: int = 1) -> int:
    y0, y1 = max(0, y - half_width), min(volume.shape[1], y + half_width + 1)
    x0, x1 = max(0, x - half_width), min(volume.shape[2], x + half_width + 1)
    z_profile = volume[:, y0:y1, x0:x1].max(axis=(1, 2))
    return int(np.argmax(z_profile))


def find_candidates(
    volume: np.ndarray,
    max_candidates: int,
    xy_radius: int,
    z_radius: int,
) -> list[tuple[int, int, int, float, float]]:
    mip = volume.max(axis=0).astype(np.float32)
    detection_image = gaussian_filter(mip, sigma=1.0)
    background = float(np.median(detection_image))
    noise = robust_sigma(detection_image)
    percentile_threshold = float(np.percentile(detection_image, 99.8))
    threshold = max(background + 8.0 * noise, percentile_threshold)
    peaks = peak_local_max(
        detection_image,
        min_distance=max(2 * xy_radius, 24),
        threshold_abs=threshold,
        exclude_border=xy_radius + 1,
        num_peaks=max_candidates * 3,
    )

    candidates: list[tuple[int, int, int, float, float]] = []
    for y, x in peaks:
        z = central_z_for_xy(volume, int(y), int(x))
        if z < z_radius or z >= volume.shape[0] - z_radius:
            continue
        patch = volume[z - z_radius : z + z_radius + 1, y - xy_radius : y + xy_radius + 1, x - xy_radius : x + xy_radius + 1]
        border = np.concatenate(
            [
                patch[:, :2, :].ravel(),
                patch[:, -2:, :].ravel(),
                patch[:, :, :2].ravel(),
                patch[:, :, -2:].ravel(),
            ]
        )
        local_background = float(np.median(border))
        local_noise = robust_sigma(border)
        peak_value = float(patch.max())
        snr = (peak_value - local_background) / local_noise
        if snr < 8.0:
            continue
        candidates.append((z, int(y), int(x), peak_value, snr))

    candidates.sort(key=lambda item: item[3], reverse=True)
    return candidates[:max_candidates]


def fit_channel(
    raw_volume: np.ndarray,
    median_volume: np.ndarray,
    channel_index: int,
    channel_name: str,
    voxel_z_nm: float,
    voxel_y_nm: float,
    voxel_x_nm: float,
    max_candidates: int,
    xy_radius: int,
    z_radius: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidates = find_candidates(median_volume, max_candidates, xy_radius, z_radius)
    rows: list[dict[str, Any]] = []
    patch_records: list[dict[str, Any]] = []

    for candidate_index, (z, y, x, peak_value, snr) in enumerate(candidates):
        slices = (
            slice(z - z_radius, z + z_radius + 1),
            slice(y - xy_radius, y + xy_radius + 1),
            slice(x - xy_radius, x + xy_radius + 1),
        )
        raw_patch = raw_volume[slices]
        median_patch = median_volume[slices]
        if raw_patch.shape != (2 * z_radius + 1, 2 * xy_radius + 1, 2 * xy_radius + 1):
            continue
        if float(raw_patch.max()) >= float(np.iinfo(raw_volume.dtype).max):
            continue

        record: dict[str, Any] = {
            "candidate_index": candidate_index,
            "channel_index": channel_index,
            "channel_name": channel_name,
            "z": z,
            "y": y,
            "x": x,
            "peak_value_median": peak_value,
            "candidate_snr": snr,
            "raw_patch": raw_patch,
            "median_patch": median_patch,
        }
        any_accepted = False
        for preprocessing, patch in (("raw", raw_patch), ("median3x3", median_patch)):
            fit = fit_gaussian3d(patch)
            if fit is None:
                continue
            sigma_z_nm = fit["sigma_z_px"] * voxel_z_nm
            sigma_y_nm = fit["sigma_y_px"] * voxel_y_nm
            sigma_x_nm = fit["sigma_x_px"] * voxel_x_nm
            sigma_xy_nm = math.sqrt(sigma_y_nm * sigma_x_nm)
            point_sigma_xy_nm = math.sqrt(max(sigma_xy_nm**2 - BEAD_AXIS_SIGMA_NM**2, 0.0))
            fit_row = {
                "candidate_index": candidate_index,
                "channel_index": channel_index,
                "channel_name": channel_name,
                "preprocessing": preprocessing,
                "z": z,
                "y": y,
                "x": x,
                "peak_value_median": peak_value,
                "candidate_snr": snr,
                **fit,
                "sigma_z_nm": sigma_z_nm,
                "sigma_y_nm": sigma_y_nm,
                "sigma_x_nm": sigma_x_nm,
                "sigma_xy_nm": sigma_xy_nm,
                "fwhm_z_nm": FWHM_PER_SIGMA * sigma_z_nm,
                "fwhm_xy_nm": FWHM_PER_SIGMA * sigma_xy_nm,
                "point_psf_sigma_xy_nm_approx": point_sigma_xy_nm,
                "point_psf_fwhm_xy_nm_approx": FWHM_PER_SIGMA * point_sigma_xy_nm,
            }
            rows.append(fit_row)
            if preprocessing == "median3x3" and fit["r_squared"] >= MIN_FIT_R_SQUARED:
                any_accepted = True
        if any_accepted:
            patch_records.append(record)

    return pd.DataFrame(rows), patch_records


def accepted_fits(fits: pd.DataFrame, preprocessing: str = "median3x3") -> pd.DataFrame:
    selected = fits[
        (fits["preprocessing"] == preprocessing)
        & (fits["r_squared"] >= MIN_FIT_R_SQUARED)
        & (fits["sigma_xy_px"].between(0.7, 10.0))
        & (fits["sigma_z_px"].between(0.3, 5.0))
    ].copy()
    return selected


def make_recommendations(
    fits: pd.DataFrame,
    bead_metadata: OpticalMetadata,
    sample_metadata: OpticalMetadata,
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {
        "basis": {
            "primary_preprocessing": "3x3 XY median filter, identical to Gao processed_1",
            "bead_diameter_nm": 200.0,
            "bead_uniform_sphere_axis_sigma_nm": BEAD_AXIS_SIGMA_NM,
            "sample_voxel_nm": [
                sample_metadata.voxel_z_nm,
                sample_metadata.voxel_y_nm,
                sample_metadata.voxel_x_nm,
            ],
        },
        "channels": {},
    }

    for channel_index, channel_name in enumerate(bead_metadata.channel_names):
        selected = accepted_fits(fits[fits["channel_index"] == channel_index])
        if selected.empty:
            continue

        sigma_z_nm = float(selected["sigma_z_nm"].median())
        sigma_y_nm = float(selected["sigma_y_nm"].median())
        sigma_x_nm = float(selected["sigma_x_nm"].median())
        sigma_xy_nm = math.sqrt(sigma_y_nm * sigma_x_nm)
        sigma_sample_px = np.array(
            [
                sigma_z_nm / sample_metadata.voxel_z_nm,
                sigma_y_nm / sample_metadata.voxel_y_nm,
                sigma_x_nm / sample_metadata.voxel_x_nm,
            ]
        )
        point_sigma_xy_nm = math.sqrt(max(sigma_xy_nm**2 - BEAD_AXIS_SIGMA_NM**2, 0.0))
        point_sigma_sample_xy_px = point_sigma_xy_nm / math.sqrt(
            sample_metadata.voxel_y_nm * sample_metadata.voxel_x_nm
        )
        observed_sigma_sample_xy_px = math.sqrt(sigma_sample_px[1] * sigma_sample_px[2])

        r_mass_radius = max(1, int(round(1.585 * observed_sigma_sample_xy_px)))
        trackpy_diameter = odd_ceil(FWHM_PER_SIGMA * observed_sigma_sample_xy_px)
        trackpy_separation = trackpy_diameter + 1

        recommendations["channels"][str(channel_index)] = {
            "channel_name": channel_name,
            "accepted_beads": int(len(selected)),
            "observed_bead_sigma_nm_zyx": [sigma_z_nm, sigma_y_nm, sigma_x_nm],
            "observed_bead_fwhm_nm_zyx": [
                FWHM_PER_SIGMA * sigma_z_nm,
                FWHM_PER_SIGMA * sigma_y_nm,
                FWHM_PER_SIGMA * sigma_x_nm,
            ],
            "log_kernel_size_px_zyx_primary": sigma_sample_px.tolist(),
            "log_kernel_size_px_xy_point_psf_lower_bound": point_sigma_sample_xy_px,
            "minimum_distance_px_zyx_primary": sigma_sample_px.tolist(),
            "minimum_distance_integer_radius_zyx": np.ceil(sigma_sample_px).astype(int).tolist(),
            "r_mass_radius_px_primary": r_mass_radius,
            "r_mass_radius_px_sensitivity": sorted(
                {
                    max(1, r_mass_radius - 1),
                    r_mass_radius,
                    r_mass_radius + 1,
                    max(1, int(round(1.3 * observed_sigma_sample_xy_px))),
                    max(1, int(round(2.0 * observed_sigma_sample_xy_px))),
                }
            ),
            "trackpy_diameter_px_primary": trackpy_diameter,
            "trackpy_diameter_px_sensitivity": sorted(
                {
                    max(3, trackpy_diameter - 2),
                    trackpy_diameter,
                    trackpy_diameter + 2,
                }
            ),
            "trackpy_separation_px_primary": trackpy_separation,
            "trackpy_separation_px_sensitivity": sorted(
                {
                    max(1, trackpy_diameter - 1),
                    trackpy_diameter + 1,
                    trackpy_diameter + 3,
                }
            ),
        }
    return recommendations


def plot_detection_overlays(
    median_mips: dict[int, np.ndarray],
    fits: pd.DataFrame,
    channel_names: Iterable[str],
    output_path: Path,
) -> None:
    channel_names = list(channel_names)
    channel_indices = sorted(median_mips)
    fig, axes = plt.subplots(1, len(channel_indices), figsize=(6 * len(channel_indices), 6))
    axes = np.atleast_1d(axes)
    for axis, channel_index in zip(axes, channel_indices):
        channel_name = channel_names[channel_index]
        mip = median_mips[channel_index]
        lo, hi = np.percentile(mip, [1.0, 99.95])
        axis.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
        selected = accepted_fits(fits[fits["channel_index"] == channel_index])
        axis.scatter(selected["x"], selected["y"], s=35, facecolors="none", edgecolors="red", linewidths=0.8)
        axis.set_title(f"C{channel_index} {channel_name}\naccepted={len(selected)}")
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_width_distributions(fits: pd.DataFrame, output_path: Path) -> None:
    accepted = fits[
        (fits["r_squared"] >= MIN_FIT_R_SQUARED)
        & (fits["sigma_xy_px"].between(0.7, 10.0))
        & (fits["sigma_z_px"].between(0.3, 5.0))
    ].copy()
    channels = sorted(accepted["channel_index"].unique())
    fig, axes = plt.subplots(2, len(channels), figsize=(5 * len(channels), 8), squeeze=False)
    for column, channel_index in enumerate(channels):
        channel = accepted[accepted["channel_index"] == channel_index]
        labels = []
        lateral = []
        axial = []
        for preprocessing in ("raw", "median3x3"):
            subset = channel[channel["preprocessing"] == preprocessing]
            labels.append(preprocessing)
            lateral.append(subset["fwhm_xy_nm"].to_numpy())
            axial.append(subset["fwhm_z_nm"].to_numpy())
        axes[0, column].boxplot(lateral, tick_labels=labels, showfliers=False)
        axes[0, column].set_ylabel("Lateral FWHM (nm)")
        axes[0, column].set_title(f"C{channel_index}: lateral")
        axes[1, column].boxplot(axial, tick_labels=labels, showfliers=False)
        axes[1, column].set_ylabel("Axial FWHM (nm)")
        axes[1, column].set_title(f"C{channel_index}: axial")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32)
    background = float(np.percentile(patch, 10.0))
    shifted = np.clip(patch - background, 0.0, None)
    maximum = float(shifted.max())
    return shifted / maximum if maximum > 0 else shifted


def plot_example_beads(
    patch_records: list[dict[str, Any]],
    channel_names: Iterable[str],
    output_path: Path,
) -> None:
    channel_names = list(channel_names)
    channel_indices = sorted({int(record["channel_index"]) for record in patch_records})
    fig, axes = plt.subplots(len(channel_indices), 4, figsize=(14, 4 * len(channel_indices)), squeeze=False)
    for row_index, channel_index in enumerate(channel_indices):
        channel_name = channel_names[channel_index]
        records = [record for record in patch_records if record["channel_index"] == channel_index]
        if not records:
            continue
        record = max(records, key=lambda item: item["candidate_snr"])
        for column, preprocessing in enumerate(("raw_patch", "median_patch")):
            patch = normalize_patch(record[preprocessing])
            z_center, y_center, x_center = np.unravel_index(int(np.argmax(patch)), patch.shape)
            axes[row_index, 2 * column].imshow(patch.max(axis=0), cmap="magma", vmin=0, vmax=1)
            axes[row_index, 2 * column].set_title(f"C{channel_index} {channel_name}\n{preprocessing} XY MIP")
            axes[row_index, 2 * column + 1].imshow(
                patch[:, y_center, :],
                cmap="magma",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[row_index, 2 * column + 1].set_title(f"{preprocessing} XZ")
        for axis in axes[row_index]:
            axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    bead_metadata: OpticalMetadata,
    sample_metadata: OpticalMetadata,
    fits: pd.DataFrame,
    recommendations: dict[str, Any],
) -> None:
    modality_label = "SoRa" if bead_metadata.sora else "conventional CSU-W1"
    lines = [
        f"# {modality_label} 200 nm bead解析",
        "",
        "## 撮影条件の照合",
        "",
        f"- 細胞: `{sample_metadata.path}`",
        f"- bead: `{bead_metadata.path}`",
        f"- 対物: {sample_metadata.objective_name}; {sample_metadata.objective_magnification:.0f}×, NA {sample_metadata.objective_na:.2f}",
        f"- modality: {modality_label}",
        f"- zoom: 細胞 {sample_metadata.zoom_magnification:g}× / bead {bead_metadata.zoom_magnification:g}×",
        f"- XY sampling: 細胞 {sample_metadata.voxel_x_nm:.3f} nm/px / bead {bead_metadata.voxel_x_nm:.3f} nm/px",
        f"- Z step: 細胞 {sample_metadata.voxel_z_nm:.1f} nm / bead {bead_metadata.voxel_z_nm:.1f} nm",
        "",
        "## 推定方法",
        "",
        "- Gao `processed_1`と同じ3×3 XY median-filtered stackを主解析に使用した。",
        f"- 200 nm beadを3D異方性Gaussianへ個別fitし、R²≥{MIN_FIT_R_SQUARED:.2f}のfitを採用した。",
        "- raw fitも併記し、median filterによる見かけ幅の変化を診断した。",
        "- beadは点光源ではないため、観測幅をLoGの主初期値、均一球近似でdeconvolutionした値を小さい側の感度条件とした。",
        "",
        "## 3×3 median filterの影響",
        "",
        "| channel | paired beads | XY FWHM median/raw | amplitude median/raw | ΔR² (median−raw) |",
        "|---|---:|---:|---:|---:|",
    ]
    for channel_index, channel_name in enumerate(bead_metadata.channel_names):
        channel_fits = fits[fits["channel_index"] == channel_index]
        raw = channel_fits[channel_fits["preprocessing"] == "raw"][
            ["candidate_index", "fwhm_xy_nm", "amplitude", "r_squared"]
        ].rename(
            columns={
                "fwhm_xy_nm": "fwhm_xy_nm_raw",
                "amplitude": "amplitude_raw",
                "r_squared": "r_squared_raw",
            }
        )
        filtered = channel_fits[channel_fits["preprocessing"] == "median3x3"][
            ["candidate_index", "fwhm_xy_nm", "amplitude", "r_squared", "sigma_xy_px", "sigma_z_px"]
        ].rename(
            columns={
                "fwhm_xy_nm": "fwhm_xy_nm_median",
                "amplitude": "amplitude_median",
                "r_squared": "r_squared_median",
            }
        )
        filtered = filtered[
            (filtered["r_squared_median"] >= MIN_FIT_R_SQUARED)
            & (filtered["sigma_xy_px"].between(0.7, 10.0))
            & (filtered["sigma_z_px"].between(0.3, 5.0))
        ]
        paired = filtered.merge(raw, on="candidate_index", how="inner")
        if paired.empty:
            continue
        lines.append(
            f"| C{channel_index} {channel_name} | {len(paired)} | "
            f"{(paired['fwhm_xy_nm_median'] / paired['fwhm_xy_nm_raw']).median():.4f} | "
            f"{(paired['amplitude_median'] / paired['amplitude_raw']).median():.4f} | "
            f"{(paired['r_squared_median'] - paired['r_squared_raw']).median():.4f} |"
        )
    lines.extend(
        [
            "",
            "median filterは線形なphoton-count保存処理ではないが、本データではXY幅への影響は約1%以下で、",
            "特に445 nmチャネルのfit安定性を大きく改善した。Gao互換主解析はmedian後画像へ統一し、",
            "raw画像は感度QCにのみ使用する。",
            "",
        "## チャネル別推奨初期値",
        "",
        ]
    )
    for channel_key, channel in recommendations["channels"].items():
        lines.extend(
            [
                f"### C{channel_key}: {channel['channel_name']}",
                "",
                f"- 採用bead数: {channel['accepted_beads']}",
                f"- 観測bead σ (Z,Y,X; nm): `{[round(value, 2) for value in channel['observed_bead_sigma_nm_zyx']]}`",
                f"- 観測bead FWHM (Z,Y,X; nm): `{[round(value, 2) for value in channel['observed_bead_fwhm_nm_zyx']]}`",
                f"- `log_kernel_size` (Z,Y,X; sample px): `{[round(value, 3) for value in channel['log_kernel_size_px_zyx_primary']]}`",
                f"- `minimum_distance` (Z,Y,X; sample px): `{[round(value, 3) for value in channel['minimum_distance_px_zyx_primary']]}`",
                f"- `r_mass` radius: `{channel['r_mass_radius_px_primary']} px`; sensitivity `{channel['r_mass_radius_px_sensitivity']}`",
                f"- trackpy `diameter`: `{channel['trackpy_diameter_px_primary']} px`; sensitivity `{channel['trackpy_diameter_px_sensitivity']}`",
                f"- trackpy `separation`: `{channel['trackpy_separation_px_primary']} px`; sensitivity `{channel['trackpy_separation_px_sensitivity']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 運用時に丸める主初期値",
            "",
            "- mTetR big-FISH: `log_kernel_size=(0.5, 4.0, 4.0) px`、`minimum_distance=(1.0, 4.0, 4.0) px`。",
            "- mTetR `r_mass` radius: `6 px`（感度条件5/6/7/8 px）。",
            "- MCP `r_mass` radius: `6 px`（感度条件5/6/7/8 px）。",
            "- SNAPtag/MCP trackpy: `diameter=11 px`、`separation=12 px`（感度条件diameter 9/11/13、separation 10/12/14）。",
            "",
            "## 注意",
            "",
            "- bead PSFだけでは生物学的mTetR/MCP locusの実幅は決まらない。FOV1–10の実輝点で候補値を再検証する。",
            "- `minimum_distance`はPSF幅から得た重複ピーク抑制の初期値であり、実際の近接spot間隔の下限ではない。",
            "- `r_mass`閾値と積分半径は結合しているため、最終半径で`r_mTetR≥1.015`と`r_MCP≥1.025`の保持率を再確認する。",
            "",
            "## 出力",
            "",
            "- `bead_gaussian_fits.csv`: bead単位のraw/median 3D Gaussian fit",
            "- `bead_detection_overlays.png`: 採用bead位置",
            "- `bead_width_raw_vs_median.png`: rawとmedian後のFWHM比較",
            "- `bead_examples_raw_vs_median.png`: 代表bead XY/XZ",
            "- `parameter_recommendations.json`: 機械可読な推奨値",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bead", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--xy-radius", type=int, default=15)
    parser.add_argument("--z-radius", type=int, default=4)
    parser.add_argument(
        "--channel-index",
        type=int,
        default=None,
        help="Fit one channel only. Use separate processes when RAM is limited.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bead_metadata = read_optical_metadata(args.bead)
    sample_metadata = read_optical_metadata(args.sample)

    if bead_metadata.objective_name != sample_metadata.objective_name:
        raise RuntimeError("Bead and sample objective names do not match")
    if bead_metadata.sora != sample_metadata.sora:
        raise RuntimeError("Bead and sample SoRa modality do not match")
    if not math.isclose(bead_metadata.zoom_magnification, sample_metadata.zoom_magnification):
        raise RuntimeError("Bead and sample zoom magnification do not match")
    if not math.isclose(bead_metadata.voxel_x_nm, sample_metadata.voxel_x_nm, rel_tol=1e-6):
        raise RuntimeError("Bead and sample XY sampling do not match")

    fit_frames: list[pd.DataFrame] = []
    all_patch_records: list[dict[str, Any]] = []
    median_mips: dict[int, np.ndarray] = {}
    if args.channel_index is None:
        channel_indices = list(range(len(bead_metadata.channel_names)))
    else:
        if not 0 <= args.channel_index < len(bead_metadata.channel_names):
            raise ValueError(f"Invalid channel index: {args.channel_index}")
        channel_indices = [args.channel_index]

    for channel_index in channel_indices:
        channel_name = bead_metadata.channel_names[channel_index]
        raw_volume = load_channel_zyx(args.bead, channel_index)
        median_volume = median_filter(raw_volume, size=(1, 3, 3))
        median_mips[channel_index] = median_volume.max(axis=0)
        fits, patch_records = fit_channel(
            raw_volume=raw_volume,
            median_volume=median_volume,
            channel_index=channel_index,
            channel_name=channel_name,
            voxel_z_nm=bead_metadata.voxel_z_nm,
            voxel_y_nm=bead_metadata.voxel_y_nm,
            voxel_x_nm=bead_metadata.voxel_x_nm,
            max_candidates=args.max_candidates,
            xy_radius=args.xy_radius,
            z_radius=args.z_radius,
        )
        fit_frames.append(fits)
        all_patch_records.extend(patch_records)
        print(
            f"C{channel_index} {channel_name}: "
            f"fits={len(fits)}, accepted_median={len(accepted_fits(fits))}",
            flush=True,
        )
        del raw_volume, median_volume
        gc.collect()

    all_fits = pd.concat(fit_frames, ignore_index=True)
    all_fits.to_csv(args.output / "bead_gaussian_fits.csv", index=False)
    recommendations = make_recommendations(all_fits, bead_metadata, sample_metadata)
    (args.output / "parameter_recommendations.json").write_text(
        json.dumps(recommendations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "optical_metadata.json").write_text(
        json.dumps(
            {
                "bead": asdict(bead_metadata),
                "sample": asdict(sample_metadata),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_detection_overlays(
        median_mips,
        all_fits,
        bead_metadata.channel_names,
        args.output / "bead_detection_overlays.png",
    )
    plot_width_distributions(all_fits, args.output / "bead_width_raw_vs_median.png")
    plot_example_beads(
        all_patch_records,
        bead_metadata.channel_names,
        args.output / "bead_examples_raw_vs_median.png",
    )
    write_markdown_report(
        args.output / "bead_parameter_report.md",
        bead_metadata,
        sample_metadata,
        all_fits,
        recommendations,
    )
    print(json.dumps(recommendations, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
