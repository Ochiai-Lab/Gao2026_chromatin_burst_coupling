#!/usr/bin/env python3
"""Ordered multi-dataset wrapper for the fixed-cell SoRa Gao 3D workflow."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
import torch

import sora_gao3d_snapshot_pipeline as gao
import sora_snapshot_pipeline as segmentation


PROJECT = Path(__file__).resolve().parent
DATASET_MANIFEST = PROJECT / "sora_fixed_dataset_manifest_20260719.json"
AGGREGATE_OUTPUT_NAME = "sora_snapshot_outputs_all_datasets_gao3d_final"
REFERENCE_SEGMENTATION_NAME = (
    "sora_snapshot_outputs_chamberA_field1_final_"
    "aa4x_sigma11_d110_cp05_dilate15_gao"
)
REFERENCE_ANALYSIS_NAME = "sora_snapshot_outputs_chamberA_field1_gao3d_final"
SEGMENTATION_SUFFIX = "aa4x_sigma11_d110_cp05_dilate15"


@dataclass(frozen=True)
class DatasetBundle:
    record: dict[str, Any]
    segmentation_config: segmentation.AnalysisConfig
    analysis_config: gao.AnalysisConfig

    @property
    def analysis_id(self) -> str:
        return str(self.record["analysis_id"])

    @property
    def is_completed_reference(self) -> bool:
        return self.record["analysis_status"] == "completed_reference"


def _load_manifest() -> dict[str, Any]:
    payload = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("dataset_count") != 8:
        raise ValueError("Expected exactly eight SoRa datasets.")
    return payload


def _analysis_label(record: dict[str, Any]) -> str:
    return (
        f"{record['streaming_tag_locus']} × {record['mintbody']} — "
        f"{record['replicate']} ({record['chamber']} Field{record['field']})"
    )


def build_dataset_bundles() -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    for index, record in enumerate(_load_manifest()["records"]):
        analysis_id = str(record["analysis_id"])
        is_reference = record["analysis_status"] == "completed_reference"
        segmentation_name = (
            REFERENCE_SEGMENTATION_NAME
            if is_reference
            else f"sora_segmentation_{analysis_id}_{SEGMENTATION_SUFFIX}"
        )
        analysis_name = (
            REFERENCE_ANALYSIS_NAME
            if is_reference
            else str(record["planned_output_name"])
        )
        common = {
            "input_nd2": Path(record["path"]),
            "project_dir": PROJECT,
            "analysis_id": analysis_id,
            "analysis_label": _analysis_label(record),
            "streaming_tag_locus": str(record["streaming_tag_locus"]),
            "mintbody_label": str(record["mintbody"]),
            "replicate": str(record["replicate"]),
            "chamber": str(record["chamber"]),
            "expected_fov_count": int(record["fov_count"]),
            "random_seed": 20260719 + index * 100_000,
        }
        segmentation_config = segmentation.AnalysisConfig(
            **common,
            output_name=segmentation_name,
        )
        analysis_config = gao.AnalysisConfig(
            **common,
            output_name=analysis_name,
            segmentation_source_name=segmentation_name,
            require_reference_equivalence=is_reference,
        )
        bundles.append(
            DatasetBundle(
                record=record,
                segmentation_config=segmentation_config,
                analysis_config=analysis_config,
            )
        )
    return bundles


def aggregate_config() -> gao.AnalysisConfig:
    reference = build_dataset_bundles()[0].analysis_config
    return gao.AnalysisConfig(
        input_nd2=reference.input_nd2,
        project_dir=PROJECT,
        output_name=AGGREGATE_OUTPUT_NAME,
        segmentation_source_name=REFERENCE_SEGMENTATION_NAME,
        analysis_id="all_datasets_pooled",
        analysis_label="All 8 datasets pooled",
        streaming_tag_locus="Nanog/Sox2",
        mintbody_label="Ser5ph/H3K27ac",
        replicate="rep1/rep2",
        chamber="ChamberA/ChamberB",
        expected_fov_count=None,
        require_reference_equivalence=False,
    )


def bundle_table(bundles: Iterable[DatasetBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        record = bundle.record
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "locus": record["streaming_tag_locus"],
                "mintbody": record["mintbody"],
                "replicate": record["replicate"],
                "chamber": record["chamber"],
                "field": int(record["field"]),
                "fov_count": int(record["fov_count"]),
                "analysis_status": record["analysis_status"],
                "input_nd2": str(bundle.analysis_config.input_nd2),
                "segmentation_root": str(
                    bundle.segmentation_config.output_root
                ),
                "analysis_root": str(bundle.analysis_config.output_root),
            }
        )
    return pd.DataFrame(rows)


def preflight_datasets(
    bundles: Iterable[DatasetBundle],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = gao.collect_metadata(bundle.analysis_config)
        channel_map = metadata["channel_mapping"]
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
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
        )
    output = aggregate_config().output_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "dataset_preflight.csv", index=False)
    return frame


def first_pending_bundle(bundles: Iterable[DatasetBundle]) -> DatasetBundle:
    return next(bundle for bundle in bundles if not bundle.is_completed_reference)


def run_segmentation_smoke_test(
    bundle: DatasetBundle,
    fov_id: int = 0,
) -> dict[str, Any]:
    regions = segmentation.run_segmentation(
        bundle.segmentation_config,
        fov_ids=[int(fov_id)],
    )
    overlay = (
        bundle.segmentation_config.output_root
        / "segmentation_overlays"
        / f"fov_{fov_id:03d}_selected_regions.png"
    )
    return {
        "analysis_id": bundle.analysis_id,
        "fov_id": int(fov_id),
        "selected_regions": int(regions["selected"].astype(bool).sum()),
        "overlay": str(overlay),
    }


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all_segmentations(
    bundles: Iterable[DatasetBundle],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        print(f"=== Segmentation: {bundle.analysis_id} ===", flush=True)
        if bundle.is_completed_reference:
            regions = gao.run_segmentation(bundle.analysis_config)
            method = "reuse_completed_reference"
        else:
            segmentation.run_segmentation(bundle.segmentation_config)
            regions = gao.run_segmentation(bundle.analysis_config)
            method = "cellpose_then_read_only_reuse"
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "fov_count": int(bundle.record["fov_count"]),
                "selected_regions": int(
                    regions["selected"].astype(bool).sum()
                ),
                "method": method,
                "segmentation_root": str(
                    bundle.segmentation_config.output_root
                ),
            }
        )
        _release_accelerator_memory()
    frame = pd.DataFrame(rows)
    output = aggregate_config().output_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "segmentation_batch_summary.csv", index=False)
    return frame


def _manifest_complete(config: gao.AnalysisConfig, fov_id: int) -> bool:
    path = (
        config.output_root
        / "per_fov_cache"
        / f"fov_{fov_id:03d}_cache_manifest.json"
    )
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        payload.get("complete")
        and payload.get("algorithm_version") == gao.ALGORITHM_VERSION
    )


def _fixed_chunks(values: list[int], size: int) -> list[list[int]]:
    size = max(1, int(size))
    return [
        values[start : start + size]
        for start in range(0, len(values), size)
    ]


def _process_quantification_chunk(
    payload: tuple[gao.AnalysisConfig, list[int]],
) -> dict[str, Any]:
    config, fov_ids = payload
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    completed = 0
    with nd2.ND2File(config.input_nd2) as handle:
        dask_array = handle.to_dask()
        for ordinal, fov_id in enumerate(fov_ids, start=1):
            gao._process_one_fov(fov_id, dask_array, config)
            completed += 1
            if ordinal % 5 == 0 or ordinal == len(fov_ids):
                print(
                    f"{config.analysis_id} worker={os.getpid()} "
                    f"{ordinal}/{len(fov_ids)} FOV; last={fov_id:03d}",
                    flush=True,
                )
    return {
        "analysis_id": config.analysis_id,
        "pid": int(os.getpid()),
        "assigned": int(len(fov_ids)),
        "completed": int(completed),
    }


def run_quantification_parallel(
    config: gao.AnalysisConfig,
    *,
    workers: int = 4,
    chunk_size: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gao.prepare_directories(config)
    metadata = gao.collect_metadata(config)
    n_positions = int(metadata["sizes"]["P"])
    pending = [
        fov_id
        for fov_id in range(n_positions)
        if not _manifest_complete(config, fov_id)
    ]
    chunks = _fixed_chunks(pending, chunk_size)
    results: list[dict[str, Any]] = []
    if chunks:
        worker_count = max(1, min(int(workers), len(chunks)))
        print(
            {
                "analysis_id": config.analysis_id,
                "n_positions": n_positions,
                "already_complete": n_positions - len(pending),
                "pending": len(pending),
                "workers": worker_count,
                "chunk_size": int(chunk_size),
            },
            flush=True,
        )
        context = mp.get_context("spawn")
        with context.Pool(
            processes=worker_count,
            maxtasksperchild=1,
        ) as pool:
            tasks = [(config, chunk) for chunk in chunks]
            for result in pool.imap_unordered(
                _process_quantification_chunk,
                tasks,
                chunksize=1,
            ):
                results.append(result)
                print({"worker_complete": result}, flush=True)

    remaining = [
        fov_id
        for fov_id in range(n_positions)
        if not _manifest_complete(config, fov_id)
    ]
    if remaining:
        raise RuntimeError(
            f"{config.analysis_id}: incomplete cache FOVs {remaining[:20]}"
        )
    candidates, exclusions = gao.run_quantification(config)
    summary = {
        "analysis_id": config.analysis_id,
        "fov_count": n_positions,
        "completed_cache_manifests": n_positions,
        "saved_candidates": int(len(candidates)),
        "rank1_candidates": int(
            candidates["is_primary_candidate"].astype(bool).sum()
        ),
        "exclusion_records": int(len(exclusions)),
        "workers": results,
    }
    gao.write_json(
        config.output_root / "parallel_cache_run_summary.json",
        summary,
    )
    return candidates, exclusions, summary


def run_all_quantifications(
    bundles: Iterable[DatasetBundle],
    *,
    workers: int = 4,
    chunk_size: int = 15,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        print(f"=== Gao 3D: {bundle.analysis_id} ===", flush=True)
        _, _, summary = run_quantification_parallel(
            bundle.analysis_config,
            workers=workers,
            chunk_size=chunk_size,
        )
        rows.append(
            {
                key: value
                for key, value in summary.items()
                if key != "workers"
            }
        )
    frame = pd.DataFrame(rows)
    output = aggregate_config().output_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "quantification_batch_summary.csv", index=False)
    return frame


def _contact_sheet(
    image_paths: list[Path],
    titles: list[str],
    output: Path,
    *,
    columns: int = 2,
    panel_width: float = 8.5,
    panel_height: float = 6.7,
) -> Path:
    rows = int(np.ceil(len(image_paths) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_width * columns, panel_height * rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    for axis, path, title in zip(axes_array, image_paths, titles):
        axis.imshow(plt.imread(path))
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    for axis in axes_array[len(image_paths) :]:
        axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def run_threshold_calibrations(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    diagnostics: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []
    individual_figures: list[Path] = []
    individual_titles: list[str] = []

    for bundle in bundles:
        config = bundle.analysis_config
        candidates = pd.read_csv(
            config.output_root / "tables" / "mtetr_mcp_spot_candidates.csv"
        )
        tagged = candidates.copy()
        tagged.insert(0, "analysis_id", bundle.analysis_id)
        tagged.insert(1, "streaming_tag_locus", record_value(bundle, "streaming_tag_locus"))
        tagged.insert(2, "mintbody", record_value(bundle, "mintbody"))
        tagged.insert(3, "replicate", record_value(bundle, "replicate"))
        combined_frames.append(tagged)

        decision, artifacts, sensitivity = gao.calibrate_thresholds(
            candidates,
            config,
        )
        paths = gao.make_calibration_figures(
            candidates,
            decision,
            artifacts,
            sensitivity,
            config,
        )
        individual_figures.append(Path(paths["calibration_png"]))
        individual_titles.append(bundle.analysis_id)
        diagnostics.append(
            {
                "analysis_id": bundle.analysis_id,
                "rank1_candidate_count": decision["rank1_candidate_count"],
                "calibration_count_crop_mean_at_least_90": (
                    decision[
                        "rank1_calibration_count_crop_mean_at_least_90"
                    ]
                ),
                "mtetr_kde_diagnostic": decision[
                    "mtetr_kde_crosspoint_raw_diagnostic"
                ],
                "rank1_secondary_kde_diagnostic": decision[
                    "rank1_vs_secondary_kde_crosspoint_raw_diagnostic"
                ],
                "mcp_kde_diagnostic": decision[
                    "mcp_kde_crosspoint_raw_diagnostic"
                ],
                "selected_r_mtetr": decision["selected_mtetr_r_mass"],
                "selected_r_mcp": decision["selected_mcp_r_mass"],
            }
        )

    pooled_config = aggregate_config()
    gao.prepare_directories(pooled_config)
    pooled_candidates = pd.concat(combined_frames, ignore_index=True)
    pooled_candidates.to_csv(
        pooled_config.output_root
        / "tables"
        / "all_datasets_spot_candidates.csv",
        index=False,
    )
    pooled_decision, pooled_artifacts, pooled_sensitivity = (
        gao.calibrate_thresholds(pooled_candidates, pooled_config)
    )
    pooled_paths = gao.make_calibration_figures(
        pooled_candidates,
        pooled_decision,
        pooled_artifacts,
        pooled_sensitivity,
        pooled_config,
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame.to_csv(
        pooled_config.output_root
        / "tables"
        / "threshold_diagnostics_by_dataset.csv",
        index=False,
    )
    contact_sheet = _contact_sheet(
        individual_figures,
        individual_titles,
        pooled_config.output_root
        / "figures"
        / "threshold_calibration_individual_contact_sheet.png",
    )
    return {
        "diagnostics": diagnostics_frame,
        "pooled_decision": pooled_decision,
        "pooled_figure": Path(pooled_paths["calibration_png"]),
        "individual_contact_sheet": contact_sheet,
        "individual_figures": individual_figures,
        "pooled_candidate_count": int(len(pooled_candidates)),
    }


def record_value(bundle: DatasetBundle, key: str) -> str:
    return str(bundle.record[key])


def _plot_state_counts(frame: pd.DataFrame, output: Path) -> Path:
    state_order = [
        "Active",
        "Inactive",
        "Excluded_below_mTetR",
        "Excluded_no_valid_crop",
    ]
    pivot = (
        frame.pivot(index="analysis_id", columns="state", values="count")
        .fillna(0)
        .reindex(columns=state_order, fill_value=0)
    )
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=["tab:orange", "tab:blue", "0.65", "0.25"],
    )
    axis.set_ylabel("Nuclear regions")
    axis.set_xlabel("")
    axis.set_title(
        "State calls by dataset\n"
        "r_mTetR≥1.045, r_MCP≥1.1, distance≤0.39 µm"
    )
    axis.legend(frameon=False, ncol=2)
    axis.tick_params(axis="x", rotation=35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def run_states_and_locus_summaries(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    long_state_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gao_figures: list[Path] = []
    gao_titles: list[str] = []

    for bundle in bundles:
        config = bundle.analysis_config
        candidates = pd.read_csv(
            config.output_root / "tables" / "mtetr_mcp_spot_candidates.csv"
        )
        decision = json.loads(
            (config.output_root / "threshold_decision.json").read_text(
                encoding="utf-8"
            )
        )
        states = gao.call_states(candidates, decision, config)
        gao.make_rank1_crop_qc(states, decision, config)
        counts = states["state"].value_counts().to_dict()
        for state, count in counts.items():
            long_state_rows.append(
                {
                    "analysis_id": bundle.analysis_id,
                    "state": state,
                    "count": int(count),
                }
            )
        active = int(counts.get("Active", 0))
        inactive = int(counts.get("Inactive", 0))
        gao_status = "not_applicable_missing_active_or_inactive"
        if active > 0 and inactive > 0:
            figure, _ = gao.make_locus_summary(states, config)
            gao_figures.append(Path(figure))
            gao_titles.append(bundle.analysis_id)
            gao_status = "completed"

        metadata = gao.collect_metadata(config)
        regions = pd.read_csv(
            config.output_root
            / "tables"
            / "segmentation_regions_all_fovs.csv"
        )
        report = gao.validate_outputs(
            config,
            metadata,
            regions,
            candidates,
            states,
            decision,
        )
        eligible = active + inactive
        summary_rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "rank1_candidates": int(len(states)),
                "valid_crops": int(
                    np.isfinite(states["mTetR_r_mass"]).sum()
                ),
                "eligible_mtetr": eligible,
                "active": active,
                "inactive": inactive,
                "excluded_below_mtetr": int(
                    counts.get("Excluded_below_mTetR", 0)
                ),
                "excluded_no_valid_crop": int(
                    counts.get("Excluded_no_valid_crop", 0)
                ),
                "active_fraction_of_eligible": (
                    float(active / eligible) if eligible else np.nan
                ),
                "gao_summary_status": gao_status,
                "all_validations_passed": report[
                    "all_validations_passed"
                ],
            }
        )

    pooled_config = aggregate_config()
    tables = pooled_config.output_root / "tables"
    figures = pooled_config.output_root / "figures"
    state_long = pd.DataFrame(long_state_rows)
    state_summary = pd.DataFrame(summary_rows)
    state_long.to_csv(tables / "state_counts_by_dataset_long.csv", index=False)
    state_summary.to_csv(
        tables / "state_summary_by_dataset.csv",
        index=False,
    )
    state_figure = _plot_state_counts(
        state_long,
        figures / "state_counts_by_dataset.png",
    )
    gao_contact_sheet = None
    if gao_figures:
        gao_contact_sheet = _contact_sheet(
            gao_figures,
            gao_titles,
            figures / "gao_fig1c_individual_contact_sheet.png",
            panel_width=8.0,
            panel_height=6.2,
        )
    return {
        "state_summary": state_summary,
        "state_counts_figure": state_figure,
        "gao_contact_sheet": gao_contact_sheet,
    }


def finalize_batch(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    pooled_config = aggregate_config()
    analysis_manifests: dict[str, Any] = {}
    for bundle in bundles:
        path = bundle.analysis_config.output_root / "analysis_manifest.json"
        analysis_manifests[bundle.analysis_id] = json.loads(
            path.read_text(encoding="utf-8")
        )
    state_summary = pd.read_csv(
        pooled_config.output_root
        / "tables"
        / "state_summary_by_dataset.csv"
    )
    diagnostics = pd.read_csv(
        pooled_config.output_root
        / "tables"
        / "threshold_diagnostics_by_dataset.csv"
    )
    pooled_decision = json.loads(
        (pooled_config.output_root / "threshold_decision.json").read_text(
            encoding="utf-8"
        )
    )
    validations = {
        "eight_dataset_manifests": len(analysis_manifests) == 8,
        "all_dataset_manifests_passed": all(
            item["all_validations_passed"]
            for item in analysis_manifests.values()
        ),
        "eight_state_summaries": len(state_summary) == 8,
        "eight_threshold_diagnostics": len(diagnostics) == 8,
        "pooled_thresholds_fixed": bool(
            pooled_decision["selected_mtetr_r_mass"] == 1.045
            and pooled_decision["selected_mcp_r_mass"] == 1.1
        ),
        "individual_threshold_contact_sheet": (
            pooled_config.output_root
            / "figures"
            / "threshold_calibration_individual_contact_sheet.png"
        ).is_file(),
        "pooled_threshold_figure": (
            pooled_config.output_root
            / "figures"
            / "threshold_calibration_four_panel.png"
        ).is_file(),
        "state_counts_figure": (
            pooled_config.output_root
            / "figures"
            / "state_counts_by_dataset.png"
        ).is_file(),
    }
    report = {
        "analysis": "Eight fixed-cell SoRa datasets, Gao-compatible 3D batch",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "dataset_count": len(bundles),
        "total_fov_count": int(
            sum(bundle.record["fov_count"] for bundle in bundles)
        ),
        "selected_thresholds": {
            "r_mTetR": 1.045,
            "r_MCP": 1.1,
            "distance_um": 0.39,
        },
        "state_summary": state_summary.to_dict(orient="records"),
        "threshold_diagnostics": diagnostics.to_dict(orient="records"),
        "validations": validations,
        "all_validations_passed": bool(all(validations.values())),
    }
    gao.write_json(
        pooled_config.output_root / "batch_analysis_manifest.json",
        report,
    )
    if not report["all_validations_passed"]:
        failed = [key for key, value in validations.items() if not value]
        raise RuntimeError(f"Batch validation failed: {failed}")
    return report
