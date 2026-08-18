#!/usr/bin/env python3
"""Ordered eight-dataset Normal Gao 3D snapshot analysis.

The four biological conditions are each represented by ChamberA/rep1 and
ChamberB/rep2.  Segmentation and 3D quantification are run per dataset so that
FOV-level cache and QC remain traceable.  Threshold figures are generated at
three levels: eight datasets, four conditions pooled across replicates, and
all Normal datasets pooled.
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

import gc
import json
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
import numpy as np
import pandas as pd
import torch

import normal_field8_gao_pipeline as normal_seg
import normal_field8_sora_aligned_pipeline as normal_aligned
import sora_gao3d_snapshot_pipeline as gao
import sora_multidataset_pipeline as sora_batch


CODE_DIR = Path(__file__).resolve().parent
PROJECT = Path(
    os.environ.get(
        "SNAPSHOT_ANALYSIS_PROJECT_DIR",
        str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO'),
    )
).resolve()
DATASET_MANIFEST = CODE_DIR / "normal_fixed_dataset_manifest_20260720.json"
REVISION_TAG = "revision_20260809"
AGGREGATE_OUTPUT_NAME = (
    "normal_snapshot_outputs_all_datasets_gao3d_revision_20260809"
)
REFERENCE_ANALYSIS_ID = "nanog_ser5ph_rep1_normal"
REFERENCE_SEGMENTATION_NAME = (
    "normal_snapshot_outputs_chamberA_field8_gao_stage1"
)
REFERENCE_ANALYSIS_NAME = (
    "normal_snapshot_outputs_chamberA_field8_sora_aligned_gao3d"
)
ALGORITHM_VERSION = "normal_multidataset_all_qc_random_controls_v2"


@dataclass(frozen=True)
class NormalDatasetBundle:
    record: dict[str, Any]
    segmentation_config: normal_seg.AnalysisConfig
    analysis_config: normal_aligned.AnalysisConfig

    @property
    def analysis_id(self) -> str:
        return str(self.record["analysis_id"])

    @property
    def condition_id(self) -> str:
        return (
            f"{str(self.record['streaming_tag_locus']).lower()}_"
            f"{str(self.record['mintbody']).lower()}"
        )

    @property
    def is_reference(self) -> bool:
        return self.analysis_id == REFERENCE_ANALYSIS_ID


def _load_records() -> list[dict[str, Any]]:
    records = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if len(records) != 8:
        raise ValueError("Expected exactly eight Normal datasets.")
    if not all(record["all_metadata_checks_passed"] for record in records):
        raise ValueError("At least one Normal dataset failed metadata checks.")
    return records


def _resolve_manifest_input_path(path_text: str) -> Path:
    """Rebase recorded /Volumes paths under GAO2026_RAW_ROOT."""
    recorded = Path(path_text)
    if not recorded.is_absolute():
        return GAO2026_RAW_ROOT / recorded
    parts = recorded.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return GAO2026_RAW_ROOT.joinpath(*parts[2:])
    return recorded


def _analysis_label(record: dict[str, Any]) -> str:
    return (
        f"{record['streaming_tag_locus']} × {record['mintbody']} — "
        f"{record['replicate']} "
        f"({record['chamber']} Field{record['field']}, Normal)"
    )


def build_dataset_bundles() -> list[NormalDatasetBundle]:
    bundles: list[NormalDatasetBundle] = []
    for index, record in enumerate(_load_records()):
        analysis_id = str(record["analysis_id"])
        is_reference = analysis_id == REFERENCE_ANALYSIS_ID
        segmentation_name = (
            REFERENCE_SEGMENTATION_NAME
            if is_reference
            else f"normal_segmentation_{analysis_id}_gao"
        )
        analysis_name = (
            f"normal_snapshot_outputs_{analysis_id}_gao3d_{REVISION_TAG}"
        )
        label = _analysis_label(record)
        segmentation_config = normal_seg.AnalysisConfig(
            input_nd2=_resolve_manifest_input_path(record["path"]),
            project_dir=PROJECT,
            output_name=segmentation_name,
            analysis_id=analysis_id,
            analysis_label=label,
            expected_fov_count=int(record["fov_count"]),
        )
        analysis_config = normal_aligned.AnalysisConfig(
            input_nd2=_resolve_manifest_input_path(record["path"]),
            project_dir=PROJECT,
            output_name=analysis_name,
            segmentation_source_name=segmentation_name,
            analysis_id=analysis_id,
            analysis_label=label,
            streaming_tag_locus=str(record["streaming_tag_locus"]),
            mintbody_label=str(record["mintbody"]),
            replicate=str(record["replicate"]),
            chamber=str(record["chamber"]),
            expected_fov_count=int(record["fov_count"]),
            random_seed=20260720 + index * 100_000,
        )
        bundles.append(
            NormalDatasetBundle(
                record=record,
                segmentation_config=segmentation_config,
                analysis_config=analysis_config,
            )
        )
    return bundles


def aggregate_config() -> normal_aligned.AnalysisConfig:
    reference = next(
        bundle for bundle in build_dataset_bundles() if bundle.is_reference
    )
    return normal_aligned.AnalysisConfig(
        input_nd2=reference.analysis_config.input_nd2,
        project_dir=PROJECT,
        output_name=AGGREGATE_OUTPUT_NAME,
        segmentation_source_name=REFERENCE_SEGMENTATION_NAME,
        analysis_id="normal_all_datasets_pooled",
        analysis_label="All 8 Normal datasets pooled",
        streaming_tag_locus="Nanog/Sox2",
        mintbody_label="Ser5ph/H3K27ac",
        replicate="rep1/rep2",
        chamber="ChamberA/ChamberB",
        expected_fov_count=None,
        random_seed=20260720,
    )


def condition_config(
    condition_id: str,
    bundles: Iterable[NormalDatasetBundle],
) -> normal_aligned.AnalysisConfig:
    matched = [
        bundle for bundle in bundles if bundle.condition_id == condition_id
    ]
    if len(matched) != 2:
        raise ValueError(
            f"{condition_id}: expected two replicate datasets, got "
            f"{len(matched)}"
        )
    record = matched[0].record
    reference = matched[0].analysis_config
    return normal_aligned.AnalysisConfig(
        input_nd2=reference.input_nd2,
        project_dir=PROJECT,
        output_name=(
            f"{AGGREGATE_OUTPUT_NAME}/condition_thresholds/{condition_id}"
        ),
        segmentation_source_name=reference.segmentation_source_name,
        analysis_id=f"{condition_id}_pooled_replicates",
        analysis_label=(
            f"{record['streaming_tag_locus']} × {record['mintbody']} — "
            "rep1 + rep2 pooled (Normal)"
        ),
        streaming_tag_locus=str(record["streaming_tag_locus"]),
        mintbody_label=str(record["mintbody"]),
        replicate="rep1+rep2",
        chamber="ChamberA+ChamberB",
        expected_fov_count=None,
        random_seed=20260720,
    )


def bundle_table(
    bundles: Iterable[NormalDatasetBundle],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        record = bundle.record
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "condition_id": bundle.condition_id,
                "locus": record["streaming_tag_locus"],
                "mintbody": record["mintbody"],
                "replicate": record["replicate"],
                "chamber": record["chamber"],
                "field": int(record["field"]),
                "fov_count": int(record["fov_count"]),
                "input_nd2": str(bundle.analysis_config.input_nd2),
                "segmentation_root": str(
                    bundle.segmentation_config.output_root
                ),
                "analysis_root": str(bundle.analysis_config.output_root),
                "reference_cache_reused": bundle.is_reference,
            }
        )
    return pd.DataFrame(rows)


def preflight_datasets(
    bundles: Iterable[NormalDatasetBundle],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = gao.collect_metadata(bundle.analysis_config)
        channel_map = metadata["channel_mapping"]
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "condition_id": bundle.condition_id,
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
                "all_checks_passed": bool(
                    int(metadata["sizes"]["P"])
                    == int(bundle.record["fov_count"])
                    and int(metadata["sizes"]["Z"]) == 11
                    and int(metadata["sizes"]["C"]) == 3
                    and np.isclose(
                        float(metadata["voxel_size_um"]["x"]),
                        0.065,
                    )
                    and np.isclose(
                        float(metadata["voxel_size_um"]["z"]),
                        0.5,
                    )
                ),
            }
        )
    config = aggregate_config()
    gao.prepare_directories(config)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        config.output_root / "tables" / "dataset_preflight.csv",
        index=False,
    )
    return frame


def _segmentation_complete(bundle: NormalDatasetBundle) -> bool:
    root = bundle.segmentation_config.output_root
    expected = int(bundle.record["fov_count"])
    return bool(
        (
            root / "tables" / "segmentation_regions_all_fovs.csv"
        ).is_file()
        and len(
            list(
                (root / "segmentation_masks").glob(
                    "fov_*_nuclei_mask.tiff"
                )
            )
        )
        == expected
        and len(
            list(
                (root / "segmentation_overlays").glob(
                    "fov_*_nuclei_overlay.png"
                )
            )
        )
        == expected
    )


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_segmentation_smoke_test(
    bundle: NormalDatasetBundle,
    fov_id: int = 0,
) -> dict[str, Any]:
    if not _segmentation_complete(bundle):
        normal_seg.run_segmentation(
            bundle.segmentation_config,
            fov_ids=[int(fov_id)],
        )
    regions = pd.read_csv(
        bundle.segmentation_config.output_root
        / "segmentation_cache"
        / f"fov_{int(fov_id):03d}_regions.csv"
    )
    return {
        "analysis_id": bundle.analysis_id,
        "fov_id": int(fov_id),
        "selected_regions": int(regions["selected"].astype(bool).sum()),
        "overlay": str(
            bundle.segmentation_config.output_root
            / "segmentation_overlays"
            / f"fov_{int(fov_id):03d}_nuclei_overlay.png"
        ),
    }


def first_pending_segmentation(
    bundles: Iterable[NormalDatasetBundle],
) -> NormalDatasetBundle:
    return next(
        bundle
        for bundle in bundles
        if not _segmentation_complete(bundle)
    )


def run_all_segmentations(
    bundles: Iterable[NormalDatasetBundle],
) -> pd.DataFrame:
    bundles = list(bundles)
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        print(f"=== Segmentation: {bundle.analysis_id} ===", flush=True)
        if _segmentation_complete(bundle):
            method = "reuse_complete_cache"
        else:
            normal_seg.run_segmentation(bundle.segmentation_config)
            method = "cellpose_nuclei"
        n_fovs = int(bundle.record["fov_count"])
        representative = tuple(
            int(value)
            for value in np.linspace(0, n_fovs - 1, 6)
        )
        normal_seg.make_segmentation_contact_sheet(
            bundle.segmentation_config,
            fov_ids=representative,
        )
        regions = normal_aligned.reuse_reviewed_segmentation(
            bundle.analysis_config
        )
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
    config = aggregate_config()
    frame = pd.DataFrame(rows)
    frame.to_csv(
        config.output_root / "tables" / "segmentation_batch_summary.csv",
        index=False,
    )
    paths = [
        bundle.segmentation_config.output_root
        / "figures"
        / "segmentation_representative_contact_sheet.png"
        for bundle in bundles
    ]
    titles = [bundle.analysis_id for bundle in bundles]
    sora_batch._contact_sheet(
        paths,
        titles,
        config.output_root
        / "figures"
        / "segmentation_by_dataset_contact_sheet.png",
        columns=2,
        panel_width=7.8,
        panel_height=6.0,
    )
    return frame


def run_all_quantifications(
    bundles: Iterable[NormalDatasetBundle],
    *,
    workers: int = 10,
    chunk_size: int = 5,
) -> pd.DataFrame:
    bundles = list(bundles)
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        print(f"=== Gao 3D: {bundle.analysis_id} ===", flush=True)
        _, _, summary = sora_batch.run_quantification_parallel(
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
        n_fovs = int(bundle.record["fov_count"])
        representative = tuple(
            int(value)
            for value in np.linspace(0, n_fovs - 1, 6)
        )
        normal_aligned.make_spot_contact_sheet(
            bundle.analysis_config,
            fov_ids=representative,
        )
    config = aggregate_config()
    frame = pd.DataFrame(rows)
    frame.to_csv(
        config.output_root / "tables" / "quantification_batch_summary.csv",
        index=False,
    )
    paths = [
        bundle.analysis_config.output_root
        / "figures"
        / "spot_detection_representative_contact_sheet.png"
        for bundle in bundles
    ]
    sora_batch._contact_sheet(
        paths,
        [bundle.analysis_id for bundle in bundles],
        config.output_root
        / "figures"
        / "spot_detection_by_dataset_contact_sheet.png",
        columns=2,
        panel_width=7.8,
        panel_height=6.0,
    )
    return frame


def _tag_candidates(
    candidates: pd.DataFrame,
    bundle: NormalDatasetBundle,
) -> pd.DataFrame:
    tagged = candidates.copy()
    tagged.insert(0, "analysis_id", bundle.analysis_id)
    tagged.insert(1, "condition_id", bundle.condition_id)
    tagged.insert(
        2,
        "streaming_tag_locus",
        str(bundle.record["streaming_tag_locus"]),
    )
    tagged.insert(3, "mintbody", str(bundle.record["mintbody"]))
    tagged.insert(4, "replicate", str(bundle.record["replicate"]))
    return tagged


def _decorate_decision(
    decision: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    decision.update(
        {
            "status": "fixed_cell_thresholds_applied_to_Normal",
            "scope": scope,
            "selected_mtetr_source": "fixed-cell threshold 1.15",
            "selected_mcp_source": "fixed-cell threshold 1.15",
            "interpretation_boundary": (
                "Fixation changes mTetR localization and intensity. The "
                "fixed-cell r_mTetR/r_MCP values are classification cutoffs; "
                "KDE crosspoints are diagnostic only and no direct numerical "
                "equivalence to live-cell thresholds is claimed."
            ),
        }
    )
    return decision


def _threshold_figure(
    candidates: pd.DataFrame,
    config: normal_aligned.AnalysisConfig,
    scope: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    gao.prepare_directories(config)
    decision, artifacts, sensitivity = gao.calibrate_thresholds(
        candidates,
        config,
    )
    decision = _decorate_decision(decision, scope)
    gao.write_json(config.output_root / "threshold_decision.json", decision)
    paths = gao.make_calibration_figures(
        candidates,
        decision,
        artifacts,
        sensitivity,
        config,
    )
    return decision, {key: Path(value) for key, value in paths.items()}


def run_threshold_calibrations(
    bundles: Iterable[NormalDatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    config = aggregate_config()
    gao.prepare_directories(config)
    tagged_frames: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, Any]] = []
    dataset_figures: list[Path] = []

    for bundle in bundles:
        candidates = pd.read_csv(
            bundle.analysis_config.output_root
            / "tables"
            / "mtetr_mcp_spot_candidates.csv"
        )
        tagged_frames.append(_tag_candidates(candidates, bundle))
        decision, paths = _threshold_figure(
            candidates,
            bundle.analysis_config,
            f"dataset:{bundle.analysis_id}",
        )
        dataset_figures.append(paths["calibration_png"])
        dataset_rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "condition_id": bundle.condition_id,
                "rank1_candidate_count": decision[
                    "rank1_candidate_count"
                ],
                "calibration_count_crop_mean_at_least_90": decision[
                    "rank1_calibration_count_crop_mean_at_least_90"
                ],
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

    all_candidates = pd.concat(tagged_frames, ignore_index=True)
    all_candidates.to_csv(
        config.output_root
        / "tables"
        / "all_datasets_spot_candidates.csv",
        index=False,
    )
    dataset_diagnostics = pd.DataFrame(dataset_rows)
    dataset_diagnostics.to_csv(
        config.output_root
        / "tables"
        / "threshold_diagnostics_by_dataset.csv",
        index=False,
    )
    dataset_contact = sora_batch._contact_sheet(
        dataset_figures,
        [bundle.analysis_id for bundle in bundles],
        config.output_root
        / "figures"
        / "threshold_calibration_by_dataset_contact_sheet.png",
        columns=2,
    )

    condition_rows: list[dict[str, Any]] = []
    condition_figures: list[Path] = []
    condition_ids = sorted(
        {bundle.condition_id for bundle in bundles}
    )
    for condition_id in condition_ids:
        condition_candidates = all_candidates.loc[
            all_candidates["condition_id"] == condition_id
        ].copy()
        condition_cfg = condition_config(condition_id, bundles)
        gao.prepare_directories(condition_cfg)
        condition_candidates.to_csv(
            condition_cfg.output_root
            / "tables"
            / "condition_pooled_spot_candidates.csv",
            index=False,
        )
        decision, paths = _threshold_figure(
            condition_candidates,
            condition_cfg,
            f"condition:{condition_id}:rep1+rep2",
        )
        condition_figures.append(paths["calibration_png"])
        condition_rows.append(
            {
                "condition_id": condition_id,
                "candidate_rows": int(len(condition_candidates)),
                "rank1_candidate_count": decision[
                    "rank1_candidate_count"
                ],
                "mtetr_kde_diagnostic": decision[
                    "mtetr_kde_crosspoint_raw_diagnostic"
                ],
                "mcp_kde_diagnostic": decision[
                    "mcp_kde_crosspoint_raw_diagnostic"
                ],
                "selected_r_mtetr": decision["selected_mtetr_r_mass"],
                "selected_r_mcp": decision["selected_mcp_r_mass"],
            }
        )
    condition_diagnostics = pd.DataFrame(condition_rows)
    condition_diagnostics.to_csv(
        config.output_root
        / "tables"
        / "threshold_diagnostics_by_condition.csv",
        index=False,
    )
    condition_contact = sora_batch._contact_sheet(
        condition_figures,
        condition_ids,
        config.output_root
        / "figures"
        / "threshold_calibration_by_condition_contact_sheet.png",
        columns=2,
    )

    pooled_decision, pooled_paths = _threshold_figure(
        all_candidates,
        config,
        "all_8_Normal_datasets_pooled",
    )
    return {
        "dataset_diagnostics": dataset_diagnostics,
        "condition_diagnostics": condition_diagnostics,
        "pooled_decision": pooled_decision,
        "dataset_contact_sheet": dataset_contact,
        "condition_contact_sheet": condition_contact,
        "pooled_figure": pooled_paths["calibration_png"],
        "pooled_sensitivity_figure": pooled_paths["sensitivity_png"],
        "pooled_candidate_count": int(len(all_candidates)),
    }


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
        "Normal state calls by dataset\n"
        "r_mTetR≥1.15, r_MCP≥1.15, distance≤0.39 µm"
    )
    axis.legend(frameon=False, ncol=2)
    axis.tick_params(axis="x", rotation=35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def run_states_and_locus_summaries(
    bundles: Iterable[NormalDatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    config = aggregate_config()
    state_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gao_figures: list[Path] = []

    for bundle in bundles:
        analysis_config = bundle.analysis_config
        candidates = pd.read_csv(
            analysis_config.output_root
            / "tables"
            / "mtetr_mcp_spot_candidates.csv"
        )
        decision = json.loads(
            (
                analysis_config.output_root / "threshold_decision.json"
            ).read_text(encoding="utf-8")
        )
        states = gao.call_states(candidates, decision, analysis_config)
        gao.make_rank1_crop_qc(states, decision, analysis_config)
        counts = states["state"].value_counts().to_dict()
        for state, count in counts.items():
            state_rows.append(
                {
                    "analysis_id": bundle.analysis_id,
                    "condition_id": bundle.condition_id,
                    "state": str(state),
                    "count": int(count),
                }
            )
        active = int(counts.get("Active", 0))
        inactive = int(counts.get("Inactive", 0))
        gao_status = "not_applicable_missing_active_or_inactive"
        if active > 0 and inactive > 0:
            figure, _ = gao.make_locus_summary(states, analysis_config)
            gao_figures.append(Path(figure))
            gao_status = "completed"
        report = normal_aligned.validate_outputs(analysis_config)
        eligible = active + inactive
        summary_rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "condition_id": bundle.condition_id,
                "locus": bundle.record["streaming_tag_locus"],
                "mintbody": bundle.record["mintbody"],
                "replicate": bundle.record["replicate"],
                "fov_count": int(bundle.record["fov_count"]),
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

    tables = config.output_root / "tables"
    figures = config.output_root / "figures"
    state_long = pd.DataFrame(state_rows)
    state_summary = pd.DataFrame(summary_rows)
    state_long.to_csv(
        tables / "state_counts_by_dataset_long.csv",
        index=False,
    )
    state_summary.to_csv(
        tables / "state_summary_by_dataset.csv",
        index=False,
    )
    condition_summary = (
        state_summary.groupby(
            ["condition_id", "locus", "mintbody"],
            as_index=False,
        )[
            [
                "fov_count",
                "rank1_candidates",
                "valid_crops",
                "eligible_mtetr",
                "active",
                "inactive",
                "excluded_below_mtetr",
                "excluded_no_valid_crop",
            ]
        ]
        .sum()
    )
    condition_summary["active_fraction_of_eligible"] = (
        condition_summary["active"]
        / condition_summary["eligible_mtetr"].replace(0, np.nan)
    )
    condition_summary.to_csv(
        tables / "state_summary_by_condition.csv",
        index=False,
    )
    pooled_summary = pd.DataFrame(
        [
            {
                "dataset_count": int(len(state_summary)),
                "fov_count": int(state_summary["fov_count"].sum()),
                "rank1_candidates": int(
                    state_summary["rank1_candidates"].sum()
                ),
                "valid_crops": int(state_summary["valid_crops"].sum()),
                "eligible_mtetr": int(
                    state_summary["eligible_mtetr"].sum()
                ),
                "active": int(state_summary["active"].sum()),
                "inactive": int(state_summary["inactive"].sum()),
                "excluded_below_mtetr": int(
                    state_summary["excluded_below_mtetr"].sum()
                ),
                "excluded_no_valid_crop": int(
                    state_summary["excluded_no_valid_crop"].sum()
                ),
            }
        ]
    )
    pooled_summary["active_fraction_of_eligible"] = (
        pooled_summary["active"]
        / pooled_summary["eligible_mtetr"].replace(0, np.nan)
    )
    pooled_summary.to_csv(
        tables / "state_summary_all_datasets_pooled.csv",
        index=False,
    )
    state_figure = _plot_state_counts(
        state_long,
        figures / "state_counts_by_dataset.png",
    )
    gao_contact = sora_batch._contact_sheet(
        gao_figures,
        [
            bundle.analysis_id
            for bundle in bundles
            if (
                bundle.analysis_config.output_root
                / "figures"
                / "gao_fig1c_method_composite.png"
            ).is_file()
        ],
        figures / "gao_fig1c_by_dataset_contact_sheet.png",
        columns=2,
        panel_width=8.0,
        panel_height=6.2,
    )
    return {
        "state_summary": state_summary,
        "condition_summary": condition_summary,
        "pooled_summary": pooled_summary,
        "state_counts_figure": state_figure,
        "gao_contact_sheet": gao_contact,
    }


def finalize_batch(
    bundles: Iterable[NormalDatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    config = aggregate_config()
    tables = config.output_root / "tables"
    figures = config.output_root / "figures"
    manifests = {
        bundle.analysis_id: json.loads(
            (
                bundle.analysis_config.output_root
                / "analysis_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for bundle in bundles
    }
    dataset_diagnostics = pd.read_csv(
        tables / "threshold_diagnostics_by_dataset.csv"
    )
    condition_diagnostics = pd.read_csv(
        tables / "threshold_diagnostics_by_condition.csv"
    )
    state_summary = pd.read_csv(
        tables / "state_summary_by_dataset.csv"
    )
    pooled_decision = json.loads(
        (config.output_root / "threshold_decision.json").read_text(
            encoding="utf-8"
        )
    )
    total_fovs = int(
        sum(int(bundle.record["fov_count"]) for bundle in bundles)
    )
    cache_counts = {
        bundle.analysis_id: len(
            list(
                (
                    bundle.analysis_config.output_root
                    / "per_fov_cache"
                ).glob("fov_*_cache_manifest.json")
            )
        )
        for bundle in bundles
    }
    validations = {
        "eight_datasets": len(bundles) == 8,
        "four_conditions_two_replicates_each": bool(
            bundle_table(bundles)
            .groupby("condition_id")
            .size()
            .eq(2)
            .all()
        ),
        "total_1300_fovs": total_fovs == 1300,
        "all_fov_caches_complete": all(
            cache_counts[bundle.analysis_id]
            == int(bundle.record["fov_count"])
            for bundle in bundles
        ),
        "eight_analysis_manifests": len(manifests) == 8,
        "all_analysis_manifests_passed": all(
            manifest["all_validations_passed"]
            for manifest in manifests.values()
        ),
        "eight_dataset_threshold_diagnostics": (
            len(dataset_diagnostics) == 8
        ),
        "four_condition_threshold_diagnostics": (
            len(condition_diagnostics) == 4
        ),
        "eight_state_summaries": len(state_summary) == 8,
        "fixed_thresholds_1p15": bool(
            np.isclose(
                pooled_decision["selected_mtetr_r_mass"],
                1.15,
            )
            and np.isclose(
                pooled_decision["selected_mcp_r_mass"],
                1.15,
            )
        ),
        "dataset_threshold_contact_sheet": (
            figures
            / "threshold_calibration_by_dataset_contact_sheet.png"
        ).is_file(),
        "condition_threshold_contact_sheet": (
            figures
            / "threshold_calibration_by_condition_contact_sheet.png"
        ).is_file(),
        "all_datasets_pooled_threshold_figure": (
            figures / "threshold_calibration_four_panel.png"
        ).is_file(),
        "state_counts_figure": (
            figures / "state_counts_by_dataset.png"
        ).is_file(),
        "gao_contact_sheet": (
            figures / "gao_fig1c_by_dataset_contact_sheet.png"
        ).is_file(),
    }
    report = {
        "analysis": (
            "Eight fixed-cell Normal datasets, Gao-compatible 3D batch"
        ),
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "dataset_count": len(bundles),
        "condition_count": 4,
        "total_fov_count": total_fovs,
        "selected_thresholds": {
            "r_mTetR": 1.15,
            "r_MCP": 1.15,
            "distance_um": 0.39,
        },
        "cache_counts": cache_counts,
        "state_summary": state_summary.to_dict(orient="records"),
        "dataset_threshold_diagnostics": dataset_diagnostics.to_dict(
            orient="records"
        ),
        "condition_threshold_diagnostics": condition_diagnostics.to_dict(
            orient="records"
        ),
        "validations": validations,
        "all_validations_passed": bool(all(validations.values())),
    }
    gao.write_json(
        config.output_root / "batch_analysis_manifest.json",
        report,
    )
    if not report["all_validations_passed"]:
        failed = [
            key for key, passed in validations.items() if not passed
        ]
        raise RuntimeError(f"Normal batch validation failed: {failed}")
    return report
