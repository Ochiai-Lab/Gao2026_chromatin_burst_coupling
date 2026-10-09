"""Publication-only SoRa paired-field analysis for public replicate 1."""
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
import multiprocessing as mp
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sora_gao3d_snapshot_pipeline as gao
import sora_multidataset_pipeline as legacy_batch
import sora_snapshot_pipeline as segmentation


CODE_DIR = Path(__file__).resolve().parent
PROJECT = Path(
    os.environ.get(
        "SORA_PUBLICATION_PROJECT_DIR",
        str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO'),
    )
).resolve()
DATASET_MANIFEST = Path(
    os.environ.get(
        "SORA_PUBLICATION_MANIFEST",
        CODE_DIR / "sora_fixed_publication_manifest.json",
    )
)
AGGREGATE_OUTPUT_NAME = (
    "sora_snapshot_outputs_all_replicates_publication_gao3d_"
    "rmass0139um_r1035_mcp105_allqcrandom_revision_20260810"
)
REVISION_TAG = "rmass0139um_allqcrandom_revision_20260810"
ALL_QC_RANDOM_SOURCE_TAG = "rmass039um_revision_20260809"
FIXED_R_MTETR = 1.035
FIXED_R_MCP = 1.05
MCP_DISTANCE_UM = 0.39


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
        """Compatibility property used by legacy helper functions."""
        return self.record["analysis_status"] == "completed_existing"

    @property
    def is_existing(self) -> bool:
        return self.is_completed_reference


def _load_manifest() -> dict[str, Any]:
    payload = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if int(payload.get("dataset_count", -1)) != 8:
        raise ValueError("Expected exactly 8 acquisition-level datasets.")
    if int(payload.get("biological_replicate_group_count", -1)) != 4:
        raise ValueError("Expected 4 condition × biological-replicate groups.")
    return payload


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
    label = (
        f"{record['streaming_tag_locus']} × {record['mintbody']} — "
        f"{record['biological_replicate']} "
        f"({record['chamber']} Field{record['field']})"
    )
    note = str(record.get("qc_note", "")).strip()
    return f"{label}; QC: {note}" if note else label


def build_dataset_bundles() -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    for index, record in enumerate(_load_manifest()["records"]):
        common = {
            "input_nd2": _resolve_manifest_input_path(record["path"]),
            "project_dir": PROJECT,
            "analysis_id": str(record["analysis_id"]),
            "analysis_label": _analysis_label(record),
            "streaming_tag_locus": str(
                record["streaming_tag_locus"]
            ),
            "mintbody_label": str(record["mintbody"]),
            "replicate": str(record["biological_replicate"]),
            "chamber": str(record["chamber"]),
            "expected_fov_count": int(record["fov_count"]),
            "random_seed": int(record["random_seed"]),
        }
        segmentation_name = str(record["planned_segmentation_name"])
        analysis_name = (
            f"{record['planned_output_name']}_{REVISION_TAG}"
        )
        segmentation_config = segmentation.AnalysisConfig(
            **common,
            output_name=segmentation_name,
        )
        analysis_config = gao.AnalysisConfig(
            **common,
            output_name=analysis_name,
            segmentation_source_name=segmentation_name,
            fixed_mtetr_r_mass_threshold=FIXED_R_MTETR,
            fixed_mcp_r_mass_threshold=FIXED_R_MCP,
            require_reference_equivalence=False,
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
        segmentation_source_name=str(
            build_dataset_bundles()[0].record[
                "planned_segmentation_name"
            ]
        ),
        analysis_id="all_publication_pooled",
        analysis_label="All 8 SoRa acquisitions pooled (public replicate 1)",
        streaming_tag_locus="Nanog/Sox2",
        mintbody_label="Ser5ph/H3K27ac",
        replicate="rep1",
        chamber="ChamberA/B/C/D",
        expected_fov_count=None,
        fixed_mtetr_r_mass_threshold=FIXED_R_MTETR,
        fixed_mcp_r_mass_threshold=FIXED_R_MCP,
        require_reference_equivalence=False,
    )


# Reuse tested restartable per-FOV machinery while redirecting its globals to
# the new manifest and aggregate root.
legacy_batch.DATASET_MANIFEST = DATASET_MANIFEST
legacy_batch.AGGREGATE_OUTPUT_NAME = AGGREGATE_OUTPUT_NAME
legacy_batch.build_dataset_bundles = build_dataset_bundles
legacy_batch.aggregate_config = aggregate_config


def bundle_table(bundles: Iterable[DatasetBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        record = bundle.record
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "condition_id": record["condition_id"],
                "biological_replicate": record[
                    "biological_replicate"
                ],
                "chamber": record["chamber"],
                "field": int(record["field"]),
                "fov_count": int(record["fov_count"]),
                "analysis_status": record["analysis_status"],
                "qc_note": record.get("qc_note", ""),
                "input_nd2": str(bundle.analysis_config.input_nd2),
                "segmentation_root": str(
                    bundle.segmentation_config.output_root
                ),
                "analysis_root": str(bundle.analysis_config.output_root),
            }
        )
    return pd.DataFrame(rows)


def pooling_membership_table(
    bundles: Iterable[DatasetBundle],
) -> pd.DataFrame:
    """Make the acquisition-to-biological-sample pooling explicit."""

    rows = [
        {
            "condition_id": bundle.record["condition_id"],
            "biological_replicate": bundle.record[
                "biological_replicate"
            ],
            "chamber": bundle.record["chamber"],
            "field": int(bundle.record["field"]),
            "analysis_id": bundle.analysis_id,
            "qc_note": bundle.record.get("qc_note", ""),
        }
        for bundle in bundles
    ]
    acquisition = pd.DataFrame(rows)
    grouped_rows: list[dict[str, Any]] = []
    for (condition_id, replicate), frame in acquisition.groupby(
        ["condition_id", "biological_replicate"],
        sort=True,
    ):
        ordered = frame.sort_values(["chamber", "field", "analysis_id"])
        fields = ordered["field"].astype(int).tolist()
        analysis_ids = ordered["analysis_id"].astype(str).tolist()
        chambers = ordered["chamber"].astype(str).unique().tolist()
        expected_count = 2 if replicate in {"rep1"} else 1
        grouped_rows.append(
            {
                "condition_id": condition_id,
                "biological_replicate": replicate,
                "sample_group_id": f"{condition_id}_{replicate}",
                "source_chambers": "+".join(chambers),
                "source_fields": "+".join(str(value) for value in fields),
                "source_analysis_ids": " + ".join(analysis_ids),
                "acquisition_count": int(len(ordered)),
                "expected_acquisition_count": expected_count,
                "pooling_check_passed": bool(
                    len(ordered) == expected_count
                ),
                "qc_notes": " | ".join(
                    note
                    for note in ordered["qc_note"].astype(str)
                    if note.strip()
                ),
            }
        )
    output = pd.DataFrame(grouped_rows)
    expected_pairs = {
        "nanog_ser5ph": "1+8",
        "nanog_h3k27ac": "2+7",
        "sox2_ser5ph": "3+6",
        "sox2_h3k27ac": "4+5",
    }
    paired = output["biological_replicate"].isin(["rep1"])
    output.loc[paired, "pooling_check_passed"] &= output.loc[
        paired, "source_fields"
    ].eq(output.loc[paired, "condition_id"].map(expected_pairs))
    table_path = aggregate_config().output_root / "tables"
    table_path.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        table_path / "biological_sample_pooling_membership.csv",
        index=False,
    )
    if not output["pooling_check_passed"].all():
        failed = output.loc[
            ~output["pooling_check_passed"], "sample_group_id"
        ].tolist()
        raise RuntimeError(f"Field-pair pooling validation failed: {failed}")
    return output


def preflight_datasets(
    bundles: Iterable[DatasetBundle],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = gao.collect_metadata(bundle.analysis_config)
        sizes = metadata["sizes"]
        voxel = metadata["voxel_size_um"]
        channel_map = metadata["channel_mapping"]
        checks = {
            "fov_count": int(sizes["P"])
            == int(bundle.record["fov_count"]),
            "z_count": int(sizes["Z"]) == 11,
            "channel_count": int(sizes["C"]) == 3,
            "image_shape": (
                int(sizes["Y"]) == 2304 and int(sizes["X"]) == 2304
            ),
            "voxel_xy": np.isclose(
                float(voxel["x"]),
                0.023214285714285715,
            )
            and np.isclose(
                float(voxel["y"]),
                0.023214285714285715,
            ),
            "voxel_z": np.isclose(float(voxel["z"]), 0.5),
            "laser_order": [
                int(item["laser_nm"]) for item in channel_map
            ]
            == [640, 515, 445],
            "manifest_metadata": bool(
                bundle.record["all_metadata_checks_passed"]
            ),
        }
        rows.append(
            {
                "analysis_id": bundle.analysis_id,
                "condition_id": bundle.record["condition_id"],
                "biological_replicate": bundle.record[
                    "biological_replicate"
                ],
                "chamber": bundle.record["chamber"],
                "field": int(bundle.record["field"]),
                "fov_count": int(sizes["P"]),
                "z_count": int(sizes["Z"]),
                "channel_count": int(sizes["C"]),
                "voxel_x_um": float(voxel["x"]),
                "voxel_z_um": float(voxel["z"]),
                "channel_names": "|".join(
                    str(item["nd2_name"]) for item in channel_map
                ),
                "exposures_ms": "|".join(
                    str(item["exposure_ms"]) for item in channel_map
                ),
                "qc_note": bundle.record.get("qc_note", ""),
                "all_checks_passed": bool(all(checks.values())),
            }
        )
    frame = pd.DataFrame(rows)
    output = aggregate_config().output_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "dataset_preflight.csv", index=False)
    if not frame["all_checks_passed"].all():
        failed = frame.loc[
            ~frame["all_checks_passed"], "analysis_id"
        ].tolist()
        raise RuntimeError(f"Preflight failed: {failed}")
    return frame


def first_pending_bundle(
    bundles: Iterable[DatasetBundle],
) -> DatasetBundle:
    return next(bundle for bundle in bundles if not bundle.is_existing)


def run_segmentation_smoke_test(
    bundle: DatasetBundle,
    fov_id: int = 0,
) -> dict[str, Any]:
    return legacy_batch.run_segmentation_smoke_test(bundle, fov_id=fov_id)


def _read_regions(bundle: DatasetBundle) -> pd.DataFrame:
    return pd.read_csv(
        bundle.analysis_config.output_root
        / "tables"
        / "segmentation_regions_all_fovs.csv"
    )


def _segmentation_summary_row(
    bundle: DatasetBundle,
    regions: pd.DataFrame,
    method: str,
) -> dict[str, Any]:
    return {
        "analysis_id": bundle.analysis_id,
        "condition_id": bundle.record["condition_id"],
        "biological_replicate": bundle.record["biological_replicate"],
        "chamber": bundle.record["chamber"],
        "fov_count": int(bundle.record["fov_count"]),
        "selected_regions": int(regions["selected"].astype(bool).sum()),
        "method": method,
        "qc_note": bundle.record.get("qc_note", ""),
        "segmentation_root": str(bundle.segmentation_config.output_root),
    }


def _segment_new_bundle(bundle: DatasetBundle) -> dict[str, Any]:
    """Process one acquisition in an isolated process.

    Acquisition-level output directories never overlap.  A spawn context keeps
    CUDA initialization process-local while allowing CPU-heavy preprocessing
    and PNG rendering to proceed in parallel.
    """

    print(f"=== Segmentation worker: {bundle.analysis_id} ===", flush=True)
    segmentation.run_segmentation(bundle.segmentation_config)
    regions = gao.run_segmentation(bundle.analysis_config)
    row = _segmentation_summary_row(
        bundle,
        regions,
        method="cellpose_nuclei_new_then_read_only_reuse",
    )
    legacy_batch._release_accelerator_memory()
    return row


def _segmentation_cache_complete(bundle: DatasetBundle) -> bool:
    root = bundle.segmentation_config.output_root
    expected = int(bundle.record["fov_count"])
    return bool(
        (
            bundle.analysis_config.output_root
            / "tables"
            / "segmentation_regions_all_fovs.csv"
        ).is_file()
        and len(
            list(
                (root / "segmentation_inputs").glob(
                    "fov_*_all_channels_mixed.png"
                )
            )
        )
        == expected
        and len(
            list(
                (root / "segmentation_masks").glob(
                    "fov_*_selected_mask.tiff"
                )
            )
        )
        == expected
        and len(
            list(
                (root / "segmentation_overlays").glob(
                    "fov_*_selected_regions.png"
                )
            )
        )
        == expected
    )


def run_all_segmentations(
    bundles: Iterable[DatasetBundle],
    max_workers: int = 1,
) -> pd.DataFrame:
    bundle_list = list(bundles)
    max_workers = max(1, int(max_workers))
    rows: list[dict[str, Any]] = []
    pending: list[DatasetBundle] = []
    for bundle in bundle_list:
        if _segmentation_cache_complete(bundle):
            print(
                "=== Segmentation complete-cache reuse: "
                f"{bundle.analysis_id} ===",
                flush=True,
            )
            regions = _read_regions(bundle)
            rows.append(
                _segmentation_summary_row(
                    bundle,
                    regions,
                    method="read_only_reuse_complete_cache",
                )
            )
        else:
            pending.append(bundle)

    if max_workers == 1:
        rows.extend(_segment_new_bundle(bundle) for bundle in pending)
    elif pending:
        print(
            "Running acquisition-independent segmentation with "
            f"{max_workers} spawned workers.",
            flush=True,
        )
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        ) as executor:
            rows.extend(executor.map(_segment_new_bundle, pending))

    frame = pd.DataFrame(rows)
    order = {
        bundle.analysis_id: index
        for index, bundle in enumerate(bundle_list)
    }
    frame["_manifest_order"] = frame["analysis_id"].map(order)
    frame = (
        frame.sort_values("_manifest_order")
        .drop(columns="_manifest_order")
        .reset_index(drop=True)
    )
    output = aggregate_config().output_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "segmentation_batch_summary.csv", index=False)
    return frame


QUANTIFICATION_TABLE_FILES = (
    "segmentation_regions_all_fovs.csv",
    "mtetr_mcp_spot_candidates.csv",
    "spot_candidate_exclusions.csv",
    "mtetr_3d_detection_summary.csv",
    "fov001_010_sweep_equivalence.csv",
)


def _six_pixel_source_root(bundle: DatasetBundle) -> Path:
    return PROJECT / str(bundle.record["planned_output_name"])


def _all_qc_random_source_root(bundle: DatasetBundle) -> Path:
    return PROJECT / (
        f"{bundle.record['planned_output_name']}_"
        f"{ALL_QC_RANDOM_SOURCE_TAG}"
    )


def _ensure_file_link(
    source: Path,
    target: Path,
    *,
    overwrite: bool,
) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return "reused"
        if not overwrite:
            raise RuntimeError(f"Refusing to replace link: {target}")
        target.unlink()
    elif target.exists():
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite file: {target}")
        target.unlink()
    target.symlink_to(source)
    return "created"


def _ensure_directory_link(
    source: Path,
    target: Path,
    *,
    overwrite: bool,
) -> str:
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return "reused"
        if not overwrite:
            raise RuntimeError(f"Refusing to replace link: {target}")
        target.unlink()
    elif target.exists():
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            raise RuntimeError(
                "Target cache directory is not an empty setup artifact; "
                f"rename it before retrying: {target}"
            )
    target.symlink_to(source, target_is_directory=True)
    return "created"


def _write_json_guarded(
    path: Path,
    payload: dict[str, Any],
    *,
    identity_keys: tuple[str, ...],
    overwrite: bool,
) -> str:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if all(existing.get(key) == payload.get(key) for key in identity_keys):
            return "reused"
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite JSON: {path}")
    gao.write_json(path, payload)
    return "created"


def _validate_six_pixel_source(source_root: Path) -> None:
    decision_path = source_root / "threshold_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    radius = int(
        decision["final_spot_parameters"]["r_mass_radius_px"]
    )
    if radius != 6:
        raise RuntimeError(
            f"Expected a six-pixel r_mass source, found {radius}: "
            f"{source_root}"
        )


def _validate_all_qc_random_source(
    source_root: Path,
    expected_fovs: int,
) -> list[dict[str, Any]]:
    manifest_paths = sorted(
        (source_root / "per_fov_cache").glob(
            "fov_*_cache_manifest.json"
        )
    )
    if len(manifest_paths) != expected_fovs:
        raise RuntimeError(
            "Unexpected all-QC random-control manifest count: "
            f"{len(manifest_paths)} != {expected_fovs}; {source_root}"
        )
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in manifest_paths
    ]
    if not all(
        bool(item.get("complete"))
        and "all_qc_random_controls" in str(
            item.get("algorithm_version", "")
        )
        for item in manifests
    ):
        raise RuntimeError(
            f"Incomplete all-QC random-control source: {source_root}"
        )
    return manifests


def _seed_one_six_pixel_cache(
    bundle: DatasetBundle,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    target_root = bundle.analysis_config.output_root
    locus_source = _six_pixel_source_root(bundle)
    random_source = _all_qc_random_source_root(bundle)
    expected_fovs = int(bundle.record["fov_count"])

    _validate_six_pixel_source(locus_source)
    random_manifests = _validate_all_qc_random_source(
        random_source,
        expected_fovs,
    )
    locus_cache = locus_source / "per_fov_cache"
    if len(list(locus_cache.glob("fov_*_cache_manifest.json"))) != expected_fovs:
        raise RuntimeError(
            f"Incomplete six-pixel locus cache: {locus_source}"
        )

    target_root.mkdir(parents=True, exist_ok=True)
    _ensure_directory_link(
        locus_cache,
        target_root / "per_fov_cache",
        overwrite=overwrite,
    )
    _ensure_directory_link(
        random_source / "per_fov_cache",
        target_root / "random_control_cache",
        overwrite=overwrite,
    )
    for name in QUANTIFICATION_TABLE_FILES:
        source = locus_source / "tables" / name
        if source.is_file():
            _ensure_file_link(
                source,
                target_root / "tables" / name,
                overwrite=overwrite,
            )
    _ensure_file_link(
        locus_source / "nd2_metadata.json",
        target_root / "nd2_metadata.json",
        overwrite=overwrite,
    )
    _ensure_directory_link(
        locus_source / "qc" / "spot_detection_overlays",
        target_root / "qc" / "spot_detection_overlays",
        overwrite=overwrite,
    )

    candidates = pd.read_csv(
        target_root / "tables" / "mtetr_mcp_spot_candidates.csv"
    )
    exclusions = pd.read_csv(
        target_root / "tables" / "spot_candidate_exclusions.csv"
    )
    selected_nuclei = int(
        pd.read_csv(
            target_root / "tables" / "segmentation_regions_all_fovs.csv"
        )["selected"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    random_controls = int(
        sum(
            int(item.get("n_all_qc_random_crops", 0))
            for item in random_manifests
        )
    )
    if random_controls != selected_nuclei:
        raise RuntimeError(
            "Expected one all-QC random control per selected nucleus: "
            f"{random_controls} != {selected_nuclei}; {bundle.analysis_id}"
        )

    summary = {
        "analysis_id": bundle.analysis_id,
        "fov_count": expected_fovs,
        "completed_cache_manifests": expected_fovs,
        "saved_candidates": int(len(candidates)),
        "rank1_candidates": int(
            candidates["is_primary_candidate"].astype(bool).sum()
        ),
        "exclusion_records": int(len(exclusions)),
        "workers": [],
        "method": "read_only_composed_six_pixel_locus_and_all_qc_random",
    }
    _write_json_guarded(
        target_root / "parallel_cache_run_summary.json",
        summary,
        identity_keys=(
            "analysis_id",
            "fov_count",
            "completed_cache_manifests",
            "saved_candidates",
            "rank1_candidates",
            "exclusion_records",
            "method",
        ),
        overwrite=overwrite,
    )
    source_manifest = {
        "analysis_id": bundle.analysis_id,
        "algorithm_version": gao.ALGORITHM_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "fov_count": expected_fovs,
        "r_mass_radius_px": 6,
        "r_mass_radius_um": float(bundle.analysis_config.r_mass_pad_um),
        "mcp_distance_threshold_um": MCP_DISTANCE_UM,
        "locus_quantification_source": str(locus_source),
        "all_qc_random_control_source": str(random_source),
        "n_qc_selected_nuclei": selected_nuclei,
        "n_all_qc_random_controls": random_controls,
        "reuse_boundary": (
            "six-pixel candidate tables and locus crops are read-only; "
            "random_qc arrays are read-only and independent of r_mass"
        ),
    }
    _write_json_guarded(
        target_root / "cache_source_manifest.json",
        source_manifest,
        identity_keys=(
            "analysis_id",
            "algorithm_version",
            "complete",
            "fov_count",
            "r_mass_radius_px",
            "locus_quantification_source",
            "all_qc_random_control_source",
            "n_qc_selected_nuclei",
            "n_all_qc_random_controls",
        ),
        overwrite=overwrite,
    )
    return {
        **summary,
        "target_root": str(target_root),
        "locus_source": str(locus_source),
        "random_control_source": str(random_source),
        "n_all_qc_random_controls": random_controls,
    }


def seed_revised_six_pixel_caches(
    bundles: Iterable[DatasetBundle],
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compose new read-only cache roots without reprocessing ND2 images.

    The target root is deliberately separate from both source analyses. A
    mismatched existing file is never replaced unless ``overwrite=True``;
    non-empty cache directories are never removed by this function.
    """

    rows = [
        _seed_one_six_pixel_cache(bundle, overwrite=overwrite)
        for bundle in bundles
    ]
    frame = pd.DataFrame(rows)
    output = aggregate_config().output_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "cache_source_summary.csv"
    if summary_path.is_file() and not overwrite:
        existing = pd.read_csv(summary_path)
        stable_columns = [
            "analysis_id",
            "fov_count",
            "saved_candidates",
            "rank1_candidates",
            "n_all_qc_random_controls",
            "target_root",
            "locus_source",
            "random_control_source",
        ]
        if not existing[stable_columns].equals(frame[stable_columns]):
            raise RuntimeError(
                f"Refusing to overwrite cache source summary: {summary_path}"
            )
    else:
        frame.to_csv(summary_path, index=False)
    return frame


def _existing_quantification_summary(
    bundle: DatasetBundle,
) -> dict[str, Any]:
    candidates = pd.read_csv(
        bundle.analysis_config.output_root
        / "tables"
        / "mtetr_mcp_spot_candidates.csv"
    )
    exclusions = pd.read_csv(
        bundle.analysis_config.output_root
        / "tables"
        / "spot_candidate_exclusions.csv"
    )
    cache_count = len(
        list(
            (
                bundle.analysis_config.output_root / "per_fov_cache"
            ).glob("fov_*_cache_manifest.json")
        )
    )
    return {
        "analysis_id": bundle.analysis_id,
        "fov_count": int(bundle.record["fov_count"]),
        "completed_cache_manifests": cache_count,
        "saved_candidates": int(len(candidates)),
        "rank1_candidates": int(
            candidates["is_primary_candidate"].astype(bool).sum()
        ),
        "exclusion_records": int(len(exclusions)),
        "method": "read_only_reuse_completed_acquisitions",
    }


def _quantification_cache_complete(bundle: DatasetBundle) -> bool:
    root = bundle.analysis_config.output_root
    expected = int(bundle.record["fov_count"])
    source_manifest_path = root / "cache_source_manifest.json"
    if not source_manifest_path.is_file():
        return False
    try:
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False
    return bool(
        source_manifest.get("complete")
        and source_manifest.get("algorithm_version")
        == gao.ALGORITHM_VERSION
        and int(source_manifest.get("r_mass_radius_px", -1)) == 6
        and (
            root / "tables" / "mtetr_mcp_spot_candidates.csv"
        ).is_file()
        and (
            root / "tables" / "spot_candidate_exclusions.csv"
        ).is_file()
        and len(
            list(
                (root / "per_fov_cache").glob(
                    "fov_*_cache_manifest.json"
                )
            )
        )
        == expected
        and len(
            list(
                (root / "per_fov_cache").glob(
                    "fov_*_rank1_crops.npz"
                )
            )
        )
        == expected
        and len(
            list(
                (root / "random_control_cache").glob(
                    "fov_*_rank1_crops.npz"
                )
            )
        )
        == expected
    )


def run_all_quantifications(
    bundles: Iterable[DatasetBundle],
    *,
    workers: int = 8,
    chunk_size: int = 15,
    allow_raw_recompute: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        print(f"=== Gao 3D: {bundle.analysis_id} ===", flush=True)
        if _quantification_cache_complete(bundle):
            summary = _existing_quantification_summary(bundle)
            summary["method"] = (
                "read_only_composed_six_pixel_locus_and_all_qc_random"
            )
        else:
            if not allow_raw_recompute:
                raise RuntimeError(
                    "The composed six-pixel cache is incomplete for "
                    f"{bundle.analysis_id}. Run "
                    "seed_revised_six_pixel_caches() first. Raw ND2 "
                    "recomputation is disabled by default."
                )
            _, _, raw_summary = (
                legacy_batch.run_quantification_parallel(
                    bundle.analysis_config,
                    workers=workers,
                    chunk_size=chunk_size,
                )
            )
            summary = {
                key: value
                for key, value in raw_summary.items()
                if key != "workers"
            }
            summary["method"] = "new_restartable_per_fov_quantification"
        summary.update(
            {
                "condition_id": bundle.record["condition_id"],
                "biological_replicate": bundle.record[
                    "biological_replicate"
                ],
                "chamber": bundle.record["chamber"],
            }
        )
        rows.append(summary)
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
    columns: int = 3,
    panel_width: float = 7.2,
    panel_height: float = 5.7,
) -> Path:
    return legacy_batch._contact_sheet(
        image_paths,
        titles,
        output,
        columns=columns,
        panel_width=panel_width,
        panel_height=panel_height,
    )


def _gao_placeholder(
    bundle: DatasetBundle,
    states: pd.DataFrame,
    status: str,
) -> Path:
    """Keep acquisitions with insufficient state groups visible in Gao QC."""

    output = (
        aggregate_config().output_root
        / "figures"
        / "gao_placeholders"
        / f"{bundle.analysis_id}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = states["state"].value_counts()
    active = int(counts.get("Active", 0))
    inactive = int(counts.get("Inactive", 0))
    valid = int(np.isfinite(states["mTetR_r_mass"]).sum())
    qc_note = str(bundle.record.get("qc_note", "")).strip() or "none"
    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5,
        0.56,
        (
            f"{bundle.analysis_id}\n\n"
            "Gao Fig.1c summary not generated\n"
            "Insufficient Active/Inactive loci\n\n"
            f"Active={active}; Inactive={inactive}; valid crops={valid}\n"
            f"status={status}\n"
            f"QC: {qc_note}"
        ),
        ha="center",
        va="center",
        fontsize=14,
        wrap=True,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def representative_segmentation_contact(
    bundles: Iterable[DatasetBundle],
) -> Path:
    bundles = list(bundles)
    paths = [
        bundle.segmentation_config.output_root
        / "segmentation_overlays"
        / "fov_000_selected_regions.png"
        for bundle in bundles
    ]
    titles = [
        (
            f"{bundle.analysis_id}\n"
            f"{bundle.record.get('qc_note', '')}"
        ).strip()
        for bundle in bundles
    ]
    return _contact_sheet(
        paths,
        titles,
        aggregate_config().output_root
        / "figures"
        / "segmentation_representatives_8_acquisitions.png",
        columns=3,
    )


def representative_spot_contact(
    bundles: Iterable[DatasetBundle],
) -> Path:
    bundles = list(bundles)
    paths = [
        bundle.analysis_config.output_root
        / "qc"
        / "spot_detection_overlays"
        / "fov_000_3d_spot_overlay.png"
        for bundle in bundles
    ]
    return _contact_sheet(
        paths,
        [bundle.analysis_id for bundle in bundles],
        aggregate_config().output_root
        / "figures"
        / "spot_detection_representatives_8_acquisitions.png",
        columns=3,
    )


def _group_config(
    records: list[dict[str, Any]],
    *,
    group_kind: str,
    group_id: str,
    label: str,
) -> gao.AnalysisConfig:
    first = records[0]
    return gao.AnalysisConfig(
        input_nd2=Path(first["path"]),
        project_dir=PROJECT,
        output_name=(
            f"{AGGREGATE_OUTPUT_NAME}/{group_kind}/{group_id}"
        ),
        segmentation_source_name=str(
            first["planned_segmentation_name"]
        ),
        analysis_id=f"{group_kind}_{group_id}",
        analysis_label=label,
        streaming_tag_locus="/".join(
            sorted(
                {
                    str(record["streaming_tag_locus"])
                    for record in records
                }
            )
        ),
        mintbody_label="/".join(
            sorted({str(record["mintbody"]) for record in records})
        ),
        replicate="/".join(
            sorted(
                {
                    str(record["biological_replicate"])
                    for record in records
                }
            )
        ),
        chamber="/".join(
            sorted({str(record["chamber"]) for record in records})
        ),
        expected_fov_count=None,
        fixed_mtetr_r_mass_threshold=FIXED_R_MTETR,
        fixed_mcp_r_mass_threshold=FIXED_R_MCP,
        require_reference_equivalence=False,
    )


def _acquisition_result_config(
    bundle: DatasetBundle,
) -> gao.AnalysisConfig:
    """Separate threshold/state outputs from revision 3D caches."""

    source = bundle.analysis_config
    return gao.AnalysisConfig(
        input_nd2=source.input_nd2,
        project_dir=PROJECT,
        output_name=(
            f"{AGGREGATE_OUTPUT_NAME}/acquisition_results/"
            f"{bundle.analysis_id}"
        ),
        segmentation_source_name=str(
            bundle.record["planned_segmentation_name"]
        ),
        analysis_id=bundle.analysis_id,
        analysis_label=_analysis_label(bundle.record),
        streaming_tag_locus=str(
            bundle.record["streaming_tag_locus"]
        ),
        mintbody_label=str(bundle.record["mintbody"]),
        replicate=str(bundle.record["biological_replicate"]),
        chamber=str(bundle.record["chamber"]),
        expected_fov_count=int(bundle.record["fov_count"]),
        random_seed=source.random_seed,
        fixed_mtetr_r_mass_threshold=FIXED_R_MTETR,
        fixed_mcp_r_mass_threshold=FIXED_R_MCP,
        require_reference_equivalence=False,
    )


def _link_rank1_crop_archives(
    source_root: Path,
    target_root: Path,
    *,
    fov_ids: Iterable[int],
    fov_offset: int = 0,
) -> None:
    """Expose locus and all-QC random archives to one result root."""

    cache_pairs = (
        ("per_fov_cache", "per_fov_cache"),
        ("random_control_cache", "random_control_cache"),
    )
    for source_name, target_name in cache_pairs:
        source_cache = source_root / source_name
        if not source_cache.is_dir():
            raise FileNotFoundError(source_cache)
        target_cache = target_root / target_name
        target_cache.mkdir(parents=True, exist_ok=True)
        for fov_id in sorted(set(int(value) for value in fov_ids)):
            source = (
                source_cache / f"fov_{fov_id:03d}_rank1_crops.npz"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            target = (
                target_cache
                / f"fov_{fov_offset + fov_id:03d}_rank1_crops.npz"
            )
            if target.is_symlink():
                if target.resolve() == source.resolve():
                    continue
                raise RuntimeError(f"Refusing to replace crop link: {target}")
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite crop file: {target}")
            target.symlink_to(source)


def _group_gao_placeholder(
    config: gao.AnalysisConfig,
    states: pd.DataFrame,
    reason: str,
) -> Path:
    output = (
        config.output_root
        / "figures"
        / "gao_fig1c_method_composite_placeholder.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = states["state"].value_counts()
    figure, axis = plt.subplots(
        figsize=(7.4, 5.8),
        constrained_layout=True,
    )
    axis.axis("off")
    axis.text(
        0.5,
        0.62,
        config.analysis_label,
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        wrap=True,
    )
    axis.text(
        0.5,
        0.42,
        (
            "Combined biological-replicate Gao summary unavailable\n"
            f"reason: {reason}\n"
            f"Active={int(counts.get('Active', 0)):,}, "
            f"Inactive={int(counts.get('Inactive', 0)):,}"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def _decision_row(
    decision: dict[str, Any],
    **identity: Any,
) -> dict[str, Any]:
    return {
        **identity,
        "rank1_candidate_count": int(
            decision["rank1_candidate_count"]
        ),
        "calibration_count_crop_mean_at_least_90": int(
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
        "selected_r_mtetr": float(
            decision["selected_mtetr_r_mass"]
        ),
        "selected_r_mcp": float(decision["selected_mcp_r_mass"]),
    }


def _calibrate_group(
    candidates: pd.DataFrame,
    records: list[dict[str, Any]],
    *,
    group_kind: str,
    group_id: str,
    label: str,
) -> tuple[dict[str, Any], Path]:
    config = _group_config(
        records,
        group_kind=group_kind,
        group_id=group_id,
        label=label,
    )
    gao.prepare_directories(config)
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
    return decision, Path(paths["calibration_png"])


def run_threshold_calibrations(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    dataset_diagnostics: list[dict[str, Any]] = []
    dataset_frames: list[pd.DataFrame] = []
    dataset_figures: list[Path] = []
    record_by_id = {
        bundle.analysis_id: bundle.record for bundle in bundles
    }

    for bundle in bundles:
        source_config = bundle.analysis_config
        config = _acquisition_result_config(bundle)
        candidates = pd.read_csv(
            source_config.output_root
            / "tables"
            / "mtetr_mcp_spot_candidates.csv"
        )
        tagged = candidates.copy()
        tagged.insert(0, "analysis_id", bundle.analysis_id)
        tagged.insert(1, "condition_id", bundle.record["condition_id"])
        tagged.insert(
            2,
            "biological_replicate",
            bundle.record["biological_replicate"],
        )
        tagged.insert(3, "chamber", bundle.record["chamber"])
        tagged.insert(4, "field", int(bundle.record["field"]))
        dataset_frames.append(tagged)
        gao.prepare_directories(config)
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
        figure = Path(paths["calibration_png"])
        dataset_figures.append(figure)
        dataset_diagnostics.append(
            _decision_row(
                decision,
                analysis_id=bundle.analysis_id,
                condition_id=bundle.record["condition_id"],
                biological_replicate=bundle.record[
                    "biological_replicate"
                ],
                chamber=bundle.record["chamber"],
                qc_note=bundle.record.get("qc_note", ""),
            )
        )

    pooled_candidates = pd.concat(dataset_frames, ignore_index=True)
    pooled_config = aggregate_config()
    gao.prepare_directories(pooled_config)
    pooled_candidates.to_csv(
        pooled_config.output_root
        / "tables"
        / "all_acquisitions_spot_candidates.csv",
        index=False,
    )

    biological_rows: list[dict[str, Any]] = []
    biological_figures: list[Path] = []
    biological_titles: list[str] = []
    group_columns = ["condition_id", "biological_replicate"]
    for (condition_id, replicate), frame in pooled_candidates.groupby(
        group_columns,
        sort=True,
    ):
        group_id = f"{condition_id}_{replicate}"
        records = [
            record
            for record in record_by_id.values()
            if record["condition_id"] == condition_id
            and record["biological_replicate"] == replicate
        ]
        label = (
            f"{records[0]['streaming_tag_locus']} × "
            f"{records[0]['mintbody']} — {replicate}; "
            f"{'+'.join(sorted({r['chamber'] for r in records}))}"
        )
        decision, figure = _calibrate_group(
            frame,
            records,
            group_kind="biological_replicate_thresholds",
            group_id=group_id,
            label=label,
        )
        biological_figures.append(figure)
        biological_titles.append(group_id)
        biological_rows.append(
            _decision_row(
                decision,
                condition_id=condition_id,
                biological_replicate=replicate,
                acquisition_count=len(records),
            )
        )

    condition_rows: list[dict[str, Any]] = []
    condition_figures: list[Path] = []
    condition_titles: list[str] = []
    for condition_id, frame in pooled_candidates.groupby(
        "condition_id",
        sort=True,
    ):
        records = [
            record
            for record in record_by_id.values()
            if record["condition_id"] == condition_id
        ]
        label = (
            f"{records[0]['streaming_tag_locus']} × "
            f"{records[0]['mintbody']} — public replicate 1 pooled"
        )
        decision, figure = _calibrate_group(
            frame,
            records,
            group_kind="condition_thresholds",
            group_id=condition_id,
            label=label,
        )
        condition_figures.append(figure)
        condition_titles.append(condition_id)
        condition_rows.append(
            _decision_row(
                decision,
                condition_id=condition_id,
                acquisition_count=len(records),
            )
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

    tables = pooled_config.output_root / "tables"
    dataset_frame = pd.DataFrame(dataset_diagnostics)
    biological_frame = pd.DataFrame(biological_rows)
    condition_frame = pd.DataFrame(condition_rows)
    dataset_frame.to_csv(
        tables / "threshold_diagnostics_by_acquisition.csv",
        index=False,
    )
    biological_frame.to_csv(
        tables
        / "threshold_diagnostics_by_biological_replicate.csv",
        index=False,
    )
    condition_frame.to_csv(
        tables / "threshold_diagnostics_by_condition.csv",
        index=False,
    )

    figures = pooled_config.output_root / "figures"
    dataset_contact = _contact_sheet(
        dataset_figures,
        [bundle.analysis_id for bundle in bundles],
        figures / "threshold_calibration_by_acquisition.png",
        columns=3,
    )
    biological_contact = _contact_sheet(
        biological_figures,
        biological_titles,
        figures / "threshold_calibration_by_biological_replicate.png",
        columns=2,
    )
    condition_contact = _contact_sheet(
        condition_figures,
        condition_titles,
        figures / "threshold_calibration_by_condition.png",
        columns=2,
    )
    return {
        "acquisition_diagnostics": dataset_frame,
        "biological_replicate_diagnostics": biological_frame,
        "condition_diagnostics": condition_frame,
        "pooled_decision": pooled_decision,
        "pooled_figure": Path(pooled_paths["calibration_png"]),
        "acquisition_contact_sheet": dataset_contact,
        "biological_replicate_contact_sheet": biological_contact,
        "condition_contact_sheet": condition_contact,
        "pooled_candidate_count": int(len(pooled_candidates)),
    }


def _state_summary_row(
    bundle: DatasetBundle,
    states: pd.DataFrame,
    *,
    all_validations_passed: bool,
    gao_summary_status: str,
) -> dict[str, Any]:
    counts = states["state"].value_counts()
    active = int(counts.get("Active", 0))
    inactive = int(counts.get("Inactive", 0))
    eligible = active + inactive
    return {
        "analysis_id": bundle.analysis_id,
        "condition_id": bundle.record["condition_id"],
        "locus": bundle.record["streaming_tag_locus"],
        "mintbody": bundle.record["mintbody"],
        "biological_replicate": bundle.record[
            "biological_replicate"
        ],
        "chamber": bundle.record["chamber"],
        "field": int(bundle.record["field"]),
        "fov_count": int(bundle.record["fov_count"]),
        "qc_note": bundle.record.get("qc_note", ""),
        "rank1_candidates": int(len(states)),
        "valid_crops": int(np.isfinite(states["mTetR_r_mass"]).sum()),
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
        "gao_summary_status": gao_summary_status,
        "all_validations_passed": bool(all_validations_passed),
    }


def _aggregate_state_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    additive = [
        "fov_count",
        "rank1_candidates",
        "valid_crops",
        "eligible_mtetr",
        "active",
        "inactive",
        "excluded_below_mtetr",
        "excluded_no_valid_crop",
    ]
    output = (
        frame.groupby(group_columns, as_index=False, sort=True)[additive]
        .sum()
    )
    acquisition_counts = (
        frame.groupby(group_columns, as_index=False, sort=True)
        .size()
        .rename(columns={"size": "acquisition_count"})
    )
    output = output.merge(acquisition_counts, on=group_columns)
    output["active_fraction_of_eligible"] = (
        output["active"]
        / output["eligible_mtetr"].replace(0, np.nan)
    )
    return output


def _plot_state_counts(
    state_long: pd.DataFrame,
    output: Path,
) -> Path:
    state_order = [
        "Active",
        "Inactive",
        "Excluded_below_mTetR",
        "Excluded_no_valid_crop",
    ]
    pivot = (
        state_long.pivot(
            index="analysis_id",
            columns="state",
            values="count",
        )
        .fillna(0)
        .reindex(columns=state_order, fill_value=0)
    )
    fig, axis = plt.subplots(figsize=(22, 7.5), constrained_layout=True)
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=["tab:orange", "tab:blue", "0.65", "0.25"],
    )
    axis.set_ylabel("Nuclear regions")
    axis.set_xlabel("")
    axis.set_title(
        "State calls by acquisition\n"
        f"r_mTetR≥{FIXED_R_MTETR:g}, "
        f"r_MCP≥{FIXED_R_MCP:g}, distance≤{MCP_DISTANCE_UM:g} µm"
    )
    axis.legend(frameon=False, ncol=2)
    axis.tick_params(axis="x", rotation=55, labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_biological_replicates(
    summary: pd.DataFrame,
    output: Path,
) -> Path:
    condition_order = [
        "nanog_ser5ph",
        "nanog_h3k27ac",
        "sox2_ser5ph",
        "sox2_h3k27ac",
    ]
    replicate_order = ["rep1"]
    pivot = (
        summary.pivot(
            index="condition_id",
            columns="biological_replicate",
            values="active_fraction_of_eligible",
        )
        .reindex(index=condition_order, columns=replicate_order)
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    pivot.plot(
        kind="bar",
        ax=axis,
        color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Active fraction among mTetR-eligible loci")
    axis.set_xlabel("")
    axis.set_title(
        "Biological replicate comparison\n"
        "public replicate 1 pool paired fields after acquisition-level QC"
    )
    axis.legend(title="Biological replicate", frameon=False, ncol=4)
    axis.tick_params(axis="x", rotation=20)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_paired_field_results(
    summary: pd.DataFrame,
    output: Path,
) -> Path:
    """Show the eight Public replicate 1 paired-field samples as pooled results."""

    paired = summary.loc[
        summary["biological_replicate"].isin(["rep1"])
    ].copy()
    condition_order = {
        "nanog_ser5ph": 0,
        "nanog_h3k27ac": 1,
        "sox2_ser5ph": 2,
        "sox2_h3k27ac": 3,
    }
    replicate_order = {"rep1": 0}
    paired["_condition_order"] = paired["condition_id"].map(
        condition_order
    )
    paired["_replicate_order"] = paired["biological_replicate"].map(
        replicate_order
    )
    paired = paired.sort_values(
        ["_condition_order", "_replicate_order"]
    ).reset_index(drop=True)
    labels = [
        (
            f"{row.condition_id}\n{row.biological_replicate} "
            f"(Field {row.source_fields})"
        )
        for row in paired.itertuples(index=False)
    ]
    x_positions = np.arange(len(paired))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    axes[0].bar(
        x_positions,
        paired["inactive"],
        color="tab:blue",
        label="Inactive",
    )
    axes[0].bar(
        x_positions,
        paired["active"],
        bottom=paired["inactive"],
        color="tab:orange",
        label="Active",
    )
    axes[0].set_ylabel("mTetR-eligible loci")
    axes[0].set_title(
        "Public replicate 1 paired-field pooled state counts"
    )
    axes[0].legend(frameon=False)
    axes[1].bar(
        x_positions,
        paired["active_fraction_of_eligible"],
        color=[
            "#F58518" if rep == "rep1" else "#E45756"
            for rep in paired["biological_replicate"]
        ],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Active fraction among mTetR-eligible loci")
    axes[1].set_title(
        "Counts are pooled before calculating the fraction"
    )
    for axis in axes:
        axis.set_xticks(x_positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def _state_formula_passes(states: pd.DataFrame) -> bool:
    valid = np.isfinite(states["mTetR_r_mass"])
    eligible = valid & states["mTetR_r_mass"].ge(FIXED_R_MTETR)
    active = (
        eligible
        & np.isfinite(states["mcp_r_mass"])
        & states["mcp_r_mass"].ge(FIXED_R_MCP)
        & np.isfinite(states["mcp_distance_um"])
        & states["mcp_distance_um"].le(MCP_DISTANCE_UM)
    )
    expected = np.full(len(states), "Inactive", dtype=object)
    expected[~valid] = "Excluded_no_valid_crop"
    expected[valid & ~eligible] = "Excluded_below_mTetR"
    expected[active] = "Active"
    return bool(
        np.array_equal(
            expected,
            states["state"].astype(str).to_numpy(),
        )
        and np.isclose(
            states["mTetR_r_mass_threshold"],
            FIXED_R_MTETR,
        ).all()
        and np.isclose(
            states["mcp_r_mass_threshold"],
            FIXED_R_MCP,
        ).all()
        and np.isclose(
            states["mcp_distance_threshold_um"],
            MCP_DISTANCE_UM,
        ).all()
    )


def _make_biological_group_gao(
    group_bundles: list[DatasetBundle],
    states_by_analysis: dict[str, pd.DataFrame],
) -> tuple[Path, str, int]:
    """Pool raw crop archives across paired fields before Gao aggregation."""

    records = [bundle.record for bundle in group_bundles]
    condition_id = str(records[0]["condition_id"])
    replicate = str(records[0]["biological_replicate"])
    fields = "+".join(
        str(int(record["field"]))
        for record in sorted(records, key=lambda item: int(item["field"]))
    )
    group_id = f"{condition_id}_{replicate}"
    label = (
        f"{records[0]['streaming_tag_locus']} × "
        f"{records[0]['mintbody']} — {replicate}; pooled Field {fields}"
    )
    config = _group_config(
        records,
        group_kind="biological_replicate_results",
        group_id=group_id,
        label=label,
    )
    gao.prepare_directories(config)
    combined_frames: list[pd.DataFrame] = []
    for acquisition_index, bundle in enumerate(
        sorted(
            group_bundles,
            key=lambda item: int(item.record["field"]),
        )
    ):
        source_states = states_by_analysis[bundle.analysis_id].copy()
        source_states.insert(
            0,
            "source_analysis_id",
            bundle.analysis_id,
        )
        source_states.insert(
            1,
            "source_field",
            int(bundle.record["field"]),
        )
        source_states.insert(
            2,
            "source_fov_id",
            source_states["fov_id"].astype(int),
        )
        offset = acquisition_index * 1000
        _link_rank1_crop_archives(
            bundle.analysis_config.output_root,
            config.output_root,
            fov_ids=range(int(bundle.record["fov_count"])),
            fov_offset=offset,
        )
        source_states["fov_id"] = (
            source_states["fov_id"].astype(int) + offset
        )
        combined_frames.append(source_states)
    combined = pd.concat(combined_frames, ignore_index=True)
    combined.to_csv(
        config.output_root
        / "tables"
        / "pooled_locus_state_calls.csv",
        index=False,
    )
    counts = combined["state"].value_counts()
    if int(counts.get("Active", 0)) and int(
        counts.get("Inactive", 0)
    ):
        figure, _ = gao.make_locus_summary(combined, config)
        return Path(figure), "completed_pooled_raw_crops", int(len(combined))
    reason = "insufficient_active_or_inactive_loci"
    return (
        _group_gao_placeholder(config, combined, reason),
        reason,
        int(len(combined)),
    )


def run_states_and_locus_summaries(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    state_frames: list[pd.DataFrame] = []
    states_by_analysis: dict[str, pd.DataFrame] = {}

    for bundle in bundles:
        source_config = bundle.analysis_config
        config = _acquisition_result_config(bundle)
        candidates = pd.read_csv(
            source_config.output_root
            / "tables"
            / "mtetr_mcp_spot_candidates.csv"
        )
        decision = json.loads(
            (config.output_root / "threshold_decision.json").read_text(
                encoding="utf-8"
            )
        )
        states = gao.call_states(candidates, decision, config)
        _link_rank1_crop_archives(
            source_config.output_root,
            config.output_root,
            fov_ids=range(int(bundle.record["fov_count"])),
        )
        gao.make_rank1_crop_qc(states, decision, config)
        validation_passed = _state_formula_passes(states)
        gao_status = "deferred_to_biological_replicate_pool"
        states_by_analysis[bundle.analysis_id] = states

        tagged = states.copy()
        tagged.insert(0, "analysis_id", bundle.analysis_id)
        tagged.insert(1, "condition_id", bundle.record["condition_id"])
        tagged.insert(
            2,
            "biological_replicate",
            bundle.record["biological_replicate"],
        )
        tagged.insert(3, "chamber", bundle.record["chamber"])
        tagged.insert(4, "field", int(bundle.record["field"]))
        state_frames.append(tagged)
        counts = states["state"].value_counts().to_dict()
        for state, count in counts.items():
            long_rows.append(
                {
                    "analysis_id": bundle.analysis_id,
                    "condition_id": bundle.record["condition_id"],
                    "biological_replicate": bundle.record[
                        "biological_replicate"
                    ],
                    "chamber": bundle.record["chamber"],
                    "state": state,
                    "count": int(count),
                }
            )
        summary_rows.append(
            _state_summary_row(
                bundle,
                states,
                all_validations_passed=validation_passed,
                gao_summary_status=gao_status,
            )
        )

    pooled = aggregate_config()
    tables = pooled.output_root / "tables"
    figures = pooled.output_root / "figures"
    acquisition_summary = pd.DataFrame(summary_rows)
    state_long = pd.DataFrame(long_rows)
    all_states = pd.concat(state_frames, ignore_index=True)
    biological_summary = _aggregate_state_summary(
        acquisition_summary,
        [
            "condition_id",
            "locus",
            "mintbody",
            "biological_replicate",
        ],
    )
    membership = pooling_membership_table(bundles)
    membership_columns = [
        "condition_id",
        "biological_replicate",
        "sample_group_id",
        "source_chambers",
        "source_fields",
        "source_analysis_ids",
        "expected_acquisition_count",
        "pooling_check_passed",
        "qc_notes",
    ]
    biological_summary = biological_summary.merge(
        membership[membership_columns],
        on=["condition_id", "biological_replicate"],
        how="left",
        validate="one_to_one",
    )
    condition_summary = _aggregate_state_summary(
        acquisition_summary,
        ["condition_id", "locus", "mintbody"],
    )
    pooled_summary = pd.DataFrame(
        [
            {
                "dataset_count": len(acquisition_summary),
                "biological_replicate_group_count": len(
                    biological_summary
                ),
                "fov_count": int(acquisition_summary["fov_count"].sum()),
                "rank1_candidates": int(
                    acquisition_summary["rank1_candidates"].sum()
                ),
                "valid_crops": int(
                    acquisition_summary["valid_crops"].sum()
                ),
                "eligible_mtetr": int(
                    acquisition_summary["eligible_mtetr"].sum()
                ),
                "active": int(acquisition_summary["active"].sum()),
                "inactive": int(acquisition_summary["inactive"].sum()),
                "excluded_below_mtetr": int(
                    acquisition_summary[
                        "excluded_below_mtetr"
                    ].sum()
                ),
                "excluded_no_valid_crop": int(
                    acquisition_summary[
                        "excluded_no_valid_crop"
                    ].sum()
                ),
            }
        ]
    )
    pooled_summary["active_fraction_of_eligible"] = (
        pooled_summary["active"]
        / pooled_summary["eligible_mtetr"].replace(0, np.nan)
    )

    acquisition_summary.to_csv(
        tables / "state_summary_by_acquisition.csv",
        index=False,
    )
    biological_summary.to_csv(
        tables / "state_summary_by_biological_replicate.csv",
        index=False,
    )
    condition_summary.to_csv(
        tables / "state_summary_by_condition.csv",
        index=False,
    )
    pooled_summary.to_csv(
        tables / "state_summary_all_publication_pooled.csv",
        index=False,
    )
    state_long.to_csv(
        tables / "state_counts_by_acquisition_long.csv",
        index=False,
    )
    all_states.to_csv(
        tables / "all_acquisitions_locus_state_calls.csv",
        index=False,
    )

    state_counts_figure = _plot_state_counts(
        state_long,
        figures / "state_counts_by_acquisition.png",
    )
    biological_figure = _plot_biological_replicates(
        biological_summary,
        figures / "active_fraction_by_biological_replicate.png",
    )
    paired_field_figure = _plot_paired_field_results(
        biological_summary,
        figures / "paired_field_pooled_results_publication.png",
    )
    bundle_groups: dict[tuple[str, str], list[DatasetBundle]] = {}
    for bundle in bundles:
        key = (
            str(bundle.record["condition_id"]),
            str(bundle.record["biological_replicate"]),
        )
        bundle_groups.setdefault(key, []).append(bundle)
    group_figures: list[Path] = []
    group_titles: list[str] = []
    group_status: dict[str, str] = {}
    for row in biological_summary.itertuples(index=False):
        key = (str(row.condition_id), str(row.biological_replicate))
        figure, status, pooled_state_count = _make_biological_group_gao(
            bundle_groups[key],
            states_by_analysis,
        )
        group_figures.append(figure)
        group_titles.append(
            f"{row.sample_group_id}: Field {row.source_fields}"
        )
        group_status[str(row.sample_group_id)] = status
        if pooled_state_count != int(row.rank1_candidates):
            raise RuntimeError(
                "Pooled state count mismatch for "
                f"{row.sample_group_id}: {pooled_state_count} != "
                f"{int(row.rank1_candidates)}"
            )
    biological_summary["gao_summary_status"] = biological_summary[
        "sample_group_id"
    ].map(group_status)
    biological_summary.to_csv(
        tables / "state_summary_by_biological_replicate.csv",
        index=False,
    )
    gao_contact = _contact_sheet(
        group_figures,
        group_titles,
        figures / "gao_fig1c_by_biological_replicate.png",
        columns=2,
        panel_width=7.4,
        panel_height=5.8,
    )
    return {
        "acquisition_summary": acquisition_summary,
        "biological_replicate_summary": biological_summary,
        "paired_field_summary": biological_summary.loc[
            biological_summary["biological_replicate"].isin(
                ["rep1"]
            )
        ].reset_index(drop=True),
        "pooling_membership": membership,
        "condition_summary": condition_summary,
        "pooled_summary": pooled_summary,
        "state_counts_figure": state_counts_figure,
        "biological_replicate_figure": biological_figure,
        "paired_field_figure": paired_field_figure,
        "gao_contact_sheet": gao_contact,
        "gao_biological_replicate_contact_sheet": gao_contact,
    }


def _focus_qc_flagged(record: dict[str, Any]) -> bool:
    note = str(record.get("qc_note", "")).lower()
    return "focus-out" in note or "substantially above" in note


def run_focus_qc_sensitivity(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    """Repeat affected biological-sample summaries without focus-QC failures."""

    bundles = list(bundles)
    flagged = [bundle for bundle in bundles if _focus_qc_flagged(bundle.record)]
    output_root = aggregate_config().output_root / "focus_qc_sensitivity"
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    flagged_rows = bundle_table(flagged)
    flagged_rows.to_csv(
        tables / "excluded_acquisitions.csv",
        index=False,
    )
    all_states = pd.read_csv(
        aggregate_config().output_root
        / "tables"
        / "all_acquisitions_locus_state_calls.csv"
    )
    comparison_rows: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    figure_titles: list[str] = []

    affected_groups = sorted(
        {
            (
                str(bundle.record["condition_id"]),
                str(bundle.record["biological_replicate"]),
            )
            for bundle in flagged
        }
    )
    for condition_id, replicate in affected_groups:
        group_bundles = [
            bundle
            for bundle in bundles
            if str(bundle.record["condition_id"]) == condition_id
            and str(bundle.record["biological_replicate"]) == replicate
        ]
        retained = [
            bundle
            for bundle in group_bundles
            if not _focus_qc_flagged(bundle.record)
        ]
        if not retained:
            raise RuntimeError(
                f"No focus-QC-passing acquisition for {condition_id} {replicate}"
            )
        group_id = f"{condition_id}_{replicate}"
        label = (
            f"{retained[0].record['streaming_tag_locus']} x "
            f"{retained[0].record['mintbody']} - {replicate}; "
            "focus-QC-passing acquisitions only"
        )
        config = _group_config(
            [bundle.record for bundle in retained],
            group_kind="focus_qc_sensitivity",
            group_id=group_id,
            label=label,
        )
        gao.prepare_directories(config)
        state_frames: list[pd.DataFrame] = []
        for acquisition_index, bundle in enumerate(
            sorted(retained, key=lambda item: int(item.record["field"]))
        ):
            source_states = pd.read_csv(
                _acquisition_result_config(bundle).output_root
                / "tables"
                / "locus_state_calls.csv"
            )
            source_states.insert(
                0,
                "source_analysis_id",
                bundle.analysis_id,
            )
            source_states.insert(
                1,
                "source_fov_id",
                source_states["fov_id"].astype(int),
            )
            offset = acquisition_index * 1000
            _link_rank1_crop_archives(
                bundle.analysis_config.output_root,
                config.output_root,
                fov_ids=range(int(bundle.record["fov_count"])),
                fov_offset=offset,
            )
            source_states["fov_id"] = (
                source_states["fov_id"].astype(int) + offset
            )
            state_frames.append(source_states)
        sensitivity_states = pd.concat(state_frames, ignore_index=True)
        sensitivity_states.to_csv(
            config.output_root / "tables" / "pooled_locus_state_calls.csv",
            index=False,
        )
        state_counts = sensitivity_states["state"].value_counts()
        if not (
            int(state_counts.get("Active", 0))
            and int(state_counts.get("Inactive", 0))
        ):
            raise RuntimeError(
                f"Insufficient states after focus-QC exclusion: {group_id}"
            )
        figure, _ = gao.make_locus_summary(sensitivity_states, config)
        figure_paths.append(Path(figure))
        figure_titles.append(group_id)

        sensitivity_stats = pd.read_csv(
            config.output_root
            / "tables"
            / "gao_fig1c_core_annulus_statistics.csv"
        )
        primary_stats = pd.read_csv(
            aggregate_config().output_root
            / "biological_replicate_results"
            / group_id
            / "tables"
            / "gao_fig1c_core_annulus_statistics.csv"
        )
        primary_group_states = all_states.loc[
            (all_states["condition_id"] == condition_id)
            & (all_states["biological_replicate"] == replicate)
        ]
        primary_counts = primary_group_states["state"].value_counts()
        for mode, stats, counts in (
            ("all_acquisitions", primary_stats, primary_counts),
            ("focus_qc_passing_only", sensitivity_stats, state_counts),
        ):
            for row in stats.itertuples(index=False):
                comparison_rows.append(
                    {
                        "condition_id": condition_id,
                        "biological_replicate": replicate,
                        "analysis_mode": mode,
                        "channel": row.channel,
                        "active": int(counts.get("Active", 0)),
                        "inactive": int(counts.get("Inactive", 0)),
                        "active_median_core_minus_annulus": (
                            row.active_median_core_minus_annulus
                        ),
                        "inactive_median_core_minus_annulus": (
                            row.inactive_median_core_minus_annulus
                        ),
                        "pvalue_two_sided": row.pvalue_two_sided,
                    }
                )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        tables / "focus_qc_sensitivity_comparison.csv",
        index=False,
    )
    contact_sheet = _contact_sheet(
        figure_paths,
        figure_titles,
        figures / "focus_qc_passing_gao_summaries.png",
        columns=2,
        panel_width=7.4,
        panel_height=5.8,
    )
    return {
        "flagged_acquisitions": flagged_rows,
        "comparison": comparison,
        "contact_sheet": contact_sheet,
    }


def _write_acquisition_analysis_manifest(
    bundle: DatasetBundle,
) -> dict[str, Any]:
    """Validate split cache/state outputs and write one acquisition manifest."""

    source = bundle.analysis_config
    result = _acquisition_result_config(bundle)
    expected_fovs = int(bundle.record["fov_count"])
    cache_summary = json.loads(
        (source.output_root / "parallel_cache_run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = pd.read_csv(
        source.output_root / "tables" / "mtetr_mcp_spot_candidates.csv"
    )
    states = pd.read_csv(
        result.output_root / "tables" / "locus_state_calls.csv"
    )
    decision = json.loads(
        (result.output_root / "threshold_decision.json").read_text(
            encoding="utf-8"
        )
    )
    segmentation_regions = pd.read_csv(
        source.output_root / "tables" / "segmentation_regions_all_fovs.csv"
    )
    locus_cache_manifest_paths = sorted(
        (source.output_root / "per_fov_cache").glob(
            "fov_*_cache_manifest.json"
        )
    )
    locus_cache_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in locus_cache_manifest_paths
    ]
    random_cache_manifest_paths = sorted(
        (source.output_root / "random_control_cache").glob(
            "fov_*_cache_manifest.json"
        )
    )
    random_cache_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in random_cache_manifest_paths
    ]
    cache_source_manifest = json.loads(
        (source.output_root / "cache_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    selected_nuclei = int(
        segmentation_regions["selected"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    random_controls = int(
        sum(
            int(item.get("n_all_qc_random_crops", 0))
            for item in random_cache_manifests
        )
    )
    random_coverage = (
        random_controls / selected_nuclei if selected_nuclei else np.nan
    )
    validations = {
        "parallel_cache_summary_complete": bool(
            int(cache_summary["fov_count"]) == expected_fovs
            and int(cache_summary["completed_cache_manifests"])
            == expected_fovs
        ),
        "all_locus_cache_manifests_present": bool(
            len(locus_cache_manifest_paths) == expected_fovs
        ),
        "all_locus_cache_manifests_complete": bool(
            len(locus_cache_manifests) == expected_fovs
            and all(
                bool(item.get("complete"))
                for item in locus_cache_manifests
            )
        ),
        "all_random_cache_manifests_present": bool(
            len(random_cache_manifest_paths) == expected_fovs
        ),
        "all_random_cache_manifests_complete": bool(
            len(random_cache_manifests) == expected_fovs
            and all(
                bool(item.get("complete"))
                for item in random_cache_manifests
            )
        ),
        "all_qc_random_control_manifest_fields_present": bool(
            len(random_cache_manifests) == expected_fovs
            and all(
                (
                    int(item.get("n_nuclei", 0)) == 0
                    or (
                        "n_qc_nuclei_for_random_controls" in item
                        and "n_all_qc_random_crops" in item
                    )
                )
                and "all_qc_random_controls" in str(
                    item.get("algorithm_version", "")
                )
                for item in random_cache_manifests
            )
        ),
        "cache_source_manifest_passed": bool(
            cache_source_manifest.get("complete")
            and cache_source_manifest.get("algorithm_version")
            == gao.ALGORITHM_VERSION
            and int(
                cache_source_manifest.get("r_mass_radius_px", -1)
            )
            == 6
        ),
        "all_qc_random_control_population_nonzero": bool(
            random_controls > 0
        ),
        "all_qc_random_control_coverage_at_least_95pct": bool(
            np.isfinite(random_coverage) and random_coverage >= 0.95
        ),
        "state_rows_match_rank1_candidates": bool(
            len(states)
            == int(candidates["is_primary_candidate"].astype(bool).sum())
        ),
        "state_formula_passed": bool(_state_formula_passes(states)),
        "fixed_thresholds_passed": bool(
            np.isclose(
                float(decision["selected_mtetr_r_mass"]),
                FIXED_R_MTETR,
            )
            and np.isclose(
                float(decision["selected_mcp_r_mass"]),
                FIXED_R_MCP,
            )
            and np.isclose(
                float(decision["mcp_near_distance_um"]),
                MCP_DISTANCE_UM,
            )
        ),
        "rank1_crop_qc_exists": bool(
            (
                result.output_root / "qc" / "rank1_crop_qc.png"
            ).is_file()
        ),
        "r_mass_radius_is_6px_approximately_0p139um": bool(
            source.final_r_mass_radius_px == 6
            and np.isclose(source.r_mass_pad_um, 0.139, atol=0.002)
        ),
    }
    manifest = {
        "analysis": (
            f"{source.analysis_label} split cache/state revision analysis"
        ),
        "algorithm_version": (
            "sora_acquisition_split_output_validation_"
            "six_pixel_rmass_all_qc_random_v2"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": bundle.analysis_id,
        "source_cache_root": str(source.output_root),
        "acquisition_result_root": str(result.output_root),
        "expected_fov_count": expected_fovs,
        "n_rank1_candidates": int(len(states)),
        "n_qc_selected_nuclei": selected_nuclei,
        "n_all_qc_random_controls": random_controls,
        "random_control_coverage": float(random_coverage),
        "selected_thresholds": {
            "r_mTetR": FIXED_R_MTETR,
            "r_MCP": FIXED_R_MCP,
            "distance_um": MCP_DISTANCE_UM,
            "r_mass_radius_px": int(source.final_r_mass_radius_px),
            "r_mass_radius_um": float(source.r_mass_pad_um),
        },
        "validations": validations,
        "all_validations_passed": bool(all(validations.values())),
    }
    gao.write_json(source.output_root / "analysis_manifest.json", manifest)
    if not manifest["all_validations_passed"]:
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(
            f"Acquisition validation failed for {bundle.analysis_id}: "
            f"{failed}"
        )
    return manifest


def finalize_batch(
    bundles: Iterable[DatasetBundle],
) -> dict[str, Any]:
    bundles = list(bundles)
    pooled = aggregate_config()
    analysis_manifests: dict[str, Any] = {}
    cache_counts: dict[str, int] = {}
    for bundle in bundles:
        manifest_path = (
            bundle.analysis_config.output_root / "analysis_manifest.json"
        )
        existing_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        if existing_manifest and bool(
            existing_manifest.get("all_validations_passed")
        ):
            analysis_manifests[bundle.analysis_id] = existing_manifest
        else:
            analysis_manifests[bundle.analysis_id] = (
                _write_acquisition_analysis_manifest(bundle)
            )
        cache_counts[bundle.analysis_id] = len(
            list(
                (
                    bundle.analysis_config.output_root / "per_fov_cache"
                ).glob("fov_*_cache_manifest.json")
            )
        )

    tables = pooled.output_root / "tables"
    figures = pooled.output_root / "figures"
    acquisition_summary = pd.read_csv(
        tables / "state_summary_by_acquisition.csv"
    )
    biological_summary = pd.read_csv(
        tables / "state_summary_by_biological_replicate.csv"
    )
    condition_summary = pd.read_csv(
        tables / "state_summary_by_condition.csv"
    )
    acquisition_diagnostics = pd.read_csv(
        tables / "threshold_diagnostics_by_acquisition.csv"
    )
    biological_diagnostics = pd.read_csv(
        tables
        / "threshold_diagnostics_by_biological_replicate.csv"
    )
    condition_diagnostics = pd.read_csv(
        tables / "threshold_diagnostics_by_condition.csv"
    )
    pooling_membership = pd.read_csv(
        tables / "biological_sample_pooling_membership.csv"
    )
    pooled_decision = json.loads(
        (pooled.output_root / "threshold_decision.json").read_text(
            encoding="utf-8"
        )
    )
    expected_cache_counts = {
        bundle.analysis_id: int(bundle.record["fov_count"])
        for bundle in bundles
    }
    validations = {
        "8_acquisition_manifests": len(analysis_manifests) == 8,
        "all_acquisition_manifests_passed": all(
            bool(item["all_validations_passed"])
            for item in analysis_manifests.values()
        ),
        "total_publication_fovs": sum(expected_cache_counts.values()) == 3200,
        "all_fov_caches_complete": cache_counts
        == expected_cache_counts,
        "8_acquisition_summaries": len(acquisition_summary) == 8,
        "4_biological_replicate_summaries": (
            len(biological_summary) == 4
        ),
        "4_condition_summaries": len(condition_summary) == 4,
        "8_acquisition_threshold_diagnostics": (
            len(acquisition_diagnostics) == 8
        ),
        "4_biological_replicate_threshold_diagnostics": (
            len(biological_diagnostics) == 4
        ),
        "4_condition_threshold_diagnostics": (
            len(condition_diagnostics) == 4
        ),
        "pooled_thresholds_fixed": bool(
            np.isclose(
                float(pooled_decision["selected_mtetr_r_mass"]),
                FIXED_R_MTETR,
            )
            and np.isclose(
                float(pooled_decision["selected_mcp_r_mass"]),
                FIXED_R_MCP,
            )
        ),
        "all_acquisition_thresholds_fixed": bool(
            np.isclose(
                acquisition_diagnostics["selected_r_mtetr"],
                FIXED_R_MTETR,
            ).all()
            and np.isclose(
                acquisition_diagnostics["selected_r_mcp"],
                FIXED_R_MCP,
            ).all()
        ),
        "all_field_pairs_explicit_and_valid": bool(
            len(pooling_membership) == 4
            and pooling_membership["pooling_check_passed"].astype(
                bool
            ).all()
            and (
                pooling_membership.loc[
                    pooling_membership[
                        "biological_replicate"
                    ].isin(["rep1"]),
                    "acquisition_count",
                ]
                == 2
            ).all()
        ),
        "qc_notes_retained": bool(
            acquisition_summary["qc_note"]
            .fillna("")
            .str.contains("Z stack|focus-out", regex=True)
            .sum()
            == 2
        ),
        "focus_qc_sensitivity_completed": all(
            path.is_file() and path.stat().st_size > 0
            for path in [
                pooled.output_root
                / "focus_qc_sensitivity"
                / "tables"
                / "excluded_acquisitions.csv",
                pooled.output_root
                / "focus_qc_sensitivity"
                / "tables"
                / "focus_qc_sensitivity_comparison.csv",
                pooled.output_root
                / "focus_qc_sensitivity"
                / "figures"
                / "focus_qc_passing_gao_summaries.png",
            ]
        ),
        "required_figures": all(
            path.is_file() and path.stat().st_size > 0
            for path in [
                figures / "threshold_calibration_by_acquisition.png",
                figures
                / "threshold_calibration_by_biological_replicate.png",
                figures / "threshold_calibration_by_condition.png",
                figures / "threshold_calibration_four_panel.png",
                figures / "state_counts_by_acquisition.png",
                figures / "active_fraction_by_biological_replicate.png",
                figures / "paired_field_pooled_results_publication.png",
                figures / "segmentation_representatives_8_acquisitions.png",
                figures
                / "spot_detection_representatives_8_acquisitions.png",
                figures / "gao_fig1c_by_biological_replicate.png",
            ]
        ),
    }
    report = {
        "analysis": (
            "Fixed-cell SoRa public replicate 1, 8 acquisition Gao-compatible "
            "3D batch"
        ),
        "algorithm_version": (
            "sora_publication_gao3d_v3_all_qc_random_"
            "rmass6px_focus_sensitivity"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "dataset_count": len(bundles),
        "biological_replicate_group_count": len(biological_summary),
        "condition_count": len(condition_summary),
        "total_fov_count": int(sum(expected_cache_counts.values())),
        "selected_thresholds": {
            "r_mTetR": FIXED_R_MTETR,
            "r_MCP": FIXED_R_MCP,
            "distance_um": MCP_DISTANCE_UM,
            "r_mass_radius_px": 6,
            "r_mass_radius_um": float(
                build_dataset_bundles()[0].analysis_config.r_mass_pad_um
            ),
        },
        "reuse_boundary": {
            "all_acquisitions": (
                "reviewed segmentation masks and six-pixel locus "
                "quantification reused read-only; all-QC random controls "
                "reused read-only from the 20260809 revision cache in a "
                "separate date-stamped output root"
            ),
            "publication_field_pairs": (
                "1+8, 2+7, 3+6, and 4+5 pooled after "
                "acquisition-level QC; raw crop archives pooled before "
                "biological-replicate Gao Fig.1c aggregation"
            ),
        },
        "cache_counts": cache_counts,
        "acquisition_state_summary": acquisition_summary.to_dict(
            orient="records"
        ),
        "biological_replicate_state_summary": (
            biological_summary.to_dict(orient="records")
        ),
        "condition_state_summary": condition_summary.to_dict(
            orient="records"
        ),
        "validations": validations,
        "all_validations_passed": bool(all(validations.values())),
    }
    gao.write_json(
        pooled.output_root / "batch_analysis_manifest.json",
        report,
    )
    if not report["all_validations_passed"]:
        failed = [key for key, value in validations.items() if not value]
        raise RuntimeError(f"Batch validation failed: {failed}")
    return report
