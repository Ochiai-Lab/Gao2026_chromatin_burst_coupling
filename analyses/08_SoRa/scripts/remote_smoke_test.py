#!/usr/bin/env python3
"""Run one Normal and one SoRa FOV through the revised cache writer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

import normal_field8_sora_aligned_pipeline as normal_aligned
import normal_multidataset_pipeline as normal_batch
import sora_gao3d_snapshot_pipeline as gao
import sora_publication_pipeline as sora_batch


def inspect_cache(path: Path) -> dict[str, int]:
    with np.load(path) as data:
        required = {
            "cell_ids",
            "locus_crops",
            "random_crops",
            "random_qc_cell_ids",
            "random_qc_z_indices",
            "random_qc_y_px",
            "random_qc_x_px",
            "random_qc_crops",
        }
        missing = required.difference(data.files)
        if missing:
            raise AssertionError(f"Missing arrays in {path}: {sorted(missing)}")
        if len(data["random_qc_cell_ids"]) != len(data["random_qc_crops"]):
            raise AssertionError(f"Random-control count mismatch: {path}")
        return {
            "rank1_crops": int(len(data["locus_crops"])),
            "all_qc_random_crops": int(len(data["random_qc_crops"])),
        }


def main() -> None:
    normal_source = normal_batch.build_dataset_bundles()[0].analysis_config
    normal_config = replace(
        normal_source,
        output_name="normal_revision_20260809_smoke_test_one_fov",
        overwrite_quantification=True,
    )
    normal_aligned.reuse_reviewed_segmentation(normal_config)
    gao.run_quantification(normal_config, fov_ids=[0])
    normal_counts = inspect_cache(
        normal_config.output_root
        / "per_fov_cache"
        / "fov_000_rank1_crops.npz"
    )

    sora_source = sora_batch.build_dataset_bundles()[0].analysis_config
    sora_config = replace(
        sora_source,
        output_name="sora_revision_20260809_smoke_test_one_fov",
        overwrite_quantification=True,
        require_reference_equivalence=False,
    )
    gao.run_segmentation(sora_config)
    gao.run_quantification(sora_config, fov_ids=[0])
    sora_counts = inspect_cache(
        sora_config.output_root
        / "per_fov_cache"
        / "fov_000_rank1_crops.npz"
    )

    assert normal_config.final_r_mass_radius_px == 6
    assert np.isclose(normal_config.r_mass_pad_um, 0.39)
    assert sora_config.final_r_mass_radius_px == 17
    assert np.isclose(sora_config.r_mass_pad_um, 0.39, atol=0.01)
    print(
        {
            "normal": normal_counts,
            "sora": sora_counts,
            "normal_r_mass_um": normal_config.r_mass_pad_um,
            "sora_r_mass_um": sora_config.r_mass_pad_um,
        }
    )


if __name__ == "__main__":
    main()
