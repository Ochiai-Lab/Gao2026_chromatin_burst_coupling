#!/usr/bin/env python3
"""Build clean revision copies of the Normal and SoRa snapshot notebooks."""

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
from typing import Any


SOURCE_DIR = Path(
    str(GAO2026_RAW_ROOT / 'Public-TS-873-2/Microscope/Ko/260717-SoRa_Fixed/ 260718-analysis-HO')
)
OUTPUT_DIR = Path(__file__).resolve().parent


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def clean_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "gao2026",
        "language": "python",
        "name": "gao2026",
    }
    return notebook


COMMON_IMPORT = """from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image as DisplayImage
from IPython.display import display

CODE_DIR = Path.cwd().resolve()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MODULE_FILES = [
    "sora_snapshot_pipeline.py",
    "sora_gao3d_snapshot_pipeline.py",
    "sora_multidataset_pipeline.py",
    "normal_field8_gao_pipeline.py",
    "normal_field8_sora_aligned_pipeline.py",
    "normal_multidataset_pipeline.py",
    "sora_rep1_rep4_pipeline.py",
]
MODULE_HASHES = {
    name: hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest()
    for name in MODULE_FILES
    if (CODE_DIR / name).is_file()
}
display(pd.Series(MODULE_HASHES, name="sha256").to_frame())
print({"hostname": platform.node(), "python": sys.executable})
"""


def build_normal() -> Path:
    source = SOURCE_DIR / "normal_fixed_snapshot_all_datasets_gao3d.ipynb"
    notebook = clean_notebook(json.loads(source.read_text(encoding="utf-8")))

    # Remove the unused method-specification markdown/code pair (old cells 6-7).
    del notebook["cells"][6:8]
    notebook["cells"][0] = markdown_cell(
        """# Normal fixed-cell snapshot analysis: revision 20260809

This is a clean revision copy of the eight-dataset Normal analysis. Reviewed
fixed-cell segmentation masks are reused read-only. Three-dimensional spot
quantification is recomputed in date-stamped output directories.

Key revisions:

- one random nuclear control is sampled independently from every QC-passing
  nucleus with a valid crop position, regardless of mTetR detection or state;
- the fixed-cell state thresholds are retained because fixation changes mTetR
  localization and intensity relative to live-cell imaging;
- the unused notebook-only method functions were removed; the imported,
  SHA-256-recorded production modules are the single source of truth.
"""
    )
    notebook["cells"][1] = code_cell(
        COMMON_IMPORT
        + """
import normal_multidataset_pipeline as batch

importlib.reload(batch)
BUNDLES = batch.build_dataset_bundles()
CONFIG = batch.aggregate_config()
DATASETS = batch.bundle_table(BUNDLES)

display(DATASETS)
assert len(DATASETS) == 8
assert DATASETS["fov_count"].sum() == 1300
assert DATASETS.groupby("condition_id").size().eq(2).all()
assert "revision_20260809" in str(CONFIG.output_root)
print({"aggregate_output": str(CONFIG.output_root)})
"""
    )
    notebook["cells"][4] = markdown_cell(
        """## 2. Common fixed-cell conditions and physical scaling

The Normal data have 65-nm XY sampling and 500-nm Z steps. The 3D mTetR
detection, 1.17-um crop radius, 0.39-um `r_mass` aperture, Trackpy feature
scale, and 0.39-um MCP distance rule follow the physical scales used in the
reference analysis.

The numerical Active/Inactive cutoffs remain `r_mTetR=1.15` and
`r_MCP=1.15`. These are fixed-cell classification thresholds: fixation alters
mTetR localization and intensity, so direct numerical equivalence to the
live-cell thresholds is not claimed. This distinction must be stated in the
Methods.
"""
    )
    notebook["cells"][5] = code_cell(
        """PARAMETERS = pd.DataFrame(
    [
        {
            "analysis_id": bundle.analysis_id,
            "LoG_sigma_zyx_px": str(
                bundle.analysis_config.bigfish_log_kernel_size_zyx_px
            ),
            "minimum_distance_zyx_px": str(
                bundle.analysis_config.bigfish_minimum_distance_zyx_px
            ),
            "crop_radius_px": bundle.analysis_config.crop_radius_px,
            "r_mass_radius_px": bundle.analysis_config.final_r_mass_radius_px,
            "r_mass_radius_um": bundle.analysis_config.r_mass_pad_um,
            "trackpy_diameter_px": bundle.analysis_config.final_trackpy_diameter_px,
            "trackpy_separation_px": bundle.analysis_config.final_trackpy_separation_px,
            "r_mTetR_threshold": bundle.analysis_config.fixed_mtetr_r_mass_threshold,
            "r_MCP_threshold": bundle.analysis_config.fixed_mcp_r_mass_threshold,
            "MCP_distance_um": bundle.analysis_config.mcp_near_distance_um,
            "threshold_context": "fixed-cell acquisition-specific",
        }
        for bundle in BUNDLES
    ]
)
display(PARAMETERS)
assert PARAMETERS["r_mass_radius_px"].eq(6).all()
assert np.isclose(PARAMETERS["r_mass_radius_um"], 0.39).all()
assert PARAMETERS["r_mTetR_threshold"].eq(1.15).all()
assert PARAMETERS["r_MCP_threshold"].eq(1.15).all()
assert PARAMETERS["MCP_distance_um"].eq(0.39).all()
"""
    )
    notebook["cells"][6] = markdown_cell(
        """## 3. Segmentation smoke test

The reviewed Normal Cellpose masks are reused without changing the fixed-cell
segmentation strategy. Only revision quantification and downstream output are
written to new date-stamped directories.
"""
    )
    notebook["cells"][11] = markdown_cell(
        """## 4. Revised 3D mTetR/MCP/SNAPtag quantification

For every QC-passing nuclear mask, a random position and central Z plane are
sampled independently of successful mTetR detection. The all-QC random crops
are saved separately in each FOV cache and provide the random mean used for
locus-centered baseline subtraction. Rank-1 locus crops and spot metrics are
otherwise generated by the same production path as the previous analysis.
"""
    )
    notebook["cells"][15] = markdown_cell(
        """## 5. Fixed-cell `r_mTetR` / `r_MCP` diagnostics

Diagnostic KDE and sensitivity figures are generated at dataset, condition,
and pooled levels. Classification remains fixed at `1.15 / 1.15`; these values
are specific to the fixed-cell mTetR localization and intensity distribution,
not a claim of direct equivalence to live-cell thresholds.
"""
    )
    notebook["cells"][17] = markdown_cell(
        """## 6. State calls and locus-centered aggregation

Each dataset uses the same fixed-cell state rule. The random mean is computed
from all QC-passing nuclei with valid crop positions, including nuclei without
a detected mTetR candidate. Active and Inactive locus images are then
baseline-subtracted and normalized with one shared positive pixel maximum per
channel, as in the reference Gao aggregation.
"""
    )
    notebook["cells"][21] = markdown_cell(
        """## Interpretation notes

1. The Normal data are fixed-cell controls, not a reanalysis of the live-cell
   movies.
2. Fixed-cell mTetR localization and intensity differ from live cells; the
   `1.15 / 1.15` thresholds and segmentation are therefore documented as
   acquisition-specific adaptations.
3. Random controls are independent of mTetR detection and state assignment.
4. Biological replicates remain the unit for reproducibility; pooled cell
   counts do not replace replicate-level interpretation.
"""
    )

    output = OUTPUT_DIR / (
        "normal_fixed_snapshot_all_datasets_gao3d_revision_20260809.ipynb"
    )
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return output


def build_sora() -> Path:
    source = SOURCE_DIR / "sora_fixed_snapshot_rep1_rep4_gao3d.ipynb"
    notebook = clean_notebook(json.loads(source.read_text(encoding="utf-8")))
    notebook["cells"][0] = markdown_cell(
        """# Fixed-cell SoRa snapshot analysis: revision 20260809

This clean revision copy reuses the reviewed SoRa fixed-cell segmentation
masks and recomputes 3D quantification for all 24 acquisitions.

Key revisions:

- the `r_mass` radius is 17 SoRa pixels (0.395 um), matching the 0.39-um
  physical aperture of the reference analysis;
- random controls are sampled from pre-dilation masks for all QC-passing
  nuclei, independently of mTetR detection and state;
- the fixed-cell `1.035 / 1.05` state thresholds are retained and documented
  as fixation/acquisition-specific rather than live-cell-equivalent;
- a separate sensitivity analysis excludes the two focus-QC-flagged Field 8
  acquisitions while retaining the all-acquisition analysis as primary.
"""
    )
    notebook["cells"][1] = code_cell(
        COMMON_IMPORT
        + """
import sora_rep1_rep4_pipeline as batch

importlib.reload(batch)
BUNDLES = batch.build_dataset_bundles()
AGGREGATE_CONFIG = batch.aggregate_config()
DATASETS = batch.bundle_table(BUNDLES)

display(DATASETS)
assert len(DATASETS) == 24
assert DATASETS["fov_count"].sum() == 8503
assert "revision_20260809" in str(AGGREGATE_CONFIG.output_root)
assert AGGREGATE_CONFIG.final_r_mass_radius_px == 17
print({"aggregate_output": str(AGGREGATE_CONFIG.output_root)})
"""
    )
    notebook["cells"][4] = markdown_cell(
        """## 2. Common fixed-cell and resolution-adapted conditions

The reviewed SoRa segmentation remains unchanged. Downstream 3D spot
detection retains the resolution-adapted LoG and Trackpy feature scales, while
the intensity `r_mass` aperture is changed to 17 pixels (0.395 um) to match the
reference 0.39-um physical radius. Crop size, MCP distance, and core/annulus
metrics remain defined in physical units.

The fixed-cell thresholds `r_mTetR=1.035` and `r_MCP=1.05` are retained because
fixation changes mTetR localization and intensity. The Methods should report
this as a fixed-cell calibration rather than direct reuse of live-cell numeric
cutoffs.
"""
    )
    notebook["cells"][5] = code_cell(
        """COMMON_PARAMETERS = {
    "segmentation": {
        "downsample": 4,
        "background_sigma_full_px": 30,
        "signal_sigma_full_px": 11,
        "cellpose_model": "nuclei",
        "diameter_4x_px": 110,
        "cellprob_threshold": 0.5,
        "flow_threshold": 0.4,
        "dilation_4x_px": 15,
        "random_control_mask": "pre-dilation selected nuclear mask",
    },
    "spot": {
        "median_kernel_zyx": (1, 3, 3),
        "log_kernel_zyx_px": (0.5, 4.0, 4.0),
        "minimum_distance_zyx_px": (1.0, 4.0, 4.0),
        "threshold": "central-plane nuclear mean / 20",
        "r_mass_radius_px": AGGREGATE_CONFIG.final_r_mass_radius_px,
        "r_mass_radius_um": AGGREGATE_CONFIG.r_mass_pad_um,
        "trackpy_diameter_px": 11,
        "trackpy_separation_px": 12,
        "crop_size_px": 101,
    },
    "state": {
        "r_mTetR": 1.035,
        "r_MCP": 1.05,
        "distance_um": 0.39,
        "threshold_context": "fixed-cell acquisition-specific",
    },
    "random_control": {
        "population": "all QC-passing nuclei with valid crop positions",
        "independent_of_mTetR_detection": True,
    },
}
display(pd.json_normalize(COMMON_PARAMETERS, sep=".").T)
assert AGGREGATE_CONFIG.final_r_mass_radius_px == 17
assert np.isclose(AGGREGATE_CONFIG.r_mass_pad_um, 0.39, atol=0.01)
"""
    )
    notebook["cells"][12] = markdown_cell(
        """## 5. Revised 3D mTetR/MCP/SNAPtag quantification

Quantification is recomputed for all 24 acquisitions in date-stamped outputs.
Each FOV cache contains rank-1 locus crops plus a separate random-control array
sampled from every QC-passing pre-dilation nuclear mask with a valid crop
position. Random sampling is independent of mTetR detection and state.

The SoRa `r_mass` radius is 17 px (0.395 um). The resolution-adapted Trackpy
diameter/separation and 3D Big-FISH detection settings are retained.
"""
    )
    notebook["cells"][9] = code_cell(
        """SEGMENTATION_SUMMARY_PATH = (
    AGGREGATE_CONFIG.output_root / "tables" / "segmentation_batch_summary.csv"
)
if SEGMENTATION_SUMMARY_PATH.is_file():
    SEGMENTATION_SUMMARY = pd.read_csv(SEGMENTATION_SUMMARY_PATH)
else:
    SEGMENTATION_SUMMARY = batch.run_all_segmentations(
        BUNDLES,
        max_workers=8,
    )
display(SEGMENTATION_SUMMARY)
"""
    )
    notebook["cells"][11] = code_cell(
        """SEGMENTATION_CONTACT = (
    AGGREGATE_CONFIG.output_root
    / "figures"
    / "segmentation_representatives_24_acquisitions.png"
)
if not SEGMENTATION_CONTACT.is_file():
    SEGMENTATION_CONTACT = batch.representative_segmentation_contact(BUNDLES)
display(DisplayImage(filename=str(SEGMENTATION_CONTACT)))
"""
    )
    notebook["cells"][13] = code_cell(
        """QUANTIFICATION_SUMMARY_PATH = (
    AGGREGATE_CONFIG.output_root / "tables" / "quantification_batch_summary.csv"
)
if QUANTIFICATION_SUMMARY_PATH.is_file():
    QUANTIFICATION_SUMMARY = pd.read_csv(QUANTIFICATION_SUMMARY_PATH)
else:
    QUANTIFICATION_SUMMARY = batch.run_all_quantifications(
        BUNDLES,
        workers=24,
        chunk_size=10,
    )
display(QUANTIFICATION_SUMMARY)
"""
    )
    notebook["cells"][15] = code_cell(
        """SPOT_CONTACT = (
    AGGREGATE_CONFIG.output_root
    / "figures"
    / "spot_detection_representatives_24_acquisitions.png"
)
if not SPOT_CONTACT.is_file():
    SPOT_CONTACT = batch.representative_spot_contact(BUNDLES)
display(DisplayImage(filename=str(SPOT_CONTACT)))
"""
    )
    notebook["cells"][17] = code_cell(
        """THRESHOLD_TABLES = AGGREGATE_CONFIG.output_root / "tables"
THRESHOLD_FIGURES = AGGREGATE_CONFIG.output_root / "figures"
THRESHOLD_REQUIRED = [
    THRESHOLD_TABLES / "threshold_diagnostics_by_acquisition.csv",
    THRESHOLD_TABLES / "threshold_diagnostics_by_biological_replicate.csv",
    THRESHOLD_TABLES / "threshold_diagnostics_by_condition.csv",
    AGGREGATE_CONFIG.output_root / "threshold_decision.json",
]
if all(path.is_file() for path in THRESHOLD_REQUIRED):
    THRESHOLD_RESULTS = {
        "acquisition_diagnostics": pd.read_csv(THRESHOLD_REQUIRED[0]),
        "biological_replicate_diagnostics": pd.read_csv(THRESHOLD_REQUIRED[1]),
        "condition_diagnostics": pd.read_csv(THRESHOLD_REQUIRED[2]),
        "pooled_decision": json.loads(THRESHOLD_REQUIRED[3].read_text()),
        "pooled_figure": THRESHOLD_FIGURES / "threshold_calibration_four_panel.png",
        "acquisition_contact_sheet": THRESHOLD_FIGURES / "threshold_calibration_by_acquisition.png",
        "biological_replicate_contact_sheet": THRESHOLD_FIGURES / "threshold_calibration_by_biological_replicate.png",
        "condition_contact_sheet": THRESHOLD_FIGURES / "threshold_calibration_by_condition.png",
        "pooled_candidate_count": len(
            pd.read_csv(
                THRESHOLD_TABLES / "all_acquisitions_spot_candidates.csv",
                usecols=[0],
            )
        ),
    }
else:
    THRESHOLD_RESULTS = batch.run_threshold_calibrations(BUNDLES)
display(THRESHOLD_RESULTS["acquisition_diagnostics"])
display(THRESHOLD_RESULTS["biological_replicate_diagnostics"])
display(THRESHOLD_RESULTS["condition_diagnostics"])
display(DisplayImage(filename=str(THRESHOLD_RESULTS["acquisition_contact_sheet"])))
display(DisplayImage(filename=str(THRESHOLD_RESULTS["biological_replicate_contact_sheet"])))
display(DisplayImage(filename=str(THRESHOLD_RESULTS["condition_contact_sheet"])))
"""
    )
    notebook["cells"][16] = markdown_cell(
        """## 6. Fixed-cell threshold diagnostics

Diagnostic distributions are saved at acquisition, biological-replicate,
condition, and pooled levels. State calls remain fixed at
`r_mTetR=1.035`, `r_MCP=1.05`, and MCP distance <=0.39 um. These numerical
values are documented as fixed-cell thresholds because fixation changes mTetR
localization and intensity.
"""
    )
    notebook["cells"][20] = markdown_cell(
        """## 8. State calls and Gao-style locus aggregation

The all-acquisition analysis remains primary. Random-mean subtraction uses all
QC-passing nuclei with valid random crops rather than only nuclei with a
detected rank-1 mTetR locus. Biological samples are formed only after
acquisition-level state calling and QC.
"""
    )
    notebook["cells"][21] = code_cell(
        """STATE_TABLES = AGGREGATE_CONFIG.output_root / "tables"
STATE_FIGURES = AGGREGATE_CONFIG.output_root / "figures"
STATE_REQUIRED = [
    STATE_TABLES / "state_summary_by_acquisition.csv",
    STATE_TABLES / "state_summary_by_biological_replicate.csv",
    STATE_TABLES / "state_summary_by_condition.csv",
    STATE_TABLES / "state_summary_all_rep1_rep4_pooled.csv",
    STATE_TABLES / "biological_sample_pooling_membership.csv",
    STATE_FIGURES / "gao_fig1c_by_biological_replicate.png",
]
if all(path.is_file() for path in STATE_REQUIRED):
    biological_summary = pd.read_csv(STATE_REQUIRED[1])
    STATE_RESULTS = {
        "acquisition_summary": pd.read_csv(STATE_REQUIRED[0]),
        "biological_replicate_summary": biological_summary,
        "paired_field_summary": biological_summary.loc[
            biological_summary["biological_replicate"].isin(["rep3", "rep4"])
        ].reset_index(drop=True),
        "pooling_membership": pd.read_csv(STATE_REQUIRED[4]),
        "condition_summary": pd.read_csv(STATE_REQUIRED[2]),
        "pooled_summary": pd.read_csv(STATE_REQUIRED[3]),
        "state_counts_figure": STATE_FIGURES / "state_counts_by_acquisition.png",
        "biological_replicate_figure": STATE_FIGURES / "active_fraction_by_biological_replicate.png",
        "paired_field_figure": STATE_FIGURES / "paired_field_pooled_results_rep3_rep4.png",
        "gao_contact_sheet": STATE_REQUIRED[5],
    }
else:
    STATE_RESULTS = batch.run_states_and_locus_summaries(BUNDLES)
display(STATE_RESULTS["acquisition_summary"])
display(STATE_RESULTS["biological_replicate_summary"])
display(STATE_RESULTS["paired_field_summary"])
display(STATE_RESULTS["pooling_membership"])
display(STATE_RESULTS["condition_summary"])
display(STATE_RESULTS["pooled_summary"])
display(DisplayImage(filename=str(STATE_RESULTS["state_counts_figure"])))
display(DisplayImage(filename=str(STATE_RESULTS["biological_replicate_figure"])))
display(DisplayImage(filename=str(STATE_RESULTS["paired_field_figure"])))
if STATE_RESULTS["gao_contact_sheet"] is not None:
    display(DisplayImage(filename=str(STATE_RESULTS["gao_contact_sheet"])))
"""
    )

    focus_markdown = markdown_cell(
        """## 9. Focus-QC sensitivity analysis

Two acquisition-level QC notes are retained: ChamberC Field 8 was acquired
substantially above the intended plane and ChamberD Field 8 was designated
focus-out. The primary analysis includes all acquisitions. This separate
sensitivity analysis excludes only those two acquisitions and recomputes the
affected Nanog-Ser5ph Rep3/Rep4 locus summaries.
"""
    )
    focus_code = code_cell(
        """FOCUS_ROOT = AGGREGATE_CONFIG.output_root / "focus_qc_sensitivity"
FOCUS_REQUIRED = [
    FOCUS_ROOT / "tables" / "excluded_acquisitions.csv",
    FOCUS_ROOT / "tables" / "focus_qc_sensitivity_comparison.csv",
    FOCUS_ROOT / "figures" / "focus_qc_passing_gao_summaries.png",
]
if all(path.is_file() for path in FOCUS_REQUIRED):
    FOCUS_QC = {
        "flagged_acquisitions": pd.read_csv(FOCUS_REQUIRED[0]),
        "comparison": pd.read_csv(FOCUS_REQUIRED[1]),
        "contact_sheet": FOCUS_REQUIRED[2],
    }
else:
    FOCUS_QC = batch.run_focus_qc_sensitivity(BUNDLES)
display(FOCUS_QC["flagged_acquisitions"])
display(FOCUS_QC["comparison"])
display(DisplayImage(filename=str(FOCUS_QC["contact_sheet"])))
"""
    )
    notebook["cells"][22:22] = [focus_markdown, focus_code]
    notebook["cells"][24] = markdown_cell(
        """## 10. Completeness validation

Validate all 24 acquisition outputs, 8,503 FOV caches, all-QC random-control
arrays, fixed-cell state calls, biological-replicate summaries, and the
focus-QC sensitivity outputs.
"""
    )

    notebook["cells"][-1] = markdown_cell(
        """## Interpretation notes

- The all-acquisition result is primary; the focus-QC exclusion is explicitly
  labeled as sensitivity analysis.
- The `r_mass` aperture is physically matched to approximately 0.39 um, while
  the SoRa LoG and Trackpy feature scales remain resolution-adapted.
- Fixed-cell thresholds and segmentation need not numerically match live-cell
  analysis because fixation changes mTetR localization and intensity; this is
  documented in the Methods.
- Higher-resolution persistence of Active-associated locus-centered signal
  supports localization within the SoRa optical-resolution volume, but does
  not establish molecular colocalization or a sub-resolution distance.
"""
    )

    output = OUTPUT_DIR / (
        "sora_fixed_snapshot_rep1_rep4_gao3d_revision_20260809.ipynb"
    )
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    for path in (build_normal(), build_sora()):
        print(path)
