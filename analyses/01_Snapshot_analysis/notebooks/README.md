# 01_Snapshot_analysis

This folder contains the notebook to reproduce the analysis corresponding to the manuscript section:
- Snapshot image processing and analysis of STREAMING-tag cells

Associated manuscript:
- Gao et al., Minute-scale coupling of chromatin marks and transcriptional bursts (2026, in preparation)


## What this analysis does

This module processes snapshot (fixed-cell) microscopy images of STREAMING-tag cells and performs:
- segmentation (Cellpose-based workflow, if enabled in the notebook)
- per-cell intensity quantification
- locus-centered measurements (spot- or locus-centered summaries, depending on the notebook implementation)
- figure-oriented summaries for downstream interpretation

State labels:
- Active = ON
- Inactive = OFF

Some tables may use ON/OFF; we treat them as equivalent to Active/Inactive.


## Notebooks

Run the notebook in this folder:

- notebooks/01_Snapshot_analysis.ipynb


## Data layout

Place snapshot data under:

````

data/01_Snapshot_analysis/
nd2/
*.nd2

```

If your snapshot data are not ND2, adapt the notebook input section accordingly.
The notebook should be written to avoid absolute paths and to use paths relative to the repository root.


## Outputs

Generated files should be written under:

- analyses/01_Snapshot_analysis/outputs/

This directory is gitignored.
Do not commit outputs.


## Key parameters to check (in the notebook)

Before running, verify:
- input data directory (expected under data/01_Snapshot_analysis/)
- channel indices / channel names used for quantification
- segmentation settings (if Cellpose is used)
- thresholds used for filtering cells and for any state/spot classification
- any randomness/seed settings if applicable


## Public-release notes

Recommended steps before committing:
- Clear notebook outputs to avoid embedding local paths:
  - Kernel -> Restart & Clear Output -> Save
- Ensure STREAMING-tag and mTetR naming is consistent in markdown cells
