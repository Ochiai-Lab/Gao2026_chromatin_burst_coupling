# 02_Cross_correlation

This folder contains notebooks to reproduce the analysis corresponding to the manuscript section:
- Time-lapse image preprocessing and analysis

Associated manuscript:
- Gao et al., Minute-scale coupling of chromatin marks and transcriptional bursts (2026, in preparation)


## What this analysis does

This module analyzes time-lapse movies and produces cross-correlation analyses between:
- SNAPtag (chromatin mark reporter)
- MCP channel (nascent transcription reporter)

The locus is anchored using:
- mTetR (genomic anchor; locus marker)

State labels:
- Active = ON
- Inactive = OFF

Some intermediate CSV files may use ON/OFF. We treat ON and Active as equivalent, and OFF and Inactive as equivalent.


## Notebooks and execution order

This analysis is a two-step pipeline.

1) Preprocess:
- notebooks/1_Timelapse_preprocess.ipynb
  - reads raw movies
  - performs preprocessing, detection/tracking steps as implemented
  - exports intermediate tables (CSV) for downstream analysis

2) Analysis:
- notebooks/2_Timelapse_analysis.ipynb
  - reads the preprocessed outputs (CSVs)
  - computes cross-correlations and summary statistics
  - exports figures/tables for the manuscript


## Data layout

Place time-lapse data under:

```

data/02_Cross_correlation/
(your movies here)

```

If you use subfolders per condition, document it and update the notebook input configuration accordingly.


## Outputs

Generated files should be written under:

- analyses/02_Cross_correlation/outputs/

This directory is gitignored.
Do not commit outputs.


## Key parameters to check (in the notebooks)

Preprocess notebook:
- input directory
- channel mapping (SNAP, MCP, mTetR)
- detection thresholds / filtering rules
- tracking/linking parameters (max displacement, minimum track length)
- any manual steps (if a GUI viewer is used, document how to skip in headless runs)

Analysis notebook:
- which CSVs are read
- lag range for correlation
- bootstrap/permutation settings (if applicable)
- figure export paths


## Public-release notes

- Clear notebook outputs before committing:
  - Kernel -> Restart & Clear Output -> Save
- Keep naming consistent:
  - STREAMING-tag (not STtag)
  - mTetR (genomic anchor)
  - Active/Inactive with explicit equivalence to ON/OFF
