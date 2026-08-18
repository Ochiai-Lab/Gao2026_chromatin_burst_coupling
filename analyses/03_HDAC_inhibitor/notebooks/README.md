# 03_HDAC_inhibitor

This folder contains the notebook to reproduce the analysis corresponding to the manuscript section:
- Analysis of HDAC inhibitor time-lapse data

Associated manuscript:
- Gao et al., Minute-scale coupling of chromatin marks and transcriptional bursts (2026, in preparation)



## What this analysis does

This module analyzes HDAC inhibitor time-lapse movies using a custom Python pipeline.

High-level overview (as described in the manuscript):
- input: two-channel TIFF movies (time x channel x y x x) and optional per-frame cell label images
  - MCP channel: nascent transcription reporter
  - mTetR channel: locus marker (genomic anchor)
- robust per-frame normalization (median and MAD) to facilitate spot detection
- spot detection using Big-FISH with channel-specific normalized thresholds
  (e.g., THRESH_MCP = 1.25; THRESH_mTetR = 1.10)
- rescue of weak mTetR frames by local maximum search within a 6-pixel radius
  using a lowered threshold (0.85 × THRESH_mTetR = 0.935), before fallback inference
- optional assignment of spots to cells using label masks (or treat the field as a single region)
- locus trajectory construction (nearest-neighbor linking with a maximum displacement)
- transcriptional state calling at each frame:
  - Active (ON) if MCP is detected within a proximity threshold from the locus position
  - Inactive (OFF) otherwise
  - isolated one-frame activations can be relabeled as Inactive to suppress spurious calls
- extraction of contiguous Active and Inactive runs (dwell times)
- censoring annotation for runs overlapping movie start/end
- aggregation across movies/cells by condition and locus
- outputs:
  - representative rasters
  - duty cycles per cell
  - dwell-time distributions and Kaplan-Meier analysis

State labels:
- Active = ON
- Inactive = OFF

Some intermediate CSV files may use ON/OFF; we treat ON and Active as equivalent, and OFF and Inactive as equivalent.


## Notebooks

Run the notebook in this folder:

- notebooks/inhibitor_response_analysis.ipynb


## Data layout

Place data under:

```

data/03_HDAC_inhibitor/
(movies)
(optional label masks)

```

If label masks are used, describe the naming rule in the notebook and keep it consistent.
Avoid hard-coded absolute paths; use paths relative to repository root.


## Outputs

Generated files should be written under:

- analyses/03_HDAC_inhibitor/outputs/

This directory is gitignored.
Do not commit outputs.


## Key parameters to check (in the notebook)

- frame interval (minutes) and pixel size (nm)
- normalized threshold for spot detection
- linking constraints (max displacement, minimum length)
- proximity threshold for calling Active (ON)
- censoring rules and whether left-censored runs are excluded from the primary analysis
- condition mapping (how movies are grouped into inhibitor conditions)


## Public-release notes

- Clear notebook outputs before committing:
  - Kernel -> Restart & Clear Output -> Save
- Keep naming consistent in markdown and comments:
  - STREAMING-tag (not STtag)
  - mTetR (genomic anchor)
  - Active/Inactive with explicit equivalence to ON/OFF

