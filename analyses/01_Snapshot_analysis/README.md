# 01_Snapshot_analysis

This module is part of the Gao2026 `v0.9.0-rc1` release candidate.

- `notebooks/`: output-stripped analysis notebooks.
- `scripts/`: helper code where applicable.
- Inputs are selected through `GAO2026_DATA_ROOT` and, for raw-image stages only, `GAO2026_RAW_ROOT`.
- Generated outputs must be written under `GAO2026_OUTPUT_ROOT` and are not tracked by Git.

## Public replicate convention

- Public `rep1` is the acquisition used for the main representative figure panel.
- Other acquisitions for the same target/factor are numbered `rep2`, `rep3`, and so on.
- `pkl_file_position_original.xlsx` records `primary_figure_dataset`,
  `include_in_state_dependence_summary`, and `source_identity` so that panel
  provenance and aggregate-analysis inclusion remain explicit.
- The Sox2 H3K9ac acquisition used in Main Fig. 1c is `rep1`. The 250909
  acquisition used in the Supplementary Fig. 4c correlation matrix is `rep2`.
