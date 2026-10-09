# 08_SoRa

Fixed-cell SoRa analysis used for the higher-resolution validation.

- The analysis constants match the manuscript Methods: median `3 x 3` filter,
  six-pixel `r_mass = 0.139 um`, mTetR/MCP ratio thresholds `1.035/1.05`, and
  Active-state distance cutoff `0.39 um`.
- `scripts/replot_quantitative_panels_from_tables.py` regenerates the radial
  profiles and Active/Inactive quantitative summaries from deposited tables.
- The publication replot uses only the displayed experiment, designated
  `rep1` in this public package. All four displayed N/P combinations and
  the article Source Data use this same experiment.
- Full spot detection, focus-QC sensitivity, and image extraction require the
  controlled raw-image/intermediate data.
- The four `gao_fig1c_scaled_images.npz` files used for the representative
  image panel are absent, so that panel cannot be regenerated from the current
  deposit. The final plotting notebook also depended on kernel-resident helper
  definitions and should not be treated as a self-contained raw-data pipeline.
