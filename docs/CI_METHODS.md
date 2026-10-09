# Article Source Data comparison-effect intervals

`scripts/recalculate_source_data_ci.py` recreates the additional comparison
intervals in the article Source Data. It reads the separately supplied
`--source-data Source_Data.xlsx` and the extracted versioned data directory
specified by `--data-root` or `GAO2026_DATA_ROOT`. It never writes to the
workbook or replaces its P, q, U or W values.

The workbook block, Excel row, comparison identifiers and output column are
recorded in `comparison_effect_intervals.csv`. The script verifies every
recalculated estimate/bound against the saved value, and also checks original
sample counts, medians and U statistics where those define the comparison.
`validation.json` records input SHA-256 hashes, software versions, coverage
and the unchanged workbook hash. `numeric_checks.json` retains individual
numeric comparisons. A discrepancy raises an exception; missing intervals
are not silently filled or inferred from P values.

## Inputs and sampling units

| Article block | Observation input | Sampling unit and comparison |
| --- | --- | --- |
| Fig. 1, 3, 4; S4/S10 radial comparisons | `01_Snapshot_analysis/pkl/<dataset_id>.pkl` and workbook comparison rows | Locus instances, Active minus Inactive, separately for each dataset, channel and radius |
| S3e/S9c expression-group effects and AUC | `01_Snapshot_analysis/ci_inputs/mean_intensity_long_from_csv.csv` | Original nuclear-intensity observations within gene, probe, replicate, well and channel |
| S4 additional nuclear-intensity summaries | `01_Snapshot_analysis/ci_inputs/snapshot_cell_measurements.csv` | Original nuclear mean intensities, Active minus Inactive, separate from the S4b central-intensity violin metric |
| S4b central-intensity effects | Workbook individual central 3x3 SNAP means and their plotted log10 values | Displayed locus instances only, Active minus Inactive, using each displayed replicate including corrected Sox2/H3K4me3 replicate 1 |
| S4c Spearman correlations | `01_Snapshot_analysis/ci_inputs/snapshot_snap_mcp_center_observations.csv` | Paired locus-centered SNAP and relative MCP observations within the displayed dataset |
| S4f SoRa | Workbook individual core-minus-annulus values, traceable to `08_SoRa/biological_replicate_results/*_rep1/tables/gao_fig1c_core_annulus_values.csv` | Quality-filtered locus instances in the displayed experiment, public replicate 1, Active minus Inactive |
| Fig. 2b/S6a | Workbook gene-wise precomputed paired Active/Inactive delta-z values | Genes, within marker and activity-rank quartile, or all genes for overall summaries |
| Fig. 2f/S6b | Workbook gene-wise log2 Active/Inactive ratios | Genes, within target and bin; exponentiated bounds give the ratio interval |
| S12c-d | `10_HDAC_time_windows/Fig5_<gene>_duty_cycle_by_window_cell.csv` | Cells within each window and treatment; treatment minus DMSO |
| S13a-f | Workbook selected pooled cell measurements, traceable to `14_acute_inhibitor/fig_s13_signal_per_cell.csv` | Included cells after experiment-specific normalization and correction; group_2 minus group_1 separately within panel, metric and time point |
| Fig. 5e-f | `12_Fig5_integrated/<gene>/tetr_mcp_hybrid_analysis/aggregated_perframe.csv` plus workbook per-cell duty cycles | Cells, treatment minus DMSO; duty cycles are checked against the final frame calls |

The workbook is an explicit public input, not a private work-table JSON.
It supplies the article's selected comparisons, measurements where stated
above, and saved reference intervals. Legacy `data/revision/` source labels
in the workbook map to the unified module folders in this release. The
published SoRa experiment is designated public replicate 1. The other experiment and mixed
tables are not in this archive. Binary fields such as `on_off` and the PKL
keys `on_images`/`off_images` remain compatibility identifiers; CSV state
values are Active/Inactive. Only load PKLs from a trusted deposit.

## Methods and deterministic seeds

- Median effects use 9,999 independent empirical-bootstrap draws. For two
  groups the median draws are generated independently, then subtracted in
  the direction stated above. Bounds are the 2.5th and 97.5th percentiles.
  The exact bootstrap-median distribution is sampled using uniform order
  statistics (beta draws); even sample sizes retain the joint middle-order
  statistics. This is an efficient implementation of the original iid
  observation bootstrap, including ties, not a different sampling scheme.
- SNAP expression-group median effects use `log2(intensity + 1e-6)`.
  S4b uses the saved `log10(max(central SNAP mean, 1e-12))` values. Other
  transforms and baseline/photobleaching corrections are taken from the
  original included observations; no extra transformation is introduced.
- The seed is the little-endian integer from the first four bytes of
  SHA-256 of the UTF-8 `str(comparison_key)`. Tuple keys, marker/channel
  order and draw order match the original implementation. Fig. 2b's
  stable block seed tokens 87, 15904 and 131 are retained intentionally.
- Spearman intervals use 1,999 paired percentile bootstrap resamples with
  SciPy, batch 32. Within each resample both variables receive the same
  indices. Ranks are recalculated, including average ranks for ties.
- Radial mean differences use the Welch standard error, Satterthwaite
  degrees of freedom and the two-sided t interval. Original float32
  background subtraction and shared per-dataset peak scaling are preserved.
- Signed AUC uses tie-aware placements and placement-variance normal
  intervals bounded to [0, 1]. Best-direction AUC intervals transform the
  signed interval with `max(AUC, 1-AUC)`; their lower bound is 0.5 when the
  signed interval contains 0.5. Fold-change bounds are `2**log2_bound`.

These are pointwise intervals conditional on the analyzed imaging series
or gene observations, not confidence intervals across independent biological
experiments. They do not account for between-experiment variation, model
frames as independent cells, or pool different radii/windows/time points.
No simultaneous or multiplicity-adjusted interval is implied.

Use the repository's pinned `environment.yml` for exact regeneration.
The focused CI dependencies are Python, NumPy, pandas and SciPy; workbook
parsing uses only Python's standard library. `--self-test` compares the
bootstrap-median distribution with explicit resampling. `--sections`
allows a selected analysis family to be rerun without changing its seeds.

SoRa comparison bootstrap seeds are explicitly stored as fixed integers in the script. They are invariant to public replicate-label changes; all other seed keys retain the documented SHA-256 rule.
