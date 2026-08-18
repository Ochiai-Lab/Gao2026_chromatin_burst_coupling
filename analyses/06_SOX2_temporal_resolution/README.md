# 06_SOX2_temporal_resolution

SOX2-MCP cross-correlation and LLI analyses at 2-min, 1-min, and 30-s sampling
intervals used in Supplementary Fig. 11.

- `notebooks/2_Timelapse_replot_from_deposited_tables.ipynb` regenerates the
  final 1-min, 30-s, and LLI panels from deposited final tables.
- The per-cell trace CSV files required to repeat filtering, cross-correlation,
  bootstrapping, and permutation testing are not included in the current data
  package.
- Accordingly, this module currently supports an exact final-table replot, not
  an independent full recomputation from traces.
