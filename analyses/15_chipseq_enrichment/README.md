# 15_chipseq_enrichment

Genome-bin and gene-window ChIP-seq enrichment analysis used for Supplementary Fig. 5e.

- Upstream analysis notebook: `notebooks/enrichment_calculation_v1.ipynb`
- Final-table figure script: `scripts/replot_fig_s5e_from_tables.py`
- Deposited final tables: `15_chipseq_enrichment/` (the earlier
  `revision/15_chipseq_enrichment/` layout is also accepted).
- The deposited tables contain the accession map, per-gene-window values, genome-wide 0.5-Mb-bin values, target summaries, and mm10 chromosome sizes used in the figure.
- Full recomputation requires the public bigWig tracks listed in `fig_s5e_accession_map.csv`. Place a corresponding `bw-path.csv` under `GAO2026_EXTERNAL_ROOT/chipseq_enrichment/`, or set `BW_TABLE_PATH` directly.
- Generated figures are written below `GAO2026_OUTPUT_ROOT/15_chipseq_enrichment/` and are excluded from Git.
