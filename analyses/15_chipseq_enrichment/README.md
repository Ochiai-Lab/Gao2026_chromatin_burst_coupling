# 15_chipseq_enrichment

Genome-bin and gene-window enrichment analysis for Supplementary Fig. 5e.

- Upstream notebook: `notebooks/enrichment_calculation_v2.ipynb`.
- Final-table figure script: `scripts/replot_fig_s5e_from_tables.py`.
- Tables: `15_chipseq_enrichment/` below `GAO2026_DATA_ROOT`.
- Version 1.0.2 corrects the four TSS-centered windows. Version 1.0.1
  incorrectly used the boundaries of 1-Mb browser display regions as TSSs.
- TSSs now come from NCBI Mus musculus Updated Annotation Release
  108.20200622 (GRCm38.p6, GCF_000001635.26), selecting RefSeq Select
  transcripts. The table and local GFF excerpt record their provenance.
- TSSs are 0-based first transcribed bases; windows are 0-based half-open.
  For reverse-strand Usp5, TSS = transcript end (1-based) minus one.
- Window size, the 18 bigWig tracks, genome-bin means, finite-value filtering,
  joint population standardization (ddof=0) and plotting format are unchanged.
- `fig_s5e_v1_v2_comparison.csv` documents both windows and the numerical changes.
- Full gene-window extraction requires the public bigWigs in the accession
  map and `BW_TABLE_PATH`, or a `bw-path.csv` below
  `GAO2026_EXTERNAL_ROOT/chipseq_enrichment/`. This external path table is
  not included in the archive. Final-table plotting needs no raw bigWigs.
- The original genome browser panels 5a-d are not recomputed. Their track
  labels indicate RefSeq Curated; the exact historical annotation download
  release was not retained. The version-2 TSS annotation is explicitly pinned.
- Output files are saved below `GAO2026_OUTPUT_ROOT/15_chipseq_enrichment/`.
