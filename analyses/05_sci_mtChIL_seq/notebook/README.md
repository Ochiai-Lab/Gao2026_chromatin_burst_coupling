# sci-mtChIL-seq reanalysis (NIH3T3)

This directory contains a publication-aligned reanalysis of published sci-mtChIL-seq data in NIH3T3 cells, where elongating RNAPII (Ser2ph) is profiled together with a paired histone modification in the same single cells. The analysis supports the manuscript’s locus-resolved interpretation using an orthogonal, fixed-cell approach (Figure 2c–f).

The main output is the distribution of gene-wise log2(Active/Inactive) CPT ratios across non-overlapping 500-gene activity-frequency bins.

## What this analysis does

Given a processed gene × cell summary table:

1. Defines transcriptional state per gene per cell:
   Active if RNAPII Ser2ph count > 0 in a promoter-proximal window (TSS −1 kb to +5 kb); otherwise Inactive.
2. For each gene and paired histone mark, computes mean CPT in Active cells and mean CPT in Inactive cells.
3. Ranks genes by activity frequency:
   f_active = N_active / (N_active + N_inactive)
4. Splits genes into consecutive non-overlapping rank bins (default: 500 genes per bin).
5. Computes per-gene log2 ratios within each bin:
   log2( (mean_CPT_active + 0.01) / (mean_CPT_inactive + 0.01) )
6. Tests deviation from 0 (ratio 1) using a two-sided one-sample Wilcoxon signed-rank test and applies BH-FDR correction across all (target × bin) tests.

This matches the Methods description in the manuscript.

## Input

Required file:
- avgCPT_rnapU1D5_pairU5D5.csv
  Example path: data/05_sci_mtChIL_seq/avgCPT_rnapU1D5_pairU5D5.csv

Expected columns (names may vary; see the notebook’s “column mapping” cell):
- gene
- target (paired histone modification; e.g., H3K27ac, H3K4me3, H3K27me3)
- avg_posi: mean paired-target CPT in Active cells
- avg_nega: mean paired-target CPT in Inactive cells
- nCell_posi: number of Active cells
- nCell_nega: number of Inactive cells

## Quickstart

1. Place the input CSV under data/05_sci_mtChIL_seq/

2. Run the notebook

Open and run:
  sci_mtChIL_seq_reanalysis.ipynb


## Outputs

By default, outputs are written to:
  outputs/

Typical outputs include:
- Figure PDF/PNG for log2(Active/Inactive) CPT ratios across 500-gene bins (for manuscript Fig. 2f)
- Supplementary figure for N_active cells per gene across bins (for Supplementary Fig. 6b)
- A stats table with p-values, BH-FDR q-values, and significance labels

## Method details (as implemented)

- Activity frequency ranking uses f_active computed from the same processed table.
- Binning uses non-overlapping consecutive rank bins (1–500, 501–1000, …).
- Ratio pseudocount: 0.01 is added to both mean CPT values.
- Statistics: two-sided one-sample Wilcoxon signed-rank test on log2 ratios vs 0, with BH-FDR correction across all target × bin tests.

## Notes and troubleshooting

- Sparse signals at low activity ranks: lower bins tend to have fewer Active cells per gene, which can reduce statistical power and attenuate apparent coupling. The “N_active cells per gene” supplementary plot is included to document this sampling limitation.
- If you regenerate the processed table from raw sequencing, ensure that the window definitions (RNAPII: −1 kb to +5 kb; paired target: ±5 kb) match the manuscript.

## Citation

If you use this code, please cite:
- Fujii et al., Nat Commun, 2025, https://www.nature.com/articles/s41467-025-67016-9
- the Gao et al. manuscript (this repository)
