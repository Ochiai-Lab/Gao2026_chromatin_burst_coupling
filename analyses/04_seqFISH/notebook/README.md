# seq-DNA/RNA/IF-FISH reanalysis (seqFISH)

This directory contains a minimal, publication-aligned reanalysis of published multimodal seq-DNA/RNA/IF-FISH data in mESCs, used to support allele-resolved enrichment of chromatin features and transcription-related factors (manuscript Figure 2a–b and Supplementary Fig. 6a).

The analysis reproduces the gene-wise Δz summary stratified by gene activity (Active-allele ratio rank quartiles).

## What this analysis does

1. Builds a unique allele key (FOV × cell × allele) to match DNA/IF allele loci with seq-RNA-FISH transcription calls.
2. Labels each allele-locus instance as Active if the same allele key is present in the seq-RNA-FISH table, otherwise Inactive.
3. Standardizes immunofluorescence (IF) intensities using gene-wise z-scores (per marker, across all alleles of the same gene).
4. Computes per-gene Δz for each IF marker:
   Δz = median(z | Active) − median(z | Inactive)
5. Ranks genes by Active-allele ratio and partitions them into rank quartiles.
6. Visualizes Δz distributions and tests whether Δz values deviate from 0 using two-sided Wilcoxon signed-rank tests with BH-FDR correction (q-values).

This matches the Methods description in the manuscript.

## Inputs

Only the following two input files are required.

- DNA/IF spot table (allele-level)
  - Path (example): data/04_seqFISH/Spot_coordinates_with_IF_intensity.csv
  - Expected content: one row per allele-locus instance with identifiers (FOV, cell, allele, gene) and IF intensities for each marker.
- Transcription state summary (RNA on/off table)
  - Path (example): data/04_seqFISH/Transcriptio_state_summary.csv
  - Expected content: allele keys (FOV, cell, allele, gene) for alleles called transcriptionally Active.

Note: the exact column names can vary between releases. The notebook contains a small “column mapping” cell; if your CSV headers differ, edit only that mapping rather than downstream code.

## Quickstart

1. Place the two input CSV files under data/04_seqFISH/

2. Run the notebook

Open and run:
  seqFISH_reanalysis.ipynb


## Outputs

By default, outputs are written to:
  outputs/

Typical outputs include:
- Figure PDF/PNG for Δz by activity quartiles (for manuscript Fig. 2b)
- Supplementary figure for Active-allele counts per gene (if generated; for Supplementary Fig. 6a)
- A “long” table with per-gene, per-marker Δz values
- A stats table with p-values, BH-FDR q-values, and significance labels

## Method details (as implemented)

- Active allele definition: allele key appears in the RNA on/off table.
- Gene-wise z-score: for each gene × marker, compute mean and s.d. across all alleles of that gene (Active and Inactive pooled), then z = (I − μ_g)/σ_g; if σ_g = 0, set z = 0.
- Gene inclusion: at least one Active allele and both Active and Inactive alleles available for Δz estimation.
- Quartiles: genes are ranked by Active-allele ratio and split into four consecutive rank quartiles.
- Statistics: two-sided Wilcoxon signed-rank test on per-gene Δz vs 0, with BH-FDR correction across all marker × quartile tests. Stars reflect q-values.

## Notes and troubleshooting

- Low-activity genes: lower quartiles can have fewer Active alleles, which reduces power and can attenuate apparent state bias. We therefore provide Supplementary Fig. 6a (Active-allele counts per gene) to document the underlying sample size per quartile.
- Reproducibility: random seeds are fixed where jitter is used for plotting.

## Citation

If you use this code, please cite:
- Ohishi et al., Sci Adv, 2024, https://www.science.org/doi/10.1126/sciadv.adn0020
- the Gao et al. manuscript (this repository)
