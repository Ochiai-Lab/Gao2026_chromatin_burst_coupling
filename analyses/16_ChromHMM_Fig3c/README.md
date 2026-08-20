# Fig. 3c ChromHMM analysis

This module preserves the exact analysis logic supplied for the current Fig. 3c.

## Deposited inputs

- `16_ChromHMM_Fig3c/emissions_12.txt`: ChromHMM emission probabilities.
- `16_ChromHMM_Fig3c/state_beds/*.bed`: genomic intervals for the 12 displayed states.
- `16_ChromHMM_Fig3c/chromState_signal_matrix.tsv`: length-weighted mean bigWig signal for each factor and state.
- `16_ChromHMM_Fig3c/fig3c_*_long.csv`: exact values plotted in the two heatmaps.
- `16_ChromHMM_Fig3c/fig3c_bigwig_accession_map.csv`: public-input accession and display-label mapping.

## Fast reproduction from deposited matrices

```bash
python analyses/16_ChromHMM_Fig3c/scripts/replot_fig3c.py \
  --data-root "$GAO2026_DATA_ROOT" \
  --output-root "$GAO2026_OUTPUT_ROOT"
```

This produces `16_ChromHMM_Fig3c/Fig.3c.pdf` and the three long-form plotting tables.

## Full factor-matrix recomputation

Download the public bigWig inputs listed in `fig3c_bigwig_accession_map.csv`, name each file `<accession>.bw`, and run:

```bash
python analyses/16_ChromHMM_Fig3c/scripts/recompute_fig3c_signal_matrix.py \
  --state-dir "$GAO2026_DATA_ROOT/16_ChromHMM_Fig3c/state_beds" \
  --bigwig-dir "$GAO2026_EXTERNAL_ROOT/16_ChromHMM_Fig3c/bw" \
  --output "$GAO2026_OUTPUT_ROOT/16_ChromHMM_Fig3c/chromState_signal_matrix.tsv"
```

The calculation calls `pyBigWig.stats(..., type="mean", nBins=1)` for every state interval, multiplies each interval mean by its length, and divides the summed signal by the summed length. This is computationally intensive and is not run by the default processed-data reproduction suite.

Only bigWig files whose accession is listed in the script's `SAMPLE_MAP` are processed; any other file in the directory is ignored.
