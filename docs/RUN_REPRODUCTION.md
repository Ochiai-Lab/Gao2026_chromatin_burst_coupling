# Reproduction workflow

## 1. Environment

```bash
conda env create -f environment.yml
conda activate gao2026
```

`environment.yml` records the versions used for the manuscript analysis.
CUDA-enabled PyTorch installation is platform specific; keep PyTorch 2.2.2
when replacing its CPU build with a CUDA build.

## 2. Data

Set `GAO2026_DATA_ROOT` to the extracted unified data directory. The directory
must contain top-level module directories `01_Snapshot_analysis` through
`16_ChromHMM_Fig3c`. For compatibility, scripts that only need revision
tables also accept modules 06-16 below `revision/`.

## 3. Run

```bash
python scripts/run_reproduction_suite.py \
  --data-root "$GAO2026_DATA_ROOT" \
  --output-root "$PWD/reproduction_outputs"
```

The run produces:

- `reproduction_run_manifest.tsv` and `.json`
- one log per task under `logs/`
- executed copies of seven notebooks under `executed_notebooks/`
- generated tables and PDFs under module-numbered output directories

Then validate the regenerated numerical tables:

```bash
python scripts/validate_reproduced_outputs.py \
  --data-root "$GAO2026_DATA_ROOT" \
  --output-root "$PWD/reproduction_outputs"
```

## 4. Interpretation of reproduction levels

- `recomputed_from_deposited_processed_inputs`: analysis statistics are
  recalculated from the processed observations deposited for that module.
- `recomputed_from_exact_deposited_aggregate_inputs`: downstream Fig. 5
  statistics and plots are recalculated from the exact per-frame/run aggregate
  files read by the manuscript notebook.
- `replot_from_deposited_final_tables`: the plot is regenerated, but an
  upstream image/tracking/statistical stage cannot be repeated because its
  input is not deposited.
- `no_representative_images`: numerical panels are regenerated, while the
  representative image panel requires a non-deposited image array or raw ND2.

## 5. Known limits

1. Snapshot: publication-aware replicate mapping and the missing publication
   acquisition have been reconciled. All 95 Illustrator-linked PDFs are now
   pixel-exact or near-identical to regenerated outputs. Raw image-to-PKL
   preprocessing was not repeated.
2. Fig. 3c: the final emission and factor-state matrices, state BEDs, exact
   plotting code, and plotted values are deposited. The default suite performs
   an exact matrix-to-figure replot. Full factor-matrix recomputation requires
   the 18 public bigWig files listed by module 16 and is computationally intensive.
3. SOX2 temporal resolution: 1-min and 30-s final tables are deposited, but the
   three trace-level CSV inputs required to repeat the entire calculation are
   not currently included.
4. SoRa and beads: numerical final tables are deposited. Representative image
   arrays and raw bead ND2 files are not included.
5. ChIP-seq enrichment: final tables are deposited. Full recomputation requires
   the 18 public bigWig inputs documented by module 15.

Use the matching `v1.0.2` code and data archives. See `VALIDATION.md` for the
2026-10-06 corrected-TSS normalization and final-table reproduction checks
and their documented scope. These tests do not imply that
non-deposited raw-image processing has been repeated.
