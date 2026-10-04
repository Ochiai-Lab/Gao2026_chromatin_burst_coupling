# Gao2026 chromatin burst coupling

Analysis code for Gao et al., "Minute-scale coupling of chromatin marks and
transcriptional bursts". Version: `1.0.1`.

## Scope

The repository contains the original five analysis modules plus revision modules for SOX2 temporal-resolution analysis, MSD, fixed-cell SoRa imaging, bead-resolution validation, HDAC-inhibitor time windows, threshold sensitivity, integrated Fig. 5 analyses, MARCS reanalysis, acute inhibitor imaging, ChIP-seq enrichment, and the Fig. 3c ChromHMM analysis.

Notebook outputs, generated figures, data files, credentials, personal paths,
and absolute NAS paths are excluded from the clean GitHub code package. Raw-image stages
require separately controlled microscopy data. Processed-data analyses and
final-table replots use the deposited data package.

The companion archive is `Gao2026_data_v1.0.1.zip`. Code and data are packaged
separately so that this repository can be used without downloading raw images.
The article Source Data workbook accompanies the journal article and is not
duplicated in the data archive. See `docs/VALIDATION.md` for the tested workflow
and its limits.

## Transcription-state labels

CSV columns `state`, `raw_state`, and `kind` use `Active`/`Inactive` in version
1.0.1. The state-aware readers accept both these labels and the earlier
`ON`/`OFF` aliases, so the underlying classification and numerical analyses
are unchanged. Internal aliases, binary 0/1 calls, field names such as
`on_images`/`off_images`, and legacy filenames remain for compatibility.
CSV export explicitly converts state labels without mutating the analysis
DataFrame. No measurement, threshold, statistical value, or raw image was
changed by this label-only update.

Run the focused compatibility tests with `python -m unittest discover -s tests`.

Only data displayed in the manuscript, observations used to calculate a
displayed aggregate, and direct processed provenance for a displayed analysis
belong in the public package. For example, the third Sox2 H3K9ac Snapshot
acquisition is retained because it contributes to Supplementary Fig. 3e,
whereas the unpublished Sox2 THZ1 branch is excluded.

## Configuration

```bash
export GAO2026_DATA_ROOT=/path/to/deposited/data
export GAO2026_RAW_ROOT=/path/to/optional/external_raw
export GAO2026_EXTERNAL_ROOT=/path/to/optional/external_public_inputs
export GAO2026_OUTPUT_ROOT=$PWD/outputs
```

Alternatively, copy `config.example.yml` to `config.yml` and use the same roots.

## Analysis order

1. `01_Snapshot_analysis`
2. `02_Cross_correlation`
3. `03_HDAC_inhibitor`
4. `04_seqFISH`
5. `05_sci_mtChIL_seq`
6. `06_SOX2_temporal_resolution`
7. `07_MSD`
8. `08_SoRa`
9. `09_bead_resolution`
10. `10_HDAC_time_windows`
11. `11_HDAC_threshold_sensitivity`
12. `12_Fig5_integrated`
13. `13_MARCS`
14. `14_acute_inhibitor`
15. `15_chipseq_enrichment`
16. `16_ChromHMM_Fig3c`

See `docs/FIGURE_CODE_DATA_MAP.tsv` for figure-to-code-to-data mapping and the module README files for input/output details.

## Reproduction

The unified data archive has one top-level directory per module:
`01_Snapshot_analysis` through `16_ChromHMM_Fig3c`. The scripts also accept
the earlier split layout in which modules 06-16 are below `revision/`.

Run the complete suite supported by deposited processed data:

```bash
conda env create -f environment.yml
conda activate gao2026
python scripts/run_reproduction_suite.py \
  --data-root /path/to/Gao2026_data \
  --output-root /path/to/reproduction_outputs
python scripts/validate_reproduced_outputs.py \
  --data-root /path/to/Gao2026_data \
  --output-root /path/to/reproduction_outputs
```

The runner writes an execution log and status manifest for every task. See
`docs/RUN_REPRODUCTION.md` and `docs/REPRODUCIBILITY_SCOPE.tsv` before
interpreting a generated plot.

## Deposited data layout

| Modules | Processed input under `GAO2026_DATA_ROOT` | Scope |
| --- | --- | --- |
| 01–05 | `01_Snapshot_analysis/` through `05_sci_mtChIL_seq/` | Historical inputs with audited additions or replacements where required |
| 06 | `06_SOX2_temporal_resolution/` | 2-min, 1-min, and 30-s cross-correlation/LLI tables |
| 07 | `07_MSD/` | Per-trajectory and replicate MSD tables |
| 08 | `08_SoRa/` | Fixed-cell radial profiles, replicate summaries, and focus-QC sensitivity |
| 09 | `09_bead_resolution/` | Per-bead FWHM, pairing, and QC tables |
| 10 | `10_HDAC_time_windows/` | Cell-level duty-cycle windows and statistics |
| 11 | `11_HDAC_threshold_sensitivity/` | MCP/mTetR threshold sweeps and effect summaries |
| 12 | `12_Fig5_integrated/` | Exact Nanog/Sox2 aggregate inputs and regenerated integrated tables |
| 13 | `13_MARCS/` | Source workbooks and all plotted protein/complex tables |
| 14 | `14_acute_inhibitor/` | Corrected six-panel Fig. S13 data with local and nuclear metrics |
| 15 | `15_chipseq_enrichment/` | Supplementary Fig. 5e gene-window, genome-bin, accession, and summary tables |
| 16 | `16_ChromHMM_Fig3c/` | Fig. 3c state BEDs, emission probabilities, factor-by-state signals, z-scores, and public accession map |

Raw-image stages cannot run without a separately supplied
`GAO2026_RAW_ROOT`. MARCS can be recomputed from the deposited source
workbooks. ChIP-seq and ChromHMM factor-matrix full recomputation require the
public bigWig files listed in the module READMEs; their deposited-table replots
do not.

Run the processed-table validation before analysis:

```bash
python scripts/validate_processed_release.py --data-root "$GAO2026_DATA_ROOT"
```

## Licensing

Code is released under the MIT License. Deposited data are licensed separately under CC BY 4.0.
