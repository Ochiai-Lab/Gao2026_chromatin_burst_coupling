# Validation

On 2026-09-26, all 15 tasks in `scripts/run_reproduction_suite.py` completed
successfully using the processed inputs extracted from the companion archive.
All 27 comparisons in `scripts/validate_reproduced_outputs.py` passed with
`rtol=1e-12` and `atol=1e-12`. The largest observed numeric difference was
1.1102230246251565e-16.

The validation environment used Python 3.9.12, NumPy 1.26.4, pandas 2.3.1,
SciPy 1.13.1, matplotlib 3.9.4, nbconvert 7.16.6, and openpyxl 3.1.5.
These are the versions of this reproduction run; they do not replace the
original analysis environment recorded in `environment.yml`.

## Scope

The suite mixes recomputation from processed observations with final-table
replots. See `REPRODUCIBILITY_SCOPE.tsv` and `RUN_REPRODUCTION.md` for each
module's scope. Passing these tests does not establish independent reproduction
of raw-image preprocessing, segmentation, tracking, or every upstream statistic.

The version 1.0.0 data archive replaces absolute source-path metadata in
115 Snapshot PKLs and the `image_path` column of one Snapshot CSV with
`external_snapshot_sources/<path identifier>/<original filename>`.
The identifier preserves distinct source references without disclosing an
absolute filesystem path. Numeric values, image arrays, conditions, indexes,
and non-path metadata are verified unchanged after serialization. All other
archive members are unchanged. This metadata-only change does not affect the
quantitative analyses. Aliased images are not newly supplied by this release.

Notebook execution outputs are intentionally absent from the source repository.
Run the reproduction suite to generate outputs in a separate output directory.

## Version 1.0.1 label-only update (2026-10-05)

Transcription-state CSV values were standardized to Active/Inactive in
447 tables (720,649 values in
state/raw_state/kind columns). All 9,393,634 other
cells in those tables were checked against version 1.0.0 without numerical
parsing or rounding. The remaining 1624
archive members, including all raw microscopy files, are byte-identical.
All 2071 new archive members passed SHA-256 readback checks;
no standalone ON/OFF CSV values remain. Binary calls and legacy field names
are retained for compatibility, as documented in the code README.

All 15 processed-data reproduction tasks and all 27 numerical comparisons
passed again for version 1.0.1, using rtol=1e-12 and atol=1e-12. The largest
absolute difference was 1.1102230246251565e-16. Three focused CSV-boundary tests
also passed. This remains processed-input validation and final-table
replotting, not independent repetition of upstream raw-image analysis.

## Version 1.0.2 corrected TSS windows (2026-10-06)

Supplementary Fig. 5e previously used the boundaries of the 1-Mb browser
display regions as TSSs. Version 1.0.2 uses explicit RefSeq Select transcript
TSSs from NCBI Mus musculus Updated Annotation Release 108.20200622
(GRCm38.p6). The transcript IDs, original 1-based coordinates, 0-based TSSs,
display intervals and v1/v2 numerical comparison are deposited in module 15.
All four 0.5-Mb windows were re-extracted from the same 18 bigWig tracks.
The unchanged genome-bin means and finite-value filtering were retained;
joint normalization was recomputed from genome-bin and corrected gene-window
signals. Missing values remain missing, not zero. Plotting style is unchanged.

The complete corrected upstream notebook executed without cell errors using
Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, pyBigWig 0.3.25 and Matplotlib
3.10.9. All 72 corrected gene values matched the final-table normalization.
Before using the background cache, 72 original gene-window integrations and
108 sampled genome-bin integrations were checked against the source bigWigs
(rtol=1e-12, atol=1e-12; missing values matched as missing).

The processed-data integrity validator and the module-15 final-table script
passed in the general Python 3.9.12 reproduction environment. The module-15
script checks all annotated TSSs and 500-kb windows and recomputes all joint
z-scores (rtol=1e-12, atol=1e-12). Other modules were not rerun: their source
code and deposited inputs remain unchanged from the documented v1.0.1 runs.
The historical 15-task/27-comparison checks above describe those prior runs,
not a new full-suite execution in this release.

Every member of the v1.0.2 data ZIP passed SHA-256 and size readback checks.
The numerical update is confined to module 15. No newly acquired microscopy
raw data or generated figure files were added.
