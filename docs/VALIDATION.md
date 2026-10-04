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
