# 14_acute_inhibitor

Same-cell acute inhibitor analysis used for Supplementary Fig. 13.

- Upstream analysis notebook: `notebooks/02_acute_inhibitor_signal_analysis.ipynb`
- Final-table figure script: `scripts/replot_supplementary_fig13_from_tables.py`
- Deposited final tables: `14_acute_inhibitor/` (the earlier
  `revision/14_acute_inhibitor/` layout is also accepted).
- The corrected tables contain both `nuclear_bg` and `local_bg` metrics and the
  six panels in the current legend: H3K27ac after HDAC inhibition at Nanog and
  Sox2, plus H3K27ac, p300, HDAC1, and HDAC3 after THZ1 at Nanog.
- The experiment counts represented in panels a-f are 2, 3, 2, 2, 3, and 3,
  respectively. The earlier release candidate lacked the nuclear metric and
  the third Sox2 HDAC-inhibitor experiment and must not be used.
- Full upstream reprocessing requires the Cell 4 and QC tables generated from the controlled microscopy data. Set `GAO2026_RAW_ROOT` or `LIVE_INHIBITOR_DATA_ROOT` to their parent directory.
- Set `GAO2026_OUTPUT_ROOT` or `LIVE_INHIBITOR_OUTPUT_ROOT` for generated outputs.
- Newly acquired ND2/TIFF files are not included in the public package.
