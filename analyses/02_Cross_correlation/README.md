# 02_Cross_correlation

Cross-correlation and lead-lag-index analysis used in Fig. 4.

- The deposited input contains all 11 trace CSV datasets used in the original
  multiple-testing correction.
- The notebook computes per-cell cross-correlation curves, bootstrap confidence
  intervals, LLI, time-shift permutation P values, and Benjamini-Hochberg Q
  values across all 11 datasets before displaying SOX2, BRD4, SIN3A, and HDAC3.
- Re-execution reproduced the reported N, LLI medians, 95% confidence
  intervals, P values, and Q values with numerical difference zero.
- Raw TIFF-to-trace preprocessing is outside this processed-data module and was
  not repeated in the release audit.
