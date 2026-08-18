# 09_bead_resolution

Fluorescent-bead validation of conventional CSU-W1 and SoRa lateral imaging
performance.

- The exact preprocessing described in Methods is retained: conventional
  CSU-W1 images are fitted without filtering, whereas SoRa images receive a
  median `3 x 3` filter before fitting.
- `scripts/replot_panel_e_from_tables.py` regenerates the paired FWHM summary
  from deposited per-bead and pairing tables.
- Raw ND2 stacks are not included, so spot detection, doublet QC, Gaussian
  refitting, and the representative image panel cannot be rerun from the
  current public package.
- Deposited medians are 290.76/238.00 nm (640 nm, N=22), 274.97/216.22 nm
  (515 nm, N=17), and 244.48/213.45 nm (445 nm, N=19) for CSU-W1/SoRa.
