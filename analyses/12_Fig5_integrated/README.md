# 12_Fig5_integrated

Integrated Nanog and Sox2 HDAC-inhibitor analyses used for Fig. 5.

- The public notebook now reads the exact aggregate CSV inputs used for final
  figure generation.
- Earlier release-candidate state-by-frame and dwell tables were derived
  exports and were not the exact notebook inputs; they have been replaced for
  reproducibility testing.
- Re-execution reproduced all nine final integrated output tables with numeric
  difference zero.
- Left-censored events are excluded as stated in the figure; right-censored
  events remain represented by the survival-analysis inputs.
