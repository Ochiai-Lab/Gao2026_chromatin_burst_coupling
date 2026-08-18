#!/usr/bin/env python3
"""Run all figure replots supported by deposited processed tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


MODULE_SCRIPTS = (
    "analyses/08_SoRa/scripts/replot_quantitative_panels_from_tables.py",
    "analyses/09_bead_resolution/scripts/replot_panel_e_from_tables.py",
    "analyses/10_HDAC_time_windows/scripts/replot_time_windows_from_tables.py",
    "analyses/11_HDAC_threshold_sensitivity/scripts/replot_threshold_sensitivity_from_tables.py",
    "analyses/14_acute_inhibitor/scripts/replot_supplementary_fig13_from_tables.py",
    "analyses/15_chipseq_enrichment/scripts/replot_fig_s5e_from_tables.py",
    "analyses/16_ChromHMM_Fig3c/scripts/replot_fig3c.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GAO2026_DATA_ROOT", "data")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("GAO2026_OUTPUT_ROOT", "outputs")),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for relative_script in MODULE_SCRIPTS:
        script = repository_root / relative_script
        command = [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ]
        print(f"\n[RUN] {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append((relative_script, result.returncode))

    if failures:
        formatted = ", ".join(f"{script} (exit {code})" for script, code in failures)
        raise SystemExit(f"Deposited-table replots failed: {formatted}")
    print(f"\nAll deposited-table replots completed under {output_root}")


if __name__ == "__main__":
    main()
