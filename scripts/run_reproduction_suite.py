#!/usr/bin/env python3
"""Run all Gao2026 analyses supported by the deposited processed data.

The manifest distinguishes analyses recomputed from deposited processed inputs
from plots regenerated from deposited final plotting tables. Raw microscopy
stages are intentionally not represented as successful runs when their inputs
are absent.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Task:
    task_id: str
    reproduction_level: str
    kind: str
    relative_path: str
    timeout_seconds: int = 3600


TASKS = (
    Task(
        "00_validate_processed_data",
        "data_integrity_validation",
        "validator",
        "scripts/validate_processed_release.py",
    ),
    Task(
        "02_cross_correlation",
        "recomputed_from_deposited_processed_inputs",
        "notebook",
        "analyses/02_Cross_correlation/notebooks/2_Timelapse_analysis.ipynb",
    ),
    Task(
        "04_seqFISH",
        "recomputed_from_deposited_processed_inputs",
        "notebook",
        "analyses/04_seqFISH/notebook/seqFISH_reanalysis.ipynb",
    ),
    Task(
        "05_sci_mtChIL_seq",
        "recomputed_from_deposited_processed_inputs",
        "notebook",
        "analyses/05_sci_mtChIL_seq/notebook/sci_mtChIL_seq_reanalysis.ipynb",
    ),
    Task(
        "06_SOX2_temporal_resolution",
        "replot_from_deposited_final_tables",
        "notebook",
        "analyses/06_SOX2_temporal_resolution/notebooks/2_Timelapse_replot_from_deposited_tables.ipynb",
    ),
    Task(
        "07_MSD",
        "replot_from_deposited_final_tables",
        "notebook",
        "analyses/07_MSD/notebooks/MSD_publication_replot_from_deposited_tables.ipynb",
    ),
    Task(
        "08_SoRa_quantitative",
        "replot_from_deposited_final_tables_no_representative_images",
        "script",
        "analyses/08_SoRa/scripts/replot_quantitative_panels_from_tables.py",
    ),
    Task(
        "09_bead_resolution_panel_e",
        "replot_from_deposited_final_tables_no_representative_images",
        "script",
        "analyses/09_bead_resolution/scripts/replot_panel_e_from_tables.py",
    ),
    Task(
        "10_HDAC_time_windows",
        "replot_from_deposited_final_tables",
        "script",
        "analyses/10_HDAC_time_windows/scripts/replot_time_windows_from_tables.py",
    ),
    Task(
        "11_HDAC_threshold_sensitivity",
        "replot_from_deposited_final_tables",
        "script",
        "analyses/11_HDAC_threshold_sensitivity/scripts/replot_threshold_sensitivity_from_tables.py",
    ),
    Task(
        "12_Fig5_integrated",
        "recomputed_from_exact_deposited_aggregate_inputs",
        "notebook",
        "analyses/12_Fig5_integrated/notebooks/figure5_integrated.ipynb",
    ),
    Task(
        "13_MARCS",
        "recomputed_from_deposited_source_workbooks",
        "notebook",
        "analyses/13_MARCS/notebooks/MARCS_HDAC_reanalysis_notebook.ipynb",
    ),
    Task(
        "14_acute_inhibitor",
        "replot_from_corrected_deposited_final_tables",
        "script",
        "analyses/14_acute_inhibitor/scripts/replot_supplementary_fig13_from_tables.py",
    ),
    Task(
        "15_chipseq_enrichment",
        "replot_from_deposited_final_tables",
        "script",
        "analyses/15_chipseq_enrichment/scripts/replot_fig_s5e_from_tables.py",
    ),
    Task(
        "16_ChromHMM_Fig3c",
        "replot_from_deposited_final_matrices",
        "script",
        "analyses/16_ChromHMM_Fig3c/scripts/replot_fig3c.py",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command_for(
    task: Task,
    repository_root: Path,
    data_root: Path,
    output_root: Path,
    executed_notebook_root: Path,
) -> list[str]:
    target = repository_root / task.relative_path
    if task.kind == "notebook":
        executed_notebook_root.mkdir(parents=True, exist_ok=True)
        return [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={task.timeout_seconds}",
            "--output",
            f"{task.task_id}.executed.ipynb",
            "--output-dir",
            str(executed_notebook_root),
            str(target),
        ]
    if task.kind == "script":
        return [
            sys.executable,
            str(target),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ]
    if task.kind == "validator":
        return [
            sys.executable,
            str(target),
            "--data-root",
            str(data_root),
        ]
    raise ValueError(f"Unsupported task kind: {task.kind}")


def write_manifest(rows: list[dict[str, object]], output_root: Path) -> None:
    manifest_json = output_root / "reproduction_run_manifest.json"
    manifest_tsv = output_root / "reproduction_run_manifest.tsv"
    manifest_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0]) if rows else []
    with manifest_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
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
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional task IDs. Omit to run the complete processed-data suite.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    notebook_root = output_root / "executed_notebooks"
    mpl_root = output_root / "mplconfig"
    for path in (log_root, notebook_root, mpl_root):
        path.mkdir(parents=True, exist_ok=True)

    selected = [task for task in TASKS if not args.only or task.task_id in args.only]
    unknown = set(args.only or ()) - {task.task_id for task in TASKS}
    if unknown:
        raise SystemExit(f"Unknown task IDs: {sorted(unknown)}")

    environment = os.environ.copy()
    environment.update(
        {
            "GAO2026_DATA_ROOT": str(data_root),
            "GAO2026_OUTPUT_ROOT": str(output_root),
            "MPLCONFIGDIR": str(mpl_root),
            "PYTHONUNBUFFERED": "1",
        }
    )

    rows: list[dict[str, object]] = []
    for task in selected:
        command = command_for(
            task,
            repository_root,
            data_root,
            output_root,
            notebook_root,
        )
        target = repository_root / task.relative_path
        log_path = log_root / f"{task.task_id}.log"
        started_at = utc_now()
        started = time.monotonic()
        print(f"[RUN] {task.task_id}: {' '.join(command)}", flush=True)
        if not target.exists():
            return_code = 127
            log_path.write_text(f"Missing target: {target}\n", encoding="utf-8")
        else:
            try:
                with log_path.open("w", encoding="utf-8") as log_handle:
                    completed = subprocess.run(
                        command,
                        cwd=repository_root,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=task.timeout_seconds + 60,
                    )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                return_code = 124
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(
                        f"\nTimed out after {task.timeout_seconds + 60} seconds.\n"
                    )
        elapsed = round(time.monotonic() - started, 3)
        status = "pass" if return_code == 0 else "fail"
        print(f"[{status.upper()}] {task.task_id} ({elapsed:.1f} s)", flush=True)
        row = {
            **asdict(task),
            "status": status,
            "return_code": return_code,
            "started_at_utc": started_at,
            "elapsed_seconds": elapsed,
            "log_path": str(log_path.relative_to(output_root)),
        }
        rows.append(row)
        write_manifest(rows, output_root)

    failures = [row for row in rows if row["status"] != "pass"]
    print(
        f"Completed {len(rows)} tasks: "
        f"{len(rows) - len(failures)} pass, {len(failures)} fail."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
