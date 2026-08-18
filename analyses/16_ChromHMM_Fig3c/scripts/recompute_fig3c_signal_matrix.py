#!/usr/bin/env python3
"""Recompute the Fig. 3c factor-by-ChromHMM-state signal matrix.

This is the student notebook's original length-weighted pyBigWig calculation,
exposed as a portable command-line script. Public bigWig files are external
inputs because they are identified by public accessions and are not duplicated
in the deposited data archive.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyBigWig


SAMPLE_MAP = {
    "SRX3898502": "BRD4",
    "ERX2789783": "HDAC3",
    "SRX2875257": "RNAPII",
    "SRX029864": "SIN3A",
    "SRX047138": "HDAC2",
    "SRX1091746": "NCOR2",
    "SRX1091748": "H3K27ac",
    "SRX2539820": "HDAC1",
    "SRX4403303": "CHD4",
    "SRX194535": "p300",
    "SRX6098709": "NCOR1",
    "SRX13298161": "ATAC-seq",
    "ENCFF806NDV": "H3K4me3",
    "ENCFF705OWT": "H3K9ac",
    "ENCFF519ZAT": "H3K9me3",
    "ENCFF182FTP": "H3K27me3",
    "SRX017056": "RNAPII-Ser5ph",
    "SRX017057": "RNAPII-Ser2ph",
    "SRX873340": "SIRT6",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--bigwig-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    states: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for bed_path in sorted(args.state_dir.glob("*.bed")):
        state_name = "_".join(bed_path.name.split("_")[1:]).replace(".bed", "")
        intervals_by_chromosome: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        with bed_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                chrom, start, end = line.split()[:3]
                intervals_by_chromosome[chrom].append((int(start), int(end)))
        states[state_name] = dict(intervals_by_chromosome)

    state_names = list(states)
    result = pd.DataFrame(index=[], columns=state_names, dtype=float)

    for bigwig_path in sorted(args.bigwig_dir.glob("*.bw")):
        sample_id = bigwig_path.stem
        factor_name = SAMPLE_MAP.get(sample_id, sample_id)
        print(f"Processing: {sample_id} -> {factor_name}", flush=True)
        bigwig = pyBigWig.open(str(bigwig_path))
        row_values: dict[str, float] = {}

        for state_name, intervals_by_chromosome in states.items():
            total_signal = 0.0
            total_bp = 0
            for chromosome, intervals in intervals_by_chromosome.items():
                if chromosome not in bigwig.chroms():
                    continue
                for start, end in intervals:
                    length = end - start
                    if length <= 0:
                        continue
                    mean_value = bigwig.stats(
                        chromosome,
                        start,
                        end,
                        type="mean",
                        nBins=1,
                    )[0]
                    if mean_value is None:
                        continue
                    total_signal += mean_value * length
                    total_bp += length
            row_values[state_name] = total_signal / total_bp if total_bp else 0.0

        bigwig.close()
        result.loc[factor_name] = pd.Series(row_values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t")
    print(f"Saved matrix to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
