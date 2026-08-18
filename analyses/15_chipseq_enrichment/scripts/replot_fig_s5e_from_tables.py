#!/usr/bin/env python3
"""Replot Supplementary Fig. 5e from deposited joint-z-score tables.

The upstream bigWig integration cannot be repeated without the public bigWig
files. This plotting block is the final manuscript plotting code with the two
exact exported tables substituted for the in-memory data frames.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_module_root(data_root: Path) -> Path:
    for candidate in (
        data_root / "revision" / "15_chipseq_enrichment",
        data_root / "15_chipseq_enrichment",
        data_root,
    ):
        if (candidate / "fig_s5e_genome_bins_joint_zscore.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Supplementary Fig. 5e tables below {data_root}")


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
    module_root = resolve_module_root(args.data_root.resolve())
    output_root = args.output_root.resolve() / "15_chipseq_enrichment"
    output_root.mkdir(parents=True, exist_ok=True)

    genome_table = pd.read_csv(module_root / "fig_s5e_genome_bins_joint_zscore.csv")
    gene_table = pd.read_csv(module_root / "fig_s5e_gene_windows_joint_zscore.csv")
    genome_z_df = genome_table.loc[genome_table["included_in_boxplot"].astype(bool)].copy()
    gene_z_df = gene_table.loc[gene_table["included_as_gene_point"].astype(bool)].copy()
    genome_z_df = genome_z_df[["target_order", "target", "z_joint"]]
    gene_z_df = gene_z_df[["target_order", "target", "gene", "z_joint"]]

    target_order = (
        genome_z_df[["target_order", "target"]]
        .drop_duplicates()
        .sort_values("target_order")
    )
    targets_plot = target_order["target"].tolist()
    # This is the appearance order of genes in the final in-memory results_df.
    # The exported table is target-major, so pd.unique() on that file would
    # silently assign the manuscript colors to different genes.
    genes = [gene for gene in ("Dnmt3L", "Nanog", "Sox2", "Usp5") if gene in set(gene_z_df["gene"])]
    expected_targets = set(targets_plot)
    if set(gene_z_df["target"]) != expected_targets:
        raise AssertionError("Gene-point and genome-bin target sets differ")
    if genome_z_df["z_joint"].isna().any() or gene_z_df["z_joint"].isna().any():
        raise AssertionError("Non-finite joint z-scores are present in plotted rows")

    plt.style.use("seaborn-v0_8")
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        figure, axis = plt.subplots(figsize=(max(8, 0.6 * len(targets_plot)), 4))
        box_data = [
            genome_z_df.loc[genome_z_df["target"].eq(target), "z_joint"].to_numpy()
            for target in targets_plot
        ]
        axis.boxplot(
            box_data,
            positions=np.arange(len(targets_plot)),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

        color_map = plt.get_cmap("tab10")
        gene_to_color = {gene: color_map(index % 10) for index, gene in enumerate(genes)}
        n_genes = max(1, len(genes))
        labeled: set[str] = set()
        for gene_index, gene in enumerate(genes):
            subset = gene_z_df.loc[gene_z_df["gene"].eq(gene)]
            x_base = np.array([targets_plot.index(target) for target in subset["target"]], dtype=float)
            offset = (gene_index - (n_genes - 1) / 2) * 0.10
            axis.scatter(
                x_base + offset,
                subset["z_joint"].to_numpy(),
                s=50,
                color=gene_to_color[gene],
                edgecolor="black",
                linewidth=0.4,
                label=gene if gene not in labeled else None,
                zorder=3,
            )
            labeled.add(gene)

        axis.set_xticks(np.arange(len(targets_plot)))
        axis.set_xticklabels(targets_plot, rotation=45, ha="right")
        axis.set_xlabel("target")
        axis.set_ylabel("Joint z-score")
        axis.set_title("Per-target joint z-scores (genome-wide fixed bins + gene windows)")
        legend = axis.legend(title="gene", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        for text in legend.get_texts():
            text.set_fontstyle("italic")
        figure.tight_layout()
        output_pdf = output_root / "joint_zscore_per_target_fixedbins_genewindows.pdf"
        figure.savefig(output_pdf, format="pdf", bbox_inches="tight")
        plt.close(figure)

    validation = target_order.copy()
    validation["n_genome_bins"] = validation["target"].map(genome_z_df.groupby("target").size())
    validation["n_gene_points"] = validation["target"].map(gene_z_df.groupby("target").size())
    validation["table_validation"] = "PASS"
    validation.to_csv(output_root / "fig_s5e_table_validation.csv", index=False)
    print(output_pdf)


if __name__ == "__main__":
    main()
