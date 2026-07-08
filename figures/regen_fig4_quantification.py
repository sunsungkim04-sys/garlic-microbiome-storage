#!/usr/bin/env python3
"""
fig4_quant_local.py — Figure 4 (quantification, 3 panels) regenerated locally on the
even-month frame (0/2/4/6 M).

Supersedes figures/regen_figures.py::fig4_quant(), which
  (a) exempted panel A from the even-month filter (`if not title.startswith("A.")`),
  (b) hardcoded panel-A x-ticks 0M..5M, and
  (c) hardcoded "Kruskal-Wallis H = 15.83, p = 0.007" (a 6-group statistic).

Panel A (CFU) has no month-6 measurement, so it carries three timepoints (0/2/4 M).
All statistics below are recomputed from Manuscript_figures/_data/quantification_summary.csv;
because the per-group value ranges are non-overlapping, the rank-based Kruskal-Wallis and
Dunn results are exactly determined by the summary statistics (see scripts/verify_fig4_stats.py).

    A. CFU        n = 9  (3 x 3)   KW H = 7.20,  df = 2, p = 0.027   Dunn: 0M a / 2M b / 4M ab
    B. 16S qPCR   n = 12 (4 x 3)   KW H = 4.38,  df = 3, p = 0.22 (NS)
    C. Fusarium   n = 12 (4 x 3)   KW H = 10.38, df = 3, p = 0.016

Run:  python3 figures/regen_fig4_quantification.py   (writes figures/output/)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures" / "output"
OUT.mkdir(parents=True, exist_ok=True)

EVEN_MONTHS = [0, 2, 4, 6]
QPCR_LOD = 1.0

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Recomputed on the even-month frame. See module docstring.
KW_TEXT = {
    "A.": "Kruskal-Wallis  H = 7.20, p = 0.027",
    "B.": "Kruskal-Wallis  H = 4.38, p = 0.22 (NS)",
    "C.": "Kruskal-Wallis  H = 10.38, p = 0.016",
}

PANELS = [
    ("A. Bacterial colony count (log CFU/g)", "#9b5fb5", "colony_CFU"),
    ("B. 16S rRNA gene abundance (log copies/g)", "#5b9bd5", "bacteria_16S"),
    ("C. Fusarium spp. abundance (nominal log$_{10}$ copies/g)", "#f0a45a", "fusarium"),
]


def save_both(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  OK  {name}.{{png,pdf}}")


def main():
    qsum = pd.read_csv(DATA / "quantification_summary.csv")
    qsum["month_int"] = qsum["month"].str.replace("M", "", regex=False).astype(int)

    stray = sorted(set(qsum["month_int"]) - set(EVEN_MONTHS))
    if stray:
        raise SystemExit(f"quantification_summary.csv still contains odd months: {stray}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (title, color, dataset) in zip(axes, PANELS):
        sub = qsum[qsum["dataset"] == dataset].sort_values("month_int")
        if sub.empty:
            raise SystemExit(f"dataset {dataset!r} missing from quantification_summary.csv")

        x = sub["month_int"].values
        y = sub["mean"].values
        yerr = sub["sd"].values

        if title.startswith("C."):
            # Fusarium qPCR: standard curve not preserved (nominal). Near-baseline
            # timepoints are shown as <LOD with open markers, consistent with Fig 10b.
            above = y >= QPCR_LOD
            ax.plot(x, np.where(above, y, QPCR_LOD - 0.05), "-", color=color,
                    linewidth=1.8, alpha=0.5, zorder=2)
            if above.any():
                ax.errorbar(x[above], y[above], yerr=yerr[above], fmt="o", color=color,
                            markersize=8, capsize=4, linewidth=0, markeredgecolor=color, zorder=4)
            if (~above).any():
                ax.scatter(x[~above], [QPCR_LOD - 0.05] * int((~above).sum()), s=80, marker="o",
                           facecolors="white", edgecolors=color, linewidths=1.6, zorder=4)
            ax.axhline(QPCR_LOD, color="#666", lw=0.9, linestyle="--", zorder=1)
            ax.text(0.985, QPCR_LOD + 0.05, "LOD", transform=ax.get_yaxis_transform(),
                    fontsize=8, va="bottom", ha="right", color="#444", style="italic")
            ax.text(0.5, 0.05, "below detection limit  (<LOD)", transform=ax.transAxes,
                    fontsize=8, color="#666", style="italic", ha="center")
            ipk = int(np.argmax(y))
            ax.annotate(f"* peak {y[ipk]:.2f}", xy=(x[ipk], y[ipk]),
                        xytext=(x[ipk] + 0.35, y[ipk]), fontsize=8.5, color=color,
                        fontweight="bold")
            ax.set_ylim(QPCR_LOD - 0.4, float(np.max(y)) + 0.9)
        else:
            ax.errorbar(x, y, yerr=yerr, marker="o", color=color, linewidth=1.8,
                        capsize=4, markersize=8, markeredgecolor=color)
            # No linear trend line on panel A: CFU rises to month 2 and then plateaus,
            # so a least-squares line would imply a monotonic increase the data do not show.
            if not title.startswith("A.") and len(x) >= 2:
                coef = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 50)
                ax.plot(xline, np.polyval(coef, xline), linestyle="--", color=color,
                        alpha=0.4, linewidth=1.5)

        if title.startswith("A."):
            lo, hi = float(np.min(y - yerr)), float(np.max(y + yerr))
            span = hi - lo
            ax.set_ylim(lo - 0.30 * span, hi + 0.35 * span)
            # Dunn (BH) compact-letter display; CFU was not assayed at month 6.
            for xi, yi, ei, lt in zip(x, y, yerr, sub["letter"].values):
                ax.text(xi, yi + ei + 0.03 * span, lt, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=color)
            ax.text(6, lo - 0.18 * span, "6M\nnot assayed", ha="center", va="center",
                    fontsize=7.5, color="#888", style="italic")

        # Common time axis across all three panels.
        ax.set_xlim(-0.45, 6.45)
        ax.set_xticks(EVEN_MONTHS)
        ax.set_xticklabels([f"{m}M" for m in EVEN_MONTHS])
        ax.set_xlabel("Month")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.4)

        kwtxt = next((v for k, v in KW_TEXT.items() if title.startswith(k)), None)
        if kwtxt:
            ax.text(0.04, 0.96, kwtxt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="0.6", alpha=0.92))

    plt.tight_layout()
    save_both(fig, "Figure_4_quantification")


if __name__ == "__main__":
    main()
