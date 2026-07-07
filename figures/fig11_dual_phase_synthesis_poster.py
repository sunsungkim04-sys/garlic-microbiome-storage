#!/usr/bin/env python3
"""
Fig 10 — Dual-phase senescence clock — POSTER edition.

Same data/tracks as fig11_dual_phase_synthesis.py, restyled for a poster:
  - vertical single-column layout (reads top-down in one poster column)
  - everything scaled up for 1-2 m viewing distance
      base 10->14 pt, ticks 9->12, lines 2.2->3.2, markers larger
  - punchier phase fills + a clearly visible 2 M "invasion peak" anchor line
  - de-cluttered: only the key call-outs survive (peak value, phase labels,
    16S peak, synchrony threshold). Detailed stats (Mantel rho / partial r /
    restricted-perm NS / KW p) are dropped -> live in the poster body text.

Output: Manuscript_figures/Main/Fig10_dual_phase_synthesis_poster.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA_MAIN = ROOT / "Manuscript_figures" / "_data"
DATA_INVEST = ROOT / "Attachments_investigation"
OUT = ROOT / "Manuscript_figures" / "Main"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.linewidth": 1.0,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

EVEN_MONTHS = [0, 2, 4, 6]
PHASE1 = (1.5, 3.5)
PHASE2 = (3.5, 6.5)

# soft pastel phase fills (light, readable behind data)
PHASE1_FILL = "#FBE3D0"   # pastel peach
PHASE2_FILL = "#DCE7F3"   # pastel blue
PHASE1_EDGE = "#C79A77"
PHASE2_EDGE = "#7E97BC"

COLOR_HOST = "#8B4513"
COLOR_HOST_FILL = "#D9B89C"
COLOR_INVADER = "#C0282E"
COLOR_16S = "#1F77B4"
COLOR_ITS = "#2CA02C"
COLOR_SYNC = "#7B5BAE"
ANCHOR_COLOR = "#B5341B"

SYNC_THRESHOLD = 0.20
QPCR_LOD = 1.0
ANCHOR_X = 2.0
XLIM = (-0.6, 6.7)


def load_track_a():
    return pd.DataFrame({"month": EVEN_MONTHS, "severity": [0.0, 1.0, 2.0, 3.5]})


def load_track_b():
    q = pd.read_csv(DATA_MAIN / "quantification_summary.csv")
    f = q[q["dataset"] == "fusarium"].copy()
    f["month_int"] = f["month"].str.rstrip("M").astype(int)
    f = f[f["month_int"].isin(EVEN_MONTHS)].sort_values("month_int")
    return f[["month_int", "mean", "sd"]].rename(columns={"month_int": "month"})


def load_track_c():
    s16 = pd.read_csv(DATA_INVEST / "per_sample_summary_freq5.csv")
    s16 = s16[s16["month"].isin(EVEN_MONTHS)]
    s16_agg = s16.groupby("month")["alpha_shannon_d130"].agg(["mean", "std"]).reset_index()
    s16_agg = s16_agg.rename(columns={"mean": "shannon_mean", "std": "shannon_sd"})
    s16_agg["marker_kind"] = "16S"
    its = pd.read_csv(DATA_INVEST / "qtof/audit/its_full_shannon_per_sample.tsv", sep="\t")
    its["month"] = its["timepoint"].str.rstrip("M").astype(int)
    its = its[its["month"].isin(EVEN_MONTHS)]
    its_agg = its.groupby("month")["shannon_genus"].agg(["mean", "std"]).reset_index()
    its_agg = its_agg.rename(columns={"mean": "shannon_mean", "std": "shannon_sd"})
    its_agg["marker_kind"] = "ITS"
    return pd.concat([s16_agg, its_agg], ignore_index=True)


def load_track_d():
    r = pd.read_csv(DATA_INVEST / "procrustes_16S_vs_ITS_residuals.csv")
    r = r[r["storage_month"].isin(EVEN_MONTHS)]
    agg = r.groupby("storage_month")["procrustes_residual"].agg(["mean", "std"]).reset_index()
    agg = agg.rename(columns={"storage_month": "month", "mean": "res_mean", "std": "res_sd"})
    return agg


def shade_phases(ax, ymin, ymax, alpha=0.45):
    ax.add_patch(Rectangle((PHASE1[0], ymin), PHASE1[1] - PHASE1[0], ymax - ymin,
                           color=PHASE1_FILL, alpha=alpha, zorder=0, linewidth=0))
    ax.add_patch(Rectangle((PHASE2[0], ymin), PHASE2[1] - PHASE2[0], ymax - ymin,
                           color=PHASE2_FILL, alpha=alpha, zorder=0, linewidth=0))
    ax.axvline(3.5, color="#333", lw=1.0, linestyle=":", alpha=0.6, zorder=1)
    # 2 M invasion-peak anchor — now clearly visible across every track
    ax.axvline(ANCHOR_X, color=ANCHOR_COLOR, lw=1.6, linestyle=(0, (5, 3)),
               alpha=0.55, zorder=1)


def style_track(ax, panel_letter, ymin, ymax, ylabel, ylabel_color="#222"):
    ax.set_xlim(*XLIM)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, color=ylabel_color, fontsize=14, labelpad=10, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#555")
    ax.spines["bottom"].set_color("#555")
    ax.tick_params(colors="#333", labelsize=12, width=1.0, length=4)
    ax.grid(True, axis="y", alpha=0.20, linestyle="-", linewidth=0.6, color="#aaa")
    ax.text(0.015, 0.95, panel_letter, transform=ax.transAxes,
            fontsize=19, fontweight="bold", va="top", color="#111",
            bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="#666", lw=1.0))


def main():
    a = load_track_a()
    b = load_track_b()
    c = load_track_c()
    d = load_track_d()

    fig = plt.figure(figsize=(9.2, 12.6))
    gs = fig.add_gridspec(
        nrows=5, ncols=1,
        height_ratios=[0.34, 1.0, 1.05, 1.05, 1.05],
        hspace=0.26,
    )
    ax_header = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    ax_b = fig.add_subplot(gs[2])
    ax_c = fig.add_subplot(gs[3])
    ax_d = fig.add_subplot(gs[4])

    # ============ Header band ============
    ax_header.set_xlim(*XLIM)
    ax_header.set_ylim(0, 1)
    ax_header.set_xticks([])
    ax_header.set_yticks([])
    for s in ax_header.spines.values():
        s.set_visible(False)
    ax_header.add_patch(Rectangle((PHASE1[0], 0.18), PHASE1[1] - PHASE1[0], 0.68,
                                  color=PHASE1_FILL, ec=PHASE1_EDGE, lw=1.6, zorder=2))
    ax_header.add_patch(Rectangle((PHASE2[0], 0.18), PHASE2[1] - PHASE2[0], 0.68,
                                  color=PHASE2_FILL, ec=PHASE2_EDGE, lw=1.6, zorder=2))
    ax_header.text(2.5, 0.52, "Phase 1\nInvasion window", ha="center", va="center",
                   fontsize=14, fontweight="bold", color="#6B3F1F", zorder=3)
    ax_header.text(5.0, 0.52, "Phase 2\nDecoupling", ha="center", va="center",
                   fontsize=14, fontweight="bold", color="#2B3F66", zorder=3)
    ax_header.annotate(
        "2 M peak", xy=(ANCHOR_X, 0.18), xytext=(ANCHOR_X, -0.16),
        ha="center", va="top", fontsize=12, fontweight="bold", color=ANCHOR_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=ANCHOR_COLOR, lw=1.8,
                        shrinkA=0, shrinkB=2),
        annotation_clip=False,
    )

    # ============ Track A — Host integrity ============
    ymin, ymax = -0.3, 4.6
    shade_phases(ax_a, ymin, ymax)
    ax_a.fill_between(a["month"], 0, a["severity"], color=COLOR_HOST_FILL, alpha=0.55, zorder=2)
    ax_a.plot(a["month"], a["severity"], "-", color=COLOR_HOST, lw=3.4, zorder=3)
    ax_a.plot(a["month"], a["severity"], "o", color=COLOR_HOST, ms=15, zorder=4,
              markerfacecolor=COLOR_HOST, markeredgecolor="white", markeredgewidth=2.0)
    for _, row in a.iterrows():
        ax_a.text(row["month"] + 0.20, row["severity"] - 0.06, f"{row['severity']:.1f}",
                  fontsize=12, color=COLOR_HOST, fontweight="bold", va="center")
    ax_a.text(0.99, 0.07, "illustrative (n=1)", transform=ax_a.transAxes,
              fontsize=11, ha="right", va="bottom", style="italic", color="#888")
    style_track(ax_a, "a", ymin, ymax, "Phenotype\nseverity (0–4)", ylabel_color=COLOR_HOST)
    ax_a.set_yticks([0, 1, 2, 3, 4])

    # ============ Track B — Invader load ============
    b_plot = b.copy()
    b_plot["above_lod"] = b_plot["mean"] >= QPCR_LOD
    ymin, ymax = -0.3, 5.6
    shade_phases(ax_b, ymin, ymax)
    ax_b.add_patch(Rectangle((XLIM[0], ymin), XLIM[1] - XLIM[0], QPCR_LOD - ymin,
                             color="#dcdcdc", alpha=0.6, zorder=1, linewidth=0))
    ax_b.axhline(QPCR_LOD, color="#555", lw=1.2, linestyle="--", zorder=2)
    ax_b.text(XLIM[1] - 0.05, QPCR_LOD + 0.20, "LOD",
              fontsize=11, va="bottom", ha="right", color="#444", style="italic")
    b_plot["plot_y"] = np.where(b_plot["above_lod"], b_plot["mean"], QPCR_LOD - 0.05)
    ax_b.plot(b_plot["month"], b_plot["plot_y"], "-", color=COLOR_INVADER,
              lw=2.6, alpha=0.5, zorder=3)
    above = b_plot[b_plot["above_lod"]]
    if len(above) > 0:
        ax_b.errorbar(above["month"], above["mean"], yerr=above["sd"], fmt="o",
                      color=COLOR_INVADER, ms=16, capsize=6, capthick=1.6, lw=1.6, zorder=4,
                      markerfacecolor=COLOR_INVADER, markeredgecolor="white", markeredgewidth=2.0)
    below = b_plot[~b_plot["above_lod"]]
    if len(below) > 0:
        ax_b.scatter(below["month"], [QPCR_LOD - 0.05] * len(below), s=190, marker="o",
                     facecolors="white", edgecolors=COLOR_INVADER, linewidths=2.2, zorder=4)
    peak_row = b[b["month"] == 2].iloc[0]
    ax_b.annotate(f"★ peak  {peak_row['mean']:.2f} log$_{{10}}$ copies/g",
                  xy=(2, peak_row["mean"]), xytext=(2.6, peak_row["mean"] + 0.62),
                  fontsize=13.5, color=COLOR_INVADER, fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color=COLOR_INVADER, lw=1.2))
    ax_b.text(0.5, 0.40, "below detection limit", fontsize=11,
              color="#666", style="italic", va="center")
    style_track(ax_b, "b", ymin, ymax, "Fusarium qPCR\nlog$_{10}$ copies/g",
                ylabel_color=COLOR_INVADER)
    ax_b.set_yticks([0, 1, 2, 3, 4, 5])

    # ============ Track C — Community alpha ============
    c16 = c[c["marker_kind"] == "16S"].sort_values("month")
    cits = c[c["marker_kind"] == "ITS"].sort_values("month")
    ymin = -0.2
    ymax = (max(c16["shannon_mean"].max() + c16["shannon_sd"].fillna(0).max(),
                cits["shannon_mean"].max() + cits["shannon_sd"].fillna(0).max()) * 1.20)
    shade_phases(ax_c, ymin, ymax)
    ax_c.errorbar(c16["month"], c16["shannon_mean"], yerr=c16["shannon_sd"], fmt="o-",
                  color=COLOR_16S, lw=3.2, ms=15, capsize=5, capthick=1.6, zorder=4,
                  markerfacecolor=COLOR_16S, markeredgecolor="white", markeredgewidth=1.8,
                  label="16S Shannon  ●")
    ax_c.errorbar(cits["month"], cits["shannon_mean"], yerr=cits["shannon_sd"], fmt="^-",
                  color=COLOR_ITS, lw=3.2, ms=15, capsize=5, capthick=1.6, zorder=4,
                  markerfacecolor=COLOR_ITS, markeredgecolor="white", markeredgewidth=1.8,
                  label="ITS Shannon  ▲")
    peak16s_y = c16[c16["month"] == 2]["shannon_mean"].iloc[0]
    ax_c.annotate("16S peak", xy=(2, peak16s_y), xytext=(2.7, peak16s_y + 0.28),
                  fontsize=13, color=COLOR_16S, fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color=COLOR_16S, lw=1.0))
    ax_c.legend(loc="upper right", frameon=False, fontsize=12.5)
    style_track(ax_c, "c", ymin, ymax, "Shannon\nindex")
    ax_c.tick_params(axis="y", labelsize=12)

    # ============ Track D — Cross-kingdom sync ============
    ymin, ymax = 0, max(d["res_mean"].max() + d["res_sd"].fillna(0).max(), 0.40) * 1.12
    shade_phases(ax_d, ymin, ymax)
    ax_d.add_patch(Rectangle((XLIM[0], 0), XLIM[1] - XLIM[0], SYNC_THRESHOLD,
                             color="#E6E0F2", alpha=0.5, zorder=1, linewidth=0))
    ax_d.axhline(SYNC_THRESHOLD, color="#555", lw=1.2, linestyle="--", zorder=2)
    ax_d.errorbar(d["month"], d["res_mean"], yerr=d["res_sd"], fmt="o-",
                  color=COLOR_SYNC, lw=3.4, ms=15, capsize=5, capthick=1.6, zorder=4,
                  markerfacecolor=COLOR_SYNC, markeredgecolor="white", markeredgewidth=1.8)
    ax_d.text(XLIM[1] - 0.05, SYNC_THRESHOLD + 0.008, "synchrony threshold",
              fontsize=11, va="bottom", ha="right", color="#444", style="italic")
    style_track(ax_d, "d", ymin, ymax, "Procrustes\nresidual", ylabel_color=COLOR_SYNC)

    # ============ X axis (bottom track only) ============
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(EVEN_MONTHS)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", direction="in", length=4, color="#999")
    ax_d.set_xticks(EVEN_MONTHS)
    ax_d.set_xticklabels([f"{m} M" for m in EVEN_MONTHS], fontsize=14)
    ax_d.set_xlabel("Storage time", fontsize=15, labelpad=8, fontweight="bold")

    # ============ Title ============
    fig.text(0.5, 0.972, "Dual-phase senescence clock",
             ha="center", fontsize=21, fontweight="bold", color="#111")
    fig.text(0.5, 0.951, "host-driven invasion window  →  kingdom decoupling",
             ha="center", fontsize=14, style="italic", color="#555")

    out_base = OUT / "Fig10_dual_phase_synthesis_poster"
    fig.savefig(out_base.with_suffix(".png"), dpi=400)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {out_base}.png + .pdf")


if __name__ == "__main__":
    main()
