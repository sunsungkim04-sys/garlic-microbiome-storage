#!/usr/bin/env python3
"""
Fig 10 — Dual-phase senescence clock (polished v2).

Stacked tracks on common storage-time axis (0/2/4/6M):
  (a) Host integrity      — phenotype severity score 0–4 (n=1 illustrative)
  (b) Invader load        — Fusarium qPCR log copies/g (n=3 ± SD), <LOD band
  (c) Community alpha     — 16S Shannon ● + ITS Shannon ▲ (n=3 ± SD)
  (d) Cross-kingdom sync  — Procrustes per-sample residual (n=3 ± SD)

Design polish:
  - dedicated Phase header bar above all tracks (single annotated band)
  - subtle 2M anchor vertical line crossing every track (Phase 1 peak)
  - Track A: filled area below curve (host degradation ramp)
  - Track B: gray <LOD band + clean above-LOD marker
  - tighter track spacing, larger title + italic subtitle
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA_MAIN = ROOT / "Manuscript_figures" / "_data"
DATA_INVEST = ROOT / "Attachments_investigation"
OUT = ROOT / "Manuscript_figures" / "Main"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

EVEN_MONTHS = [0, 2, 4, 6]
PHASE1 = (1.5, 3.5)
PHASE2 = (3.5, 6.5)

# more saturated phase colors (deeper but still light)
PHASE1_FILL  = "#FBD7B8"   # warm peach
PHASE2_FILL  = "#C9DAEC"   # cool blue
PHASE1_EDGE  = "#A66F3D"
PHASE2_EDGE  = "#3D5A80"

# Tableau-inspired palette
COLOR_HOST    = "#8B4513"
COLOR_HOST_FILL = "#D9B89C"
COLOR_INVADER = "#C0282E"
COLOR_16S     = "#1F77B4"
COLOR_ITS     = "#2CA02C"
COLOR_SYNC    = "#7B5BAE"

SYNC_THRESHOLD = 0.20
QPCR_LOD = 1.0
ANCHOR_X = 2.0   # Phase 1 peak

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
    # phase boundary
    ax.axvline(3.5, color="#444", lw=0.6, linestyle=":", alpha=0.55, zorder=1)
    # 2M anchor — subtle vertical accent
    ax.axvline(ANCHOR_X, color="#666", lw=0.5, linestyle="-", alpha=0.25, zorder=1)


def style_track(ax, panel_letter, ymin, ymax, ylabel, ylabel_color="#333"):
    ax.set_xlim(*XLIM)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, color=ylabel_color, fontsize=10, labelpad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")
    ax.tick_params(colors="#444", labelsize=9)
    ax.grid(True, axis="y", alpha=0.18, linestyle="-", linewidth=0.5, color="#aaa")
    # panel label in a soft circle
    ax.text(0.012, 0.93, panel_letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", color="#222",
            bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec="#888", lw=0.6))


def main():
    a = load_track_a()
    b = load_track_b()
    c = load_track_c()
    d = load_track_d()

    fig = plt.figure(figsize=(8.4, 11.2))
    gs = fig.add_gridspec(
        nrows=5, ncols=1,
        height_ratios=[0.32, 1.0, 1.05, 1.05, 1.05],
        hspace=0.28,
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
    # phase bands
    ax_header.add_patch(Rectangle((PHASE1[0], 0.15), PHASE1[1] - PHASE1[0], 0.7,
                                  color=PHASE1_FILL, ec=PHASE1_EDGE, lw=1.0, zorder=2))
    ax_header.add_patch(Rectangle((PHASE2[0], 0.15), PHASE2[1] - PHASE2[0], 0.7,
                                  color=PHASE2_FILL, ec=PHASE2_EDGE, lw=1.0, zorder=2))
    ax_header.text(2.5, 0.50, "Phase 1\nInvasion window", ha="center", va="center",
                   fontsize=10.5, fontweight="bold", color="#6B3F1F", zorder=3)
    ax_header.text(5.0, 0.50, "Phase 2\nDecoupling", ha="center", va="center",
                   fontsize=10.5, fontweight="bold", color="#2B3F66", zorder=3)
    # anchor pointer at 2M
    ax_header.annotate(
        "anchor",
        xy=(ANCHOR_X, 0.15), xytext=(ANCHOR_X, -0.10),
        ha="center", va="top", fontsize=8.5, color="#A33",
        arrowprops=dict(arrowstyle="-|>", color="#A33", lw=1.2,
                        shrinkA=0, shrinkB=2),
        annotation_clip=False,
    )

    # ============ Track A — Host integrity ============
    ymin, ymax = -0.3, 4.5
    shade_phases(ax_a, ymin, ymax)
    # area fill under the curve (gradient effect via single fill_between)
    ax_a.fill_between(a["month"], 0, a["severity"], color=COLOR_HOST_FILL, alpha=0.55, zorder=2)
    ax_a.plot(a["month"], a["severity"], "-", color=COLOR_HOST, lw=2.4, zorder=3)
    ax_a.plot(a["month"], a["severity"], "o", color=COLOR_HOST, ms=11, zorder=4,
              markerfacecolor=COLOR_HOST, markeredgecolor="white", markeredgewidth=1.4)
    # annotations: tiny score badges to the right of each marker
    for _, row in a.iterrows():
        ax_a.text(row["month"] + 0.18, row["severity"] - 0.05, f"{row['severity']:.1f}",
                  fontsize=8.5, color=COLOR_HOST, fontweight="bold", va="center")
    ax_a.text(0.99, 0.06, "host integrity (illustrative, n=1)", transform=ax_a.transAxes,
              fontsize=8, ha="right", va="bottom", style="italic", color="#777")
    style_track(ax_a, "a", ymin, ymax, "Phenotype\nseverity score\n(0 – 4)",
                ylabel_color=COLOR_HOST)
    ax_a.set_yticks([0, 1, 2, 3, 4])

    # ============ Track B — Invader load ============
    b_plot = b.copy()
    b_plot["above_lod"] = b_plot["mean"] >= QPCR_LOD

    ymin, ymax = -0.3, 5.5
    shade_phases(ax_b, ymin, ymax)
    # <LOD shaded band
    ax_b.add_patch(Rectangle((XLIM[0], ymin), XLIM[1] - XLIM[0], QPCR_LOD - ymin,
                             color="#ddd", alpha=0.55, zorder=1, linewidth=0))
    ax_b.axhline(QPCR_LOD, color="#666", lw=0.9, linestyle="--", zorder=2)
    ax_b.text(XLIM[1] - 0.05, QPCR_LOD + 0.18, "LOD",
              fontsize=8.5, va="bottom", ha="right", color="#444", style="italic")

    # connecting line (clipped at LOD for <LOD)
    b_plot["plot_y"] = np.where(b_plot["above_lod"], b_plot["mean"], QPCR_LOD - 0.05)
    ax_b.plot(b_plot["month"], b_plot["plot_y"], "-", color=COLOR_INVADER,
              lw=2.0, alpha=0.5, zorder=3)
    above = b_plot[b_plot["above_lod"]]
    if len(above) > 0:
        ax_b.errorbar(above["month"], above["mean"], yerr=above["sd"], fmt="o",
                      color=COLOR_INVADER, ms=12, capsize=5, capthick=1.2, lw=1.2, zorder=4,
                      markerfacecolor=COLOR_INVADER, markeredgecolor="white", markeredgewidth=1.4)
    below = b_plot[~b_plot["above_lod"]]
    if len(below) > 0:
        ax_b.scatter(below["month"], [QPCR_LOD - 0.05] * len(below), s=130, marker="o",
                     facecolors="white", edgecolors=COLOR_INVADER, linewidths=1.6, zorder=4)
    # peak annotation
    peak_row = b[b["month"] == 2].iloc[0]
    ax_b.annotate(f"★ peak  {peak_row['mean']:.2f} (Cq ≈ 15, nominal)",
                  xy=(2, peak_row["mean"]), xytext=(2.55, peak_row["mean"] + 0.55),
                  fontsize=10, color=COLOR_INVADER, fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color=COLOR_INVADER, lw=0.9))
    # <LOD label on the grey band (one label, not per-marker)
    ax_b.text(0.5, 0.35, "near-baseline (Cq ~30–34, <LOD-style)", fontsize=8.5,
              color="#666", style="italic", va="center")
    style_track(ax_b, "b", ymin, ymax,
                "Fusarium qPCR\nnominal log$_{10}$ copies/g\n(n = 3, ± SD)",
                ylabel_color=COLOR_INVADER)
    ax_b.set_yticks([0, 1, 2, 3, 4, 5])

    # ============ Track C — Community alpha ============
    c16 = c[c["marker_kind"] == "16S"].sort_values("month")
    cits = c[c["marker_kind"] == "ITS"].sort_values("month")
    ymin = -0.2
    ymax = (max(c16["shannon_mean"].max() + c16["shannon_sd"].fillna(0).max(),
                cits["shannon_mean"].max() + cits["shannon_sd"].fillna(0).max()) * 1.18)
    shade_phases(ax_c, ymin, ymax)
    ax_c.errorbar(c16["month"], c16["shannon_mean"], yerr=c16["shannon_sd"], fmt="o-",
                  color=COLOR_16S, lw=2.2, ms=11, capsize=4, capthick=1.2, zorder=4,
                  markerfacecolor=COLOR_16S, markeredgecolor="white", markeredgewidth=1.3,
                  label="16S Shannon  ●")
    ax_c.errorbar(cits["month"], cits["shannon_mean"], yerr=cits["shannon_sd"], fmt="^-",
                  color=COLOR_ITS, lw=2.2, ms=11, capsize=4, capthick=1.2, zorder=4,
                  markerfacecolor=COLOR_ITS, markeredgecolor="white", markeredgewidth=1.3,
                  label="ITS Shannon  ▲")
    # callout for 16S 2M peak
    peak16s_y = c16[c16["month"] == 2]["shannon_mean"].iloc[0]
    ax_c.annotate(f"16S peak (KW p = 0.016)",
                  xy=(2, peak16s_y), xytext=(2.7, peak16s_y + 0.25),
                  fontsize=9, color=COLOR_16S, fontweight="bold", style="italic",
                  arrowprops=dict(arrowstyle="-", color=COLOR_16S, lw=0.8))
    ax_c.legend(loc="upper right", frameon=False, fontsize=9.5)
    style_track(ax_c, "c", ymin, ymax,
                "Shannon index\n(n = 3, ± SD)")
    ax_c.tick_params(axis="y", labelsize=9)

    # ============ Track D — Cross-kingdom sync ============
    ymin, ymax = 0, max(d["res_mean"].max() + d["res_sd"].fillna(0).max(), 0.40) * 1.10
    shade_phases(ax_d, ymin, ymax)
    # synchrony threshold shaded zone (below = synced)
    ax_d.add_patch(Rectangle((XLIM[0], 0), XLIM[1] - XLIM[0], SYNC_THRESHOLD,
                             color="#E6E0F2", alpha=0.45, zorder=1, linewidth=0))
    ax_d.axhline(SYNC_THRESHOLD, color="#555", lw=0.9, linestyle="--", zorder=2)
    ax_d.errorbar(d["month"], d["res_mean"], yerr=d["res_sd"], fmt="o-",
                  color=COLOR_SYNC, lw=2.4, ms=11, capsize=4, capthick=1.2, zorder=4,
                  markerfacecolor=COLOR_SYNC, markeredgecolor="white", markeredgewidth=1.3)
    ax_d.text(XLIM[1] - 0.05, SYNC_THRESHOLD + 0.005,
              f"synchrony threshold (residual < {SYNC_THRESHOLD:.2f})",
              fontsize=8.5, va="bottom", ha="right", color="#444", style="italic")
    # Mantel statistic
    ax_d.text(0.011, 0.04,
              "Mantel ρ = 0.82  |  partial r = 0.65 (free perm p = 0.0001;\nwithin-month restricted perm NS, p > 0.05)",
              transform=ax_d.transAxes, fontsize=8.0, color=COLOR_SYNC, style="italic",
              fontweight="bold", va="bottom")
    style_track(ax_d, "d", ymin, ymax,
                "Procrustes\nresidual\n(n = 3, ± SD)",
                ylabel_color=COLOR_SYNC)

    # ============ X axis (only on bottom track) ============
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(EVEN_MONTHS)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", direction="in", length=3, color="#aaa")
    ax_d.set_xticks(EVEN_MONTHS)
    ax_d.set_xticklabels([f"{m} M" for m in EVEN_MONTHS], fontsize=11)
    ax_d.set_xlabel("Storage time", fontsize=11, labelpad=6)

    # ============ Title ============
    fig.text(0.5, 0.965,
             "Dual-phase senescence clock",
             ha="center", fontsize=15, fontweight="bold", color="#222")
    fig.text(0.5, 0.945,
             "proposed host-driven invasion window  →  kingdom decoupling",
             ha="center", fontsize=11, style="italic", color="#555")

    out_base = OUT / "Fig10_dual_phase_synthesis"
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {out_base}.png + .pdf")


if __name__ == "__main__":
    main()
