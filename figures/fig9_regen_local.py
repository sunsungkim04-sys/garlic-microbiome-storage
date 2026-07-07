#!/usr/bin/env python3
"""
fig9_regen_local.py — Fig 9 (single-panel, 2026-05-27 refocus + 2026-05-27 framing-trim).

Single clear message: **Erwiniaceae vs Burkholderia commodity contrast**.
Garlic shows a 4M Erwiniaceae soft-rot–competent bloom; the onion soft-rot
agent Burkholderia is absent across all storage months.

Earlier 3-panel layout (Fusarium qPCR vs ITS dual-axis; row-normalised lag
heatmap) was removed because (i) the qPCR-vs-ITS method-mismatch story is
already the dedicated content of Fig 8, and (ii) the fungal pioneer → bacterial
soft-rot temporal succession is shown more cleanly in Fig 10's dual-phase
synthesis. Combining all three into one figure conflated commodity contrast
with cross-kingdom temporal coupling.

Framing trim (2026-05-27): Phase 1/2 header bar + anchor pointer also removed
to keep Fig 9 self-contained — the Phase 1/2 framework belongs to Fig 10
(the synthesis figure) and forcing it onto every panel risks "thesis
over-impose" framing critique. Fig 9 now stands alone as a clean commodity
contrast observation; the Phase context is established in text + Fig 10.

Reads from ../../Manuscript_figures/_data/ and writes
Fig9_Erwiniaceae_vs_Burkholderia.{png,pdf} to ../../Manuscript_figures/Main/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "Manuscript_figures" / "_data"
OUT = ROOT / "Manuscript_figures" / "Main"

EVEN_G = [1, 3, 5, 7]
G2M = {1: 0, 3: 2, 5: 4, 7: 6}
MONTHS = [0, 2, 4, 6]
XLIM = (-0.6, 6.7)

C_ERW  = "#2C5FA8"
C_BURK = "#9A9A9A"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_g(col):
    return int(col.split("_G")[1].split("_R")[0])


def even_cols(df):
    return [c for c in df.columns if "_G" in c and parse_g(c) in EVEN_G]


def family_relpct(table_path, tax_path, families):
    ft = pd.read_csv(table_path, sep="\t", skiprows=1, index_col=0)
    tax = pd.read_csv(tax_path, sep="\t").set_index("Feature ID")
    T = tax["Taxon"]
    keep = T[(T.str.startswith("d__Bacteria") | T.str.startswith("d__Archaea"))
             & ~T.str.contains("Chloroplast|Mitochondria", case=False, na=False)].index
    ft = ft.loc[ft.index.intersection(keep)]
    ft = ft[ft.sum(axis=1) >= 5]

    def fam(t):
        if pd.isna(t):
            return "NA"
        parts = [p.strip() for p in t.split(";")]
        f = next((p for p in parts if p.startswith("f__")), None)
        return f[3:] if f and len(f) > 3 else "Unclassified"

    agg = ft.groupby(tax.loc[ft.index, "Taxon"].apply(fam)).sum()
    rel = agg / agg.sum(axis=0) * 100.0
    cols = even_cols(rel)
    out = {}
    for famname in families:
        row = rel.loc[famname] if famname in rel.index else pd.Series(0.0, index=cols)
        out[famname] = {G2M[G]: row[[c for c in cols if parse_g(c) == G]].agg(["mean", "std", "min", "max"])
                        for G in EVEN_G}
    return out


def main():
    erw = "Erwiniaceae"
    burk = "Burkholderiaceae"
    fam16 = family_relpct(DATA / "16S_feature-table-dada2.txt",
                          DATA / "16S_taxonomy.tsv",
                          [erw, burk])

    def mvec(d):
        return np.array([d[m]["mean"] for m in MONTHS])

    def svec(d):
        return np.array([d[m]["std"] for m in MONTHS])

    erw_m, erw_s = mvec(fam16[erw]), svec(fam16[erw])
    burk_m, burk_s = mvec(fam16[burk]), svec(fam16[burk])

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ymin, ymax = -3, 45

    # Erwiniaceae trajectory (the story)
    ax.errorbar(MONTHS, erw_m, yerr=erw_s, marker="o", ms=12, lw=2.5,
                color=C_ERW, capsize=5, capthick=1.3,
                markerfacecolor=C_ERW, markeredgecolor="white", markeredgewidth=1.4,
                label="Erwiniaceae  (garlic 4M bloom)", zorder=4)
    # Burkholderiaceae trajectory (negative control)
    ax.errorbar(MONTHS, burk_m, yerr=burk_s, marker="s", ms=8, lw=1.6,
                color=C_BURK,
                markerfacecolor=C_BURK, markeredgecolor="white", markeredgewidth=1.0,
                capsize=3, label="Burkholderiaceae  (onion soft-rot agent, absent)",
                zorder=3)

    # 4M peak annotation
    peak = erw_m[MONTHS.index(4)]
    rng = (fam16[erw][4]["min"], fam16[erw][4]["max"])
    ax.annotate(f"★ 4M Erwiniaceae bloom\n{peak:.1f}% (range {rng[0]:.1f}–{rng[1]:.1f}%, n=3)",
                xy=(4, peak), xytext=(4.6, peak - 4),
                fontsize=10.5, color=C_ERW, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_ERW, lw=1.0))

    # 0M / 2M / 6M annotations for Erwiniaceae
    for m, val in zip(MONTHS, erw_m):
        if m == 4:
            continue
        label = "0%" if val < 0.1 else f"{val:.1f}%"
        ax.text(m, val + 1.4, label, ha="center", va="bottom",
                fontsize=8.5, color=C_ERW, fontweight="bold")

    # Burkholderiaceae annotation box
    burk_4m = burk_m[MONTHS.index(4)]
    ax.text(0.985, 0.24,
            "Burkholderiaceae ≈ 0% across all months\n"
            "(genus Burkholderia 0% at every timepoint;\n"
            "onion-type soft-rot signal absent in garlic)",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", lw=0.7))

    # axis styling
    ax.set_xticks(MONTHS)
    ax.set_xticklabels([f"{m} M" for m in MONTHS], fontsize=11)
    ax.set_xlim(*XLIM)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Relative abundance (%) — 16S family level", fontsize=10.5, labelpad=8)
    ax.set_xlabel("Storage time", fontsize=11, labelpad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")
    ax.tick_params(colors="#444", labelsize=10)
    ax.grid(True, axis="y", alpha=0.18, linestyle="-", linewidth=0.5, color="#aaa")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    # caveat footer
    fig.text(0.5, 0.02,
             "16S V4 read length resolves Erwiniaceae to family level only "
             "(Pectobacterium / Erwinia / Pantoea / Brenneria not distinguished).",
             ha="center", fontsize=8.5, style="italic", color="#666")

    # title + subtitle
    fig.text(0.5, 0.965,
             "Erwiniaceae vs Burkholderia commodity contrast",
             ha="center", fontsize=15, fontweight="bold", color="#222")
    fig.text(0.5, 0.935,
             "4M soft-rot–competent window in garlic;  onion-type Burkholderia signal absent",
             ha="center", fontsize=11, style="italic", color="#555")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Fig9_Erwiniaceae_vs_Burkholderia"
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print("wrote", stem.with_suffix(".png"))
    print("Erwiniaceae:", dict(zip(MONTHS, np.round(erw_m, 3))))
    print("Burkholderiaceae:", dict(zip(MONTHS, np.round(burk_m, 3))))


if __name__ == "__main__":
    main()
