#!/usr/bin/env python3
"""
Fig 2 Panel A — family-level ITS succession (even-month) — POSTER edition.

Same data/colors as regen_figures.fig2_succession() Panel A, restyled:
  - NO title / no (A) tag / NO PERMANOVA box (added on the poster if needed)
  - bars grouped by storage month: flush within a month, clean gap between
    months, and NO floating edge margins (the "띄워져 보이는" problem)
  - two-level x axis: small R1/R2/R3 + large 0/2/4/6 M + "Storage month"
  - larger legend + bigger fonts (1-2 m viewing distance)

Reads LOCAL data under Manuscript_figures/_data (no server paths).
Output: Manuscript_figures/Main/Fig2_succession_panelA_poster.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Manuscript_figures" / "_data"
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

EVEN_G = [1, 3, 5, 7]
G2M = {1: 0, 3: 2, 5: 4, 7: 6}

FAMILY_COLORS = {
    "Aspergillaceae": "#e76f8d",
    "Wallemiaceae": "#3a9d6b",
    "Nectriaceae": "#dfc442",
    "Cladosporiaceae": "#5c8fc1",
    "Cuniculitremaceae": "#1f3a93",
    "Incertae sedis": "#a05fa5",
    "Pleosporaceae": "#a8c8dd",
    "Other": "#bdbdbd",
}
FAM_ORDER = ["Aspergillaceae", "Wallemiaceae", "Nectriaceae", "Cladosporiaceae",
             "Cuniculitremaceae", "Incertae sedis", "Pleosporaceae", "Other"]


# ---------- helpers (copied from regen_figures.py) ----------
def load_feature_table(path):
    return pd.read_csv(path, sep="\t", skiprows=1, index_col=0)


def load_taxonomy(path):
    df = pd.read_csv(path, sep="\t")
    df.set_index("Feature ID", inplace=True)
    return df


def parse_sample_id(sid):
    if "_G" in sid and "_R" in sid:
        return int(sid.split("_G")[1].split("_R")[0]), int(sid.split("_R")[1])
    return None, None


def get_family(taxon_str):
    if pd.isna(taxon_str) or taxon_str == "Unassigned":
        return "Unassigned"
    parts = [p.strip() for p in taxon_str.split(";")]
    f_part = next((p for p in parts if p.startswith("f__")), None)
    if f_part and len(f_part) > 3:
        return f_part[3:]
    if "Incertae_sedis" in taxon_str or "Incertae sedis" in taxon_str:
        return "Incertae sedis"
    return "Unclassified"


def apply_filters_ITS(table, tax):
    table = table[table.sum(axis=1) >= 5]
    fungi = tax[tax["Taxon"].str.startswith("k__Fungi", na=False)].index
    return table.loc[table.index.intersection(fungi)]


def agg_by_family(table, tax):
    asv2f = tax.loc[table.index, "Taxon"].apply(get_family)
    return table.groupby(asv2f).sum()


def evenmonth_subset(table):
    return table[[c for c in table.columns if "_G" in c and parse_sample_id(c)[0] in EVEN_G]]


def order_evenmonth_samples(cols):
    return [c for *_, c in sorted(parse_sample_id(c) + (c,) for c in cols)]


# ---------- build ----------
def main():
    tab = load_feature_table(DATA / "ITS_feature-table-dada2.txt")
    tax = load_taxonomy(DATA / "ITS_taxonomy.tsv")
    tab = apply_filters_ITS(tab, tax)
    tab = evenmonth_subset(tab)
    cols = order_evenmonth_samples(tab.columns)
    tab = tab[cols]
    tab_f = agg_by_family(tab, tax)
    rel = tab_f / tab_f.sum(axis=0) * 100
    fam_present = [f for f in FAM_ORDER[:-1] if f in rel.index]
    rel_plot = rel.loc[fam_present].copy()
    rel_plot.loc["Other"] = rel.drop(fam_present, errors="ignore").sum(axis=0)

    # x positions: flush within month, gap between months, no edge margin
    GROUP_GAP = 0.9
    reps_by_g = {}
    for c in cols:
        g, r = parse_sample_id(c)
        reps_by_g.setdefault(g, []).append((r, c))
    xpos, rep_labels, group_centers, group_labels = [], [], [], []
    cursor = 0.0
    for g in sorted(reps_by_g):
        start = cursor
        for r, _c in sorted(reps_by_g[g]):
            xpos.append(cursor)
            rep_labels.append(f"R{r}")
            cursor += 1.0
        group_centers.append((start + cursor - 1.0) / 2.0)
        group_labels.append(f"{G2M[g]} M")
        cursor += GROUP_GAP

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    bottom = np.zeros(len(cols))
    for fam in fam_present + ["Other"]:
        if fam not in rel_plot.index:
            continue
        ax.bar(xpos, rel_plot.loc[fam].values, bottom=bottom, label=fam,
               color=FAMILY_COLORS.get(fam, "#cccccc"),
               edgecolor="white", linewidth=0.6, width=1.0)
        bottom += rel_plot.loc[fam].values

    # y axis
    ax.set_ylabel("Relative abundance (%)", fontsize=16, fontweight="bold", labelpad=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="y", labelsize=13, width=1.0, length=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # kill floating edges
    ax.set_xlim(xpos[0] - 0.6, xpos[-1] + 0.6)

    # two-level x axis
    ax.set_xticks(xpos)
    ax.set_xticklabels(rep_labels, fontsize=10.5, color="#666")
    ax.tick_params(axis="x", length=0)
    for xc, lab in zip(group_centers, group_labels):
        ax.text(xc, -0.085, lab, ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=16, fontweight="bold", color="#222", clip_on=False)
    ax.set_xlabel("Storage month", fontsize=16, fontweight="bold", labelpad=34)

    # legend
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.015, 0.5), frameon=False,
                    fontsize=13, title="Fungal family", labelspacing=0.55,
                    handlelength=1.4, handleheight=1.4, borderaxespad=0.0)
    leg.get_title().set_fontsize(14)
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    base = OUT / "Fig2_succession_panelA_poster"
    fig.savefig(base.with_suffix(".png"), dpi=400)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {base}.png + .pdf")


if __name__ == "__main__":
    main()
