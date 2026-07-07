#!/usr/bin/env python3
"""
Fig 2 — ITS even-month top-12 stacked bar — POSTER edition.

Same data/colors as regen_figures.fig1_ITS_stacked(), restyled for a poster:
  - NO title (added separately on the poster)
  - bars grouped by storage month with a clear gap between groups
  - two-level x axis: small R1/R2/R3 under each bar + large 0/2/4/6 M group
    label, plus a "Storage month" axis title
  - larger legend, bigger fonts overall (1-2 m viewing distance)

Reads the LOCAL copies under Manuscript_figures/_data (no server paths).
Output: Manuscript_figures/Main/Fig2_ITS_stacked_evenmonth_poster.{png,pdf}
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

ITS_COLORS = {
    "Penicillium": "#1f77b4",
    "Wallemia": "#aec7e8",
    "Fusarium": "#ff7f0e",
    "Cladosporium": "#ffbb78",
    "Fungi_gen_Incertae_sedis": "#2ca02c",
    "f__Pleosporaceae": "#98df8a",
    "Kockovaella": "#d62728",
    "Aspergillus": "#ff9896",
    "Staurosphaeria": "#9467bd",
    "o__Hypocreales": "#c5b0d5",
    "Occultifur": "#8c564b",
    "f__Plectosphaerellaceae": "#c49c94",
    "Other": "#dadada",
}


# ---------- helpers (copied from regen_figures.py, server paths removed) ----------
def load_feature_table(path):
    return pd.read_csv(path, sep="\t", skiprows=1, index_col=0)


def load_taxonomy(path):
    df = pd.read_csv(path, sep="\t")
    df.set_index("Feature ID", inplace=True)
    return df


def parse_sample_id(sid):
    if "_G" in sid and "_R" in sid:
        g = int(sid.split("_G")[1].split("_R")[0])
        r = int(sid.split("_R")[1])
        return g, r
    return None, None


def get_genus(taxon_str):
    if pd.isna(taxon_str) or taxon_str == "Unassigned":
        return "Unassigned"
    parts = [p.strip() for p in taxon_str.split(";")]
    g_part = next((p for p in parts if p.startswith("g__")), None)
    if g_part and len(g_part) > 3 and g_part[3:]:
        return g_part[3:]
    f_part = next((p for p in parts if p.startswith("f__")), None)
    if f_part and len(f_part) > 3:
        return f"f__{f_part[3:]}"
    o_part = next((p for p in parts if p.startswith("o__")), None)
    if o_part and len(o_part) > 3:
        return f"o__{o_part[3:]}"
    if "Incertae_sedis" in taxon_str or "Incertae sedis" in taxon_str:
        return "Fungi_gen_Incertae_sedis"
    return "Unclassified"


def prettify_taxon_label(name):
    if name.startswith("f__"):
        return f"{name[3:]} (family)"
    if name.startswith("o__"):
        return f"{name[3:]} (order)"
    if name == "Fungi_gen_Incertae_sedis":
        return "Fungi (incertae sedis)"
    return name


def apply_filters_ITS(table, tax):
    table = table[table.sum(axis=1) >= 5]
    fungi = tax[tax["Taxon"].str.startswith("k__Fungi", na=False)].index
    return table.loc[table.index.intersection(fungi)]


def agg_by_genus(table, tax):
    asv2g = tax.loc[table.index, "Taxon"].apply(get_genus)
    return table.groupby(asv2g).sum()


def evenmonth_subset(table):
    cols = [c for c in table.columns if "_G" in c and parse_sample_id(c)[0] in EVEN_G]
    return table[cols]


def order_evenmonth_samples(cols):
    keyed = sorted((parse_sample_id(c) + (c,) for c in cols))
    return [c for *_, c in keyed]


# ---------- build ----------
def main():
    tab = load_feature_table(DATA / "ITS_feature-table-dada2.txt")
    tax = load_taxonomy(DATA / "ITS_taxonomy.tsv")
    tab = apply_filters_ITS(tab, tax)
    tab = evenmonth_subset(tab)
    cols = order_evenmonth_samples(tab.columns)
    tab = tab[cols]
    tab_g = agg_by_genus(tab, tax)
    rel = tab_g / tab_g.sum(axis=0)
    top12 = rel.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    rel_plot = rel.loc[top12].copy()
    rel_plot.loc["Other"] = rel.drop(top12).sum(axis=0)

    # x positions: group bars by month, leave a gap between groups
    GROUP_GAP = 0.9
    reps_by_g = {}
    for c in cols:
        g, r = parse_sample_id(c)
        reps_by_g.setdefault(g, []).append((r, c))
    xpos, rep_labels, group_centers, group_labels = [], [], [], []
    cursor = 0.0
    for g in sorted(reps_by_g):
        reps = sorted(reps_by_g[g])
        start = cursor
        for r, _c in reps:
            xpos.append(cursor)
            rep_labels.append(f"R{r}")
            cursor += 1.0
        group_centers.append((start + cursor - 1.0) / 2.0)
        group_labels.append(f"{G2M[g]} M")
        cursor += GROUP_GAP

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    bottom = np.zeros(len(cols))
    for genus in list(rel_plot.index):
        color = ITS_COLORS.get(genus, "#cccccc")
        vals = rel_plot.loc[genus].values
        ax.bar(xpos, vals, bottom=bottom, label=prettify_taxon_label(genus),
               color=color, edgecolor="white", linewidth=0.6, width=1.0)
        bottom += vals

    # y axis
    ax.set_ylabel("Relative abundance", fontsize=16, fontweight="bold", labelpad=10)
    ax.set_ylim(0, 1.001)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=13, width=1.0, length=4)
    ax.margins(x=0.01)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # two-level x axis: small replicate ticks + big month-group labels below
    ax.set_xticks(xpos)
    ax.set_xticklabels(rep_labels, fontsize=10.5, color="#666")
    ax.tick_params(axis="x", length=0)
    ymin = ax.get_ylim()[0]
    for xc, lab in zip(group_centers, group_labels):
        ax.text(xc, -0.085, lab, ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=16, fontweight="bold", color="#222", clip_on=False)
    ax.set_xlabel("Storage month", fontsize=16, fontweight="bold", labelpad=34)

    # legend — larger, titled
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.015, 0.5), frameon=False,
                    fontsize=13, title="Top-12 fungal genera", labelspacing=0.55,
                    handlelength=1.4, handleheight=1.4, borderaxespad=0.0)
    leg.get_title().set_fontsize(14)
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    base = OUT / "Fig2_ITS_stacked_evenmonth_poster"
    fig.savefig(base.with_suffix(".png"), dpi=400)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {base}.png + .pdf")


if __name__ == "__main__":
    main()
