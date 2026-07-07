#!/usr/bin/env python3
"""
regen_figures.py — v11.4.3 figure regeneration as vector PDFs.

Style matched to existing PNGs (v11.3.x).
Output: /home1/minseo1101/garlic_project/manuscript/figures/v11.4_regen/
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

BASE = Path("/home1/minseo1101/garlic_project")
DATA = BASE / "data/qiime2_reanalysis"
OUT = BASE / "manuscript/figures/v11.4_regen"
OUT.mkdir(parents=True, exist_ok=True)

# Even-month mapping
EVEN_G = [1, 3, 5, 7]
G2M = {1: 0, 3: 2, 5: 4, 7: 6}
MONTH_COLORS_VIRIDIS = {0: "#440154", 2: "#3b528b", 4: "#21918c", 6: "#fde725"}

# ITS Top-12 genus palette (matched to Fig 1 PNG)
ITS_COLORS = {
    "Penicillium": "#1f77b4",       # dark blue
    "Wallemia": "#aec7e8",          # light blue
    "Fusarium": "#ff7f0e",          # orange
    "Cladosporium": "#ffbb78",      # peach
    "Fungi_gen_Incertae_sedis": "#2ca02c",  # dark green
    "f__Pleosporaceae": "#98df8a",  # light green
    "Kockovaella": "#d62728",       # red
    "Aspergillus": "#ff9896",       # pink-red
    "Staurosphaeria": "#9467bd",    # purple
    "o__Hypocreales": "#c5b0d5",    # light purple
    "Occultifur": "#8c564b",        # brown
    "f__Plectosphaerellaceae": "#c49c94",  # tan
    "Other": "#dadada",
}

# 16S Top-12 genus palette (matched to Fig 3A PNG)
S16_COLORS = {
    "Escherichia-Shigella": "#1f77b4",
    "Staphylococcus": "#aec7e8",
    "f__Erwiniaceae": "#ffbb78",
    "Bacillus": "#98df8a",
    "Saccharopolyspora": "#d62728",
    "Brevibacterium": "#9467bd",
    "Rhodococcus": "#8c564b",
    "Paenibacillus": "#c49c94",
    "Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium": "#f7b6d2",
    "Leifsonia": "#c7c7c7",
    "Pseudomonas": "#bcbd22",
    "f__Micrococcaceae": "#17becf",
    "Other": "#dadada",
}

# Family palette for Fig 2 (matched)
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

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_both(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  ✅ {name}.{{png,pdf}}")


def parse_sample_id(sid):
    if "_G" in sid and "_R" in sid:
        g = int(sid.split("_G")[1].split("_R")[0])
        r = int(sid.split("_R")[1])
        full_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        return g, r, full_map.get(g)
    return None, None, None


def load_feature_table(path):
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    return df


def load_taxonomy(path):
    df = pd.read_csv(path, sep="\t")
    df.set_index("Feature ID", inplace=True)
    return df


def get_genus(taxon_str):
    if pd.isna(taxon_str) or taxon_str == "Unassigned":
        return "Unassigned"
    parts = [p.strip() for p in taxon_str.split(";")]
    g_part = next((p for p in parts if p.startswith("g__")), None)
    if g_part and len(g_part) > 3:
        val = g_part[3:]
        if val:
            # Strip italic markers
            return val
    # Fall back to family/order/incertae
    f_part = next((p for p in parts if p.startswith("f__")), None)
    if f_part and len(f_part) > 3:
        return f"f__{f_part[3:]}"
    o_part = next((p for p in parts if p.startswith("o__")), None)
    if o_part and len(o_part) > 3:
        return f"o__{o_part[3:]}"
    # Incertae sedis handling
    if "Incertae_sedis" in taxon_str or "Incertae sedis" in taxon_str:
        return "Fungi_gen_Incertae_sedis"
    return "Unclassified"


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


def prettify_taxon_label(name):
    """Display-only cleanup of raw QIIME taxonomy strings for legends.
    Internal grouping/color keys keep the raw form; only the printed label changes."""
    if name.startswith("f__"):
        return f"{name[3:]} (family)"
    if name.startswith("o__"):
        return f"{name[3:]} (order)"
    if name == "Fungi_gen_Incertae_sedis":
        return "Fungi (incertae sedis)"
    return name


def apply_filters_ITS(table, tax):
    table = table[table.sum(axis=1) >= 5]
    fungi_asvs = tax[tax["Taxon"].str.startswith("k__Fungi", na=False)].index
    table = table.loc[table.index.intersection(fungi_asvs)]
    return table


def apply_filters_16S(table, tax):
    table = table[table.sum(axis=1) >= 5]
    T = tax["Taxon"]
    keep = T[(T.str.startswith("d__Bacteria") | T.str.startswith("d__Archaea"))
             & ~T.str.contains("Chloroplast|Mitochondria", case=False, na=False)].index
    table = table.loc[table.index.intersection(keep)]
    return table


def rarefy(table, depth, seed=42):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(0, index=table.index, columns=table.columns, dtype=int)
    for col in table.columns:
        counts = table[col].values.astype(int)
        total = counts.sum()
        if total < depth:
            out[col] = 0
            continue
        probs = counts / total
        sampled = rng.multinomial(depth, probs)
        out[col] = sampled
    out = out.loc[(out.sum(axis=1) > 0), (out.sum(axis=0) >= depth)]
    return out


def agg_by_genus(table, tax):
    asv2g = tax.loc[table.index, "Taxon"].apply(get_genus)
    return table.groupby(asv2g).sum()


def agg_by_family(table, tax):
    asv2f = tax.loc[table.index, "Taxon"].apply(get_family)
    return table.groupby(asv2f).sum()


def evenmonth_subset(table):
    cols = [c for c in table.columns if "_G" in c and parse_sample_id(c)[0] in EVEN_G]
    return table[cols]


def order_evenmonth_samples(cols):
    """Order columns: G1_R1, G1_R2, G1_R3, G3_R1, ... G7_R3"""
    keyed = []
    for c in cols:
        g, r, _ = parse_sample_id(c)
        keyed.append((g, r, c))
    keyed.sort()
    return [c for _, _, c in keyed]


# ============================================================
# Fig 1: ITS evenmonth top-12 stacked bar (style: 0-1.0 scale, G_R labels)
# ============================================================
def fig1_ITS_stacked():
    print("Fig 1 — ITS evenmonth top-12 stacked bar")
    tab = load_feature_table(DATA / "ITS_old/exported/feature-table-dada2.txt")  # pre-freq; apply_filters applies freq>=5
    tax = load_taxonomy(DATA / "ITS_old/exported/taxonomy.tsv")
    tab = apply_filters_ITS(tab, tax)
    # composition = non-rarefied relative abundance (matches text; rarefaction reserved for diversity)
    tab = evenmonth_subset(tab)
    cols = order_evenmonth_samples(tab.columns)
    tab = tab[cols]
    tab_g = agg_by_genus(tab, tax)
    rel = tab_g / tab_g.sum(axis=0)  # 0–1.0
    top12 = rel.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    rel_plot = rel.loc[top12].copy()
    rel_plot.loc["Other"] = rel.drop(top12).sum(axis=0)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    labels = [f"{G2M[parse_sample_id(c)[0]]}M_R{parse_sample_id(c)[1]}" for c in cols]
    bottom = np.zeros(len(cols))
    for genus in list(rel_plot.index):
        color = ITS_COLORS.get(genus, "#cccccc")
        vals = rel_plot.loc[genus].values
        ax.bar(range(len(cols)), vals, bottom=bottom, label=prettify_taxon_label(genus), color=color,
               edgecolor="white", linewidth=0.3, width=0.85)
        bottom += vals
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative abundance")
    ax.set_ylim(0, 1.01)
    _last = None
    for _i, _c in enumerate(cols):
        _g = parse_sample_id(_c)[0]
        if _last is not None and _g != _last:
            ax.axvline(_i - 0.5, color='black', linewidth=1.2)
        _last = _g
    ax.text(0.02, 0.02, "PERMANOVA (Jaccard)\nF = 4.44, p = 0.001, PERMDISP NS", transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92))
    ax.set_title("ITS top-12 fungal genera — even-month (n = 12)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    plt.tight_layout()
    save_both(fig, "Fig1_ITS_stacked_evenmonth")


# ============================================================
# Fig 2: Fungal succession framework — (A) family 7-month + (B) genus trajectory evenmonth
# ============================================================
def fig2_succession():
    print("Fig 2 — Fungal succession framework (even-month main frame)")
    tab = load_feature_table(DATA / "ITS_old/exported/feature-table-dada2.txt")  # pre-freq; apply_filters applies freq>=5
    tax = load_taxonomy(DATA / "ITS_old/exported/taxonomy.tsv")
    tab = apply_filters_ITS(tab, tax)
    # composition = non-rarefied relative abundance (matches text; rarefaction reserved for diversity)
    # Even-month subset for BOTH panels (matches caption "even-month main frame, n=3 per timepoint")
    tab_em = evenmonth_subset(tab)
    cols_em = order_evenmonth_samples(tab_em.columns)
    tab_em = tab_em[cols_em]
    # Panel A: family-level even-month frame
    tab_f = agg_by_family(tab_em, tax)
    rel_f = tab_f / tab_f.sum(axis=0) * 100
    fam_order = ["Aspergillaceae", "Wallemiaceae", "Nectriaceae", "Cladosporiaceae",
                 "Cuniculitremaceae", "Incertae sedis", "Pleosporaceae", "Other"]
    fam_present = [f for f in fam_order[:-1] if f in rel_f.index]
    rel_top_f = rel_f.loc[fam_present].copy()
    rel_top_f.loc["Other"] = rel_f.drop(fam_present, errors="ignore").sum(axis=0)
    # Panel B: dominant genera trajectory (even-month only)
    tab_g_em = agg_by_genus(tab_em, tax)
    rel_g = tab_g_em / tab_g_em.sum(axis=0) * 100
    sample_month = {c: G2M[parse_sample_id(c)[0]] for c in rel_g.columns}
    targets = ["Penicillium", "Wallemia", "Fusarium", "Aspergillus"]
    target_colors = {"Penicillium": "#e76f8d", "Wallemia": "#3a9d6b",
                     "Fusarium": "#dfc442", "Aspergillus": "#5e4b8b"}
    trajectory = []
    for m in [0, 2, 4, 6]:
        cols_m = [c for c, mm in sample_month.items() if mm == m]
        if not cols_m:
            continue
        for g in targets:
            vals = rel_g.loc[g, cols_m].values if g in rel_g.index else np.zeros(len(cols_m))
            trajectory.append((m, g, np.mean(vals), np.std(vals)))
    md = pd.DataFrame(trajectory, columns=["month", "genus", "mean", "sd"])

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel A
    x_positions = np.arange(len(cols_em))
    bottom = np.zeros(len(cols_em))
    for fam in fam_present + ["Other"]:
        if fam not in rel_top_f.index:
            continue
        color = FAMILY_COLORS.get(fam, "#cccccc")
        vals = rel_top_f.loc[fam].values
        ax1.bar(x_positions, vals, bottom=bottom, label=fam, color=color,
                edgecolor="white", linewidth=0.4, width=0.85)
        bottom += vals
    # Month group labels at bottom
    month_centers = {}
    for i, c in enumerate(cols_em):
        m = G2M[parse_sample_id(c)[0]]
        month_centers.setdefault(m, []).append(i)
    ax1.set_xticks([np.mean(v) for v in month_centers.values()])
    ax1.set_xticklabels([f"{m}M" for m in month_centers.keys()])
    ax1.set_ylabel("Relative abundance (%)")
    ax1.set_ylim(0, 100)
    _last = None
    for _i, _c in enumerate(cols_em):
        _g = parse_sample_id(_c)[0]
        if _last is not None and _g != _last:
            ax1.axvline(_i - 0.5, color='black', linewidth=1.2)
        _last = _g
    ax1.text(0.02, 0.02, "PERMANOVA (Jaccard)\nF = 4.44, p = 0.001, PERMDISP NS", transform=ax1.transAxes, va="bottom", ha="left", fontsize=8.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92))
    ax1.set_xlabel("Storage month  (even-month main frame, n = 3 per timepoint)")
    ax1.text(-0.06, 1.02, "(A)", transform=ax1.transAxes, fontsize=14, fontweight="bold")
    # Legend below
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False,
               fontsize=8, title="Family")

    # Panel B
    for g in targets:
        sub = md[md["genus"] == g]
        ax2.errorbar(sub["month"], sub["mean"], yerr=sub["sd"], label=g,
                     color=target_colors[g], marker="o", linewidth=2, capsize=3, markersize=7)
    ax2.set_xlabel("Storage month")
    ax2.set_ylabel("Relative abundance (%)")
    ax2.set_ylim(-5, 110)
    ax2.set_xticks([0, 2, 4, 6])
    ax2.set_xticklabels(["0M", "2M", "4M", "6M"])
    ax2.legend(loc="center right", frameon=True, framealpha=0.9, fontsize=9)
    ax2.text(-0.08, 1.02, "(B)", transform=ax2.transAxes, fontsize=14, fontweight="bold")
    ax2.grid(linestyle=":", alpha=0.4)
    # Annotations
    pen_2m = md[(md["genus"] == "Penicillium") & (md["month"] == 2)]
    if not pen_2m.empty:
        ax2.annotate("Penicillium dominant", xy=(2, pen_2m["mean"].iloc[0]),
                     xytext=(2.3, pen_2m["mean"].iloc[0] - 5),
                     color="#e76f8d", fontsize=10, style="italic", fontweight="bold")
    fus_peak = md[(md["genus"] == "Fusarium") & (md["month"] == 4)]
    if not fus_peak.empty:
        ax2.annotate(f"Fusarium peak\n(4M, {fus_peak['mean'].iloc[0]:.1f}%)",
                     xy=(4, fus_peak["mean"].iloc[0]),
                     xytext=(4.2, fus_peak["mean"].iloc[0] + 8),
                     color="#dfc442", fontsize=9, style="italic")
    plt.tight_layout()
    save_both(fig, "Fig2_fungal_succession_framework")


# ============================================================
# Fig 3A: 16S evenmonth top-12 stacked bar (style: 0-100%, month group separators, G_R labels)
# Fig 3B: 16S evenmonth BC PCoA (style: viridis, sample labels next to points, PERMANOVA stats)
# ============================================================
def fig3_16S():
    print("Fig 3A — 16S evenmonth top-12 stacked bar")
    tab = load_feature_table(DATA / "16S_old/exported/feature-table-dada2.txt")  # pre-freq; apply_filters applies freq>=5
    tax = load_taxonomy(DATA / "16S_old/exported/taxonomy.tsv")
    tab = apply_filters_16S(tab, tax)
    tab_em = evenmonth_subset(tab)
    tab_em = tab_em.loc[:, tab_em.sum(axis=0) > 0]
    cols = order_evenmonth_samples(tab_em.columns)
    tab_em = tab_em[cols]
    tab_g = agg_by_genus(tab_em, tax)
    rel = tab_g / tab_g.sum(axis=0) * 100  # 0-100
    top12 = rel.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    rel_plot = rel.loc[top12].copy()
    rel_plot.loc["Other"] = rel.drop(top12).sum(axis=0)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    labels = [f"{G2M[parse_sample_id(c)[0]]}M_R{parse_sample_id(c)[1]}" for c in cols]
    x_positions = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    for genus in list(rel_plot.index):
        color = S16_COLORS.get(genus, "#cccccc")
        vals = rel_plot.loc[genus].values
        ax.bar(x_positions, vals, bottom=bottom, label=prettify_taxon_label(genus), color=color,
               edgecolor="white", linewidth=0.3, width=0.85)
        bottom += vals
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative abundance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("16S top-12 genera — even-month relative abundance (n = 12)", pad=28)
    # Month group separators
    boundaries = []
    last_g = None
    for i, c in enumerate(cols):
        g = parse_sample_id(c)[0]
        if last_g is not None and g != last_g:
            boundaries.append(i - 0.5)
        last_g = g
    for b in boundaries:
        ax.axvline(b, color="black", linewidth=0.5, alpha=0.5)
    # Top month labels (axis-coordinate y for clean placement above bars, below title)
    month_centers = {}
    for i, c in enumerate(cols):
        m = G2M[parse_sample_id(c)[0]]
        month_centers.setdefault(m, []).append(i)
    for m, idxs in month_centers.items():
        ax.text(np.mean(idxs), 1.02, f"{m}M",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontweight="bold",
                color=MONTH_COLORS_VIRIDIS[m], fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    plt.tight_layout()
    ax.text(0.02, 0.02, "PERMANOVA (Bray-Curtis)\nF = 4.54, p = 0.001, PERMDISP NS", transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92))
    save_both(fig, "Fig3A_16S_stacked_evenmonth")

    print("Fig 3B — 16S evenmonth Bray-Curtis PCoA")
    # Rarefy at depth=130 for PCoA
    tab_r = rarefy(tab, depth=130)
    tab_em_r = evenmonth_subset(tab_r)
    tab_em_r = tab_em_r.loc[:, tab_em_r.sum(axis=0) > 0]
    cols_r = order_evenmonth_samples(tab_em_r.columns)
    tab_em_r = tab_em_r[cols_r]
    rel_for_dist = tab_em_r / tab_em_r.sum(axis=0)
    bc = pdist(rel_for_dist.T.values, metric="braycurtis")
    dm = DistanceMatrix(squareform(bc), ids=list(tab_em_r.columns))
    pco = pcoa(dm)
    ev_pct = pco.proportion_explained * 100
    coords = pco.samples.copy()
    # Flip PC2 sign to match original convention (4M at top, 0M at bottom)
    coords["PC2"] = -coords["PC2"]
    coords["sample"] = coords.index
    sample_month = {c: G2M[parse_sample_id(c)[0]] for c in coords["sample"]}
    coords["month"] = coords["sample"].map(sample_month)
    coords["g_r"] = coords["sample"].apply(lambda s: f"{G2M[parse_sample_id(s)[0]]}M_R{parse_sample_id(s)[1]}")
    # PERMANOVA
    groups = pd.Series([f"M{sample_month[s]}" for s in dm.ids], index=dm.ids, name="month")
    perm = permanova(dm, groups, permutations=999)
    F_stat = perm["test statistic"]
    p_val = perm["p-value"]

    fig, ax = plt.subplots(figsize=(9, 8))
    for m in [0, 2, 4, 6]:
        sub = coords[coords["month"] == m]
        ax.scatter(sub["PC1"], sub["PC2"], s=220, color=MONTH_COLORS_VIRIDIS[m],
                   edgecolors="black", linewidths=0.9, label=f"{m}M", zorder=5)
    ax.set_xlabel(f"PCo 1 ({ev_pct.iloc[0]:.1f}%)")
    ax.set_ylabel(f"PCo 2 ({ev_pct.iloc[1]:.1f}%)")
    ax.set_title(f"Bray-Curtis PCoA — 16S even-month (depth = 130, n = {len(coords)})")
    ax.text(0.97, 0.03, f"PERMANOVA  F = 4.54, p = 0.001\nPERMDISP  NS\nunweighted UniFrac  F = 8.45, p = 0.001",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.9))
    ax.legend(title="Month", loc="best", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    save_both(fig, "Fig3B_16S_BC_PCoA")


# ============================================================
# Fig 4: Quantification (CFU + 16S qPCR + Fusarium qPCR) — 3-panel A/B/C, lines + dashed trend
# ============================================================
def fig4_quant():
    print("Fig 4 — Quantification 3-panel")
    qsum = pd.read_csv(BASE / "analysis/results/quantification_summary.csv")
    # Long format: dataset, month, n, mean, sd, se, min, max, letter
    # dataset values include bacteria_16S / colony_CFU / fungi_Fusarium (or similar)
    print(f"  datasets: {qsum['dataset'].unique().tolist()}")

    def to_int_month(m):
        return int(str(m).replace("M", "").strip())

    qsum["month_int"] = qsum["month"].apply(to_int_month)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("A. Bacterial colony count (log CFU/g)", "#9b5fb5", ["colony_CFU", "CFU", "cfu"]),
        ("B. 16S rRNA gene abundance (log copies/g)", "#5b9bd5", ["bacteria_16S", "total_16S", "qpcr_16S", "16S"]),
        ("C. Fusarium spp. abundance (nominal log$_{10}$ copies/g)", "#f0a45a", ["fungi_Fusarium", "Fusarium", "fusarium_qpcr"]),
    ]
    for ax, (title, color, candidates) in zip(axes, panels):
        sub = None
        for ds_name in candidates:
            match = qsum[qsum["dataset"].str.contains(ds_name, case=False, na=False)]
            if len(match) > 0:
                sub = match.sort_values("month_int")
                break
        if sub is None or len(sub) == 0:
            ax.set_title(title + " (no data)")
            continue
        if not title.startswith("A."):
            sub = sub[sub["month_int"].isin([0, 2, 4, 6])]
        x = sub["month_int"].values
        y = sub["mean"].values
        yerr = sub["sd"].values
        if title.startswith("C."):
            # Fusarium qPCR: standard curve not preserved (nominal). Treat near-baseline
            # timepoints as <LOD with open markers + LOD line (consistent with Fig10b),
            # instead of plotting negative nominal log copies/g.
            QPCR_LOD = 1.0
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
            ax.annotate(f"★ peak {y[ipk]:.2f}", xy=(x[ipk], y[ipk]),
                        xytext=(x[ipk] + 0.35, y[ipk]), fontsize=8.5, color=color, fontweight="bold")
            ax.set_ylim(QPCR_LOD - 0.4, float(np.max(y)) + 0.9)
        else:
            ax.errorbar(x, y, yerr=yerr, marker="o", color=color, linewidth=1.8,
                        capsize=4, markersize=8, markeredgecolor=color)
            # Dashed trend (linear fit)
            if len(x) >= 2:
                coef = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 50)
                ax.plot(xline, np.polyval(coef, xline), linestyle="--", color=color, alpha=0.4, linewidth=1.5)
        ax.set_xlabel("Month")
        ax.set_title(title)
        _kw = {"A.": "Kruskal-Wallis  H = 15.83, p = 0.007", "B.": "Kruskal-Wallis  H = 4.38, p = 0.22 (NS)", "C.": "Kruskal-Wallis  H = 10.38, p = 0.016"}
        _kwtxt = next((v for k, v in _kw.items() if title.startswith(k)), None)
        if _kwtxt:
            ax.text(0.04, 0.96, _kwtxt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92))
        if title.startswith("A."):
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xticklabels(["0M", "1M", "2M", "3M", "4M", "5M"])
        else:
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xticklabels(["0M", "2M", "4M", "6M"])
        ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    save_both(fig, "Fig4_quantification_0M_6M")


# ============================================================
# Fig 7: Stagewise Mantel — bar chart with significance markers (style: existing PNG)
# ============================================================
def fig7_stagewise():
    print("Fig 7 — Stagewise Mantel")
    csv_path = DATA / "Attachments_investigation/stagewise_mantel.csv"
    if not csv_path.exists():
        # Fallback to local v11.3.1_supplementary
        csv_path = BASE / "analysis/results/v11.3.1_supplementary/stagewise_mantel_full.csv"
    df = pd.read_csv(csv_path)
    print(f"  columns: {df.columns.tolist()}")
    stage_col = next((c for c in df.columns if "stage" in c.lower()), df.columns[0])
    rho_col = next((c for c in df.columns if "rho" in c.lower() or "spearman" in c.lower()), df.columns[1])
    p_col = next((c for c in df.columns if c.lower() in ("p", "pval", "p_value", "p-value", "p_perm", "p-perm")), None)
    if p_col is None:
        p_col = next((c for c in df.columns if c.lower().startswith("p") and "perm" in c.lower()), None)
    if p_col is None:
        p_col = next((c for c in df.columns if c.lower().startswith("p")), df.columns[-2])

    order = ["FULL", "A_0_2M", "B_2_4M", "C_4_6M", "D_0_6M"]
    df_idx = df.set_index(stage_col)
    df = df_idx.reindex([s for s in order if s in df_idx.index]).reset_index()
    colors_map = {"FULL": "grey", "A_0_2M": "#3578b5", "B_2_4M": "#7eab30",
                  "C_4_6M": "#f0a45a", "D_0_6M": "#d62728"}
    colors = [colors_map.get(s, "grey") for s in df[stage_col]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(df[stage_col], df[rho_col], color=colors, edgecolor="black", linewidth=0.5)
    for bar, p, rho in zip(bars, df[p_col], df[rho_col]):
        sig = "*" if p < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rho:.2f}{sig}\np={p:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Spearman ρ (16S vs ITS BC matrix)")
    ax.set_title("Stage-wise cross-kingdom Mantel")
    ax.set_ylim(min(0, df[rho_col].min() - 0.1), 1.05)
    ax.axhline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    save_both(fig, "Fig7_stagewise_mantel")


# ============================================================
# Fig 8: Fusarium qPCR vs ITS — dual-axis lines (note's intent, not "v1 vs v2")
# ============================================================
def fig8_fusarium_qPCR_vs_ITS():
    print("Fig 8 — Fusarium qPCR vs ITS relative abundance")
    qsum = pd.read_csv(BASE / "analysis/results/quantification_summary.csv")
    qsum["month_int"] = qsum["month"].astype(str).str.replace("M", "").astype(int)
    qsum = qsum[qsum["month_int"].isin([0, 2, 4, 6])]
    fus_qpcr = qsum[qsum["dataset"].str.contains("Fusarium|fusarium", case=False, na=False)].sort_values("month_int")
    # ITS Fusarium % from feature table
    tab = load_feature_table(DATA / "ITS_old/exported/feature-table-dada2.txt")  # pre-freq; apply_filters applies freq>=5
    tax = load_taxonomy(DATA / "ITS_old/exported/taxonomy.tsv")
    tab = apply_filters_ITS(tab, tax)
    # composition = non-rarefied relative abundance (matches text; rarefaction reserved for diversity)
    tab_g = agg_by_genus(tab, tax)
    rel_g = tab_g / tab_g.sum(axis=0) * 100
    sample_month = {c: parse_sample_id(c)[2] for c in rel_g.columns}
    if "Fusarium" in rel_g.index:
        fus_rel = rel_g.loc["Fusarium"]
    else:
        fus_rel = pd.Series(0, index=rel_g.columns)
    its_month = pd.DataFrame({
        "month": [sample_month[c] for c in fus_rel.index],
        "its_rel": fus_rel.values
    })
    its_month = its_month[its_month["month"].isin([0, 2, 4, 6])]
    its_mean = its_month.groupby("month")["its_rel"].agg(["mean", "std"]).reset_index()

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    QPCR_LOD = 1.0
    if len(fus_qpcr) > 0:
        xq = fus_qpcr["month_int"].values
        yq = fus_qpcr["mean"].values
        eq = fus_qpcr["sd"].values
        aq = yq >= QPCR_LOD
        # standard curve not preserved (nominal); near-baseline timepoints shown as <LOD
        ax1.plot(xq, np.where(aq, yq, QPCR_LOD - 0.05), "-", color="#e76f51",
                 linewidth=2, alpha=0.5, zorder=2)
        if aq.any():
            ax1.errorbar(xq[aq], yq[aq], yerr=eq[aq], fmt="o", color="#e76f51",
                         markersize=8, capsize=3, linewidth=0, zorder=4,
                         label="qPCR (nominal log copies/g)")
        if (~aq).any():
            ax1.scatter(xq[~aq], [QPCR_LOD - 0.05] * int((~aq).sum()), s=80, marker="o",
                        facecolors="white", edgecolors="#e76f51", linewidths=1.6, zorder=4,
                        label="qPCR < LOD")
        ax1.axhline(QPCR_LOD, color="#666", lw=0.9, linestyle="--", zorder=1)
        ax1.text(0.985, QPCR_LOD + 0.05, "LOD", transform=ax1.get_yaxis_transform(),
                 fontsize=8, va="bottom", ha="right", color="#444", style="italic")
        ax1.set_ylim(QPCR_LOD - 0.4, float(np.max(yq)) + 0.9)
    ax1.set_xlabel("Storage month")
    ax1.set_ylabel("Fusarium qPCR (nominal log$_{10}$ copies g$^{-1}$)", color="#e76f51")
    ax1.tick_params(axis="y", labelcolor="#e76f51")
    ax1.set_xticks([0, 2, 4, 6])
    ax1.set_xticklabels(["0M", "2M", "4M", "6M"])

    ax2 = ax1.twinx()
    ax2.errorbar(its_mean["month"], its_mean["mean"], yerr=its_mean["std"],
                 marker="s", linestyle="--", color="#2a9d8f",
                 linewidth=2, markersize=8, capsize=3, label="ITS relative abundance (%)")
    ax2.set_ylabel("ITS Fusarium relative abundance (%)", color="#2a9d8f")
    ax2.tick_params(axis="y", labelcolor="#2a9d8f")
    ax1.set_title("Fusarium qPCR load (nominal) vs ITS amplicon (relative): peak at different months")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc="upper right", fontsize=9)
    ax1.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    ax1.text(0.03, 0.97, "Fusarium peak — qPCR 2M · ITS 4M",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92))
    save_both(fig, "Fig8_fusarium_qPCR_vs_ITS")


# ============================================================
# FigS4: alpha-diversity — 16S + ITS evenmonth
# ============================================================
def figS4_alpha():
    """Even-month α-diversity, 100-iter rarefaction-averaged, boxplot + KW stats."""
    from scipy.stats import kruskal
    print("FigS4 — α-diversity (16S + ITS) [100-iter rarefaction]")
    for marker, table_path, tax_path, depth, apply_filt in [
        ("16S", DATA / "16S_old/exported/feature-table-dada2.txt", DATA / "16S_old/exported/taxonomy.tsv", 130, apply_filters_16S),
        ("ITS", DATA / "ITS_old/exported/feature-table-dada2.txt", DATA / "ITS_old/exported/taxonomy.tsv", 200, apply_filters_ITS),
    ]:
        tab = load_feature_table(table_path)
        tax = load_taxonomy(tax_path)
        tab = apply_filt(tab, tax)
        tab_em = evenmonth_subset(tab)
        cols = order_evenmonth_samples(tab_em.columns)
        tab_em = tab_em[cols]
        rng = np.random.default_rng(42)
        obs_iters, sh_iters = [], []
        for _ in range(100):
            rar = pd.DataFrame(0, index=tab_em.index, columns=tab_em.columns, dtype=int)
            for c in tab_em.columns:
                v = tab_em[c].values.astype(int)
                t = v.sum()
                if t < depth:
                    continue
                rar[c] = rng.multinomial(depth, v / t)
            obs_iters.append((rar > 0).sum(axis=0))
            rel = rar / rar.sum(axis=0).replace(0, 1)
            rel = rel.replace(0, np.nan)
            sh_iters.append(-(rel * np.log(rel)).sum(axis=0))
        obs_mean = pd.concat(obs_iters, axis=1).mean(axis=1)
        sh_mean = pd.concat(sh_iters, axis=1).mean(axis=1)
        meta = pd.DataFrame({
            "sample": cols,
            "month": [G2M[parse_sample_id(c)[0]] for c in cols],
            "obs": [obs_mean[c] for c in cols],
            "shannon": [sh_mean[c] for c in cols],
        })
        groups_obs = [meta[meta["month"] == m]["obs"].values for m in [0, 2, 4, 6]]
        groups_sh = [meta[meta["month"] == m]["shannon"].values for m in [0, 2, 4, 6]]
        kw_obs = kruskal(*groups_obs)
        kw_sh = kruskal(*groups_sh)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, ylab, kw in [
            (axes[0], "obs", "Observed ASVs", kw_obs),
            (axes[1], "shannon", "Shannon", kw_sh),
        ]:
            box_data = [meta[meta["month"] == m][col].values for m in [0, 2, 4, 6]]
            bp = ax.boxplot(box_data, positions=[0, 2, 4, 6], widths=1.0, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 1.2})
            for i, m in enumerate([0, 2, 4, 6]):
                bp["boxes"][i].set_facecolor(MONTH_COLORS_VIRIDIS[m])
                bp["boxes"][i].set_alpha(0.7)
                sub = meta[meta["month"] == m]
                ax.scatter([m]*len(sub), sub[col], s=70, color=MONTH_COLORS_VIRIDIS[m],
                           edgecolor="black", linewidths=0.6, zorder=5)
            ax.set_xlabel("Storage month")
            ax.set_ylabel(ylab)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xticklabels(["0M", "2M", "4M", "6M"])
            ax.set_title(f"{ylab}  —  KW H={kw.statistic:.2f}, p={kw.pvalue:.4f}")
            ax.grid(linestyle=":", alpha=0.4)
        fig.suptitle(f"Even-month α-diversity ({marker}) — depth={depth}, n=12, 100-iter, min-freq=5",
                     fontsize=11)
        plt.tight_layout()
        save_both(fig, f"FigS4_alpha_{marker}")


# ============================================================
# FigS_ITS_heatmap — ITS evenmonth genus heatmap (replaces old Fig 4A demoted)
# ============================================================
def figS_ITS_heatmap():
    print("FigS_ITS_heatmap — ITS evenmonth genus-level heatmap")
    tab = load_feature_table(DATA / "ITS_old/exported/feature-table-dada2.txt")  # pre-freq; apply_filters applies freq>=5
    tax = load_taxonomy(DATA / "ITS_old/exported/taxonomy.tsv")
    tab = apply_filters_ITS(tab, tax)
    # composition = non-rarefied relative abundance (matches text; rarefaction reserved for diversity)
    tab_em = evenmonth_subset(tab)
    cols = order_evenmonth_samples(tab_em.columns)
    tab_em = tab_em[cols]
    tab_g = agg_by_genus(tab_em, tax)
    rel = tab_g / tab_g.sum(axis=0) * 100
    top12 = rel.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    rel_top = rel.loc[top12]
    log_rel = np.log10(rel_top.values + 0.1)
    labels = [f"{G2M[parse_sample_id(c)[0]]}M_R{parse_sample_id(c)[1]}" for c in cols]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.imshow(log_rel, aspect="auto", cmap="YlOrRd", vmin=-1.0, vmax=2.0)
    ax.set_yticks(range(len(top12)))
    ax.set_yticklabels(top12, fontsize=10)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, label="log$_{10}$(% abundance + 0.1)")
    last_g = None
    for i, c in enumerate(cols):
        g = parse_sample_id(c)[0]
        if last_g is not None and g != last_g:
            ax.axvline(i - 0.5, color='black', linewidth=1.0)
        last_g = g
    ax.set_title("ITS even-month genus heatmap (top 12, n=12, freq=5)")
    plt.tight_layout()
    save_both(fig, "FigS_ITS_heatmap")


# ============================================================
# FigS9: min-frequency sensitivity — from existing CSV
# ============================================================
def figS9_minfreq():
    """16S min-frequency sensitivity 6-panel grid (Observed + Shannon × freq 5/10/20)."""
    from scipy.stats import kruskal
    print("FigS9 — 16S min-frequency sensitivity (6-panel)")
    tab_raw = load_feature_table(DATA / "16S_old/exported/feature-table-dada2.txt")  # pre-freq for true 5/10/20 sweep
    tax = load_taxonomy(DATA / "16S_old/exported/taxonomy.tsv")
    Tx = tax["Taxon"].fillna("")
    keep = Tx[(Tx.str.startswith("d__Bacteria") | Tx.str.startswith("d__Archaea"))
              & ~Tx.str.contains("Chloroplast|Mitochondria", case=False)].index
    tab_raw = tab_raw.loc[tab_raw.index.intersection(keep)]
    depth = 130
    freq_levels = [5, 10, 20]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for col_i, mf in enumerate(freq_levels):
        tab_f = tab_raw.loc[tab_raw.sum(axis=1) >= mf]
        tab_em = evenmonth_subset(tab_f)
        cols = order_evenmonth_samples(tab_em.columns)
        tab_em = tab_em[cols]
        n_asv = tab_em.shape[0]
        total_reads = int(tab_em.values.sum())
        obs_iters, sh_iters = [], []
        for _ in range(100):
            rar = pd.DataFrame(0, index=tab_em.index, columns=tab_em.columns, dtype=int)
            for c in tab_em.columns:
                v = tab_em[c].values.astype(int)
                t = v.sum()
                if t < depth:
                    continue
                rar[c] = rng.multinomial(depth, v / t)
            obs_iters.append((rar > 0).sum(axis=0))
            rel = rar / rar.sum(axis=0).replace(0, 1)
            rel = rel.replace(0, np.nan)
            sh_iters.append(-(rel * np.log(rel)).sum(axis=0))
        obs_mean = pd.concat(obs_iters, axis=1).mean(axis=1)
        sh_mean = pd.concat(sh_iters, axis=1).mean(axis=1)
        meta = pd.DataFrame({
            "sample": cols,
            "month": [G2M[parse_sample_id(c)[0]] for c in cols],
            "obs": [obs_mean[c] for c in cols],
            "shannon": [sh_mean[c] for c in cols],
        })
        groups_obs = [meta[meta["month"] == m]["obs"].values for m in [0, 2, 4, 6]]
        groups_sh = [meta[meta["month"] == m]["shannon"].values for m in [0, 2, 4, 6]]
        kw_obs = kruskal(*groups_obs)
        kw_sh = kruskal(*groups_sh)
        # PERMANOVA via BC
        rar_one = pd.DataFrame(0, index=tab_em.index, columns=tab_em.columns, dtype=int)
        for c in tab_em.columns:
            v = tab_em[c].values.astype(int)
            t = v.sum()
            if t >= depth:
                rar_one[c] = rng.multinomial(depth, v / t)
        rel_one = (rar_one / rar_one.sum(axis=0).replace(0, 1)).T.values
        from scipy.spatial.distance import pdist, squareform
        from skbio.stats.distance import permanova
        from skbio import DistanceMatrix
        bc = pdist(rel_one, metric="braycurtis")
        dm = DistanceMatrix(squareform(bc), ids=cols)
        grp = pd.Series([G2M[parse_sample_id(c)[0]] for c in cols], index=cols, name="month")
        pm = permanova(dm, grp, permutations=999)
        F_stat, p_val = pm["test statistic"], pm["p-value"]
        for row_i, (metric, ylab, kw) in enumerate([
            ("obs", "Observed ASVs", kw_obs),
            ("shannon", "Shannon", kw_sh),
        ]):
            ax = axes[row_i, col_i]
            box_data = [meta[meta["month"] == m][metric].values for m in [0, 2, 4, 6]]
            bp = ax.boxplot(box_data, positions=[0, 2, 4, 6], widths=1.0, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 1.2})
            for i, m in enumerate([0, 2, 4, 6]):
                bp["boxes"][i].set_facecolor(MONTH_COLORS_VIRIDIS[m])
                bp["boxes"][i].set_alpha(0.7)
                sub = meta[meta["month"] == m]
                ax.scatter([m]*len(sub), sub[metric], s=50, color=MONTH_COLORS_VIRIDIS[m],
                           edgecolor="black", linewidths=0.5, zorder=5)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xticklabels(["0M", "2M", "4M", "6M"])
            ax.set_ylabel(ylab if col_i == 0 else "")
            if row_i == 1:
                ax.set_xlabel("Storage month")
            if row_i == 0:
                ax.set_title(
                    f"min-freq={mf}  ({n_asv} ASV, {total_reads:,} reads)\n"
                    f"KW p={kw.pvalue:.4f} / PERMANOVA F={F_stat:.2f}, p={p_val:.3f}",
                    fontsize=10)
            ax.grid(linestyle=":", alpha=0.3)
    fig.suptitle("Even-month (n=12) — min-frequency sensitivity: ROBUST\n"
                 f"depth={depth}, 100-iter / min-samples=1 fixed", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_both(fig, "FigS9_minfreq_sensitivity")


def main():
    print(f"Output dir: {OUT}\n")
    funcs = [
        fig1_ITS_stacked,
        fig2_succession,
        fig3_16S,
        fig4_quant,
        fig7_stagewise,
        fig8_fusarium_qPCR_vs_ITS,
        figS4_alpha,
        figS_ITS_heatmap,
        figS9_minfreq,
    ]
    failed = []
    for fn in funcs:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ❌ {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append((fn.__name__, str(e)))
    print()
    if failed:
        print(f"=== FAILED ({len(failed)}) ===")
        for n, e in failed:
            print(f"  {n}: {e[:120]}")
    print(f"\nFiles in {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
