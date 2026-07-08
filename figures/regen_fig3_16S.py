#!/usr/bin/env python3
"""
fig3_16S_local.py — Figure 3 (bacterial community: A stacked bar, B Bray-Curtis PCoA)
regenerated locally on the even-month frame.

Why this script exists
----------------------
The rendered Figure 3 carried a hardcoded statistics box. It first read "PERMDISP NS",
which is false, and was then hand-edited to "PERMDISP p = 0.03" — a value taken from a
*with-replacement* rarefaction draw, while the F = 4.54 printed beside it comes from the
*without-replacement* draw used everywhere else (garlic_16S_depth_sweep.py, Table S6).
No committed script reproduced the figure at all.

Here every number in the annotation is computed from the same distance matrix that the
figure plots. Rarefaction subsamples without replacement (seed = 42, depth = 130), matching
the main pipeline, so panel B reports PERMANOVA F = 4.54, p = 0.001 and PERMDISP F = 12.40,
p = 0.022 — a consistent pair.

The min-frequency filter is applied to the 12 analysed samples (587 ASVs), matching
scripts/regen_minfreq_sensitivity.py.

unweighted UniFrac (F = 8.45) needs the phylogeny, which is not part of this archive;
it is carried in as a constant from the depth sweep and is flagged as such below.

Run:  python3 figures/regen_fig3_16S.py   (writes figures/output/)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures" / "output"
OUT.mkdir(parents=True, exist_ok=True)

EVEN_G = [1, 3, 5, 7]
G2M = {1: 0, 3: 2, 5: 4, 7: 6}
SEED, DEPTH = 42, 130

# Carried in from garlic_16S_depth_sweep.py (needs the phylogeny; not computable here).
UNIFRAC_TEXT = "unweighted UniFrac  F = 8.45, p = 0.001"

MONTH_COLORS_VIRIDIS = {0: "#440154", 2: "#3b528b", 4: "#21918c", 6: "#fde725"}
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

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------------ helpers
def parse_sample_id(sid):
    g = int(sid.split("_G")[1].split("_R")[0])
    r = int(sid.split("_R")[1])
    return g, r, G2M.get(g)


def get_genus(taxon_str):
    if pd.isna(taxon_str) or taxon_str == "Unassigned":
        return "Unassigned"
    parts = [p.strip() for p in taxon_str.split(";")]
    for prefix in ("g__", "f__", "o__"):
        part = next((p for p in parts if p.startswith(prefix)), None)
        if part and len(part) > 3:
            return part[3:] if prefix == "g__" else part
    return "Unassigned"


def prettify(name):
    if name.startswith("f__"):
        return f"{name[3:]} (family)"
    if name.startswith("o__"):
        return f"{name[3:]} (order)"
    return name


def rarefy_without_replacement(counts, depth, rng):
    pool = np.repeat(np.arange(len(counts)), counts)
    sub = rng.choice(pool, size=depth, replace=False)
    out = np.zeros_like(counts)
    u, c = np.unique(sub, return_counts=True)
    out[u] = c
    return out


def bray_curtis(mat):
    n = mat.shape[0]
    D = np.zeros((n, n))
    rs = mat.sum(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            den = rs[i] + rs[j]
            D[i, j] = D[j, i] = np.abs(mat[i] - mat[j]).sum() / den if den > 0 else 0.0
    return D


def pcoa(D):
    n = D.shape[0]
    A = -0.5 * D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    w, v = eigh(H @ A @ H)
    idx = np.argsort(-w)
    w, v = w[idx], v[:, idx]
    keep = w > 1e-9
    coords = v[:, keep] * np.sqrt(w[keep])
    return coords, w[keep] / w[keep].sum()


def permanova(D, groups, n_perm=999, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n, uniq = len(groups), np.unique(groups)
    SST = (D ** 2).sum() / (2 * n)

    def ssw(g):
        s = 0.0
        for u in uniq:
            i = np.where(g == u)[0]
            if len(i) >= 2:
                s += (D[np.ix_(i, i)] ** 2).sum() / (2 * len(i))
        return s

    SSW, a = ssw(groups), len(uniq)
    F = ((SST - SSW) / (a - 1)) / (SSW / (n - a))
    R2 = (SST - SSW) / SST
    n_ge = 1
    for _ in range(n_perm):
        s = ssw(rng.permutation(groups))
        if s > 0 and ((SST - s) / (a - 1)) / (s / (n - a)) >= F:
            n_ge += 1
    return dict(F=F, p=n_ge / (n_perm + 1), R2=R2,
                adjR2=1 - (1 - R2) * (n - 1) / (n - a))


def permdisp(D, groups, n_perm=999, seed=SEED):
    rng = np.random.default_rng(seed)
    coords, _ = pcoa(D)
    groups = np.asarray(groups)
    n, uniq = len(groups), np.unique(groups)

    def disp(g):
        d = np.zeros(n)
        for u in uniq:
            i = np.where(g == u)[0]
            d[i] = np.linalg.norm(coords[i] - coords[i].mean(axis=0), axis=1)
        return d

    def F_of(g):
        d, gm, a = disp(g), None, len(uniq)
        gm = d.mean()
        ssa = sum(len(np.where(g == u)[0]) * (d[np.where(g == u)[0]].mean() - gm) ** 2 for u in uniq)
        ssw = sum(((d[np.where(g == u)[0]] - d[np.where(g == u)[0]].mean()) ** 2).sum() for u in uniq)
        return (ssa / (a - 1)) / (ssw / (n - a)) if ssw > 0 else np.nan

    F_obs = F_of(groups)
    n_ge = 1 + sum(F_of(rng.permutation(groups)) >= F_obs for _ in range(n_perm))
    return dict(F=F_obs, p=n_ge / (n_perm + 1), per_group=disp(groups))


def save_both(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  OK  {name}.{{png,pdf}}")


# --------------------------------------------------------------------- data
def load():
    tab = pd.read_csv(DATA / "16S_feature-table-dada2.txt", sep="\t", skiprows=1, index_col=0)
    tax = pd.read_csv(DATA / "16S_taxonomy.tsv", sep="\t").set_index("Feature ID")
    cols = sorted([c for c in tab.columns if c.startswith("old_")
                   and parse_sample_id(c)[0] in EVEN_G],
                  key=lambda c: parse_sample_id(c)[:2])
    tab = tab[cols]                                    # 12 analysed samples first ...
    tab = tab[tab.sum(axis=1) >= 5]                    # ... then the min-frequency filter
    T = tax["Taxon"]
    keep = T[(T.str.startswith("d__Bacteria") | T.str.startswith("d__Archaea"))
             & ~T.str.contains("Chloroplast|Mitochondria", case=False, na=False)].index
    tab = tab.loc[tab.index.intersection(keep)]
    return tab.loc[:, cols], tax, cols


def main():
    tab, tax, cols = load()
    print(f"  even-month filtered table: {tab.shape[0]} ASVs x {tab.shape[1]} samples")
    months = np.array([G2M[parse_sample_id(c)[0]] for c in cols])

    # ---------------- Panel B statistics (computed once, reused in both annotations)
    rng = np.random.default_rng(SEED)
    rare = np.stack([rarefy_without_replacement(tab[c].values.astype(np.int64), DEPTH, rng)
                     for c in cols])
    D = bray_curtis(rare)
    pm, pdp = permanova(D, months), permdisp(D, months)
    print(f"  PERMANOVA F = {pm['F']:.2f}, p = {pm['p']:.3f}, R2 = {pm['R2']:.2f}, "
          f"adjR2 = {pm['adjR2']:.2f}")
    print(f"  PERMDISP  F = {pdp['F']:.2f}, p = {pdp['p']:.3f}")
    for m in (0, 2, 4, 6):
        print(f"    dispersion {m}M: {pdp['per_group'][months == m].mean():.3f}")

    stat_a = (f"PERMANOVA (Bray-Curtis)\n"
              f"F = {pm['F']:.2f}, p = {pm['p']:.3f}; PERMDISP F = {pdp['F']:.2f}, p = {pdp['p']:.3f}")
    stat_b = (f"PERMANOVA  F = {pm['F']:.2f}, p = {pm['p']:.3f}\n"
              f"PERMDISP  F = {pdp['F']:.2f}, p = {pdp['p']:.3f} (unequal dispersion)\n"
              f"{UNIFRAC_TEXT}")

    # ---------------- Panel A: top-12 genus stacked bar
    genus = tax.loc[tab.index, "Taxon"].apply(get_genus)
    rel = tab.groupby(genus).sum()
    rel = rel / rel.sum(axis=0) * 100
    top12 = rel.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    plot = rel.loc[top12].copy()
    plot.loc["Other"] = rel.drop(top12).sum(axis=0)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    for g in plot.index:
        vals = plot.loc[g].values
        ax.bar(x, vals, bottom=bottom, label=prettify(g),
               color=S16_COLORS.get(g, "#cccccc"), edgecolor="white", linewidth=0.3, width=0.85)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{G2M[parse_sample_id(c)[0]]}M_R{parse_sample_id(c)[1]}" for c in cols],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative abundance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("16S top-12 genera — even-month relative abundance (n = 12)", pad=28)
    last = None
    for i, c in enumerate(cols):
        g = parse_sample_id(c)[0]
        if last is not None and g != last:
            ax.axvline(i - 0.5, color="black", linewidth=0.5, alpha=0.5)
        last = g
    centers = {}
    for i, c in enumerate(cols):
        centers.setdefault(G2M[parse_sample_id(c)[0]], []).append(i)
    for m, idxs in centers.items():
        ax.text(np.mean(idxs), 1.02, f"{m}M", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontweight="bold",
                color=MONTH_COLORS_VIRIDIS[m], fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    ax.text(0.02, 0.02, stat_a, transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            zorder=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                 edgecolor="0.6", alpha=0.92))
    plt.tight_layout()
    save_both(fig, "Figure_3A_16S_stacked")

    # ---------------- Panel B: PCoA
    coords, expl = pcoa(D)
    pc1, pc2 = coords[:, 0], -coords[:, 1]      # PC2 sign flip: 4M at bottom, matching prior figure
    fig, ax = plt.subplots(figsize=(9, 8))
    for m in (0, 2, 4, 6):
        sel = months == m
        ax.scatter(pc1[sel], pc2[sel], s=220, color=MONTH_COLORS_VIRIDIS[m],
                   edgecolors="black", linewidths=0.9, label=f"{m}M", zorder=5)
    ax.set_xlabel(f"PCo 1 ({expl[0] * 100:.1f}%)")
    ax.set_ylabel(f"PCo 2 ({expl[1] * 100:.1f}%)")
    ax.set_title(f"Bray-Curtis PCoA — 16S even-month (depth = {DEPTH}, n = {len(cols)})")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(title="Month", loc="upper right", frameon=True)
    ax.text(0.97, 0.03, stat_b, transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            zorder=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                 edgecolor="0.6", alpha=0.92))
    plt.tight_layout()
    save_both(fig, "Figure_3B_16S_BC_PCoA")


if __name__ == "__main__":
    main()
