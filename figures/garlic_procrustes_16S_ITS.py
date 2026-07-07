#!/usr/bin/env python
"""
Cross-kingdom Procrustes analysis: 16S vs ITS OLD even-month (n=12)
- Build BC distance matrices for 16S and ITS (same 12 samples)
- PCoA -> Procrustes superimposition + permutation p
- Mantel test (Spearman) on BC matrices
"""
import os, sys, tempfile, shutil, zipfile, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.spatial import procrustes as scipy_procrustes
from skbio import OrdinationResults
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix

BASE   = os.path.expanduser("~/garlic_project/data/qiime2_reanalysis")
OUTDIR = BASE
os.makedirs(OUTDIR, exist_ok=True)

EVEN_SAMPLES = [f"old_G{g}_R{r}" for g in (1,3,5,7) for r in (1,2,3)]  # 12 samples


# ---------- helpers ----------
def extract_qza(qza_path, suffix):
    td = tempfile.mkdtemp(prefix=f"qza_{suffix}_")
    with zipfile.ZipFile(qza_path) as z:
        z.extractall(td)
    return td


def load_feature_table(qza_path):
    td = extract_qza(qza_path, "tbl")
    biom_files = glob.glob(os.path.join(td, "*/data/feature-table.biom"))
    if not biom_files:
        raise RuntimeError("no biom in qza")
    from biom import load_table
    tbl = load_table(biom_files[0])
    df = tbl.to_dataframe(dense=True)  # rows=features, cols=samples
    df = df.astype(float)
    shutil.rmtree(td)
    return df


def load_taxonomy(qza_path):
    td = extract_qza(qza_path, "tax")
    tsvs = glob.glob(os.path.join(td, "*/data/taxonomy.tsv"))
    df = pd.read_csv(tsvs[0], sep="\t")
    df = df.rename(columns={"Feature ID": "feature_id", "Taxon": "taxon"})
    df = df.set_index("feature_id")
    shutil.rmtree(td)
    return df["taxon"].astype(str)


def is_contam_16s(tax):
    t = tax.lower()
    if "mitochondria" in t or "chloroplast" in t:
        return True
    if t.startswith("unassigned") or t.strip() == "":
        return True
    if "d__eukaryota" in t:
        return True
    return False


def is_keep_its(tax):
    t = tax.lower()
    if t.startswith("unassigned") or t.strip() == "":
        return False
    if "k__fungi" in t:
        return True
    return False


def rarefy_table(df_counts, depth, seed=42):
    """df_counts: features x samples. Rarefy each sample column to `depth`."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(0, index=df_counts.index, columns=df_counts.columns, dtype=int)
    keep = []
    for s in df_counts.columns:
        col = df_counts[s].values.astype(int)
        n = col.sum()
        if n < depth:
            print(f"  drop {s} (depth={n} < {depth})")
            continue
        # multinomial without replacement via expanded indices
        idx = np.repeat(np.arange(len(col)), col)
        chosen = rng.choice(idx, size=depth, replace=False)
        vc = np.bincount(chosen, minlength=len(col))
        out[s] = vc
        keep.append(s)
    return out[keep]


def bc_distance(df_counts):
    # samples x features
    M = df_counts.T.values.astype(float)
    # relative abundance
    M = M / M.sum(axis=1, keepdims=True)
    D = squareform(pdist(M, metric="braycurtis"))
    return pd.DataFrame(D, index=df_counts.columns, columns=df_counts.columns)


def filter_features(tbl, taxonomy, contam_fn=None, keep_fn=None, freq_min=5):
    # taxonomy filter
    if contam_fn is not None:
        bad = set(taxonomy[taxonomy.apply(contam_fn)].index)
        tbl = tbl.loc[~tbl.index.isin(bad)]
    if keep_fn is not None:
        good = set(taxonomy[taxonomy.apply(keep_fn)].index)
        tbl = tbl.loc[tbl.index.isin(good)]
    # frequency filter
    tbl = tbl.loc[tbl.sum(axis=1) >= freq_min]
    return tbl


def build_bc(domain, table_qza, tax_qza, depth, samples, kind):
    print(f"\n=== {domain} ===")
    tbl = load_feature_table(table_qza)
    tax = load_taxonomy(tax_qza)
    print(f"  raw features={tbl.shape[0]}, samples={tbl.shape[1]}")
    # restrict to even-month samples
    sub = [s for s in samples if s in tbl.columns]
    miss = set(samples) - set(sub)
    if miss:
        print(f"  WARN missing: {miss}")
    tbl = tbl[sub]
    if kind == "16S":
        tbl = filter_features(tbl, tax, contam_fn=is_contam_16s, keep_fn=None, freq_min=5)
    else:
        tbl = filter_features(tbl, tax, contam_fn=None, keep_fn=is_keep_its, freq_min=5)
    print(f"  filtered features={tbl.shape[0]}")
    rar = rarefy_table(tbl, depth=depth, seed=42)
    print(f"  rarefied: n_samples={rar.shape[1]}, depth={depth}")
    D = bc_distance(rar)
    return rar, D


# ---------- run ----------
samples = EVEN_SAMPLES

print("Building 16S OLD even-month BC matrix")
rar_16s, D_16s = build_bc(
    "16S",
    table_qza=os.path.join(BASE, "16S_old", "table-dada2.qza"),
    tax_qza=os.path.join(BASE, "16S_old", "taxonomy.qza"),
    depth=130, samples=samples, kind="16S")

print("Building ITS OLD even-month BC matrix")
rar_its, D_its = build_bc(
    "ITS",
    table_qza=os.path.join(BASE, "ITS_old", "table-dada2.qza"),
    tax_qza=os.path.join(BASE, "ITS_old", "taxonomy.qza"),
    depth=200, samples=samples, kind="ITS")

# align sample order intersection
common = [s for s in samples if s in D_16s.index and s in D_its.index]
print(f"\nCommon samples (n={len(common)}): {common}")
D_16s = D_16s.loc[common, common]
D_its = D_its.loc[common, common]

# PCoA
dm_16s = DistanceMatrix(D_16s.values, ids=common)
dm_its = DistanceMatrix(D_its.values, ids=common)
ord_16s = pcoa(dm_16s)
ord_its = pcoa(dm_its)
coord_16s = ord_16s.samples.iloc[:, :2].values  # PCo1, PCo2
coord_its = ord_its.samples.iloc[:, :2].values
ev_16s = ord_16s.proportion_explained.iloc[:2].values * 100
ev_its = ord_its.proportion_explained.iloc[:2].values * 100

# Procrustes
mtx1, mtx2, disparity = scipy_procrustes(coord_16s, coord_its)

def procrustes_disparity(X, Y):
    m1, m2, d = scipy_procrustes(X, Y)
    return d

# permutation test: shuffle ITS row labels (sample identities)
rng = np.random.default_rng(123)
n_perm = 999
null = np.zeros(n_perm)
for i in range(n_perm):
    perm = rng.permutation(coord_its.shape[0])
    null[i] = procrustes_disparity(coord_16s, coord_its[perm])
obs = disparity
p_proc = (np.sum(null <= obs) + 1) / (n_perm + 1)

# Mantel (Spearman)
iu = np.triu_indices(len(common), k=1)
v16 = D_16s.values[iu]
vit = D_its.values[iu]
mantel_r, _ = spearmanr(v16, vit)
null_m = np.zeros(n_perm)
for i in range(n_perm):
    perm = rng.permutation(len(common))
    D_its_p = D_its.values[np.ix_(perm, perm)]
    r, _ = spearmanr(v16, D_its_p[iu])
    null_m[i] = r
p_mantel = (np.sum(null_m >= mantel_r) + 1) / (n_perm + 1)

print("\n=== Results ===")
print(f"Procrustes M^2 (disparity) = {obs:.4f}")
print(f"Procrustes p (999 perm)    = {p_proc:.4f}")
print(f"Mantel r (Spearman)         = {mantel_r:.4f}")
print(f"Mantel p (999 perm)         = {p_mantel:.4f}")

# Save CSV
stats_df = pd.DataFrame([{
    "n_samples": len(common),
    "procrustes_disparity_M2": obs,
    "procrustes_p_999perm": p_proc,
    "mantel_r_spearman": mantel_r,
    "mantel_p_999perm": p_mantel,
    "ev_16s_pc1_pc2": f"{ev_16s[0]:.1f}/{ev_16s[1]:.1f}",
    "ev_its_pc1_pc2": f"{ev_its[0]:.1f}/{ev_its[1]:.1f}",
}])
stats_csv = os.path.join(OUTDIR, "procrustes_16S_vs_ITS_stats.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"saved -> {stats_csv}")

# per-sample residuals (distance between matched configurations after procrustes)
resid = np.linalg.norm(mtx1 - mtx2, axis=1)
resid_df = pd.DataFrame({"sample": common, "procrustes_residual": resid}).sort_values("procrustes_residual")
resid_csv = os.path.join(OUTDIR, "procrustes_16S_vs_ITS_residuals.csv")
resid_df.to_csv(resid_csv, index=False)
print(f"saved -> {resid_csv}")

# ----- plots -----
# colors per sample (groups 1,3,5,7)
group_colors = {"G1": "#4575b4", "G3": "#74add1", "G5": "#fdae61", "G7": "#d73027"}
def sample_color(s):
    for g, c in group_colors.items():
        if g in s:
            return c
    return "gray"
colors = [sample_color(s) for s in common]

# paired PCoA
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, coord, ev, title in (
    (axes[0], coord_16s, ev_16s, "16S OLD even-month"),
    (axes[1], coord_its, ev_its, "ITS OLD even-month"),
):
    ax.scatter(coord[:, 0], coord[:, 1], c=colors, s=110, edgecolor="k", linewidth=0.8, zorder=3)
    for i, s in enumerate(common):
        ax.annotate(s.replace("old_", ""), (coord[i, 0], coord[i, 1]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PCo1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PCo2 ({ev[1]:.1f}%)")
    ax.set_title(title)
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.axvline(0, color="grey", lw=0.4, ls=":")
# group legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=f"{g} (Month {int(g[1])*2})") for g, c in group_colors.items()]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False)
plt.tight_layout()
p_paired = os.path.join(OUTDIR, "procrustes_paired_PCoA_16S_ITS.png")
plt.savefig(p_paired, dpi=200, bbox_inches="tight")
plt.close()
print(f"saved -> {p_paired}")

# Procrustes superimposition
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(mtx1[:, 0], mtx1[:, 1], c=colors, marker="o", s=110, edgecolor="k", linewidth=0.8, label="16S", zorder=3)
ax.scatter(mtx2[:, 0], mtx2[:, 1], c=colors, marker="^", s=110, edgecolor="k", linewidth=0.8, label="ITS", zorder=3)
for i, s in enumerate(common):
    ax.plot([mtx1[i, 0], mtx2[i, 0]], [mtx1[i, 1], mtx2[i, 1]],
            color=colors[i], lw=1.0, alpha=0.7, zorder=2)
    ax.annotate(s.replace("old_", ""), (mtx1[i, 0], mtx1[i, 1]),
                fontsize=7.5, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Procrustes dim 1")
ax.set_ylabel("Procrustes dim 2")
ax.set_title(f"Procrustes 16S vs ITS — M$^2$={obs:.3f}, p={p_proc:.3f}\nMantel ρ={mantel_r:.3f}, p={p_mantel:.3f}")
# combined legend
from matplotlib.lines import Line2D
shape_legend = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="k", markersize=10, label="16S"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="grey", markeredgecolor="k", markersize=10, label="ITS"),
]
leg1 = ax.legend(handles=shape_legend, loc="upper right", frameon=True)
ax.add_artist(leg1)
ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
plt.tight_layout()
p_super = os.path.join(OUTDIR, "procrustes_superimposition_16S_ITS.png")
plt.savefig(p_super, dpi=200, bbox_inches="tight")
plt.close()
print(f"saved -> {p_super}")

print("\nDONE")
