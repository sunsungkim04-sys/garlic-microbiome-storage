#!/usr/bin/env python3
"""
regen_supp.py — Generate remaining Supplementary vector PDFs.
- FigS_compositional (CLR/Aitchison sensitivity)
- FigS_RF (Random Forest feature importance)
- FigS5_NEW_PCoA_16S, FigS6_NEW_stacked_ITS, FigS7_NEW_alpha_ITS (NEW batch)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix

BASE = Path("/home1/minseo1101/garlic_project")
DATA = BASE / "data/qiime2_reanalysis"
TRACK_A = BASE / "analysis/results/v11.3.1_supplementary"
OUT = BASE / "manuscript/figures/v11.4_regen"
OUT.mkdir(parents=True, exist_ok=True)

MONTH_COLORS_VIRIDIS = {0: "#440154", 2: "#3b528b", 4: "#21918c", 6: "#fde725"}

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


def parse_new_sample_id(sid):
    """new_G1_R1 → month=0; new_G2_R1 → month=1; ... new_G4 → month=3."""
    if "_G" in sid and "_R" in sid:
        try:
            g = int(sid.split("_G")[1].split("_R")[0])
            r = int(sid.split("_R")[1])
            return g - 1, r  # month = group - 1
        except (ValueError, IndexError):
            return None, None
    return None, None


# ============================================================
# FigS_compositional — CLR PCoA from existing TSV
# ============================================================
def figS_compositional():
    print("FigS_compositional — CLR/Aitchison PCoA")
    candidates = [
        TRACK_A / "Supplementary_Table_S_compositional.tsv",
        TRACK_A / "ancombc_16S.tsv",
        TRACK_A / "ancombc_ITS.tsv",
    ]
    for p in candidates:
        if p.exists():
            print(f"  found: {p.name}")
    # Build from raw 16S + ITS tables: compute CLR + Aitchison PCoA for each marker
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, marker, tab_path, tax_path in [
        (axes[0], "16S", DATA / "16S_old/exported/feature-table-dada2.txt", DATA / "16S_old/exported/taxonomy.tsv"),
        (axes[1], "ITS", DATA / "ITS_old/exported/feature-table-dada2.txt", DATA / "ITS_old/exported/taxonomy.tsv"),
    ]:
        tab = pd.read_csv(tab_path, sep="\t", skiprows=1, index_col=0)
        tax = pd.read_csv(tax_path, sep="\t").set_index("Feature ID")["Taxon"]
        # canonical filter (freq=5): 16S keep Bacteria/Archaea (drop chloro/mito/Eukaryota); ITS keep k__Fungi
        if marker == "16S":
            keep = tax[(tax.str.startswith("d__Bacteria") | tax.str.startswith("d__Archaea"))
                       & ~tax.str.contains("Chloroplast|Mitochondria", case=False, na=False)].index
        else:
            keep = tax[tax.str.startswith("k__Fungi", na=False)].index
        tab = tab.loc[tab.index.intersection(keep)]
        # even-month subset + freq>=5
        cols = [c for c in tab.columns if "_G" in c and int(c.split("_G")[1].split("_R")[0]) in [1, 3, 5, 7]]
        tab = tab[cols]
        tab = tab.loc[tab.sum(axis=1) >= 5]
        # CLR transform (pseudo-count 0.5)
        mat = tab.values + 0.5
        log_mat = np.log(mat)
        clr = log_mat - log_mat.mean(axis=0)
        # Aitchison distance = Euclidean on CLR
        dist = pdist(clr.T, metric="euclidean")
        dm = DistanceMatrix(squareform(dist), ids=tab.columns.tolist())
        pco = pcoa(dm)
        ev = pco.proportion_explained * 100
        coords = pco.samples.copy()
        coords["sample"] = coords.index
        coords["month"] = coords["sample"].apply(lambda s: {1: 0, 3: 2, 5: 4, 7: 6}[int(s.split("_G")[1].split("_R")[0])])
        for m in [0, 2, 4, 6]:
            sub = coords[coords["month"] == m]
            if marker == "16S" and m == 0:
                # 16S 0M may have low reads
                if len(sub) == 0:
                    continue
            ax.scatter(sub["PC1"], sub["PC2"], s=130, color=MONTH_COLORS_VIRIDIS[m],
                       edgecolor="black", linewidths=0.6, label=f"{m}M", zorder=5)
        ax.set_xlabel(f"PCo 1 ({ev.iloc[0]:.1f}%)")
        ax.set_ylabel(f"PCo 2 ({ev.iloc[1]:.1f}%)")
        ax.set_title(f"{marker} — Aitchison (CLR) PCoA")
        ax.legend(title="Month", frameon=False, fontsize=9)
        ax.grid(linestyle=":", alpha=0.4)
    fig.suptitle("Supplementary Figure S_compositional — CLR / Aitchison sensitivity", fontsize=12)
    plt.tight_layout()
    save_both(fig, "FigS_compositional")


# ============================================================
# FigS_RF — Random Forest feature importance
# ============================================================
def figS_RF():
    print("FigS_RF — Random Forest feature importance")
    rf_csv = TRACK_A / "rf_feature_importance_top20.tsv"
    if not rf_csv.exists():
        rf_csv = TRACK_A / "Supplementary_Table_S_RF.tsv"
    if not rf_csv.exists():
        print("  ❌ no RF data")
        return
    df = pd.read_csv(rf_csv, sep="\t")
    print(f"  columns: {df.columns.tolist()}")
    feat_col = next((c for c in df.columns if "asv" in c.lower() or "feature" in c.lower() or "taxon" in c.lower() or "name" in c.lower()), df.columns[0])
    imp_col = next((c for c in df.columns if "importance" in c.lower() or "imp" in c.lower()), df.columns[1])
    df = df.sort_values(imp_col, ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(df)), df[imp_col], color="#5b9bd5", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df[feat_col], fontsize=9)
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Supplementary Figure S_RF — RF feature importance (top 20)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    save_both(fig, "FigS_RF")


# ============================================================
# FigS5 / S6 / S7 — NEW batch
# ============================================================
def figS_NEW_batch():
    print("FigS5/6/7 — NEW batch (independent lot)")
    for marker, tab_path, tax_path, depth, is_its in [
        ("16S", DATA / "16S_new/exported/feature-table.txt", DATA / "16S_new/exported/taxonomy.tsv", 60, False),
        ("ITS", DATA / "ITS_new/exported/feature-table.txt", DATA / "ITS_new/exported/taxonomy.tsv", 100, True),
    ]:
        if not tab_path.exists():
            print(f"  ❌ {marker}: feature table not found ({tab_path})")
            continue
        tab = pd.read_csv(tab_path, sep="\t", skiprows=1, index_col=0)
        tax = pd.read_csv(tax_path, sep="\t").set_index("Feature ID")
        tab = tab[tab.sum(axis=1) >= 5]
        if is_its:
            fungi = tax[tax["Taxon"].str.startswith("k__Fungi", na=False)].index
            tab = tab.loc[tab.index.intersection(fungi)]
        else:
            bad = tax[tax["Taxon"].str.contains("Chloroplast|Mitochondria", na=False, regex=True)].index
            tab = tab.loc[tab.index.difference(bad)]
        # rarefy
        rng = np.random.default_rng(42)
        rar = pd.DataFrame(0, index=tab.index, columns=tab.columns, dtype=int)
        for col in tab.columns:
            counts = tab[col].values.astype(int)
            total = counts.sum()
            if total < depth:
                continue
            rar[col] = rng.multinomial(depth, counts / total)
        rar = rar.loc[(rar.sum(axis=1) > 0), (rar.sum(axis=0) >= depth)]
        # Per-sample meta
        meta = []
        for c in rar.columns:
            m, r = parse_new_sample_id(c)
            if m is not None:
                meta.append((c, m, r))
        meta_df = pd.DataFrame(meta, columns=["sample", "month", "rep"]).sort_values(["month", "rep"])
        rar = rar[meta_df["sample"].values]

        if marker == "16S":
            # FigS5: 16S BC PCoA (NEW batch)
            rel = rar / rar.sum(axis=0)
            bc = pdist(rel.T.values, metric="braycurtis")
            dm = DistanceMatrix(squareform(bc), ids=rar.columns.tolist())
            pco = pcoa(dm)
            ev = pco.proportion_explained * 100
            coords = pco.samples.copy()
            coords["sample"] = coords.index
            coords = coords.merge(meta_df, on="sample")
            fig, ax = plt.subplots(figsize=(8, 6.5))
            new_colors = {0: "#440154", 1: "#414487", 2: "#2a788e", 3: "#22a884"}
            for m in sorted(coords["month"].unique()):
                sub = coords[coords["month"] == m]
                ax.scatter(sub["PC1"], sub["PC2"], s=150, color=new_colors.get(m, "#cccccc"),
                           edgecolor="black", linewidths=0.7, label=f"{m}M", zorder=5)
            ax.set_xlabel(f"PCo 1 ({ev.iloc[0]:.1f}%)")
            ax.set_ylabel(f"PCo 2 ({ev.iloc[1]:.1f}%)")
            ax.set_title("Supplementary Figure S5 — NEW batch 16S Bray-Curtis PCoA")
            ax.legend(title="Month", frameon=False, fontsize=9)
            ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            save_both(fig, "FigS5_NEW_PCoA_16S")
        else:
            # FigS6: ITS stacked bar (NEW batch)
            def get_genus(t):
                if pd.isna(t) or t == "Unassigned":
                    return "Unassigned"
                parts = [p.strip() for p in t.split(";")]
                g = next((p for p in parts if p.startswith("g__")), None)
                if g and len(g) > 3:
                    return g[3:]
                return "Unclassified"
            asv2g = tax.loc[rar.index, "Taxon"].apply(get_genus)
            tab_g = rar.groupby(asv2g).sum()
            rel = tab_g / tab_g.sum(axis=0) * 100
            top12 = rel.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
            rel_plot = rel.loc[top12].copy()
            rel_plot.loc["Other"] = rel.drop(top12).sum(axis=0)
            fig, ax = plt.subplots(figsize=(13, 6))
            labels = [f"M{m}_R{r}" for m, r in zip(meta_df["month"], meta_df["rep"])]
            colors = plt.cm.tab20(np.linspace(0, 1, len(rel_plot)))
            bottom = np.zeros(len(meta_df))
            for i, genus in enumerate(rel_plot.index):
                ax.bar(range(len(meta_df)), rel_plot.loc[genus].values, bottom=bottom,
                       label=genus, color=colors[i], edgecolor="white", linewidth=0.3, width=0.85)
                bottom += rel_plot.loc[genus].values
            ax.set_xticks(range(len(meta_df)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Relative abundance (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Supplementary Figure S6 — NEW batch ITS top-12 genus stacked bar")
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
            plt.tight_layout()
            save_both(fig, "FigS6_NEW_stacked_ITS")

            # FigS7: ITS alpha-diversity (NEW batch)
            obs = (rar > 0).sum(axis=0)
            rel_alpha = rar / rar.sum(axis=0)
            rel_alpha = rel_alpha.replace(0, np.nan)
            shannon = -(rel_alpha * np.log(rel_alpha)).sum(axis=0)
            meta_df["obs"] = [obs[s] for s in meta_df["sample"]]
            meta_df["shannon"] = [shannon[s] for s in meta_df["sample"]]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            new_colors = {0: "#440154", 1: "#414487", 2: "#2a788e", 3: "#22a884"}
            for ax, col, ylab in [(axes[0], "obs", "Observed ASV richness"),
                                  (axes[1], "shannon", "Shannon")]:
                for m in sorted(meta_df["month"].unique()):
                    sub = meta_df[meta_df["month"] == m]
                    ax.scatter([m]*len(sub), sub[col], s=130, color=new_colors.get(m, "#cccccc"),
                               edgecolor="black", linewidths=0.7, zorder=5)
                ax.set_xlabel("Storage month")
                ax.set_ylabel(ylab)
                ax.set_xticks(sorted(meta_df["month"].unique()))
                ax.grid(linestyle=":", alpha=0.4)
            fig.suptitle("Supplementary Figure S7 — NEW batch ITS α-diversity", fontsize=12)
            plt.tight_layout()
            save_both(fig, "FigS7_NEW_alpha_ITS")


def main():
    print(f"Output dir: {OUT}\n")
    funcs = [figS_compositional, figS_RF, figS_NEW_batch]
    for fn in funcs:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ❌ {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\nFiles in {OUT}:")
    for f in sorted(OUT.iterdir()):
        if "compositional" in f.name or "RF" in f.name or "NEW" in f.name or "S5" in f.name or "S6" in f.name or "S7" in f.name:
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
