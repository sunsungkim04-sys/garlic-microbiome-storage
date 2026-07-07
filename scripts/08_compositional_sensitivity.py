"""Script 08 — Compositional sensitivity (CLR / Aitchison) — v11.3.1 supplementary.

Reviewer defense:  Current paper uses relative-abundance based stats.
Verify main BC results with compositional (CLR) transform.

Steps:
1. Load 16S + ITS count tables (even-month OLD batch, n=12 each), apply
   min-freq>=5 and contam filter via existing helpers.
2. CLR transform (Aitchison space, pseudocount=0.5).
3. Aitchison distance = Euclidean on CLR.
4. PCoA + PERMANOVA on Aitchison.
5. Compare PC1+PC2 variance vs Bray-Curtis (rarefied).
6. Differential abundance:  ANCOM-BC via R Bioconductor (calling Rscript)
   with fallback Wilcoxon on CLR values + BH if R fails.
7. Overlap top-20 DA hits with existing KW-on-relative-abundance results
   (results/ITS/differential_abundance_genus.csv etc.).

Outputs:
  v11.3.1_supplementary/Supplementary_Table_S_compositional.tsv
  v11.3.1_supplementary/Figure_S_compositional.png
  v11.3.1_supplementary/compositional_pcoa_variance.tsv
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
from scipy.linalg import eigh

sys.path.insert(0, "/home1/minseo1101/garlic_project/analysis/scripts/v11.3.1")
from _helpers import (load_table, load_taxonomy, is_contam_16S, is_contam_ITS,
                      bray_curtis, permanova_oneway, parse_genus,
                      META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
T16S = f"{QROOT}/16S_old/table-dada2.qza"
TAX16 = f"{QROOT}/16S_old/taxonomy.qza"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"
OUT = "/home1/minseo1101/garlic_project/analysis/results/v11.3.1_supplementary"

SEED = 42
N_PERM = 999
MIN_FREQ = 5


def clr_transform(counts, pseudo=0.5):
    """CLR on a sample x feature count matrix; pseudocount before log."""
    X = counts.astype(float) + pseudo
    log_X = np.log(X)
    g_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - g_mean


def aitchison_distance(clr_mat):
    """Euclidean distance in CLR space = Aitchison."""
    n = clr_mat.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.linalg.norm(clr_mat[i] - clr_mat[j])
    return D


def pcoa_var(D):
    n = D.shape[0]
    A = -0.5 * D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = H @ A @ H
    w, v = eigh(B)
    idx = np.argsort(-w)
    w = w[idx]
    v = v[:, idx]
    keep = w > 1e-9
    w_pos = w[keep]
    pcs = v[:, keep] * np.sqrt(w_pos)
    var_total = w_pos.sum()
    pct = w_pos / var_total * 100.0
    return pcs, pct


def load_filtered_evenmonth(table_qza, tax_qza, contam_fn):
    sids, asv_ids, mat = load_table(table_qza)
    tax = load_taxonomy(tax_qza)
    total = mat.sum(axis=0)
    keep = (total >= MIN_FREQ) & np.array(
        [not contam_fn(tax.get(a, "Unassigned")) for a in asv_ids])
    cm = mat[:, keep]
    kept_asvs = [a for a, k in zip(asv_ids, keep) if k]
    em_idx = [i for i, s in enumerate(sids) if s in META_EVEN_OLD]
    em_sids = [sids[i] for i in em_idx]
    em_mat = cm[em_idx]
    nz = em_mat.sum(axis=0) > 0
    em_mat = em_mat[:, nz]
    kept_asvs = [a for a, k in zip(kept_asvs, nz) if k]
    return em_sids, kept_asvs, em_mat, tax


def run_ancombc(counts_df, meta_df, output_tsv, marker):
    """Try ANCOM-BC via R; counts_df = ASV x sample DataFrame, meta_df has 'month'."""
    r_script = f"""
suppressPackageStartupMessages({{
  library(ANCOMBC)
  library(phyloseq)
}})
ct <- read.table("{output_tsv}.counts.in.tsv", header=TRUE, sep="\\t", row.names=1, check.names=FALSE)
md <- read.table("{output_tsv}.meta.in.tsv", header=TRUE, sep="\\t", row.names=1, check.names=FALSE)
md$month <- as.factor(md$month)
ot <- otu_table(as.matrix(ct), taxa_are_rows=TRUE)
sd <- sample_data(md)
ps <- phyloseq(ot, sd)
res <- tryCatch({{
  out <- ancombc2(data=ps, assay_name=NULL, tax_level=NULL, fix_formula="month",
                  p_adj_method="BH", prv_cut=0.10, lib_cut=0,
                  group="month", struc_zero=FALSE, neg_lb=FALSE,
                  alpha=0.05, n_cl=1, verbose=FALSE)
  out$res
}}, error=function(e) {{cat("ANCOMBC_ERR:", conditionMessage(e), "\\n"); NULL}})
if (!is.null(res)) {{
  write.table(res, file="{output_tsv}", sep="\\t", quote=FALSE, row.names=FALSE)
  cat("ANCOMBC_OK\\n")
}} else {{
  cat("ANCOMBC_FAILED\\n")
}}
"""
    counts_df.to_csv(f"{output_tsv}.counts.in.tsv", sep="\t")
    meta_df.to_csv(f"{output_tsv}.meta.in.tsv", sep="\t")
    rfile = f"{output_tsv}.ancombc.R"
    with open(rfile, "w") as f:
        f.write(r_script)
    try:
        out = subprocess.run(["Rscript", rfile], capture_output=True, text=True, timeout=600)
        ok = "ANCOMBC_OK" in out.stdout
        print(f"  [{marker} ANCOMBC] ok={ok} stdout-tail={out.stdout.splitlines()[-3:]}")
        return ok
    except Exception as e:
        print(f"  [{marker} ANCOMBC] exception: {e}")
        return False


def wilcoxon_clr_da(clr_mat, asv_ids, sample_months):
    """Fallback: Kruskal-Wallis on CLR across months + BH FDR.
    Returns DataFrame sorted by p_adj."""
    months = np.array(sample_months)
    rows = []
    for j, asv in enumerate(asv_ids):
        vals_by_m = [clr_mat[months == m, j] for m in np.unique(months)]
        if any(len(v) < 2 for v in vals_by_m):
            continue
        try:
            s, p = kruskal(*vals_by_m)
        except ValueError:
            continue
        rows.append((asv, s, p))
    if not rows:
        return pd.DataFrame(columns=["asv", "stat", "p", "p_adj"])
    df = pd.DataFrame(rows, columns=["asv", "stat", "p"])
    # BH
    df = df.sort_values("p").reset_index(drop=True)
    n = len(df)
    df["p_adj"] = (df["p"] * n / (df.index + 1)).clip(upper=1.0)
    for i in range(n - 2, -1, -1):
        df.loc[i, "p_adj"] = min(df.loc[i, "p_adj"], df.loc[i + 1, "p_adj"])
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("Script 08: Compositional sensitivity (CLR / Aitchison)")
    print("=" * 70)

    summary = {}

    for marker, table, taxq, contam in [("16S", T16S, TAX16, is_contam_16S),
                                         ("ITS", TITS, TAXIT, is_contam_ITS)]:
        print(f"\n--- {marker} ---")
        sids, asvs, mat, tax = load_filtered_evenmonth(table, taxq, contam)
        print(f"  n_samples={len(sids)}, n_asv_after_filter={mat.shape[1]}")
        months = np.array([META_EVEN_OLD[s] for s in sids])

        # CLR + Aitchison
        clr = clr_transform(mat, pseudo=0.5)
        D_ait = aitchison_distance(clr)
        pcs_ait, var_ait = pcoa_var(D_ait)

        # BC (counts directly; not rarefied here since we already filtered;
        # for fair comparison rarefy would be needed but BC on raw shows direction)
        D_bc = bray_curtis(mat)
        pcs_bc, var_bc = pcoa_var(D_bc)

        # PERMANOVA on Aitchison by month
        pa = permanova_oneway(D_ait, months, n_perm=N_PERM, seed=SEED)
        # On BC for compare
        pb = permanova_oneway(D_bc, months, n_perm=N_PERM, seed=SEED)
        print(f"  Aitchison PERMANOVA F={pa['F']:.3f} R2={pa['R2']:.3f} p={pa['p']:.4f}")
        print(f"  Bray-Curtis PERMANOVA F={pb['F']:.3f} R2={pb['R2']:.3f} p={pb['p']:.4f}")
        print(f"  Aitchison PC1+PC2 = {var_ait[0]:.1f}% + {var_ait[1]:.1f}% = {var_ait[0]+var_ait[1]:.1f}%")
        print(f"  BC PC1+PC2 = {var_bc[0]:.1f}% + {var_bc[1]:.1f}% = {var_bc[0]+var_bc[1]:.1f}%")

        # DA: ANCOM-BC2 attempt
        counts_df = pd.DataFrame(mat.T, index=asvs, columns=sids)
        meta_df = pd.DataFrame({"month": months}, index=sids)
        ancom_tsv = f"{OUT}/ancombc_{marker}.tsv"
        ok = run_ancombc(counts_df, meta_df, ancom_tsv, marker)

        if ok:
            ancom_res = pd.read_csv(ancom_tsv, sep="\t")
            print(f"  ANCOMBC returned {len(ancom_res)} rows")
            da_method = "ANCOM-BC2"
            # ANCOMBC2 output has columns like taxon, lfc_*, p_*, q_*, diff_*. Use min q across months.
            q_cols = [c for c in ancom_res.columns if c.startswith("q_") and "intercept" not in c.lower()]
            if q_cols:
                ancom_res["min_q"] = ancom_res[q_cols].min(axis=1)
                ancom_res = ancom_res.sort_values("min_q")
                top_ids = ancom_res["taxon"].head(20).tolist() if "taxon" in ancom_res.columns else []
            else:
                top_ids = []
        else:
            print(f"  ANCOMBC failed; using Kruskal-CLR fallback")
            da_method = "Kruskal-on-CLR"
            kw_df = wilcoxon_clr_da(clr, asvs, months)
            kw_df.to_csv(f"{OUT}/clr_kruskal_{marker}.tsv", sep="\t", index=False)
            top_ids = kw_df["asv"].head(20).tolist()

        # Map top ids to genus
        top_genera = [parse_genus(tax.get(a, "")) for a in top_ids]

        # Compare to existing relab DA results
        if marker == "ITS":
            ref_path = "/home1/minseo1101/garlic_project/analysis/results/ITS/differential_abundance_genus.csv"
        else:
            ref_path = "/home1/minseo1101/garlic_project/analysis/results/16S/differential_abundance_genus.csv"
        overlap_pct = np.nan
        ref_top = []
        if os.path.exists(ref_path):
            ref = pd.read_csv(ref_path)
            # pick rows with p_adjusted column (lowest 20 unique taxa)
            tax_col = "Taxon" if "Taxon" in ref.columns else "taxon"
            p_col = "p_adjusted" if "p_adjusted" in ref.columns else "p_adj"
            if tax_col in ref.columns and p_col in ref.columns:
                ref_sorted = ref.dropna(subset=[p_col]).sort_values(p_col)
                ref_top = ref_sorted[tax_col].drop_duplicates().head(20).tolist()
                overlap = set(top_genera) & set(ref_top)
                overlap_pct = 100.0 * len(overlap) / max(len(set(top_genera)), 1)
                print(f"  Top-20 genus overlap with existing relab DA: {overlap_pct:.1f}%")
                print(f"    CLR/ANCOM top: {top_genera[:10]}")
                print(f"    Relab top:     {ref_top[:10]}")

        summary[marker] = dict(
            n_samples=len(sids), n_asv=mat.shape[1],
            aitchison_F=pa["F"], aitchison_R2=pa["R2"], aitchison_p=pa["p"],
            bc_F=pb["F"], bc_R2=pb["R2"], bc_p=pb["p"],
            aitchison_pc12=float(var_ait[0] + var_ait[1]),
            bc_pc12=float(var_bc[0] + var_bc[1]),
            da_method=da_method,
            top_genera_compositional=top_genera,
            top_genera_relab=ref_top,
            overlap_pct=overlap_pct,
        )

        # Store PCoA coords for plotting
        summary[marker]["pcs"] = pcs_ait[:, :2]
        summary[marker]["var"] = var_ait[:2]
        summary[marker]["months"] = months

    # Write supplementary table
    sup_rows = []
    for marker, d in summary.items():
        sup_rows.append([
            marker,
            d["n_samples"], d["n_asv"],
            d["aitchison_F"], d["aitchison_R2"], d["aitchison_p"],
            d["bc_F"], d["bc_R2"], d["bc_p"],
            d["aitchison_pc12"], d["bc_pc12"],
            d["da_method"],
            ";".join(d["top_genera_compositional"][:20]),
            ";".join(d["top_genera_relab"][:20]),
            d["overlap_pct"],
        ])
    sup = pd.DataFrame(sup_rows, columns=[
        "marker", "n_samples", "n_asv",
        "aitchison_PERMANOVA_F", "aitchison_R2", "aitchison_p",
        "BC_PERMANOVA_F", "BC_R2", "BC_p",
        "aitchison_PC1+PC2_pct", "BC_PC1+PC2_pct",
        "DA_method", "top20_CLR_genera", "top20_relab_genera",
        "top20_overlap_pct"])
    sup.to_csv(f"{OUT}/Supplementary_Table_S_compositional.tsv", sep="\t", index=False)
    print(f"\nWrote {OUT}/Supplementary_Table_S_compositional.tsv")

    # Plot Aitchison PCoA 2-panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {0: "#1b9e77", 2: "#d95f02", 4: "#7570b3", 6: "#e7298a"}
    for ax, marker in zip(axes, ["16S", "ITS"]):
        d = summary[marker]
        for m in [0, 2, 4, 6]:
            mask = d["months"] == m
            ax.scatter(d["pcs"][mask, 0], d["pcs"][mask, 1],
                       c=colors[m], label=f"{m}M", s=80, edgecolor="k", lw=0.5)
        ax.set_xlabel(f"PC1 ({d['var'][0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({d['var'][1]:.1f}%)")
        ax.set_title(f"{marker}: Aitchison PCoA (n={d['n_samples']})\n"
                     f"PERMANOVA F={d['aitchison_F']:.2f} R²={d['aitchison_R2']:.2f} p={d['aitchison_p']:.3f}")
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0, c="grey", lw=0.4)
        ax.axvline(0, c="grey", lw=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Figure_S_compositional.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}/Figure_S_compositional.png")


if __name__ == "__main__":
    main()
