"""Script 11 — Network bootstrap stability.

The cross-kingdom co-occurrence network (§3.4) has 60 edges
(|rho|>0.6, FDR<0.05).  Reviewer concern: are edges stable to
resampling?

Method:
  - Recreate the genus x sample matrix as in
    scripts/cross_kingdom_analysis.py (top-15 bacterial & top-15 fungal
    genera, relative abundance, matched sample pairs).
  - Original edges: load from
    results/cross_kingdom/network_edges.csv  (60 edges).
  - Bootstrap N=100 with-replacement resamples of the matched
    sample set; refit Spearman on top-15×top-15; apply BH;
    record edges that satisfy |rho|>0.6 AND q<0.05.
  - For each original edge, stability_fraction = # bootstraps where
    edge re-appears / N.

Output:
  v11.3.1_supplementary/Supplementary_Table_S_network_bootstrap.tsv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/home1/minseo1101/garlic_project/analysis/scripts/v11.3.1")
from _helpers import (load_table, load_taxonomy, is_contam_16S, is_contam_ITS,
                      parse_genus, META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
T16S = f"{QROOT}/16S_old/table-dada2.qza"
TAX16 = f"{QROOT}/16S_old/taxonomy.qza"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"
ORIG = "/home1/minseo1101/garlic_project/analysis/results/cross_kingdom/network_edges.csv"
OUT = "/home1/minseo1101/garlic_project/analysis/results/v11.3.1_supplementary"

N_BOOT = 100
RHO_THRESH = 0.6
Q_THRESH = 0.05
SEED = 42


def load_genus_matrix(table_qza, tax_qza, contam_fn):
    sids, asvs, mat = load_table(table_qza)
    tax = load_taxonomy(tax_qza)
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not contam_fn(tax.get(a, "")) for a in asvs])
    mat = mat[:, keep]
    asvs = [a for a, k in zip(asvs, keep) if k]
    em_idx = [i for i, s in enumerate(sids) if s in META_EVEN_OLD]
    em_sids = [sids[i] for i in em_idx]
    em_mat = mat[em_idx]
    # Genus-level
    gnames = [parse_genus(tax.get(a, "")) for a in asvs]
    df = pd.DataFrame(em_mat.T, index=gnames)
    df = df.groupby(level=0).sum()
    df.columns = em_sids
    # relative abundance per sample
    df = df.div(df.sum(axis=0), axis=1)
    return df


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = ranked
    return out


def build_network(bact, fungi):
    """bact, fungi = top-15 relative-abundance DataFrames, same columns.
    Returns list of (b, f, rho, q) for all pairs satisfying threshold."""
    rhos = np.zeros((bact.shape[0], fungi.shape[0]))
    pvals = np.ones_like(rhos)
    for i, bn in enumerate(bact.index):
        bv = bact.loc[bn].values
        for j, fn in enumerate(fungi.index):
            fv = fungi.loc[fn].values
            r, p = spearmanr(bv, fv)
            rhos[i, j] = r
            pvals[i, j] = p if not np.isnan(p) else 1.0
    q = bh_fdr(pvals.flatten()).reshape(pvals.shape)
    edges = []
    for i in range(bact.shape[0]):
        for j in range(fungi.shape[0]):
            edges.append((bact.index[i], fungi.index[j], rhos[i, j], q[i, j]))
    return edges


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading bacterial + fungal genus matrices …")
    bact = load_genus_matrix(T16S, TAX16, is_contam_16S)
    fungi = load_genus_matrix(TITS, TAXIT, is_contam_ITS)

    common_samples = [s for s in bact.columns if s in fungi.columns]
    print(f"  Common samples: {len(common_samples)}")
    bact = bact[common_samples]
    fungi = fungi[common_samples]

    # Top-15 by mean abundance (mirrors cross_kingdom_analysis.py)
    bact_top = bact.loc[bact.mean(axis=1).nlargest(15).index]
    fungi_top = fungi.loc[fungi.mean(axis=1).nlargest(15).index]
    print(f"  Top-15 bacteria: {list(bact_top.index)}")
    print(f"  Top-15 fungi:   {list(fungi_top.index)}")

    # Build original network from current data (recompute to get current edges)
    print("\nBuilding observed network …")
    obs_edges = build_network(bact_top, fungi_top)
    obs_sig = [(b, f, r, q) for (b, f, r, q) in obs_edges
               if abs(r) > RHO_THRESH and q < Q_THRESH]
    print(f"  Significant edges (|rho|>{RHO_THRESH}, q<{Q_THRESH}): {len(obs_sig)}")

    # Also load published 60 edges
    if os.path.exists(ORIG):
        orig_df = pd.read_csv(ORIG)
        print(f"  Original cross_kingdom/network_edges.csv has {len(orig_df)} rows")
    else:
        orig_df = None

    obs_edge_keys = {frozenset([b, f]): (r, q) for (b, f, r, q) in obs_sig}

    # Bootstrap
    print(f"\nBootstrapping N={N_BOOT} …")
    rng = np.random.default_rng(SEED)
    n = len(common_samples)
    boot_counts = {k: 0 for k in obs_edge_keys}
    for b_iter in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        bb = bact_top.iloc[:, idx]
        ff = fungi_top.iloc[:, idx]
        # need at least one unique sample variation; if all idx same -> skip
        if len(set(idx.tolist())) < 3:
            continue
        bedges = build_network(bb, ff)
        bsig = {frozenset([b, f]) for (b, f, r, q) in bedges
                if abs(r) > RHO_THRESH and q < Q_THRESH}
        for k in obs_edge_keys:
            if k in bsig:
                boot_counts[k] += 1
        if (b_iter + 1) % 20 == 0:
            print(f"  iter {b_iter+1}/{N_BOOT}")

    rows = []
    for k, (rho_obs, q_obs) in obs_edge_keys.items():
        parts = list(k)
        t1, t2 = parts[0], parts[1]
        frac = boot_counts[k] / N_BOOT
        rows.append((f"{t1}--{t2}", t1, t2, rho_obs, q_obs, boot_counts[k], frac))

    out_df = pd.DataFrame(rows, columns=["edge_id", "taxon1", "taxon2",
                                          "original_rho", "original_q",
                                          "n_bootstraps_sig", "stability_fraction"])
    out_df = out_df.sort_values("stability_fraction", ascending=False)
    out_df.to_csv(f"{OUT}/Supplementary_Table_S_network_bootstrap.tsv", sep="\t", index=False)
    print(f"\nWrote {OUT}/Supplementary_Table_S_network_bootstrap.tsv")
    stable = (out_df["stability_fraction"] >= 0.80).sum()
    print(f"  Edges stable at ≥80%: {stable}/{len(out_df)} ({100*stable/max(1,len(out_df)):.1f}%)")
    print(f"  Edges stable at ≥50%: {(out_df['stability_fraction']>=0.5).sum()}/{len(out_df)}")


if __name__ == "__main__":
    main()
