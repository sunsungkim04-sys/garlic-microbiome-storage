"""Script 3 (★★) — 16S OLD rarefaction depth sweep.

Depths 50/100/130/200/500/1000. 0M Escherichia contam의 PERMANOVA F 기여 정량.

Output:
  Attachments_investigation/16S_depth_sweep.csv
  Attachments_investigation/16S_depth_sweep.png
"""
import os
import sys
import csv
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (load_table, load_taxonomy, is_contam_16S,
                      rarefy_counts_to_depth, alpha_metrics,
                      bray_curtis, permanova_oneway, permdisp,
                      META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
OUT = f"{QROOT}/Attachments_investigation"
TABLE = f"{QROOT}/16S_old/table-dada2.qza"
TAX = f"{QROOT}/16S_old/taxonomy.qza"
TREE_QZA = f"{QROOT}/16S_old/rooted-tree.qza"

DEPTHS = [50, 100, 130, 200, 500, 1000]
N_PERM = 999
SEED = 42


def load_tree(qza):
    with zipfile.ZipFile(qza) as z:
        nwk_name = [n for n in z.namelist() if n.endswith("/data/tree.nwk")][0]
        return z.read(nwk_name).decode()


def unifrac_unweighted(mat, asv_ids, newick):
    """Unweighted UniFrac via skbio."""
    from skbio import TreeNode
    from io import StringIO
    from skbio.diversity import beta_diversity
    tree = TreeNode.read(StringIO(newick))
    # Map ASVs to those present in tree
    leaf_names = {t.name for t in tree.tips()}
    keep = [i for i, a in enumerate(asv_ids) if a in leaf_names]
    M = mat[:, keep]
    ids = [asv_ids[i] for i in keep]
    pres = (M > 0).astype(int)
    # build sample IDs as needed for the call
    D = beta_diversity("unweighted_unifrac", pres, ids=None,
                       taxa=ids, tree=tree, validate=False)
    return D.data


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading...")
    sample_ids, asv_ids, mat = load_table(TABLE)
    tax = load_taxonomy(TAX)

    # min-freq=5 + taxa filter
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not is_contam_16S(tax.get(a, "Unassigned")) for a in asv_ids])
    clean_mat = mat[:, keep]
    clean_asvs = [a for a, k in zip(asv_ids, keep) if k]
    print(f"clean: {clean_mat.shape}")

    # Even-month subset
    even_idx = [i for i, s in enumerate(sample_ids) if s in META_EVEN_OLD]
    em_samples = [sample_ids[i] for i in even_idx]
    em = clean_mat[even_idx]
    nz = em.sum(axis=0) > 0
    em = em[:, nz]
    em_asvs = [clean_asvs[i] for i, z in enumerate(nz) if z]
    groups_all = np.array([META_EVEN_OLD[s] for s in em_samples])
    print(f"even-month clean: {em.shape}")
    print("sample-read totals:")
    for s, r in sorted(zip(em_samples, em.sum(axis=1).astype(int))):
        print(f"  {s}: {r}")

    newick = load_tree(TREE_QZA)

    rows = []
    for depth in DEPTHS:
        # Identify samples passing depth
        sample_tot = em.sum(axis=1)
        ok = sample_tot >= depth
        kept_samples = [em_samples[i] for i, k in enumerate(ok) if k]
        kept_idx = np.where(ok)[0]
        groups = groups_all[ok]
        sub_em = em[ok]
        n_retained = len(kept_samples)
        if n_retained < 4 or len(np.unique(groups)) < 2:
            print(f"depth={depth}: too few samples ({n_retained}) — skip")
            rows.append([depth, n_retained, np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, ",".join(kept_samples)])
            continue
        # rarefy
        rng = np.random.default_rng(SEED)
        rare = np.zeros_like(sub_em)
        for i, row in enumerate(sub_em):
            rare[i] = rarefy_counts_to_depth(row, depth, rng)
        D_bc = bray_curtis(rare)
        perm_bc = permanova_oneway(D_bc, groups, n_perm=N_PERM, seed=SEED)
        disp_bc = permdisp(D_bc, groups, n_perm=N_PERM, seed=SEED)

        # UniFrac
        try:
            D_uf = unifrac_unweighted(rare, em_asvs, newick)
            perm_uf = permanova_oneway(D_uf, groups, n_perm=N_PERM, seed=SEED)
            disp_uf = permdisp(D_uf, groups, n_perm=N_PERM, seed=SEED)
            F_uf, p_uf, dF_uf, dp_uf = perm_uf["F"], perm_uf["p"], disp_uf["F"], disp_uf["p"]
        except Exception as e:
            print(f"  UniFrac failed at depth={depth}: {e}")
            F_uf = p_uf = dF_uf = dp_uf = np.nan

        # Alpha + ranking
        alpha = [alpha_metrics(r) for r in rare]
        obs = np.array([a[0] for a in alpha])
        sh = np.array([a[1] for a in alpha])
        month_obs = {}
        for m in np.unique(groups):
            month_obs[int(m)] = obs[groups == m].mean()
        obs_rank = sorted(month_obs, key=month_obs.get, reverse=True)
        ranking_str = ">".join(f"{m}M" for m in obs_rank)
        print(f"depth={depth}  n={n_retained}  "
              f"BC F={perm_bc['F']:.2f} p={perm_bc['p']:.3f} disp={disp_bc['F']:.2f} dp={disp_bc['p']:.3f}  "
              f"UniFrac F={F_uf:.2f}  rank={ranking_str}")
        rows.append([depth, n_retained,
                     perm_bc["F"], perm_bc["p"], disp_bc["F"], disp_bc["p"],
                     F_uf, p_uf, dF_uf, dp_uf, ranking_str,
                     ",".join(kept_samples)])

    # CSV
    csv_path = f"{OUT}/16S_depth_sweep.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["depth", "n_retained", "BC_F", "BC_p", "PERMDISP_BC_F", "PERMDISP_BC_p",
                    "UniFrac_F", "UniFrac_p", "PERMDISP_UF_F", "PERMDISP_UF_p",
                    "alpha_obs_ranking", "kept_samples"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Plot
    depths_pl = [r[0] for r in rows]
    bc_F = [r[2] for r in rows]
    uf_F = [r[6] for r in rows]
    bc_disp = [r[4] for r in rows]
    n_ret = [r[1] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(depths_pl, bc_F, "o-", label="BC F", color="#2c7fb8")
    axes[0].plot(depths_pl, uf_F, "s-", label="unweighted UniFrac F", color="#1a9850")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Rarefy depth")
    axes[0].set_ylabel("PERMANOVA F")
    axes[0].set_title("PERMANOVA F vs depth")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(depths_pl, bc_disp, "o-", color="#d95f0e")
    axes[1].axhline(1.0, ls="--", color="grey")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Rarefy depth")
    axes[1].set_ylabel("PERMDISP F")
    axes[1].set_title("BC PERMDISP F vs depth")
    axes[1].grid(alpha=0.3)

    axes[2].plot(depths_pl, n_ret, "o-", color="#7570b3")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Rarefy depth")
    axes[2].set_ylabel("# samples retained")
    axes[2].set_title("Samples retained")
    axes[2].grid(alpha=0.3)

    fig.suptitle("v11.3.1 16S OLD depth sweep", y=1.02)
    plt.tight_layout()
    png_path = f"{OUT}/16S_depth_sweep.png"
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
