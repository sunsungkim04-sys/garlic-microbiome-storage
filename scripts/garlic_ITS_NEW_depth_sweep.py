"""Script 5 (★) — ITS NEW BC dispersion audit via depth sweep.

NEW F=1.11 (NS): real lot variation or low-read dispersion artifact?

Output:
  Attachments_investigation/ITS_NEW_depth_sweep.csv
  Attachments_investigation/ITS_NEW_depth_sweep.png
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (load_table, load_taxonomy, is_contam_ITS,
                      rarefy_counts_to_depth, alpha_metrics,
                      bray_curtis, jaccard_binary,
                      permanova_oneway, permdisp, META_NEW)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
OUT = f"{QROOT}/Attachments_investigation"
TABLE = f"{QROOT}/ITS_new/table-dada2.qza"
TAX = f"{QROOT}/ITS_new/taxonomy.qza"

DEPTHS = [50, 100, 200, 500]
N_PERM = 999
SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading ITS NEW...")
    sample_ids, asv_ids, mat = load_table(TABLE)
    tax = load_taxonomy(TAX)

    # min-freq=5 + k__Fungi keep + Unassigned drop
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not is_contam_ITS(tax.get(a, "Unassigned")) for a in asv_ids])
    clean_mat = mat[:, keep]
    clean_asvs = [a for a, k in zip(asv_ids, keep) if k]
    print(f"clean: {clean_mat.shape}  (NEW raw {mat.shape})")

    new_idx = [i for i, s in enumerate(sample_ids) if s in META_NEW]
    new_samples = [sample_ids[i] for i in new_idx]
    cm = clean_mat[new_idx]
    nz = cm.sum(axis=0) > 0
    cm = cm[:, nz]
    groups_all = np.array([META_NEW[s] for s in new_samples])
    print(f"NEW clean: {cm.shape}")
    print("sample-read totals:")
    for s, r in sorted(zip(new_samples, cm.sum(axis=1).astype(int))):
        print(f"  {s}: {r}")

    rows = []
    for depth in DEPTHS:
        sample_tot = cm.sum(axis=1)
        ok = sample_tot >= depth
        kept = [new_samples[i] for i, k in enumerate(ok) if k]
        groups = groups_all[ok]
        sub = cm[ok]
        n = len(kept)
        if n < 4 or len(np.unique(groups)) < 2:
            print(f"depth={depth}: too few samples ({n}) — skip")
            rows.append([depth, n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, ",".join(kept)])
            continue
        rng = np.random.default_rng(SEED)
        rare = np.zeros_like(sub)
        for i, row in enumerate(sub):
            rare[i] = rarefy_counts_to_depth(row, depth, rng)
        D_bc = bray_curtis(rare)
        D_ja = jaccard_binary(rare)
        perm_bc = permanova_oneway(D_bc, groups, n_perm=N_PERM, seed=SEED)
        disp_bc = permdisp(D_bc, groups, n_perm=N_PERM, seed=SEED)
        perm_ja = permanova_oneway(D_ja, groups, n_perm=N_PERM, seed=SEED)
        disp_ja = permdisp(D_ja, groups, n_perm=N_PERM, seed=SEED)
        print(f"depth={depth}  n={n}  BC F={perm_bc['F']:.2f} p={perm_bc['p']:.3f} disp={disp_bc['F']:.2f} dp={disp_bc['p']:.3f}  "
              f"Jaccard F={perm_ja['F']:.2f} p={perm_ja['p']:.3f}")
        rows.append([depth, n,
                     perm_bc["F"], perm_bc["p"], disp_bc["F"], disp_bc["p"],
                     perm_ja["F"], perm_ja["p"], disp_ja["F"], disp_ja["p"],
                     ",".join(kept)])

    csv_path = f"{OUT}/ITS_NEW_depth_sweep.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["depth", "n_retained",
                    "BC_F", "BC_p", "PERMDISP_BC_F", "PERMDISP_BC_p",
                    "Jaccard_F", "Jaccard_p", "PERMDISP_Jac_F", "PERMDISP_Jac_p",
                    "kept_samples"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    depths_pl = [r[0] for r in rows]
    bc_F = [r[2] for r in rows]
    bc_disp = [r[4] for r in rows]
    ja_F = [r[6] for r in rows]
    n_ret = [r[1] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(depths_pl, bc_F, "o-", label="BC", color="#2c7fb8")
    axes[0].plot(depths_pl, ja_F, "s-", label="Jaccard", color="#1a9850")
    axes[0].set_xlabel("Rarefy depth")
    axes[0].set_ylabel("PERMANOVA F")
    axes[0].set_title("ITS NEW PERMANOVA F")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(depths_pl, bc_disp, "o-", color="#d95f0e")
    axes[1].axhline(1.0, ls="--", color="grey")
    axes[1].set_xlabel("Rarefy depth")
    axes[1].set_ylabel("PERMDISP F")
    axes[1].set_title("ITS NEW BC PERMDISP F")
    axes[1].grid(alpha=0.3)

    axes[2].plot(depths_pl, n_ret, "o-", color="#7570b3")
    axes[2].set_xlabel("Rarefy depth")
    axes[2].set_ylabel("# samples retained")
    axes[2].set_title("Samples retained")
    axes[2].grid(alpha=0.3)

    fig.suptitle("v11.3.1 ITS NEW depth sweep — lot variation vs dispersion audit", y=1.02)
    plt.tight_layout()
    png_path = f"{OUT}/ITS_NEW_depth_sweep.png"
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
