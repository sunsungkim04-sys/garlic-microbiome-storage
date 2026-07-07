"""Script 6 (★) — Stage-wise cross-kingdom Mantel.

Which transition (0→2, 2→4, 4→6, 0→6) is most strongly synchronized?

Output:
  Attachments_investigation/stagewise_mantel.csv
  Attachments_investigation/stagewise_mantel.png
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (load_table, load_taxonomy, is_contam_16S, is_contam_ITS,
                      rarefy_counts_to_depth, bray_curtis,
                      mantel_spearman, META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
OUT = f"{QROOT}/Attachments_investigation"
T16S = f"{QROOT}/16S_old/table-dada2.qza"
TAX16 = f"{QROOT}/16S_old/taxonomy.qza"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"

DEPTH_16S = 130
DEPTH_ITS = 200
N_PERM = 999
SEED = 42

STAGES = {
    "A_0_2M": ([0, 2], "0 → 2 M (early invasion)"),
    "B_2_4M": ([2, 4], "2 → 4 M (mid)"),
    "C_4_6M": ([4, 6], "4 → 6 M (late)"),
    "D_0_6M": ([0, 6], "0 → 6 M (full contrast)"),
}


def get_rarefied_bc(table, tax_qza, depth, contam_fn):
    sample_ids, asv_ids, mat = load_table(table)
    tax = load_taxonomy(tax_qza)
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not contam_fn(tax.get(a, "Unassigned")) for a in asv_ids])
    cm = mat[:, keep]
    even_idx = [i for i, s in enumerate(sample_ids) if s in META_EVEN_OLD]
    em_samples = [sample_ids[i] for i in even_idx]
    em = cm[even_idx]
    nz = em.sum(axis=0) > 0
    em = em[:, nz]
    rng = np.random.default_rng(SEED)
    rare = np.zeros_like(em)
    keep_mask = np.ones(em.shape[0], dtype=bool)
    for i, row in enumerate(em):
        rr = rarefy_counts_to_depth(row, depth, rng)
        if rr is None:
            keep_mask[i] = False
        else:
            rare[i] = rr
    em_samples = [s for s, k in zip(em_samples, keep_mask) if k]
    rare = rare[keep_mask]
    return em_samples, rare


def subset_distance(samples, dist, target_samples):
    idx = [samples.index(s) for s in target_samples if s in samples]
    sub_samples = [samples[i] for i in idx]
    sub = dist[np.ix_(idx, idx)]
    return sub_samples, sub


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Building rarefied tables...")
    s16, r16 = get_rarefied_bc(T16S, TAX16, DEPTH_16S, is_contam_16S)
    sit, rit = get_rarefied_bc(TITS, TAXIT, DEPTH_ITS, is_contam_ITS)
    print(f"16S samples: {s16}")
    print(f"ITS samples: {sit}")

    common = [s for s in s16 if s in sit]
    print(f"Common (rarefied both): {len(common)}")

    # Reorder
    r16_sub = r16[[s16.index(s) for s in common]]
    rit_sub = rit[[sit.index(s) for s in common]]

    D16 = bray_curtis(r16_sub)
    DIT = bray_curtis(rit_sub)

    rows = []
    for tag, (months, label) in STAGES.items():
        sel = [s for s in common if META_EVEN_OLD[s] in months]
        idx = [common.index(s) for s in sel]
        D16s = D16[np.ix_(idx, idx)]
        DITs = DIT[np.ix_(idx, idx)]
        if len(sel) < 4:
            rows.append([tag, label, len(sel), np.nan, np.nan])
            continue
        man = mantel_spearman(D16s, DITs, n_perm=N_PERM, seed=SEED)
        # mean Procrustes residual approximation: average of off-diagonal abs diff
        iu = np.triu_indices(len(sel), k=1)
        mean_diff = float(np.mean(np.abs(D16s[iu] - DITs[iu])))
        print(f"{tag} (n={len(sel)}): rho={man['rho']:.3f}  p={man['p']:.4f}  mean|d16-dITS|={mean_diff:.3f}")
        rows.append([tag, label, len(sel), man["rho"], man["p"], mean_diff])

    # Full
    man_full = mantel_spearman(D16, DIT, n_perm=N_PERM, seed=SEED)
    iu = np.triu_indices(len(common), k=1)
    full_diff = float(np.mean(np.abs(D16[iu] - DIT[iu])))
    rows.insert(0, ["FULL", "All 12 samples", len(common),
                    man_full["rho"], man_full["p"], full_diff])
    print(f"FULL (n={len(common)}): rho={man_full['rho']:.3f}  p={man_full['p']:.4f}")

    csv_path = f"{OUT}/stagewise_mantel.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["stage", "label", "n", "spearman_rho", "p_perm",
                    "mean_abs_BC_diff_16S_vs_ITS"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Plot bar
    stages = [r[0] for r in rows]
    rhos = [r[3] for r in rows]
    ps = [r[4] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stages, rhos, color=["#999"] + ["#2c7fb8", "#7fb000", "#fc8d59", "#d62728"])
    for bar, p in zip(bars, ps):
        h = bar.get_height()
        sig = "*" if (p is not None and not np.isnan(p) and p < 0.05) else ""
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}{sig}\np={p:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Spearman ρ (16S vs ITS BC matrix)")
    ax.set_title("v11.3.1 Stage-wise cross-kingdom Mantel")
    ax.set_ylim(min(0, min(rhos) - 0.1), 1.05)
    ax.axhline(0, color="grey", lw=0.5)
    plt.tight_layout()
    png_path = f"{OUT}/stagewise_mantel.png"
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
