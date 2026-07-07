"""Script 09 — Stage-resolved Mantel, FULL coverage (odd-month transitions).

Extends garlic_stagewise_mantel.py (which only does even-month 0,2,4,6).
Adds 1→2, 2→3, 3→4, 4→5, 5→6 where data exists.

Note: 16S 0M (G1) excluded by chloroplast 99% issue per manuscript;
ITS 0M is fine.  For stages where one marker is missing samples,
we use only the markers we have, dropping those samples.

We use META_ALL_OLD with all available months (0,1,2,3,4,5,6 — with the
caveat that month 0 might be unavailable for 16S).

Outputs:
  v11.3.1_supplementary/stagewise_mantel_full.csv
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, "/home1/minseo1101/garlic_project/analysis/scripts/v11.3.1")
from _helpers import (load_table, load_taxonomy, is_contam_16S, is_contam_ITS,
                      rarefy_counts_to_depth, bray_curtis,
                      mantel_spearman, META_ALL_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
T16S = f"{QROOT}/16S_old/table-dada2.qza"
TAX16 = f"{QROOT}/16S_old/taxonomy.qza"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"
OUT = "/home1/minseo1101/garlic_project/analysis/results/v11.3.1_supplementary"

DEPTH_16S = 130
DEPTH_ITS = 200
N_PERM = 999
SEED = 42

# Even-month stages from existing pipeline
EVEN_STAGES = {
    "A_0_2M": [0, 2],
    "B_2_4M": [2, 4],
    "C_4_6M": [4, 6],
    "D_0_6M": [0, 6],
}
# Odd-month and bridging stages
ODD_STAGES = {
    "E_1_2M": [1, 2],
    "F_2_3M": [2, 3],
    "G_3_4M": [3, 4],
    "H_4_5M": [4, 5],
    "I_5_6M": [5, 6],
    "J_0_1M": [0, 1],
    "K_1_3M": [1, 3],
    "L_3_5M": [3, 5],
}


def get_rarefied(table, tax_qza, depth, contam_fn):
    sample_ids, asv_ids, mat = load_table(table)
    tax = load_taxonomy(tax_qza)
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array(
        [not contam_fn(tax.get(a, "Unassigned")) for a in asv_ids])
    cm = mat[:, keep]
    all_idx = [i for i, s in enumerate(sample_ids) if s in META_ALL_OLD]
    ids = [sample_ids[i] for i in all_idx]
    sub = cm[all_idx]
    nz = sub.sum(axis=0) > 0
    sub = sub[:, nz]
    rng = np.random.default_rng(SEED)
    rare = np.zeros_like(sub)
    keep_mask = np.ones(sub.shape[0], dtype=bool)
    for i, row in enumerate(sub):
        rr = rarefy_counts_to_depth(row, depth, rng)
        if rr is None:
            keep_mask[i] = False
        else:
            rare[i] = rr
    ids = [s for s, k in zip(ids, keep_mask) if k]
    rare = rare[keep_mask]
    return ids, rare


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading rarefied tables…")
    s16, r16 = get_rarefied(T16S, TAX16, DEPTH_16S, is_contam_16S)
    sit, rit = get_rarefied(TITS, TAXIT, DEPTH_ITS, is_contam_ITS)
    print(f"  16S samples retained (depth≥{DEPTH_16S}): {len(s16)}")
    print(f"  ITS samples retained (depth≥{DEPTH_ITS}): {len(sit)}")

    common = [s for s in s16 if s in sit]
    print(f"  Common (both): {len(common)}  → {common}")
    if len(common) < 4:
        print("Too few common samples — aborting")
        return

    # Reorder
    r16_sub = r16[[s16.index(s) for s in common]]
    rit_sub = rit[[sit.index(s) for s in common]]
    D16 = bray_curtis(r16_sub)
    DIT = bray_curtis(rit_sub)

    rows = []
    # Full
    man_full = mantel_spearman(D16, DIT, n_perm=N_PERM, seed=SEED)
    rows.append(["FULL", f"All {len(common)} common samples", len(common),
                 sorted(set(META_ALL_OLD[s] for s in common)),
                 man_full["rho"], man_full["p"]])
    print(f"FULL n={len(common)} rho={man_full['rho']:.3f} p={man_full['p']:.4f}")

    for tag, months in {**EVEN_STAGES, **ODD_STAGES}.items():
        sel = [s for s in common if META_ALL_OLD[s] in months]
        if len(sel) < 4:
            rows.append([tag, f"{months[0]}M↔{months[1]}M", len(sel), months, np.nan, np.nan])
            print(f"{tag} months={months} n={len(sel)} SKIPPED (n<4)")
            continue
        idx = [common.index(s) for s in sel]
        D16s = D16[np.ix_(idx, idx)]
        DITs = DIT[np.ix_(idx, idx)]
        man = mantel_spearman(D16s, DITs, n_perm=N_PERM, seed=SEED)
        rows.append([tag, f"{months[0]}M↔{months[1]}M", len(sel), months,
                     man["rho"], man["p"]])
        sig = "*" if (man["p"] is not None and not np.isnan(man["p"]) and man["p"] < 0.05) else ""
        print(f"{tag} months={months} n={len(sel)} rho={man['rho']:.3f} p={man['p']:.4f}{sig}")

    out_path = f"{OUT}/stagewise_mantel_full.csv"
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["stage", "label", "n", "months", "spearman_rho", "p_perm"])
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
