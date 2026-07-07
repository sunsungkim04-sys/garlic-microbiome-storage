"""Quick 16S depth=2000 PERMANOVA + PERMDISP for Table S6."""
import os, sys, zipfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (load_table, load_taxonomy, is_contam_16S,
                      rarefy_counts_to_depth, bray_curtis,
                      permanova_oneway, permdisp, META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
TABLE = f"{QROOT}/16S_old/table-dada2.qza"
TAX = f"{QROOT}/16S_old/taxonomy.qza"
DEPTHS = [2000, 3000]

sample_ids, asv_ids, mat = load_table(TABLE)
tax = load_taxonomy(TAX)
total = mat.sum(axis=0)
keep = (total >= 5) & np.array([not is_contam_16S(tax.get(a, "Unassigned")) for a in asv_ids])
mat = mat[:, keep]
asvs = [a for a,k in zip(asv_ids, keep) if k]

even_idx = [i for i,s in enumerate(sample_ids) if s in META_EVEN_OLD]
em_samples = [sample_ids[i] for i in even_idx]
em = mat[even_idx]
nz = em.sum(axis=0) > 0
em = em[:, nz]
groups = np.array([META_EVEN_OLD[s] for s in em_samples])

read_sums = em.sum(axis=1)
print("sample_read_sums", read_sums.tolist())

for depth in DEPTHS:
    rng = np.random.default_rng(42)
    keep_idx = [i for i,s in enumerate(read_sums) if s >= depth]
    if len(keep_idx) < 4:
        print("depth=", depth, "insufficient samples", len(keep_idx), "SKIP")
        continue
    em_sub = em[keep_idx]
    gr_sub = groups[keep_idx]
    rar = np.array([rarefy_counts_to_depth(row, depth, rng) for row in em_sub])
    bc = bray_curtis(rar)
    perm = permanova_oneway(bc, gr_sub, n_perm=999, seed=42)
    disp = permdisp(bc, gr_sub, n_perm=999, seed=42)
    F1, p1 = perm["F"], perm["p"]
    F2, p2 = disp["F"], disp["p"]
    print("depth=", depth, "n=", len(keep_idx), "groups=", sorted(set(gr_sub)),
          "BC_F=", round(F1,3), "p=", round(p1,4),
          "PERMDISP_F=", round(F2,3), "p=", round(p2,4))
