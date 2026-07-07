"""Script 7 (★) — ITS NEW supplementary OLD-parity analyses.

(7.1) Indicator analysis per timepoint
(7.2) DA Wilcoxon + BH-FDR
(7.3) Cut audit freq=5 → freq=10
(7.4) Min-freq sensitivity sweep (5/10/20)
(7.5) Family-level OLD vs NEW comparison

Outputs to Attachments_investigation/:
  indicator_NEW_freq5_ITS.csv
  da_NEW_genera_wilcoxon_ITS.csv
  cut_asv_audit_NEW_ITS.csv
  minfreq_sensitivity_NEW_ITS.{csv,png}
  family_NEW_vs_OLD_ITS.csv
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (load_table, load_taxonomy, parse_genus, parse_family,
                      is_contam_ITS, rarefy_counts_to_depth, alpha_metrics,
                      bray_curtis, permanova_oneway, permdisp,
                      META_EVEN_OLD, META_NEW)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
OUT = f"{QROOT}/Attachments_investigation"
T_NEW = f"{QROOT}/ITS_new/table-dada2.qza"
TAX_NEW = f"{QROOT}/ITS_new/taxonomy.qza"
T_OLD = f"{QROOT}/ITS_old/table-dada2.qza"
TAX_OLD = f"{QROOT}/ITS_old/taxonomy.qza"

DEPTH = 100
N_PERM = 999
SEED = 42
FREQS = [5, 10, 20]


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    fdr = ranked * n / (np.arange(n) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty(n)
    out[order] = fdr
    return np.clip(out, 0, 1)


def indval(mat_rel, groups, target):
    """Dufrene-Legendre IndVal: A * B (A=fidelity, B=specificity).
    mat_rel: relative abundance per sample (rows sum to 1).
    Returns indval per feature for `target`."""
    g = np.asarray(groups)
    in_t = g == target
    out_t = ~in_t
    n_in = in_t.sum()
    n_out = out_t.sum()
    if n_in == 0 or n_out == 0:
        return np.zeros(mat_rel.shape[1])
    A = np.where(mat_rel.sum(axis=0) > 0,
                 mat_rel[in_t].mean(axis=0) /
                 (mat_rel[in_t].mean(axis=0) + mat_rel[out_t].mean(axis=0) + 1e-12),
                 0)
    B = (mat_rel[in_t] > 0).sum(axis=0) / n_in
    return A * B


def indval_perm(mat_rel, groups, target, n_perm=999, seed=42):
    rng = np.random.default_rng(seed)
    obs = indval(mat_rel, groups, target)
    null_ge = np.ones_like(obs)
    for _ in range(n_perm):
        g_p = rng.permutation(groups)
        nul = indval(mat_rel, g_p, target)
        null_ge += (nul >= obs).astype(int)
    return obs, null_ge / (n_perm + 1)


def aggregate_taxa(mat, asv_list, tax_dict, level_fn):
    keys = [level_fn(tax_dict.get(a, "Unassigned")) for a in asv_list]
    unique = sorted(set(keys))
    out = np.zeros((mat.shape[0], len(unique)))
    idx = {k: i for i, k in enumerate(unique)}
    for j, k in enumerate(keys):
        out[:, idx[k]] += mat[:, j]
    return out, unique


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------- Load NEW
    print("Loading ITS NEW...")
    sids_n, asvs_n, mat_n = load_table(T_NEW)
    tax_n = load_taxonomy(TAX_NEW)
    total_n = mat_n.sum(axis=0)
    contam_n = np.array([is_contam_ITS(tax_n.get(a, "Unassigned")) for a in asvs_n])

    # ---- (7.3) Cut audit freq=5 → freq=10 (before contam filter, but excluding contam)
    print("\n[7.3] Cut audit freq=5 → freq=10")
    cut_mask = (total_n >= 5) & (total_n < 10) & ~contam_n
    cut_path = f"{OUT}/cut_asv_audit_NEW_ITS.csv"
    n_named = 0
    confs = []
    with open(cut_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["asv_id", "total_reads", "n_samples_present",
                    "max_reads", "category", "taxonomy"])
        for i, asv in enumerate(asvs_n):
            if not cut_mask[i]:
                continue
            tot = int(total_n[i])
            npres = int((mat_n[:, i] > 0).sum())
            mx = int(mat_n[:, i].max())
            tax = tax_n.get(asv, "Unassigned")
            g = parse_genus(tax)
            cat = f"NAMED_GENUS: {g}" if not g.startswith(("f__", "o__", "c__", "p__")) and g != "Unassigned" else f"FAMILY/HIGHER: {g}"
            if cat.startswith("NAMED_GENUS"):
                n_named += 1
            w.writerow([asv, tot, npres, mx, cat, tax])
    n_cut = int(cut_mask.sum())
    print(f"  cut ASVs (freq 5–9): {n_cut}; named_genus: {n_named} ({100*n_named/max(n_cut,1):.1f}%)")
    print(f"  Wrote {cut_path}")

    # ---- (7.4) Min-freq sensitivity sweep (5/10/20)
    print("\n[7.4] Min-freq sensitivity sweep")
    new_idx = [i for i, s in enumerate(sids_n) if s in META_NEW]
    new_samples = [sids_n[i] for i in new_idx]
    new_mat_full = mat_n[new_idx]
    new_groups = np.array([META_NEW[s] for s in new_samples])

    sweep_rows = []
    n_indicator_baseline = None
    for mf in FREQS:
        keep = (total_n >= mf) & ~contam_n
        m = new_mat_full[:, keep]
        asvs = [a for a, k in zip(asvs_n, keep) if k]
        nz = m.sum(axis=0) > 0
        m = m[:, nz]
        asvs = [a for a, k in zip(asvs, nz) if k]
        # filter samples by min reads
        s_tot = m.sum(axis=1)
        ok = s_tot >= DEPTH
        ms = m[ok]
        groups = new_groups[ok]
        n = len(groups)
        if n < 4 or len(np.unique(groups)) < 2:
            sweep_rows.append([mf, m.shape[1], n, np.nan, np.nan, np.nan, np.nan])
            continue
        rng = np.random.default_rng(SEED)
        rare = np.zeros_like(ms)
        for i, row in enumerate(ms):
            rare[i] = rarefy_counts_to_depth(row, DEPTH, rng)
        # KW + PERMANOVA
        alpha = [alpha_metrics(r) for r in rare]
        obs = np.array([a[0] for a in alpha])
        sh = np.array([a[1] for a in alpha])
        from scipy.stats import kruskal
        kw_obs = kruskal(*[obs[groups == g] for g in sorted(np.unique(groups))]).pvalue
        kw_sh = kruskal(*[sh[groups == g] for g in sorted(np.unique(groups))]).pvalue
        D = bray_curtis(rare)
        perm = permanova_oneway(D, groups, n_perm=N_PERM, seed=SEED)
        # indicator count at 1M (NEW 1M)
        n_ind = "—"  # placeholder, will fill below
        sweep_rows.append([mf, m.shape[1], n, perm["F"], perm["p"], kw_obs, kw_sh])
        print(f"  freq={mf}: ASV={m.shape[1]}  n={n}  F={perm['F']:.2f} p={perm['p']:.3f}  KW_obs p={kw_obs:.3f}")

    sweep_csv = f"{OUT}/minfreq_sensitivity_NEW_ITS.csv"
    with open(sweep_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["min_freq", "n_ASV", "n_samples_retained", "BC_F", "BC_p",
                    "KW_obs_p", "KW_shannon_p"])
        for r in sweep_rows:
            w.writerow(r)
    print(f"  Wrote {sweep_csv}")

    # plot
    mf_pl = [r[0] for r in sweep_rows]
    F_pl = [r[3] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mf_pl, F_pl, "o-", color="#2c7fb8")
    ax.set_xlabel("Min-frequency")
    ax.set_ylabel("PERMANOVA F")
    ax.set_title("ITS NEW min-freq sensitivity")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/minfreq_sensitivity_NEW_ITS.png", dpi=160, bbox_inches="tight")
    print(f"  Wrote {OUT}/minfreq_sensitivity_NEW_ITS.png")

    # ---- (7.1) + (7.2) Indicator + DA at min_freq=5
    print("\n[7.1+7.2] Indicator + DA (freq=5)")
    keep = (total_n >= 5) & ~contam_n
    clean = mat_n[:, keep]
    cleanasvs = [a for a, k in zip(asvs_n, keep) if k]
    new_clean = clean[new_idx]
    nz = new_clean.sum(axis=0) > 0
    new_clean = new_clean[:, nz]
    cleanasvs = [a for a, k in zip(cleanasvs, nz) if k]

    # Aggregate to genus
    genus_mat, genus_list = aggregate_taxa(new_clean, cleanasvs, tax_n, parse_genus)
    print(f"  NEW genera: {len(genus_list)}")
    # Rarefy
    s_tot = new_clean.sum(axis=1)
    ok = s_tot >= DEPTH
    keep_samples = [new_samples[i] for i, k in enumerate(ok) if k]
    keep_groups = new_groups[ok]
    print(f"  retained samples ({DEPTH}+): {len(keep_samples)} / {len(new_samples)}")
    if len(keep_samples) >= 4 and len(np.unique(keep_groups)) >= 2:
        rng = np.random.default_rng(SEED)
        sub_mat = new_clean[ok]
        sub_genus = genus_mat[ok]
        # relative abundance for IndVal
        gtot = sub_genus.sum(axis=1, keepdims=True)
        gtot[gtot == 0] = 1
        rel = sub_genus / gtot

        ind_rows = []
        for t in sorted(np.unique(keep_groups)):
            ind, p = indval_perm(rel, keep_groups, int(t), n_perm=999, seed=SEED)
            fdr = bh_fdr(p)
            for j, g in enumerate(genus_list):
                if ind[j] > 0:
                    ind_rows.append([int(t), g, float(ind[j]), float(p[j]), float(fdr[j])])
        ind_rows.sort(key=lambda x: (x[0], -x[2]))
        ind_csv = f"{OUT}/indicator_NEW_freq5_ITS.csv"
        with open(ind_csv, "w") as f:
            w = csv.writer(f)
            w.writerow(["timepoint", "genus", "indval", "p_perm", "fdr_BH"])
            for r in ind_rows:
                w.writerow(r)
        print(f"  Wrote {ind_csv} ({len(ind_rows)} rows)")

        # DA Wilcoxon
        from scipy.stats import mannwhitneyu
        da_rows = []
        for t in sorted(np.unique(keep_groups)):
            in_g = keep_groups == t
            out_g = ~in_g
            pvals = []
            stats_ = []
            log2fc = []
            for j in range(rel.shape[1]):
                a = rel[in_g, j]
                b = rel[out_g, j]
                if a.sum() == 0 and b.sum() == 0:
                    pvals.append(1.0)
                    stats_.append(np.nan)
                    log2fc.append(np.nan)
                    continue
                try:
                    u, p = mannwhitneyu(a, b, alternative="greater")
                except Exception:
                    u, p = np.nan, 1.0
                pvals.append(p)
                stats_.append(u)
                mean_a = a.mean()
                mean_b = b.mean()
                lfc = np.log2((mean_a + 1e-6) / (mean_b + 1e-6))
                log2fc.append(lfc)
            fdr = bh_fdr(pvals)
            for j, g in enumerate(genus_list):
                if pvals[j] < 0.5:
                    da_rows.append([int(t), g, float(stats_[j]) if not np.isnan(stats_[j]) else "",
                                    float(log2fc[j]), float(pvals[j]), float(fdr[j])])
        da_rows.sort(key=lambda x: (x[0], x[4]))
        da_csv = f"{OUT}/da_NEW_genera_wilcoxon_ITS.csv"
        with open(da_csv, "w") as f:
            w = csv.writer(f)
            w.writerow(["timepoint", "genus", "U", "log2FC", "p_mannwhitney_greater", "fdr_BH"])
            for r in da_rows:
                w.writerow(r)
        print(f"  Wrote {da_csv} ({len(da_rows)} rows)")
        n_sig = sum(1 for r in da_rows if r[5] < 0.05)
        print(f"  FDR < 0.05: {n_sig}")

    # ---- (7.5) Family-level OLD vs NEW
    print("\n[7.5] Family-level OLD vs NEW comparison")
    sids_o, asvs_o, mat_o = load_table(T_OLD)
    tax_o = load_taxonomy(TAX_OLD)
    total_o = mat_o.sum(axis=0)
    contam_o = np.array([is_contam_ITS(tax_o.get(a, "Unassigned")) for a in asvs_o])
    keep_o = (total_o >= 5) & ~contam_o
    clean_o = mat_o[:, keep_o]
    asvs_o_k = [a for a, k in zip(asvs_o, keep_o) if k]

    fam_old, fam_old_list = aggregate_taxa(clean_o, asvs_o_k, tax_o, parse_family)
    fam_new, fam_new_list = aggregate_taxa(clean, cleanasvs, tax_n, parse_family)

    # rel per batch (sum across all samples)
    old_sum = fam_old.sum(axis=0)
    new_sum = fam_new.sum(axis=0)
    old_rel = old_sum / max(old_sum.sum(), 1)
    new_rel = new_sum / max(new_sum.sum(), 1)
    old_map = dict(zip(fam_old_list, old_rel))
    new_map = dict(zip(fam_new_list, new_rel))
    all_fams = sorted(set(fam_old_list) | set(fam_new_list))
    fam_csv = f"{OUT}/family_NEW_vs_OLD_ITS.csv"
    with open(fam_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["family", "pct_OLD", "pct_NEW", "diff_NEW-OLD"])
        for fam in all_fams:
            a = old_map.get(fam, 0.0)
            b = new_map.get(fam, 0.0)
            w.writerow([fam, f"{a*100:.3f}", f"{b*100:.3f}", f"{(b-a)*100:.3f}"])
    print(f"  Wrote {fam_csv} ({len(all_fams)} families)")

    print("\n=== Script 7 complete ===")


if __name__ == "__main__":
    main()
