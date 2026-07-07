"""Script 2 (★★★) — Dynamic m-s=1 ablation.

638 singleton ASV random drop (5 seed × 5 drop_fraction = 25 iter).
filter-result coupling 우려 차단.

Output:
  Attachments_investigation/singleton_ablation.csv
  Attachments_investigation/singleton_ablation.png
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (load_table, load_taxonomy, parse_genus, is_contam_16S,
                      rarefy_counts_to_depth, alpha_metrics,
                      bray_curtis, permanova_oneway, META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
OUT = f"{QROOT}/Attachments_investigation"
TABLE = f"{QROOT}/16S_old/table-dada2.qza"
TAX = f"{QROOT}/16S_old/taxonomy.qza"
SINGLETON_CSV = f"{QROOT}/singleton_asv_audit.csv"

DEPTH = 130
N_PERM = 999
SEEDS = [42, 100, 200, 300, 400]
DROP_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]

# 8 strict indicator genera from manuscript §3.2 / 01_16S §5.3
INDICATOR_GENERA = {"Bacillus", "Pseudonocardia", "Acinetobacter",
                    "Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium",
                    "Pseudogracilibacillus", "Sphingobacterium", "Hathewaya"}
# alternative shorter Allorhizobium key
ALT_GENUS_MAP = {"Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium": "Allorhizobium"}


def load_singleton_ids(path):
    """Return set of ASV ids with n_samples_present == 1."""
    ids = set()
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                if int(row["n_samples_present"]) == 1:
                    ids.add(row["asv_id"])
            except Exception:
                pass
    return ids


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading table + taxonomy...")
    sample_ids, asv_ids, mat = load_table(TABLE)
    tax = load_taxonomy(TAX)

    # min-freq=5 + taxa filter
    total = mat.sum(axis=0)
    pass_freq = total >= 5
    not_contam = np.array([not is_contam_16S(tax.get(a, "Unassigned")) for a in asv_ids])
    keep_mask = pass_freq & not_contam
    clean_mat = mat[:, keep_mask]
    clean_asvs = np.array([a for a, k in zip(asv_ids, keep_mask) if k])
    print(f"clean table: {clean_mat.shape}")

    # Even-month subset
    even_idx = [i for i, s in enumerate(sample_ids) if s in META_EVEN_OLD]
    even_samples = [sample_ids[i] for i in even_idx]
    em = clean_mat[even_idx]
    nz = em.sum(axis=0) > 0
    em = em[:, nz]
    em_asvs = clean_asvs[nz]
    groups = np.array([META_EVEN_OLD[s] for s in even_samples])
    print(f"even-month: {em.shape}")

    # Identify singletons within even-month frame (n_samples_present == 1 in EM)
    pres_em = (em > 0).sum(axis=0)
    em_singletons = set(em_asvs[pres_em == 1].tolist())
    print(f"Singletons in even-month (n_present==1): {len(em_singletons)}")

    # Also load the precomputed singleton list (those flagged by audit across whole table)
    audit_singletons = load_singleton_ids(SINGLETON_CSV)
    print(f"Singletons in audit CSV: {len(audit_singletons)}")
    # Use union present in EM
    drop_pool = np.array([a in (em_singletons | audit_singletons) for a in em_asvs])
    drop_pool_idx = np.where(drop_pool)[0]
    print(f"Drop pool size in EM table: {len(drop_pool_idx)}")

    # Indicator detection (em level)
    asv_genus = np.array([parse_genus(tax.get(a, "Unassigned")) for a in em_asvs])

    def count_indicators_preserved(used_idx):
        sub_genus = asv_genus[used_idx]
        present = set(sub_genus)
        return sum(1 for g in INDICATOR_GENERA if g in present)

    baseline_indicators = count_indicators_preserved(np.arange(em.shape[1]))
    print(f"Baseline indicator genera in even-month: {baseline_indicators}/{len(INDICATOR_GENERA)}")

    rows = []
    for drop_frac in DROP_FRACS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            n_drop = int(round(drop_frac * len(drop_pool_idx)))
            drop_choice = rng.choice(drop_pool_idx, size=n_drop, replace=False) if n_drop > 0 else np.array([], dtype=int)
            keep = np.ones(em.shape[1], dtype=bool)
            keep[drop_choice] = False
            sub_mat = em[:, keep]
            n_asv_rem = int(sub_mat.shape[1])

            # Rarefy
            rng2 = np.random.default_rng(seed + 7)
            sub_rare = np.zeros_like(sub_mat)
            ok = True
            for i, row in enumerate(sub_mat):
                rr = rarefy_counts_to_depth(row, DEPTH, rng2)
                if rr is None:
                    ok = False
                    break
                sub_rare[i] = rr
            if not ok:
                rows.append([drop_frac, seed, n_asv_rem, np.nan, np.nan, np.nan, np.nan, False, np.nan])
                continue

            # Alpha + ranking
            alpha = [alpha_metrics(r) for r in sub_rare]
            obs = np.array([a[0] for a in alpha])
            sh = np.array([a[1] for a in alpha])
            month_obs = {m: obs[groups == m].mean() for m in (0, 2, 4, 6)}
            month_sh = {m: sh[groups == m].mean() for m in (0, 2, 4, 6)}
            obs_rank = sorted(month_obs, key=month_obs.get, reverse=True)
            sh_rank = sorted(month_sh, key=month_sh.get, reverse=True)
            target_rank = [2, 4, 6, 0]
            ranking_match = obs_rank == target_rank and sh_rank == target_rank

            # KW p (alpha by month)
            from scipy.stats import kruskal
            groups_lists_obs = [obs[groups == m] for m in (0, 2, 4, 6)]
            kw_obs = kruskal(*groups_lists_obs).pvalue
            groups_lists_sh = [sh[groups == m] for m in (0, 2, 4, 6)]
            kw_sh = kruskal(*groups_lists_sh).pvalue

            # BC + PERMANOVA
            D = bray_curtis(sub_rare)
            perm = permanova_oneway(D, groups, n_perm=N_PERM, seed=seed + 11)

            # Indicator preserved
            n_ind = count_indicators_preserved(np.where(keep)[0])

            rows.append([drop_frac, seed, n_asv_rem,
                         perm["F"], perm["p"], kw_obs, kw_sh,
                         ranking_match, n_ind])
            print(f"drop={drop_frac:.2f} seed={seed}  n_asv={n_asv_rem}  "
                  f"F={perm['F']:.2f} p={perm['p']:.3f}  KW_obs={kw_obs:.3f}  "
                  f"match={ranking_match}  ind={n_ind}")

    # CSV
    csv_path = f"{OUT}/singleton_ablation.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["drop_fraction", "seed", "n_ASV_remaining",
                    "BC_F", "BC_p", "KW_obs_p", "KW_shannon_p",
                    "ranking_match", "n_indicator_preserved"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Aggregate per drop_fraction
    agg = {}
    for r in rows:
        df, _, n_asv, F, p, kw_o, kw_s, rm, ni = r
        agg.setdefault(df, []).append(r)
    drop_fracs_sorted = sorted(agg)
    F_mean = [np.nanmean([r[3] for r in agg[d]]) for d in drop_fracs_sorted]
    KW_mean = [np.nanmean([r[5] for r in agg[d]]) for d in drop_fracs_sorted]
    RM_pct = [100.0 * np.mean([1 if r[7] else 0 for r in agg[d]]) for d in drop_fracs_sorted]
    IND_mean = [np.nanmean([r[8] for r in agg[d]]) for d in drop_fracs_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(drop_fracs_sorted, F_mean, "o-", color="#2c7fb8")
    axes[0, 0].set_title("PERMANOVA Bray-Curtis F\n(avg across 5 seeds)")
    axes[0, 0].set_xlabel("Singleton drop fraction")
    axes[0, 0].set_ylabel("F")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(drop_fracs_sorted, KW_mean, "o-", color="#fc8d59")
    axes[0, 1].axhline(0.05, ls="--", color="grey")
    axes[0, 1].set_title("Kruskal-Wallis Observed-ASV p\n(avg across 5 seeds)")
    axes[0, 1].set_xlabel("Singleton drop fraction")
    axes[0, 1].set_ylabel("p")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(drop_fracs_sorted, RM_pct, "o-", color="#1a9850")
    axes[1, 0].set_title("Ranking 2M>4M>6M>0M match\n(% of 5 seeds)")
    axes[1, 0].set_xlabel("Singleton drop fraction")
    axes[1, 0].set_ylabel("% seeds")
    axes[1, 0].set_ylim(-5, 105)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(drop_fracs_sorted, IND_mean, "o-", color="#9e3fd6")
    axes[1, 1].axhline(baseline_indicators, ls="--", color="grey",
                       label=f"baseline = {baseline_indicators}")
    axes[1, 1].set_title("Indicator genera preserved\n(avg across 5 seeds, of 7)")
    axes[1, 1].set_xlabel("Singleton drop fraction")
    axes[1, 1].set_ylabel("# of indicator genera")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("v11.3.1 Singleton ablation — m-s=1 robustness check", y=1.02)
    plt.tight_layout()
    png_path = f"{OUT}/singleton_ablation.png"
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
