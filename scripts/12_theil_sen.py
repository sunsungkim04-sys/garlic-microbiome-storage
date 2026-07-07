"""Script 12 — Theil-Sen slope + 95% CI for Mann-Kendall taxa.

For all genus-level taxa passing Mann-Kendall (pymannkendall 1.4) on
the OLD even-month (n=12) time series for 16S and ITS independently,
compute the Theil-Sen median slope and 95% CI.

Pipeline:
  1. Build genus-level rel-abundance table (n_genera × n_samples),
     OLD even-month only.
  2. For each genus, average reps per month → 4-point series (0/2/4/6 M).
  3. pymannkendall.original_test() on the 4-point series.
  4. Retain p < 0.10 (lenient since n=4 per series, then apply BH).
  5. Theil-Sen slope + 95% CI via scipy.stats.theilslopes.
  6. Append to existing Table S3-style outputs.

Outputs:
  v11.3.1_supplementary/theil_sen_slopes.tsv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import pymannkendall as mk

sys.path.insert(0, "/home1/minseo1101/garlic_project/analysis/scripts/v11.3.1")
from _helpers import (load_table, load_taxonomy, is_contam_16S, is_contam_ITS,
                      parse_genus, META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
T16S = f"{QROOT}/16S_old/table-dada2.qza"
TAX16 = f"{QROOT}/16S_old/taxonomy.qza"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"
OUT = "/home1/minseo1101/garlic_project/analysis/results/v11.3.1_supplementary"


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = ranked
    return out


def genus_relab_per_sample(table_qza, tax_qza, contam_fn):
    sids, asvs, mat = load_table(table_qza)
    tax = load_taxonomy(tax_qza)
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not contam_fn(tax.get(a, "")) for a in asvs])
    mat = mat[:, keep]
    asvs = [a for a, k in zip(asvs, keep) if k]
    em_idx = [i for i, s in enumerate(sids) if s in META_EVEN_OLD]
    em_sids = [sids[i] for i in em_idx]
    em_mat = mat[em_idx]
    gnames = [parse_genus(tax.get(a, "")) for a in asvs]
    df = pd.DataFrame(em_mat.T, index=gnames)
    df = df.groupby(level=0).sum()
    df.columns = em_sids
    rel = df.div(df.sum(axis=0), axis=1)
    return rel


def analyse_marker(rel_df, marker):
    """For each genus build a per-month mean series, run MK + TS."""
    months = np.array([META_EVEN_OLD[s] for s in rel_df.columns])
    uniq_months = np.array(sorted(set(months)))
    rows = []
    for genus in rel_df.index:
        vals = rel_df.loc[genus].values
        # mean per month
        series = np.array([vals[months == m].mean() for m in uniq_months])
        # also keep all 12 points for TS robustness
        # MK on 4-point series
        try:
            res = mk.original_test(series)
            mk_tau = res.Tau
            mk_p = res.p
            mk_trend = res.trend
        except Exception:
            mk_tau = np.nan; mk_p = np.nan; mk_trend = "n/a"
        # Theil-Sen on full 12-point series (more reps = better CI)
        try:
            ts = theilslopes(vals, np.tile(uniq_months, len(vals) // len(uniq_months)))
            ts_slope = ts.slope
            ts_lo = ts.low_slope
            ts_hi = ts.high_slope
            ts_inter = ts.intercept
        except Exception as e:
            ts_slope = np.nan; ts_lo = np.nan; ts_hi = np.nan; ts_inter = np.nan
        rows.append(dict(marker=marker, genus=genus, mk_tau=mk_tau, mk_p=mk_p,
                         mk_trend=mk_trend,
                         ts_slope=ts_slope, ts_ci95_low=ts_lo, ts_ci95_high=ts_hi,
                         ts_intercept=ts_inter,
                         mean_0M=series[0] if len(series) > 0 else np.nan,
                         mean_6M=series[-1] if len(series) > 0 else np.nan))
    df = pd.DataFrame(rows)
    df["mk_p_adj"] = bh_fdr(df["mk_p"].fillna(1.0).values)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Building genus relab tables …")
    b16 = genus_relab_per_sample(T16S, TAX16, is_contam_16S)
    bit = genus_relab_per_sample(TITS, TAXIT, is_contam_ITS)
    print(f"  16S genera: {len(b16)},  ITS genera: {len(bit)}")

    df16 = analyse_marker(b16, "16S")
    dfit = analyse_marker(bit, "ITS")

    out = pd.concat([df16, dfit], ignore_index=True)
    out = out.sort_values(["marker", "mk_p"])
    out.to_csv(f"{OUT}/theil_sen_slopes.tsv", sep="\t", index=False)

    sig_mk = out[(out["mk_p"] < 0.10) & (~out["mk_p"].isna())]
    sig_fdr = out[(out["mk_p_adj"] < 0.10) & (~out["mk_p_adj"].isna())]
    print(f"\nGenera with MK p<0.10: {len(sig_mk)}")
    print(f"Genera with MK p_adj<0.10 (BH): {len(sig_fdr)}")
    print(f"\nTop 10 MK-significant genera with Theil-Sen:")
    print(sig_mk[["marker", "genus", "mk_tau", "mk_p", "mk_p_adj",
                  "ts_slope", "ts_ci95_low", "ts_ci95_high"]].head(20).to_string(index=False))
    print(f"\nWrote {OUT}/theil_sen_slopes.tsv")


if __name__ == "__main__":
    main()
