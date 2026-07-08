#!/usr/bin/env python3
"""
regen_minfreq_sensitivity.py — regenerate supplementary/TableS8_minfreq_sensitivity.tsv
(manuscript Table S6) from this archive alone.

Reads data/{16S,ITS}_feature-table-dada2.txt + data/{16S,ITS}_taxonomy.tsv and reports,
for min-frequency 5 / 10 / 20, the retained ASV count, read total, rarefied alpha diversity,
Bray-Curtis (and Jaccard) PERMANOVA, month-2 indicator counts, and the top month-2 genera.

Rarefaction subsamples WITHOUT replacement, matching garlic_16S_depth_sweep.py and the
manuscript's Figure 3B: at min-frequency = 5 the 16S PERMANOVA reproduces F = 4.54.
Alpha columns are means over 100 draws (seeds 42..141); beta uses the seed-42 draw.

    python scripts/regen_minfreq_sensitivity.py            # print
    python scripts/regen_minfreq_sensitivity.py --write    # rewrite the TSV
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "TableS8_minfreq_sensitivity.tsv"

SAMPLES = [f"old_G{g}_R{r}" for g in (1, 3, 5, 7) for r in (1, 2, 3)]
G2M = {1: 0, 3: 2, 5: 4, 7: 6}
MONTHS = np.array([G2M[int(s.split("_")[1][1:])] for s in SAMPLES])
FREQ_LEVELS = [5, 10, 20]
DEPTH = {"16S": 130, "ITS": 200}
SEED, N_ITER = 42, 100


def is_contam_16S(t):
    t = t.lower()
    return ("mitochondria" in t or "chloroplast" in t or t.startswith("unassigned")
            or t == "" or "d__eukaryota" in t)


def is_contam_ITS(t):
    t = t.lower()
    return t.startswith("unassigned") or t == "" or "k__fungi" not in t


def parse_genus(tax):
    levels = {}
    for p in tax.split(";"):
        p = p.strip()
        if "__" in p:
            k, v = p.split("__", 1)
            levels[k.strip()] = v.strip()
    g = levels.get("g", "")
    if g and g.lower() not in ("", "uncultured", "unidentified"):
        return g
    for r in ("f", "o", "c", "p"):
        if levels.get(r, ""):
            return f"{r}__{levels[r]}"
    return "Unassigned"


def rarefy(counts, depth, rng):
    if counts.sum() < depth:
        return None
    pool = np.repeat(np.arange(len(counts)), counts)
    sub = rng.choice(pool, size=depth, replace=False)
    out = np.zeros_like(counts)
    u, c = np.unique(sub, return_counts=True)
    out[u] = c
    return out


def alpha(counts):
    nz = counts[counts > 0]
    if nz.sum() == 0:
        return 0, 0.0
    p = nz / nz.sum()
    return int(len(nz)), float(-np.sum(p * np.log(p)))


def _dist(mat, binary=False):
    n = mat.shape[0]
    D = np.zeros((n, n))
    if binary:
        pres = (mat > 0).astype(int)
        for i, j in itertools.combinations(range(n), 2):
            a, b = (pres[i] & pres[j]).sum(), (pres[i] | pres[j]).sum()
            D[i, j] = D[j, i] = 1.0 - a / b if b else 0.0
    else:
        rs = mat.sum(axis=1)
        for i, j in itertools.combinations(range(n), 2):
            den = rs[i] + rs[j]
            D[i, j] = D[j, i] = np.abs(mat[i] - mat[j]).sum() / den if den > 0 else 0.0
    return D


def permanova(D, groups, n_perm=999, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n, uniq = len(groups), np.unique(groups)
    SST = (D ** 2).sum() / (2 * n)

    def ssw(g):
        return sum((D[np.ix_(i, i)] ** 2).sum() / (2 * len(i))
                   for i in (np.where(g == u)[0] for u in uniq) if len(i) >= 2)

    SSW, a = ssw(groups), len(uniq)
    F = ((SST - SSW) / (a - 1)) / (SSW / (n - a))
    n_ge = 1 + sum(
        ((SST - s) / (a - 1)) / (s / (n - a)) >= F
        for s in (ssw(rng.permutation(groups)) for _ in range(n_perm)) if s > 0)
    return float(F), n_ge / (n_perm + 1)


def indicators(sub):
    """Month-2 indicators present in >= 2 of 3 month-2 replicates.
    strict: absent from every other sample.  loose: present in <= 1 other sample."""
    m2 = sub.loc[:, [s for s, m in zip(SAMPLES, MONTHS) if m == 2]]
    other = sub.loc[:, [s for s, m in zip(SAMPLES, MONTHS) if m != 2]]
    reps, n_other = (m2 > 0).sum(axis=1), (other > 0).sum(axis=1)
    return int(((reps >= 2) & (n_other == 0)).sum()), int(((reps >= 2) & (n_other <= 1)).sum())


def top_2M(sub, tax):
    m2 = sub.loc[:, [s for s, m in zip(SAMPLES, MONTHS) if m == 2]]
    genus = pd.Series([parse_genus(tax.get(a, "")) for a in sub.index], index=sub.index)
    by_g = m2.groupby(genus).sum()
    rel = (by_g.div(by_g.sum(axis=0), axis=1).mean(axis=1) * 100).sort_values(ascending=False)
    return rel.index[0], round(float(rel.iloc[0]), 2), rel.index[1], round(float(rel.iloc[1]), 2)


def run(marker, contam_fn):
    tbl = pd.read_csv(ROOT / "data" / f"{marker}_feature-table-dada2.txt",
                      sep="\t", skiprows=1, index_col=0)[SAMPLES]
    tax = pd.read_csv(ROOT / "data" / f"{marker}_taxonomy.tsv",
                      sep="\t", index_col=0)["Taxon"].to_dict()
    not_contam = np.array([not contam_fn(tax.get(a, "Unassigned")) for a in tbl.index])
    depth, rows = DEPTH[marker], []

    for mf in FREQ_LEVELS:
        sub = tbl.loc[(tbl.values.sum(axis=1) >= mf) & not_contam]
        counts = sub.T.values.astype(np.int64)

        obs, sh = [], []
        for it in range(N_ITER):
            rng = np.random.default_rng(SEED + it)
            a = [alpha(r) for r in (rarefy(row, depth, rng) for row in counts)]
            obs.append([x[0] for x in a])
            sh.append([x[1] for x in a])
        obs, sh = np.mean(obs, axis=0), np.mean(sh, axis=0)
        obs_m = {m: float(np.mean(obs[MONTHS == m])) for m in (0, 2, 4, 6)}
        sh_m = {m: float(np.mean(sh[MONTHS == m])) for m in (0, 2, 4, 6)}
        kw_o = kruskal(*[obs[MONTHS == m] for m in (0, 2, 4, 6)])
        kw_s = kruskal(*[sh[MONTHS == m] for m in (0, 2, 4, 6)])

        rng = np.random.default_rng(SEED)
        rare = np.stack([rarefy(r, depth, rng) for r in counts])
        F_bc, p_bc = permanova(_dist(rare), MONTHS)
        F_jc, p_jc = permanova(_dist(rare, binary=True), MONTHS)

        strict, loose = indicators(sub)
        g1, a1, g2, a2 = top_2M(sub, tax)
        rank = lambda d: ">".join(f"{m}M" for m in sorted(d, key=lambda k: -d[k]))

        rows.append(dict(
            marker=marker, min_freq=mf, n_asv_clean=sub.shape[0], total_reads=int(sub.values.sum()),
            obs_0M=round(obs_m[0], 2), obs_2M=round(obs_m[2], 2), obs_4M=round(obs_m[4], 2),
            obs_6M=round(obs_m[6], 2), sh_0M=round(sh_m[0], 3), sh_2M=round(sh_m[2], 3),
            sh_4M=round(sh_m[4], 3), sh_6M=round(sh_m[6], 3),
            obs_ranking=rank(obs_m), sh_ranking=rank(sh_m),
            kw_obs_H=round(float(kw_o.statistic), 2), kw_obs_p=round(float(kw_o.pvalue), 4),
            kw_sh_H=round(float(kw_s.statistic), 2), kw_sh_p=round(float(kw_s.pvalue), 4),
            permanova_F=round(F_bc, 3) if marker == "16S" else np.nan,
            permanova_p=p_bc if marker == "16S" else np.nan,
            n_strict_2M_indicator=strict, n_loose_2M_indicator=loose,
            top1_2M_genus=g1, top1_2M_abund=a1, top2_2M_genus=g2, top2_2M_abund=a2,
            permanova_F_BC=round(F_bc, 3) if marker == "ITS" else np.nan,
            permanova_p_BC=p_bc if marker == "ITS" else np.nan,
            permanova_F_Jaccard=round(F_jc, 3) if marker == "ITS" else np.nan,
            permanova_p_Jaccard=p_jc if marker == "ITS" else np.nan))
        print(f"  {marker} mf={mf:>2}: {sub.shape[0]:>3} ASV / {int(sub.values.sum()):>7,} reads  "
              f"BC F={F_bc:.3f} p={p_bc:.3f}  strict/loose={strict}/{loose}  top1 {g1} {a1}%")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    df = pd.DataFrame(run("16S", is_contam_16S) + run("ITS", is_contam_ITS))
    if args.write:
        df.to_csv(OUT, sep="\t", index=False)
        print(f"\nwrote {OUT.relative_to(ROOT)}")
