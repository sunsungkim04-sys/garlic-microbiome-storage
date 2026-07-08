#!/usr/bin/env python3
"""
regen_figS9_minfreq.py — Figure S9 (16S min-frequency sensitivity, 6-panel).

The submitted Figure S9 printed 707 / 494 / 338 ASVs and PERMANOVA F = 4.25 / 4.43 / 4.94.
Those came from a filter applied to a 20-sample table (and from a with-replacement
rarefaction), so the figure disagreed with both Table S6 (587 / 414 / 284;
F = 4.539 / 4.711 / 4.720) and section 2.4 (587 ASVs).

This regenerates the figure from the same pipeline as scripts/regen_minfreq_sensitivity.py:
min-frequency applied to the 12 analysed samples, rarefaction without replacement at depth 130,
alpha diversity averaged over 100 draws (seeds 42..141), PERMANOVA on the seed-42 draw.

Run:  python3 figures/regen_figS9_minfreq.py   (writes figures/output/)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures" / "output"

EVEN = [f"old_G{g}_R{r}" for g in (1, 3, 5, 7) for r in (1, 2, 3)]
G2M = {1: 0, 3: 2, 5: 4, 7: 6}
MONTHS = np.array([G2M[int(s.split("_")[1][1:])] for s in EVEN])
MONTH_COLORS = {0: "#440154", 2: "#3b528b", 4: "#21918c", 6: "#fde725"}
FREQ_LEVELS = [5, 10, 20]
DEPTH, SEED, N_ITER = 130, 42, 100

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def is_contam_16S(t):
    t = t.lower()
    return ("mitochondria" in t or "chloroplast" in t or t.startswith("unassigned")
            or t == "" or "d__eukaryota" in t)


def rarefy(counts, depth, rng):
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


def bray_curtis(mat):
    n = mat.shape[0]
    D = np.zeros((n, n))
    rs = mat.sum(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            den = rs[i] + rs[j]
            D[i, j] = D[j, i] = np.abs(mat[i] - mat[j]).sum() / den if den > 0 else 0.0
    return D


def permanova(D, groups, n_perm=999, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n, uniq = len(groups), np.unique(groups)
    SST = (D ** 2).sum() / (2 * n)

    def ssw(g):
        s = 0.0
        for u in uniq:
            i = np.where(g == u)[0]
            if len(i) >= 2:
                s += (D[np.ix_(i, i)] ** 2).sum() / (2 * len(i))
        return s

    SSW, a = ssw(groups), len(uniq)
    F = ((SST - SSW) / (a - 1)) / (SSW / (n - a))
    n_ge = 1
    for _ in range(n_perm):
        s = ssw(rng.permutation(groups))
        if s > 0 and ((SST - s) / (a - 1)) / (s / (n - a)) >= F:
            n_ge += 1
    return F, n_ge / (n_perm + 1)


def main():
    tbl = pd.read_csv(DATA / "16S_feature-table-dada2.txt", sep="\t", skiprows=1, index_col=0)[EVEN]
    tax = pd.read_csv(DATA / "16S_taxonomy.tsv", sep="\t", index_col=0)["Taxon"].to_dict()
    not_contam = np.array([not is_contam_16S(tax.get(a, "Unassigned")) for a in tbl.index])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for col, mf in enumerate(FREQ_LEVELS):
        sub = tbl.loc[(tbl.values.sum(axis=1) >= mf) & not_contam]
        counts = sub.T.values.astype(np.int64)
        n_asv, total_reads = sub.shape[0], int(sub.values.sum())

        obs_it, sh_it = [], []
        for it in range(N_ITER):
            rng = np.random.default_rng(SEED + it)
            a = [alpha(r) for r in (rarefy(row, DEPTH, rng) for row in counts)]
            obs_it.append([x[0] for x in a])
            sh_it.append([x[1] for x in a])
        obs, sh = np.mean(obs_it, axis=0), np.mean(sh_it, axis=0)
        kw_obs = kruskal(*[obs[MONTHS == m] for m in (0, 2, 4, 6)])
        kw_sh = kruskal(*[sh[MONTHS == m] for m in (0, 2, 4, 6)])

        rng = np.random.default_rng(SEED)
        rare = np.stack([rarefy(r, DEPTH, rng) for r in counts])
        F, p = permanova(bray_curtis(rare), MONTHS)

        for row, (vals, ylab, kw) in enumerate([(obs, "Observed ASVs", kw_obs),
                                                (sh, "Shannon", kw_sh)]):
            ax = axes[row, col]
            box = [vals[MONTHS == m] for m in (0, 2, 4, 6)]
            bp = ax.boxplot(box, positions=[0, 2, 4, 6], widths=1.0, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 1.2})
            for i, m in enumerate((0, 2, 4, 6)):
                bp["boxes"][i].set_facecolor(MONTH_COLORS[m])
                bp["boxes"][i].set_alpha(0.7)
                ax.scatter([m] * 3, vals[MONTHS == m], s=50, color=MONTH_COLORS[m],
                           edgecolor="black", linewidths=0.5, zorder=5)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xticklabels(["0M", "2M", "4M", "6M"])
            ax.set_ylabel(ylab if col == 0 else "")
            ax.grid(linestyle=":", alpha=0.35)
            if row == 1:
                ax.set_xlabel("Storage month")
            if row == 0:
                ax.set_title(f"min-freq={mf}  ({n_asv} ASV, {total_reads:,} reads)\n"
                             f"KW p={kw.pvalue:.4f} / PERMANOVA F={F:.3f}, p={p:.3f}",
                             fontsize=10)
            else:
                ax.set_title(f"KW p={kw.pvalue:.4f}", fontsize=9)
        print(f"  min-freq={mf:>2}: {n_asv} ASV, {total_reads:,} reads, "
              f"PERMANOVA F={F:.3f} p={p:.3f}, KW obs p={kw_obs.pvalue:.4f}")

    fig.suptitle("Even-month (n = 12) — min-frequency sensitivity: ROBUST\n"
                 f"depth={DEPTH}, without-replacement rarefaction, alpha averaged over "
                 f"{N_ITER} draws / min-samples=1 fixed", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Figure_S9_minfreq_sensitivity.png", dpi=300)
    fig.savefig(OUT / "Figure_S9_minfreq_sensitivity.pdf")
    plt.close(fig)
    print("  OK  Figure_S9_minfreq_sensitivity.{png,pdf}")


if __name__ == "__main__":
    main()
