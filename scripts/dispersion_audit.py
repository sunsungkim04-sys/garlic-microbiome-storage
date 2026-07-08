#!/usr/bin/env python3
"""
dispersion_audit.py — reproduce every PERMANOVA / PERMDISP statistic the manuscript reports.

A figure should not print a number no script can regenerate; the same holds for the body text.
From this archive alone this script recomputes:

  16S (587 ASVs, depth 130)   Bray-Curtis  PERMANOVA F, R2, adj R2 ; PERMDISP F, p  (+ 10-seed sweep)
                              Jaccard      PERMDISP p
                              unweighted UniFrac  PERMANOVA F, R2, adj R2 ; PERMDISP p (+ 10-seed sweep)
                              within-month dispersion (mean distance to group centroid)
  ITS (95 ASVs, depth 200)    Bray-Curtis  PERMANOVA F   (dispersion-inflated; not the effect size)
                              Jaccard      PERMANOVA F, adj R2 ; PERMDISP p   (the reported effect size)

Rarefaction is without replacement at seed 42, matching scripts/garlic_16S_depth_sweep.py and
figures/regen_fig3_16S.py. unweighted UniFrac is computed from data/16S_rooted_tree.nwk with a
dependency-free implementation validated against scikit-bio (identical to three decimal places).

The phylogeny has 494 tips and covers 414 of the 587 analysed 16S ASVs, so UniFrac is computed on
that intersection; Bray-Curtis and Jaccard use all 587.

Run:  python3 scripts/dispersion_audit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVEN = [f"old_G{g}_R{r}" for g in (1, 3, 5, 7) for r in (1, 2, 3)]
MONTHS = np.array([{1: 0, 3: 2, 5: 4, 7: 6}[int(s.split("_")[1][1:])] for s in EVEN])
SEED = 42


def is_contam_ITS(t):
    t = t.lower()
    return t.startswith("unassigned") or t == "" or "k__fungi" not in t


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


def parse_newick(s):
    """Return list of (name_or_None, branch_length, list_of_child_indices) as a flat node array."""
    s = s.strip().rstrip(";")
    nodes = []          # (name, length, children)
    pos = 0

    def parse_node():
        nonlocal pos
        children = []
        if s[pos] == "(":
            pos += 1
            while True:
                children.append(parse_node())
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == ")":
                    pos += 1
                    break
        # label
        start = pos
        while pos < len(s) and s[pos] not in ",():":
            pos += 1
        name = s[start:pos] or None
        length = 0.0
        if pos < len(s) and s[pos] == ":":
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in ",()":
                pos += 1
            length = float(s[start:pos])
        nodes.append((name, length, children))
        return len(nodes) - 1

    root = parse_node()
    return nodes, root


def unweighted_unifrac_matrix(pres, taxa, nodes, root):
    """pres: n_samples x n_taxa boolean. Returns n x n UniFrac distances."""
    n = pres.shape[0]
    tip_index = {t: i for i, t in enumerate(taxa)}
    masks = np.zeros(len(nodes), dtype=np.int64)
    for idx, (name, _, children) in enumerate(nodes):
        if not children:
            j = tip_index.get(name)
            if j is not None:
                m = 0
                for s in range(n):
                    if pres[s, j]:
                        m |= (1 << s)
                masks[idx] = m
        else:
            m = 0
            for c in children:
                m |= masks[c]
            masks[idx] = m
    lengths = np.array([nd[1] for nd in nodes])
    keep = np.arange(len(nodes)) != root          # the root has no branch
    masks, lengths = masks[keep], lengths[keep]

    D = np.zeros((n, n))
    for i in range(n):
        bi = 1 << i
        for j in range(i + 1, n):
            bj = 1 << j
            in_i = (masks & bi) != 0
            in_j = (masks & bj) != 0
            observed = in_i | in_j
            unshared = in_i ^ in_j
            denom = lengths[observed].sum()
            D[i, j] = D[j, i] = lengths[unshared].sum() / denom if denom > 0 else 0.0
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
    R2 = (SST - SSW) / SST
    n_ge = 1
    for _ in range(n_perm):
        s = ssw(rng.permutation(groups))
        if s > 0 and ((SST - s) / (a - 1)) / (s / (n - a)) >= F:
            n_ge += 1
    return F, n_ge / (n_perm + 1), R2, 1 - (1 - R2) * (n - 1) / (n - a)


def permdisp(D, groups, n_perm=999, seed=SEED):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    A = -0.5 * D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    w, v = eigh(H @ A @ H)
    i = np.argsort(-w)
    w, v = w[i], v[:, i]
    k = w > 1e-9
    C = v[:, k] * np.sqrt(w[k])
    groups = np.asarray(groups)
    uniq = np.unique(groups)

    def F_of(g):
        d = np.zeros(n)
        for u in uniq:
            ii = np.where(g == u)[0]
            d[ii] = np.linalg.norm(C[ii] - C[ii].mean(axis=0), axis=1)
        gm = d.mean()
        ssa = sum(len(np.where(g == u)[0]) * (d[np.where(g == u)[0]].mean() - gm) ** 2 for u in uniq)
        ssw = sum(((d[np.where(g == u)[0]] - d[np.where(g == u)[0]].mean()) ** 2).sum() for u in uniq)
        return (ssa / (len(uniq) - 1)) / (ssw / (n - len(uniq))) if ssw > 0 else np.nan

    Fo = F_of(groups)
    n_ge = 1 + sum(F_of(rng.permutation(groups)) >= Fo for _ in range(n_perm))
    return Fo, n_ge / (n_perm + 1)




def bray_curtis(mat):
    n = mat.shape[0]
    D = np.zeros((n, n))
    rs = mat.sum(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            den = rs[i] + rs[j]
            D[i, j] = D[j, i] = np.abs(mat[i] - mat[j]).sum() / den if den > 0 else 0.0
    return D


def jaccard(mat):
    n = mat.shape[0]
    p = (mat > 0).astype(int)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = (p[i] & p[j]).sum(), (p[i] | p[j]).sum()
            D[i, j] = D[j, i] = 1.0 - a / b if b else 0.0
    return D


def dispersion_per_group(D, groups):
    n = D.shape[0]
    A = -0.5 * D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    w, v = eigh(H @ A @ H)
    i = np.argsort(-w)
    w, v = w[i], v[:, i]
    k = w > 1e-9
    C = v[:, k] * np.sqrt(w[k])
    return {int(u): float(np.linalg.norm(C[np.where(groups == u)[0]]
                                         - C[np.where(groups == u)[0]].mean(axis=0), axis=1).mean())
            for u in np.unique(groups)}


def load(marker, contam_fn, min_freq=5):
    tbl = pd.read_csv(DATA / f"{marker}_feature-table-dada2.txt", sep="\t", skiprows=1, index_col=0)[EVEN]
    tax = pd.read_csv(DATA / f"{marker}_taxonomy.tsv", sep="\t", index_col=0)["Taxon"].to_dict()
    nc = np.array([not contam_fn(tax.get(a, "Unassigned")) for a in tbl.index])
    return tbl.loc[(tbl.values.sum(axis=1) >= min_freq) & nc]


def rarefied(sub, depth, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rarefy(sub[c].values.astype(np.int64), depth, rng) for c in EVEN])


def main():
    print("=" * 78)
    print("16S — depth 130, seed 42 (subsampling without replacement)")
    s16 = load("16S", is_contam_16S)
    r16 = rarefied(s16, 130, SEED)
    print(f"  table: {s16.shape[0]} ASVs x {s16.shape[1]} samples")

    Dbc = bray_curtis(r16)
    F, p, R2, adj = permanova(Dbc, MONTHS)
    dF, dp = permdisp(Dbc, MONTHS)
    print(f"  Bray-Curtis  PERMANOVA F = {F:.3f}  p = {p:.3f}  R2 = {R2:.3f}  adj R2 = {adj:.3f}")
    print(f"  Bray-Curtis  PERMDISP  F = {dF:.3f}  p = {dp:.3f}")
    disp = dispersion_per_group(Dbc, MONTHS)
    print(f"  within-month dispersion: {' / '.join(f'{disp[m]:.2f}' for m in (0, 2, 4, 6))} at 0/2/4/6M")

    jF, jp = permdisp(jaccard(r16), MONTHS)
    print(f"  Jaccard      PERMDISP  F = {jF:.3f}  p = {jp:.3f}")

    nodes, root = parse_newick(open(DATA / "16S_rooted_tree.nwk").read())
    tips = {nd[0] for nd in nodes if not nd[2] and nd[0]}
    keep = [i for i, a in enumerate(s16.index) if a in tips]
    taxa = [s16.index[i] for i in keep]
    print(f"  phylogeny: {len(tips)} tips, covering {len(keep)} of {s16.shape[0]} analysed ASVs")
    Du = unweighted_unifrac_matrix(r16[:, keep] > 0, taxa, nodes, root)
    uF, up, uR2, uadj = permanova(Du, MONTHS)
    udF, udp = permdisp(Du, MONTHS)
    print(f"  unw. UniFrac PERMANOVA F = {uF:.3f}  p = {up:.3f}  R2 = {uR2:.3f}  adj R2 = {uadj:.3f}")
    print(f"  unw. UniFrac PERMDISP  F = {udF:.3f}  p = {udp:.3f}")

    print("\n  PERMDISP across ten rarefaction draws (seeds 42-51):")
    bc_ps, uf_ps = [], []
    for s in range(42, 52):
        r = rarefied(s16, 130, s)
        bc_ps.append(permdisp(bray_curtis(r), MONTHS)[1])
        uf_ps.append(permdisp(unweighted_unifrac_matrix(r[:, keep] > 0, taxa, nodes, root), MONTHS)[1])
    bc_ps, uf_ps = np.array(bc_ps), np.array(uf_ps)
    print(f"    Bray-Curtis   p = {bc_ps.min():.3f}-{bc_ps.max():.3f};  significant in {(bc_ps < 0.05).sum()}/10")
    print(f"    unw. UniFrac  p = {uf_ps.min():.3f}-{uf_ps.max():.3f};  non-significant in {(uf_ps >= 0.05).sum()}/10")

    print("\n" + "=" * 78)
    print("ITS — depth 200, seed 42 (subsampling without replacement)")
    sIT = load("ITS", is_contam_ITS)
    rIT = rarefied(sIT, 200, SEED)
    print(f"  table: {sIT.shape[0]} ASVs x {sIT.shape[1]} samples")
    F, p, _, _ = permanova(bray_curtis(rIT), MONTHS)
    print(f"  Bray-Curtis  PERMANOVA F = {F:.3f}  p = {p:.3f}   (dispersion-inflated; not the effect size)")
    Dj = jaccard(rIT)
    F, p, R2, adj = permanova(Dj, MONTHS)
    dF, dp = permdisp(Dj, MONTHS)
    print(f"  Jaccard      PERMANOVA F = {F:.3f}  p = {p:.3f}  R2 = {R2:.3f}  adj R2 = {adj:.3f}")
    print(f"  Jaccard      PERMDISP  F = {dF:.3f}  p = {dp:.3f}   (dispersion-clean effect size)")


if __name__ == "__main__":
    main()
