"""Shared helpers for v11.3.1 supplementary analyses.

All 7 scripts (sample_identity, singleton_ablation, 16S_depth_sweep,
procrustes_jt, ITS_NEW_depth, stagewise_mantel, ITS_NEW_complete)
import this module. Place at /tmp/v11.3.1/_helpers.py on lab101.
"""
import h5py
import zipfile
import io
import numpy as np


def load_table(qza):
    """Return (sample_ids, asv_ids, sample x asv int64 matrix) from a qza."""
    with zipfile.ZipFile(qza) as z:
        biom_name = [n for n in z.namelist() if n.endswith("/data/feature-table.biom")][0]
        bio = io.BytesIO(z.read(biom_name))
    with h5py.File(bio, "r") as f:
        sample_ids = [s.decode() for s in f["sample/ids"][:]]
        obs_ids = [o.decode() for o in f["observation/ids"][:]]
        data = f["sample/matrix/data"][:]
        indices = f["sample/matrix/indices"][:]
        indptr = f["sample/matrix/indptr"][:]
        mat = np.zeros((len(sample_ids), len(obs_ids)), dtype=np.int64)
        for i in range(len(sample_ids)):
            seg = slice(indptr[i], indptr[i + 1])
            mat[i, indices[seg]] = data[seg].astype(np.int64)
    return sample_ids, obs_ids, mat


def load_taxonomy(qza):
    with zipfile.ZipFile(qza) as z:
        tsv = [n for n in z.namelist() if n.endswith("/data/taxonomy.tsv")][0]
        text = z.read(tsv).decode()
    out = {}
    for i, line in enumerate(text.splitlines()):
        if i == 0:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def parse_genus(tax):
    """Return genus if present and meaningful, else falls back to f__/o__/c__ tag."""
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
        x = levels.get(r, "")
        if x:
            return f"{r}__{x}"
    return "Unassigned"


def parse_family(tax):
    """Return family level taxonomy; fall back through o/c if missing."""
    levels = {}
    for p in tax.split(";"):
        p = p.strip()
        if "__" in p:
            k, v = p.split("__", 1)
            levels[k.strip()] = v.strip()
    f = levels.get("f", "")
    if f:
        return f
    for r in ("o", "c", "p"):
        x = levels.get(r, "")
        if x:
            return f"{r}__{x}"
    return "Unassigned"


def is_contam_16S(tax):
    t = tax.lower()
    if "mitochondria" in t or "chloroplast" in t:
        return True
    if t.startswith("unassigned") or t == "":
        return True
    if "d__eukaryota" in t:
        return True
    return False


def is_contam_ITS(tax):
    """Keep k__Fungi, drop Unassigned/empty."""
    t = tax.lower()
    if t.startswith("unassigned") or t == "":
        return True
    if "k__fungi" not in t:
        return True
    return False


def rarefy_counts_to_depth(counts, depth, rng):
    """Single-iteration multinomial rarefaction without replacement.
    Returns subsampled count vector (same shape) or None if total < depth."""
    if counts.sum() < depth:
        return None
    pool = np.repeat(np.arange(len(counts)), counts)
    sub = rng.choice(pool, size=depth, replace=False)
    out = np.zeros_like(counts)
    u, c = np.unique(sub, return_counts=True)
    out[u] = c
    return out


def alpha_metrics(counts):
    """Observed ASVs + Shannon (natural log) from a single count vector."""
    nz = counts[counts > 0]
    if nz.sum() == 0:
        return 0, 0.0
    p = nz / nz.sum()
    return int(len(nz)), float(-np.sum(p * np.log(p)))


def bray_curtis(mat):
    """n x f matrix → n x n BC distance matrix."""
    n = mat.shape[0]
    D = np.zeros((n, n))
    rs = mat.sum(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            num = np.abs(mat[i] - mat[j]).sum()
            den = rs[i] + rs[j]
            D[i, j] = D[j, i] = num / den if den > 0 else 0.0
    return D


def jaccard_binary(mat):
    n = mat.shape[0]
    pres = (mat > 0).astype(int)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = pres[i] & pres[j]
            b = pres[i] | pres[j]
            bs = b.sum()
            D[i, j] = D[j, i] = 1.0 - (a.sum() / bs) if bs > 0 else 0.0
    return D


def permanova_oneway(D, groups, n_perm=999, seed=42):
    """Two-or-more group PERMANOVA. Returns dict with F, p, R2."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n = len(groups)
    SST = (D ** 2).sum() / (2 * n)
    unique = np.unique(groups)

    def ss_within(g):
        ssw = 0.0
        for u in unique:
            idx = np.where(g == u)[0]
            if len(idx) < 2:
                continue
            sub = D[np.ix_(idx, idx)]
            ssw += (sub ** 2).sum() / (2 * len(idx))
        return ssw

    SSW = ss_within(groups)
    SSA = SST - SSW
    a = len(unique)
    F = (SSA / (a - 1)) / (SSW / (n - a)) if SSW > 0 else np.nan
    R2 = SSA / SST if SST > 0 else np.nan

    n_ge = 1
    for _ in range(n_perm):
        perm = rng.permutation(groups)
        SSW_p = ss_within(perm)
        SSA_p = SST - SSW_p
        F_p = (SSA_p / (a - 1)) / (SSW_p / (n - a)) if SSW_p > 0 else 0.0
        if F_p >= F:
            n_ge += 1
    p = n_ge / (n_perm + 1)
    return dict(F=float(F), p=float(p), R2=float(R2))


def permdisp(D, groups, n_perm=999, seed=42):
    """PERMDISP (homogeneity of multivariate dispersions, Anderson 2006).
    Distances from group centroid in PCoA space → ANOVA F + permutation p."""
    from scipy.linalg import eigh
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    A = -0.5 * D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = H @ A @ H
    w, v = eigh(B)
    idx = np.argsort(-w)
    w, v = w[idx], v[:, idx]
    keep = w > 1e-9
    coords = v[:, keep] * np.sqrt(w[keep])
    groups = np.asarray(groups)
    unique = np.unique(groups)

    def disp_F(g):
        d_to_centroid = np.zeros(n)
        for u in unique:
            ii = np.where(g == u)[0]
            cen = coords[ii].mean(axis=0)
            d_to_centroid[ii] = np.linalg.norm(coords[ii] - cen, axis=1)
        # One-way ANOVA F on d_to_centroid by g
        gm = d_to_centroid.mean()
        ssa = 0.0
        ssw = 0.0
        for u in unique:
            ii = np.where(g == u)[0]
            sub = d_to_centroid[ii]
            ssa += len(ii) * (sub.mean() - gm) ** 2
            ssw += ((sub - sub.mean()) ** 2).sum()
        a = len(unique)
        return (ssa / (a - 1)) / (ssw / (n - a)) if ssw > 0 else np.nan

    F_obs = disp_F(groups)
    n_ge = 1
    for _ in range(n_perm):
        F_p = disp_F(rng.permutation(groups))
        if not np.isnan(F_p) and F_p >= F_obs:
            n_ge += 1
    p = n_ge / (n_perm + 1)
    return dict(F=float(F_obs), p=float(p))


def mantel_spearman(D1, D2, n_perm=999, seed=42):
    """Mantel test (Spearman rank correlation) with permutation p."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(D1.shape[0], k=1)
    v1 = D1[iu]
    v2 = D2[iu]
    rho, _ = spearmanr(v1, v2)
    n = D1.shape[0]
    n_ge = 1
    for _ in range(n_perm):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        rho_p, _ = spearmanr(v1, D2p[iu])
        if rho_p >= rho:
            n_ge += 1
    p = n_ge / (n_perm + 1)
    return dict(rho=float(rho), p=float(p))


def filter_table(mat, asv_ids, tax_dict, min_freq=5, is_contam_fn=is_contam_16S):
    """Apply min-freq + taxa filter. Returns (filtered_mat, kept_asvs)."""
    total = mat.sum(axis=0)
    pass_freq = total >= min_freq
    not_contam = np.array([not is_contam_fn(tax_dict.get(a, "Unassigned")) for a in asv_ids])
    keep = pass_freq & not_contam
    return mat[:, keep], [a for a, k in zip(asv_ids, keep) if k]


META_EVEN_OLD = {
    "old_G1_R1": 0, "old_G1_R2": 0, "old_G1_R3": 0,
    "old_G3_R1": 2, "old_G3_R2": 2, "old_G3_R3": 2,
    "old_G5_R1": 4, "old_G5_R2": 4, "old_G5_R3": 4,
    "old_G7_R1": 6, "old_G7_R2": 6, "old_G7_R3": 6,
}

META_ALL_OLD = {
    "old_G1_R1": 0, "old_G1_R2": 0, "old_G1_R3": 0,
    "old_G2_R1": 1, "old_G2_R2": 1, "old_G2_R3": 1,
    "old_G3_R1": 2, "old_G3_R2": 2, "old_G3_R3": 2,
    "old_G4_R1": 3, "old_G4_R2": 3,
    "old_G5_R1": 4, "old_G5_R2": 4, "old_G5_R3": 4,
    "old_G6_R1": 5, "old_G6_R2": 5, "old_G6_R3": 5,
    "old_G7_R1": 6, "old_G7_R2": 6, "old_G7_R3": 6,
}

META_NEW = {
    "new_G1_R1": 0, "new_G1_R2": 0, "new_G1_R3": 0, "new_G1_R4": 0, "new_G1_R5": 0,
    "new_G2_R1": 1, "new_G2_R2": 1, "new_G2_R3": 1, "new_G2_R4": 1, "new_G2_R5": 1,
    "new_G3_R1": 2, "new_G3_R2": 2, "new_G3_R3": 2, "new_G3_R4": 2, "new_G3_R5": 2,
    "new_G4_R1": 3, "new_G4_R2": 3, "new_G4_R3": 3, "new_G4_R4": 3, "new_G4_R5": 3,
}
