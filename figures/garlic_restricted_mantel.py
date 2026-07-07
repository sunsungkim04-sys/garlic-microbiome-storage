#!/usr/bin/env python
"""
Restricted (within-month) permutation partial Mantel — robustness check for
reviewer concern that cross-kingdom r=0.65 may be inflated by within-month
(same-bulb) non-independence.

Reproduces published simple Mantel rho (~0.8215) and partial r (~0.65),
then re-tests partial r significance under:
  (1) FREE permutation (as published, 9999 random)
  (2) RESTRICTED within-month permutation (exhaustive 6^4 = 1296):
      ITS sample labels shuffled ONLY within each month block, preserving
      the storage-time structure that partial Mantel already conditions on.
"""
import os, sys, tempfile, shutil, zipfile, glob, itertools
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

BASE = os.path.expanduser("~/garlic_project/data/qiime2_reanalysis")
EVEN = [f"old_G{g}_R{r}" for g in (1,3,5,7) for r in (1,2,3)]  # 0/2/4/6M x 3

def extract(qza, suf):
    td = tempfile.mkdtemp(prefix=f"qza_{suf}_")
    with zipfile.ZipFile(qza) as z: z.extractall(td)
    return td
def load_tbl(qza):
    td = extract(qza, "t"); from biom import load_table
    b = glob.glob(os.path.join(td, "*/data/feature-table.biom"))[0]
    df = load_table(b).to_dataframe(dense=True).astype(float); shutil.rmtree(td); return df
def load_tax(qza):
    td = extract(qza, "x"); t = glob.glob(os.path.join(td, "*/data/taxonomy.tsv"))[0]
    df = pd.read_csv(t, sep="\t").rename(columns={"Feature ID":"fid","Taxon":"tax"}).set_index("fid")
    shutil.rmtree(td); return df["tax"].astype(str)
def contam16(t):
    t = t.lower(); return ("mitochondria" in t or "chloroplast" in t or
        t.startswith("unassigned") or t.strip()=="" or "d__eukaryota" in t)
def keepits(t):
    t = t.lower(); return (not (t.startswith("unassigned") or t.strip()=="")) and ("k__fungi" in t)
def rarefy(df, depth, seed=42):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=int); keep=[]
    for s in df.columns:
        col = df[s].values.astype(int); n = col.sum()
        if n < depth: print(f"  drop {s} ({n}<{depth})"); continue
        idx = np.repeat(np.arange(len(col)), col)
        ch = rng.choice(idx, size=depth, replace=False)
        out[s] = np.bincount(ch, minlength=len(col)); keep.append(s)
    return out[keep]
def bc(df):
    M = df.T.values.astype(float); M = M / M.sum(axis=1, keepdims=True)
    return pd.DataFrame(squareform(pdist(M, metric="braycurtis")), index=df.columns, columns=df.columns)
def build(tblq, taxq, depth, kind):
    print(f"=== build {kind} (depth={depth}) ===")
    tbl = load_tbl(tblq); tax = load_tax(taxq)
    sub = [s for s in EVEN if s in tbl.columns]; tbl = tbl[sub]
    if kind == "16S":
        bad = set(tax[tax.apply(contam16)].index); tbl = tbl.loc[~tbl.index.isin(bad)]
    else:
        good = set(tax[tax.apply(keepits)].index); tbl = tbl.loc[tbl.index.isin(good)]
    tbl = tbl.loc[tbl.sum(axis=1) >= 5]
    return bc(rarefy(tbl, depth))

D16 = build(f"{BASE}/16S_old/table-dada2.qza", f"{BASE}/16S_old/taxonomy.qza", 130, "16S")
Dit = build(f"{BASE}/ITS_old/table-dada2.qza", f"{BASE}/ITS_old/taxonomy.qza", 200, "ITS")
common = [s for s in EVEN if s in D16.index and s in Dit.index]
print(f"\nn common = {len(common)}: {common}")
D16 = D16.loc[common, common]; Dit = Dit.loc[common, common]
D16.to_csv(os.path.expanduser("~/garlic_16S_bc_n12.csv"))
Dit.to_csv(os.path.expanduser("~/garlic_ITS_bc_n12.csv"))

iu = np.triu_indices(len(common), k=1)
v16 = D16.values[iu]
months = np.array([int(s.split("_G")[1].split("_")[0]) - 1 for s in common])
T = np.abs(months[:, None] - months[None, :]); vtime = T[iu]

def pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))
def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab, rac, rbc = pear(ra, rb), pear(ra, rc), pear(rb, rc)
    return (rab - rac*rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2)), rab

Dv = Dit.values
obs, simple = partial(v16, Dv[iu], vtime)
print("\n========== REPRODUCE PUBLISHED ==========")
print(f"simple Mantel rho (16S~ITS)   = {simple:.4f}   (published 0.8215)")
print(f"partial Mantel r (|time ctrl) = {obs:.4f}   (published 0.65)")

# (1) FREE permutation
rng = np.random.default_rng(42); N = 9999; cnt = 1
for _ in range(N):
    p = rng.permutation(len(common)); vp = Dv[np.ix_(p, p)][iu]
    r, _ = partial(v16, vp, vtime)
    if r >= obs: cnt += 1
p_free = cnt / (N + 1)

# (2) RESTRICTED within-month permutation (exhaustive)
groups = {}
for i, s in enumerate(common): groups.setdefault(months[i], []).append(i)
gkeys = sorted(groups)
perms_per = [list(itertools.permutations(groups[g])) for g in gkeys]
null = []
for combo in itertools.product(*perms_per):
    perm = np.arange(len(common))
    for g, order in zip(gkeys, combo): perm[groups[g]] = list(order)
    vp = Dv[np.ix_(perm, perm)][iu]
    r, _ = partial(v16, vp, vtime)
    null.append(r)
null = np.array(null)
p_restr = (np.sum(null >= obs)) / len(null)

print("\n========== PERMUTATION SCHEMES ==========")
print(f"(1) FREE perm        p = {p_free:.4f}   (N=9999, published scheme)")
print(f"(2) RESTRICTED within-month p = {p_restr:.4f}   (exhaustive n_perm={len(null)})")
print(f"    restricted null partial r: min={null.min():.3f} med={np.median(null):.3f} "
      f"mean={null.mean():.3f} max={null.max():.3f} | obs={obs:.3f}")
print(f"    obs percentile in restricted null = {100*(np.sum(null<=obs)/len(null)):.1f}%")
print("=========================================")
