import sys, numpy as np
sys.path.insert(0, "/tmp")
import garlic_procrustes_16S_ITS as g   # runs build -> g.D_16s, g.D_its, g.common
from scipy.stats import rankdata

common = g.common
D16 = g.D_16s.loc[common, common].values
Dit = g.D_its.loc[common, common].values
iu  = np.triu_indices(len(common), k=1)
v16, vit = D16[iu], Dit[iu]

# storage-month distance matrix (old_G1=0M,G3=2M,G5=4M,G7=6M -> month=g-1)
months = np.array([int(s.split("_G")[1].split("_")[0]) - 1 for s in common])
T = np.abs(months[:, None] - months[None, :])
vtime = T[iu]

def pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))

def partial(a, b, c):  # partial Spearman of a,b given c
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab, rac, rbc = pear(ra, rb), pear(ra, rc), pear(rb, rc)
    rp = (rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2))
    return rp, rab, rac, rbc

obs, rab, rac, rbc = partial(v16, vit, vtime)
rng = np.random.default_rng(42); nperm = 9999; cnt = 1
for _ in range(nperm):
    p = rng.permutation(len(common))
    vitp = Dit[np.ix_(p, p)][iu]
    rp, _, _, _ = partial(v16, vitp, vtime)
    if rp >= obs:
        cnt += 1
pval = cnt / (nperm + 1)

print("\n========== PARTIAL MANTEL ==========")
print(f"n samples = {len(common)} ; months = {list(months)}")
print(f"simple Mantel  16S~ITS  Spearman r = {rab:.4f}   (should match published 0.8215)")
print(f"               16S~time Spearman r = {rac:.4f}")
print(f"               ITS~time Spearman r = {rbc:.4f}")
print(f"PARTIAL Mantel 16S~ITS | time   r = {obs:.4f}   p(9999 perm) = {pval:.4f}")
print("====================================")
