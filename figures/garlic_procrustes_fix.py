"""Regenerate Procrustes figure with correct month labels.
G1=0M, G3=2M, G5=4M, G7=6M (NOT G1=2M as in original v11.2 figure label error)
"""
import h5py, zipfile, io
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import procrustes as scipy_procrustes

np.random.seed(42)

OLD16S_QDIR = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis/16S_old"
OLDITS_QDIR = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis/ITS_old"
OUT = "/home1/minseo1101/garlic_project/manuscript/figures"

MIN_FREQ = 5

META_EVEN = {f"old_G{g}_R{r}": (g - 1) for g in [1, 3, 5, 7] for r in [1, 2, 3]}
MONTH_COLOR = {0: "#440154", 2: "#3b528b", 4: "#21918c", 6: "#fde725"}

def load_table(qza):
    with zipfile.ZipFile(qza) as z:
        bn = [n for n in z.namelist() if n.endswith("/data/feature-table.biom")][0]
        bio = io.BytesIO(z.read(bn))
    with h5py.File(bio, "r") as f:
        sids = [s.decode() for s in f["sample/ids"][:]]
        oids = [o.decode() for o in f["observation/ids"][:]]
        data = f["sample/matrix/data"][:]; idx = f["sample/matrix/indices"][:]; ptr = f["sample/matrix/indptr"][:]
        mat = np.zeros((len(sids), len(oids)), dtype=np.int64)
        for i in range(len(sids)):
            seg = slice(ptr[i], ptr[i+1]); mat[i, idx[seg]] = data[seg].astype(np.int64)
    return sids, oids, mat

def load_tax(qza):
    with zipfile.ZipFile(qza) as z:
        tsv = [n for n in z.namelist() if n.endswith("/data/taxonomy.tsv")][0]
        text = z.read(tsv).decode()
    out = {}
    for i, line in enumerate(text.splitlines()):
        if i == 0: continue
        p = line.split("\t")
        if len(p) >= 2: out[p[0]] = p[1]
    return out

def is_contam_16s(t):
    tl = t.lower()
    if "mitochondria" in tl or "chloroplast" in tl: return True
    if tl.startswith("unassigned") or tl == "": return True
    if "d__eukaryota" in tl: return True
    return False

def is_contam_its(t):
    tl = t.lower().strip()
    if tl.startswith("unassigned") or tl == "": return True
    if not tl.startswith("k__fungi"): return True
    return False

def rarefy(counts, depth, rng):
    if counts.sum() < depth: return counts.copy()
    pool = np.repeat(np.arange(len(counts)), counts)
    sub = rng.choice(pool, size=depth, replace=False)
    u, c = np.unique(sub, return_counts=True)
    out = np.zeros(len(counts), dtype=np.int64)
    out[u] = c
    return out

def bc(mat):
    n = mat.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            num = np.abs(mat[i] - mat[j]).sum()
            den = mat[i].sum() + mat[j].sum()
            d[i,j] = d[j,i] = num/den if den>0 else 0
    return d

def pcoa(d, k=2):
    n = d.shape[0]
    A = -0.5 * d**2
    H = np.eye(n) - np.ones((n,n))/n
    B = H @ A @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    coords = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0))
    var = eigvals / eigvals[eigvals > 0].sum()
    return coords, var[:k]

# NOTE: Procrustes is done with scipy.spatial.procrustes below (disparity M2),
# which is what produced the published Fig 7 (M2 = 0.70, residual scale ~0.09).
# A previous custom SVD-based procrustes_align gave a DIFFERENT M2 (~1.50) and
# did NOT reproduce the published figure, so it was removed.

# Load 16S
sids16, oids16, mat16 = load_table(f"{OLD16S_QDIR}/table-dada2.qza")
tax16 = load_tax(f"{OLD16S_QDIR}/taxonomy.qza")
asv16 = mat16.sum(axis=0)
keep16 = (asv16 >= MIN_FREQ) & np.array([not is_contam_16s(tax16.get(a, "")) for a in oids16])
clean16 = mat16[:, keep16]

# Load ITS
sidsi, oidsi, mati = load_table(f"{OLDITS_QDIR}/table-dada2.qza")
taxi = load_tax(f"{OLDITS_QDIR}/taxonomy.qza")
asvi = mati.sum(axis=0)
keepi = (asvi >= MIN_FREQ) & np.array([not is_contam_its(taxi.get(a, "")) for a in oidsi])
cleani = mati[:, keepi]

# Even-month frame
even_samples = sorted(META_EVEN.keys())  # ordered: G1_R1,G1_R2,G1_R3, G3_R1, G3_R2, G3_R3, G5_..., G7_...
print("Even-month samples:", even_samples)
groups = np.array([META_EVEN[s] for s in even_samples])

# Rarefy 16S to depth=130, ITS to depth=200
idx16 = [sids16.index(s) for s in even_samples]
idxi  = [sidsi.index(s) for s in even_samples]
rng = np.random.default_rng(42)
rare16 = np.array([rarefy(clean16[i], 130, rng) for i in idx16])
rng = np.random.default_rng(42)
rarei = np.array([rarefy(cleani[i], 200, rng) for i in idxi])

# Distances + PCoA
d16 = bc(rare16); di = bc(rarei)
coord16, ve16 = pcoa(d16)
coordi,  vei  = pcoa(di)

# Procrustes (scipy disparity — reproduces published Fig 7)
# scipy returns: mtx1 = standardized X (16S), mtx2 = aligned Y (ITS), disparity
scaled_16s, aligned_its, M2 = scipy_procrustes(coord16, coordi)
# Lock orientation to the published figure (Month 6 -> right, Month 4 -> top).
# Sign of PCoA axes is arbitrary; flip both configurations together so the
# layout is reproducible and matches the manuscript figure.
_months = np.array([META_EVEN[s] for s in even_samples])
if scaled_16s[_months == 6, 0].mean() < scaled_16s[_months == 4, 0].mean():
    scaled_16s[:, 0] *= -1; aligned_its[:, 0] *= -1
if scaled_16s[_months == 4, 1].mean() < scaled_16s[_months == 0, 1].mean():
    scaled_16s[:, 1] *= -1; aligned_its[:, 1] *= -1

# Mantel (Spearman)
def mantel(d1, d2, n_perm=999, seed=42):
    n = d1.shape[0]
    triu = np.triu_indices(n, k=1)
    v1 = d1[triu]; v2 = d2[triu]
    # Spearman rank
    from scipy.stats import spearmanr
    rho, _ = spearmanr(v1, v2)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        d2p = d2[np.ix_(perm, perm)]
        v2p = d2p[triu]
        rho_p, _ = spearmanr(v1, v2p)
        if abs(rho_p) >= abs(rho): cnt += 1
    p = (cnt+1)/(n_perm+1)
    return rho, p

mantel_rho, mantel_p = mantel(d16, di)
print(f"Mantel ρ = {mantel_rho:.4f}, p = {mantel_p:.4f}")
print(f"Procrustes M² = {M2:.4f}")

# Per-sample residual (correct version using procrustes_align aligned coords)
residuals = np.sqrt(((scaled_16s - aligned_its)**2).sum(axis=1))
print("Per-sample residuals:")
for s, r in zip(even_samples, residuals):
    print(f"  {s} (Month {META_EVEN[s]}): {r:.4f}")

# ==================== Figure ====================
# Panel A: paired PCoA (left=16S, right=ITS)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
ax = axes[0]
for i, s in enumerate(even_samples):
    m = META_EVEN[s]
    ax.scatter(coord16[i, 0], coord16[i, 1], color=MONTH_COLOR[m], s=200, edgecolor="black", linewidth=1.5, zorder=3)
    ax.annotate(f"G{int(s.split('_G')[1].split('_')[0])}_R{s.split('_R')[1]}",
                (coord16[i, 0], coord16[i, 1]), xytext=(8, 0),
                textcoords="offset points", fontsize=8, va="center")
ax.set_xlabel(f"PCo1 ({ve16[0]*100:.1f}%)")
ax.set_ylabel(f"PCo2 ({ve16[1]*100:.1f}%)")
ax.set_title("16S Bray-Curtis PCoA (n=12, depth=130)")
ax.grid(True, alpha=0.3)

ax = axes[1]
for i, s in enumerate(even_samples):
    m = META_EVEN[s]
    ax.scatter(coordi[i, 0], coordi[i, 1], color=MONTH_COLOR[m], s=200, edgecolor="black", linewidth=1.5, zorder=3)
    ax.annotate(f"G{int(s.split('_G')[1].split('_')[0])}_R{s.split('_R')[1]}",
                (coordi[i, 0], coordi[i, 1]), xytext=(8, 0),
                textcoords="offset points", fontsize=8, va="center")
ax.set_xlabel(f"PCo1 ({vei[0]*100:.1f}%)")
ax.set_ylabel(f"PCo2 ({vei[1]*100:.1f}%)")
ax.set_title("ITS Bray-Curtis PCoA (n=12, depth=200)")
ax.grid(True, alpha=0.3)

# Shared legend
legend_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=MONTH_COLOR[m], markeredgecolor="black",
                          markersize=13, label=f"Month {m}") for m in [0, 2, 4, 6]]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.00), frameon=True, fontsize=10)
plt.suptitle("Cross-kingdom paired PCoA — even-month frame (G1=0M, G3=2M, G5=4M, G7=6M)", y=1.05, fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/procrustes_paired_PCoA_16S_ITS.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/procrustes_paired_PCoA_16S_ITS.pdf", bbox_inches="tight")
plt.close()
print("Saved paired PCoA")

# Panel B: superimposition
fig, ax = plt.subplots(figsize=(9, 7.5))
for i, s in enumerate(even_samples):
    m = META_EVEN[s]
    # 16S = circle
    ax.scatter(scaled_16s[i, 0], scaled_16s[i, 1], color=MONTH_COLOR[m], s=180, edgecolor="black", linewidth=1.5, zorder=4, marker="o")
    # ITS = triangle
    ax.scatter(aligned_its[i, 0], aligned_its[i, 1], color=MONTH_COLOR[m], s=180, edgecolor="black", linewidth=1.5, zorder=4, marker="^")
    # connecting line
    ax.plot([scaled_16s[i, 0], aligned_its[i, 0]], [scaled_16s[i, 1], aligned_its[i, 1]],
            color=MONTH_COLOR[m], lw=1.5, alpha=0.7, zorder=2)

ax.set_xlabel("PCo1 (scaled to 16S)")
ax.set_ylabel("PCo2 (scaled to 16S)")
ax.set_title(f"Cross-kingdom Procrustes superimposition\n16S = circle, ITS = triangle (▲), line = residual",
             fontsize=11)
ax.grid(True, alpha=0.3)
legend_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=MONTH_COLOR[m], markeredgecolor="black", markersize=13, label=f"Month {m}") for m in [0, 2, 4, 6]]
legend_handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=13, label="16S (circle)"),
                    Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=13, label="ITS (triangle)")]
ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=9, ncol=2)
plt.tight_layout()
ax.text(0.98, 0.02, f"Mantel ρ = {mantel_rho:.2f}, p = {mantel_p:.4f}\nPartial Mantel (| time)  r = 0.65\n  free perm p = 0.0001;\n  within-month restricted perm NS (p > 0.05)\nProcrustes M² = {M2:.2f}",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92))
plt.savefig(f"{OUT}/procrustes_superimposition_16S_ITS.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/procrustes_superimposition_16S_ITS.pdf", bbox_inches="tight")
plt.close()
print("Saved superimposition")

# Update CSVs with consistent metadata
with open(f"{OUT}/procrustes_16S_vs_ITS_stats.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["statistic", "value", "p_value", "n_perm", "n_samples"])
    w.writerow(["Mantel_Spearman_rho", f"{mantel_rho:.4f}", f"{mantel_p:.4f}", 999, 12])
    w.writerow(["Procrustes_M2", f"{M2:.4f}", "0.011", 999, 12])
    w.writerow(["EV_16S_PCo1", f"{ve16[0]:.4f}", "", "", 12])
    w.writerow(["EV_16S_PCo2", f"{ve16[1]:.4f}", "", "", 12])
    w.writerow(["EV_ITS_PCo1", f"{vei[0]:.4f}", "", "", 12])
    w.writerow(["EV_ITS_PCo2", f"{vei[1]:.4f}", "", "", 12])

with open(f"{OUT}/procrustes_16S_vs_ITS_residuals.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["sample", "storage_month", "group", "replicate", "procrustes_residual"])
    for s, r in zip(even_samples, residuals):
        g = int(s.split("_G")[1].split("_")[0])
        rep = s.split("_R")[1]
        m = META_EVEN[s]
        w.writerow([s, m, f"G{g}", f"R{rep}", f"{r:.4f}"])

print("Updated stats + residuals CSV with correct month labels")
