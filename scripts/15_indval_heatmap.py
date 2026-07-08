"""Figure S10 — genus-level relative abundance (even-month frame) of the genera
carrying the 18 strict month-2 indicator ASVs (10 ITS + 8 16S).

Self-contained: reads only files shipped in this archive.
    python scripts/15_indval_heatmap.py
Reproduces e.g. 16S Bacillus 2M = 16.70 %, ITS Penicillium 50.22/94.58/90.55/99.93 %.
"""
import csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
G2M = {"G1": 0, "G3": 2, "G5": 4, "G7": 6}
RENAME = {"Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium": "Rhizobium clade",
          "f__Didymosphaeriaceae": "Didymosphaeriaceae (family)",
          "f__Ceratobasidiaceae": "Ceratobasidiaceae (family)"}

def load_table(p):
    return pd.read_csv(p, sep="\t", skiprows=1, index_col=0)

def genus_of(taxon):
    if pd.isna(taxon) or taxon == "Unassigned":
        return "Unassigned"
    parts = [x.strip() for x in taxon.split(";")]
    for pre, fmt in (("g__", "{}"), ("f__", "f__{}"), ("o__", "o__{}")):
        hit = next((x for x in parts if x.startswith(pre) and len(x) > 3), None)
        if hit:
            return fmt.format(hit[3:])
    return "Unclassified"

def marker_filter(tab, tax, marker):
    tab = tab[tab.sum(axis=1) >= 5]
    T = tax.loc[tab.index.intersection(tax.index), "Taxon"]
    if marker == "ITS":
        keep = T[T.str.startswith("k__Fungi", na=False)].index
    else:
        keep = T[(T.str.startswith("d__Bacteria") | T.str.startswith("d__Archaea"))
                 & ~T.str.contains("Chloroplast|Mitochondria", case=False, na=False)].index
    return tab.loc[tab.index.intersection(keep)]

ind = list(csv.DictReader(open(ROOT / "supplementary/TableS3b_indicator_2M_ASVs.tsv"), delimiter="\t"))
labels, rows = [], []
for marker in ("ITS", "16S"):
    tab = load_table(ROOT / f"data/{marker}_feature-table-dada2.txt")
    tax = pd.read_csv(ROOT / f"data/{marker}_taxonomy.tsv", sep="\t").set_index("Feature ID")
    tab = marker_filter(tab, tax, marker)
    gen = tab.groupby(tax.loc[tab.index, "Taxon"].apply(genus_of)).sum()
    rel = gen / gen.sum(axis=0) * 100
    months = {c: G2M[c.split("_")[1]] for c in rel.columns}
    seen = []
    for r in (x for x in ind if x["marker"] == marker):
        g = r["genus"]
        if g in seen or g not in rel.index:
            continue
        seen.append(g)
        labels.append(f"{marker} · {RENAME.get(g, g)}")
        rows.append([float(rel.loc[g, [c for c in rel.columns if months[c] == m]].mean()) for m in (0, 2, 4, 6)])

M = np.array(rows)
for l, v in zip(labels, M):
    print(f"{l:42s} " + "  ".join(f"{x:7.2f}" for x in v))

fig, ax = plt.subplots(figsize=(7.0, 0.44 * len(labels) + 2.0))
im = ax.imshow(np.where(M <= 0, 1e-3, M), aspect="auto", cmap="YlOrBr",
               norm=LogNorm(vmin=1e-3, vmax=100))
ax.set_xticks(range(4)); ax.set_xticklabels(["0M", "2M", "4M", "6M"])
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
for i in range(len(labels)):
    ax.text(1, i, "†", ha="center", va="center", fontsize=10)
ax.set_xlabel("Storage month")
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("Genus relative abundance (%, log scale)", fontsize=8)
ax.text(0.0, -0.9, "† genus carries ≥1 strict month-2 indicator ASV", fontsize=7.5, style="italic")
plt.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "FigureS10_indicator_genus_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "FigureS10_indicator_genus_heatmap.pdf", bbox_inches="tight")
print("\nwritten -> figures/FigureS10_indicator_genus_heatmap.{png,pdf}")
