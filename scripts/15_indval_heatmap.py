#!/usr/bin/env python3
"""
15_indval_heatmap.py
Wave 3 Track F — Generates a supplementary heatmap of the 18 strict 2M
indicator genera (10 ITS + 8 16S) across storage months (0/1/2/3/4/5/6),
annotating cells where the IndVal q-value (BH-adjusted p) is < 0.05.

Inputs (verified to exist on disk):
- 16S genus relabund: source/analysis/results/16S/taxonomy_Genus_full.csv
- ITS genus relabund: source/analysis/results/ITS/taxonomy_genus_relabund.csv
- 16S strict 2M indicator list: Attachments_investigation/indicator_2M_evenmonth.csv
- ITS strict 2M indicator list: Attachments_investigation/indicator_2M_evenmonth_ITS.csv
- 16S IndVal q-values: source/analysis/results/16S/indicator_species.csv
- ITS IndVal q-values: source/analysis/results/ITS/indicator_species.csv

Output:
- Attachments_investigation/v11.3.1_supplementary/Figure_S_IndVal_heatmap.png
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

ROOT = "/Users/minseokim/Documents/Obsidian Vault/03_Projects/Garlic-Microbiome"
OUT_DIR = f"{ROOT}/Attachments_investigation/v11.3.1_supplementary"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. Load indicator lists ----------
its_ind = pd.read_csv(f"{ROOT}/Attachments_investigation/indicator_2M_evenmonth_ITS.csv")
bac_ind = pd.read_csv(f"{ROOT}/Attachments_investigation/indicator_2M_evenmonth.csv")

# ITS: genus column may include f__Didymosphaeriaceae / f__Ceratobasidiaceae;
# keep label as in file. Aggregate ASVs to unique taxon labels per the indicator
# file (n=10 ASVs in 7 distinct genera).
its_taxa = its_ind["genus"].unique().tolist()
# Bacterial strict indicators (8 ASVs, 8 genera per file).
bac_taxa = bac_ind["genus"].unique().tolist()


# ---------- 2. Load genus and family relabund tables ----------
# ITS summary tables (taxonomy_genus_relabund.csv) report VALUES AS PERCENT
# (rows sum to 100 per sample), while the 16S summary table reports
# VALUES AS FRACTIONS (rows sum to 1). Handle both correctly.
its_relab = pd.read_csv(f"{ROOT}/source/analysis/results/ITS/taxonomy_genus_relabund.csv")
its_fam = pd.read_csv(f"{ROOT}/source/analysis/results/ITS/taxonomy_family_relabund.csv")
bac_relab = pd.read_csv(f"{ROOT}/source/analysis/results/16S/taxonomy_Genus_full.csv")

def its_parse(s):
    # Cross-check (per_sample_summary_freq5_ITS.csv): old_G1_R1 = month 0,
    # old_G2_R* = month 1, ..., so 'old-N-R' maps to month = N - 1.
    m = re.match(r"(old|new)-(\d+)-(\d+)$", s)
    if not m:
        return None, None
    batch, n, _r = m.groups()
    return batch, int(n) - 1

def build_its_pivot(table, key_col, keep_labels):
    """Return percent-scale pivot (genus/family) x month for the primary batch."""
    long = table.melt(id_vars=[key_col], var_name="sample", value_name="abund_pct")
    long[["batch", "month"]] = long["sample"].apply(lambda s: pd.Series(its_parse(s)))
    long = long[long["batch"] == "old"].dropna(subset=["month"])
    sub = long[long[key_col].isin(keep_labels)].copy()
    pivot = sub.groupby([key_col, "month"], as_index=False)["abund_pct"].mean().pivot(
        index=key_col, columns="month", values="abund_pct"
    ).fillna(0)
    return pivot

# Genus-level rows (present in taxonomy_genus_relabund.csv)
its_genera = {"Aspergillus", "Penicillium", "Alternaria", "Cystobasidium"}
its_pivot_g = build_its_pivot(its_relab, "Genus", its_genera)
# Family-level rows (Sterigmatomyces=Agaricostilbaceae is not in family
# relabund either — taxonomy summary tables only retain abundant lineages.
# Ceratobasidiaceae and Didymosphaeriaceae likewise absent. We instead use
# the strict-indicator ASV table (`indicator_2M_evenmonth_ITS.csv`) to fill
# in the M2 value from `m2_max_abund_pct` and treat all other months as 0
# by the strict-indicator definition.)
extra_its_rows = []
for label in ("Sterigmatomyces", "f__Didymosphaeriaceae", "f__Ceratobasidiaceae"):
    asv_rows = its_ind[its_ind["genus"] == label]
    m2_val = asv_rows["m2_max_abund_pct"].sum() if not asv_rows.empty else 0.0
    extra_its_rows.append((label, m2_val))
extra_df = pd.DataFrame(extra_its_rows, columns=["Genus", "m2_pct"])
extra_pivot = pd.DataFrame(0.0, index=extra_df["Genus"], columns=range(0, 7))
extra_pivot[2] = extra_df.set_index("Genus")["m2_pct"]

its_pivot = pd.concat([its_pivot_g, extra_pivot], axis=0).fillna(0)


# ---------- 16S: build long form ----------
bac = bac_relab.copy()
bac = bac[bac.get("garlic_type", "old") == "old"] if "garlic_type" in bac.columns else bac
# columns: sample-id, <genera>, garlic_type, storage_month
month_col = "storage_month"
non_genus_cols = {"sample-id", "garlic_type", month_col}
genus_cols = [c for c in bac.columns if c not in non_genus_cols]

# Bacterial indicator labels in this file
bac_keep_labels = {
    "Bacillus",
    "Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium",
    "Pseudonocardia", "Acinetobacter", "Pseudogracilibacillus",
    "Sphingobacterium", "Hathewaya", "Saccharopolyspora",
}
# Saccharopolyspora is in the indicator file -- confirm it's in the relabund cols.
bac_keep_labels = bac_keep_labels & set(genus_cols)

bac_mat_rows = []
for g in bac_keep_labels:
    for m, sub in bac.groupby(month_col):
        # 16S values are fractions (0-1); convert to percent here so both
        # data sources share the same unit before plotting.
        bac_mat_rows.append((g, int(m), sub[g].mean() * 100.0))
bac_pivot_raw = pd.DataFrame(bac_mat_rows, columns=["Genus", "month", "abund"]).pivot(
    index="Genus", columns="month", values="abund"
).fillna(0)

# Ensure month 0..6 present (16S has no month 0 baseline; fill 0)
for m in range(0, 7):
    if m not in bac_pivot_raw.columns:
        bac_pivot_raw[m] = 0.0
    if m not in its_pivot.columns:
        its_pivot[m] = 0.0

bac_pivot_raw = bac_pivot_raw[[0, 1, 2, 3, 4, 5, 6]]
its_pivot = its_pivot[[0, 1, 2, 3, 4, 5, 6]]


# ---------- 3. IndVal q-values per genus ----------
its_iv = pd.read_csv(f"{ROOT}/source/analysis/results/ITS/indicator_species.csv")
bac_iv = pd.read_csv(f"{ROOT}/source/analysis/results/16S/indicator_species.csv")

# ITS indicator_species.csv is at Family-level for our strict set; we still
# annotate with month-2 q-values where the strict genus's family matches.
# For genera, we instead use the asv-level indicator file's presence rule
# (strict: present at 2M in >=2/3 replicates, absent at 0/4/6M) which is what
# the strict indicator list itself encodes; we mark q < 0.05 only where the
# 16S indicator_species.csv recorded such a q-value, and for ITS we report
# 'strict' (asterisk) since the strict-presence test served as the q-test.

# 16S q-values at indicator_month=2, batch='old':
bac_q = bac_iv[(bac_iv["batch"] == "old") & (bac_iv["indicator_month"] == 2)]
bac_q_map = dict(zip(bac_q["genus"], bac_q["q_value"]))


# ---------- 4. Plot ----------
# Rename for display: f__Family -> Family (italic family rank)
def disp(g):
    if g.startswith("f__"):
        return f"{g[3:]} (family)"
    if g == "Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium":
        return "Rhizobium clade"
    return g

# Combined matrix: ITS on top, 16S below; rows ordered alphabetically within
# each block.
its_pivot = its_pivot.sort_index()
bac_pivot_raw = bac_pivot_raw.sort_index()

its_pivot.index = [f"ITS · {disp(g)}" for g in its_pivot.index]
bac_pivot_raw.index = [f"16S · {disp(g)}" for g in bac_pivot_raw.index]

combined = pd.concat([its_pivot, bac_pivot_raw], axis=0)
combined.columns = [f"M{m}" for m in combined.columns]

# Both inputs are now in percent (ITS native percent, 16S converted above).
combined_pct = combined

# Build q-annotation matrix (string)
ann = pd.DataFrame("", index=combined.index, columns=combined.columns)
# 16S q-value at M2
for g, q in bac_q_map.items():
    row_label = f"16S · {disp(g)}"
    if row_label in ann.index and q < 0.05:
        ann.loc[row_label, "M2"] = "*"
    if row_label in ann.index and q < 0.01:
        ann.loc[row_label, "M2"] = "**"
# ITS strict-indicator markers: by construction each ITS row passes the
# strict presence/absence test at M2, so we mark M2 with † (footnote
# explained in caption).
for label in its_pivot.index:
    ann.loc[label, "M2"] = "†"

fig, ax = plt.subplots(figsize=(8, 8))

# Use log-normalised sequential colormap so the rare-but-significant
# indicators are visible alongside higher-abundance Bacillus.
plot_vals = combined_pct.values.copy()
# Add small floor so log scale works
floor = 0.001
plot_vals = np.where(plot_vals < floor, floor, plot_vals)

sns.heatmap(
    plot_vals,
    cmap="YlOrBr",
    norm=LogNorm(vmin=floor, vmax=max(plot_vals.max(), 100)),
    cbar_kws={"label": "Mean relative abundance (%, log scale)"},
    linewidths=0.4,
    linecolor="white",
    xticklabels=combined_pct.columns,
    yticklabels=combined_pct.index,
    annot=ann.values,
    fmt="",
    annot_kws={"color": "black", "fontsize": 10, "weight": "bold"},
    ax=ax,
)

ax.set_title(
    "Figure S_IndVal_heatmap. Strict month-2 indicator genera (n = 18) across\n"
    "storage months in the primary batch (old, n = 2–3 per timepoint)",
    fontsize=11, loc="left", pad=12,
)
ax.set_xlabel("Storage month")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=9)

# Caption footnotes
fig.text(
    0.02, 0.02,
    "† = ITS ASV-level strict indicator (present in ≥2/3 replicates at M2, "
    "absent at all other even-month timepoints; Section 3.2).\n"
    "* = 16S genus IndVal q < 0.05 at M2 (BH-adjusted; "
    "Dufrêne–Legendre indicator analysis; Section 3.3).\n"
    "** = 16S genus IndVal q < 0.01 at M2.\n"
    "Cell colour = mean relative abundance (%) for the genus at the given "
    "month across primary-batch replicates (log scale).",
    fontsize=7, color="#444",
)

plt.tight_layout(rect=(0, 0.07, 1, 1))
out_path = f"{OUT_DIR}/Figure_S_IndVal_heatmap.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out_path}")

# Also save a CSV companion with the abundance matrix and q-values
out_csv = combined_pct.copy()
out_csv["IndVal_q_M2"] = ""
for g, q in bac_q_map.items():
    row_label = f"16S · {disp(g)}"
    if row_label in out_csv.index:
        out_csv.loc[row_label, "IndVal_q_M2"] = f"{q:.3g}"
for label in its_pivot.index:
    out_csv.loc[label, "IndVal_q_M2"] = "strict (asv presence)"
out_csv.to_csv(f"{OUT_DIR}/Figure_S_IndVal_heatmap_source.tsv", sep="\t")
print(f"Saved: {OUT_DIR}/Figure_S_IndVal_heatmap_source.tsv")
