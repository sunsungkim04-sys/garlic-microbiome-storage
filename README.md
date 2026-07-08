# Garlic storage microbiome — dual-marker time series

Data and analysis code for *"Microbial community restructuring precedes visible decay by two
months in cold-stored garlic"* (Ok & Kim et al., LWT — Food Science and Technology).

Cold-stored garlic (*Allium sativum* L.), single lot, 4–6 °C, profiled by 16S rRNA V4 and
fungal ITS2 amplicon sequencing, qPCR, culture (CFU), and cryo-SEM.

## Data availability
- **Raw reads:** NCBI SRA, BioProject `PRJNA1490345`.
- **This archive (Zenodo, concept DOI [`10.5281/zenodo.21230814`](https://doi.org/10.5281/zenodo.21230814) — always resolves to the latest version):**
  processed feature tables, taxonomy, qPCR/CFU data, and analysis code.

## Scope of this archive
Amplicon data cover the **even-month timepoints analysed in the manuscript: 0, 2, 4 and 6 months,
n = 3 per timepoint (12 samples, 24 libraries)**. Sample IDs `G1`, `G3`, `G5`, `G7` = months
`0`, `2`, `4`, `6`.

`data/quantification_summary.csv` reports culture (CFU) and qPCR, which were prepared from
separate homogenates. qPCR covers 0/2/4/6 M; CFU covers 0/2/4 M (month 6 was not assayed).

ASV counts in this archive: 16S DADA2 982 → 587 after min-frequency = 5 and Bacteria/Archaea
filtering; ITS DADA2 235 → 95 after min-frequency = 5 and `k__Fungi` filtering.

## Repository layout
```
data/            16S/ITS DADA2 feature tables, min-frequency=5 filtered tables, taxonomy, qPCR/CFU table
                 and the rooted 16S phylogeny (Newick)
supplementary/   consolidated Table S2/S3a/S3b/S6 TSVs (see supplementary/README.md)
scripts/         analysis scripts (compositional sensitivity, stage Mantel, IndVal, depth sweep,
                 min-frequency sensitivity, ...)
figures/         figure-regeneration scripts (output written to figures/output/)
```

## Reproduce
QIIME2 v2024.10 (16S SILVA 138.1 V4 99%, ITS UNITE v9 dynamic), then Python 3.11
(scikit-bio 0.6.0, SciPy 1.11.4, statsmodels, scikit-posthocs).

`python scripts/15_indval_heatmap.py` regenerates manuscript Figure S10 from this archive alone
(e.g. 16S *Bacillus* 2M = 16.70 %, ITS *Penicillium* = 50.22 / 94.58 / 90.55 / 99.93 %).

`python scripts/regen_minfreq_sensitivity.py` regenerates manuscript Table S6
(`supplementary/TableS8_minfreq_sensitivity.tsv`) from this archive alone; at min-frequency = 5
the 16S Bray-Curtis PERMANOVA reproduces the manuscript's F = 4.54.

`python scripts/dispersion_audit.py` reproduces every PERMANOVA and PERMDISP statistic reported in
the manuscript, including unweighted UniFrac from `data/16S_rooted_tree.nwk` (16S Bray-Curtis
F = 4.54, PERMDISP F = 12.40, p = 0.022; unweighted UniFrac F = 9.57; ITS Jaccard F = 4.44,
PERMDISP p = 0.76). No external phylogenetics package is required.

`python figures/regen_fig3_16S.py` and `python figures/regen_fig4_quantification.py`
regenerate manuscript Figures 3 and 4 into `figures/output/`. Both compute every statistic
printed on the figure from the data plotted — nothing is hardcoded. Figure 3 reports
PERMANOVA F = 4.54, p = 0.001 with PERMDISP F = 12.40, p = 0.022; Figure 4 reports the CFU
Kruskal-Wallis H = 7.20, p = 0.027. `python scripts/verify_fig4_stats.py` re-derives the
Figure 4 rank statistics and compact-letter display independently.

## Versions
- **v2.2** — unweighted UniFrac corrected to the 587-ASV table (F = 8.45 -> 9.57, R2 = 0.78,
  adjusted R2 = 0.70); `figures/regen_figS9_minfreq.py` added.
- **v2.3** — `data/16S_rooted_tree.nwk` (494 tips) and `scripts/dispersion_audit.py` added, so every
  PERMANOVA/PERMDISP number in the manuscript, unweighted UniFrac included, is reproducible from
  this archive with no external phylogenetics dependency. The phylogeny covers 414 of the 587
  analysed 16S ASVs; Bray-Curtis and Jaccard use all 587.
- **v2.1** — min-frequency filtering is now applied to the 12 analysed samples, so every table in
  this archive is reproducible from the archive itself (16S min-freq=5: 587 ASVs; Table S6 updated).
  `data/quantification_summary.csv` restricted to the analysed timepoints; `figures/regen_figures.py`
  panel 4A corrected to 0/2/4 M (Kruskal-Wallis H = 7.20, p = 0.027); added
  `scripts/regen_minfreq_sensitivity.py`, `figures/regen_fig3_16S.py`,
  `figures/regen_fig4_quantification.py` and `scripts/verify_fig4_stats.py`, so Figures 3 and 4
  and Table S6 are all reproducible from this archive. The Figure 3 statistics box previously read
  "PERMDISP NS"; PERMDISP is F = 12.40, p = 0.022 on the matrix the figure plots.
- **v2.0** — amplicon dataset restricted to the even-month timepoints (0/2/4/6 M, n = 12) analysed
  in the manuscript; `scripts/15_indval_heatmap.py` rewritten to be self-contained and to match
  manuscript Figure S10.
- **v1.0** — initial release; superseded.

## Citation
See the manuscript. Code/data released under CC-BY-4.0 (see `LICENSE`).
