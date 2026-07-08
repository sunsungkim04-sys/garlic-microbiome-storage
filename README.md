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
separate homogenates and are given for every month sampled.

ASV counts in this archive: 16S DADA2 982 → 587 after min-frequency = 5 and Bacteria/Archaea
filtering; ITS DADA2 235 → 95 after min-frequency = 5 and `k__Fungi` filtering.

## Repository layout
```
data/            processed 16S/ITS feature tables (DADA2), taxonomy, qPCR/CFU table
supplementary/   consolidated Table S2/S3a/S3b/S6 TSVs (see supplementary/README.md)
scripts/         analysis scripts (compositional sensitivity, stage Mantel, IndVal, depth sweep, ...)
figures/         figure-regeneration scripts
```

## Reproduce
QIIME2 v2024.10 (16S SILVA 138.1 V4 99%, ITS UNITE v9 dynamic), then Python 3.11
(scikit-bio 0.6.0, SciPy 1.11.4, statsmodels, scikit-posthocs).

`python scripts/15_indval_heatmap.py` regenerates manuscript Figure S10 from this archive alone
(e.g. 16S *Bacillus* 2M = 16.70 %, ITS *Penicillium* = 50.22 / 94.58 / 90.55 / 99.93 %).

## Versions
- **v2.0** — amplicon dataset restricted to the even-month timepoints (0/2/4/6 M, n = 12) analysed
  in the manuscript; `scripts/15_indval_heatmap.py` rewritten to be self-contained and to match
  manuscript Figure S10.
- **v1.0** — initial release; superseded.

## Citation
See the manuscript. Code/data released under CC-BY-4.0 (see `LICENSE`).
