# Garlic storage microbiome — 6-month dual-marker time series

Data and analysis code for *"Community monitoring flags spoilage risk in cold-stored garlic
two months before visible decay"* (Ok & Kim et al., LWT — Food Science and Technology).

Cold-stored garlic (*Allium sativum* L.), single lot, 0–6 months (4–6 °C), profiled by 16S
rRNA V4 and fungal ITS2 amplicon sequencing, qPCR, culture (CFU), and cryo-SEM.

## Data availability
- **Raw reads:** NCBI SRA, BioProject `PRJNA1490345` (20 samples × 16S/ITS = 40 libraries).
- **This archive (Zenodo [`10.5281/zenodo.21230815`](https://doi.org/10.5281/zenodo.21230815)):** processed feature tables, taxonomy, qPCR data, and analysis code.

## Repository layout
```
data/            processed 16S/ITS feature tables (DADA2), taxonomy, qPCR table
supplementary/   consolidated Table S2/S3a/S3b/S8 TSVs (see supplementary/README.md)
scripts/         analysis scripts (compositional sensitivity, stage Mantel, IndVal, depth sweep, ...)
figures/         figure-regeneration scripts
```

## Reproduce
QIIME2 v2024.10 (16S SILVA 138.1 V4 99%, ITS UNITE v9 dynamic), then Python 3.11
(scikit-bio 0.6.0, SciPy 1.11.4, statsmodels, scikit-posthocs). Sample IDs `G1–G7` = months `0–6`.

## Citation
See the manuscript. Code/data released under CC-BY-4.0 (see `LICENSE`).
