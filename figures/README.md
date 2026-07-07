v11.4 figure regeneration scripts (publication-ready vector PDF output).

regen_figures.py — Main Fig 1, 2, 3A, 3B, 4, 7, 8 + Supp S4 alpha, S_ITS_heatmap, S9 minfreq
regen_supp.py    — Supp S_compositional, S_RF, S5_NEW_PCoA_16S, S6_NEW_stacked_ITS, S7_NEW_alpha_ITS

Run on lab101 (skbio required from qiime2 env):
  ssh lab101 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate qiime2-amplicon-2024.10 && python3 ~/garlic_project/manuscript/figures/regen_figures.py'

Output: lab101:~/garlic_project/manuscript/figures/v11.4_regen/

