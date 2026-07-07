"""Script 13 — Phylogenetic signal for 10 strict month-2 indicator ASVs.

10 indicators from indicator_2M_evenmonth_ITS.csv (ITS, OLD batch).

Faith's PD requires a phylogenetic tree.  ITS_old does not currently
ship a rooted-tree.qza, so we build one on the fly with mafft+fasttree
restricted to the kept ASV set (already filtered).

Steps:
  1. Extract ITS rep-seqs from rep-seqs-clean.qza (or dada2 fallback).
  2. Filter rep-seqs to the kept ASVs that pass the same contam +
     min-freq filter (matches results table).
  3. Build alignment (mafft --auto) and tree (fasttree -nt -gtr).
  4. Compute Faith's PD = sum of branch lengths in subtree induced
     by the 10 indicator ASVs.
  5. Null: N=999 random ASV draws of size 10 from the same pool.
  6. z-score = (obs - null_mean) / null_std, empirical p =
     fraction of null PD ≥ obs.
  7. NRI/NTI: also computable from pairwise dist matrix; we compute
     them but explicitly note small-set caveat (size=10).

Outputs:
  v11.3.1_supplementary/phylo_signal_summary.txt
  v11.3.1_supplementary/phylo_signal_indicators.tsv
"""
import os
import sys
import subprocess
import zipfile
import io
import numpy as np
import pandas as pd

sys.path.insert(0, "/home1/minseo1101/garlic_project/analysis/scripts/v11.3.1")
from _helpers import load_table, load_taxonomy, is_contam_ITS, parse_genus

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"
REPSEQ = f"{QROOT}/ITS_old/rep-seqs-clean.qza"
INDICATOR_CSV = f"{QROOT}/indicator_2M_evenmonth_ITS.csv"
OUT = "/home1/minseo1101/garlic_project/analysis/results/v11.3.1_supplementary"
WORK = f"{OUT}/_phylo_work"

SEED = 42
N_PERM = 999


def extract_repseqs(qza_path, out_fasta):
    """rep-seqs.qza → out_fasta (DNA sequences keyed by ASV id)."""
    with zipfile.ZipFile(qza_path) as z:
        fpath = [n for n in z.namelist() if n.endswith("dna-sequences.fasta")][0]
        with z.open(fpath) as f:
            with open(out_fasta, "wb") as o:
                o.write(f.read())


def parse_fasta(path):
    out = {}
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    out[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        out[cur_id] = "".join(cur_seq)
    return out


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")


def newick_parse_branch_lengths(nwk):
    """Return dict {leaf: cumulative-root-distance} and list of edges
    (parent_idx, child_idx, length, child_is_leaf, leaf_label).
    Implements a minimal Newick parser tolerant of unrooted output."""
    # Use ete3? Not available presumably.  Use a manual parser.
    s = nwk.strip().rstrip(";")
    # nodes
    nodes = []  # each: dict(children=[], length=0.0, label=None)
    stack = []
    def new_node():
        nodes.append(dict(children=[], length=0.0, label=None))
        return len(nodes) - 1
    root = new_node()
    cur = root
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            child = new_node()
            nodes[cur]["children"].append(child)
            stack.append(cur)
            cur = child
            i += 1
        elif c == ")":
            # done with this node
            cur = stack.pop()
            i += 1
            # next may be label or :length
            j = i
            while j < len(s) and s[j] not in ",():":
                j += 1
            if j > i:
                nodes[cur]["label"] = s[i:j]
            i = j
            if i < len(s) and s[i] == ":":
                j = i + 1
                while j < len(s) and s[j] not in ",()":
                    j += 1
                nodes[cur]["length"] = float(s[i+1:j])
                i = j
        elif c == ",":
            i += 1
        else:
            # leaf label
            j = i
            while j < len(s) and s[j] not in ",():":
                j += 1
            label = s[i:j]
            length = 0.0
            if ":" in label:
                lab, lng = label.split(":", 1)
                label = lab
                length = float(lng)
            leaf = new_node()
            nodes[leaf]["label"] = label
            nodes[leaf]["length"] = length
            nodes[cur]["children"].append(leaf)
            i = j
    return nodes, root


def faiths_pd(nodes, root, leaf_set):
    """Sum of edge lengths in minimal subtree containing leaf_set."""
    # For each node, determine if it's an ancestor of any leaf in set.
    in_set = [False] * len(nodes)
    leaf_idx = {}
    def index_leaves(n):
        if not nodes[n]["children"]:
            leaf_idx[nodes[n]["label"]] = n
            return
        for c in nodes[n]["children"]:
            index_leaves(c)
    index_leaves(root)
    target = set(leaf_set) & set(leaf_idx.keys())
    if not target:
        return 0.0
    def mark(n):
        if not nodes[n]["children"]:
            return nodes[n]["label"] in target
        any_child = False
        for c in nodes[n]["children"]:
            r = mark(c)
            if r:
                any_child = True
        in_set[n] = any_child or (nodes[n]["label"] in target)
        return in_set[n]
    mark(root)
    # Sum lengths of edges between in_set nodes (descend from root)
    def sum_pd(n):
        total = 0.0
        for c in nodes[n]["children"]:
            if in_set[c]:
                total += nodes[c]["length"] + sum_pd(c)
        return total
    return sum_pd(root)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    # Load indicator ASVs (10 of them)
    ind = pd.read_csv(INDICATOR_CSV)
    indicator_asvs = ind["asv"].tolist()
    print(f"Indicator ASVs (n={len(indicator_asvs)}): {indicator_asvs}")

    # Load full kept-ASV pool (after filter — same as paper's pool)
    sids, asvs, mat = load_table(TITS)
    tax = load_taxonomy(TAXIT)
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not is_contam_ITS(tax.get(a, "")) for a in asvs])
    asvs_pool = [a for a, k in zip(asvs, keep) if k]
    print(f"ITS kept ASV pool size: {len(asvs_pool)}")

    # Sanity: how many indicator ASVs are in the pool?
    in_pool = [a for a in indicator_asvs if a in asvs_pool]
    print(f"  Indicator ASVs present in pool: {len(in_pool)}/{len(indicator_asvs)}")

    # Extract rep-seqs
    fasta_all = f"{WORK}/repseqs_all.fasta"
    extract_repseqs(REPSEQ, fasta_all)
    seqs_all = parse_fasta(fasta_all)
    print(f"  rep-seqs fasta: {len(seqs_all)} sequences")

    # Filter to pool
    seqs_pool = {a: seqs_all[a] for a in asvs_pool if a in seqs_all}
    print(f"  rep-seqs after pool filter: {len(seqs_pool)}")
    fasta_pool = f"{WORK}/repseqs_pool.fasta"
    write_fasta(seqs_pool, fasta_pool)

    # Align with mafft
    aln = f"{WORK}/aln.fasta"
    if not os.path.exists(aln) or os.path.getsize(aln) == 0:
        print("Running mafft (this may take ~minutes)…")
        with open(aln, "w") as f:
            r = subprocess.run(["mafft", "--auto", "--thread", "4", fasta_pool],
                               stdout=f, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                print("mafft failed:", r.stderr[-500:])
                sys.exit(1)
    # FastTree
    tree_path = f"{WORK}/tree.nwk"
    if not os.path.exists(tree_path) or os.path.getsize(tree_path) == 0:
        print("Running FastTree…")
        r = subprocess.run(["fasttree", "-nt", "-gtr", "-quiet", aln],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("fasttree failed:", r.stderr[-500:])
            sys.exit(1)
        with open(tree_path, "w") as f:
            f.write(r.stdout)

    nwk = open(tree_path).read().strip()
    print(f"Tree built ({len(nwk)} chars)")
    nodes, root = newick_parse_branch_lengths(nwk)
    leaves = [n["label"] for n in nodes if not n["children"] and n["label"]]
    print(f"  Tree leaves: {len(leaves)}")

    # Observed PD
    obs_pd = faiths_pd(nodes, root, in_pool)
    print(f"Observed Faith's PD (n_indicator={len(in_pool)}): {obs_pd:.4f}")

    # Null distribution
    rng = np.random.default_rng(SEED)
    pool_set = [a for a in asvs_pool if a in seqs_pool]  # only those in tree
    print(f"  Drawing N={N_PERM} random sets of size {len(in_pool)} …")
    null_pd = np.empty(N_PERM)
    for k in range(N_PERM):
        draw = rng.choice(pool_set, size=len(in_pool), replace=False).tolist()
        null_pd[k] = faiths_pd(nodes, root, draw)
    mean_null = null_pd.mean()
    std_null = null_pd.std(ddof=1)
    z = (obs_pd - mean_null) / std_null if std_null > 0 else np.nan
    p_emp = (np.sum(null_pd >= obs_pd) + 1) / (N_PERM + 1)
    p_low = (np.sum(null_pd <= obs_pd) + 1) / (N_PERM + 1)  # for clustering test
    print(f"Null mean={mean_null:.4f}  sd={std_null:.4f}")
    print(f"  z-score = {z:.3f}")
    print(f"  Empirical p (PD ≥ obs, dispersion test): {p_emp:.4f}")
    print(f"  Empirical p (PD ≤ obs, clustering test): {p_low:.4f}")

    # Save summary
    summary_path = f"{OUT}/phylo_signal_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Phylogenetic signal — 10 strict month-2 ITS indicator ASVs\n")
        f.write(f"Date: 2026-05-13 v11.3.1 supplementary\n\n")
        f.write(f"n_indicator (in tree): {len(in_pool)}\n")
        f.write(f"pool_size: {len(pool_set)}\n")
        f.write(f"observed_Faith_PD: {obs_pd:.4f}\n")
        f.write(f"null_PD_mean: {mean_null:.4f}\n")
        f.write(f"null_PD_sd:   {std_null:.4f}\n")
        f.write(f"z_score:      {z:.4f}\n")
        f.write(f"p_PD_geq_obs (over-dispersed test): {p_emp:.4f}\n")
        f.write(f"p_PD_leq_obs (clustered test):     {p_low:.4f}\n\n")
        f.write(f"Interpretation:\n")
        if p_low < 0.05:
            f.write("  Indicator ASVs are phylogenetically CLUSTERED (clade-conserved).\n")
        elif p_emp < 0.05:
            f.write("  Indicator ASVs are phylogenetically OVER-DISPERSED (multiple clades).\n")
        else:
            f.write("  No significant phylogenetic signal detected (random tree placement).\n")
        f.write("\nNRI/NTI: not computed in this script. Mean pairwise distance among\n")
        f.write("the 10 ASVs vs null sets could give NRI; we report Faith's PD only\n")
        f.write("here for simplicity and because n=10 is small for NRI/NTI to be stable.\n")

    print(f"Wrote {summary_path}")

    # Save indicator table with tree info
    rows = []
    for a in indicator_asvs:
        rows.append(dict(asv=a, in_pool=(a in pool_set), in_tree=(a in leaves),
                         taxonomy=tax.get(a, ""), genus=parse_genus(tax.get(a, ""))))
    pd.DataFrame(rows).to_csv(f"{OUT}/phylo_signal_indicators.tsv",
                              sep="\t", index=False)
    print(f"Wrote {OUT}/phylo_signal_indicators.tsv")


if __name__ == "__main__":
    main()
