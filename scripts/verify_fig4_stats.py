#!/usr/bin/env python3
"""
verify_fig4_stats.py — recompute the Figure 4 statistics on the even-month frame.

Per-replicate CFU/qPCR values were not preserved; quantification_summary.csv keeps
n / mean / sd / se / min / max per group. For CFU (0/2/4 M) and Fusarium (0/2/4/6 M)
the per-group value ranges are strictly non-overlapping, so every replicate's rank is
uniquely determined by the summary statistics. Kruskal-Wallis and Dunn's test depend
only on ranks, hence both are recovered exactly.

16S qPCR groups overlap, so its statistic cannot be re-derived from the summary; the
published even-month value (H = 4.38, df = 3, p = 0.22) is carried through unchanged.

Run:  python3 scripts/verify_fig4_stats.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "quantification_summary.csv"

EXPECTED = {
    "colony_CFU": dict(H=7.20, df=2, p=0.027),
    "fusarium": dict(H=10.38, df=3, p=0.016),
}


def ranks_from_ranges(ranges):
    """ranges: {group: (min, max)}, n = 3 each. Requires strict separation."""
    order = sorted(ranges, key=lambda g: ranges[g][0])
    for a, b in zip(order, order[1:]):
        if ranges[a][1] >= ranges[b][0]:
            raise ValueError(f"ranges of {a} and {b} overlap; ranks not determined")
    return {g: [3 * i + 1, 3 * i + 2, 3 * i + 3] for i, g in enumerate(order)}, order


def kruskal_from_ranks(rank_map):
    N = sum(len(v) for v in rank_map.values())
    H = 12 / (N * (N + 1)) * sum(sum(v) ** 2 / len(v) for v in rank_map.values()) - 3 * (N + 1)
    df = len(rank_map) - 1
    return H, df, float(stats.chi2.sf(H, df))


def dunn_bh(rank_map):
    N = sum(len(v) for v in rank_map.values())
    mean_rank = {g: float(np.mean(v)) for g, v in rank_map.items()}
    raw = {}
    for a, b in itertools.combinations(rank_map, 2):
        se = np.sqrt((N * (N + 1) / 12) * (1 / len(rank_map[a]) + 1 / len(rank_map[b])))
        raw[(a, b)] = float(2 * stats.norm.sf(abs(mean_rank[a] - mean_rank[b]) / se))
    ordered = sorted(raw.items(), key=lambda kv: kv[1])
    m, prev, adj = len(ordered), 1.0, {}
    for i, (key, p) in enumerate(reversed(ordered)):
        prev = min(prev, p * m / (m - i))
        adj[key] = prev
    return raw, adj


def _maximal_cliques(nodes, adjacent):
    """Bron-Kerbosch without pivoting; group counts here are tiny."""
    out = []

    def neighbours(v):
        return {u for u in nodes if u != v and adjacent(v, u)}

    def bk(r, p, x):
        if not p and not x:
            out.append(set(r))
            return
        for v in list(p):
            nv = neighbours(v)
            bk(r | {v}, p & nv, x & nv)
            p = p - {v}
            x = x | {v}

    bk(set(), set(nodes), set())
    return out


def compact_letters(order, adj, alpha=0.05):
    """Compact-letter display: letters = maximal cliques of the 'not different' graph."""
    def same(a, b):
        return adj.get((a, b), adj.get((b, a), 1.0)) >= alpha

    cliques = _maximal_cliques(order, same)
    # deterministic letter order: by the earliest group each clique contains
    cliques.sort(key=lambda c: min(order.index(g) for g in c))
    out = {g: "" for g in order}
    for letter, clique in zip("abcdefg", cliques):
        for g in clique:
            out[g] += letter
    return out


def main():
    q = pd.read_csv(CSV)
    q["month_int"] = q["month"].str.replace("M", "", regex=False).astype(int)
    ok = True

    for dataset, exp in EXPECTED.items():
        sub = q[q["dataset"] == dataset].sort_values("month_int")
        recs = sub.to_dict("records")
        ranges = {r["month"]: (r["min"], r["max"]) for r in recs}
        rank_map, order = ranks_from_ranges(ranges)
        H, df, p = kruskal_from_ranks(rank_map)
        raw, adj = dunn_bh(rank_map)

        print(f"\n{dataset}  (n = {len(recs) * 3}; groups {[r['month'] for r in recs]})")
        print(f"  separation (low -> high): {' < '.join(order)}")
        print(f"  Kruskal-Wallis  H = {H:.4f}  df = {df}  p = {p:.5f}"
              f"   [expected H = {exp['H']}, df = {exp['df']}, p ~ {exp['p']}]")
        if not (abs(H - exp["H"]) < 0.01 and df == exp["df"] and abs(p - exp["p"]) < 0.001):
            ok = False
            print("  !! MISMATCH vs expected")

        for (a, b), pa in sorted(adj.items()):
            print(f"    Dunn {a} vs {b}: p_raw = {raw[(a, b)]:.5f}  "
                  f"p_adj(BH) = {pa:.4f}  {'SIG' if pa < 0.05 else 'ns'}")

        got = compact_letters(order, adj)
        stored = {r["month"]: r["letter"] for r in recs}
        print(f"  letters recomputed  {dict(sorted(got.items()))}")
        print(f"  letters in csv      {dict(sorted(stored.items()))}")
        for m, lt in stored.items():
            if set(lt) != set(got[m]):
                ok = False
                print(f"  !! letter mismatch at {m}: csv = {lt!r}, recomputed = {got[m]!r}")

    sub16 = q[q["dataset"] == "bacteria_16S"]
    print(f"\nbacteria_16S  (n = {len(sub16) * 3}) — group ranges overlap, so the statistic is not")
    print("  re-derivable from the summary; published even-month H = 4.38, df = 3, p = 0.22 (NS).")
    all_a = sorted(set(sub16["letter"])) == ["a"]
    print(f"  all letters 'a' (no pairwise difference): {all_a}")
    ok = ok and all_a

    months = sorted(q["month_int"].unique())
    print(f"\nmonths present in quantification_summary.csv: {months}")
    if any(m % 2 for m in months):
        ok = False
        print("  !! odd months still present")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
