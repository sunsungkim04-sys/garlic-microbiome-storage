"""Script 1 (★★★) — Cross-marker bulb identity check.

16S OLD vs ITS OLD: same bulb same aliquot? Procrustes ρ=0.82의 전제 검증.

Output:
  Attachments_investigation/sample_identity_audit.csv
  Attachments_investigation/sample_identity_audit.md
"""
import os
import re
import csv
import sys
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import load_table

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
OUT = f"{QROOT}/Attachments_investigation"
M16S = f"{QROOT}/16S_old/manifest.tsv"
MITS = f"{QROOT}/ITS_old/manifest.tsv"
T16S = f"{QROOT}/16S_old/table-dada2.qza"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
SESS = "/home1/minseo1101/garlic_project/SESSION_NOTES.md"
GDRAFT_DIR = "/home1/minseo1101/garlic_project/source"


def parse_manifest(path):
    out = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            sid, fp = line.rstrip("\n").split("\t", 1)
            fn = os.path.basename(fp)
            out[sid] = fn
    return out


def parse_fastq_meta(fn):
    """Z-{recv}-Z-24-{lot}-old-{group}-{rep}_S{snum}_L001_R1_001.fastq.gz"""
    m = re.match(r"Z-(\d+)-Z-24-(\d+)-old-(\d+)-(\d+)_S(\d+)_L001_R1_001\.fastq\.gz", fn)
    if not m:
        return None
    return dict(receipt=m.group(1), lot=m.group(2),
                group=int(m.group(3)), rep=int(m.group(4)),
                snum=int(m.group(5)))


def main():
    os.makedirs(OUT, exist_ok=True)

    m16 = parse_manifest(M16S)
    mit = parse_manifest(MITS)

    sids_16 = set(m16)
    sids_it = set(mit)
    common = sorted(sids_16 & sids_it, key=lambda s: (int(re.search(r"G(\d+)", s).group(1)),
                                                       int(re.search(r"R(\d+)", s).group(1))))
    only_16 = sids_16 - sids_it
    only_it = sids_it - sids_16

    print(f"16S samples: {len(sids_16)}  ITS samples: {len(sids_it)}")
    print(f"Common SampleIDs: {len(common)}")
    print(f"Only in 16S: {sorted(only_16)}")
    print(f"Only in ITS: {sorted(only_it)}")

    # FASTQ meta per sample
    meta16 = {s: parse_fastq_meta(m16[s]) for s in common}
    metait = {s: parse_fastq_meta(mit[s]) for s in common}

    # Unique receipts/lots
    rec16 = {meta16[s]["receipt"] for s in common}
    recit = {metait[s]["receipt"] for s in common}
    lot16 = {meta16[s]["lot"] for s in common}
    lotit = {metait[s]["lot"] for s in common}
    print(f"\n16S receipt: {rec16}  lot: {lot16}")
    print(f"ITS receipt: {recit}  lot: {lotit}")

    # Read counts per sample (from table-dada2)
    print("\nLoading raw DADA2 totals (16S)...")
    s16, _, mat16 = load_table(T16S)
    print("Loading raw DADA2 totals (ITS)...")
    sit, _, matit = load_table(TITS)
    tot16 = dict(zip(s16, mat16.sum(axis=1).astype(int)))
    totit = dict(zip(sit, matit.sum(axis=1).astype(int)))

    # Spearman over the common samples
    arr16 = np.array([tot16.get(s, 0) for s in common], dtype=float)
    arrit = np.array([totit.get(s, 0) for s in common], dtype=float)
    from scipy.stats import spearmanr, pearsonr
    rs, ps = spearmanr(arr16, arrit)
    rp, pp = pearsonr(arr16, arrit)

    print(f"\nSpearman (raw DADA2 totals 16S vs ITS): rho={rs:.3f} p={ps:.3g}")
    print(f"Pearson  (raw DADA2 totals 16S vs ITS): r={rp:.3f}  p={pp:.3g}")

    # Group-level check: same bulb if both come from same harvest. Group order match.
    # Receipt date check
    recv_diff_days_note = "16S Z-202109 (Sep 2021) vs ITS Z-202112 (Dec 2021) — different sequencing dates"

    # CSV per sample
    csv_path = f"{OUT}/sample_identity_audit.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["sample", "in_16S", "in_ITS", "group_16S", "rep_16S",
                    "group_ITS", "rep_ITS", "group_match", "rep_match",
                    "lot_16S", "lot_ITS", "receipt_16S", "receipt_ITS",
                    "total_reads_16S", "total_reads_ITS"])
        for s in sorted(sids_16 | sids_it):
            in16 = s in sids_16
            init = s in sids_it
            m1 = meta16.get(s) if in16 else None
            m2 = metait.get(s) if init else None
            w.writerow([s, in16, init,
                        m1["group"] if m1 else "",
                        m1["rep"] if m1 else "",
                        m2["group"] if m2 else "",
                        m2["rep"] if m2 else "",
                        (m1 and m2 and m1["group"] == m2["group"]) if (m1 and m2) else "",
                        (m1 and m2 and m1["rep"] == m2["rep"]) if (m1 and m2) else "",
                        m1["lot"] if m1 else "",
                        m2["lot"] if m2 else "",
                        m1["receipt"] if m1 else "",
                        m2["receipt"] if m2 else "",
                        tot16.get(s, ""), totit.get(s, "")])
    print(f"\nWrote {csv_path}")

    # Heuristic verdict
    all_group_rep_match = all(
        meta16[s]["group"] == metait[s]["group"] and meta16[s]["rep"] == metait[s]["rep"]
        for s in common
    )
    same_n = len(common) == len(sids_16) == len(sids_it)

    # Grep SESSION_NOTES + Q&A
    kw_pat = r"split|aliquot|same bulb|독립|동일.{0,3}(추출|시료|벌브|구근)|구근.{0,5}분"
    hits = []
    for path in [SESS]:
        try:
            r = subprocess.run(["grep", "-niE", "-A1", "-B1", kw_pat, path],
                               capture_output=True, text=True)
            if r.stdout:
                hits.append(f"### {path}\n```\n{r.stdout}\n```")
        except Exception as e:
            hits.append(f"### {path} — grep failed: {e}")
    # Also scan source dir
    try:
        r = subprocess.run(["grep", "-RniE", "-A1", "-B1", "--include=*.md", "--include=*.txt",
                            kw_pat, GDRAFT_DIR],
                           capture_output=True, text=True)
        if r.stdout:
            hits.append(f"### {GDRAFT_DIR} (md/txt)\n```\n{r.stdout[:4000]}\n```")
    except Exception as e:
        hits.append(f"### {GDRAFT_DIR} — grep failed: {e}")

    # Verdict
    if all_group_rep_match and same_n:
        verdict = "CONFIRMED_SAME_BULB_LIKELY"
        reason = ("SampleID 21/21 일치 + filename 의 (group, rep) tag 완전 일치 + "
                  "manifest 명명 규칙 (`old_G{n}_R{r}` ↔ `old-{n}-{r}`) 정합. "
                  "단 16S Z-202109 / ITS Z-202112 receipt 분리 → "
                  "DNA aliquot split 인지 fresh re-extraction 인지는 SESSION_NOTES 확인 필요.")
    elif same_n:
        verdict = "PARTIAL_MISMATCH"
        reason = "SampleID 매칭 OK 이나 (group, rep) tag mismatch — 직접 확인 필요"
    else:
        verdict = "DIFFERENT_SAMPLE_POOL"
        reason = "SampleID 집합 불일치"

    md = [
        "# Sample identity audit (v11.3.1 script 1)",
        "",
        f"- 16S OLD sample count: **{len(sids_16)}**",
        f"- ITS OLD sample count: **{len(sids_it)}**",
        f"- Common SampleIDs: **{len(common)}**",
        f"- Only in 16S: `{sorted(only_16)}`",
        f"- Only in ITS: `{sorted(only_it)}`",
        "",
        "## FASTQ filename metadata",
        f"- 16S receipt(s): `{rec16}`   lot(s): `{lot16}`",
        f"- ITS receipt(s): `{recit}`   lot(s): `{lotit}`",
        f"- Note: {recv_diff_days_note}",
        "",
        "## Read-count correlation (DADA2 raw totals)",
        f"- Spearman ρ = {rs:.3f}, p = {ps:.3g}",
        f"- Pearson  r = {rp:.3f}, p = {pp:.3g}",
        "",
        "## Verdict",
        f"**{verdict}**",
        "",
        reason,
        "",
        "## Keyword scan (SESSION_NOTES + source/)",
        ""
    ] + hits + [
        "",
        "## Manuscript action",
        ("- `CONFIRMED_SAME_BULB_LIKELY` → §3.4 그대로 유지. §2.1 에 "
         "\"matched bulb identity confirmed via filename (group, rep) tag\" 한 줄."),
        ("- `PARTIAL_MISMATCH` / `Independent extraction` → §3.4 framing 약화: "
         "\"matched timepoint, separate extraction (different sequencing receipts)\""),
        "- `Unknown` 잔존 시 § 4.6 limitation 한 줄.",
    ]
    md_path = f"{OUT}/sample_identity_audit.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {md_path}")
    print(f"\nVERDICT: {verdict}")
    print(reason)


if __name__ == "__main__":
    main()
