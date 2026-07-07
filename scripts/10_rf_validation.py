"""Script 10 — Random Forest validation (ITS ASV → storage month).

Methods §2.5: RF, 500 trees, 5-fold CV.
Data: ITS_old even-month (G1/G3/G5/G7, n=12, 3 reps × 4 months).

CAVEAT (in script header for honesty):
  - n=12 samples, 4 classes (months 0,2,4,6), 3 reps/class.
  - Stratified 5-fold CV on n=12/k=4 classes is impossible
    (need ≥5 per class).  We use LeaveOneOut CV (12 splits) as a
    more honest alternative and 3-fold StratifiedKFold for comparison.
  - With n=3 per class, any reported accuracy must be interpreted
    cautiously; report per-class scores as descriptive only.

Outputs:
  v11.3.1_supplementary/Supplementary_Table_S_RF.tsv  (perf summary)
  v11.3.1_supplementary/rf_confusion_matrix.tsv
  v11.3.1_supplementary/rf_feature_importance_top20.tsv
  v11.3.1_supplementary/Figure_S_RF.png  (confusion matrix + feature importance bar)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              classification_report, accuracy_score)

sys.path.insert(0, "/home1/minseo1101/garlic_project/analysis/scripts/v11.3.1")
from _helpers import (load_table, load_taxonomy, is_contam_ITS, parse_genus,
                      META_EVEN_OLD)

QROOT = "/home1/minseo1101/garlic_project/data/qiime2_reanalysis"
TITS = f"{QROOT}/ITS_old/table-dada2.qza"
TAXIT = f"{QROOT}/ITS_old/taxonomy.qza"
OUT = "/home1/minseo1101/garlic_project/analysis/results/v11.3.1_supplementary"

SEED = 42
N_TREES = 500


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading ITS data …")
    sids, asvs, mat = load_table(TITS)
    tax = load_taxonomy(TAXIT)

    # filter
    total = mat.sum(axis=0)
    keep = (total >= 5) & np.array([not is_contam_ITS(tax.get(a, "Unassigned"))
                                     for a in asvs])
    mat = mat[:, keep]
    asvs = [a for a, k in zip(asvs, keep) if k]
    em_idx = [i for i, s in enumerate(sids) if s in META_EVEN_OLD]
    em_sids = [sids[i] for i in em_idx]
    X = mat[em_idx]
    # relative-abundance normalize per sample
    row_sum = X.sum(axis=1, keepdims=True)
    X = X / np.where(row_sum == 0, 1, row_sum)
    y = np.array([META_EVEN_OLD[s] for s in em_sids])
    print(f"  X shape: {X.shape}   y: {y}")

    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED,
                                 n_jobs=-1)

    # LOO CV
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(rf, X, y, cv=loo, n_jobs=1)
    acc_loo = accuracy_score(y, y_pred_loo)
    print(f"  LOO accuracy: {acc_loo:.3f}")

    # Stratified 3-fold (n_splits=3, n=12/4classes=3 reps so works)
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    y_pred_3 = cross_val_predict(rf, X, y, cv=skf3, n_jobs=1)
    acc_3 = accuracy_score(y, y_pred_3)
    print(f"  Stratified 3-fold accuracy: {acc_3:.3f}")

    # Methods says 5-fold; we attempt but skf=5 fails with 3 per class
    try:
        skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred_5 = cross_val_predict(rf, X, y, cv=skf5, n_jobs=1)
        acc_5 = accuracy_score(y, y_pred_5)
        print(f"  Stratified 5-fold accuracy: {acc_5:.3f}")
    except Exception as e:
        acc_5 = np.nan
        print(f"  5-fold CV failed: {e}")

    # Confusion matrix (use LOO predictions — most stable)
    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred_loo, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{m}M" for m in labels],
                         columns=[f"pred_{m}M" for m in labels])
    cm_df.to_csv(f"{OUT}/rf_confusion_matrix.tsv", sep="\t")
    print(f"  Confusion matrix:\n{cm_df}")

    pr, rc, f1, sup = precision_recall_fscore_support(y, y_pred_loo, labels=labels,
                                                       zero_division=0)
    perf = pd.DataFrame({"month": labels, "precision": pr, "recall": rc,
                         "f1": f1, "support": sup})
    print(f"\nPer-class:\n{perf}")

    # Feature importance — fit once on full data
    rf_full = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
    rf_full.fit(X, y)
    imp = rf_full.feature_importances_
    order = np.argsort(-imp)[:20]
    fi_rows = []
    for rank, j in enumerate(order, 1):
        asv = asvs[j]
        tx = tax.get(asv, "")
        fi_rows.append([rank, asv, parse_genus(tx), imp[j], tx])
    fi = pd.DataFrame(fi_rows, columns=["rank", "asv", "genus", "importance", "taxonomy"])
    fi.to_csv(f"{OUT}/rf_feature_importance_top20.tsv", sep="\t", index=False)

    # Summary table
    summary = pd.DataFrame([
        {"metric": "LOO_accuracy", "value": acc_loo},
        {"metric": "Stratified3-fold_accuracy", "value": acc_3},
        {"metric": "Stratified5-fold_accuracy", "value": acc_5},
        {"metric": "n_samples", "value": len(y)},
        {"metric": "n_classes", "value": len(labels)},
        {"metric": "n_features", "value": X.shape[1]},
        {"metric": "n_trees", "value": N_TREES},
    ])
    summary.to_csv(f"{OUT}/Supplementary_Table_S_RF.tsv", sep="\t", index=False)
    perf.to_csv(f"{OUT}/rf_per_class.tsv", sep="\t", index=False)

    # Plot 2-panel: confusion + feature importance
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    im = a1.imshow(cm, cmap="Blues")
    a1.set_xticks(range(len(labels)))
    a1.set_xticklabels([f"{m}M" for m in labels])
    a1.set_yticks(range(len(labels)))
    a1.set_yticklabels([f"{m}M" for m in labels])
    a1.set_xlabel("Predicted")
    a1.set_ylabel("True")
    a1.set_title(f"RF Confusion Matrix\nLOO accuracy={acc_loo:.2f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            a1.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=a1, fraction=0.046)

    top = fi.head(15)
    labels_short = [f"{g}\n({a[:6]})" for g, a in zip(top["genus"], top["asv"])]
    a2.barh(range(len(top)), top["importance"], color="#1f78b4")
    a2.set_yticks(range(len(top)))
    a2.set_yticklabels(labels_short, fontsize=7)
    a2.invert_yaxis()
    a2.set_xlabel("Feature importance")
    a2.set_title("RF top-15 ASVs (genus | ASV-id prefix)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/Figure_S_RF.png", dpi=160, bbox_inches="tight")
    print(f"\nWrote outputs to {OUT}/")


if __name__ == "__main__":
    main()
