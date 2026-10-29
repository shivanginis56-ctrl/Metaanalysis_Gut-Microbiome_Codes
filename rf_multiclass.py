"""
rf_multiclass.py
────────────────
Multi-class Random Forest: predict race (4 classes) from CLR-transformed
microbiome profiles across all 7,302 samples.

Unlike the binary per-race RF (one-vs-rest), a single 4-class RF captures
how each taxon contributes to distinguishing ALL races simultaneously.
Feature importances reflect taxa that are collectively most informative
for separating the race groups.

Outputs
───────
  rf_multiclass_importances.csv   — per-taxon MDI importance + rank
  rf_multiclass_report.txt        — classification report + OOB score
  reports/figures/fig_rf_multiclass.pdf  — top-30 importance bar chart
                                          + confusion matrix

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.rf_multiclass
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config      import METADATA_FILE, ABUNDANCE_FILE, ROOT_DIR, RF_RANDOM_STATE
from pipeline.data_loader import load_metadata, group_samples, load_abundance, filter_rare_taxa
from pipeline.transforms  import clr_transform

RACES        = ["African", "Asian", "Caucasian", "Hispanic"]
FIGURES_DIR  = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
N_ESTIMATORS = 500       # more trees → stable importances for 4-class
N_JOBS       = -1
CV_FOLDS     = 5


def _banner(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}", flush=True)


def main():
    t0 = time.time()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       MULTI-CLASS RF  —  Race Signature Detection        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Load & preprocess ─────────────────────────────────────────────────────
    _banner("Load data")
    meta   = load_metadata(METADATA_FILE)
    groups = group_samples(meta)

    race_groups: dict[str, list[str]] = {}
    for (bp, race), sids in groups.items():
        race_groups.setdefault(race, []).extend(sids)
    print(f"  Race groups: { {r: len(s) for r, s in race_groups.items()} }")

    taxa_names, otu_ids_list, raw_df = load_abundance(ABUNDANCE_FILE)
    known  = [s for s in raw_df.columns if s in meta]
    raw_df = raw_df[known]

    filtered_df      = filter_rare_taxa(raw_df)
    filtered_taxa    = list(filtered_df.index)
    filtered_otu_ids = [otu_ids_list[taxa_names.index(t)] for t in filtered_taxa]
    print(f"  After pre-filter: {len(filtered_taxa)} taxa")

    _banner("CLR transform")
    clr_df = clr_transform(filtered_df)
    print(f"  CLR matrix: {clr_df.shape[0]} taxa × {clr_df.shape[1]} samples")

    # ── Build X, y ────────────────────────────────────────────────────────────
    _banner("Build feature matrix")
    available = set(clr_df.columns)

    sample_ids  = []
    race_labels = []
    for race in RACES:
        sids = [s for s in race_groups.get(race, []) if s in available]
        sample_ids.extend(sids)
        race_labels.extend([race] * len(sids))

    X = clr_df[sample_ids].T.values.astype(float)   # (n_samples, n_taxa)
    y = np.array(race_labels)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"  X shape: {X.shape}")
    print(f"  Class distribution: { {r: (y==r).sum() for r in RACES} }")
    print(f"  Note: class_weight='balanced' corrects for size imbalance (Hispanic n=132)")

    # ── Cross-validated RF ────────────────────────────────────────────────────
    _banner(f"Cross-validated RF  ({CV_FOLDS}-fold stratified, n_estimators={N_ESTIMATORS})")

    rf_cv = RandomForestClassifier(
        n_estimators = N_ESTIMATORS,
        max_features = "sqrt",
        random_state = RF_RANDOM_STATE,
        n_jobs       = N_JOBS,
        class_weight = "balanced",
        oob_score    = False,   # incompatible with CV split
    )

    cv_results = cross_validate(
        rf_cv, X, y_enc,
        cv      = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RF_RANDOM_STATE),
        scoring = ["accuracy", "f1_macro", "f1_weighted"],
        return_train_score = False,
        n_jobs  = N_JOBS,
    )

    print(f"\n  CV Accuracy   : {cv_results['test_accuracy'].mean():.3f} "
          f"± {cv_results['test_accuracy'].std():.3f}")
    print(f"  CV F1 (macro) : {cv_results['test_f1_macro'].mean():.3f} "
          f"± {cv_results['test_f1_macro'].std():.3f}")
    print(f"  CV F1 (wtd)   : {cv_results['test_f1_weighted'].mean():.3f} "
          f"± {cv_results['test_f1_weighted'].std():.3f}")

    # ── Full-data fit for importances + OOB ──────────────────────────────────
    _banner("Full-data fit  (for feature importances + OOB score)")

    rf_full = RandomForestClassifier(
        n_estimators = N_ESTIMATORS,
        max_features = "sqrt",
        random_state = RF_RANDOM_STATE,
        n_jobs       = N_JOBS,
        class_weight = "balanced",
        oob_score    = True,
    )
    rf_full.fit(X, y_enc)
    print(f"  OOB accuracy: {rf_full.oob_score_:.3f}")

    # Per-class importances via permutation on OOB samples (approximation)
    importances = rf_full.feature_importances_
    ranked_idx  = np.argsort(importances)[::-1]

    # Build importance DataFrame
    imp_df = pd.DataFrame({
        "rank":         np.arange(1, len(filtered_taxa) + 1),
        "otu_id":       [filtered_otu_ids[i] for i in ranked_idx],
        "genus":        [filtered_taxa[i]    for i in ranked_idx],
        "mdi_importance": importances[ranked_idx],
    })

    # Compare top taxa against manual 96
    manual_df       = pd.read_csv(ROOT_DIR / "step3_final_core.csv")
    manual_per_race = manual_df.groupby("Race")["OTU_ID"].apply(set).to_dict()
    man_all_ids     = set().union(*manual_per_race.values())

    _banner("Performance vs Manual 96 taxa  (top-N by MDI importance)")
    print(f"\n  {'Threshold':<10} {'n_taxa':>7} {'Overlap/96':>12} "
          f"{'Precision':>10} {'Recall':>10} {'F1':>8}")
    print(f"  {'─'*10} {'─'*7} {'─'*12} {'─'*10} {'─'*10} {'─'*8}")

    for top_n in [50, 70, 96, 100, 150, 200]:
        top_ids  = set(imp_df.head(top_n)["otu_id"])
        overlap  = top_ids & man_all_ids
        p  = len(overlap) / len(top_ids)  if top_ids  else 0
        r  = len(overlap) / len(man_all_ids) if man_all_ids else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        print(f"  Top-{top_n:<6} {top_n:>7} {len(overlap):>9}/96 "
              f"{p:>10.3f} {r:>10.3f} {f1:>8.3f}")

    # ── Classification report on full data ───────────────────────────────────
    _banner("Classification report  (full-data predictions)")
    y_pred  = rf_full.predict(X)
    report  = classification_report(y_enc, y_pred,
                                    target_names=le.classes_, digits=3)
    cm      = confusion_matrix(y_enc, y_pred)
    print(report)

    # ── Save outputs ─────────────────────────────────────────────────────────
    _banner("Save outputs")
    imp_df.to_csv(ROOT_DIR / "rf_multiclass_importances.csv", index=False)

    report_lines = [
        "Multi-Class RF — Race Signature Detection",
        "=" * 50,
        f"n_estimators : {N_ESTIMATORS}",
        f"n_features   : {len(filtered_taxa)} (post pre-filter)",
        f"n_samples    : {X.shape[0]}",
        f"class_weight : balanced",
        "",
        f"CV Accuracy   : {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}",
        f"CV F1 (macro) : {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}",
        f"OOB Accuracy  : {rf_full.oob_score_:.3f}",
        "",
        "Classification Report (full data):",
        report,
        "",
        "Top-30 taxa by MDI importance:",
        imp_df.head(30).to_string(index=False),
    ]
    report_path = ROOT_DIR / "rf_multiclass_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Saved rf_multiclass_importances.csv")
    print(f"  Saved rf_multiclass_report.txt")

    # ── Figures ───────────────────────────────────────────────────────────────
    _banner("Generating figures")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: top-30 importance bar chart
    ax = axes[0]
    top30 = imp_df.head(30).iloc[::-1]
    colors = ["#E74C3C" if r in man_all_ids else "#3498DB"
              for r in top30["otu_id"]]
    ax.barh(top30["genus"], top30["mdi_importance"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("MDI Importance", fontsize=10)
    ax.set_title("Top-30 Taxa — Multi-Class RF\n"
                 "Red = in Manual 96  |  Blue = RF-only",
                 fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: confusion matrix
    ax2 = axes[1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=le.classes_)
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title(f"Confusion Matrix (full-data)\n"
                  f"OOB Accuracy = {rf_full.oob_score_:.3f}", fontsize=11)

    fig.suptitle(
        f"Multi-Class RF  |  {len(filtered_taxa)} taxa × {X.shape[0]} samples  |  "
        f"CV F1 (macro) = {cv_results['test_f1_macro'].mean():.3f}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig_path = FIGURES_DIR / "fig_rf_multiclass.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path.name}")

    print(f"\n✓  Done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
