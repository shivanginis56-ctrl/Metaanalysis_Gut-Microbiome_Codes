"""
feature_selector.py
────────────────────
ML-based feature selection to identify the "core microbiome" of a group.

Two-stage approach per (bioproject, race) group:
────────────────────────────────────────────────
Stage 1 – Random Forest ranking
    A binary RF classifier distinguishes the target group (label = 1) from a
    carefully chosen negative set (label = 0).  MDI feature importances rank
    all available taxa.  The top RF_TOP_N_FOR_RFECV taxa are forwarded to
    Stage 2.

Stage 2 – RFECV (Recursive Feature Elimination with Cross-Validation)
    Starting from the top-ranked taxa, RFECV iteratively removes the least
    important features while monitoring cross-validated ROC-AUC.  The feature
    subset that maximises CV performance is returned as the "ML core".

Negative set strategy
─────────────────────
- Multi-race bioproject  →  other races within the SAME bioproject.
  (Finds taxa that distinguish this racial group within the same cohort.)
- Single-race bioproject →  other bioprojects containing the SAME race.
  (Finds taxa that characterise this cohort vs. other same-race cohorts,
   capturing population/geography-specific signatures.)

Output per group
────────────────
A list of dicts, each representing one selected taxon:
  {otu_id, genus, rf_importance, rfecv_selected, rank}
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from pipeline.config import (
    RF_N_ESTIMATORS, RF_MAX_FEATURES, RF_RANDOM_STATE, RF_N_JOBS,
    RF_TOP_N_FOR_RFECV,
    RFECV_STEP, RFECV_CV_FOLDS, RFECV_SCORING,
    MIN_POSITIVE_SAMPLES, SINGLE_RACE_NEGATIVE_STRATEGY,
    MAX_SAMPLES_PER_CLASS,
)


# ── Negative set construction ─────────────────────────────────────────────────

def build_negative_set(
    target_bp: str,
    target_race: str,
    groups: dict[tuple[str, str], list[str]],
    clr_df: pd.DataFrame,
) -> list[str]:
    """
    Select sample IDs for the negative class of the binary classification.

    Logic:
      - If the bioproject contains ≥2 races, use other races from that project.
      - Otherwise (single-race project), use same race from other bioprojects,
        per SINGLE_RACE_NEGATIVE_STRATEGY in config.

    Parameters
    ----------
    target_bp   : bioproject of the target group.
    target_race : race label of the target group.
    groups      : full {(bioproject, race): [sample_id, ...]} mapping.
    clr_df      : CLR-transformed matrix (taxa × samples) — used only to verify
                  which sample IDs are actually available.

    Returns
    -------
    list[str]  Sample IDs for the negative class (already filtered to those
               present in clr_df).
    """
    available = set(clr_df.columns)

    # All races in the target bioproject
    races_in_bp = [race for (bp, race) in groups if bp == target_bp]

    if len(races_in_bp) > 1:
        # Multi-race bioproject: use other races in the same project
        neg_ids = [
            sid
            for (bp, race), sids in groups.items()
            if bp == target_bp and race != target_race
            for sid in sids
            if sid in available
        ]
    else:
        # Single-race bioproject: use same race from other bioprojects
        neg_ids = [
            sid
            for (bp, race), sids in groups.items()
            if bp != target_bp and race == target_race
            for sid in sids
            if sid in available
        ]

    return neg_ids


# ── Stage 1: Random Forest importance ranking ─────────────────────────────────

def rank_by_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_n: int = RF_TOP_N_FOR_RFECV,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Train a Random Forest on (X, y) and return the top_n features by
    Mean Decrease in Impurity (MDI) importance.

    Parameters
    ----------
    X            : (n_samples, n_features) CLR-transformed feature matrix.
    y            : (n_samples,) binary labels (1 = target group, 0 = negative).
    feature_names: taxon names aligned with columns of X.
    top_n        : number of top features to forward to RFECV.

    Returns
    -------
    X_top         : (n_samples, top_n) sub-matrix with most important features.
    top_names     : list of taxon names for the top_n features.
    importances   : full array of MDI importances (length = n_features).
    """
    rf = RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        max_features = RF_MAX_FEATURES,
        random_state = RF_RANDOM_STATE,
        n_jobs       = RF_N_JOBS,
        class_weight = "balanced",   # compensate for class imbalance
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:top_n]

    X_top     = X[:, top_idx]
    top_names = [feature_names[i] for i in top_idx]

    return X_top, top_names, importances, top_idx


# ── Stage 2: RFECV ────────────────────────────────────────────────────────────

def run_rfecv(
    X_top: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, RFECV]:
    """
    Apply Recursive Feature Elimination with CV to select the optimal
    feature subset from the RF-ranked top features.

    Parameters
    ----------
    X_top : (n_samples, top_n) matrix from Stage 1.
    y     : binary labels.

    Returns
    -------
    selected_mask : boolean array (length = top_n), True = retained by RFECV.
    rfecv         : fitted RFECV object (contains cv_results_ for inspection).
    """
    base_rf = RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        max_features = RF_MAX_FEATURES,
        random_state = RF_RANDOM_STATE,
        n_jobs       = RF_N_JOBS,
        class_weight = "balanced",
    )

    cv = StratifiedKFold(n_splits=RFECV_CV_FOLDS, shuffle=True,
                         random_state=RF_RANDOM_STATE)

    rfecv = RFECV(
        estimator  = base_rf,
        step       = RFECV_STEP,
        cv         = cv,
        scoring    = RFECV_SCORING,
        min_features_to_select = 1,
        n_jobs     = RF_N_JOBS,
    )
    rfecv.fit(X_top, y)

    return rfecv.support_, rfecv


# ── Main entry point ──────────────────────────────────────────────────────────

def select_core_taxa(
    target_bp: str,
    target_race: str,
    groups: dict,
    clr_df: pd.DataFrame,
    taxa_names: list[str],
    otu_ids: list[int],
) -> list[dict] | None:
    """
    Full two-stage ML selection for one (bioproject, race) group.

    Parameters
    ----------
    target_bp   : bioproject identifier.
    target_race : race label.
    groups      : {(bp, race): [sample_id, ...]} for all groups.
    clr_df      : CLR-transformed DataFrame (taxa × all samples).
    taxa_names  : taxon names aligned with clr_df.index.
    otu_ids     : 1-based OTU IDs aligned with taxa_names.

    Returns
    -------
    list[dict] or None
        Each dict: {otu_id, genus, rf_importance, rfecv_selected, rank}
        Returns None if the group is too small to run ML reliably.
    """
    available   = set(clr_df.columns)
    pos_ids     = [s for s in groups[(target_bp, target_race)] if s in available]
    neg_ids     = build_negative_set(target_bp, target_race, groups, clr_df)

    # Guard: skip groups with too few positive samples
    if len(pos_ids) < MIN_POSITIVE_SAMPLES:
        print(f"    SKIP: only {len(pos_ids)} positive samples "
              f"(need ≥ {MIN_POSITIVE_SAMPLES})")
        return None

    if len(neg_ids) == 0:
        print("    SKIP: no negative samples found for comparison")
        return None

    # Optional: subsample large classes to keep runtime tractable
    rng = np.random.default_rng(RF_RANDOM_STATE)
    if MAX_SAMPLES_PER_CLASS and len(pos_ids) > MAX_SAMPLES_PER_CLASS:
        pos_ids = list(rng.choice(pos_ids, MAX_SAMPLES_PER_CLASS, replace=False))
        print(f"    Subsampled positives → {MAX_SAMPLES_PER_CLASS}", flush=True)
    if MAX_SAMPLES_PER_CLASS and len(neg_ids) > MAX_SAMPLES_PER_CLASS:
        neg_ids = list(rng.choice(neg_ids, MAX_SAMPLES_PER_CLASS, replace=False))
        print(f"    Subsampled negatives → {MAX_SAMPLES_PER_CLASS}", flush=True)

    # Build feature matrix: samples as rows, taxa as columns
    # clr_df is (taxa × samples); transpose to (samples × taxa)
    all_ids = pos_ids + neg_ids
    X_df    = clr_df[all_ids].T                         # (n_samples, n_taxa)
    X       = X_df.values.astype(float)
    y       = np.array([1] * len(pos_ids) + [0] * len(neg_ids))

    feature_names = list(clr_df.index)                  # taxa in same order as X cols

    print(f"    Pos={len(pos_ids):,}  Neg={len(neg_ids):,}  "
          f"Features={X.shape[1]:,}", flush=True)

    # ── Stage 1: RF importance ranking ───────────────────────────────────────
    print("    Stage 1: Random Forest … ", end="", flush=True)
    X_top, top_names, all_importances, top_idx = rank_by_random_forest(
        X, y, feature_names, top_n=RF_TOP_N_FOR_RFECV
    )
    print(f"top {len(top_names)} features selected", flush=True)

    # ── Stage 2: RFECV ───────────────────────────────────────────────────────
    print("    Stage 2: RFECV … ", end="", flush=True)
    selected_mask, rfecv_obj = run_rfecv(X_top, y)
    n_selected = selected_mask.sum()
    # best_score is the maximum across all CV steps (each step tests a different
    # feature-count subset; n_features_ is the count, NOT an index into the array)
    best_score = rfecv_obj.cv_results_["mean_test_score"].max()
    print(f"{n_selected} features retained  "
          f"(best CV {RFECV_SCORING}={best_score:.3f})", flush=True)

    # ── Assemble results ──────────────────────────────────────────────────────
    # Map top_names back to global OTU IDs and global importances
    taxon_to_otu   = dict(zip(taxa_names, otu_ids))
    taxon_to_imp   = dict(zip(feature_names, all_importances))

    results = []
    for local_i, taxon in enumerate(top_names):
        results.append({
            "otu_id":         taxon_to_otu.get(taxon, -1),
            "genus":          taxon,
            "rf_importance":  float(taxon_to_imp.get(taxon, 0.0)),
            "rfecv_selected": bool(selected_mask[local_i]),
        })

    # Sort by RF importance descending; assign rank
    results.sort(key=lambda x: -x["rf_importance"])
    for rank, entry in enumerate(results, 1):
        entry["rank"] = rank

    return results
