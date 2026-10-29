"""
rf_per_race.py
──────────────
Runs the same RF + RFECV pipeline as main.py, but aggregated per race
across ALL bioprojects instead of per (bioproject × race).

For each race:
  Positive  — all samples of that race (all bioprojects)
  Negative  — all samples of every other race (all bioprojects)

This mirrors the AE setup exactly, enabling a direct comparison.

Outputs
───────
  rf_per_race.csv           — per-race × taxon results (importance + selected)
  rf_per_race_summary.csv   — precision / recall / F1 vs manual 96 taxa
                              at multiple score thresholds

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.rf_per_race
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    METADATA_FILE, ABUNDANCE_FILE, ROOT_DIR,
    RF_N_ESTIMATORS, RF_MAX_FEATURES, RF_RANDOM_STATE, RF_N_JOBS,
    RF_TOP_N_FOR_RFECV, RFECV_STEP, RFECV_CV_FOLDS, RFECV_SCORING,
    MAX_SAMPLES_PER_CLASS,
)
from pipeline.data_loader    import load_metadata, group_samples, load_abundance, filter_rare_taxa
from pipeline.transforms     import clr_transform
from pipeline.feature_selector import rank_by_random_forest, run_rfecv

RACES = ["African", "Asian", "Caucasian", "Hispanic"]


def _banner(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}", flush=True)


def compute_metrics(predicted_ids: set, manual_ids: set) -> dict:
    overlap   = predicted_ids & manual_ids
    precision = len(overlap) / len(predicted_ids) if predicted_ids else 0.0
    recall    = len(overlap) / len(manual_ids)    if manual_ids    else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_predicted": len(predicted_ids),
        "overlap":     len(overlap),
        "precision":   round(precision, 3),
        "recall":      round(recall,    3),
        "f1":          round(f1,        3),
    }


def main():
    t0 = time.time()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       RF PER-RACE CORE MICROBIOME PIPELINE               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Load data ─────────────────────────────────────────────────────────────
    _banner("Load data")
    meta   = load_metadata(METADATA_FILE)
    groups = group_samples(meta)

    # Aggregate to per-race
    race_groups: dict[str, list[str]] = {}
    for (bp, race), sids in groups.items():
        race_groups.setdefault(race, []).extend(sids)
    print(f"  Race groups: { {r: len(s) for r, s in race_groups.items()} }")

    taxa_names, otu_ids_list, raw_df = load_abundance(ABUNDANCE_FILE)
    known = [s for s in raw_df.columns if s in meta]
    raw_df = raw_df[known]

    filtered_df      = filter_rare_taxa(raw_df)
    filtered_taxa    = list(filtered_df.index)
    filtered_otu_ids = [otu_ids_list[taxa_names.index(t)] for t in filtered_taxa]
    print(f"  After pre-filter: {len(filtered_taxa)} taxa")

    # ── CLR transform ─────────────────────────────────────────────────────────
    _banner("CLR transform")
    clr_df = clr_transform(filtered_df)
    print(f"  CLR matrix: {clr_df.shape[0]} taxa × {clr_df.shape[1]} samples")

    # ── Manual reference ──────────────────────────────────────────────────────
    manual_df       = pd.read_csv(ROOT_DIR / "step3_final_core.csv")
    manual_per_race = manual_df.groupby("Race")["OTU_ID"].apply(set).to_dict()
    man_all_ids     = set().union(*manual_per_race.values())
    print(f"\n  Manual reference: {len(man_all_ids)} unique taxa across all races")

    available = set(clr_df.columns)
    rng       = np.random.default_rng(RF_RANDOM_STATE)

    all_rows    = []
    summary_rows = []

    # ── Per-race RF + RFECV ───────────────────────────────────────────────────
    _banner("RF + RFECV per race  (positive = race, negative = all other races)")

    for race in RACES:
        print(f"\n  ══ Race: {race} ══")

        pos_ids = [s for s in race_groups.get(race, []) if s in available]
        neg_ids = [
            s for r, sids in race_groups.items()
            if r != race
            for s in sids
            if s in available
        ]

        if len(pos_ids) < 10:
            print(f"  SKIP: only {len(pos_ids)} positive samples")
            continue

        # Subsample large classes
        if MAX_SAMPLES_PER_CLASS and len(pos_ids) > MAX_SAMPLES_PER_CLASS:
            pos_ids = list(rng.choice(pos_ids, MAX_SAMPLES_PER_CLASS, replace=False))
            print(f"  Subsampled positives → {MAX_SAMPLES_PER_CLASS}")
        if MAX_SAMPLES_PER_CLASS and len(neg_ids) > MAX_SAMPLES_PER_CLASS:
            neg_ids = list(rng.choice(neg_ids, MAX_SAMPLES_PER_CLASS, replace=False))
            print(f"  Subsampled negatives → {MAX_SAMPLES_PER_CLASS}")

        print(f"  Pos={len(pos_ids):,}  Neg={len(neg_ids):,}  Features={len(filtered_taxa)}")

        all_ids = pos_ids + neg_ids
        X = clr_df[all_ids].T.values.astype(float)
        y = np.array([1] * len(pos_ids) + [0] * len(neg_ids))

        # Stage 1: RF
        print("  Stage 1: Random Forest … ", end="", flush=True)
        X_top, top_names, all_importances, top_idx = rank_by_random_forest(
            X, y, filtered_taxa, top_n=RF_TOP_N_FOR_RFECV
        )
        print(f"top {len(top_names)} features selected")

        # Stage 2: RFECV
        print("  Stage 2: RFECV … ", end="", flush=True)
        selected_mask, rfecv_obj = run_rfecv(X_top, y)
        n_selected = selected_mask.sum()
        best_score = rfecv_obj.cv_results_["mean_test_score"].max()
        print(f"{n_selected} features retained  (best CV {RFECV_SCORING}={best_score:.3f})")

        # Assemble results
        taxon_to_otu = dict(zip(filtered_taxa, filtered_otu_ids))
        taxon_to_imp = dict(zip(filtered_taxa, all_importances))

        race_results = []
        for local_i, taxon in enumerate(top_names):
            race_results.append({
                "race":           race,
                "otu_id":         taxon_to_otu.get(taxon, -1),
                "genus":          taxon,
                "rf_importance":  float(taxon_to_imp.get(taxon, 0.0)),
                "rfecv_selected": bool(selected_mask[local_i]),
            })
        race_results.sort(key=lambda x: -x["rf_importance"])
        for rank, entry in enumerate(race_results, 1):
            entry["rank"] = rank
        all_rows.extend(race_results)

        # ── Metrics at multiple thresholds ────────────────────────────────────
        manual_ids = manual_per_race.get(race, set())
        imp_series = pd.Series(
            {r["otu_id"]: r["rf_importance"] for r in race_results}
        ).sort_values(ascending=False)

        rfecv_ids = {r["otu_id"] for r in race_results if r["rfecv_selected"]}
        top50_ids = set(imp_series.head(50).index)
        top70_ids = set(imp_series.head(70).index)
        top96_ids = set(imp_series.head(96).index)
        top100_ids= set(imp_series.head(100).index)

        for label, pred_ids in [
            ("RFECV",   rfecv_ids),
            ("Top-50",  top50_ids),
            ("Top-70",  top70_ids),
            ("Top-96",  top96_ids),
            ("Top-100", top100_ids),
        ]:
            m = compute_metrics(pred_ids, manual_ids)
            summary_rows.append({"Race": race, "Threshold": label, **m})
            print(f"    {label:8s}  n={m['n_predicted']:4d}  "
                  f"ov={m['overlap']}/manual  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _banner("Save outputs")
    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(ROOT_DIR / "rf_per_race.csv",         index=False)
    summary_df.to_csv(ROOT_DIR / "rf_per_race_summary.csv", index=False)
    print(f"  Saved rf_per_race.csv ({len(results_df)} rows)")
    print(f"  Saved rf_per_race_summary.csv ({len(summary_df)} rows)")

    # ── Cross-race union vs Manual 96 ─────────────────────────────────────────
    _banner("Cross-race union vs Manual 96 taxa")
    print(f"\n  {'Threshold':<10} {'AE Unique':>10} {'Overlap/96':>12} "
          f"{'Precision':>10} {'Recall':>10} {'F1':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*8}")

    imp_per_race = {}
    for race in RACES:
        sub = results_df[results_df["race"] == race]
        if sub.empty:
            continue
        imp_per_race[race] = sub.set_index("otu_id")["rf_importance"]

    for label, top_n_val in [("RFECV", None), ("Top-50", 50),
                              ("Top-70", 70), ("Top-96", 96), ("Top-100", 100)]:
        union_ids = set()
        for race, imp in imp_per_race.items():
            if label == "RFECV":
                sub = results_df[results_df["race"] == race]
                union_ids |= set(sub[sub["rfecv_selected"]]["otu_id"])
            else:
                union_ids |= set(imp.sort_values(ascending=False).head(top_n_val).index)
        m = compute_metrics(union_ids, man_all_ids)
        print(f"  {label:<10} {m['n_predicted']:>10} {m['overlap']:>9}/96 "
              f"{m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}")

    print(f"\n✓  Done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
