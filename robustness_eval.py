"""
robustness_eval.py
──────────────────
Runs the Weighted MSE AE pipeline N_RUNS times with different random seeds
to evaluate the stability of the 95%-mass core microbiome selection.

Per-run metrics vs the 96-taxon manual reference:
  - Precision, Recall, F1   (95%-mass union, 4 races combined)
  - AP  (Average Precision of CoreScore ranking over all 4,680 taxa)
  - Macro F1  (mean per-race F1, then averaged across runs)

Output
──────
  robustness_results.csv   — per-run metrics (appended after each run)
  robustness_summary.txt   — mean ± std summary table

Resume: already-completed seeds are detected from robustness_results.csv and
skipped, so the script can be interrupted and restarted safely.

Run from Datasets/ with the venv active:
    python -m pipeline.robustness_eval
"""

import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from pipeline.config      import METADATA_FILE, ABUNDANCE_FILE, ROOT_DIR
from pipeline.data_loader import load_metadata, group_samples, load_abundance
from pipeline.autoencoder import (
    multiplicative_replacement,
    clr_transform_group,
    get_feature_importance,
)
from pipeline.autoencoder_v2 import train_autoencoder_v2

# ── Configuration ─────────────────────────────────────────────────────────────
N_RUNS          = 100
AE_HIDDEN_DIMS  = [512, 128]
AE_BOTTLENECK   = 64
AE_DROPOUT      = 0.15
AE_EPOCHS       = 150
AE_LR           = 1e-3
AE_WEIGHT_DECAY = 1e-4
AE_PATIENCE     = 20
MASS_THRESHOLD  = 0.95

RACES   = ["African", "Asian", "Caucasian", "Hispanic"]
OUT_CSV = ROOT_DIR / "robustness_results.csv"
OUT_TXT = ROOT_DIR / "robustness_summary.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess(sample_ids, raw_df, n_features):
    valid   = [s for s in sample_ids if s in raw_df.columns]
    X_raw   = raw_df[valid].T.values.astype(float)
    totals  = X_raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    X_prop  = X_raw / totals
    X_rep   = multiplicative_replacement(X_prop, delta=0.65 / n_features)
    X_clr   = clr_transform_group(X_rep)
    means   = X_clr.mean(axis=0, keepdims=True)
    stds    = X_clr.std(axis=0,  keepdims=True) + 1e-8
    return X_clr, (X_clr - means) / stds, valid


def compute_prevalence_weights(raw_df):
    presence   = (raw_df.values > 0).astype(float)
    prevalence = presence.mean(axis=1)
    return (prevalence / (prevalence.mean() + 1e-12)).astype(np.float32)


def select_95mass(scores: np.ndarray, otu_ids_arr: np.ndarray) -> set:
    """
    Select OTU IDs whose CoreScores jointly account for >= MASS_THRESHOLD
    of the total positive score (greedy top-down).
    """
    pos_mask = scores > 0
    if not pos_mask.any():
        return set()
    pos_idx    = np.where(pos_mask)[0]
    sorted_rel = np.argsort(scores[pos_idx])[::-1]
    sorted_idx = pos_idx[sorted_rel]
    cumsum     = np.cumsum(scores[sorted_idx])
    n_sel      = int(np.searchsorted(cumsum, MASS_THRESHOLD * cumsum[-1])) + 1
    return {otu_ids_arr[i] for i in sorted_idx[:n_sel]}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate_run(
    race_scores:     dict,        # race -> (n_features,) CoreScore array
    otu_ids_arr:     np.ndarray,  # (n_features,) 1-based OTU IDs
    manual_96:       set,
    manual_per_race: dict,        # race -> set of OTU IDs
) -> dict:
    """
    Compute all five metrics for one run:
      - Precision, Recall, F1  (overall union vs manual 96)
      - AP  (Average Precision of CoreScore ranking, all 4,680 taxa)
      - Macro F1  (mean of per-race F1 scores)
    """
    # ── 95%-mass per race → union ─────────────────────────────────────────────
    union_sel    = set()
    per_race_sel = {}
    for race, scores in race_scores.items():
        sel = select_95mass(scores, otu_ids_arr)
        per_race_sel[race] = sel
        union_sel |= sel

    tp = len(union_sel & manual_96)
    fp = len(union_sel - manual_96)
    fn = len(manual_96 - union_sel)
    overall_p, overall_r, overall_f1 = _prf(tp, fp, fn)

    # ── Average Precision: ranking all 4,680 taxa by max CoreScore ────────────
    max_scores = np.zeros(len(otu_ids_arr))
    for scores in race_scores.values():
        max_scores = np.maximum(max_scores, scores)

    y_true = np.isin(otu_ids_arr, list(manual_96)).astype(int)
    ap     = float(average_precision_score(y_true, max_scores))

    # ── Macro F1: F1 per race, then averaged ─────────────────────────────────
    race_f1s = []
    for race in RACES:
        sel  = per_race_sel.get(race, set())
        man  = manual_per_race.get(race, set())
        if not man:
            continue
        tp_r = len(sel & man)
        fp_r = len(sel - man)
        fn_r = len(man - sel)
        _, _, f = _prf(tp_r, fp_r, fn_r)
        race_f1s.append(f)

    return {
        "n_selected": len(union_sel),
        "precision":  overall_p,
        "recall":     overall_r,
        "f1":         overall_f1,
        "ap":         ap,
        "macro_f1":   float(np.mean(race_f1s)) if race_f1s else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()
    device   = _device()
    print(f"\n  Device : {device}")
    print(f"  N_RUNS : {N_RUNS}")

    # ── Load data once ────────────────────────────────────────────────────────
    print("\n  Loading metadata & abundance …", flush=True)
    meta = load_metadata(METADATA_FILE)
    groups = group_samples(meta)
    race_groups: dict = {}
    for (_, race), sids in groups.items():
        race_groups.setdefault(race, []).extend(sids)
    print(f"  Race sizes: { {r: len(s) for r, s in race_groups.items()} }")

    taxa_names, otu_ids, raw_df = load_abundance(ABUNDANCE_FILE)
    n_features   = len(taxa_names)
    otu_ids_arr  = np.array(otu_ids)

    feature_weights = compute_prevalence_weights(raw_df)
    print(f"  Features: {n_features}  |  Prevalence weights computed.")

    # ── Manual reference ──────────────────────────────────────────────────────
    step3 = pd.read_csv(ROOT_DIR / "step3_final_core.csv")
    manual_96       = set(step3["OTU_ID"].unique())
    manual_per_race = {
        race: set(step3.loc[step3["Race"] == race, "OTU_ID"])
        for race in RACES
    }
    print(f"  Manual core: {len(manual_96)} OTUs")
    for r, s in manual_per_race.items():
        print(f"    {r}: {len(s)} OTUs")

    # ── Resume detection ──────────────────────────────────────────────────────
    completed = set()
    if OUT_CSV.exists():
        prev      = pd.read_csv(OUT_CSV)
        completed = set(prev["seed"].tolist())
        print(f"\n  Resuming — {len(completed)}/{N_RUNS} runs already done.")

    seeds_todo = [s for s in range(N_RUNS) if s not in completed]
    if not seeds_todo:
        print("  All runs complete — regenerating summary only.")
    else:
        print(f"  Runs to do: {len(seeds_todo)}")

    file_existed  = OUT_CSV.exists()   # controls CSV header
    session_start = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for run_i, seed in enumerate(seeds_todo):
        t0 = time.time()
        print(f"\n  ── Run {seed+1:3d}/{N_RUNS}  (seed={seed}) "
              f"─────────────────────────", flush=True)
        set_seed(seed)

        # Train one Weighted MSE AE per race, extract CoreScores
        race_scores = {}
        for race in RACES:
            X_clr, X_sc, valid_ids = preprocess(
                race_groups.get(race, []), raw_df, n_features)
            model, losses = train_autoencoder_v2(
                X_clr           = X_sc,
                n_features      = n_features,
                hidden_dims     = AE_HIDDEN_DIMS,
                bottleneck      = AE_BOTTLENECK,
                dropout         = AE_DROPOUT,
                epochs          = AE_EPOCHS,
                lr              = AE_LR,
                weight_decay    = AE_WEIGHT_DECAY,
                patience        = AE_PATIENCE,
                device_str      = str(device),
                verbose         = False,
                l1_lambda       = 0.0,
                feature_weights = feature_weights,
            )
            scores, _ = get_feature_importance(
                model, X_sc, "clr_weighted_mse", device, X_clr_orig=X_clr)
            race_scores[race] = scores
            print(f"    {race:<12} {len(valid_ids):>4} samples  "
                  f"{len(losses):>3} epochs  "
                  f"loss={losses[-1]:.5f}  "
                  f"top={scores.max():.2f}",
                  flush=True)

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics              = evaluate_run(race_scores, otu_ids_arr,
                                            manual_96, manual_per_race)
        elapsed              = time.time() - t0
        metrics["seed"]      = seed
        metrics["elapsed_s"] = round(elapsed, 1)

        print(f"  → P={metrics['precision']:.3f}  "
              f"R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  "
              f"AP={metrics['ap']:.3f}  "
              f"MacroF1={metrics['macro_f1']:.3f}  "
              f"n_sel={metrics['n_selected']}  "
              f"({elapsed:.0f}s)", flush=True)

        # ── Checkpoint ────────────────────────────────────────────────────────
        write_header = (not file_existed) and (run_i == 0)
        pd.DataFrame([metrics]).to_csv(
            OUT_CSV, mode="a", header=write_header, index=False)
        file_existed = True   # header written at most once per fresh start

        # ── ETA ───────────────────────────────────────────────────────────────
        done_now  = run_i + 1
        avg_s     = (time.time() - session_start) / done_now
        remaining = len(seeds_todo) - done_now
        print(f"  ETA: ~{avg_s * remaining / 60:.0f} min remaining",
              flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    results = pd.read_csv(OUT_CSV)
    n_done  = len(results)
    sep     = "─" * 64

    header_line = (f"  {'Metric':<14} {'Mean':>9} {'± Std':>9} "
                   f"{'Min':>9} {'Max':>9}")
    metric_rows = []
    for col, label in [
        ("n_selected", "n_selected"),
        ("precision",  "Precision"),
        ("recall",     "Recall"),
        ("f1",         "F1"),
        ("ap",         "AP"),
        ("macro_f1",   "Macro F1"),
    ]:
        m, s   = results[col].mean(), results[col].std()
        lo, hi = results[col].min(),  results[col].max()
        metric_rows.append(
            f"  {label:<14} {m:>9.4f} {s:>9.4f} {lo:>9.4f} {hi:>9.4f}")

    summary_lines = [
        "",
        sep,
        f"  Robustness Summary  (N={n_done} runs, seeds 0–{n_done-1})",
        sep,
        header_line,
        sep,
        *metric_rows,
        sep,
    ]
    summary = "\n".join(summary_lines)
    print(summary)

    txt_body = (
        "Robustness Evaluation — Weighted MSE AE + 95%-mass threshold\n"
        f"N={n_done} independent runs, seeds 0–{n_done-1}\n"
        f"Architecture : 4680 → 512 → 128 → 64 → 128 → 512 → 4680\n"
        f"AE params    : epochs={AE_EPOCHS} max, patience={AE_PATIENCE}, "
        f"lr={AE_LR}, wd={AE_WEIGHT_DECAY}\n"
        f"Reference    : 96-taxon manual core (step3_final_core.csv)\n"
        f"\nMetrics\n"
        f"  Precision / Recall / F1 : 95%-mass union vs manual 96 (overall)\n"
        f"  AP                      : Average Precision of max-CoreScore ranking "
        f"over all 4,680 taxa\n"
        f"  Macro F1                : mean per-race F1 (4 races)\n"
        f"\n{summary}\n"
    )
    with open(OUT_TXT, "w") as fh:
        fh.write(txt_body)
    print(f"\n  Saved {OUT_TXT.name}")
    print(f"  Saved {OUT_CSV.name}  ({n_done} rows)")
    print(f"\n✓  Total wall time: {(time.time() - t_global) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
