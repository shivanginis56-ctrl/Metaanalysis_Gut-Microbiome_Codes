"""
taxa_stability_eval.py
──────────────────────
Re-runs the Weighted MSE AE for N_RUNS seeds and records WHICH taxa are
selected in each run (95%-mass threshold, per race + union).

Outputs
───────
  taxa_selection_frequency.csv
      otu_id, genus, n_runs_selected, freq, in_manual,
      in_manual_african, in_manual_asian, in_manual_caucasian, in_manual_hispanic
      → one row per taxon that was selected at least once

  taxa_stability_summary.txt
      Breakdown: always-selected / majority / occasional / never

Run from Datasets/ with venv active:
    python -m pipeline.taxa_stability_eval
"""

import time, random
import numpy as np
import pandas as pd
import torch

from pipeline.config      import METADATA_FILE, ABUNDANCE_FILE, ROOT_DIR
from pipeline.data_loader import load_metadata, group_samples, load_abundance
from pipeline.autoencoder import (
    multiplicative_replacement, clr_transform_group, get_feature_importance)
from pipeline.autoencoder_v2 import train_autoencoder_v2

# ── Config (identical to robustness_eval.py) ──────────────────────────────────
N_RUNS          = 100
AE_HIDDEN_DIMS  = [512, 128]
AE_BOTTLENECK   = 64
AE_DROPOUT      = 0.15
AE_EPOCHS       = 150
AE_LR           = 1e-3
AE_WEIGHT_DECAY = 1e-4
AE_PATIENCE     = 20
MASS_THRESHOLD  = 0.95
RACES           = ["African", "Asian", "Caucasian", "Hispanic"]

OUT_FREQ  = ROOT_DIR / "taxa_selection_frequency.csv"
OUT_TXT   = ROOT_DIR / "taxa_stability_summary.txt"
CKPT_CSV  = ROOT_DIR / "taxa_stability_checkpoint.csv"   # seed × otu_id flags


# ── Helpers ───────────────────────────────────────────────────────────────────

def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def preprocess(sample_ids, raw_df, n_features):
    valid  = [s for s in sample_ids if s in raw_df.columns]
    X_raw  = raw_df[valid].T.values.astype(float)
    totals = X_raw.sum(axis=1, keepdims=True); totals[totals==0] = 1.0
    X_prop = X_raw / totals
    X_rep  = multiplicative_replacement(X_prop, delta=0.65/n_features)
    X_clr  = clr_transform_group(X_rep)
    means  = X_clr.mean(axis=0, keepdims=True)
    stds   = X_clr.std(axis=0,  keepdims=True) + 1e-8
    return X_clr, (X_clr - means)/stds, valid

def compute_prevalence_weights(raw_df):
    p = (raw_df.values > 0).astype(float).mean(axis=1)
    return (p / (p.mean() + 1e-12)).astype(np.float32)

def select_95mass(scores, otu_ids_arr):
    pos = scores > 0
    if not pos.any(): return set()
    idx   = np.where(pos)[0]
    order = np.argsort(scores[idx])[::-1]
    sidx  = idx[order]
    cs    = np.cumsum(scores[sidx])
    n     = int(np.searchsorted(cs, MASS_THRESHOLD * cs[-1])) + 1
    return {otu_ids_arr[i] for i in sidx[:n]}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0     = time.time()
    device = _device()
    print(f"\n  Device: {device}  |  N_RUNS: {N_RUNS}")

    # ── Load data once ────────────────────────────────────────────────────────
    print("\n  Loading data …", flush=True)
    meta = load_metadata(METADATA_FILE)
    groups = group_samples(meta)
    race_groups = {}
    for (_, race), sids in groups.items():
        race_groups.setdefault(race, []).extend(sids)

    taxa_names, otu_ids, raw_df = load_abundance(ABUNDANCE_FILE)
    n_features  = len(taxa_names)
    otu_ids_arr = np.array(otu_ids)
    taxa_arr    = np.array(taxa_names)

    feature_weights = compute_prevalence_weights(raw_df)

    # ── Manual reference ──────────────────────────────────────────────────────
    step3 = pd.read_csv(ROOT_DIR / "step3_final_core.csv")
    manual_96       = set(step3["OTU_ID"].unique())
    manual_per_race = {r: set(step3.loc[step3["Race"]==r, "OTU_ID"]) for r in RACES}
    print(f"  Manual core: {len(manual_96)} OTUs")

    # ── Resume ────────────────────────────────────────────────────────────────
    if CKPT_CSV.exists():
        ckpt_df    = pd.read_csv(CKPT_CSV)
        done_seeds = set(ckpt_df["seed"].unique())
        print(f"  Resuming — {len(done_seeds)}/{N_RUNS} done")
    else:
        ckpt_df    = pd.DataFrame()
        done_seeds = set()

    seeds_todo   = [s for s in range(N_RUNS) if s not in done_seeds]
    file_existed = CKPT_CSV.exists()
    sess_start   = time.time()

    # ── Loop ─────────────────────────────────────────────────────────────────
    for run_i, seed in enumerate(seeds_todo):
        t_run = time.time()
        print(f"\n  ── Run {seed+1:3d}/{N_RUNS}  (seed={seed})", flush=True)
        set_seed(seed)

        race_scores = {}
        for race in RACES:
            X_clr, X_sc, _ = preprocess(race_groups.get(race,[]), raw_df, n_features)
            model, losses = train_autoencoder_v2(
                X_clr=X_sc, n_features=n_features, hidden_dims=AE_HIDDEN_DIMS,
                bottleneck=AE_BOTTLENECK, dropout=AE_DROPOUT, epochs=AE_EPOCHS,
                lr=AE_LR, weight_decay=AE_WEIGHT_DECAY, patience=AE_PATIENCE,
                device_str=str(device), verbose=False,
                l1_lambda=0.0, feature_weights=feature_weights)
            scores, _ = get_feature_importance(
                model, X_sc, "clr_weighted_mse", device, X_clr_orig=X_clr)
            race_scores[race] = scores

        # Union selection for this run
        union_sel = set()
        for race, scores in race_scores.items():
            union_sel |= select_95mass(scores, otu_ids_arr)

        elapsed = time.time() - t_run
        overlap = len(union_sel & manual_96)
        print(f"  → n_sel={len(union_sel)}  manual_overlap={overlap}/96  ({elapsed:.0f}s)",
              flush=True)

        # Save selected OTU IDs for this seed
        rows = [{"seed": seed, "otu_id": oid} for oid in union_sel]
        chunk = pd.DataFrame(rows)
        write_header = (not file_existed) and (run_i == 0)
        chunk.to_csv(CKPT_CSV, mode="a", header=write_header, index=False)
        file_existed = True

        done_now = run_i + 1
        avg_s    = (time.time() - sess_start) / done_now
        print(f"  ETA: ~{avg_s*(len(seeds_todo)-done_now)/60:.0f} min", flush=True)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n  Aggregating …", flush=True)
    ckpt_df = pd.read_csv(CKPT_CSV)
    n_done  = ckpt_df["seed"].nunique()

    freq_df = (ckpt_df.groupby("otu_id")["seed"]
                       .count()
                       .reset_index()
                       .rename(columns={"seed": "n_runs_selected"}))
    freq_df["freq"] = freq_df["n_runs_selected"] / n_done

    # Attach genus name (otu_id is 1-based index)
    id_to_genus = {otu_ids[i]: taxa_names[i] for i in range(len(otu_ids))}
    freq_df["genus"]    = freq_df["otu_id"].map(id_to_genus)
    freq_df["in_manual"] = freq_df["otu_id"].isin(manual_96)
    for race in RACES:
        freq_df[f"in_manual_{race.lower()}"] = freq_df["otu_id"].isin(manual_per_race[race])

    freq_df = freq_df.sort_values(["in_manual", "n_runs_selected"],
                                   ascending=[False, False]).reset_index(drop=True)
    freq_df.to_csv(OUT_FREQ, index=False)
    print(f"  Saved {OUT_FREQ.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    manual_rows    = freq_df[freq_df["in_manual"]]
    non_manual_rows= freq_df[~freq_df["in_manual"]]

    always     = freq_df[freq_df["n_runs_selected"] == n_done]
    majority   = freq_df[(freq_df["n_runs_selected"] >= n_done*0.5) &
                          (freq_df["n_runs_selected"] <  n_done)]
    occasional = freq_df[(freq_df["n_runs_selected"] >= 2) &
                          (freq_df["n_runs_selected"] <  n_done*0.5)]
    once       = freq_df[freq_df["n_runs_selected"] == 1]

    always_man   = always[always["in_manual"]]
    always_nman  = always[~always["in_manual"]]

    lines = [
        f"Taxa Selection Stability — Weighted MSE AE + 95%-mass threshold",
        f"N={n_done} runs (seeds 0–{n_done-1})  |  Reference: 96-taxon manual core",
        "",
        f"── Manual Core Taxa (96 total) ────────────────────────────────",
        f"  Always selected (100% of runs) : {len(always_man):>3}",
        f"  Majority selected  (≥50% runs) : {len(manual_rows[(manual_rows.n_runs_selected>=n_done*0.5) & (manual_rows.n_runs_selected<n_done)]):>3}",
        f"  Occasional         (<50% runs) : {len(manual_rows[(manual_rows.n_runs_selected>0) & (manual_rows.n_runs_selected<n_done*0.5)]):>3}",
        f"  Never selected     (0 runs)    : {96 - len(manual_rows):>3}",
        "",
        f"── Non-Manual Taxa ─────────────────────────────────────────────",
        f"  Always selected (100% of runs) : {len(always_nman):>3}",
        f"  Majority selected  (≥50% runs) : {len(non_manual_rows[non_manual_rows.n_runs_selected>=n_done*0.5]):>3}",
        f"  Occasional         (<50% runs) : {len(non_manual_rows[(non_manual_rows.n_runs_selected>0) & (non_manual_rows.n_runs_selected<n_done*0.5)]):>3}",
        "",
        f"── Always-Selected Manual Taxa ({len(always_man)}) ───────────────────────",
    ]
    for _, row in always_man.iterrows():
        lines.append(f"  OTU {row.otu_id:>4}  {row.genus}")
    lines += [
        "",
        f"── Manual Taxa NOT Always Selected ──────────────────────────────",
    ]
    missed = manual_rows[manual_rows["n_runs_selected"] < n_done].sort_values(
        "n_runs_selected", ascending=False)
    for _, row in missed.iterrows():
        lines.append(f"  OTU {int(row.otu_id):>4}  {row.n_runs_selected:>3}/{n_done} runs  "
                     f"({row.freq:.0%})  {row.genus}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_TXT, "w") as fh:
        fh.write(summary + "\n")
    print(f"\n  Saved {OUT_TXT.name}")
    print(f"\n✓  Done in {(time.time()-t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
