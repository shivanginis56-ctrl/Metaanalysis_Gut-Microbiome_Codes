"""
unsupervised_pipeline.py
────────────────────────
Within-Group Autoencoder Pipeline for Emerged Core Microbiome Identification.

For each of the 4 race groups (African, Asian, Caucasian, Hispanic):
  1. Extract all samples belonging to that race (cross-bioproject).
  2. TSS-normalise → Multiplicative Replacement (CoDA-safe zero handling).
  3. Apply CLR transformation → Euclidean compositional space.
  4. Train Autoencoder with prevalence-weighted MSE loss (v2):
       4680 → 512 → 128 → 64 → 128 → 512 → 4680
  5. Score taxa via CLR-weighted inverse MSE: max(0,mean_CLR)/(MSE+ε)
  6. Select adaptive TOP_N via Kneedle elbow on sorted CoreScore curve.
  7. Save results + generate figures vs manual reference.

Run from Datasets/ with the venv active:
    /Users/sampathuppala/Desktop/Datasets/microbiome-ml-env/bin/python \\
        -m pipeline.unsupervised_pipeline

Outputs
───────
  ae_emerged_core.csv                         – tidy long-format (all races)
  reports/figures/fig_ae_training_loss.pdf    – loss curves per race
  reports/figures/fig_ae_venn_diagrams.pdf    – Venn diagrams vs manual
  reports/figures/fig_ae_top20_importance.pdf – bar chart top-20 per race
  reports/figures/fig_ae_cross_race_overlap.pdf
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pathlib import Path

import torch
from kneed import KneeLocator

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config      import (METADATA_FILE, ABUNDANCE_FILE, ORIGINAL_CORE,
                                   ROOT_DIR, CORRECTED_BIOPROJECTS)
from pipeline.data_loader import load_metadata, group_samples, load_abundance
from pipeline.comparator  import parse_original_core
from pipeline.autoencoder import (
    multiplicative_replacement,
    clr_transform_group,
    get_feature_importance,
)
from pipeline.autoencoder_v2 import train_autoencoder_v2

# ── Reproducibility ────────────────────────────────────────────────────────────
import random as _random
RANDOM_SEED = 42
_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Configuration ──────────────────────────────────────────────────────────────
AE_HIDDEN_DIMS  = [512, 128]       # encoder hidden layer widths
AE_BOTTLENECK   = 64               # latent space dimension
AE_DROPOUT      = 0.15             # dropout rate during training
AE_EPOCHS       = 150              # max training epochs (early stopping may cut short)
AE_LR           = 1e-3             # Adam learning rate
AE_WEIGHT_DECAY = 1e-4             # L2 regularisation
AE_PATIENCE     = 20               # early-stopping patience (epochs)
IMPORTANCE_METHOD = "clr_weighted_mse"   # primary method: max(0,mean_CLR)/(MSE+ε)
TOP_N_MIN       = 1                # no floor — let the data decide
TOP_N_MAX       = 4680             # no ceiling — let the data decide
USE_WEIGHTED_MSE = True            # prevalence-weighted reconstruction loss

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV  = ROOT_DIR / "ae_emerged_core.csv"

RACE_COLORS = {
    "African":   "#E07B54",
    "Asian":     "#5B9BD5",
    "Caucasian": "#70AD47",
    "Hispanic":  "#9B59B6",
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}", flush=True)


def adaptive_top_n(scores: np.ndarray, min_n: int = TOP_N_MIN,
                   max_n: int = TOP_N_MAX) -> int:
    """
    Find the natural elbow of the sorted-descending CoreScore curve using the
    Kneedle algorithm.  Falls back to max_n if no clear knee is detected.

    The elbow marks the point where additional taxa add rapidly diminishing
    structural importance — an objective, threshold-free cutoff.
    """
    sorted_scores = np.sort(scores)[::-1]
    n = min(max_n, len(sorted_scores))
    x = np.arange(1, n + 1, dtype=float)
    y = sorted_scores[:n]

    # Normalise to [0,1] so Kneedle works regardless of score scale
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return min_n

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = (x - x.min()) / (x.max() - x.min())

    try:
        kl = KneeLocator(x_norm, y_norm, curve="convex", direction="decreasing",
                         interp_method="polynomial")
        knee_x = kl.knee        # normalised x position of knee
        if knee_x is None:
            return max_n
        # Convert back to index count
        top_n = int(round(knee_x * (n - 1))) + 1
        return max(min_n, min(max_n, top_n))
    except Exception:
        return max_n


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Per-race Processing ────────────────────────────────────────────────────────

def process_race(
    race:             str,
    sample_ids:       list[str],
    raw_df:           pd.DataFrame,
    taxa_names:       list[str],
    otu_ids:          list[int],
    device:           torch.device,
    feature_weights:  np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Full autoencoder pipeline for one race group.

    Returns
    -------
    ranked_df : DataFrame with all 4,680 taxa ranked by importance score.
    losses    : Per-epoch training loss list.
    """
    n_samples  = len(sample_ids)
    n_features = len(taxa_names)
    print(f"\n  Race: {race}  |  n_samples={n_samples}  |  n_features={n_features}")

    # ── Extract raw counts for this race ─────────────────────────────────────
    valid_ids = [s for s in sample_ids if s in raw_df.columns]
    # Transpose: rows=samples, cols=taxa  (required format for the AE)
    X_raw = raw_df[valid_ids].T.values.astype(float)   # (n_samples, n_taxa)

    # ── TSS normalise to proportions ──────────────────────────────────────────
    # Multiplicative replacement is defined for compositions (proportions that
    # sum to ~1).  Raw counts make the auto-derived delta catastrophically large
    # for sparse samples, yielding negative values → log(negative) = NaN.
    row_totals = X_raw.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1.0          # guard against fully-zero rows
    X_prop = X_raw / row_totals                # (n_samples, n_taxa), rows ≈ 1

    # ── Step A: Multiplicative Replacement on proportions ────────────────────
    # delta = 0.65 / D is the standard CoDA recommendation (Martín-Fernández
    # et al. 2003 / 2011).  Guarantees n_zeros * delta < D * (0.65/D) = 0.65 < 1
    # so the rescaling factor (1 − n_zeros·δ / row_sum) is always positive.
    print("    Step A: Multiplicative replacement (on proportions) …", flush=True)
    D = X_prop.shape[1]
    delta_coda = 0.65 / D
    X_replaced = multiplicative_replacement(X_prop, delta=delta_coda)

    # ── Step B: CLR Transformation ────────────────────────────────────────────
    print("    Step B: CLR transformation …", flush=True)
    X_clr = clr_transform_group(X_replaced)            # (n_samples, n_taxa)

    # Standardise each feature to zero mean (helps training stability)
    col_means = X_clr.mean(axis=0, keepdims=True)
    col_stds  = X_clr.std(axis=0,  keepdims=True) + 1e-8
    X_scaled  = (X_clr - col_means) / col_stds

    # ── Step C: Train Autoencoder (weighted MSE if enabled) ──────────────────
    wt_str = "prevalence-weighted MSE" if feature_weights is not None else "plain MSE"
    print(f"    Step C: Training autoencoder ({AE_EPOCHS} max epochs, "
          f"patience={AE_PATIENCE}, loss={wt_str}) …", flush=True)
    model, losses = train_autoencoder_v2(
        X_clr           = X_scaled,
        n_features      = n_features,
        hidden_dims     = AE_HIDDEN_DIMS,
        bottleneck      = AE_BOTTLENECK,
        dropout         = AE_DROPOUT,
        epochs          = AE_EPOCHS,
        lr              = AE_LR,
        weight_decay    = AE_WEIGHT_DECAY,
        patience        = AE_PATIENCE,
        device_str      = str(device),
        verbose         = True,
        feature_weights = feature_weights,
    )

    # ── Step D: Feature Importance (three methods, primary = gradient saliency)
    print(f"    Step D: Computing feature importance "
          f"(method={IMPORTANCE_METHOD}) …", flush=True)

    scores, sort_dir = get_feature_importance(
        model        = model,
        X_clr        = X_scaled,    # standardised — used for MSE
        method       = IMPORTANCE_METHOD,
        device       = device,
        X_clr_orig   = X_clr,       # raw CLR — used for mean_CLR gating
    )

    # Also compute the other two for completeness
    scores_mse, _     = get_feature_importance(model, X_scaled, "reconstruction_mse", device)
    scores_weight, _  = get_feature_importance(model, X_scaled, "weight_magnitude",   device)

    # ── Step E: Rank taxa + adaptive TOP_N ───────────────────────────────────
    if sort_dir == "asc":
        ranked_idx = np.argsort(scores)
    else:
        ranked_idx = np.argsort(scores)[::-1]

    top_n = adaptive_top_n(scores)

    ranked_df = pd.DataFrame({
        "race":             race,
        "rank":             np.arange(1, n_features + 1),
        "otu_id":           [otu_ids[i]   for i in ranked_idx],
        "genus":            [taxa_names[i] for i in ranked_idx],
        "importance_score": scores[ranked_idx],
        "importance_method":IMPORTANCE_METHOD,
        "mse_score":        scores_mse[ranked_idx],
        "weight_score":     scores_weight[ranked_idx],
        "emerged_core":     np.arange(1, n_features + 1) <= top_n,
    })

    print(f"    → Adaptive TOP_N = {top_n}  |  emerged core identified  "
          f"(final training loss = {losses[-1]:.6f})")

    return ranked_df, losses


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_training_losses(loss_dict: dict[str, list[float]]) -> None:
    """Loss curves for all 4 races on a single figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (race, losses) in zip(axes, loss_dict.items()):
        color = RACE_COLORS.get(race, "#555")
        ax.plot(losses, color=color, lw=1.8)
        ax.set_title(f"{race}  (final loss={losses[-1]:.5f})", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE Loss", fontsize=9)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Autoencoder Training Loss per Race\n"
        f"Architecture: {4680} → {AE_HIDDEN_DIMS[0]} → {AE_HIDDEN_DIMS[1]} "
        f"→ {AE_BOTTLENECK} → … → {4680}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "fig_ae_training_loss.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_venn_diagrams(
    ae_results:  dict[str, pd.DataFrame],
    manual_df:   pd.DataFrame,
) -> None:
    """
    2-set Venn diagrams per race: Autoencoder Emerged Core vs Manual Reference.
    Manual reference is the UNION of all manual core taxa for that race
    across all bioprojects.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Aggregate manual core per race (union across bioprojects)
    manual_per_race = (
        manual_df.groupby("race")["genus"]
        .apply(set)
        .to_dict()
    )

    for ax, race in zip(axes, ["African", "Asian", "Caucasian", "Hispanic"]):
        ranked_df = ae_results.get(race)
        color     = RACE_COLORS.get(race, "#888")

        if ranked_df is None:
            ax.text(0.5, 0.5, f"{race}\n(skipped)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.axis("off")
            continue

        ae_core  = set(ranked_df[ranked_df["emerged_core"]]["genus"])
        man_core = manual_per_race.get(race, set())
        shared   = ae_core & man_core
        only_ae  = ae_core  - man_core
        only_man = man_core - ae_core

        v = venn2(
            subsets       = (len(only_ae), len(only_man), len(shared)),
            set_labels    = (f"AE Core\n(n={len(ae_core)})",
                             f"Manual\n(n={len(man_core)})"),
            set_colors    = (color, "#AAAAAA"),
            alpha         = 0.6,
            ax            = ax,
        )

        # Annotate counts
        if v.get_label_by_id("10"):
            v.get_label_by_id("10").set_text(str(len(only_ae)))
        if v.get_label_by_id("01"):
            v.get_label_by_id("01").set_text(str(len(only_man)))
        if v.get_label_by_id("11"):
            v.get_label_by_id("11").set_text(str(len(shared)))

        precision = len(shared) / len(ae_core)  if ae_core  else 0
        recall    = len(shared) / len(man_core) if man_core else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0)

        ax.set_title(
            f"{race}\nP={precision:.2f}  R={recall:.2f}  F1={f1:.2f}",
            fontsize=11, pad=12,
        )

    fig.suptitle(
        "Emerged Core (Autoencoder) vs Manual Reference per Race\n"
        "Manual = union of all bioproject cores for that race",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "fig_ae_venn_diagrams.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_top20_importance(ae_results: dict[str, pd.DataFrame]) -> None:
    """
    Horizontal bar chart of top-20 taxa by importance score per race.
    Bars coloured by whether the taxon is also in the manual reference.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax, race in zip(axes, ["African", "Asian", "Caucasian", "Hispanic"]):
        ranked_df = ae_results.get(race)
        color     = RACE_COLORS.get(race, "#888")

        if ranked_df is None:
            ax.axis("off")
            continue

        top20 = ranked_df.head(20).iloc[::-1]   # reverse for horizontal bar

        ax.barh(
            top20["genus"],
            top20["importance_score"],
            color=color, alpha=0.80, edgecolor="white",
        )
        ax.set_title(f"{race}  — Top 20 Emerged Core Taxa\n"
                     f"(method: {IMPORTANCE_METHOD.replace('_', ' ')})", fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Top-20 Emerged Core Taxa per Race  |  "
        f"Autoencoder ({AE_HIDDEN_DIMS[0]}→{AE_HIDDEN_DIMS[1]}→{AE_BOTTLENECK})",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "fig_ae_top20_importance.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_emerged_core_overlap_heatmap(ae_results: dict[str, pd.DataFrame]) -> None:
    """
    Heatmap showing which top-100 emerged core taxa are shared across races.
    Useful for identifying a pan-racial 'universal' core.
    """
    # Collect all emerged core taxa across races
    all_emerged = {}
    for race, df in ae_results.items():
        all_emerged[race] = set(df[df["emerged_core"]]["genus"])

    all_taxa = sorted(set().union(*all_emerged.values()))

    # Build presence matrix
    presence = pd.DataFrame(
        {race: [1 if t in taxa else 0 for t in all_taxa]
         for race, taxa in all_emerged.items()},
        index=all_taxa,
    )
    # Show only taxa appearing in ≥2 races (shared core candidates)
    presence = presence[presence.sum(axis=1) >= 2].sort_values(
        by=list(all_emerged.keys()), ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(presence) * 0.28)))
    im = ax.imshow(presence.T.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(presence)))
    ax.set_yticks(np.arange(len(presence.columns)))
    ax.set_xticklabels(presence.index, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(presence.columns, fontsize=10)

    # Draw grid lines
    for x in np.arange(-0.5, len(presence), 1):
        ax.axvline(x, color="white", lw=0.5)
    for y in np.arange(-0.5, len(presence.columns), 1):
        ax.axhline(y, color="white", lw=0.5)

    ax.set_title(
        f"Cross-Race Overlap of Emerged Core Taxa\n"
        f"(showing taxa present in ≥2 races; total unique = {len(all_taxa)})",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "fig_ae_cross_race_overlap.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   WITHIN-GROUP AUTOENCODER CORE MICROBIOME PIPELINE     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    device = _select_device()
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    _banner("Load metadata & abundance matrix")
    meta   = load_metadata(METADATA_FILE)
    groups = group_samples(meta)

    # Group by RACE only (aggregate across all bioprojects)
    race_groups: dict[str, list[str]] = {}
    for (bp, race), sids in groups.items():
        race_groups.setdefault(race, []).extend(sids)
    print(f"  Race groups: { {r: len(s) for r, s in race_groups.items()} }")

    # Load FULL abundance matrix (all 4,680 taxa — per the engineered prompt)
    taxa_names, otu_ids, raw_df = load_abundance(ABUNDANCE_FILE)
    n_features = len(taxa_names)
    print(f"  Full feature space: {n_features} taxa × {raw_df.shape[1]} samples\n")

    # ── Compute global prevalence weights (for weighted MSE loss) ─────────────
    feature_weights: np.ndarray | None = None
    if USE_WEIGHTED_MSE:
        _banner("Computing prevalence weights")
        binary_path = ROOT_DIR / "step1_binary_normalised.csv"
        binary      = pd.read_csv(binary_path, index_col=0)
        prev_arr    = binary.sum(axis=1).values / binary.shape[1]
        feature_weights = prev_arr
        print(f"  Mean global prevalence: {prev_arr.mean():.4f}  "
              f"(weights normalised to mean=1 inside trainer)")

    # ── Per-race autoencoder pipeline ─────────────────────────────────────────
    _banner("Running within-group autoencoder per race")

    all_results: list[pd.DataFrame]      = []
    ae_results:  dict[str, pd.DataFrame] = {}
    loss_dict:   dict[str, list[float]]  = {}

    for race in ["African", "Asian", "Caucasian", "Hispanic"]:
        sample_ids = race_groups.get(race, [])
        if not sample_ids:
            print(f"\n  SKIP {race}: no samples found")
            continue

        ranked_df, losses = process_race(
            race            = race,
            sample_ids      = sample_ids,
            raw_df          = raw_df,
            taxa_names      = taxa_names,
            otu_ids         = otu_ids,
            device          = device,
            feature_weights = feature_weights,
        )

        all_results.append(ranked_df)
        ae_results[race]  = ranked_df
        loss_dict[race]   = losses

    # ── Save tidy CSV ─────────────────────────────────────────────────────────
    _banner("Save emerged core CSV")
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    n_core = result_df["emerged_core"].sum()
    print(f"  Saved {len(result_df)} rows → {OUTPUT_CSV.name}")
    print(f"  ({n_core} emerged-core taxon entries across all races)")

    # ── Figures ───────────────────────────────────────────────────────────────
    _banner("Generating figures")

    print("  Training loss curves …")
    plot_training_losses(loss_dict)

    print("  Top-20 importance bar charts …")
    plot_top20_importance(ae_results)

    print("  Cross-race overlap heatmap …")
    plot_emerged_core_overlap_heatmap(ae_results)

    print("  Venn diagrams vs manual reference …")
    try:
        manual_df = parse_original_core(ORIGINAL_CORE)
        # Normalise race labels in manual_df to match our labels
        manual_df["race"] = manual_df["race"].str.strip()
        plot_venn_diagrams(ae_results, manual_df)
    except Exception as exc:
        print(f"  WARNING: Venn diagram skipped — {exc}")

    # ── Summary table ─────────────────────────────────────────────────────────
    _banner("Emerged Core Summary")
    print(f"  {'Race':<12} {'n_samples':>10} {'n_emerged_core':>15}  (adaptive TOP_N)")
    print(f"  {'─'*12} {'─'*10} {'─'*15}")
    for race in ["African", "Asian", "Caucasian", "Hispanic"]:
        n_samp = len(race_groups.get(race, []))
        n_core = (result_df[(result_df.race == race) & result_df.emerged_core].shape[0])
        print(f"  {race:<12} {n_samp:>10,} {n_core:>15}")

    elapsed = time.time() - t0
    print(f"\n✓  Autoencoder pipeline complete in {elapsed/60:.1f} min.\n")


if __name__ == "__main__":
    main()
