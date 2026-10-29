"""
kegg_functional_prediction.py
──────────────────────────────
Tax4Fun-equivalent: predicts per-sample KEGG pathway abundances from the
OTU abundance table × KO gene profiles (from KEGG REST API queries).

Method:
  1. Load per-taxon KO lists from kegg_expanded_cache.csv
  2. Build binary taxon × KO presence matrix
  3. Multiply TSS-normalised abundance (samples × taxa) by KO matrix
     → predicted KO abundance per sample
  4. Map KOs to KEGG pathway modules
  5. Aggregate to per-sample pathway abundances
  6. Compare across races + AE vs manual core taxa

Requires kegg_expanded.py to have been run first (for the KO cache).

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.kegg_functional_prediction
"""

import sys
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import ROOT_DIR

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KEGG_BASE = "https://rest.kegg.jp"
PAUSE_SEC = 0.35

PATHWAY_CATEGORIES = {
    "Carbon metabolism":        ["carbon", "glycolysis", "citrate", "tca",
                                 "gluconeogenesis", "pyruvate", "pentose"],
    "Fatty acid / lipid":       ["fatty acid", "lipid", "sterol", "cholesterol",
                                 "sphingo", "glycerolipid"],
    "Amino acid metabolism":    ["amino acid", "alanine", "valine", "leucine",
                                 "lysine", "arginine", "proline", "serine",
                                 "phenylalanine", "tryptophan", "tyrosine",
                                 "cysteine", "methionine", "histidine"],
    "Short-chain fatty acids":  ["butyrate", "propionate", "acetate",
                                 "butanoate", "propanoate", "scfa"],
    "Vitamin / cofactor":       ["vitamin", "folate", "riboflavin", "thiamine",
                                 "cobalamin", "biotin", "pantothenate"],
    "Secondary metabolites":    ["polyketide", "terpenoid", "alkaloid",
                                 "phenylpropanoid", "antibiotic", "terpene"],
    "Nitrogen / nucleotide":    ["nitrogen", "purine", "pyrimidine",
                                 "nucleotide", "urea"],
    "Carbohydrate degradation": ["starch", "sucrose", "galactose", "fructose",
                                 "cellulose", "pectin", "polysaccharide",
                                 "glycan", "xylose", "mannose"],
    "Sulfur metabolism":        ["sulfur", "cysteine", "methionine", "sulphate"],
    "Energy metabolism":        ["oxidative phosphorylation", "photosynthesis",
                                 "nitrogen fixation", "atp", "electron"],
    "Signalling / regulation":  ["signaling", "quorum", "two-component",
                                 "mapk", "phosphorelay"],
}


def categorise_pathway(name: str) -> str:
    name_lower = name.lower()
    for cat, kws in PATHWAY_CATEGORIES.items():
        if any(kw in name_lower for kw in kws):
            return cat
    return "Other metabolism"


def tss_normalize(df: pd.DataFrame) -> pd.DataFrame:
    col_sums = df.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return df.div(col_sums, axis=1)


def load_inputs():
    print("  Loading abundance table …")
    t0 = time.time()
    abund = pd.read_csv(ROOT_DIR / "Taxonomic_abundance.csv", index_col=0)
    print(f"    {abund.shape[0]} taxa × {abund.shape[1]} samples  ({time.time()-t0:.0f}s)")

    n_samples   = abund.shape[1]
    prevalence  = (abund > 0).sum(axis=1) / n_samples
    abund_filt  = abund[prevalence >= 0.05]
    abund_tss   = tss_normalize(abund_filt)
    print(f"    Pre-filtered: {abund_filt.shape[0]} taxa retained")

    meta = pd.read_csv(ROOT_DIR / "metadata_clean.csv")
    meta.index = meta["sample_id"]

    cache = pd.read_csv(ROOT_DIR / "kegg_expanded_cache.csv")
    print(f"  KEGG cache: {len(cache)} taxa  ({(cache['N_KOs']>0).sum()} with KO data)")

    return abund_tss, meta, cache


def build_ko_matrix(abund_tss: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary taxon × KO matrix for taxa in both the abundance table
    and the KEGG cache with KO data.
    Returns DataFrame with taxon names as index, KO IDs as columns.
    """
    matched_cache = cache[cache["N_KOs"] > 0].copy()

    # Map genus name → KO list
    genus_to_kos: dict[str, list[str]] = {}
    for _, row in matched_cache.iterrows():
        genus = str(row["Genus"])
        kos   = [k for k in str(row["KO_List"]).split(";") if k.startswith("K")]
        if kos:
            genus_to_kos[genus] = kos

    # Match to abundance table row names (genus names)
    all_kos: set[str] = set()
    for kos in genus_to_kos.values():
        all_kos.update(kos)

    if not all_kos:
        print("  ⚠  No KO data found in cache — run kegg_expanded.py first")
        return pd.DataFrame()

    all_kos_sorted = sorted(all_kos)
    rows: dict[str, list[int]] = {}

    for taxon in abund_tss.index:
        kos = genus_to_kos.get(taxon)
        if kos:
            presence = [1 if ko in set(kos) else 0 for ko in all_kos_sorted]
            rows[taxon] = presence

    if not rows:
        print("  ⚠  No taxa matched between abundance table and KEGG cache")
        return pd.DataFrame()

    ko_matrix = pd.DataFrame(rows, index=all_kos_sorted).T
    print(f"  KO matrix: {ko_matrix.shape[0]} taxa × {ko_matrix.shape[1]} KOs")
    return ko_matrix


def predict_ko_abundances(abund_tss: pd.DataFrame,
                          ko_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Multiply abundance (taxa × samples) by KO presence (taxa × KOs).
    Result: samples × KOs predicted abundance.
    """
    common_taxa = abund_tss.index.intersection(ko_matrix.index)
    print(f"  Taxa in prediction: {len(common_taxa)}")

    A = abund_tss.loc[common_taxa].T.values        # samples × taxa
    K = ko_matrix.loc[common_taxa].values          # taxa × KOs

    predicted = A @ K                              # samples × KOs

    result = pd.DataFrame(
        predicted,
        index=abund_tss.columns,
        columns=ko_matrix.columns,
    )
    return result


def fetch_ko_pathway_map(ko_list: list[str]) -> dict[str, list[str]]:
    """
    Fetch KO → pathway mapping from KEGG for the KOs we have.
    Uses batched requests (KEGG allows up to 10 IDs per request).
    """
    ko_pathway: dict[str, list[str]] = {ko: [] for ko in ko_list}
    batch_size = 10
    batches    = [ko_list[i:i+batch_size] for i in range(0, len(ko_list), batch_size)]

    print(f"  Fetching KO→pathway map for {len(ko_list)} KOs in {len(batches)} batches …")

    for i, batch in enumerate(batches):
        if i % 50 == 0:
            print(f"    Batch {i+1}/{len(batches)} …", flush=True)
        try:
            ids  = "+".join(batch)
            resp = requests.get(f"{KEGG_BASE}/link/pathway/{ids}", timeout=15)
            if resp.status_code == 200:
                for line in resp.text.strip().splitlines():
                    p = line.split("\t")
                    if len(p) >= 2:
                        ko = p[0].replace("ko:", "")
                        pw = p[1].replace("path:", "")
                        if ko in ko_pathway:
                            ko_pathway[ko].append(pw)
        except Exception:
            pass
        time.sleep(PAUSE_SEC)

    return ko_pathway


def ko_to_pathway_df(predicted_ko: pd.DataFrame,
                     ko_pathway_map: dict[str, list[str]],
                     ko_names: dict[str, str]) -> pd.DataFrame:
    """
    Aggregate per-sample KO abundances to pathway level.
    Returns samples × pathways DataFrame.
    """
    pathway_set: set[str] = set()
    for pws in ko_pathway_map.values():
        pathway_set.update(pws)
    pathways = sorted(pathway_set)

    samples       = predicted_ko.index.tolist()
    pathway_abund = np.zeros((len(samples), len(pathways)))
    pw_idx        = {pw: i for i, pw in enumerate(pathways)}

    for ko in predicted_ko.columns:
        if ko not in ko_pathway_map:
            continue
        ko_vals = predicted_ko[ko].values
        for pw in ko_pathway_map[ko]:
            if pw in pw_idx:
                pathway_abund[:, pw_idx[pw]] += ko_vals

    result = pd.DataFrame(pathway_abund, index=samples, columns=pathways)
    return result


def per_race_means(pathway_df: pd.DataFrame,
                   meta: pd.DataFrame) -> pd.DataFrame:
    races = ["African", "Asian", "Caucasian", "Hispanic"]
    out   = {}
    for race in races:
        samples = meta[meta["race"] == race]["sample_id"].tolist()
        samples = [s for s in samples if s in pathway_df.index]
        if samples:
            out[race] = pathway_df.loc[samples].mean(axis=0)
    return pd.DataFrame(out)


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_pathway_heatmap(race_means: pd.DataFrame, out_path: Path, top_n: int = 30):
    df = race_means.copy()
    df = df.loc[df.max(axis=1) > 0]
    if df.empty:
        print(f"  ⚠  No non-zero pathways for heatmap")
        return
    df = df.loc[df.sum(axis=1).nlargest(top_n).index]

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.3)))
    sns.heatmap(df, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Predicted pathway abundance", "shrink": 0.6})
    ax.set_title(f"Top {top_n} predicted KEGG pathways across races\n"
                 "(Tax4Fun-style: abundance × KO presence)",
                 fontsize=10, pad=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_category_race_bars(race_means: pd.DataFrame,
                            pathway_names: dict[str, str],
                            out_path: Path, top_n: int = 12):
    """Aggregate to pathway categories, grouped bars by race."""
    df = race_means.copy().T   # races × pathways
    df.columns = [pathway_names.get(c, c) for c in df.columns]

    # Categorise columns
    df_cat = df.T.copy()
    df_cat.index = [categorise_pathway(n) for n in df_cat.index]
    df_cat = df_cat.groupby(level=0).sum()
    df_cat = df_cat.loc[df_cat.max(axis=1).nlargest(top_n).index]

    fig, ax = plt.subplots(figsize=(12, 6))
    x       = np.arange(len(df_cat.index))
    n_races = len(df_cat.columns)
    w       = 0.75 / n_races
    colors  = ["#e6550d", "#3182bd", "#31a354", "#756bb1"]

    for i, (race, color) in enumerate(zip(df_cat.columns, colors)):
        ax.bar(x + i * w, df_cat[race], w, label=race, color=color, alpha=0.85)

    ax.set_xticks(x + w * (n_races - 1) / 2)
    ax.set_xticklabels([c.replace(" ", "\n") for c in df_cat.index],
                       fontsize=8, ha="center")
    ax.set_ylabel("Predicted pathway abundance", fontsize=10)
    ax.set_title(f"Top {top_n} KEGG pathway categories across races\n"
                 "(Tax4Fun-style prediction from OTU abundance × KO profiles)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "═" * 60)
    print("  KEGG Functional Prediction (Tax4Fun-equivalent)")
    print("═" * 60)

    cache_path = ROOT_DIR / "kegg_expanded_cache.csv"
    if not cache_path.exists():
        print("\n  ✗  kegg_expanded_cache.csv not found.")
        print("     Run pipeline.kegg_expanded first.\n")
        sys.exit(1)

    # 1. Load inputs
    print("\n  Loading data …")
    abund_tss, meta, cache = load_inputs()

    # 2. Build KO matrix
    print("\n  Building taxon × KO matrix …")
    ko_matrix = build_ko_matrix(abund_tss, cache)
    if ko_matrix.empty:
        sys.exit(1)

    ko_matrix.to_csv(ROOT_DIR / "kegg_ko_matrix.csv")
    print(f"  Saved kegg_ko_matrix.csv")

    # 3. Predict per-sample KO abundances
    print("\n  Predicting per-sample KO abundances …")
    predicted_ko = predict_ko_abundances(abund_tss, ko_matrix)
    print(f"  Predicted shape: {predicted_ko.shape[0]} samples × {predicted_ko.shape[1]} KOs")

    # 4. Fetch KO → pathway map
    print("\n  Fetching KO → pathway map from KEGG …")
    ko_list        = predicted_ko.columns.tolist()
    ko_pathway_map = fetch_ko_pathway_map(ko_list)

    mapped_kos = sum(1 for v in ko_pathway_map.values() if v)
    print(f"  KOs with pathway mapping: {mapped_kos} / {len(ko_list)}")

    # Fetch pathway names
    print("  Fetching pathway names …")
    pathway_names: dict[str, str] = {}
    all_pathway_ids = sorted({pw for pws in ko_pathway_map.values() for pw in pws
                               if pw.startswith("map")})
    for i in range(0, min(len(all_pathway_ids), 200), 10):
        batch = all_pathway_ids[i:i+10]
        try:
            ids  = "+".join(batch)
            resp = requests.get(f"{KEGG_BASE}/get/{ids}", timeout=15)
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.startswith("NAME"):
                        name = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                        # Associate with last ID in batch (simplification)
                        if batch:
                            pathway_names[batch[0]] = name
        except Exception:
            pass
        time.sleep(PAUSE_SEC)

    # 5. Aggregate to pathway level
    print("\n  Aggregating KOs → pathways …")
    pathway_df = ko_to_pathway_df(predicted_ko, ko_pathway_map, pathway_names)
    print(f"  Pathway table: {pathway_df.shape[0]} samples × {pathway_df.shape[1]} pathways")

    pathway_df.to_csv(ROOT_DIR / "kegg_predicted_pathways.csv")
    print(f"  Saved kegg_predicted_pathways.csv")

    # 6. Per-race means
    race_means = per_race_means(pathway_df, meta)
    race_means.to_csv(ROOT_DIR / "kegg_race_pathway_means.csv")
    print(f"  Saved kegg_race_pathway_means.csv")

    # 7. Figures
    print("\n  Generating figures …")
    plot_pathway_heatmap(
        race_means,
        FIGURES_DIR / "fig_kegg_predicted_heatmap.pdf",
        top_n=30,
    )
    plot_category_race_bars(
        race_means,
        pathway_names,
        FIGURES_DIR / "fig_kegg_predicted_race_bars.pdf",
        top_n=12,
    )

    # 8. Summary
    print("\n  ── Top predicted pathways (mean across races) ───────────────")
    if not race_means.empty:
        top5 = race_means.mean(axis=1).nlargest(5)
        for pw, val in top5.items():
            name = pathway_names.get(pw, pw)
            print(f"    {val:.4f}  {pw}  {name}")

    elapsed = (time.time() - t0) / 60
    print(f"\n✓  Done in {elapsed:.1f} min.\n")


if __name__ == "__main__":
    main()
