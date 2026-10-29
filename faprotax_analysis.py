"""
faprotax_analysis.py
────────────────────
Functional annotation of microbiome taxa using FAPROTAX, then compares
functional profiles across:
  - 4 racial groups (African, Asian, Caucasian, Hispanic)
  - AE core vs Manual core vs AE-only taxa contributions

Requires FAPROTAX to be downloaded first:
  1. Go to: https://pages.uoregon.edu/slouca/LoucaLab/archive/FAPROTAX/lib/php/index.php?section=Download
  2. Download and unzip to:  /Users/sampathuppala/Desktop/Datasets/FAPROTAX/
  3. Confirm these two files exist:
       FAPROTAX/collapse_table.py
       FAPROTAX/FAPROTAX.txt

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.faprotax_analysis
"""

import sys
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import ROOT_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────
FAPROTAX_DIR    = ROOT_DIR / "FAPROTAX_1.2.12"
COLLAPSE_SCRIPT = FAPROTAX_DIR / "collapse_table.py"
FAPROTAX_DB     = FAPROTAX_DIR / "FAPROTAX.txt"
FIGURES_DIR     = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ABUNDANCE_FILE  = ROOT_DIR / "Taxonomic_abundance.csv"
METADATA_FILE   = ROOT_DIR / "metadata_clean.csv"
MANUAL_CORE_FILE= ROOT_DIR / "step3_final_core.csv"
AE_CORE_FILE    = ROOT_DIR / "ae_core_148.csv"
AE_ONLY_FILE    = ROOT_DIR / "ae_only_56_taxa.csv"

GLOBAL_PREV_THRESHOLD = 0.05   # pre-filter: same as pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_faprotax():
    if not COLLAPSE_SCRIPT.exists() or not FAPROTAX_DB.exists():
        print("\n  ✗  FAPROTAX not found.")
        print("     Download from:")
        print("       https://pages.uoregon.edu/slouca/LoucaLab/archive/FAPROTAX/lib/php/index.php?section=Download")
        print(f"     Unzip to: {FAPROTAX_DIR}")
        print("     Then re-run this script.\n")
        sys.exit(1)
    print(f"  ✓  FAPROTAX found at {FAPROTAX_DIR}")


def tss_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Total Sum Scaling: each sample sums to 1."""
    col_sums = df.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return df.div(col_sums, axis=1)


def load_data():
    print("\n  Loading abundance table (large file ~30s) …")
    t0 = time.time()
    abund = pd.read_csv(ABUNDANCE_FILE, index_col=0)
    print(f"    Loaded {abund.shape[0]} taxa × {abund.shape[1]} samples  ({time.time()-t0:.0f}s)")

    meta = pd.read_csv(METADATA_FILE)
    meta.index = meta["sample_id"]

    manual_core  = pd.read_csv(MANUAL_CORE_FILE)
    ae_core      = pd.read_csv(AE_CORE_FILE)
    ae_only      = pd.read_csv(AE_ONLY_FILE)

    # ── Pre-filter: global prevalence >= 5% ──────────────────────────────────
    n_samples    = abund.shape[1]
    prevalence   = (abund > 0).sum(axis=1) / n_samples
    abund_filt   = abund[prevalence >= GLOBAL_PREV_THRESHOLD]
    print(f"    Pre-filtered: {abund_filt.shape[0]} taxa retained (prevalence ≥ {GLOBAL_PREV_THRESHOLD*100:.0f}%)")

    # ── TSS normalize ─────────────────────────────────────────────────────────
    abund_tss = tss_normalize(abund_filt)

    return abund_tss, meta, manual_core, ae_core, ae_only


def prepare_faprotax_input(abund_tss: pd.DataFrame, tmp_dir: str, label: str = "input") -> Path:
    """
    Build TSV for FAPROTAX v1.2.12.
    Index name must NOT start with '#' (treated as comment).
    Use -d taxonomy to tell FAPROTAX which column has the row names.
    """
    df = abund_tss.copy()
    df.index.name = "taxonomy"   # no # prefix — # causes FAPROTAX to skip as comment

    out_path = Path(tmp_dir) / f"faprotax_{label}.tsv"
    df.to_csv(out_path, sep="\t")
    return out_path


def run_faprotax(input_tsv: Path, tmp_dir: str) -> tuple[Path, Path]:
    out_tsv    = Path(tmp_dir) / "faprotax_output.tsv"
    report_txt = Path(tmp_dir) / "faprotax_report.txt"

    cmd = [
        sys.executable, str(COLLAPSE_SCRIPT),
        "-i", str(input_tsv),
        "-g", str(FAPROTAX_DB),
        "-o", str(out_tsv),
        "-r", str(report_txt),
        "-d", "taxonomy",        # column containing row names (genus names to match)
        "-f",
    ]

    print("\n  Running FAPROTAX collapse_table.py …")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  FAPROTAX stderr:")
        print(result.stderr[-2000:])
        sys.exit(1)

    print(f"  ✓  FAPROTAX finished")
    return out_tsv, report_txt


def load_faprotax_output(out_tsv: Path) -> pd.DataFrame:
    func_df = pd.read_csv(out_tsv, sep="\t", index_col=0)
    # Drop the metadata rows FAPROTAX sometimes appends
    func_df = func_df[~func_df.index.astype(str).str.startswith("#")]
    # Convert to float
    func_df = func_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return func_df


# ── Analysis ───────────────────────────────────────────────────────────────────

def per_race_profiles(func_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Mean functional abundance per race."""
    races      = ["African", "Asian", "Caucasian", "Hispanic"]
    race_means = {}
    for race in races:
        samples = meta[meta["race"] == race]["sample_id"].tolist()
        samples = [s for s in samples if s in func_df.columns]
        if samples:
            race_means[race] = func_df[samples].mean(axis=1)
    return pd.DataFrame(race_means)


def taxa_subset_contribution(
    abund_tss: pd.DataFrame,
    func_df:   pd.DataFrame,
    otu_ids:   list[int],
    meta:      pd.DataFrame,
    label:     str,
    tmp_dir:   str,
) -> pd.DataFrame:
    """
    Re-run FAPROTAX on a subset of taxa to see their functional contribution.
    Returns per-race mean functional profile from that subset.
    """
    # Map OTU IDs (1-based) to row names
    valid_idx = [abund_tss.index[i - 1] for i in otu_ids if i <= len(abund_tss.index)]
    subset    = abund_tss.loc[valid_idx]

    if subset.empty:
        print(f"    ⚠  No taxa found for subset '{label}'")
        return pd.DataFrame()

    subset_tsv = Path(tmp_dir) / f"faprotax_input_{label}.tsv"
    s = subset.copy()
    s.index.name = "taxonomy"
    s.to_csv(subset_tsv, sep="\t")

    out_tsv = Path(tmp_dir) / f"faprotax_output_{label}.tsv"
    cmd = [
        sys.executable, str(COLLAPSE_SCRIPT),
        "-i", str(subset_tsv),
        "-g", str(FAPROTAX_DB),
        "-o", str(out_tsv),
        "-d", "taxonomy",
        "-f",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_tsv.exists():
        print(f"    ⚠  FAPROTAX failed for subset '{label}'")
        return pd.DataFrame()

    sub_func = load_faprotax_output(out_tsv)
    return per_race_profiles(sub_func, meta)


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_race_heatmap(race_means: pd.DataFrame, title: str, out_path: Path, top_n: int = 30):
    """Heatmap of top_n functional groups × 4 races."""
    df = race_means.copy()
    df = df.loc[df.max(axis=1) > 0]        # drop zeros
    if df.empty:
        print(f"  ⚠  Skipping heatmap '{out_path.name}' — no non-zero functional groups")
        return
    df = df.loc[df.sum(axis=1).nlargest(top_n).index]  # top N by total abundance

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.32)))
    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Mean relative abundance", "shrink": 0.6},
        yticklabels=True,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_contribution_bar(
    full_race: pd.DataFrame,
    ae_race:   pd.DataFrame,
    manual_race: pd.DataFrame,
    out_path:  Path,
    top_n:     int = 20,
):
    """
    Stacked bar: for top functions, what fraction comes from
    AE-core vs Manual-core vs rest.
    Uses Caucasian (largest group) as representative.
    """
    race = "Caucasian"
    if race not in full_race.columns:
        return

    total      = full_race[race].copy()
    ae_contrib = ae_race[race] if race in ae_race.columns else pd.Series(dtype=float)
    man_contrib= manual_race[race] if race in manual_race.columns else pd.Series(dtype=float)

    total = total[total > 0].nlargest(top_n)
    funcs = total.index

    ae_vals  = ae_contrib.reindex(funcs).fillna(0)
    man_vals = man_contrib.reindex(funcs).fillna(0)
    rest_vals= (total - ae_vals - man_vals).clip(lower=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(funcs))
    ax.barh(y, rest_vals.values,              color="#aec7e8", label="Other taxa")
    ax.barh(y, man_vals.values,  left=rest_vals.values, color="#ffbb78", label="Manual core (96)")
    ax.barh(y, ae_vals.values,   left=(rest_vals + man_vals).values, color="#d62728", label="AE core (117)")

    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_", " ") for f in funcs], fontsize=8)
    ax.set_xlabel("Mean relative abundance (Caucasian samples)", fontsize=10)
    ax.set_title(
        f"Functional contribution by taxa group — Top {top_n} functions\n"
        "(Caucasian samples as representative)",
        fontsize=10, pad=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_race_comparison_bar(race_means: pd.DataFrame, out_path: Path, top_n: int = 20):
    """Grouped bar chart: top functions, bars coloured by race."""
    df = race_means.copy()
    df = df.loc[df.max(axis=1) > 0]
    if df.empty:
        print(f"  ⚠  Skipping bar chart '{out_path.name}' — no non-zero functional groups")
        return
    df = df.loc[df.sum(axis=1).nlargest(top_n).index]

    fig, ax = plt.subplots(figsize=(12, 7))
    x     = np.arange(len(df.index))
    n_races = len(df.columns)
    width = 0.8 / n_races
    colors = ["#e6550d", "#3182bd", "#31a354", "#756bb1"]

    for i, (race, color) in enumerate(zip(df.columns, colors)):
        ax.bar(x + i * width, df[race], width, label=race, color=color, alpha=0.85)

    ax.set_xticks(x + width * (n_races - 1) / 2)
    ax.set_xticklabels([f.replace("_", " ") for f in df.index],
                       rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean relative abundance", fontsize=10)
    ax.set_title(f"Top {top_n} FAPROTAX functional groups across races", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("\n" + "═" * 60)
    print("  FAPROTAX Functional Analysis")
    print("═" * 60)

    # 1. Check FAPROTAX
    check_faprotax()

    # 2. Load data
    abund_tss, meta, manual_core, ae_core, ae_only = load_data()

    # 3. Run FAPROTAX on full pre-filtered table
    with tempfile.TemporaryDirectory() as tmp_dir:

        print("\n  Preparing FAPROTAX input …")
        input_tsv = prepare_faprotax_input(abund_tss, tmp_dir, label="full")

        out_tsv, report_txt = run_faprotax(input_tsv, tmp_dir)
        func_df = load_faprotax_output(out_tsv)
        print(f"  Functional groups detected: {len(func_df)}")

        # 4. Per-race profiles (full table)
        race_means = per_race_profiles(func_df, meta)

        # Diagnostic: how many groups have any non-zero signal?
        nonzero_groups = (race_means.max(axis=1) > 0).sum()
        print(f"  Functional groups with non-zero abundance: {nonzero_groups} / {len(race_means)}")
        if nonzero_groups > 0:
            print("  Top matched groups:")
            for grp, val in race_means.sum(axis=1).nlargest(10).items():
                print(f"    {val:.4f}  {grp}")

        # 5. Subset contributions
        manual_otu_ids = manual_core["OTU_ID"].unique().tolist()
        ae_otu_ids     = ae_core["OTU_ID"].unique().tolist()
        ae_only_ids    = ae_only["OTU_ID"].unique().tolist()

        print("\n  Computing subset contributions …")
        print(f"    Manual core : {len(manual_otu_ids)} OTUs")
        print(f"    AE core     : {len(ae_otu_ids)} OTUs")
        print(f"    AE-only     : {len(ae_only_ids)} OTUs")

        ae_race_means     = taxa_subset_contribution(abund_tss, func_df, ae_otu_ids,     meta, "ae_core",    tmp_dir)
        manual_race_means = taxa_subset_contribution(abund_tss, func_df, manual_otu_ids, meta, "manual_core",tmp_dir)
        ae_only_race_means= taxa_subset_contribution(abund_tss, func_df, ae_only_ids,    meta, "ae_only",    tmp_dir)

        # 6. Save CSVs
        out_full    = ROOT_DIR / "faprotax_functional_profiles.csv"
        out_race    = ROOT_DIR / "faprotax_race_means.csv"
        out_ae      = ROOT_DIR / "faprotax_ae_core_means.csv"
        out_manual  = ROOT_DIR / "faprotax_manual_core_means.csv"
        out_ae_only = ROOT_DIR / "faprotax_ae_only_means.csv"

        func_df.to_csv(out_full)
        race_means.to_csv(out_race)
        if not ae_race_means.empty:     ae_race_means.to_csv(out_ae)
        if not manual_race_means.empty: manual_race_means.to_csv(out_manual)
        if not ae_only_race_means.empty:ae_only_race_means.to_csv(out_ae_only)

        print(f"\n  Saved: {out_full.name}")
        print(f"  Saved: {out_race.name}")

        # 7. Read FAPROTAX report for coverage stats
        report_path = Path(report_txt)
        if report_path.exists():
            report_lines = report_path.read_text().splitlines()
            print("\n  ── FAPROTAX Coverage Report ──────────────────────")
            for line in report_lines[:20]:
                print(f"    {line}")

        # 8. Figures
        print("\n  Generating figures …")

        plot_race_heatmap(
            race_means,
            title="FAPROTAX functional groups across races (top 30)\nAll pre-filtered taxa",
            out_path=FIGURES_DIR / "fig_faprotax_race_heatmap.pdf",
            top_n=30,
        )

        plot_race_comparison_bar(
            race_means,
            out_path=FIGURES_DIR / "fig_faprotax_race_bars.pdf",
            top_n=20,
        )

        if not ae_race_means.empty and not manual_race_means.empty:
            plot_contribution_bar(
                full_race=race_means,
                ae_race=ae_race_means,
                manual_race=manual_race_means,
                out_path=FIGURES_DIR / "fig_faprotax_contribution.pdf",
            )

        if not ae_only_race_means.empty:
            plot_race_heatmap(
                ae_only_race_means,
                title="FAPROTAX functional groups — AE-only taxa (56)\n(structural core missed by manual method)",
                out_path=FIGURES_DIR / "fig_faprotax_ae_only_heatmap.pdf",
                top_n=20,
            )

    # 9. Summary
    print("\n  ── Summary ──────────────────────────────────────────")
    top5 = race_means.mean(axis=1).nlargest(5)
    print("  Top 5 functional groups (mean across all races):")
    for func, val in top5.items():
        print(f"    {val:.4f}  {func}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n✓  Done in {elapsed:.1f} min.\n")


if __name__ == "__main__":
    main()
