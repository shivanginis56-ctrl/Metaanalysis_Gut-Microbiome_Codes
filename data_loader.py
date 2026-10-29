"""
data_loader.py
──────────────
Responsible for:
  1. Loading metadata_clean.csv → sample-level metadata dictionary.
  2. Loading Taxonomic_abundance.csv → full taxa × sample count matrix.
  3. Grouping samples by (bioproject, race) for downstream analysis.
  4. Pre-filtering globally rare taxa to reduce feature dimensionality.

All I/O is isolated here; no ML or statistics are performed in this module.
"""

import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from pipeline.config import GLOBAL_PREVALENCE_MIN


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_metadata(path: Path) -> dict[str, dict]:
    """
    Parse metadata_clean.csv.

    Returns
    -------
    dict
        {sample_id: {"bioproject": str, "race": str}}
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    meta = {r["sample_id"]: {"bioproject": r["bioproject"], "race": r["race"]}
            for r in rows}
    return meta


def group_samples(meta: dict) -> dict[tuple[str, str], list[str]]:
    """
    Partition sample IDs by their (bioproject, race) combination.

    Returns
    -------
    dict
        {(bioproject, race): [sample_id, ...]}
    """
    groups: dict[tuple, list] = defaultdict(list)
    for sid, info in meta.items():
        groups[(info["bioproject"], info["race"])].append(sid)
    return dict(groups)


# ── Abundance matrix ───────────────────────────────────────────────────────────

def load_abundance(path: Path) -> tuple[list[str], list[int], pd.DataFrame]:
    """
    Load the full Taxonomic_abundance.csv into memory.

    The CSV layout is:
        - Column 0 ("Taxon")  → taxon name used as the DataFrame index.
        - Remaining columns   → one column per sample_id, values are raw read counts.

    OTU IDs are assigned as the 1-based row position in the CSV file
    (matching the numbering in core micrbiome.csv).

    Returns
    -------
    taxa_names : list[str]
        Ordered list of genus/group names (length = n_taxa).
    otu_ids : list[int]
        1-based OTU number for each taxon.
    df : pd.DataFrame
        Shape (n_taxa, n_samples). Index = taxon name, columns = sample_id.
    """
    print("    Reading CSV … (large file, may take ~30s)", flush=True)
    df = pd.read_csv(path, index_col=0)

    taxa_names = list(df.index)
    otu_ids    = list(range(1, len(taxa_names) + 1))

    print(f"    Loaded: {len(taxa_names):,} taxa × {df.shape[1]:,} samples", flush=True)
    return taxa_names, otu_ids, df


def filter_rare_taxa(
    df: pd.DataFrame,
    min_prevalence: float = GLOBAL_PREVALENCE_MIN,
) -> pd.DataFrame:
    """
    Drop taxa that appear in fewer than `min_prevalence` fraction of samples.

    This step runs once globally and dramatically reduces the feature space
    before per-group ML analysis.

    Parameters
    ----------
    df : DataFrame (taxa × samples, raw counts)
    min_prevalence : float
        Fraction of samples in which a taxon must appear (count > 0) to be retained.

    Returns
    -------
    DataFrame with rare taxa removed.
    """
    n_samples  = df.shape[1]
    prevalence = (df > 0).sum(axis=1) / n_samples
    keep_mask  = prevalence >= min_prevalence
    n_kept     = keep_mask.sum()

    print(f"    Global pre-filter (prevalence ≥ {min_prevalence:.0%}): "
          f"{n_kept:,} / {len(prevalence):,} taxa retained", flush=True)

    return df.loc[keep_mask]


# ── Convenience summary ────────────────────────────────────────────────────────

def print_group_summary(groups: dict, meta: dict) -> None:
    """Print a formatted table of samples per bioproject × race group."""
    from collections import Counter

    bp_race: dict[str, Counter] = defaultdict(Counter)
    for sid, info in meta.items():
        bp_race[info["bioproject"]][info["race"]] += 1

    print(f"\n  {'Bioproject':<18} {'Race':<12} {'n_samples':>10}")
    print(f"  {'─'*18} {'─'*12} {'─'*10}")
    for bp in sorted(bp_race):
        for race, n in sorted(bp_race[bp].items()):
            print(f"  {bp:<18} {race:<12} {n:>10,}")
    print()
