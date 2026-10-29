"""
comparator.py
─────────────
Parses the original hand-derived "core micrbiome.csv" and compares it
against the ML pipeline's output (core_microbiome_pipeline.csv).

The original file uses a non-standard multi-table layout: each bioproject
occupies a section with racial sub-groups arranged in parallel columns.
This module contains a dedicated parser that reconstructs a tidy (long-format)
DataFrame from that layout, enabling group-by-group comparison.

Comparison metrics per (bioproject, race):
  - n_manual    : taxa in the original file
  - n_ml        : taxa selected by ML (RFECV) in the pipeline
  - shared      : taxa in both
  - precision   : shared / n_ml  (how much of ML output is in the manual list)
  - recall      : shared / n_manual  (how much of the manual list ML captured)
  - f1          : harmonic mean of precision and recall
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path


# All bioproject IDs expected in the original file
_BIOPROJECTS = {
    "PRJNA388263", "PRJNA562529", "PRJNA395034", "PRJEB11419",
    "PRJDB10527",  "PRJDB9936",   "PRJNA400332", "PRJNA661289",
    "PRJDB7891",   "PRJDB5678",   "PRJNA601994", "PRJNA709129",
    "PRJNA486009", "PRJNA401981", "PRJDB4360",   "PRJNA386260",
}

# Normalise inconsistent race labels found in the original file
_RACE_NORMALISE = {
    "Afrcian":     "African",    # typo in original
    "Caucasians":  "Caucasian",
    "Caucasian ":  "Caucasian",
    "Asian ":      "Asian",
    "Thai":        "Asian",
    "Japanese":    "Asian",
}


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_original_core(path: Path) -> pd.DataFrame:
    """
    Parse the semi-structured "core micrbiome.csv" into a tidy DataFrame.

    Each row in the output has:
        bioproject, race, otu_id, genus, prevalence, median_abundance

    The parser operates as a simple state machine:
      State A – scanning for a section header (digit + bioproject ID)
      State B – scanning for a race header row
      State C – scanning for a column header row ("Taxa", "Genus" …)
      State D – reading data rows until a blank line resets to State B/A
    """
    records = []

    with open(path, newline="") as f:
        lines = list(csv.reader(f))

    current_bp  = None
    race_cols   = {}    # {column_index: normalised_race_label}
    in_data     = False

    _RACE_LABELS = set(_RACE_NORMALISE.keys()) | {
        "African", "Asian", "Caucasian", "Hispanic"
    }

    for line in lines:
        cells = [c.strip() for c in line]
        non_empty = [c for c in cells if c]

        # ── State A: look for section header ─────────────────────────────────
        if (len(non_empty) >= 2
                and non_empty[0].isdigit()
                and non_empty[1] in _BIOPROJECTS):
            current_bp = non_empty[1]
            race_cols  = {}
            in_data    = False
            continue

        if current_bp is None:
            continue

        # ── State B: look for race header ────────────────────────────────────
        if (any(c in _RACE_LABELS for c in cells)
                and "Taxa" not in cells
                and "Genus" not in cells):
            race_cols = {}
            for ci, cell in enumerate(cells):
                if cell in _RACE_LABELS:
                    normalised = _RACE_NORMALISE.get(cell, cell)
                    race_cols[ci] = normalised
            in_data = False
            continue

        # ── State C: column header row ────────────────────────────────────────
        if "Taxa" in cells and "Genus" in cells:
            in_data = True
            continue

        if not in_data:
            continue

        # ── State D: data rows ────────────────────────────────────────────────
        # Each race block occupies 4 columns starting at its race_cols anchor:
        #   anchor+0: OTU label ("OTU 41")
        #   anchor+1: Genus name
        #   anchor+2: Prevalence
        #   anchor+3: Median Abundance
        for start_col, race_label in sorted(race_cols.items()):
            try:
                otu_raw  = cells[start_col]
                genus    = cells[start_col + 1] if start_col + 1 < len(cells) else ""
                prev_str = cells[start_col + 2] if start_col + 2 < len(cells) else ""
                med_str  = cells[start_col + 3] if start_col + 3 < len(cells) else ""
            except IndexError:
                continue

            if not otu_raw.startswith("OTU"):
                continue

            try:
                otu_id = int(otu_raw.split()[1])
                prev   = float(prev_str)
                med    = float(med_str)
            except (ValueError, IndexError):
                continue

            records.append({
                "bioproject":       current_bp,
                "race":             race_label,
                "otu_id":           otu_id,
                "genus":            genus,
                "prevalence":       round(prev, 9),
                "median_abundance": round(med,  9),
            })

    return pd.DataFrame(records)


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare(
    pipeline_df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Compare ML core microbiome output with the original manual analysis.

    Parameters
    ----------
    pipeline_df : Output of the ML pipeline (from main.py).  Must contain
                  columns: bioproject, race, otu_id, genus, rfecv_selected.
    original_df : Parsed original core microbiome (from parse_original_core).

    Returns
    -------
    summary_df  : One row per (bioproject, race) with overlap metrics.
    report_str  : Formatted text report for printing / writing to file.
    """
    # Keep only ML-selected taxa
    ml_core = pipeline_df[pipeline_df["rfecv_selected"] == True].copy()

    pipe_groups = {k: g for k, g in ml_core.groupby(["bioproject", "race"])}
    orig_groups = {k: g for k, g in original_df.groupby(["bioproject", "race"])}

    rows   = []
    lines  = []
    lines += [
        "=" * 70,
        "  CORE MICROBIOME: ML PIPELINE vs ORIGINAL MANUAL ANALYSIS",
        "=" * 70,
    ]

    all_keys = sorted(set(pipe_groups) | set(orig_groups))

    for bp, race in all_keys:
        gk = (bp, race)
        pipe = pipe_groups.get(gk)
        orig = orig_groups.get(gk)

        # Groups that exist only in the original (e.g., combined Caucasian+Asian)
        # have no ML counterpart — they are noted but skipped in metrics.
        if pipe is None:
            lines.append(f"\n  [NOT IN ML OUTPUT] {bp} / {race}")
            continue
        if orig is None:
            lines.append(f"\n  [NOT IN ORIGINAL ] {bp} / {race}  "
                         f"(ML found {len(pipe)} taxa)")
            continue

        pipe_otus = set(pipe["otu_id"])
        orig_otus = set(orig["otu_id"])
        shared    = pipe_otus & orig_otus
        only_ml   = pipe_otus - orig_otus
        only_man  = orig_otus - pipe_otus

        n_ml  = len(pipe_otus)
        n_man = len(orig_otus)
        n_sh  = len(shared)

        precision = n_sh / n_ml  if n_ml  > 0 else 0.0
        recall    = n_sh / n_man if n_man > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        rows.append({
            "bioproject": bp, "race": race,
            "n_manual": n_man, "n_ml": n_ml, "n_shared": n_sh,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
        })

        status = "EXACT" if pipe_otus == orig_otus else "PARTIAL"
        lines += [
            f"\n  {'─'*60}",
            f"  {bp} / {race}  [{status}]",
            f"    Manual core taxa : {n_man}",
            f"    ML core taxa     : {n_ml}",
            f"    Shared           : {n_sh}",
            f"    Precision        : {precision:.3f}  "
            f"(shared / ML-selected)",
            f"    Recall           : {recall:.3f}  "
            f"(shared / manual)",
            f"    F1               : {f1:.3f}",
        ]

        if only_ml:
            genera = sorted(pipe[pipe["otu_id"].isin(only_ml)]["genus"])
            lines.append(f"    Only in ML       : {sorted(only_ml)} → {genera}")
        if only_man:
            genera = sorted(orig[orig["otu_id"].isin(only_man)]["genus"])
            lines.append(f"    Only in manual   : {sorted(only_man)} → {genera}")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(rows)

    if len(summary_df):
        mean_f1       = summary_df["f1"].mean()
        mean_precision= summary_df["precision"].mean()
        mean_recall   = summary_df["recall"].mean()
        exact         = (summary_df["n_shared"] == summary_df["n_manual"]).sum()
        n_groups      = len(summary_df)

        lines += [
            f"\n{'='*70}",
            "  AGGREGATE SUMMARY",
            f"  {'─'*68}",
            f"  Groups compared    : {n_groups}",
            f"  Exact matches      : {exact}",
            f"  Mean Precision     : {mean_precision:.3f}",
            f"  Mean Recall        : {mean_recall:.3f}",
            f"  Mean F1            : {mean_f1:.3f}",
            "=" * 70,
        ]

    return summary_df, "\n".join(lines)
