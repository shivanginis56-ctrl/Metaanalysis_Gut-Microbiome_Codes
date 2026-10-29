"""
threshold_comparison.py
───────────────────────
Builds threshold_comparison.xlsx with 5 sheets:

  Sheet 0 — Summary       : overall stats for all 4 thresholds in one table
  Sheet 1 — Score > 0     : full comparison vs manual (colour-coded)
  Sheet 2 — Score > mean  : full comparison vs manual
  Sheet 3 — 90% mass      : full comparison vs manual
  Sheet 4 — 95% mass      : full comparison vs manual

Colour coding (matching core_comparison.xlsx):
  Green  (#70AD47) — in BOTH AE and Manual for that race
  Grey   (#A6A6A6) — in Manual for race X, AE has it for a DIFFERENT race
  Yellow (#FFD966) — AE only (not in manual at all)
  Red    (#FF4C4C) — in Manual but AE missed it in ALL races for this threshold

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.threshold_comparison
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import ROOT_DIR

RACES = ["African", "Asian", "Caucasian", "Hispanic"]

# Colours
GREEN  = PatternFill("solid", fgColor="70AD47")
GREY   = PatternFill("solid", fgColor="A6A6A6")
YELLOW = PatternFill("solid", fgColor="FFD966")
RED    = PatternFill("solid", fgColor="FF4C4C")
HEADER = PatternFill("solid", fgColor="2F4F4F")
SUBHDR = PatternFill("solid", fgColor="4A7C8E")

WHITE_FONT  = Font(bold=True, color="FFFFFF")
BOLD_FONT   = Font(bold=True)
THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin"),
)


def auto_width(ws):
    for col in ws.columns:
        max_len = max((len(str(c.value)) if c.value else 0) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 3, 40)


def compute_ae_core(ae_df, race, threshold_name, threshold_fn=None, n_taxa=None):
    """Return set of OTU_IDs for a race under the given threshold."""
    sub    = ae_df[ae_df["race"] == race].copy()
    scores = sub.set_index("otu_id")["importance_score"]
    pos    = scores[scores > 0]

    if threshold_name == "score_gt_0":
        return set(pos.index.tolist())
    elif threshold_name == "score_gt_mean":
        return set(scores[scores > pos.mean()].index.tolist())
    elif threshold_name == "mass_90":
        total   = pos.sum()
        sorted_s = pos.sort_values(ascending=False)
        cumsum  = sorted_s.cumsum()
        n       = int(np.searchsorted(cumsum.values, total * 0.90)) + 1
        return set(sorted_s.head(n).index.tolist())
    elif threshold_name == "mass_95":
        total   = pos.sum()
        sorted_s = pos.sort_values(ascending=False)
        cumsum  = sorted_s.cumsum()
        n       = int(np.searchsorted(cumsum.values, total * 0.95)) + 1
        return set(sorted_s.head(n).index.tolist())


def build_comparison_df(ae_df, manual_per_race, threshold_name):
    """Build tidy comparison DataFrame for one threshold."""
    ae_per_race = {r: compute_ae_core(ae_df, r, threshold_name) for r in RACES}

    # All taxa appearing in either AE or manual
    all_ids = set()
    for r in RACES:
        all_ids |= ae_per_race[r]
        all_ids |= manual_per_race.get(r, set())

    # OTU_ID → genus name
    otu_genus = ae_df.drop_duplicates("otu_id").set_index("otu_id")["genus"].to_dict()

    rows = []
    for otu in sorted(all_ids):
        row = {"OTU_ID": otu, "Genus": otu_genus.get(otu, "")}
        for r in RACES:
            row[f"AE_{r}"]     = "✓" if otu in ae_per_race[r]              else ""
            row[f"Manual_{r}"] = "✓" if otu in manual_per_race.get(r, set()) else ""

        in_ae_any     = any(otu in ae_per_race[r] for r in RACES)
        in_manual_any = any(otu in manual_per_race.get(r, set()) for r in RACES)

        if in_ae_any and in_manual_any:
            row["Category"] = "Confirmed Core"
        elif in_ae_any and not in_manual_any:
            row["Category"] = "AE Only"
        elif not in_ae_any and in_manual_any:
            row["Category"] = "Missed by AE"
        else:
            row["Category"] = "Other"

        rows.append(row)

    return pd.DataFrame(rows)


def row_fill(category, ae_race_cols, manual_race_cols, row_data):
    """Per-column fill for a comparison row."""
    fills = {}
    ae_any     = any(row_data.get(c, "") == "✓" for c in ae_race_cols)
    man_any    = any(row_data.get(c, "") == "✓" for c in manual_race_cols)
    man_set    = {c.replace("Manual_","") for c in manual_race_cols if row_data.get(c,"")=="✓"}
    ae_set     = {c.replace("AE_","")     for c in ae_race_cols     if row_data.get(c,"")=="✓"}

    if category == "Confirmed Core":
        fill = GREEN
    elif category == "AE Only":
        fill = YELLOW
    elif category == "Missed by AE":
        fill = RED
    else:
        fill = GREY
    return fill


def write_comparison_sheet(ws, df, title):
    """Write a colour-coded comparison table to a worksheet."""
    AE_COLS  = [f"AE_{r}"     for r in RACES]
    MAN_COLS = [f"Manual_{r}" for r in RACES]
    COLS = ["OTU_ID", "Genus"] + AE_COLS + MAN_COLS + ["Category"]

    # Title row
    ws.append([title])
    ws.cell(ws.max_row, 1).font = Font(bold=True, size=13)
    ws.append([])

    # Legend
    ws.append(["LEGEND:", "", "Green = Confirmed Core (both AE & Manual)",
               "", "Yellow = AE Only", "", "Red = Missed by AE",
               "", "Grey = Manual only / cross-race"])
    for i, fill in [(1, None),(3, GREEN),(5, YELLOW),(7, RED),(9, GREY)]:
        c = ws.cell(ws.max_row, i)
        if fill:
            c.fill = fill
    ws.append([])

    # Summary counts for this sheet
    n_confirmed = (df["Category"] == "Confirmed Core").sum()
    n_ae_only   = (df["Category"] == "AE Only").sum()
    n_missed    = (df["Category"] == "Missed by AE").sum()
    n_manual    = sum(len(v) for v in []) # computed below

    # Unique IDs
    ae_all  = set()
    man_all = set()
    for r in RACES:
        ae_all  |= set(df[df[f"AE_{r}"]=="✓"]["OTU_ID"])
        man_all |= set(df[df[f"Manual_{r}"]=="✓"]["OTU_ID"])

    ws.append([f"AE core: {len(ae_all)} unique taxa   |   "
               f"Manual core: {len(man_all)} unique taxa   |   "
               f"Confirmed (both): {n_confirmed}   |   "
               f"AE Only: {n_ae_only}   |   "
               f"Missed: {n_missed}"])
    ws.cell(ws.max_row, 1).font = BOLD_FONT
    ws.append([])

    # Header row
    ws.append(COLS)
    hdr_row = ws.max_row
    for i, col in enumerate(COLS, 1):
        cell = ws.cell(hdr_row, i)
        cell.fill   = HEADER
        cell.font   = WHITE_FONT
        cell.alignment = Alignment(horizontal="center")

    # Data rows
    for _, row in df[COLS].iterrows():
        ws.append(list(row))
        data_row = ws.max_row
        fill = row_fill(row["Category"],
                        [c for c in AE_COLS],
                        [c for c in MAN_COLS],
                        row.to_dict())
        for i in range(1, len(COLS) + 1):
            cell = ws.cell(data_row, i)
            cell.fill   = fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = THIN_BORDER

    auto_width(ws)


def write_summary_sheet(ws, ae_df, manual_per_race):
    """Write cross-threshold summary."""
    THRESHOLDS = {
        "Score > 0":       "score_gt_0",
        "Score > mean":    "score_gt_mean",
        "90% score mass":  "mass_90",
        "95% score mass":  "mass_95",
    }

    ws.append(["Threshold Comparison — AE Core vs Manual Core (96 taxa)"])
    ws.cell(1, 1).font = Font(bold=True, size=14)
    ws.append([])

    # Per-threshold × per-race table
    ws.append(["Threshold", "Race", "AE Core Size", "Manual Size",
               "Overlap", "Precision", "Recall", "F1",
               "Total AE Unique", "Overlap with Manual 96"])
    hdr = ws.max_row
    for i in range(1, 11):
        ws.cell(hdr, i).fill = HEADER
        ws.cell(hdr, i).font = WHITE_FONT
        ws.cell(hdr, i).alignment = Alignment(horizontal="center")

    man_all_ids = set()
    for r in RACES:
        man_all_ids |= manual_per_race.get(r, set())

    for tname, tkey in THRESHOLDS.items():
        ae_per_race = {r: compute_ae_core(ae_df, r, tkey) for r in RACES}
        ae_all_ids  = set()
        for r in RACES:
            ae_all_ids |= ae_per_race[r]

        overlap_96 = len(ae_all_ids & man_all_ids)

        for r in RACES:
            ae_ids  = ae_per_race[r]
            man_ids = manual_per_race.get(r, set())
            overlap = ae_ids & man_ids
            p  = len(overlap)/len(ae_ids)  if ae_ids  else 0
            rc = len(overlap)/len(man_ids) if man_ids else 0
            f1 = 2*p*rc/(p+rc) if (p+rc) else 0

            ws.append([
                tname, r,
                len(ae_ids), len(man_ids), len(overlap),
                round(p, 3), round(rc, 3), round(f1, 3),
                len(ae_all_ids), overlap_96,
            ])

        ws.append([])

    # Overall summary block
    ws.append(["OVERALL UNIQUE CORE COMPARISON"])
    ws.cell(ws.max_row, 1).font = Font(bold=True, size=12)
    ws.append(["Threshold", "AE Unique Taxa (union of all races)",
               "Overlap with Manual 96", "% of Manual 96 Captured",
               "AE-Only (not in manual 96)", "Missed (in manual 96, not in AE)"])
    for i in range(1, 7):
        ws.cell(ws.max_row, i).fill = SUBHDR
        ws.cell(ws.max_row, i).font = WHITE_FONT

    for tname, tkey in THRESHOLDS.items():
        ae_per_race = {r: compute_ae_core(ae_df, r, tkey) for r in RACES}
        ae_all      = set()
        for r in RACES:
            ae_all |= ae_per_race[r]
        overlap  = len(ae_all & man_all_ids)
        ae_only  = len(ae_all - man_all_ids)
        missed   = len(man_all_ids - ae_all)
        ws.append([tname, len(ae_all), overlap,
                   f"{overlap/len(man_all_ids)*100:.1f}%",
                   ae_only, missed])

    auto_width(ws)


def main():
    ae_df    = pd.read_csv(ROOT_DIR / "ae_emerged_core.csv")
    step3    = pd.read_csv(ROOT_DIR / "step3_final_core.csv")
    manual_per_race = step3.groupby("Race")["OTU_ID"].apply(set).to_dict()

    THRESHOLDS = [
        ("Score > 0",      "score_gt_0"),
        ("Score > mean",   "score_gt_mean"),
        ("90% score mass", "mass_90"),
        ("95% score mass", "mass_95"),
    ]

    wb = Workbook()
    wb.remove(wb.active)   # remove default sheet

    # Summary sheet first
    ws_sum = wb.create_sheet("Summary")
    write_summary_sheet(ws_sum, ae_df, manual_per_race)
    print("  Built Summary sheet")

    # One sheet per threshold
    for tname, tkey in THRESHOLDS:
        print(f"  Building '{tname}' sheet …", flush=True)
        df = build_comparison_df(ae_df, manual_per_race, tkey)
        ws = wb.create_sheet(tname)
        write_comparison_sheet(ws, df, f"AE vs Manual Core  |  Threshold: {tname}")

    out = ROOT_DIR / "threshold_comparison.xlsx"
    wb.save(out)
    print(f"\n  Saved {out.name}")

    # Print quick stats
    man_all = set()
    for r in RACES:
        man_all |= manual_per_race.get(r, set())

    print(f"\n  Quick stats — overlap with all 96 manual taxa:\n")
    print(f"  {'Threshold':<22} {'AE Unique':>10} {'Overlap/96':>12} {'% Captured':>12} {'Missed':>8}")
    print(f"  {'-'*68}")
    for tname, tkey in THRESHOLDS:
        ae_per_race = {r: compute_ae_core(ae_df, r, tkey) for r in RACES}
        ae_all = set()
        for r in RACES:
            ae_all |= ae_per_race[r]
        ov = len(ae_all & man_all)
        print(f"  {tname:<22} {len(ae_all):>10} {ov:>10}/96 {ov/96*100:>11.1f}% {96-ov:>8}")


if __name__ == "__main__":
    main()
