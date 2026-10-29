"""
build_final_xlsx.py
Compiles ALL final results into a single comprehensive xlsx.
Sheets: Overview | Method_Comparison | AE_Core_70 | AE_Core_119 |
        Robustness | RF_Multiclass | RF_Importances | MaAsLin2_Differential |
        KEGG_AE_Manual | KEGG_Differential | FAPROTAX_Summary | Predicted_Pathways
"""
import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                              numbers as opnumbers)
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

ROOT = Path(".")

# ── Style helpers ─────────────────────────────────────────────────────────────
HDR_FILL  = PatternFill("solid", fgColor="1F3864")
SUB_FILL  = PatternFill("solid", fgColor="2E75B6")
BEST_FILL = PatternFill("solid", fgColor="E2EFDA")
ALT_FILL  = PatternFill("solid", fgColor="EBF3FB")
WHT_FONT  = Font(bold=True, color="FFFFFF", size=10)
BLD_FONT  = Font(bold=True, size=10)
NRM_FONT  = Font(size=9)
THIN      = Border(left=Side(style="thin"),  right=Side(style="thin"),
                   top=Side(style="thin"),   bottom=Side(style="thin"))

def hdr(ws, row, col, text, width=None):
    c = ws.cell(row=row, column=col, value=text)
    c.fill = HDR_FILL; c.font = WHT_FONT; c.border = THIN
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    if width:
        ws.column_dimensions[get_column_letter(col)].width = width

def subhdr(ws, row, col, text):
    c = ws.cell(row=row, column=col, value=text)
    c.fill = SUB_FILL; c.font = WHT_FONT; c.border = THIN
    c.alignment = Alignment(horizontal="center", vertical="center")

def val(ws, row, col, v, bold=False, fill=None, fmt=None, align="left"):
    c = ws.cell(row=row, column=col, value=v)
    c.font = BLD_FONT if bold else NRM_FONT
    c.border = THIN
    c.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
    if fill:  c.fill = fill
    if fmt:   c.number_format = fmt

def auto_width(ws, max_w=50):
    for col in ws.columns:
        max_len = max(
            (len(str(c.value)) if c.value is not None else 0) for c in col
        )
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 3, max_w)

def write_df(ws, df, start_row=1, start_col=1):
    for j, col_name in enumerate(df.columns):
        hdr(ws, start_row, start_col + j, col_name)
    for i, row_data in enumerate(df.itertuples(index=False), start=start_row + 1):
        fill = ALT_FILL if i % 2 == 0 else None
        for j, v in enumerate(row_data):
            val(ws, i, start_col + j, v if pd.notna(v) else "", fill=fill)

# ── Load data ─────────────────────────────────────────────────────────────────
# AE core
tc = pd.ExcelFile("threshold_comparison.xlsx")
sm_raw = pd.read_excel("threshold_comparison.xlsx", sheet_name="Score > mean", header=None)
header_row = next(i for i, r in sm_raw.iterrows() if str(r[0]).strip() == 'OTU_ID')
ae70 = sm_raw.iloc[header_row+1:].copy()
ae70.columns = list(sm_raw.iloc[header_row])
ae70 = ae70[pd.to_numeric(ae70['OTU_ID'], errors='coerce').notna()].copy()
ae70['OTU_ID'] = ae70['OTU_ID'].astype(int)

m95_raw = pd.read_excel("threshold_comparison.xlsx", sheet_name="95% score mass", header=None)
header_row95 = next(i for i, r in m95_raw.iterrows() if str(r[0]).strip() == 'OTU_ID')
ae119 = m95_raw.iloc[header_row95+1:].copy()
ae119.columns = list(m95_raw.iloc[header_row95])
ae119 = ae119[pd.to_numeric(ae119['OTU_ID'], errors='coerce').notna()].copy()
ae119['OTU_ID'] = ae119['OTU_ID'].astype(int)

robustness  = pd.read_csv("robustness_results.csv")
rf_imp      = pd.read_csv("rf_multiclass_importances.csv")
kegg_cache  = pd.read_csv("kegg_expanded_cache.csv")
kegg_cache["N_KOs"]      = pd.to_numeric(kegg_cache["N_KOs"],      errors="coerce").fillna(0)
kegg_cache["N_Pathways"] = pd.to_numeric(kegg_cache["N_Pathways"], errors="coerce").fillna(0)
aldex       = pd.read_csv("results_long_random_effect_all_taxa_effect_cutoff.csv")
sig_aldex   = aldex[(aldex['q_value']<0.05)&(aldex['abs_effect']>=1.0)].copy()
fap_ae119   = pd.read_csv("faprotax_ae119_means.csv")
fap_ae70    = pd.read_csv("faprotax_ae70_means.csv")
fap_diff    = pd.read_csv("faprotax_diff74_means.csv")
race_pw     = pd.read_csv("kegg_race_pathway_means.csv", index_col=0)

diff_genera = list(sig_aldex['taxon'].unique())
kegg_diff_df = kegg_cache[kegg_cache['Genus'].isin(diff_genera)].copy()

# RF report text
with open("rf_multiclass_report.txt") as f:
    rf_text = f.read()

# Pathway names
pw_names = {}
try:
    pw_df = pd.read_csv("kegg_pathway_list_cache.csv")
    pw_names = dict(zip(pw_df["id"], pw_df["name"]))
except: pass

# ── Build workbook ────────────────────────────────────────────────────────────
wb = Workbook()
wb.remove(wb.active)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════
ws = wb.create_sheet("Overview")
ws.row_dimensions[1].height = 28

overview_rows = [
    ("DATASET", ""),
    ("Samples", "7,302"),
    ("Racial groups", "4 (Caucasian n=4,902 | Asian n=1,951 | African n=317 | Hispanic n=132)"),
    ("Bioprojects", "16"),
    ("Total taxa", "4,680"),
    ("Pre-filtered taxa (≥5% prevalence)", "238"),
    ("Manual core reference", "96 taxa (prevalence ≥50% AND median rel. abund ≥0.001)"),
    ("", ""),
    ("KEY RESULTS — SEED=42 (REPRODUCIBLE)", ""),
    ("AE Weighted MSE Score>mean", "70 taxa | Precision=1.000 | Recall=0.729 | F1=0.843  ★ BEST"),
    ("AE Weighted MSE 95% mass", "119 taxa | Precision=0.739 | Recall=0.917 | F1=0.819"),
    ("RF Multiclass (4-class)", "OOB Accuracy=0.907 | CV Accuracy=0.904±0.006 | F1 macro=0.625±0.010"),
    ("RF Binary RFECV", "148 taxa | Precision=0.568 | Recall=0.875 | F1=0.689"),
    ("MaAsLin2 Differential", "74 unique taxa (q<0.05 AND |effect|≥1.0, across 6 pairwise comparisons)"),
    ("", ""),
    ("ROBUSTNESS (100 runs, 95% mass threshold)", ""),
    ("F1", f"{robustness['f1'].mean():.3f} ± {robustness['f1'].std():.3f}  (range {robustness['f1'].min():.3f}–{robustness['f1'].max():.3f})"),
    ("Precision", f"{robustness['precision'].mean():.3f} ± {robustness['precision'].std():.3f}"),
    ("Recall", f"{robustness['recall'].mean():.3f} ± {robustness['recall'].std():.3f}"),
    ("n_selected (95% mass)", f"{robustness['n_selected'].mean():.1f} ± {robustness['n_selected'].std():.1f}  (range {robustness['n_selected'].min()}–{robustness['n_selected'].max()})"),
    ("", ""),
    ("FUNCTIONAL ANNOTATION", ""),
    ("KEGG coverage — AE-119 (95% mass, seed=42)", "69/119 matched (58%) | mean 1,390 KOs | 120 pathways per taxon"),
    ("KEGG coverage — AE-70  (Score>mean, seed=42)", "41/70 matched (59%) | mean 1,329 KOs | 119 pathways per taxon"),
    ("KEGG coverage — Differential (74 taxa)", "32/74 matched (43%) | mean 1,389 KOs | 124 pathways per taxon"),
    ("Predicted KEGG pathways", "392 non-zero pathways predicted across 7,302 samples"),
    ("FAPROTAX non-zero groups", "9 for AE core | 9 for Manual core | 8 for Differential"),
]

ws.column_dimensions['A'].width = 42
ws.column_dimensions['B'].width = 75
for r_idx, (k, v) in enumerate(overview_rows, start=1):
    if v == "" and k == "":
        continue
    is_section = v == ""
    ck = ws.cell(row=r_idx, column=1, value=k)
    cv = ws.cell(row=r_idx, column=2, value=v)
    if is_section:
        ck.fill = HDR_FILL; ck.font = WHT_FONT
        ws.merge_cells(start_row=r_idx, start_column=1, end_row=r_idx, end_column=2)
        ws.cell(row=r_idx, column=1).alignment = Alignment(horizontal="left", vertical="center")
    elif k in ("DATASET", "KEY RESULTS — SEED=42 (REPRODUCIBLE)",
               "ROBUSTNESS (100 runs, 95% mass threshold)", "FUNCTIONAL ANNOTATION"):
        ck.fill = SUB_FILL; ck.font = WHT_FONT
        cv.fill = SUB_FILL; cv.font = WHT_FONT
        ws.merge_cells(start_row=r_idx, start_column=1, end_row=r_idx, end_column=2)
        ws.cell(row=r_idx, column=1).alignment = Alignment(horizontal="left", vertical="center")
    else:
        ck.font = BLD_FONT; ck.border = THIN
        cv.font = NRM_FONT; cv.border = THIN
        ck.alignment = Alignment(horizontal="left", vertical="center")
        cv.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        if "BEST" in v:
            ck.fill = BEST_FILL; cv.fill = BEST_FILL

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 2: Method_Comparison
# ═══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Method_Comparison")
methods = [
    ["AE Weighted MSE (Score>mean)", 70,  70, 1.000, 0.729, 0.843, "★ Best precision — seed=42"],
    ["AE Weighted MSE (95% mass)",   119, 88, 0.739, 0.917, 0.819, "Best recall among AE"],
    ["AE Baseline (95% mass)",       133, 91, 0.684, 0.948, 0.795, ""],
    ["RF Multiclass (Top-96 MDI)",    96, 72, 0.750, 0.750, 0.750, "Discriminative"],
    ["RF Binary RFECV (union)",      148, 84, 0.568, 0.875, 0.689, "Race-discriminative"],
]
mdf = pd.DataFrame(methods, columns=["Method","Taxa selected","Overlap w/ manual 96",
                                      "Precision","Recall","F1","Notes"])
write_df(ws2, mdf, start_row=2)
ws2.cell(row=1, column=1, value="All methods benchmarked against 96-taxon manual reference core  |  seed = 42")
ws2.cell(row=1, column=1).font = BLD_FONT
ws2.cell(row=1, column=1).fill = SUB_FILL
ws2.cell(row=1, column=1).font = WHT_FONT
ws2.merge_cells(start_row=1, start_column=1, end_row=1, end_column=7)
# Highlight best row
for col in range(1, 8):
    ws2.cell(row=3, column=col).fill = BEST_FILL
auto_width(ws2)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 3: AE_Core_70  (Score>mean, seed=42 — final core)
# ═══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("AE_Core_70")
ws3.cell(row=1, column=1,
         value="AE-Derived Core — Score>mean threshold — seed=42 — 70 taxa (all confirmed by manual reference)")
ws3.cell(row=1, column=1).font = WHT_FONT
ws3.cell(row=1, column=1).fill = HDR_FILL
ws3.merge_cells(start_row=1, start_column=1, end_row=1, end_column=11)

ae70_out = ae70[['OTU_ID','Genus','AE_African','AE_Asian','AE_Caucasian','AE_Hispanic',
                  'Manual_African','Manual_Asian','Manual_Caucasian','Manual_Hispanic','Category']].copy()
ae70_out = ae70_out.rename(columns={
    'AE_African':'AE African','AE_Asian':'AE Asian','AE_Caucasian':'AE Caucasian',
    'AE_Hispanic':'AE Hispanic','Manual_African':'Manual African','Manual_Asian':'Manual Asian',
    'Manual_Caucasian':'Manual Caucasian','Manual_Hispanic':'Manual Hispanic'
})
write_df(ws3, ae70_out, start_row=2)
auto_width(ws3)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 4: AE_Core_119  (95% mass, seed=42)
# ═══════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("AE_Core_119")
ws4.cell(row=1, column=1,
         value="AE-Derived Core — 95% score mass threshold — seed=42 — 119 taxa (88 confirmed / 31 AE-only)")
ws4.cell(row=1, column=1).font = WHT_FONT
ws4.cell(row=1, column=1).fill = HDR_FILL
ws4.merge_cells(start_row=1, start_column=1, end_row=1, end_column=11)

ae119_out = ae119[['OTU_ID','Genus','AE_African','AE_Asian','AE_Caucasian','AE_Hispanic',
                    'Manual_African','Manual_Asian','Manual_Caucasian','Manual_Hispanic','Category']].copy()
ae119_out = ae119_out.rename(columns={
    'AE_African':'AE African','AE_Asian':'AE Asian','AE_Caucasian':'AE Caucasian',
    'AE_Hispanic':'AE Hispanic','Manual_African':'Manual African','Manual_Asian':'Manual Asian',
    'Manual_Caucasian':'Manual Caucasian','Manual_Hispanic':'Manual Hispanic'
})
write_df(ws4, ae119_out, start_row=2)
auto_width(ws4)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 5: Robustness_100runs
# ═══════════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("Robustness_100runs")
ws5.cell(row=1, column=1,
         value=f"AE Robustness — 100 independent seeds — 95% mass threshold vs Manual 96  |  "
               f"Mean F1={robustness['f1'].mean():.3f}±{robustness['f1'].std():.3f}")
ws5.cell(row=1, column=1).font = WHT_FONT
ws5.cell(row=1, column=1).fill = HDR_FILL
ws5.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)

rob_out = robustness[['seed','n_selected','precision','recall','f1','ap','macro_f1','elapsed_s']].copy()
rob_out.columns = ['Seed','N taxa selected','Precision','Recall','F1','Avg Precision','Macro F1','Time (s)']
write_df(ws5, rob_out, start_row=2)
# Highlight seed=42
for col in range(1, 9):
    ws5.cell(row=44+2, column=col).fill = BEST_FILL
auto_width(ws5)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 6: RF_Multiclass
# ═══════════════════════════════════════════════════════════════════════════════
ws6 = wb.create_sheet("RF_Multiclass")
rf_summary = [
    ("CV Accuracy",        "0.904 ± 0.006"),
    ("OOB Accuracy",       "0.907"),
    ("CV F1 macro",        "0.625 ± 0.010"),
    ("n_estimators",       "500"),
    ("n_features",         "238 (pre-filtered)"),
    ("n_samples",          "7,302"),
    ("class_weight",       "balanced"),
    ("Interpretation",     "Race is highly predictable from microbiome composition (~90% accuracy)"),
]
ws6.cell(row=1,column=1,value="Random Forest Multiclass — Race Prediction").font = WHT_FONT
ws6.cell(row=1,column=1).fill = HDR_FILL
ws6.merge_cells(start_row=1,start_column=1,end_row=1,end_column=2)
for r_idx,(k,v) in enumerate(rf_summary, start=2):
    ws6.cell(row=r_idx,column=1,value=k).font = BLD_FONT
    ws6.cell(row=r_idx,column=2,value=v).font = NRM_FONT

r_off = len(rf_summary)+4
ws6.cell(row=r_off,column=1,value="Top-50 feature importances (MDI) — sorted by importance").font = BLD_FONT
top50 = rf_imp.head(50)[['rank','otu_id','genus','mdi_importance']].copy()
top50.columns = ['Rank','OTU_ID','Genus','MDI Importance']
write_df(ws6, top50, start_row=r_off+1)
auto_width(ws6)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 7: MaAsLin2_Differential
# ═══════════════════════════════════════════════════════════════════════════════
ws7 = wb.create_sheet("MaAsLin2_Differential")
ws7.cell(row=1,column=1,
         value="MaAsLin2 Pairwise Results — q<0.05 AND |effect|≥1.0 — 143 rows / 74 unique taxa")
ws7.cell(row=1,column=1).font = WHT_FONT
ws7.cell(row=1,column=1).fill = HDR_FILL
ws7.merge_cells(start_row=1,start_column=1,end_row=1,end_column=10)

ald_out = sig_aldex[['taxon','comparison','race1','race2','effect','abs_effect',
                       'p_value','q_value','direction']].copy()
ald_out = ald_out.sort_values(['taxon','abs_effect'], ascending=[True,False])
ald_out.columns = ['Taxon','Comparison','Race1','Race2','Effect','|Effect|',
                    'p-value','q-value (BH)','Direction']
write_df(ws7, ald_out, start_row=2)
auto_width(ws7)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 8: KEGG_AE_119  (correct — 95% mass, seed=42)
# ═══════════════════════════════════════════════════════════════════════════════
ws8 = wb.create_sheet("KEGG_AE_119")

# Build AE-119 KEGG table from cache using correct taxa list
ae119_genera = ae119[ae119['Category'].isin(['Confirmed Core','AE Only'])]['Genus'].tolist()
ae119_grp    = ae119[ae119['Category'].isin(['Confirmed Core','AE Only'])].set_index('Genus')['Category'].to_dict()
ae119_grp    = {g: ('Shared' if v=='Confirmed Core' else 'AE-only') for g,v in ae119_grp.items()}

k8_rows = []
for g in ae119_genera:
    hit = kegg_cache[kegg_cache['Genus']==g]
    if len(hit):
        r = hit.iloc[0].copy()
    else:
        r = pd.Series({'Genus':g,'KEGG_OrgCode':'No match','KEGG_OrgDesc':'No match','N_KOs':0,'N_Pathways':0,'PathwayCategories':''})
    r['Group'] = ae119_grp.get(g,'Unknown')
    k8_rows.append(r)
k8 = pd.DataFrame(k8_rows)[['Genus','Group','KEGG_OrgCode','KEGG_OrgDesc','N_KOs','N_Pathways','PathwayCategories']].copy()
k8.columns = ['Genus','Group','KEGG Org Code','KEGG Organism','N KOs','N Pathways','Pathway Categories']
k8 = k8.sort_values(['Group','N KOs'], ascending=[True,False])

n_matched = int((k8['N KOs']>0).sum())
ws8.cell(row=1,column=1,
         value=f"KEGG Functional Annotation — AE-119 (95% mass, seed=42)  |  {n_matched}/119 matched ({100*n_matched//119}%)")
ws8.cell(row=1,column=1).font = WHT_FONT
ws8.cell(row=1,column=1).fill = HDR_FILL
ws8.merge_cells(start_row=1,start_column=1,end_row=1,end_column=7)
write_df(ws8, k8, start_row=2)
# Colour rows by group
for row_i in range(3, len(k8)+3):
    grp = ws8.cell(row=row_i, column=2).value
    fill = PatternFill("solid", fgColor="E8F0FE") if grp=="Shared" else \
           PatternFill("solid", fgColor="FFF3CD") if grp=="AE-only" else None
    if fill:
        for col_i in range(1, 8):
            ws8.cell(row=row_i, column=col_i).fill = fill
auto_width(ws8)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 9: KEGG_Differential
# ═══════════════════════════════════════════════════════════════════════════════
ws9 = wb.create_sheet("KEGG_Differential")
ws9.cell(row=1,column=1,
         value="KEGG Functional Annotation — MaAsLin2 Differential Taxa (74 taxa)  |  32/74 matched (43%)")
ws9.cell(row=1,column=1).font = WHT_FONT
ws9.cell(row=1,column=1).fill = HDR_FILL
ws9.merge_cells(start_row=1,start_column=1,end_row=1,end_column=7)

k9 = kegg_diff_df[['Genus','KEGG_OrgCode','KEGG_OrgDesc','N_KOs','N_Pathways','PathwayCategories']].copy()
k9.columns = ['Genus','KEGG Org Code','KEGG Organism','N KOs','N Pathways','Pathway Categories']
write_df(ws9, k9, start_row=2)
auto_width(ws9)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 10: FAPROTAX_Summary
# ═══════════════════════════════════════════════════════════════════════════════
ws10 = wb.create_sheet("FAPROTAX_Summary")
ws10.cell(row=1,column=1,
          value="FAPROTAX Ecological Function — mean relative abundance per functional group × race × core")
ws10.cell(row=1,column=1).font = WHT_FONT
ws10.cell(row=1,column=1).fill = HDR_FILL
ws10.merge_cells(start_row=1,start_column=1,end_row=1,end_column=6)

races = ["African","Asian","Caucasian","Hispanic"]
combined = []
for core_name, fap_df in [("AE-119 (95% mass)",fap_ae119),
                           ("AE-70  (Score>mean)",fap_ae70),
                           ("Differential-74",fap_diff)]:
    df = fap_df.copy()
    if "group" not in df.columns:
        df = df.rename(columns={df.columns[0]: "group"})
    nz = df[df[races].sum(axis=1)>0].copy()
    nz["Core"] = core_name
    combined.append(nz[["Core","group"]+races])
fap_all = pd.concat(combined, ignore_index=True)
fap_all.columns = ["Core","Functional Group","African","Asian","Caucasian","Hispanic"]
write_df(ws10, fap_all, start_row=2)
auto_width(ws10)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 11: Predicted_Pathways
# ═══════════════════════════════════════════════════════════════════════════════
ws11 = wb.create_sheet("Predicted_Pathways")
ws11.cell(row=1,column=1,
          value="Predicted KEGG Pathway Abundances by Race — Tax4Fun-equivalent  |  392 non-zero pathways")
ws11.cell(row=1,column=1).font = WHT_FONT
ws11.cell(row=1,column=1).fill = HDR_FILL
ws11.merge_cells(start_row=1,start_column=1,end_row=1,end_column=7)

map_means = race_pw[race_pw.index.str.startswith("map")].copy()
map_means = map_means[map_means[races].sum(axis=1)>0]
map_means["_mean"] = map_means[races].mean(axis=1)
map_means = map_means.sort_values("_mean", ascending=False).drop(columns="_mean")
map_means["Pathway Name"] = map_means.index.map(lambda x: pw_names.get(x,""))
pw_out = map_means.reset_index()
pw_out.columns = ["Pathway ID","African","Asian","Caucasian","Hispanic","Pathway Name"]
pw_out = pw_out[["Pathway ID","Pathway Name","African","Asian","Caucasian","Hispanic"]]
write_df(ws11, pw_out, start_row=2)
auto_width(ws11)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = ROOT / "final_results_seed42.xlsx"
wb.save(out_path)
print(f"\n✓ Saved: {out_path}")
print(f"  Sheets: {[ws.title for ws in wb.worksheets]}")
