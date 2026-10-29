"""
run_functional_correct.py
─────────────────────────
Runs KEGG (from cache) + FAPROTAX on the THREE correct sets:
  1. AE core — 119 taxa (95% mass, seed=42)
  2. AE core — 70 taxa  (Score>mean, seed=42)
  3. Differential taxa — 74 taxa (ALDEx2 pairwise q<0.05, |effect|≥1.0)

All KEGG data already in cache — no new API calls.
FAPROTAX is run on the correct OTU subsets of the 238 pre-filtered taxa.
"""
import re, sys, subprocess, tempfile, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
OUT  = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FAPROTAX_DIR    = ROOT / "FAPROTAX_1.2.12"
COLLAPSE_SCRIPT = FAPROTAX_DIR / "collapse_table.py"
FAPROTAX_DB     = FAPROTAX_DIR / "FAPROTAX.txt"
RACES = ["African", "Asian", "Caucasian", "Hispanic"]

COL = {
    "Shared":       "#0072B2",
    "Confirmed":    "#0072B2",
    "AE-only":      "#E69F00",
    "Manual-only":  "#009E73",
    "Differential": "#56B4E9",
    "African":      "#CC79A7",
    "Asian":        "#0072B2",
    "Caucasian":    "#009E73",
    "Hispanic":     "#E69F00",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_genus(name):
    name = str(name)
    name = re.sub(r"\[|\]", "", name)
    name = re.sub(r"_ugn_\d+", " sp.", name)
    name = re.sub(r"_ufm_\d+", " sp.", name)
    name = re.sub(r"\.MBIC\d+", "", name)
    return re.sub(r"\s+", " ", name.replace("_"," ").replace("."," ")).strip()

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def tss_normalize(df):
    s = df.sum(axis=0); s[s==0] = 1
    return df.div(s, axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load taxa lists
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 1: Load taxon lists ──────────────────────────────────────────")

# AE-119 (seed=42)
m95 = pd.read_excel("threshold_comparison.xlsx", sheet_name="95% score mass", header=None)
h = next(i for i,r in m95.iterrows() if str(r[0]).strip()=="OTU_ID")
ae119_full = m95.iloc[h+1:].copy(); ae119_full.columns = list(m95.iloc[h])
ae119_full = ae119_full[pd.to_numeric(ae119_full["OTU_ID"],errors="coerce").notna()].copy()
ae119_full["OTU_ID"] = ae119_full["OTU_ID"].astype(int)
ae119_full["status"] = ae119_full["Category"]
# AE-selected only (119 taxa = Confirmed Core + AE Only)
ae119 = ae119_full[ae119_full["status"].isin(["Confirmed Core","AE Only"])].copy()
ae119["Group"] = ae119["status"].map({"Confirmed Core":"Shared","AE Only":"AE-only"})

# Manual-96 (for context — "Missed by AE" rows in the 95% mass sheet)
manual96_genera = ae119_full[ae119_full["status"]=="Missed by AE"]["Genus"].tolist()

# AE-70 (seed=42)
msm = pd.read_excel("threshold_comparison.xlsx", sheet_name="Score > mean", header=None)
h2 = next(i for i,r in msm.iterrows() if str(r[0]).strip()=="OTU_ID")
ae70_full = msm.iloc[h2+1:].copy(); ae70_full.columns = list(msm.iloc[h2])
ae70_full = ae70_full[pd.to_numeric(ae70_full["OTU_ID"],errors="coerce").notna()].copy()
ae70_full["OTU_ID"] = ae70_full["OTU_ID"].astype(int)
ae70 = ae70_full[ae70_full["Category"]=="Confirmed Core"].copy()
ae70["Group"] = "Confirmed"

# Differential-74
diff_result = pd.read_csv("results_long_random_effect_all_taxa_effect_cutoff.csv")
sig = diff_result[(diff_result["q_value"]<0.05)&(diff_result["abs_effect"]>=1.0)]
diff74_genera = list(sig["taxon"].unique())

# Annotation for OTU_ID → genus mapping
annot = pd.read_csv("aldex2_taxon_annotation.csv")
extra_map = {
    "X.Eubacterium..siraeum.group":     14,
    "X.Eubacterium..sulci.group":       15,
    "X.Eubacterium..ruminantium.group": 12,
    "X.Eubacterium..saphenum.group":    13,
    "X.Eubacterium..yurii.group":       19,
    "X.Polyangium..brachysporum.group": 21,
    "Acaryochloris.MBIC11017":          42,
    "X.Clostridium..methylpentosum.group": 5,
    "Ktedonobacteraceae_ugn_28":        28,
    "Ktedonobacteraceae_ugn_406":       405,
    "Ktedonobacteraceae_ugn_29":        27,
    "Ktedonobacteraceae_ugn_27":        26,
    "X.Ruminococcus..gnavus.group":     25,
}

def get_otu_ids(genera):
    ids = []
    for g in genera:
        if g in extra_map:
            ids.append(extra_map[g])
        else:
            row = annot[annot["genus"]==g]
            if len(row): ids.append(row["otu_id"].values[0])
    return ids

diff74_otu_ids = get_otu_ids(diff74_genera)
print(f"  AE-119: {len(ae119)} taxa")
print(f"  AE-70:  {len(ae70)} taxa")
print(f"  Diff-74 OTU_IDs resolved: {len(diff74_otu_ids)}/{len(diff74_genera)}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — KEGG from cache (no API calls)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 2: KEGG annotation from cache ────────────────────────────────")

cache = pd.read_csv("kegg_expanded_cache.csv")
cache["matched"] = cache["N_KOs"].fillna(0) > 0

def build_kegg_df(genera, group_map=None):
    """group_map: dict genus→group_label, or None for single group."""
    rows = []
    for g in genera:
        hit = cache[cache["Genus"]==g]
        if len(hit):
            row = hit.iloc[0].copy()
        else:
            row = pd.Series({"Genus":g,"KEGG_OrgCode":"No match","N_KOs":0,"N_Pathways":0})
        row["label"] = clean_genus(g)
        row["Group"] = group_map.get(g,"Unknown") if group_map else "Differential"
        rows.append(row)
    df = pd.DataFrame(rows)
    df["N_KOs"]     = pd.to_numeric(df["N_KOs"],     errors="coerce").fillna(0)
    df["N_Pathways"]= pd.to_numeric(df["N_Pathways"],errors="coerce").fillna(0)
    df["matched"]   = df["N_KOs"] > 0
    df["color"]     = df["Group"].map(COL).fillna("#888888")
    return df

# AE-119
group_map_119 = dict(zip(ae119["Genus"], ae119["Group"]))
kegg_119 = build_kegg_df(ae119["Genus"].tolist(), group_map_119)
kegg_119_matched = len(kegg_119[kegg_119["matched"]])
print(f"  AE-119 KEGG matched: {kegg_119_matched}/{len(kegg_119)}")

# AE-70
group_map_70 = {g:"Confirmed" for g in ae70["Genus"].tolist()}
kegg_70 = build_kegg_df(ae70["Genus"].tolist(), group_map_70)
kegg_70_matched = len(kegg_70[kegg_70["matched"]])
print(f"  AE-70  KEGG matched: {kegg_70_matched}/{len(kegg_70)}")

# Differential-74
kegg_diff = build_kegg_df(diff74_genera)
kegg_diff_matched = len(kegg_diff[kegg_diff["matched"]])
print(f"  Diff-74 KEGG matched: {kegg_diff_matched}/{len(kegg_diff)}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FAPROTAX on correct subsets
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 3: FAPROTAX analysis ─────────────────────────────────────────")

if not COLLAPSE_SCRIPT.exists():
    print("  ✗ FAPROTAX not found — skipping FAPROTAX step")
    faprotax_ok = False
else:
    print("  ✓ FAPROTAX found")
    faprotax_ok = True

if faprotax_ok:
    print("  Loading abundance matrix (~30s)…")
    t0 = time.time()
    abund_full = pd.read_csv("Taxonomic_abundance.csv", index_col=0)
    print(f"    Loaded {abund_full.shape[0]} taxa × {abund_full.shape[1]} samples ({time.time()-t0:.0f}s)")

    # Pre-filter: prevalence >= 5%
    n_samp = abund_full.shape[1]
    prev   = (abund_full > 0).sum(axis=1) / n_samp
    abund_filt = abund_full[prev >= 0.05].copy()
    abund_tss  = tss_normalize(abund_filt)
    print(f"    Pre-filtered: {abund_tss.shape[0]} taxa")

    meta = pd.read_csv("metadata_clean.csv")
    meta.index = meta["sample_id"]

    def run_faprotax_subset(otu_ids, label):
        """Select taxa by OTU_ID (1-based iloc in FULL matrix), run FAPROTAX."""
        valid_rows = []
        for oid in otu_ids:
            if oid <= abund_full.shape[0]:
                taxon_name = abund_full.index[oid - 1]
                if taxon_name in abund_tss.index:
                    valid_rows.append(taxon_name)
        valid_rows = list(dict.fromkeys(valid_rows))  # deduplicate, preserve order
        if not valid_rows:
            print(f"    ⚠  No valid taxa for {label}")
            return None
        subset = abund_tss.loc[valid_rows].copy()
        print(f"    {label}: {len(valid_rows)} taxa in abundance matrix")

        # Write FAPROTAX input TSV — index.name="taxonomy", no # prefix
        with tempfile.TemporaryDirectory() as tmp:
            tsv_path = Path(tmp) / f"faprotax_{label}.tsv"
            out_path = Path(tmp) / f"faprotax_{label}_out.tsv"
            rep_path = Path(tmp) / f"faprotax_{label}_report.txt"
            subset.index.name = "taxonomy"
            subset.to_csv(tsv_path, sep="\t")

            result = subprocess.run(
                [sys.executable, str(COLLAPSE_SCRIPT),
                 "-i", str(tsv_path),
                 "-g", str(FAPROTAX_DB),
                 "-o", str(out_path),
                 "-r", str(rep_path),
                 "-d", "taxonomy",
                 "-f"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"    ⚠  FAPROTAX error: {result.stderr[:400]}")
                return None

            func = pd.read_csv(out_path, sep="\t", index_col=0)
            func = func[~func.index.astype(str).str.startswith("#")]
            func = func.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        # Compute per-race means
        samples = meta.index[meta.index.isin(func.columns)]
        func    = func[samples]
        race_means = {}
        for race in RACES:
            race_samps = meta.index[meta["race"]==race]
            race_samps = race_samps[race_samps.isin(func.columns)]
            race_means[race] = func[race_samps].mean(axis=1)
        return pd.DataFrame(race_means)

    # Run for each set
    fap_119  = run_faprotax_subset(ae119["OTU_ID"].tolist(), "ae119")
    fap_70   = run_faprotax_subset(ae70["OTU_ID"].tolist(),  "ae70")
    fap_diff = run_faprotax_subset(diff74_otu_ids,            "diff74")

    def save_fap(df, path):
        if df is not None:
            df.index.name = "group"
            df.reset_index().to_csv(path, index=False)
            print(f"    Saved {path}")

    save_fap(fap_119,  ROOT / "faprotax_ae119_means.csv")
    save_fap(fap_70,   ROOT / "faprotax_ae70_means.csv")
    save_fap(fap_diff, ROOT / "faprotax_diff74_means.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate all figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 4: Generate figures ──────────────────────────────────────────")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

def fig_kegg_lollipop(kegg_df, title, fname, group_labels):
    df = kegg_df[kegg_df["matched"]].sort_values("N_KOs", ascending=True).copy()
    n_match = len(df); total = len(kegg_df)
    xmax = df["N_KOs"].max() if len(df) else 1

    h = max(8, n_match * 0.29)
    fig, ax = plt.subplots(figsize=(11, h), dpi=300)
    y = np.arange(n_match)
    ax.set_xlim(0, xmax * 1.22); ax.set_ylim(-0.8, n_match - 0.2)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([0, row["N_KOs"]], [i, i],
                color=row["color"], lw=1.0, alpha=0.65, zorder=1)
    ax.scatter(df["N_KOs"].values, y, c=df["color"].values,
               s=28, zorder=3, linewidths=0.4, edgecolors="white")
    for i, (_, row) in enumerate(df.iterrows()):
        if row["N_Pathways"] > 0:
            ax.text(row["N_KOs"] + xmax*0.013, i,
                    f"({int(row['N_Pathways'])}pw)",
                    va="center", ha="left", fontsize=5.2, color="#666666")

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values, fontsize=6.5, fontstyle="italic")
    ax.set_xlabel("Number of KEGG KO gene IDs", fontsize=10, labelpad=6)
    ax.set_title(
        f"KEGG Functional Annotation — {title}\n"
        f"{n_match} of {total} taxa KEGG-matched ({100*n_match/total:.0f}%)"
        f"  ·  annotations: pathway count per taxon",
        fontsize=10, fontweight="bold", pad=10, loc="left"
    )
    if n_match:
        med = df["N_KOs"].median()
        ax.axvline(med, color="#aaaaaa", ls=":", lw=1.0)
        ax.text(med + xmax*0.013, n_match - 0.6,
                f"median {int(med):,} KOs", fontsize=7, color="#888888", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(axis="x", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    style_ax(ax)

    groups_in = df["Group"].unique()
    handles = [mpatches.Patch(color=COL.get(g,"#888888"), label=g)
               for g in group_labels if g in groups_in]
    if handles:
        ax.legend(handles=handles, fontsize=8.5, loc="lower right",
                  framealpha=0.95, edgecolor="#cccccc", title="Taxon group",
                  title_fontsize=8.5)

    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{fname}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {fname} ({n_match} matched)")

def fig_faprotax_bar(csv_path, title, fname):
    if not Path(csv_path).exists():
        print(f"  ✗ {fname}: FAPROTAX CSV missing")
        return
    df = pd.read_csv(csv_path)
    if "group" not in df.columns:
        df = df.rename(columns={df.columns[0]: "group"})
    nz = df[df[RACES].sum(axis=1) > 0].copy()
    nz["_mean"] = nz[RACES].mean(axis=1)
    nz = nz.sort_values("_mean", ascending=False).reset_index(drop=True)
    nz["group_clean"] = nz["group"].str.replace("_or_","/").str.replace("_"," ").str.title()

    n_groups = len(nz)
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.1), 5.5), dpi=300)
    bar_w = 0.18; offsets = np.linspace(-1.5*bar_w, 1.5*bar_w, 4)
    race_cols = [COL[r] for r in RACES]
    for j, (race, color) in enumerate(zip(RACES, race_cols)):
        ax.bar(np.arange(n_groups) + offsets[j], nz[race].values,
               width=bar_w, color=color, label=race, edgecolor="white", lw=0.5, zorder=3)

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(nz["group_clean"].values, fontsize=8.5, rotation=35, ha="right")
    ax.set_ylabel("Mean relative abundance", fontsize=10, labelpad=6)
    ax.set_title(
        f"FAPROTAX Ecological Function Annotation — {title}\n"
        f"{n_groups} non-zero functional groups (of 92 total in database)",
        fontsize=10, fontweight="bold", pad=10, loc="left"
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4, color="#cccccc"); ax.set_axisbelow(True)
    style_ax(ax)
    ax.legend(title="Race / Ethnicity", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.95, edgecolor="#cccccc")

    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{fname}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {fname} ({n_groups} non-zero groups)")

# KEGG figures
fig_kegg_lollipop(kegg_119, "AE-Derived Core — 95% Mass (119 taxa, seed=42)",
                  "fig_ae119_kegg", ["Shared","AE-only"])
fig_kegg_lollipop(kegg_70,  "AE-Derived Core — Score>Mean (70 taxa, seed=42)",
                  "fig_ae70_kegg",  ["Confirmed"])
fig_kegg_lollipop(kegg_diff,"Differentially Abundant Taxa — ALDEx2 Pairwise (74 taxa)",
                  "fig_diff74_kegg",["Differential"])

# FAPROTAX figures
fig_faprotax_bar("faprotax_ae119_means.csv",
                 "AE-Derived Core — 95% Mass (119 taxa, seed=42)",  "fig_ae119_faprotax")
fig_faprotax_bar("faprotax_ae70_means.csv",
                 "AE-Derived Core — Score>Mean (70 taxa, seed=42)", "fig_ae70_faprotax")
fig_faprotax_bar("faprotax_diff74_means.csv",
                 "Differentially Abundant Taxa — ALDEx2 (74 taxa)", "fig_diff74_faprotax")

print("\n── All done ──────────────────────────────────────────────────────────")
