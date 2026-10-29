"""
generate_panels.py
──────────────────
Renders each panel from Fig. 6, 7, and 8 as a standalone
publication-quality figure (PDF + PNG at 300 dpi).

Outputs (in reports/figures/):
  panel_6a_method_comparison.{pdf,png}
  panel_6b_core_composition.{pdf,png}
  panel_6c_kegg_richness.{pdf,png}
  panel_6d_predicted_pathways.{pdf,png}
  panel_7a_kegg_categories.{pdf,png}
  panel_7b_faprotax_race.{pdf,png}
  panel_8a_ae_only_prevalence.{pdf,png}
  panel_8b_pathway_categories.{pdf,png}

Run from Datasets/:
  source microbiome-ml-env/bin/activate && python3 pipeline/generate_panels.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT     = Path(".")
OUT      = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

RACE_COLORS  = {"African": "#CC79A7", "Asian": "#0072B2",
                "Caucasian": "#009E73", "Hispanic": "#E69F00"}
GROUP_COLORS = {"AE-only": "#E69F00", "Shared": "#0072B2", "Manual-only": "#009E73"}
RACES        = ["African", "Asian", "Caucasian", "Hispanic"]

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


# ── Fig. 6A — Method comparison ───────────────────────────────────────────────
def panel_6a():
    methods = [
        ("AE Weighted MSE\n(Score>mean)",  1.000, 0.729, 0.843),
        ("AE Weighted MSE\n(95% mass)",    0.739, 0.917, 0.819),
        ("RF Multiclass\n(Top-96)",        0.750, 0.750, 0.750),
        ("RF Binary\n(RFECV)",             0.568, 0.875, 0.689),
        ("AE Baseline\n(95% mass)",        0.684, 0.948, 0.795),
    ]
    methods_sorted = sorted(methods, key=lambda x: x[3])
    labels = [m[0] for m in methods_sorted]
    prec   = [m[1] for m in methods_sorted]
    rec    = [m[2] for m in methods_sorted]
    f1     = [m[3] for m in methods_sorted]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    y, w = np.arange(len(labels)), 0.25
    ax.barh(y - w, prec, w, color="#2166ac", alpha=0.88, label="Precision")
    ax.barh(y,     rec,  w, color="#4dac26", alpha=0.88, label="Recall")
    ax.barh(y + w, f1,   w, color="#d62728", alpha=0.88, label="F1-score")

    best = f1.index(max(f1))
    ax.barh(best + w, f1[best], w, color="#d62728", edgecolor="black", linewidth=1.5)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Score vs 96-taxon manual reference", fontsize=11, labelpad=6)
    ax.set_title("Method comparison vs manual core (96 taxa)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.axvline(0.8, color="#aaaaaa", ls="--", lw=0.9)
    ax.set_xlim(0, 1.18)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(axis="x", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_6a_method_comparison")


# ── Fig. 6B — Core taxa set composition ──────────────────────────────────────
def panel_6b():
    groups = ["AE Score>mean\n(70 taxa)", "Manual core\n(96 taxa)", "AE 95% mass\n(119 taxa)"]
    shared = [70, 88, 88]
    ae_add = [0,   0,  31]
    man_add= [0,   8,   0]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    y = np.arange(len(groups))
    ax.barh(y, shared, 0.5, color="#0072B2", alpha=0.88, label="Overlap with manual")
    ax.barh(y, ae_add, 0.5, left=shared,  color="#E69F00", alpha=0.88, label="AE-only additions")
    ax.barh(y, man_add,0.5, left=shared,  color="#009E73", alpha=0.88, label="Manual-only")

    totals = [s+a+m for s,a,m in zip(shared, ae_add, man_add)]
    for i, t in enumerate(totals):
        ax.text(t + 0.8, i, str(t), va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(groups, fontsize=10)
    ax.set_xlabel("Number of taxa", fontsize=11, labelpad=6)
    ax.set_title("Core taxon set composition", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(axis="x", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_6b_core_composition")


# ── Fig. 6C — KEGG KO richness by group ──────────────────────────────────────
def panel_6c():
    try:
        kegg = pd.read_csv(ROOT / "kegg_expanded_results.csv")
    except FileNotFoundError:
        print("  ✗ panel_6c: kegg_expanded_results.csv not found"); return

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    order = ["AE-only", "Shared", "Manual-only"]
    x_pos = {g: i for i, g in enumerate(order)}

    for grp, color in GROUP_COLORS.items():
        vals = kegg[kegg["Group"] == grp]["N_KOs"].dropna()
        vals = vals[vals > 0]
        if vals.empty: continue
        x = x_pos[grp]
        ax.scatter(np.full(len(vals), x) + np.random.uniform(-0.18, 0.18, len(vals)),
                   vals, color=color, alpha=0.45, s=22, zorder=2)
        ax.plot(x, vals.median(), marker="D", color=color, markersize=11, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.hlines(vals.median(), x - 0.28, x + 0.28, colors=color, lw=2.0, zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=11)
    ax.set_ylabel("KEGG KO gene count per taxon", fontsize=11, labelpad=6)
    ax.set_title("Functional gene richness by taxon group",
                 fontsize=12, fontweight="bold", pad=10)
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in order
               if g in kegg["Group"].values]
    ax.legend(handles=handles, fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_6c_kegg_richness")


# ── Fig. 6D — Predicted KEGG pathways by race (heatmap) ──────────────────────
def panel_6d():
    try:
        race_pw = pd.read_csv(ROOT / "kegg_race_pathway_means.csv", index_col=0)
    except FileNotFoundError:
        print("  ✗ panel_6d: kegg_race_pathway_means.csv not found"); return

    race_pw = race_pw[race_pw.index.str.startswith("map")]
    race_pw = race_pw.loc[race_pw.max(axis=1) > 0]
    top15   = race_pw.mean(axis=1).nlargest(30).index
    df_plot = race_pw.loc[top15]

    pathway_short = {
        "map01100": "Metabolic pathways",
        "map01110": "Secondary metabolites",
        "map01120": "Microbial metabolism",
        "map01240": "Cofactor biosynthesis",
        "map01230": "Amino acid biosynthesis",
        "map00195": "Photosynthesis",
        "map00280": "BCAA degradation",
        "map02040": "ABC transporters",
        "map00010": "Glycolysis",
        "map00020": "TCA cycle",
    }
    df_plot = df_plot.copy()
    df_plot.index = [pathway_short.get(p, p) for p in df_plot.index]
    races_in = [r for r in RACES if r in df_plot.columns]

    fig, ax = plt.subplots(figsize=(6, 10), dpi=300)
    sns.heatmap(df_plot[races_in], ax=ax, cmap="YlOrRd",
                linewidths=0.35, linecolor="white",
                cbar_kws={"label": "Predicted abundance", "shrink": 0.6})
    ax.set_title("Predicted KEGG pathway abundance by race\n(top 30 by mean, Tax4Fun-style)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    save(fig, "panel_6d_predicted_pathways")


# ── Fig. 7A — KEGG pathway categories ─────────────────────────────────────────
def panel_7a():
    try:
        comp = pd.read_csv(ROOT / "kegg_group_comparison.csv")
    except FileNotFoundError:
        print("  ✗ panel_7a: kegg_group_comparison.csv not found"); return

    pivot = comp.pivot_table(
        index="PathwayCategory", columns="Group",
        values="CountPerTaxon", fill_value=0
    )
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    col_order = [g for g in ["AE-only", "Shared", "Manual-only"] if g in pivot.columns]

    n = len(pivot.index)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.9), 5.5), dpi=300)
    x = np.arange(n); w = 0.25
    for i, grp in enumerate(col_order):
        color = GROUP_COLORS.get(grp, "#888888")
        ax.bar(x + i * w, pivot[grp], w, label=grp, color=color,
               alpha=0.88, edgecolor="white", lw=0.5, zorder=3)

    ax.set_xticks(x + w * (len(col_order) - 1) / 2)
    ax.set_xticklabels([c.replace(" ", "\n") for c in pivot.index],
                       fontsize=8.5, ha="center")
    ax.set_ylabel("Mean KEGG pathways per taxon", fontsize=11, labelpad=6)
    ax.set_title("KEGG pathway category distribution\nAE-only vs Shared vs Manual-only",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_7a_kegg_categories")


# ── Fig. 7B — FAPROTAX functional groups by race ─────────────────────────────
def panel_7b():
    try:
        fap = pd.read_csv(ROOT / "faprotax_race_means.csv", index_col=0)
    except FileNotFoundError:
        print("  ✗ panel_7b: faprotax_race_means.csv not found"); return

    races_in = [r for r in RACES if r in fap.columns]
    fap = fap[races_in]
    nz  = fap.loc[fap.max(axis=1) > 0].copy()
    nz["_mean"] = nz[races_in].mean(axis=1)
    nz  = nz.sort_values("_mean", ascending=False).drop(columns="_mean")
    nz.index = [n.replace("_or_", "/").replace("_", " ").title() for n in nz.index]

    n = len(nz); w = 0.18; offsets = np.linspace(-1.5 * w, 1.5 * w, len(races_in))
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5.5), dpi=300)
    for j, race in enumerate(races_in):
        ax.bar(np.arange(n) + offsets[j], nz[race].values,
               width=w, color=RACE_COLORS[race], label=race,
               edgecolor="white", lw=0.5, zorder=3)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(nz.index.tolist(), fontsize=9, rotation=35, ha="right")
    ax.set_ylabel("Mean relative abundance", fontsize=11, labelpad=6)
    ax.set_title("FAPROTAX functional group annotation by racial group",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(title="Race / Ethnicity", fontsize=9, title_fontsize=9,
              framealpha=0.95, edgecolor="#cccccc")
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_7b_faprotax_race")


# ── Fig. 8A — AE-only taxa: prevalence vs KO richness ────────────────────────
def panel_8a():
    try:
        kegg    = pd.read_csv(ROOT / "kegg_expanded_results.csv")
        ae_only = pd.read_csv(ROOT / "ae_only_56_taxa.csv")
    except FileNotFoundError as e:
        print(f"  ✗ panel_8a: {e}"); return

    merged = kegg[kegg["Group"] == "AE-only"].merge(
        ae_only[["OTU_ID", "GlobalPrevalence", "MeanRelAbund"]],
        on="OTU_ID", how="left"
    )
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    no_kegg  = merged[merged["N_KOs"].isna() | (merged["N_KOs"] == 0)]
    has_kegg = merged[merged["N_KOs"] > 0]

    ax.scatter(no_kegg["GlobalPrevalence"], np.zeros(len(no_kegg)),
               color="#aec7e8", alpha=0.7, s=55, marker="x", label="No KEGG match", zorder=2)
    sc = ax.scatter(has_kegg["GlobalPrevalence"], has_kegg["N_KOs"],
                    c=has_kegg["MeanRelAbund"], cmap="YlOrRd",
                    alpha=0.88, s=70, zorder=3, label="KEGG matched",
                    edgecolors="#555555", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Mean relative abundance", shrink=0.8)

    for _, row in has_kegg.nlargest(8, "N_KOs").iterrows():
        ax.annotate(str(row["Genus"]).split()[0],
                    (row["GlobalPrevalence"], row["N_KOs"]),
                    fontsize=6.5, xytext=(4, 2), textcoords="offset points",
                    color="#333333")

    ax.set_xlabel("Global prevalence (fraction of samples)", fontsize=11, labelpad=6)
    ax.set_ylabel("KEGG KO gene count per taxon", fontsize=11, labelpad=6)
    ax.set_title("AE-only taxa: prevalence vs functional richness\n"
                 "(taxa selected by AE but absent from manual reference)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_8a_ae_only_prevalence")


# ── Fig. 8B — Pathway category coverage, AE-only taxa ────────────────────────
def panel_8b():
    try:
        kegg    = pd.read_csv(ROOT / "kegg_expanded_results.csv")
        ae_only = pd.read_csv(ROOT / "ae_only_56_taxa.csv")
    except FileNotFoundError as e:
        print(f"  ✗ panel_8b: {e}"); return

    merged = kegg[kegg["Group"] == "AE-only"].merge(
        ae_only[["OTU_ID", "GlobalPrevalence", "MeanRelAbund"]],
        on="OTU_ID", how="left"
    )
    cat_counts: dict = {}
    for _, row in merged.iterrows():
        if pd.notna(row.get("PathwayCategories")) and row["PathwayCategories"]:
            for cat in str(row["PathwayCategories"]).split("; "):
                cat = cat.strip()
                if cat:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

    if not cat_counts:
        print("  ✗ panel_8b: no pathway categories found"); return

    cats_sorted = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    cats, counts = zip(*cats_sorted)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))

    fig, ax = plt.subplots(figsize=(7, max(4.5, len(cats) * 0.45)), dpi=300)
    bars = ax.barh(range(len(cats)), counts, color=colors, alpha=0.88,
                   edgecolor="white", lw=0.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=9.5)

    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=9.5)
    ax.set_xlabel("Number of AE-only taxa with pathway category", fontsize=11, labelpad=6)
    ax.set_title("KEGG pathway category coverage\nAE-only taxa (KEGG-matched subset)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", ls="--", lw=0.5, alpha=0.35, color="#cccccc"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "panel_8b_pathway_categories")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)   # for jitter in panel_6c
    print("\n── Generating standalone panels ──────────────────────────────────────")
    panel_6a(); panel_6b(); panel_6c(); panel_6d()
    panel_7a(); panel_7b()
    panel_8a(); panel_8b()
    print("\nAll panels saved to reports/figures/")
