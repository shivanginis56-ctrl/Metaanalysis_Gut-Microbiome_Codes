"""
kegg_expanded.py
────────────────
Extends kegg_pathways.py to ALL AE core + Manual core taxa (union set).
Compares pathway category distributions across three groups:
  - AE-only     : in AE core but NOT in manual core
  - Manual-only : in manual core but NOT in AE core
  - Shared      : in both AE core and manual core

Also queries KEGG KO numbers per taxon and saves them for
kegg_functional_prediction.py (per-sample pathway prediction).

Caches API results to kegg_expanded_cache.csv to avoid re-querying.

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.kegg_expanded
"""

import sys
import time
import re
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import ROOT_DIR

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KEGG_BASE  = "https://rest.kegg.jp"
PAUSE_SEC  = 0.35

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


def kegg_find_organism(genus: str) -> tuple[str, str]:
    clean = re.sub(r"[\[\]_]", " ", genus).strip().split()[0]
    try:
        resp = requests.get(f"{KEGG_BASE}/find/genome/{clean}", timeout=10)
        if resp.status_code != 200 or not resp.text.strip():
            return "", ""
        line  = resp.text.strip().splitlines()[0]
        parts = line.split("\t")
        if len(parts) < 2:
            return "", ""
        org_part = parts[1].split(";")
        return org_part[0].strip(), org_part[1].strip() if len(org_part) > 1 else ""
    except Exception:
        return "", ""


def kegg_list_pathways(org_code: str) -> list[tuple[str, str]]:
    try:
        resp = requests.get(f"{KEGG_BASE}/list/pathway/{org_code}", timeout=10)
        if resp.status_code != 200 or not resp.text.strip():
            return []
        paths = []
        for line in resp.text.strip().splitlines():
            p = line.split("\t")
            if len(p) >= 2:
                paths.append((p[0], p[1]))
        return paths
    except Exception:
        return []


def kegg_list_ko(org_code: str) -> list[str]:
    """Return list of unique KO numbers for an organism."""
    try:
        resp = requests.get(f"{KEGG_BASE}/link/ko/{org_code}", timeout=15)
        if resp.status_code != 200 or not resp.text.strip():
            return []
        kos = set()
        for line in resp.text.strip().splitlines():
            p = line.split("\t")
            if len(p) >= 2 and p[1].startswith("ko:"):
                kos.add(p[1].replace("ko:", ""))
        return sorted(kos)
    except Exception:
        return []


def load_taxa_groups() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ae_core    = pd.read_csv(ROOT_DIR / "ae_core_148.csv")
    manual_raw = pd.read_csv(ROOT_DIR / "step3_final_core.csv")
    manual_core= manual_raw[["OTU_ID", "Genus"]].drop_duplicates()

    ae_ids     = set(ae_core["OTU_ID"].tolist())
    man_ids    = set(manual_core["OTU_ID"].tolist())
    shared_ids = ae_ids & man_ids
    ae_only_ids= ae_ids - man_ids
    man_only_ids=man_ids - ae_ids

    # Add group label to AE core
    ae_core["Group"] = ae_core["OTU_ID"].apply(
        lambda x: "Shared" if x in shared_ids else "AE-only"
    )
    manual_core = manual_core.copy()
    manual_core["Group"] = manual_core["OTU_ID"].apply(
        lambda x: "Shared" if x in shared_ids else "Manual-only"
    )

    # Union: all unique taxa
    union = (
        pd.concat([
            ae_core[["OTU_ID", "Genus", "Group"]],
            manual_core[manual_core["Group"] == "Manual-only"][["OTU_ID", "Genus", "Group"]],
        ])
        .drop_duplicates(subset="OTU_ID")
        .reset_index(drop=True)
    )

    print(f"  AE core     : {len(ae_ids)} OTUs")
    print(f"  Manual core : {len(man_ids)} OTUs")
    print(f"  Shared      : {len(shared_ids)}")
    print(f"  AE-only     : {len(ae_only_ids)}")
    print(f"  Manual-only : {len(man_only_ids)}")
    print(f"  Union total : {len(union)} OTUs to query KEGG")

    return union, ae_core, manual_core


def query_kegg_with_cache(union: pd.DataFrame) -> pd.DataFrame:
    cache_path = ROOT_DIR / "kegg_expanded_cache.csv"

    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        cached_ids = set(cache["OTU_ID"].tolist())
        print(f"  Cache loaded: {len(cache)} taxa already queried")
    else:
        cache = pd.DataFrame()
        cached_ids = set()

    to_query = union[~union["OTU_ID"].isin(cached_ids)]
    print(f"  Need to query: {len(to_query)} taxa  (~{len(to_query)*0.35*3/60:.1f} min)")

    records = []
    for i, (_, row) in enumerate(to_query.iterrows(), 1):
        otu_id = int(row["OTU_ID"])
        genus  = str(row["Genus"])
        group  = str(row["Group"])

        print(f"  [{i:>3}/{len(to_query)}] {genus:<45}", end="", flush=True)

        org_code, org_desc = kegg_find_organism(genus)
        time.sleep(PAUSE_SEC)

        if not org_code:
            print("→ no KEGG hit")
            records.append({
                "OTU_ID": otu_id, "Genus": genus, "Group": group,
                "KEGG_OrgCode": "", "KEGG_OrgDesc": "No match",
                "N_Pathways": 0, "PathwayCategories": "", "TopPathways": "",
                "N_KOs": 0, "KO_List": "",
            })
            continue

        pathways = kegg_list_pathways(org_code)
        time.sleep(PAUSE_SEC)

        kos = kegg_list_ko(org_code)
        time.sleep(PAUSE_SEC)

        cats     = [categorise_pathway(name) for _, name in pathways]
        cat_str  = "; ".join(sorted(set(cats)))
        top_paths= "; ".join(name for _, name in pathways[:5])

        print(f"→ {org_code}  ({len(pathways)} pathways, {len(kos)} KOs)")

        records.append({
            "OTU_ID": otu_id, "Genus": genus, "Group": group,
            "KEGG_OrgCode": org_code, "KEGG_OrgDesc": org_desc,
            "N_Pathways": len(pathways), "PathwayCategories": cat_str,
            "TopPathways": top_paths,
            "N_KOs": len(kos), "KO_List": ";".join(kos),
        })

    new_df = pd.DataFrame(records)
    if not new_df.empty:
        combined = pd.concat([cache, new_df], ignore_index=True) if not cache.empty else new_df
        combined.to_csv(cache_path, index=False)
        print(f"  Cache saved: {len(combined)} taxa total")
        return combined

    return cache


def pathway_category_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Count pathway categories per group for comparison."""
    groups = ["AE-only", "Shared", "Manual-only"]
    rows   = []

    for group in groups:
        sub = results[results["Group"] == group]
        cat_counts: dict[str, int] = {}
        for _, row in sub.iterrows():
            if not row["PathwayCategories"]:
                continue
            for cat in str(row["PathwayCategories"]).split("; "):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        for cat, cnt in cat_counts.items():
            rows.append({"Group": group, "PathwayCategory": cat,
                         "Count": cnt, "N_Taxa": len(sub)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalise by group size
    df["CountPerTaxon"] = df["Count"] / df["N_Taxa"]
    return df


def fisher_enrichment(results: pd.DataFrame) -> pd.DataFrame:
    """Fisher's exact test: AE-only vs Manual-only per pathway category."""
    ae_sub  = results[results["Group"] == "AE-only"]
    man_sub = results[results["Group"] == "Manual-only"]

    all_cats = set()
    for _, row in results.iterrows():
        if row["PathwayCategories"]:
            all_cats.update(str(row["PathwayCategories"]).split("; "))

    rows = []
    for cat in sorted(all_cats):
        ae_yes  = sum(cat in str(r["PathwayCategories"]) for _, r in ae_sub.iterrows())
        ae_no   = len(ae_sub)  - ae_yes
        man_yes = sum(cat in str(r["PathwayCategories"]) for _, r in man_sub.iterrows())
        man_no  = len(man_sub) - man_yes

        if ae_yes + man_yes == 0:
            continue

        _, pval = fisher_exact([[ae_yes, ae_no], [man_yes, man_no]])
        rows.append({
            "PathwayCategory": cat,
            "AE_only_count":  ae_yes, "AE_only_total":  len(ae_sub),
            "Manual_only_count": man_yes, "Manual_only_total": len(man_sub),
            "AE_fraction":   ae_yes  / max(len(ae_sub),  1),
            "Man_fraction":  man_yes / max(len(man_sub), 1),
            "pvalue": pval,
        })

    df = pd.DataFrame(rows).sort_values("pvalue")
    return df


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_group_pathway_bars(comp_df: pd.DataFrame, out_path: Path):
    if comp_df.empty:
        return

    pivot = comp_df.pivot_table(
        index="PathwayCategory", columns="Group",
        values="CountPerTaxon", fill_value=0
    )
    pivot = pivot.sort_values(pivot.columns[0] if len(pivot.columns) else "AE-only",
                              ascending=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(pivot.index))
    n   = len(pivot.columns)
    w   = 0.75 / n
    colors = {"AE-only": "#d62728", "Shared": "#ff7f0e", "Manual-only": "#1f77b4"}

    for i, grp in enumerate(pivot.columns):
        color = colors.get(grp, f"C{i}")
        ax.bar(x + i * w, pivot[grp], w, label=grp, color=color, alpha=0.85)

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels([c.replace(" ", "\n") for c in pivot.index],
                       fontsize=8, ha="center")
    ax.set_ylabel("KEGG pathways per taxon", fontsize=10)
    ax.set_title(
        "KEGG Pathway Categories — AE-only vs Shared vs Manual-only taxa\n"
        "(normalised by group size)",
        fontsize=10, pad=10,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_ko_distribution(results: pd.DataFrame, out_path: Path):
    """Box plot of KO counts per group."""
    groups  = ["AE-only", "Shared", "Manual-only"]
    data    = [results[results["Group"] == g]["N_KOs"].dropna().tolist() for g in groups]
    colors  = ["#d62728", "#ff7f0e", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=groups,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Number of KO (KEGG Orthology) genes", fontsize=10)
    ax.set_title("Functional gene richness (KO count) per taxa group", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Add medians as text
    for i, d in enumerate(data, 1):
        if d:
            ax.text(i, np.median(d), f" {int(np.median(d))}",
                    va="center", fontsize=9, color="black")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_enrichment(enrich_df: pd.DataFrame, out_path: Path):
    """Horizontal bar: AE fraction vs Manual fraction per pathway category."""
    if enrich_df.empty:
        return

    df = enrich_df[enrich_df["pvalue"] < 0.2].copy()  # show nominally enriched
    if df.empty:
        df = enrich_df.head(12)

    df = df.sort_values("AE_fraction", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["AE_fraction"],  0.35, color="#d62728", alpha=0.8, label="AE-only")
    ax.barh(y + 0.2, df["Man_fraction"], 0.35, color="#1f77b4", alpha=0.8, label="Manual-only")

    ax.set_yticks(y)
    ax.set_yticklabels(df["PathwayCategory"], fontsize=9)
    ax.set_xlabel("Fraction of taxa in group with this pathway category", fontsize=9)
    ax.set_title("Pathway category enrichment: AE-only vs Manual-only taxa\n"
                 "(shown: p < 0.2, Fisher's exact test)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # p-value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(max(row["AE_fraction"], row["Man_fraction"]) + 0.01, i,
                f"p={row['pvalue']:.2f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "═" * 60)
    print("  Expanded KEGG Analysis — AE + Manual Core Taxa")
    print("═" * 60)

    # 1. Load taxa groups
    print("\n  Loading taxa groups …")
    union, ae_core, manual_core = load_taxa_groups()

    # 2. Query KEGG (with cache)
    print("\n  Querying KEGG REST API …")
    results = query_kegg_with_cache(union)

    n_hits = (results["KEGG_OrgCode"] != "").sum()
    print(f"\n  KEGG hits : {n_hits} / {len(results)} taxa  ({n_hits/len(results)*100:.0f}%)")
    print(f"  Mean pathways per taxon (matched): "
          f"{results[results['N_KOs']>0]['N_Pathways'].mean():.0f}")
    print(f"  Mean KOs per taxon (matched): "
          f"{results[results['N_KOs']>0]['N_KOs'].mean():.0f}")

    # Save per-taxon results
    out_results = ROOT_DIR / "kegg_expanded_results.csv"
    results.to_csv(out_results, index=False)
    print(f"  Saved {out_results.name}")

    # 3. Pathway category comparison
    comp_df = pathway_category_comparison(results)
    out_comp = ROOT_DIR / "kegg_group_comparison.csv"
    comp_df.to_csv(out_comp, index=False)
    print(f"  Saved {out_comp.name}")

    # 4. Fisher enrichment
    enrich_df = fisher_enrichment(results)
    out_enrich = ROOT_DIR / "kegg_enrichment.csv"
    enrich_df.to_csv(out_enrich, index=False)
    print(f"  Saved {out_enrich.name}")

    if not enrich_df.empty:
        print("\n  Top pathway enrichments (AE-only vs Manual-only):")
        for _, row in enrich_df.head(5).iterrows():
            print(f"    {row['PathwayCategory']:<30}  "
                  f"AE={row['AE_fraction']:.2f}  Man={row['Man_fraction']:.2f}  "
                  f"p={row['pvalue']:.3f}")

    # 5. Figures
    print("\n  Generating figures …")
    plot_group_pathway_bars(comp_df,
        FIGURES_DIR / "fig_kegg_group_comparison.pdf")
    plot_ko_distribution(results,
        FIGURES_DIR / "fig_kegg_ko_distribution.pdf")
    plot_enrichment(enrich_df,
        FIGURES_DIR / "fig_kegg_enrichment.pdf")

    elapsed = (time.time() - t0) / 60
    print(f"\n✓  Done in {elapsed:.1f} min.\n")


if __name__ == "__main__":
    main()
