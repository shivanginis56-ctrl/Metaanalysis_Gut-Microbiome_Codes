"""
prep_for_aldex2.py
──────────────────
Exports two R-ready CSVs for the ALDEx2 differential abundance analysis:

  aldex2_counts.csv   — filtered count matrix (238 taxa × 7302 samples)
                        rows = taxon names, cols = sample IDs
                        values = raw integer read counts
  aldex2_metadata.csv — sample metadata (sample_id, race, bioproject)

Run from Datasets/:
    microbiome-ml-env/bin/python -m pipeline.prep_for_aldex2
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config      import METADATA_FILE, ABUNDANCE_FILE, ROOT_DIR
from pipeline.data_loader import load_metadata, load_abundance

def main():
    print("  Loading metadata …")
    meta = load_metadata(METADATA_FILE)   # dict: sample_id → {bioproject, race}
    meta_df = pd.DataFrame.from_dict(meta, orient="index").reset_index()
    meta_df.columns = ["sample_id", "bioproject", "race"]
    meta_df = meta_df[meta_df["race"].isin(["African", "Asian", "Caucasian", "Hispanic"])]

    print("  Loading abundance matrix …")
    taxa_names, otu_ids, raw_df = load_abundance(ABUNDANCE_FILE)

    # Align to metadata samples
    known = [s for s in raw_df.columns if s in meta]
    raw_df = raw_df[known]

    # Filter by prevalence using positional indexing (avoids duplicate name issues)
    print("  Filtering rare taxa (prevalence >= 5%) …")
    n_samples  = raw_df.shape[1]
    prevalence = (raw_df > 0).sum(axis=1) / n_samples
    keep_idx   = [i for i, p in enumerate(prevalence) if p >= 0.05]
    print(f"  Retained {len(keep_idx)} taxa")

    # Use positional indexing — guaranteed unique OTU_IDs regardless of genus name dups
    filtered_df      = raw_df.iloc[keep_idx]
    kept_otu_ids     = [otu_ids[i]   for i in keep_idx]
    kept_genus_names = [taxa_names[i] for i in keep_idx]

    count_matrix       = filtered_df.round(0).astype(int)
    count_matrix.index = [f"OTU{oid}" for oid in kept_otu_ids]   # e.g. OTU42

    # Taxon annotation table so R can map OTU label → genus name
    taxon_ann = pd.DataFrame({
        "otu_label": count_matrix.index,
        "otu_id":    kept_otu_ids,
        "genus":     kept_genus_names,
    })

    # Save
    out_counts = ROOT_DIR / "aldex2_counts.csv"
    out_meta   = ROOT_DIR / "aldex2_metadata.csv"
    out_ann    = ROOT_DIR / "aldex2_taxon_annotation.csv"

    count_matrix.to_csv(out_counts)
    taxon_ann.to_csv(out_ann, index=False)
    print(f"  Saved aldex2_taxon_annotation.csv (OTU_ID → genus lookup)")
    meta_df.to_csv(out_meta, index=False)

    print(f"  Saved aldex2_counts.csv   ({count_matrix.shape[0]} taxa × {count_matrix.shape[1]} samples)")
    print(f"  Saved aldex2_metadata.csv ({len(meta_df)} samples)")

    # Quick summary
    print(f"\n  Race distribution in metadata:")
    print(meta_df["race"].value_counts().to_string())
    print(f"\n  Bioproject distribution:")
    print(meta_df["bioproject"].value_counts().to_string())

if __name__ == "__main__":
    main()
