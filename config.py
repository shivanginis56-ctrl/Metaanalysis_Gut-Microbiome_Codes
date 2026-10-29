"""
config.py
─────────
Central configuration for the Core Microbiome ML Pipeline.
Edit the values in this file to change data paths, ML hyperparameters,
or filtering thresholds without touching any other module.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent        # Datasets/
METADATA_FILE = ROOT_DIR / "metadata_clean.csv"
ABUNDANCE_FILE= ROOT_DIR / "Taxonomic_abundance.csv"
ORIGINAL_CORE = ROOT_DIR / "core micrbiome.csv"
PIPELINE_DIR  = ROOT_DIR / "pipeline"

# Output files written by main.py
OUTPUT_TIDY       = ROOT_DIR / "core_microbiome_pipeline.csv"   # ML results (long format)
OUTPUT_COMPARISON = ROOT_DIR / "core_microbiome_comparison.txt" # vs original manual file

# ── Data filtering (applied before ML) ────────────────────────────────────────
# Minimum prevalence across ALL samples required to keep a taxon in the feature space.
# Drops taxa that are nearly always zero (noise), reducing dimensionality.
GLOBAL_PREVALENCE_MIN = 0.05   # taxon must be present in ≥5 % of all samples

# ── CLR transform ─────────────────────────────────────────────────────────────
# Pseudocount added to every cell before log transform to handle structural zeros.
# 0.5 is standard in microbiome CLR literature (Aitchison 1986).
CLR_PSEUDOCOUNT = 0.5

# ── Random Forest hyperparameters ─────────────────────────────────────────────
RF_N_ESTIMATORS = 300   # number of trees
RF_MAX_FEATURES = "sqrt"   # features per split (default for classification)
RF_RANDOM_STATE = 42
RF_N_JOBS       = -1    # use all CPU cores

# How many top-ranked taxa (by RF importance) to pass into RFECV.
# Limits RFECV to a tractable feature space after the first RF ranking pass.
RF_TOP_N_FOR_RFECV = 100

# ── RFECV hyperparameters ─────────────────────────────────────────────────────
RFECV_STEP     = 5    # remove 5 features per iteration
RFECV_CV_FOLDS = 5    # stratified K-fold CV
RFECV_SCORING  = "roc_auc"  # optimise area under ROC curve

# ── Classification problem setup ───────────────────────────────────────────────
# Minimum positive samples required to attempt classification for a group.
# Groups smaller than this are skipped (RFECV is unreliable on very small sets).
MIN_POSITIVE_SAMPLES = 15

# When a bioproject has only one race, we compare that group against:
#   "same_race_other_bioprojects" → other bioprojects with the same race label
#   "all_others"                  → all remaining samples
SINGLE_RACE_NEGATIVE_STRATEGY = "same_race_other_bioprojects"

# ── Runtime cap: subsample large groups ───────────────────────────────────────
# RF + RFECV can be slow for groups with thousands of samples.
# If a class (positive or negative) exceeds this limit, it is randomly
# subsampled down to MAX_SAMPLES_PER_CLASS before training.
# Set to None to disable subsampling entirely.
MAX_SAMPLES_PER_CLASS = None

# ── Bioprojects flagged as "Corrected" in the original manual file ────────────
CORRECTED_BIOPROJECTS = {"PRJNA388263", "PRJNA562529", "PRJNA395034"}
