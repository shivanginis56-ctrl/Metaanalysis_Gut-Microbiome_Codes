"""
transforms.py
─────────────
Applies the Centered Log-Ratio (CLR) transformation to compositional
microbiome count data.

Why CLR?
--------
Raw or even relative-abundance microbiome data are *compositional*: counts
in each sample are constrained to sum to a constant, so standard Euclidean
distances are misleading.  CLR maps the simplex into real space (Aitchison
1986), making the data amenable to PCA, linear models, and tree-based ML.

Formula (per sample j):
    CLR(x_ij) = log(x_ij + ε) − mean_i[ log(x_ij + ε) ]

where ε is a small pseudocount that replaces structural zeros before logging.
"""

import numpy as np
import pandas as pd

from pipeline.config import CLR_PSEUDOCOUNT


def clr_transform(
    df: pd.DataFrame,
    pseudocount: float = CLR_PSEUDOCOUNT,
) -> pd.DataFrame:
    """
    Apply the CLR transformation to a taxa × samples count matrix.

    Steps
    -----
    1. Add `pseudocount` to every cell (handles structural zeros).
    2. Take the natural log of every cell.
    3. Subtract the per-sample (column-wise) geometric mean from every cell.

    Parameters
    ----------
    df : pd.DataFrame
        Shape (n_taxa, n_samples).  Values are raw integer counts.
    pseudocount : float
        Added before log to avoid log(0).  Default 0.5 (Aitchison standard).

    Returns
    -------
    pd.DataFrame
        CLR-transformed matrix, same shape and index/column names as input.
    """
    # Step 1 – add pseudocount
    X = df.values.astype(float) + pseudocount          # (n_taxa, n_samples)

    # Step 2 – log transform
    log_X = np.log(X)

    # Step 3 – subtract geometric mean (column-wise mean of logs)
    geo_mean = log_X.mean(axis=0, keepdims=True)       # (1, n_samples)
    clr_X    = log_X - geo_mean                        # broadcast subtraction

    return pd.DataFrame(clr_X, index=df.index, columns=df.columns)


def extract_group_matrix(
    abundance_df: pd.DataFrame,
    sample_ids: list[str],
) -> pd.DataFrame:
    """
    Slice the global abundance matrix down to a specific group's samples.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Full taxa × samples matrix (raw counts, pre-filtered for rare taxa).
    sample_ids : list[str]
        Sample IDs belonging to the group of interest.

    Returns
    -------
    pd.DataFrame
        Subset of abundance_df with only the requested columns.
        Missing sample IDs (not present in the matrix) are silently ignored.
    """
    valid_ids = [sid for sid in sample_ids if sid in abundance_df.columns]
    return abundance_df[valid_ids]
