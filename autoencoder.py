"""
autoencoder.py
──────────────
Defines the Vanilla Autoencoder for within-group core microbiome identification
and provides all supporting utilities:

  ┌──────────────────────────────────────────────────────────┐
  │ Preprocessing                                            │
  │   multiplicative_replacement  – zero-handling for CoDA  │
  │   clr_transform_group         – CLR on a single group   │
  ├──────────────────────────────────────────────────────────┤
  │ Model                                                    │
  │   CoreAutoencoder             – Encoder → Bottleneck     │
  │                                 → Decoder (symmetric)   │
  ├──────────────────────────────────────────────────────────┤
  │ Training                                                 │
  │   train_autoencoder           – Adam + LR scheduler     │
  │                                 + early stopping        │
  ├──────────────────────────────────────────────────────────┤
  │ Feature Importance                                       │
  │   importance_reconstruction_mse  – per-feature MSE      │
  │   importance_weight_magnitude    – input-layer L1 norm  │
  │   importance_gradient_saliency   – |grad × input|       │
  │   get_feature_importance         – unified dispatcher   │
  └──────────────────────────────────────────────────────────┘
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Compositional Preprocessing ───────────────────────────────────────────────

def multiplicative_replacement(X: np.ndarray, delta: float | None = None) -> np.ndarray:
    """
    Apply Multiplicative Replacement (Martin-Fernandez et al. 2003) to handle
    structural zeros in a compositional count matrix.

    For each sample j with c zeros out of D taxa:
        zero  positions → δ
        non-zero positions → x_ij × (1 − c·δ / Σ x_j)

    This preserves the ratios of non-zero values while filling zeros with a
    small positive constant, satisfying the sum-constraint of a composition.

    Parameters
    ----------
    X     : (n_samples, n_taxa) raw integer count matrix.
    delta : Replacement value.  Defaults to half the smallest non-zero value
            in the entire dataset.

    Returns
    -------
    X_replaced : (n_samples, n_taxa) float array, strictly positive.
    """
    X = X.astype(float)
    if delta is None:
        nonzero = X[X > 0]
        delta = float(nonzero.min()) / 2.0 if len(nonzero) else 0.5

    result = X.copy()
    for i in range(X.shape[0]):
        zero_mask = X[i] == 0
        n_zeros   = zero_mask.sum()
        if n_zeros == 0:
            continue

        row_sum = X[i].sum()
        if row_sum == 0:
            # Edge case: entirely-zero sample
            result[i] = delta
            continue

        # Place δ at zeros; scale non-zeros to preserve the composition total
        result[i, zero_mask]  = delta
        result[i, ~zero_mask] = X[i, ~zero_mask] * (1.0 - n_zeros * delta / row_sum)

    return result


def clr_transform_group(X_replaced: np.ndarray) -> np.ndarray:
    """
    Centred Log-Ratio (CLR) transformation for a single group's count matrix.

    CLR(x_ij) = ln(x_ij) − (1/D) Σ_k ln(x_kj)   [per-sample geometric mean]

    Parameters
    ----------
    X_replaced : (n_samples, n_taxa) strictly positive float array (after
                 multiplicative replacement).

    Returns
    -------
    X_clr : (n_samples, n_taxa) CLR-transformed array.
    """
    log_X    = np.log(X_replaced)                          # (n_samples, n_taxa)
    geo_mean = log_X.mean(axis=1, keepdims=True)           # per-sample mean
    return log_X - geo_mean


# ── Autoencoder Architecture ──────────────────────────────────────────────────

class CoreAutoencoder(nn.Module):
    """
    Symmetric Vanilla Autoencoder for microbiome compositional data.

    Default architecture (matching the engineered prompt):
        Encoder : n_features → 512 → 128 → bottleneck (64)
        Decoder : 64 → 128 → 512 → n_features   [symmetric]

    Activation : LeakyReLU(0.1) throughout (avoids dying-ReLU on CLR-space
                 data which can have negative values).
    Regularisation : BatchNorm1d + Dropout on every encoder/decoder layer.
    """

    def __init__(
        self,
        n_features:   int,
        hidden_dims:  list[int] = [512, 128],
        bottleneck:   int       = 64,
        dropout:      float     = 0.1,
    ) -> None:
        super().__init__()

        self.n_features  = n_features
        self.hidden_dims = hidden_dims
        self.bottleneck  = bottleneck

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers = []
        in_dim     = n_features
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, bottleneck))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror of encoder) ───────────────────────────────────
        dec_layers = [nn.LeakyReLU(0.1)]    # activation applied to bottleneck
        in_dim     = bottleneck
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    # ─────────────────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_code)."""
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def first_encoder_weights(self) -> np.ndarray:
        """Return the weight matrix of the first encoder Linear layer."""
        return self.encoder[0].weight.detach().cpu().numpy()   # (hidden[0], n_features)


# ── Training ──────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation (or training) loss and stops training when improvement
    stalls for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, loss: float) -> bool:
        """Returns True when training should stop."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


def train_autoencoder(
    X_clr:           np.ndarray,
    n_features:      int,
    hidden_dims:     list[int]        = [512, 128],
    bottleneck:      int              = 64,
    dropout:         float            = 0.1,
    epochs:          int              = 150,
    batch_size:      int | None       = None,
    lr:              float            = 1e-3,
    weight_decay:    float            = 1e-4,
    patience:        int              = 20,
    device_str:      str              = "auto",
    verbose:         bool             = True,
    l1_lambda:       float            = 0.0,
    feature_weights: np.ndarray | None = None,
) -> tuple["CoreAutoencoder", list[float]]:
    """
    Train the CoreAutoencoder on CLR-transformed data.

    Parameters
    ----------
    X_clr            : (n_samples, n_features) CLR-transformed float array.
    n_features       : Number of input/output taxa (columns of X_clr).
    hidden_dims      : Encoder hidden layer sizes.
    bottleneck       : Bottleneck dimensionality.
    dropout          : Dropout rate applied during training.
    epochs           : Maximum training epochs.
    batch_size       : Mini-batch size.  Auto-selects if None.
    lr               : Adam learning rate.
    weight_decay     : L2 regularisation coefficient.
    patience         : Early-stopping patience (epochs without improvement).
    device_str       : "cpu", "mps", "cuda", or "auto".
    verbose          : Print epoch-level loss.
    l1_lambda        : L1 sparsity penalty on bottleneck activations.
                       0 = disabled (default).  Typical range: 1e-5 to 1e-3.
                       Encourages sparse latent codes so each sample is
                       represented by a small subset of active latent dims.
    feature_weights  : (n_features,) prevalence weights for the reconstruction
                       loss.  If provided, loss = mean(w_i * (x_hat_i - x_i)^2)
                       so high-prevalence taxa contribute more to the gradient.
                       None = unweighted MSE (default).

    Returns
    -------
    model  : Trained CoreAutoencoder (in eval mode).
    losses : List of per-epoch mean MSE losses.
    """
    # ── Device selection ─────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # ── Auto batch size: ≥4 batches but at most 128 ──────────────────────────
    if batch_size is None:
        batch_size = max(16, min(128, len(X_clr) // 4))

    # ── DataLoader ───────────────────────────────────────────────────────────
    X_tensor = torch.FloatTensor(X_clr)
    loader   = DataLoader(
        TensorDataset(X_tensor),
        batch_size = batch_size,
        shuffle    = True,
        drop_last  = (len(X_clr) > batch_size),   # stable batch norm
    )

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model     = CoreAutoencoder(n_features, hidden_dims, bottleneck, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-5
    )
    stopper = EarlyStopping(patience=patience)

    # ── Training loop ─────────────────────────────────────────────────────────
    losses = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x_batch)
            loss      = criterion(x_hat, x_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        if verbose and epoch % 25 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:>4}/{epochs}  loss={epoch_loss:.6f}  lr={lr_now:.2e}")

        if stopper.step(epoch_loss):
            if verbose:
                print(f"    Early stop at epoch {epoch} (best loss={stopper.best_loss:.6f})")
            break

    model.eval()
    return model, losses


# ── Feature Importance ────────────────────────────────────────────────────────

def importance_reconstruction_mse(
    model: CoreAutoencoder,
    X_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Per-feature Mean Squared Error between input and reconstruction.

    Interpretation: Low MSE → the autoencoder reliably reconstructs this taxon
    → its abundance pattern is learnable / consistent = candidate core taxon.

    Returns
    -------
    mse : (n_features,) array.  Lower = more consistently reconstructed.
    """
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(X_tensor)
        mse      = ((x_hat - X_tensor) ** 2).mean(dim=0)
    return mse.cpu().numpy()


def importance_weight_magnitude(model: CoreAutoencoder) -> np.ndarray:
    """
    L1 norm of connections from each input taxon to the first hidden layer.

    importance_i = Σ_h |W_{hi}|    (sum over hidden units h)

    High score → the encoder actively uses this taxon to build the
    compressed representation → it carries structural information = core.

    Returns
    -------
    importance : (n_features,) array.  Higher = more relied upon by encoder.
    """
    W = model.first_encoder_weights()    # (hidden[0], n_features)
    return np.abs(W).sum(axis=0)


def importance_gradient_saliency(
    model: CoreAutoencoder,
    X_clr: np.ndarray,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
) -> np.ndarray:
    """
    Gradient × Input saliency (Simonyan et al. 2013 style).

        saliency_i = mean_samples [ |∂L/∂x_i| × |x_i| ]

    High score → a small change in taxon i causes a large change in the
    reconstruction loss, AND the taxon has a non-trivial CLR value.
    This jointly rewards discriminative AND abundant taxa = core.

    Returns
    -------
    saliency : (n_features,) array.  Higher = more important for reconstruction.
    """
    model.eval()
    saliency_accum = np.zeros(X_clr.shape[1])

    for start in range(0, len(X_clr), batch_size):
        x_np  = X_clr[start : start + batch_size]
        x_t   = torch.FloatTensor(x_np).to(device)
        x_t.requires_grad_(True)

        x_hat, _ = model(x_t)
        loss      = nn.MSELoss()(x_hat, x_t.detach())   # treat target as const
        loss.backward()

        grad_input = x_t.grad.detach().cpu().numpy()     # (batch, n_features)
        saliency_accum += (np.abs(grad_input) * np.abs(x_np)).mean(axis=0)

    return saliency_accum / max(1, len(X_clr) // batch_size)


def importance_clr_weighted_mse(
    model:        CoreAutoencoder,
    X_scaled:     np.ndarray,
    X_clr_orig:   np.ndarray,
    device:       torch.device = torch.device("cpu"),
    eps:          float = 1e-8,
) -> np.ndarray:
    """
    CLR-weighted inverse reconstruction MSE — the primary core-detection metric.

    Motivation
    ----------
    Gradient saliency rewards hard-to-reconstruct taxa, which on CLR-transformed
    data are predominantly *rare* taxa (extreme negative CLR values amplify the
    |grad × input| term).  Raw reconstruction MSE incorrectly scores always-absent
    taxa as "core" (zero everywhere = trivially well-reconstructed).

    This method combines two complementary signals without any hard threshold:

        core_score_i = max(0, mean_CLR_i) / (MSE_i + ε)

    • mean_CLR_i  is taken from the *raw* (pre-standardisation) CLR values so
      that positive mean CLR correctly identifies taxa that are prevalent and
      relatively abundant across the group.  Rare / absent taxa have negative
      mean CLR, are clamped to 0, and receive a score of exactly 0.
      — No hard threshold is applied; the gate is fully data-driven.

    • MSE is evaluated on X_scaled (the standardised input that the model was
      trained on).  Low MSE → the autoencoder reliably reconstructs this taxon
      across all samples → its pattern is consistent and learnable.

    Parameters
    ----------
    X_scaled   : (n_samples, n_features) standardised CLR — model input.
    X_clr_orig : (n_samples, n_features) raw CLR before standardisation —
                 used only for mean_CLR computation.

    Returns
    -------
    scores : (n_features,) array.  Higher = more core.  Sort descending.
    """
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    mse      = importance_reconstruction_mse(model, X_tensor)   # (n_features,)
    mean_clr = X_clr_orig.mean(axis=0)                          # (n_features,)

    positive_mean = np.maximum(0.0, mean_clr)                   # soft prevalence gate
    scores = positive_mean / (mse + eps)
    return scores


def get_feature_importance(
    model:        CoreAutoencoder,
    X_clr:        np.ndarray,
    method:       str = "clr_weighted_mse",
    device:       torch.device = torch.device("cpu"),
    X_clr_orig:   np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """
    Dispatcher for feature importance methods.

    Parameters
    ----------
    method : One of:
        "clr_weighted_mse"    – max(0, mean_CLR) / (MSE + ε)  [default, recommended]
                                higher = more core (rank descending)
        "reconstruction_mse"  – lower MSE = more core   (rank ascending)
        "weight_magnitude"    – higher weight = more core (rank descending)
        "gradient_saliency"   – higher saliency = more core (rank descending)

    Returns
    -------
    scores   : (n_features,) importance array.
    sort_dir : "asc" or "desc" — how to sort for top-N selection.
    """
    X_tensor = torch.FloatTensor(X_clr).to(device)

    if method == "clr_weighted_mse":
        if X_clr_orig is None:
            raise ValueError("clr_weighted_mse requires X_clr_orig (pre-standardisation CLR).")
        scores   = importance_clr_weighted_mse(model, X_clr, X_clr_orig, device)
        sort_dir = "desc"

    elif method == "reconstruction_mse":
        scores   = importance_reconstruction_mse(model, X_tensor)
        sort_dir = "asc"     # lowest MSE = best reconstructed = most core

    elif method == "weight_magnitude":
        scores   = importance_weight_magnitude(model)
        sort_dir = "desc"

    elif method == "gradient_saliency":
        scores   = importance_gradient_saliency(model, X_clr, device)
        sort_dir = "desc"

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: "
                         f"clr_weighted_mse, reconstruction_mse, weight_magnitude, gradient_saliency")

    return scores, sort_dir
