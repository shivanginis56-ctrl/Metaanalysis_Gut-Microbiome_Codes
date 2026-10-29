"""
autoencoder_v2.py
─────────────────
Extended autoencoder with two new training options on top of the baseline:

  1. Sparse Bottleneck (L1 penalty)
       loss = MSE + λ · mean(|z|)
       Forces each sample to be represented by a small subset of active
       latent dimensions, potentially sharpening the core to a more
       essential set of taxa.
       Controlled by: l1_lambda  (default 0.0 = off)

  2. Prevalence-Weighted MSE
       loss = mean( w_i · (x̂_i − x_i)² )
       Assigns higher reconstruction weight to high-prevalence taxa so the
       model prioritises accurately encoding taxa that are consistently
       present across the group.
       Controlled by: feature_weights  (default None = plain MSE)

Everything else (architecture, CLR preprocessing, CoreScore, etc.) is
identical to autoencoder.py — this file re-imports and extends it.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Re-use all unchanged components from the baseline
from pipeline.autoencoder import (
    CoreAutoencoder,
    EarlyStopping,
    multiplicative_replacement,
    clr_transform_group,
    importance_reconstruction_mse,
    importance_weight_magnitude,
    importance_gradient_saliency,
    importance_clr_weighted_mse,
    get_feature_importance,
)


# ── Extended Training ──────────────────────────────────────────────────────────

def train_autoencoder_v2(
    X_clr:           np.ndarray,
    n_features:      int,
    hidden_dims:     list[int]         = [512, 128],
    bottleneck:      int               = 64,
    dropout:         float             = 0.1,
    epochs:          int               = 150,
    batch_size:      int | None        = None,
    lr:              float             = 1e-3,
    weight_decay:    float             = 1e-4,
    patience:        int               = 20,
    device_str:      str               = "auto",
    verbose:         bool              = True,
    # ── new parameters ───────────────────────────────────────────────────────
    l1_lambda:       float             = 0.0,
    feature_weights: np.ndarray | None = None,
) -> tuple[CoreAutoencoder, list[float]]:
    """
    Train CoreAutoencoder with optional L1 sparsity and/or weighted MSE.

    Parameters
    ----------
    l1_lambda
        L1 penalty on bottleneck activations. 0 = disabled.
        Typical range: 1e-5 (mild) to 1e-3 (aggressive).
        Effect: forces sparse latent codes; each sample represented by
        only a few active bottleneck dimensions.

    feature_weights
        (n_features,) prevalence array.  When provided, the reconstruction
        loss becomes a weighted MSE so high-prevalence taxa dominate the
        gradient.  Weights are normalised internally to mean ≈ 1.
        None = standard unweighted MSE.

    All other parameters are identical to train_autoencoder().
    """
    # ── Device ───────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # ── Batch size ───────────────────────────────────────────────────────────
    if batch_size is None:
        batch_size = max(16, min(128, len(X_clr) // 4))

    # ── DataLoader ───────────────────────────────────────────────────────────
    X_tensor = torch.FloatTensor(X_clr)
    loader   = DataLoader(
        TensorDataset(X_tensor),
        batch_size = batch_size,
        shuffle    = True,
        drop_last  = (len(X_clr) > batch_size),
    )

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model     = CoreAutoencoder(n_features, hidden_dims, bottleneck, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-5
    )
    stopper = EarlyStopping(patience=patience)

    # Pre-build normalised feature weight tensor
    w_tensor: torch.Tensor | None = None
    if feature_weights is not None:
        w = np.array(feature_weights, dtype=np.float32)
        w = w / (w.mean() + 1e-8)           # normalise → mean weight ≈ 1
        w_tensor = torch.FloatTensor(w).to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    losses = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_hat, z = model(x_batch)

            # Reconstruction loss — weighted or plain MSE
            if w_tensor is not None:
                recon_loss = (w_tensor * (x_hat - x_batch) ** 2).mean()
            else:
                recon_loss = criterion(x_hat, x_batch)

            # L1 sparsity penalty on bottleneck
            loss = recon_loss + l1_lambda * z.abs().mean() if l1_lambda > 0.0 else recon_loss

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
