import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from typing import Dict
from src.model.architecture.autoencoder import TabularAutoencoder
from src.utils.data_class import ModelConfig, TrainConfig
from src.utils.model_utils import configure_optimizer


def _kl_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mean KL across batch and dimensions
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def fit_tabular_autoencoder(
    model: TabularAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    verbose=False,
) -> Dict[str, float]:
    """
    Trains encoder+decoder to reconstruct input windows.
    Regresor is NOT trained here.
    """
    model.to(device)
    optim = configure_optimizer(model_cfg.optim, model)

    best_val = float("inf")
    best_state = None
    bad = 0

    epochs = train_cfg.epochs_ae
    loss_fn = train_cfg.loss_fn
    grad_clip = train_cfg.grad_clip
    early_stop_patience = train_cfg.early_stop_patience

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, _ in train_loader:
            x = x.to(device)  # (B, T, 1)

            optim.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(x)

            if loss_fn == "mae":
                recon = F.l1_loss(x_hat, x)
            else:
                recon = F.mse_loss(x_hat, x)

            if model_cfg.variational:
                if logvar is None:
                    raise ValueError("variational=True but model returned logvar=None")
                kl = _kl_std_normal(mu, logvar)
                loss = recon + model_cfg.beta_kl * kl
            else:
                loss = recon

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optim.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)

                if loss_fn == "mae":
                    recon = F.l1_loss(x_hat, x)
                else:
                    recon = F.mse_loss(x_hat, x)

                if model_cfg.variational:
                    kl = _kl_std_normal(mu, logvar)
                    loss = recon + model_cfg.beta_kl * kl
                else:
                    loss = recon

                val_losses.append(float(loss.item()))

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")

        if verbose:
            print(
                f"[AE] epoch {epoch:03d} | train {train_mean:.6f} | val {val_mean:.6f}"
            )

        if val_mean < best_val:
            best_val = val_mean
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_ae": best_val}
