# train_latent_forecaster.py
from __future__ import annotations


from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json

from src.model.operations.evaluate import evaluate_forecast
from src.data.preprocess import process_sequences
from src.utils.data_class import ModelConfig, OptimConfig
from src.model.architecture.autoencoder import LatentForecasterSklearn
from src.utils.data_utils import collect_tensors_from_loader
from src.utils.model_utils import get_device


# -----------------------------
# Stage 1: Train AE (encoder+decoder)
# -----------------------------
def train_autoencoder(
    model: LatentForecasterSklearn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    loss_fn: str = "mse",
    grad_clip: Optional[float] = 1.0,
    early_stop_patience: int = 7,
) -> Dict[str, float]:
    """
    Trains encoder+decoder to reconstruct input windows.
    Regresor is NOT trained here.
    """
    optim = model.configure_optimizer()
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, _ in train_loader:
            x = x.to(device)  # (B, T, 1)

            optim.zero_grad(set_to_none=True)
            losses = model.reconstruction_loss(x, loss_fn=loss_fn)
            loss = losses["total"]
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
                losses = model.reconstruction_loss(x, loss_fn=loss_fn)
                val_losses.append(losses["total"].item())

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"[AE] epoch {epoch:03d} | train {train_mean:.6f} | val {val_mean:.6f}")

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


# -----------------------------
# Stage 2: Fit latent regressor
# -----------------------------
def fit_latent_regressor_from_loaders(
    model: LatentForecasterSklearn,
    train_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> None:
    """
    Fits the sklearn regressor on encoded train data.
    Uses all train samples by default; can subsample if needed.
    """
    model.eval()

    x_all, y_all = collect_tensors_from_loader(train_loader, device=device)

    if max_samples is not None and x_all.shape[0] > max_samples:
        idx = torch.randperm(x_all.shape[0], device=device)[:max_samples]
        x_all = x_all[idx]
        y_all = y_all[idx]

    # Fit regressor z_past -> z_future using mu-encodings
    model.fit_latent_regressor(x_all, y_all)


def run(
    csv_path: str,
    lookback: int,
    horizon: int,
    split_ratio,
    batch_size: int,
    cfg: ModelConfig,
    epochs_ae: int = 30,
    device: Optional[torch.device] = None,
    out_dir: str = "output",
) -> None:
    device = device or get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    loaders, scaler, input_shape = process_sequences(
        data=data,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        split_ratio=split_ratio,
    )
    train_loader, val_loader, test_loader = loaders

    # Sanity on shapes: X is (B, T, 1); y is (B, H)
    assert input_shape == (lookback, 1), f"Expected (lookback,1) but got {input_shape}"

    # Build model
    model = LatentForecasterSklearn(cfg).to(device)

    # Stage 1: AE training on x windows
    ae_stats = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs_ae,
        loss_fn="mse",
        grad_clip=1.0,
        early_stop_patience=7,
    )

    # Stage 2: fit latent regressor on train set
    fit_latent_regressor_from_loaders(
        model, train_loader, device=device, max_samples=None
    )

    # Evaluate forecast quality (uses regressor + decoder)
    val_metrics = evaluate_forecast(model, val_loader, device=device)
    test_metrics = evaluate_forecast(model, test_loader, device=device)

    print(f"[FORECAST] val:  {val_metrics}")
    print(f"[FORECAST] test: {test_metrics}")

    # Save artifacts
    tag = Path(csv_path).stem
    save_base = (
        out_dir / "models" / f"{tag}_lb{lookback}_h{horizon}_{cfg.regressor_name}"
    )
    model.save(str(save_base))

    # Save config + metrics
    report = {
        "csv": csv_path,
        "lookback": lookback,
        "horizon": horizon,
        "batch_size": batch_size,
        "device": str(device),
        "cfg": asdict(cfg),
        "ae": ae_stats,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(save_base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return loaders, scaler, model
