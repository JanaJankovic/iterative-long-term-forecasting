# src/model/evaluation/iterative_forecast.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.architecture.autoencoder import LatentForecasterSklearn
from src.utils.data_class import IterForecastResult


def _get_last_window_from_loader(
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns the very last sample's x window from the loader.
    Assumes shuffle=False in DataLoader.
    Output: x_last (1, T, 1) on device
    """
    last_x = None
    for x, _ in loader:
        last_x = x  # (B, T, 1)
    if last_x is None:
        raise ValueError("Loader is empty.")

    x_last = last_x[-1:].to(device)  # (1, T, 1)
    return x_last


def _inverse_minmax_1d(scaler, arr_2d: np.ndarray) -> np.ndarray:
    """
    scaler: sklearn MinMaxScaler fitted on shape (n,1)
    arr_2d: (n,1)
    """
    return scaler.inverse_transform(arr_2d)


@torch.no_grad()
def iterative_forecast_from_last_val(
    model: LatentForecasterSklearn,
    val_loader: DataLoader,
    device: torch.device,
    steps_ahead: int,
    scaler=None,
) -> IterForecastResult:
    """
    Autoregressive rollout:
      - seed = last x window from validation loader
      - repeatedly call model.predict(seed) which returns next H steps
      - append predictions to history and slide the window
      - continue until 'steps_ahead' points are generated

    Inputs:
      model.predict expects x_past: (B, T, 1)
      model.predict returns y_hat: (B, H, 1)

    Returns:
      scaled predictions always; inverse-transformed predictions if scaler provided.
    """
    model.eval()

    x_seed = _get_last_window_from_loader(val_loader, device)  # (1, T, 1)
    B, T, D = x_seed.shape
    if B != 1 or D != 1:
        raise ValueError(f"Expected seed shape (1,T,1), got {tuple(x_seed.shape)}")

    horizon = int(getattr(model.cfg, "horizon"))
    if horizon <= 0:
        raise ValueError("Model horizon must be > 0.")
    if steps_ahead <= 0:
        raise ValueError("steps_ahead must be > 0.")

    preds = []
    remaining = steps_ahead

    # rollout
    while remaining > 0:
        y_hat = model.predict(x_seed)  # (1, H, 1)
        take = min(remaining, y_hat.shape[1])
        y_take = y_hat[:, :take, :]  # (1, take, 1)
        preds.append(y_take)

        # update seed window: drop oldest 'take' and append predicted 'take'
        if take >= T:
            # if someone sets steps chunk larger than window, keep last T points of predicted
            x_seed = y_take[:, -T:, :].contiguous()
        else:
            x_seed = torch.cat([x_seed[:, take:, :], y_take], dim=1).contiguous()

        remaining -= take

    y_pred_scaled_t = torch.cat(preds, dim=1)  # (1, S, 1)
    seed_x_scaled = x_seed.detach().cpu().float().numpy().reshape(T, 1)
    y_pred_scaled = (
        y_pred_scaled_t.detach().cpu().float().numpy().reshape(steps_ahead, 1)
    )

    if scaler is None:
        return IterForecastResult(
            seed_x_scaled=seed_x_scaled,
            y_pred_scaled=y_pred_scaled,
            seed_x=None,
            y_pred=None,
        )

    seed_x = _inverse_minmax_1d(scaler, seed_x_scaled)
    y_pred = _inverse_minmax_1d(scaler, y_pred_scaled)

    return IterForecastResult(
        seed_x_scaled=seed_x_scaled,
        y_pred_scaled=y_pred_scaled,
        seed_x=seed_x,
        y_pred=y_pred,
    )
