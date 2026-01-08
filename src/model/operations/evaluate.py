from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (np.abs(b) + eps)


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """MAE, MSE, RMSE, MAPE, R2, MDA.

    Accepts:
      - (H,), (H,1) numpy/torch
      - (N,H), (N,H,1) will be flattened
    """
    yt = _to_numpy(y_true).astype(float).reshape(-1)
    yp = _to_numpy(y_pred).astype(float).reshape(-1)

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(_safe_div(np.abs(err), y_true)) * 100.0)

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    if len(yt) < 2:
        mda = float("nan")
    else:
        mda = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))))

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2, "mda": mda}
