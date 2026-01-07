from typing import Dict
import torch

from torch.utils.data import DataLoader
from src.model.architecture.autoencoder import LatentForecasterSklearn


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / (b.abs() + eps)


@torch.no_grad()
def evaluate_forecast(
    model: LatentForecasterSklearn,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Computes: MAE, MSE, RMSE, MAPE, R2, MDA.

    Shapes expected:
      x: (B, T, 1)
      y: (B, H)   (your loader)
    Model:
      y_hat: (B, H, 1)

    Returns scalar metrics over the whole loader.
    """
    model.eval()

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(device)  # (B, T, 1)
        y = y.to(device).unsqueeze(-1)  # (B, H, 1)

        y_hat = model.predict(x)  # (B, H, 1)

        y_true_all.append(y)
        y_pred_all.append(y_hat)

    if not y_true_all:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
            "mda": float("nan"),
        }

    y_true = torch.cat(y_true_all, dim=0).float()  # (N, H, 1)
    y_pred = torch.cat(y_pred_all, dim=0).float()  # (N, H, 1)

    err = y_pred - y_true

    mae = torch.mean(torch.abs(err))
    mse = torch.mean(err**2)
    rmse = torch.sqrt(mse)

    # MAPE (%), safe for near-zeros
    mape = torch.mean(_safe_div(torch.abs(err), y_true)) * 100.0

    # R2 over all points (flatten N*H*1)
    yt = y_true.view(-1)
    yp = y_pred.view(-1)
    ss_res = torch.sum((yt - yp) ** 2)
    ss_tot = torch.sum((yt - torch.mean(yt)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # MDA: mean directional accuracy on the horizon axis
    # Compare sign of delta across consecutive horizon steps.
    # If H < 2, undefined -> nan.
    if y_true.shape[1] < 2:
        mda = torch.tensor(float("nan"))
    else:
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        mda = torch.mean((torch.sign(dy_true) == torch.sign(dy_pred)).float())

    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mape": float(mape.item()),
        "r2": float(r2.item()),
        "mda": float(mda.item()) if torch.isfinite(mda) else float("nan"),
    }
