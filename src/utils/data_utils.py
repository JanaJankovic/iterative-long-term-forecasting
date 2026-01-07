import numpy as np
import torch

from typing import Tuple
from torch.utils.data import DataLoader
from src.utils.constants import LatentFeatureMode


def latent_to_features(z_past: torch.Tensor, mode: LatentFeatureMode) -> np.ndarray:
    """
    z_past: (B, T, L) torch -> features: (B, F) numpy
    """
    if z_past.ndim != 3:
        raise ValueError("z_past must be (B, T, L)")
    if mode == "last":
        feats = z_past[:, -1, :]  # (B, L)
    elif mode == "mean":
        feats = z_past.mean(dim=1)  # (B, L)
    elif mode == "flatten":
        feats = z_past.reshape(z_past.shape[0], -1)  # (B, T*L)
    else:
        raise ValueError(f"Unknown latent_feature_mode: {mode}")

    return feats.detach().cpu().float().numpy()


def latent_future_to_targets(z_future: torch.Tensor) -> np.ndarray:
    """
    z_future: (B, H, L) -> y: (B, H*L) numpy
    """
    if z_future.ndim != 3:
        raise ValueError("z_future must be (B, H, L)")
    B, H, L = z_future.shape
    return z_future.detach().cpu().float().numpy().reshape(B, H * L)


def collect_tensors_from_loader(
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collects all batches into two tensors:
      x_all: (N, T, 1)
      y_all: (N, H, 1)
    """
    xs = []
    ys = []
    for x, y in loader:
        xs.append(x.to(device))
        ys.append(y.to(device).unsqueeze(-1))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
