# latent_forecaster_sklearn.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib

import numpy as np
import torch
import torch.nn as nn

from src.model.architecture.decoders import TabularMLPDecoder
from src.model.architecture.encoders import TabularMLPEncoder


class TabularAutoencoder(nn.Module):
    """
    Convenience wrapper:
      x: (B,F) -> x_hat: (B,F)
    Exposes encoder outputs for downstream latent forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        variational: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TabularMLPEncoder(
            d_in=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            variational=variational,
        )
        self.decoder = TabularMLPDecoder(
            latent_dim=latent_dim,
            d_out=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          x_hat, mu, logvar(optional)
        """
        mu, logvar, z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
