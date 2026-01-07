# latent_forecaster_sklearn.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib

import numpy as np
import torch
import torch.nn as nn

from src.model.architecture.decoders import RNNDecoder, TabularMLPDecoder
from src.model.architecture.encoders import RNNEncoder, TabularMLPEncoder
from model.architecture.lantent import LatentRegressor
from src.utils.data_class import ModelConfig
from src.utils.model_utils import kl_div_standard_normal
from src.utils.data_utils import latent_future_to_targets, latent_to_features


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


class LatentForecasterSklearn(nn.Module):
    """
    Two-stage pipeline:
      1) Torch encoder/decoder create a structured latent space (optionally VAE-style).
      2) A simple regressor predicts future latents from past latents:
         z_past (B,T,L) -> features (B,F) -> regressor -> z_future_hat (B,H,L)
      3) Decoder maps z_future_hat to y_hat (B,H,D)

    Shapes:
      x_past: (B, T, D)
      z_past: (B, T, L)
      regressor output: (B, H*L) -> reshape to (B, H, L)
      y_hat: (B, H, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = RNNEncoder(
            d_in=cfg.d_in,
            latent_dim=cfg.latent_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            rnn_type=cfg.rnn_type,
            dropout=cfg.rnn_dropout,
            bidirectional=cfg.encoder_bidirectional,
            variational=cfg.variational,
        )

        self.decoder = RNNDecoder(
            latent_dim=cfg.latent_dim,
            d_out=cfg.d_in,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            rnn_type=cfg.rnn_type,
            dropout=cfg.rnn_dropout,
        )

        # sklearn regressor stored outside torch graph
        self.regressor: Optional[LatentRegressor] = None

    # -------------------------
    # Torch-side helpers
    # -------------------------

    def configure_optimizer(self) -> torch.optim.Optimizer:
        name = self.cfg.optim.name.lower()
        lr = float(self.cfg.optim.lr)
        wd = float(self.cfg.optim.weight_decay)

        if name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        raise ValueError(f"Unknown optimizer: {self.cfg.optim.name}")

    def forward_decoder(self, z_future: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_future)

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Uses mu (not sampled z) for stable downstream regression datasets.
        Returns: (B, T, L)
        """
        mu, logvar, z = self.encoder(x)
        return mu

    def reconstruction_loss(
        self, x: torch.Tensor, loss_fn: str = "mse"
    ) -> Dict[str, torch.Tensor]:
        """
        Train encoder+decoder as (V)AE on x -> x_hat.
        """
        mu, logvar, z = self.encoder(x)  # (B, T, L)
        x_hat = self.decoder(z)  # (B, T, D)

        if loss_fn == "mse":
            recon = torch.mean((x_hat - x) ** 2)
        elif loss_fn == "mae":
            recon = torch.mean(torch.abs(x_hat - x))
        else:
            raise ValueError("loss_fn must be 'mse' or 'mae'")

        if logvar is not None:
            kl = kl_div_standard_normal(mu, logvar)
            total = recon + self.cfg.beta_kl * kl
        else:
            kl = torch.zeros((), device=x.device, dtype=x.dtype)
            total = recon

        return {"total": total, "recon": recon, "kl": kl}

    # -------------------------
    # Regressor-side helpers
    # -------------------------

    def fit_latent_regressor(
        self,
        x_past: torch.Tensor,
        y_true: torch.Tensor,
    ) -> None:
        """
        Fits regressor to map z_past -> z_future.

        x_past: (N, T, D)
        y_true: (N, H, D)

        Target latents are computed by encoding y_true with the same encoder.
        Uses mu (not sampled z) to avoid noise in targets.
        """
        self.eval()

        with torch.no_grad():
            z_past = self.encode_mu(x_past)  # (N, T, L)
            z_future = self.encode_mu(y_true)  # (N, H, L)

        X = latent_to_features(z_past, self.cfg.latent_feature_mode)  # (N, F)
        Y = latent_future_to_targets(z_future)  # (N, H*L)

        self.regressor = LatentRegressor(
            self.cfg.regressor_name, self.cfg.regressor_params
        )
        self.regressor.fit(X, Y)

    @torch.no_grad()
    def predict(self, x_past: torch.Tensor) -> torch.Tensor:
        """
        x_past: (B, T, D)
        Returns y_hat: (B, H, D)
        """
        if self.regressor is None:
            raise RuntimeError(
                "Latent regressor not fit. Call fit_latent_regressor(...) first."
            )

        self.eval()

        z_past = self.encode_mu(x_past)  # (B, T, L)
        X = latent_to_features(z_past, self.cfg.latent_feature_mode)  # (B, F)
        z_flat = self.regressor.predict(X)  # (B, H*L)

        B = x_past.shape[0]
        H = self.cfg.horizon
        L = self.cfg.latent_dim
        z_future_hat = torch.from_numpy(np.asarray(z_flat, dtype=np.float32)).to(
            x_past.device
        )
        z_future_hat = z_future_hat.reshape(B, H, L)

        y_hat = self.decoder(z_future_hat)  # (B, H, D)
        return y_hat

    # -------------------------
    # Saving / loading
    # -------------------------

    def save(self, path: str | Path) -> None:
        """
        Saves torch weights + regressor (joblib).
        Creates:
          - <path>.pt
          - <path>.joblib
        """

        path = Path(path)
        torch.save(
            {"cfg": self.cfg, "state_dict": self.state_dict()}, path.with_suffix(".pt")
        )

        if self.regressor is None:
            joblib.dump(None, path.with_suffix(".joblib"))
        else:
            joblib.dump(self.regressor, path.with_suffix(".joblib"))

    @staticmethod
    def load(
        path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "LatentForecasterSklearn":

        path = Path(path)
        ckpt = torch.load(path.with_suffix(".pt"), map_location=map_location)
        cfg: ModelConfig = ckpt["cfg"]
        model = LatentForecasterSklearn(cfg)
        model.load_state_dict(ckpt["state_dict"])

        reg = joblib.load(path.with_suffix(".joblib"))
        model.regressor = reg
        return model
