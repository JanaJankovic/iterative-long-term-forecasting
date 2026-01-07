import torch
import torch.nn as nn
from src.utils.model_utils import reparameterize
from src.model.architecture.mlp import MLP
from typing import Optional, Tuple


class TabularMLPEncoder(nn.Module):
    """
    For tabular features:
      x: (B, F) -> mu: (B, L), logvar: (B, L) optional, z: (B, L)

    Mirrors your mentor’s Keras AE idea:
      Dense(hidden) -> Dense(latent)
    but supports variational mode if you want it.
    """

    def __init__(
        self,
        d_in: int,  # F
        latent_dim: int,  # L
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        variational: bool = False,
    ) -> None:
        super().__init__()
        self.variational = variational
        self.latent_dim = latent_dim

        out_dim = 2 * latent_dim if variational else latent_dim

        self.net = MLP(
            in_dim=d_in,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Tabular encoder expects (B,F), got {tuple(x.shape)}")

        h = self.net(x)  # (B, 2L) or (B, L)

        if self.variational:
            mu, logvar = torch.split(h, self.latent_dim, dim=-1)
            z = reparameterize(mu, logvar)
            return mu, logvar, z

        mu = h
        return mu, None, mu


class RNNEncoder(nn.Module):
    """
    x: (B, T, D) -> (mu, logvar, z): each (B, T, L)
    """

    def __init__(
        self,
        d_in: int,
        latent_dim: int,
        rnn_hidden: int,
        rnn_layers: int,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        bidirectional: bool = False,
        variational: bool = True,
    ) -> None:
        super().__init__()
        self.variational = variational
        self.latent_dim = latent_dim

        rnn_type = rnn_type.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn = rnn_cls(
            input_size=d_in,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        proj_in = rnn_hidden * (2 if bidirectional else 1)
        proj_out = 2 * latent_dim if variational else latent_dim
        self.proj = nn.Linear(proj_in, proj_out)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        h, _ = self.rnn(x)  # (B, T, H_rnn)
        p = self.proj(h)  # (B, T, 2L) or (B, T, L)

        if self.variational:
            mu, logvar = torch.split(p, self.latent_dim, dim=-1)
            z = reparameterize(mu, logvar)
            return mu, logvar, z

        mu = p
        return mu, None, mu
