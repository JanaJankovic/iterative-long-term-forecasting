import torch
import torch.nn as nn
from src.model.architecture.mlp import MLP


class TabularMLPDecoder(nn.Module):
    """
    For tabular features:
      z: (B, L) -> x_hat: (B, F)

    Mirrors your mentor’s decoder:
      Dense(hidden) -> Dense(input_dim)
    """

    def __init__(
        self,
        latent_dim: int,  # L
        d_out: int,  # F
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_out = d_out

        self.net = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=d_out,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Tabular decoder expects (B,L), got {tuple(z.shape)}")
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: got {z.shape[-1]}, expected {self.latent_dim}"
            )
        return self.net(z)  # (B, F)


class RNNDecoder(nn.Module):
    """
    z: (B, H, L) -> y: (B, H, D)
    """

    def __init__(
        self,
        latent_dim: int,
        d_out: int,
        rnn_hidden: int,
        rnn_layers: int,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_out = d_out

        rnn_type = rnn_type.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn = rnn_cls(
            input_size=latent_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(rnn_hidden, d_out)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, H, L = z.shape
        if L != self.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: got {L}, expected {self.latent_dim}"
            )
        h, _ = self.rnn(z)  # (B, H, rnn_hidden)
        y = self.out_proj(h)  # (B, H, D)
        return y
