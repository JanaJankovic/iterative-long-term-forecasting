from typing import Any, Dict
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import torch

from src.utils.data_class import LatentForecasterConfig
from src.utils.model_utils import build_regressor


def iterative_forecast_latent_1step(
    forecaster,
    z_start: np.ndarray,  # (L,)
    n_steps: int,
) -> np.ndarray:
    """
    One-step iterative rollout:
      z_t -> z_{t+1} repeated n_steps times.
    Returns: (n_steps, L)
    """
    z = z_start.reshape(1, -1)
    out = []
    for _ in range(n_steps):
        z = forecaster.predict(z)  # (1,L)
        out.append(z.reshape(-1))
    return np.vstack(out)


class TabularLatentForecastPipeline:
    """
    Pipeline pieces:
      1) TabularAutoencoder (PyTorch) to compress X_t -> z_t and decode z -> X_hat
      2) sklearn forecaster to predict z_t -> z_{t+1} (iteratively)
    """

    def __init__(self, ae_model, cfg: LatentForecasterConfig):
        self.ae = ae_model
        self.cfg = cfg
        self.forecaster = None  # set after fit

    @torch.no_grad()
    def encode_numpy(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """
        X: (N,F) -> Z: (N,L)
        Uses mu (not sampled z) if VAE.
        """
        self.ae.eval()
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        mu, logvar, z = self.ae.encoder(xb)
        Z = mu if (self.cfg.use_mu_not_sample or logvar is None) else z
        return Z.detach().cpu().numpy()

    @torch.no_grad()
    def decode_numpy(self, Z: np.ndarray, device: torch.device) -> np.ndarray:
        """
        Z: (N,L) -> X_hat: (N,F)
        """
        self.ae.eval()
        zb = torch.tensor(Z, dtype=torch.float32, device=device)
        x_hat = self.ae.decoder(zb)
        return x_hat.detach().cpu().numpy()

    def fit_latent_forecaster(self, Z_train: np.ndarray) -> None:
        """
        Fit one-step latent transition:
          Z_train[:-1] -> Z_train[1:]
        """
        base = build_regressor(self.cfg.regressor_name, self.cfg.regressor_params)
        self.forecaster = MultiOutputRegressor(base)
        self.forecaster.fit(Z_train[:-1], Z_train[1:])

    def forecast(
        self,
        X_all: np.ndarray,  # (N,F) scaled or raw, your choice
        n_train: int,
        horizon: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        1) Encode full sequence to Z_all
        2) Fit forecaster on train latents only
        3) Roll out horizon steps from last train latent
        4) Decode predicted latents to X_hat
        5) Return y_hat from column 0 (assumes column 0 is 'y' like your features_df)
        """
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        Z_all = self.encode_numpy(X_all, device=device)
        Z_train = Z_all[:n_train]
        Z_test = Z_all[n_train:]

        self.fit_latent_forecaster(Z_train)

        effective_h = min(horizon, len(Z_test))
        z_start = Z_train[-1]
        Z_hat = iterative_forecast_latent_1step(
            self.forecaster, z_start=z_start, n_steps=effective_h
        )

        X_hat = self.decode_numpy(Z_hat, device=device)  # (effective_h, F)
        y_hat = X_hat[:, 0]  # predicted series (in same space as X_all)

        return {
            "effective_horizon": effective_h,
            "Z_hat": Z_hat,
            "X_hat": X_hat,
            "y_pred": y_hat,
        }
