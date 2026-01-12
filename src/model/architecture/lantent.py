from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import torch

from src.data.preprocess import create_features_from_series
from src.utils.data_class import DataConfig, LatentForecasterConfig
from src.utils.model_utils import build_regressor
import pandas as pd


class TabularLatentTransition:
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

    def __iterative_forecast_latent_1step(
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
        Z_hat = self.__iterative_forecast_latent_1step(
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


class TabularLatentReadout:
    """
    AE: X_t -> z_t
    Regressor: z_t -> y_{t+1} (scaled)
    Rollout: predict y_{t+1}, append to history, rebuild features, repeat
    """

    def __init__(
        self,
        ae_model,
        cfg,
        data_cfg: DataConfig,
        *,
        lags: Tuple[int, ...],
        rolling_windows: Tuple[int, ...],
        calendar_cols: List[str],
        numeric_cols: List[str],
        nominal_cols: List[str],
        feature_names_in: List[
            str
        ],  # columns fed into AE (after cyclic + scaling), excludes y
        scalers: Dict[str, Any],  # {"numeric_scaler":..., "y_scaler":...}
        y_col: str = "y",
        freq: str = "D",
    ):
        self.ae = ae_model
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.lags = tuple(lags)
        self.rolling_windows = tuple(rolling_windows)

        self.calendar_cols = list(calendar_cols)
        self.numeric_bases = tuple(numeric_cols)
        self.nominal_cols = list(nominal_cols)

        self.feature_names_in = list(feature_names_in)
        self.scalers = scalers
        self.y_col = y_col
        self.freq = freq

        self.forecaster = None

        # fixed calendar cycles that work on single-row inputs
        self._cycle_map = {
            "dayofweek": (7, 0.0),
            "dow": (7, 0.0),
            "weekday": (7, 0.0),
            "month": (12, 1.0),
            "dayofyear": (366, 1.0),
            "doy": (366, 1.0),
            "hour": (24, 0.0),
            "minute": (60, 0.0),
            "second": (60, 0.0),
            "weekofyear": (52, 1.0),
            "woy": (52, 1.0),
        }

    @torch.no_grad()
    def encode_numpy(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        self.ae.eval()
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        mu, logvar, z = self.ae.encoder(xb)
        Z = (
            mu
            if (getattr(self.cfg, "use_mu_not_sample", True) or logvar is None)
            else z
        )
        return Z.detach().cpu().numpy()

    def _cyclic_encode_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.calendar_cols:
            if col not in out.columns:
                continue
            key = col.lower()
            P, offset = self._cycle_map.get(key, None) or self._cycle_map.get(
                key.replace("_", ""), None
            )
            if P is None:
                # assume already fine; just drop nothing
                continue
            v = out[col].astype(float).to_numpy()
            a = 2.0 * np.pi * ((v - offset) / float(P))
            out[f"{col}_sin"] = np.sin(a).astype(np.float32)
            out[f"{col}_cos"] = np.cos(a).astype(np.float32)
            out = out.drop(columns=[col])
        return out

    def _transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = self._cyclic_encode_inplace(features_df)

        # NEW: one-hot nominal exactly like in preprocessing
        if self.nominal_cols:
            oh = self.scalers["onehot"].transform(df[self.nominal_cols])
            if hasattr(oh, "toarray"):  # sparse -> dense
                oh = oh.toarray()
            oh_cols = self.scalers["onehot"].get_feature_names_out(self.nominal_cols)
            df_oh = pd.DataFrame(oh, index=df.index, columns=oh_cols)
            df = pd.concat([df.drop(columns=self.nominal_cols), df_oh], axis=1)

        num_scaler = self.scalers["numeric_scaler"]
        y_scaler = self.scalers["y_scaler"]

        num_cols = list(getattr(num_scaler, "feature_names_in_", []))
        if not num_cols:
            raise ValueError(
                "numeric_scaler has no feature_names_in_. Fit it on a DataFrame."
            )

        missing = [c for c in num_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing numeric cols at transform time: {missing[:10]}")

        df.loc[:, num_cols] = num_scaler.transform(df[num_cols]).astype(np.float32)

        df.loc[:, [self.y_col]] = y_scaler.transform(df[[self.y_col]]).astype(
            np.float32
        )

        return df

    def fit(
        self,
        n_train_rows: int,
        device: torch.device,
    ) -> None:

        df_raw = pd.read_csv(self.data_cfg.csv_path)

        dt = pd.to_datetime(df_raw[self.data_cfg.datetime_col], errors="coerce")
        m = dt.notna()
        df_raw = df_raw.loc[m].copy()
        dt = dt.loc[m]

        series = pd.Series(
            df_raw[self.data_cfg.target_col].astype(float).to_numpy(),
            index=dt,
            name="y",
        ).sort_index()

        feats = create_features_from_series(
            series, lags=self.lags, rolling_windows=self.rolling_windows
        )
        df = self._transform(feats)

        X_all = df[self.feature_names_in].to_numpy(dtype=np.float32)
        y_all = df[self.y_col].to_numpy(dtype=np.float32)

        # train pairs: z_t -> y_{t+1} inside train block
        X_in = X_all[: n_train_rows - 1]
        y_out = y_all[1:n_train_rows]

        Z_in = self.encode_numpy(X_in, device=device)

        self.forecaster = build_regressor(
            self.cfg.regressor_name, self.cfg.regressor_params
        )
        self.forecaster.fit(Z_in, y_out)

    def forecast(
        self,
        start_row: int,  # use n_val_end here
        horizon: int,
        device: torch.device,
        idx_all: pd.DatetimeIndex,  # pass meta["idx_all"]
    ) -> Dict[str, Any]:
        df_raw = pd.read_csv(self.data_cfg.csv_path)

        dt = pd.to_datetime(df_raw[self.data_cfg.datetime_col], errors="coerce")
        m = dt.notna()
        df_raw = df_raw.loc[m].copy()
        dt = dt.loc[m]

        series = pd.Series(
            df_raw[self.data_cfg.target_col].astype(float).to_numpy(),
            index=dt,
            name="y",
        ).sort_index()

        # start timestamp = last available engineered row BEFORE rollout
        start_ts = idx_all[start_row - 1]

        # history up to that timestamp (raw series slice)
        y_hist = series.loc[:start_ts].copy()
        y_scaler = self.scalers["y_scaler"]

        y_pred = []
        y_pred_scaled = []

        for k in range(horizon):
            feats_t = create_features_from_series(
                y_hist, lags=self.lags, rolling_windows=self.rolling_windows
            ).iloc[[-1]]

            df_t = self._transform(feats_t)
            X_t = df_t[self.feature_names_in].to_numpy(dtype=np.float32)
            Z_t = self.encode_numpy(X_t, device=device)

            y_next_s = float(self.forecaster.predict(Z_t).reshape(-1)[0])
            y_next = float(y_scaler.inverse_transform([[y_next_s]])[0, 0])

            # force next timestamp to follow engineered timeline (no weekends drift)
            next_ts = (
                idx_all[start_row + k]
                if (start_row + k) < len(idx_all)
                else (y_hist.index[-1] + pd.tseries.frequencies.to_offset(self.freq))
            )
            y_hist.loc[next_ts] = y_next

            y_pred_scaled.append(y_next_s)
            y_pred.append(y_next)

        return {
            "y_pred_scaled": np.asarray(y_pred_scaled, dtype=np.float32),
            "y_pred": np.asarray(y_pred, dtype=np.float32),
            "start_timestamp": start_ts,
            "start_row": start_row,
        }
