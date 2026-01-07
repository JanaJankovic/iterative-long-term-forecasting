from typing import Tuple
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_sequence(data, lookback, horizon, stride=1):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1, stride):
        X.append(data[i - lookback : i])
        y.append(data[i : i + horizon])
    return np.array(X), np.array(y).squeeze(-1)


def process_sequences(data, lookback, horizon, batch_size, split_ratio):
    df = data[["load"]]

    # SPLIT DATA
    total_len = len(df)
    train_end = int(total_len * split_ratio[0])
    val_end = train_end + int(total_len * split_ratio[1])

    train_split = df.iloc[:train_end].copy()
    val_split = df.iloc[train_end:val_end].copy()
    test_split = df.iloc[val_end:].copy()

    # SCALE DATA
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_split)
    val_scaled = scaler.transform(val_split)
    test_scaled = scaler.transform(test_split)

    # BUILD SEQUENCES
    X_train, y_train = build_sequence(train_scaled, lookback, horizon)
    X_val, y_val = build_sequence(val_scaled, lookback, horizon)
    X_test, y_test = build_sequence(test_scaled, lookback, horizon)

    # CREATE DATA LOADERS
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (N, L, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # (N, H)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    loaders = (
        DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
    )

    input_shape = X_train.shape[1:]

    return loaders, scaler, input_shape


def create_features_from_series(
    series: pd.Series,
    lags: Tuple[int, ...] = (1, 2, 7, 14, 30),
    rolling_windows: Tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """
    Tabular feature matrix for daily series.
    Requires datetime index (daily).
    Output columns include:
      y, calendar features, lag_k, roll_mean_k
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")

    df = pd.DataFrame({"y": series.astype(float)})

    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for win in rolling_windows:
        df[f"roll_mean_{win}"] = df["y"].rolling(window=win).mean()

    df = df.dropna()
    return df
