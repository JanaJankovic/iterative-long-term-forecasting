import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast_with_splits(
    y_eng: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
    n_val_end: int,
    start_idx: int,  # where forecast starts in engineered space (use n_val_end)
):
    y_eng = np.asarray(y_eng).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    H = len(y_pred)
    end = start_idx + H
    if end > len(y_eng):
        raise ValueError(f"end={end} exceeds len(y_eng)={len(y_eng)}")

    x = np.arange(len(y_eng))
    x_pred = np.arange(start_idx, end)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))

    plt.plot(x[:n_train], y_eng[:n_train], label="Train (true)", linewidth=1)
    plt.plot(
        x[n_train:n_val_end], y_eng[n_train:n_val_end], label="Val (true)", linewidth=1
    )
    plt.plot(x[n_val_end:], y_eng[n_val_end:], label="Test (true)", linewidth=1)

    plt.plot(x_pred, y_pred, label="Forecast", linewidth=2.5)

    plt.title("Forecast results (engineered index)")
    plt.xlabel("Engineered row index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_target(path: str, target: str):
    df = pd.read_csv(path)
    series = np.asarray(df[target]).reshape(-1)
    plt.plot(series, linewidth=1, color="blue")
    plt.title("Target")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
