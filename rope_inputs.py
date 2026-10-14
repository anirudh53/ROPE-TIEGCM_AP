"""
rope_inputs.py -- input-side helpers used by rope.py.

  UnscentedTransform  sigma points / weights (alpha=1, beta=2, kappa=0 like rope.py)
  mean_state_cov      3-hour DA covariance -> 10x10 covariance of the mean state
  get_forecaster      lazily loaded, cached F10 forecaster
  f10_daily_forecast  daily F10 forecast covering a ROPE horizon
  forecast_driver_df  driver slice with forecast F10 (+ offset) after the start hour

No import of rope.py here, so rope.py can import this module.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
F10_DIR = os.path.join(THIS_DIR, "F10_Forecasting")
F10_HISTORY = os.path.join("Data", "noaa_radio_flux_celestrak_ext.csv")   # relative to F10_DIR


# ============================================================
# Unscented transform (same alpha/beta/kappa convention as rope.py)
# ============================================================
class UnscentedTransform:
    def __init__(self, n: int, alpha: float = 1.0, beta: float = 2.0, kappa: float = 0.0):
        self.n   = n
        lam      = alpha ** 2 * (n + kappa) - n
        self.c   = n + lam
        self.Wm  = np.full(2 * n + 1, 1.0 / (2.0 * self.c))
        self.Wc  = self.Wm.copy()
        self.Wm[0] = lam / self.c
        self.Wc[0] = lam / self.c + (1.0 - alpha ** 2 + beta)

    def sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """(2n+1, n): row 0 = mean, rows 1..n = mean + L[:, i], rows n+1..2n = mean - L[:, i]."""
        cov = 0.5 * (cov + cov.T)
        try:
            L = np.linalg.cholesky(self.c * cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self.c * cov + 1e-12 * np.eye(self.n))
        return np.vstack([mean[None], mean + L.T, mean - L.T])


# ============================================================
# IC: mean state + 10x10 covariance
# ============================================================
def mean_state_cov(z_rows: np.ndarray, Sigma_joint: np.ndarray, mode: str = "hour_avg"):
    n, K = z_rows.shape
    blocks = np.stack([Sigma_joint[i * K:(i + 1) * K, i * K:(i + 1) * K] for i in range(n)])
    if mode == "hour_avg":
        S = blocks.mean(axis=0)
    elif mode == "mean":
        A = np.kron(np.full((1, n), 1.0 / n), np.eye(K))
        S = A @ Sigma_joint @ A.T
    elif mode == "last":
        S = blocks[-1]
    else:
        raise ValueError(f"ic_cov_mode must be 'hour_avg', 'mean' or 'last', got {mode!r}")
    return z_rows.mean(axis=0), 0.5 * (S + S.T)


# ============================================================
# F10 forecaster (lazy, cached per (dir, history, model_type))
# ============================================================
_FORECASTERS: Dict[tuple, Any] = {}


def get_forecaster(model_type: str = "LSTM_dynamic", f10_dir: str = F10_DIR,
                   history: str = F10_HISTORY, max_workers: int = 1):
    key = (f10_dir, history, model_type)
    if key not in _FORECASTERS:
        if f10_dir not in sys.path:
            sys.path.insert(0, f10_dir)
        from f10_forecaster import F10Forecaster
        fc = F10Forecaster(base_dir=f10_dir, data_file=history)
        # max_workers=1: concurrent Keras loads are not thread-safe (see ROPE+F10_FORECASTER notes)
        fc.load_models(model_types=[model_type], max_workers=max_workers)
        _FORECASTERS[key] = fc
    return _FORECASTERS[key]


def f10_daily_forecast(fc, start, horizon: int, model_type: str):
    """
    Daily forecast from the UT day of `start` through the day after the last forecast hour
    (that extra day is the right-hand anchor for hourly interpolation).
    Returns dates, mean, std (all daily) and the number of days that cover the horizon.
    """
    start = pd.Timestamp(start)
    day0 = start.normalize()
    last = start + pd.Timedelta(hours=horizon - 1)
    n_cover = int((last.normalize() - day0).days) + 1
    res = fc.forecast(start_date=f"{day0:%Y-%m-%d}", horizon_days=n_cover + 1,
                      model_types=[model_type])[model_type]
    return pd.DatetimeIndex(res["dates"]), np.asarray(res["mean"]), np.asarray(res["std"]), n_cover


def forecast_driver_df(driver_df: pd.DataFrame, start, horizon: int, seq_len: int,
                       f10_dates=None, f10_daily=None, offset: float = 0.0) -> pd.DataFrame:
    """
    Driver slice for one run: observed history up to and including `start`; for hours after
    `start`, f10 = forecast (daily values anchored at 00 UT, linear in time -- the same
    convention as the CelesTrak CSV) + offset. f10_dates=None keeps the observed f10 there.
    The slice keeps enough history for the 41-day average.
    """
    start = pd.Timestamp(start)
    lo = start - pd.Timedelta(hours=seq_len - 1 + 41 * 24 + 24)
    hi = start + pd.Timedelta(hours=horizon - 1)
    df = driver_df[(driver_df["datetime"] >= lo) & (driver_df["datetime"] <= hi)].copy()
    fut = (df["datetime"] > start).to_numpy()
    if f10_dates is not None:
        x  = df.loc[fut, "datetime"].to_numpy().astype("datetime64[s]").astype(np.int64)
        xp = pd.DatetimeIndex(f10_dates).to_numpy().astype("datetime64[s]").astype(np.int64)
        df.loc[fut, "f10"] = np.interp(x, xp, np.asarray(f10_daily, dtype=float))
    if offset:
        df.loc[fut, "f10"] = df.loc[fut, "f10"] + offset
    return df.reset_index(drop=True)
