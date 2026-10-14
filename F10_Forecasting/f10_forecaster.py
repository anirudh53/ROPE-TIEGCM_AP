"""
f10_forecaster.py

Clean F10.7 driver-forecasting pipeline, built to eventually feed ROPE's
driver_df. Two things this fixes vs. the original MLE_Prediction_Example.py:

  1. Horizon is a free parameter (any number of days), not hardcoded to 6.
     Dynamic mode is single-day autoregressive already -- extending it to
     N days is just running the existing inner loop N times, no chaining
     of fixed-length blocks needed.
  2. Models are loaded ONCE and cached, not reloaded on every forecast call.

Static mode is kept too (per your "have everything while experimenting"
ask) but it is capped at its trained 6-day output -- it cannot be
stretched to arbitrary horizons without retraining or block-chaining,
which is the behavior you specifically don't want. Use dynamic for
anything beyond 6 days.

Same as MLP — the LSTM ensemble uses the identical 6 lookback windows: 7, 10, 13, 16, 19, and 22 days (range(7, 23, 3)), one full LSTM sub-ensemble trained per lookback value, same as the MLP side.
Usage:
    from f10_forecaster import F10Forecaster

    fc = F10Forecaster(base_dir="F10_Forecasting", data_file="Data/noaa_radio_flux_celestrak_ext.csv")
    fc.load_models(model_types=["MLP_dynamic", "LSTM_dynamic"])

    result = fc.forecast(start_date="2000-01-15", horizon_days=30)
    # result["MLP_dynamic"]["daily"]        -> (n_models, horizon_days) unscaled f10
    # result["MLP_dynamic"]["mean"]         -> (horizon_days,)
    # result["MLP_dynamic"]["std"]          -> (horizon_days,)
    # result["MLP_dynamic"]["hourly_mean"]  -> hourly-interpolated Series
    # result["MLP_dynamic"]["hourly_std"]   -> hourly-interpolated Series
"""
from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import load_model


class _LSTMCompat(LSTM):
    """
    Drop-in LSTM replacement that silently discards the legacy 'time_major'
    kwarg -- these .h5 files were saved with an older Keras/TF that still
    passed it, and current Keras's LSTM.__init__ no longer accepts it.
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


def _cluster_models_by_shape(models: List) -> List[List]:
    """
    Groups models by their exact weight-shape signature. Normally all 10
    'iteration' models at a given (lookback, architecture) slot share one
    shape (only weight values differ). But this isn't guaranteed -- e.g.
    lookback=19/arch=3 in the LSTM dynamic ensemble actually contains two
    distinct architectures under one nominal slot. Batching only works
    within a shape-homogeneous cluster, so this splits models into however
    many clusters actually exist rather than assuming exactly one.
    """
    clusters: Dict[tuple, List] = {}
    for m in models:
        sig = tuple(tuple(w.shape) for w in m.get_weights())
        clusters.setdefault(sig, []).append(m)
    return list(clusters.values())


# ============================================================
# Data utilities (unchanged math from the original script)
# ============================================================
def get_data(data_fname: str) -> pd.DataFrame:
    f10_df = pd.read_csv(data_fname, index_col=0)
    f10_df.index = pd.to_datetime(f10_df.index)
    if f10_df.isnull().values.any():
        warnings.warn("NaN values detected in F10.7 data.", category=Warning)
    if (f10_df == 0).any().any():
        warnings.warn("Zero values detected in F10.7 data.", category=Warning)
    return f10_df


def back_average(f10_df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    f10_df = f10_df.copy()
    f10_df["backward_avg"] = f10_df["f10"].rolling(window=window).mean()[::-1]
    return f10_df.iloc[window:, :]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += ([('var%d(t)' % (j + 1)) for j in range(n_vars)] if i == 0
                  else [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)])
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ============================================================
# Single-day autoregressive rollout (generalized horizon)
# ============================================================
def dynamic_pred(model, x_row: np.ndarray, horizon_days: int) -> np.ndarray:
    """
    x_row: normalized (1, lookback*2) input, most recent lookback window.
    Returns: (horizon_days,) normalized f10 predictions, one per future day.
    """
    yhat = []
    xhat = x_row.reshape(1, -1).copy()
    for _ in range(horizon_days):
        back_ave = np.mean(np.diff(xhat[0, ::-2])) + xhat[0, -1]
        pred = model.predict(xhat, batch_size=1, verbose=0)[0]
        yhat.append(pred[0])
        dyn_output = np.array([pred[0], back_ave])
        xhat = np.roll(xhat, shift=-2, axis=-1)
        xhat[0, -2:] = dyn_output
    return np.array(yhat)


def dynamic_pred_lstm(model, x_row: np.ndarray, horizon_days: int) -> np.ndarray:
    """
    x_row: normalized (lookback*2, 1) input, most recent lookback window.
    Returns: (horizon_days,) normalized f10 predictions, one per future day.
    """
    yhat = []
    xhat = x_row.reshape(1, -1, 1).copy()
    for _ in range(horizon_days):
        back_ave = np.mean(np.diff(xhat[0, ::-2, 0])) + xhat[0, -1, 0]
        pred = model.predict(xhat, batch_size=1, verbose=0)[0]
        yhat.append(float(pred[0]))
        dyn_output = np.array([float(pred[0]), back_ave])
        xhat = np.roll(xhat, shift=-2, axis=1)
        xhat[0, -2:, 0] = dyn_output
    return np.array(yhat)


# ============================================================
# Batched Dense-ensemble forward pass -- avoids per-model .predict()
# calls entirely by stacking weights and doing one vectorized pass.
# Only applies to plain feedforward (Dense-stack) models -- i.e. the
# MLP family. LSTM would need the recurrent-cell math done by hand;
# not implemented yet, flagged as follow-up.
# ============================================================
class BatchedDenseGroup:
    """
    Wraps N Keras Sequential Dense models that share identical layer
    shapes (same architecture, different trained weights -- exactly
    what the 10 'iteration' models within one (lookback, architecture)
    slot are). Stacks their weights so a forward pass computes all N
    outputs in one vectorized NumPy call instead of N separate
    model.predict() calls.
    """

    _SUPPORTED_ACTS = {
        "tanh": np.tanh,
        "linear": lambda x: x,
        "relu": lambda x: np.maximum(x, 0),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    }

    def __init__(self, models: List):
        self.n_models = len(models)
        self.layers = []  # list of (W_stack (n,in,out), b_stack (n,out), act_fn)
        ref_shapes = [tuple(w.shape for w in m.get_weights()) for m in models]
        if len(set(ref_shapes)) != 1:
            raise ValueError("All models in a BatchedDenseGroup must share identical layer shapes.")

        for layer_idx, layer in enumerate(models[0].layers):
            act_name = getattr(layer.activation, "__name__", "linear")
            if act_name not in self._SUPPORTED_ACTS:
                raise NotImplementedError(f"Unsupported activation '{act_name}' for batched forward.")
            Ws, bs = [], []
            for m in models:
                w, b = m.layers[layer_idx].get_weights()
                Ws.append(w)
                bs.append(b)
            self.layers.append((np.stack(Ws, axis=0), np.stack(bs, axis=0), self._SUPPORTED_ACTS[act_name]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (n_models, in_dim) -- can be the same row tiled n_models times,
           or already-diverged per-model rows (used during autoregression).
        Returns: (n_models, out_dim)
        """
        h = x
        for W_stack, b_stack, act_fn in self.layers:
            h = np.einsum("ni,nio->no", h, W_stack) + b_stack
            h = act_fn(h)
        return h


def dynamic_pred_batched(group: BatchedDenseGroup, x_row: np.ndarray, horizon_days: int) -> np.ndarray:
    """
    Vectorized version of dynamic_pred: runs all group.n_models models'
    autoregressive rollouts together. Returns (n_models, horizon_days).
    """
    n = group.n_models
    xhat = np.tile(x_row.reshape(1, -1), (n, 1)).astype(np.float64)
    out = np.zeros((n, horizon_days), dtype=np.float64)
    for t in range(horizon_days):
        back_ave = np.mean(np.diff(xhat[:, ::-2], axis=1), axis=1) + xhat[:, -1]
        pred = group.forward(xhat)[:, 0]
        out[:, t] = pred
        dyn_output = np.stack([pred, back_ave], axis=1)
        xhat = np.roll(xhat, shift=-2, axis=1)
        xhat[:, -2:] = dyn_output
    return out


def static_pred_batched(group: BatchedDenseGroup, x_row: np.ndarray, eff_horizon: int) -> np.ndarray:
    """
    Vectorized version of the static forward pass: one call computes all
    group.n_models models' 6-day outputs at once. Returns (n_models, eff_horizon).
    """
    n = group.n_models
    x_in = np.tile(x_row.reshape(1, -1), (n, 1)).astype(np.float64)
    out = group.forward(x_in)
    return out[:, :eff_horizon]


# ============================================================
# Batched LSTM(+Dense)-ensemble forward pass.
# Same idea as BatchedDenseGroup -- stack weights across the 10
# same-shaped models -- but LSTM has no vectorized matrix-multiply
# equivalent of a single Dense layer, so the gate equations are
# implemented by hand and applied per-timestep, batched across models.
# ============================================================
class BatchedRNNGroup:
    """
    Wraps N Keras models whose layer stack is some sequence of LSTM and
    Dense layers (in that order, LSTM(s) first), sharing identical shapes
    across models. Stacks weights and runs the recurrence by hand so all
    N models advance together, one NumPy pass per timestep instead of N
    separate model.predict() calls.

    Uses Keras's standard LSTM gate order: kernel/recurrent_kernel/bias
    are each split into 4 equal chunks along the last axis, in order
    [input, forget, cell, output] -- this matches Keras's LSTMCell layout,
    so weights loaded via layer.get_weights() slot in directly with no
    reordering needed.
    """

    def __init__(self, models: List):
        self.n_models = len(models)
        ref_shapes = [tuple(tuple(w.shape) for w in m.get_weights()) for m in models]
        if len(set(ref_shapes)) != 1:
            raise ValueError("All models in a BatchedRNNGroup must share identical layer shapes.")

        self.specs = []
        for layer_idx, layer in enumerate(models[0].layers):
            weights_list = [m.layers[layer_idx].get_weights() for m in models]

            if isinstance(layer, LSTM):
                kernel = np.stack([w[0] for w in weights_list], axis=0)     # (n, in_dim, 4*units)
                rkernel = np.stack([w[1] for w in weights_list], axis=0)    # (n, units, 4*units)
                bias = np.stack([w[2] for w in weights_list], axis=0)       # (n, 4*units)
                acts = BatchedDenseGroup._SUPPORTED_ACTS
                act_name = getattr(layer.activation, "__name__", "tanh")
                rec_name = getattr(layer.recurrent_activation, "__name__", "sigmoid")
                if act_name not in acts or rec_name not in acts:
                    raise NotImplementedError(f"Unsupported LSTM activations '{act_name}'/'{rec_name}'.")
                self.specs.append({
                    "type": "lstm",
                    "kernel": kernel,
                    "rkernel": rkernel,
                    "bias": bias,
                    "units": rkernel.shape[1],
                    "return_sequences": bool(layer.return_sequences),
                    # per-layer activations: these models use tanh, sigmoid AND relu cell
                    # activations, so hard-coding tanh gave wrong outputs for most of them
                    "act_fn": acts[act_name],
                    "rec_act_fn": acts[rec_name],
                })
            elif isinstance(layer, Dense):
                act_name = getattr(layer.activation, "__name__", "linear")
                if act_name not in BatchedDenseGroup._SUPPORTED_ACTS:
                    raise NotImplementedError(f"Unsupported activation '{act_name}' for batched RNN forward.")
                W = np.stack([w[0] for w in weights_list], axis=0)
                b = np.stack([w[1] for w in weights_list], axis=0)
                self.specs.append({"type": "dense", "W": W, "b": b,
                                    "act_fn": BatchedDenseGroup._SUPPORTED_ACTS[act_name]})
            else:
                raise NotImplementedError(f"Unsupported layer type {type(layer).__name__} in BatchedRNNGroup.")

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _apply_lstm(self, x_seq: np.ndarray, spec: dict) -> np.ndarray:
        n, T, _ = x_seq.shape
        units = spec["units"]
        h = np.zeros((n, units), dtype=np.float64)
        c = np.zeros((n, units), dtype=np.float64)
        outputs = [] if spec["return_sequences"] else None

        for t in range(T):
            xt = x_seq[:, t, :]
            z = (np.einsum("ni,nio->no", xt, spec["kernel"])
                 + np.einsum("nh,nho->no", h, spec["rkernel"])
                 + spec["bias"])
            zi, zf, zc, zo = np.split(z, 4, axis=-1)
            act, rec_act = spec["act_fn"], spec["rec_act_fn"]
            i = rec_act(zi)
            f = rec_act(zf)
            c_tilde = act(zc)
            o = rec_act(zo)
            c = f * c + i * c_tilde
            h = o * act(c)
            if spec["return_sequences"]:
                outputs.append(h.copy())

        return np.stack(outputs, axis=1) if spec["return_sequences"] else h

    def _apply_dense(self, h: np.ndarray, spec: dict) -> np.ndarray:
        if h.ndim == 3:  # applied per-timestep if the sequence dim survived (stacked LSTMs)
            out = np.einsum("nti,nio->nto", h, spec["W"]) + spec["b"][:, None, :]
        else:
            out = np.einsum("ni,nio->no", h, spec["W"]) + spec["b"]
        return spec["act_fn"](out)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq: (n_models, T, input_dim)
        Returns: (n_models, out_dim)
        """
        h = x_seq
        for spec in self.specs:
            h = self._apply_lstm(h, spec) if spec["type"] == "lstm" else self._apply_dense(h, spec)
        return h


def dynamic_pred_lstm_batched(group: BatchedRNNGroup, x_row: np.ndarray, horizon_days: int) -> np.ndarray:
    """
    Vectorized version of dynamic_pred_lstm: runs all group.n_models
    models' autoregressive rollouts together. Returns (n_models, horizon_days).
    """
    n = group.n_models
    T = x_row.shape[0]
    xhat = np.tile(x_row.reshape(1, T, 1), (n, 1, 1)).astype(np.float64)
    out = np.zeros((n, horizon_days), dtype=np.float64)
    for t in range(horizon_days):
        back_ave = np.mean(np.diff(xhat[:, ::-2, 0], axis=1), axis=1) + xhat[:, -1, 0]
        pred = group.forward(xhat)[:, 0]
        out[:, t] = pred
        dyn_output = np.stack([pred, back_ave], axis=1)
        xhat = np.roll(xhat, shift=-2, axis=1)
        xhat[:, -2:, 0] = dyn_output
    return out


def static_pred_lstm_batched(group: BatchedRNNGroup, x_row: np.ndarray, eff_horizon: int) -> np.ndarray:
    """
    Vectorized static LSTM forward pass. Returns (n_models, eff_horizon).
    """
    n = group.n_models
    T = x_row.shape[0]
    x_in = np.tile(x_row.reshape(1, T, 1), (n, 1, 1)).astype(np.float64)
    out = group.forward(x_in)
    return out[:, :eff_horizon]


# ============================================================
# Hourly interpolation -- the merge-friendly output
# ============================================================
class HourlyInterpolator:
    """
    Converts a daily forecast (dates + values, or an ensemble of them) into
    an hourly-indexed series, ready to drop into a ROPE-style driver_df.
    """

    @staticmethod
    def to_hourly(dates: pd.DatetimeIndex, daily_values: np.ndarray,
                  method: str = "linear") -> pd.Series:
        """
        dates: (n_days,) daily timestamps (assumed to represent noon-of-day
               values, since F10.7 is reported once per day)
        daily_values: (n_days,) array
        method: 'linear' (smooth ramp between days) or 'ffill' (step function,
                constant within each day -- matches how F10.7 is often held
                constant per UT day in driver files)
        Returns hourly pd.Series covering [dates[0], dates[-1] + 23:00].
        """
        daily_idx = pd.to_datetime(dates) + pd.Timedelta(hours=12)
        s = pd.Series(np.asarray(daily_values), index=daily_idx)

        hourly_idx = pd.date_range(
            pd.to_datetime(dates[0]),
            pd.to_datetime(dates[-1]) + pd.Timedelta(hours=23),
            freq="h",
        )
        s = s.reindex(s.index.union(hourly_idx)).sort_index()

        if method == "linear":
            s = s.interpolate(method="time")
        elif method == "ffill":
            s = s.ffill()
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        s = s.reindex(hourly_idx)
        s = s.bfill().ffill()  # cover edges before first/after last daily anchor
        return s

    @staticmethod
    def ensemble_to_hourly(dates: pd.DatetimeIndex, ensemble: np.ndarray,
                            method: str = "linear") -> np.ndarray:
        """
        ensemble: (n_models, n_days) -> returns (n_models, n_hours)
        """
        rows = [HourlyInterpolator.to_hourly(dates, ensemble[m], method).values
                for m in range(ensemble.shape[0])]
        return np.vstack(rows)


# ============================================================
# Model cache -- load once, reuse across forecast() calls
# ============================================================
class _ModelCache:
    def __init__(self, base_dir: str, model_save_location: str):
        self.base_dir = base_dir
        self.model_save_location = model_save_location
        self._cache: Dict[str, object] = {}
        self._lock = threading.Lock()

    def _path(self, model_type: str, arch: int, lookback: int, iteration: int, keras_type: str) -> str:
        subfolder = "Dynamic Lookback Ensemble/" if "dynamic" in model_type else "Lookback Ensemble/"
        if keras_type == "MLP":
            fname = f"model{arch}_LB{lookback}copy_{iteration}_MLP.h5"
        else:
            fname = f"model{arch}_{lookback}copy_{iteration}_LSTM.h5"
        return os.path.join(self.base_dir, self.model_save_location, subfolder, fname)

    def get(self, model_type: str, arch: int, lookback: int, iteration: int, keras_type: str):
        key = (model_type, arch, lookback, iteration, keras_type)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Load outside the lock (I/O + deserialization can run concurrently);
        # only guard the dict read/write itself.
        custom_objects = {"LSTM": _LSTMCompat} if keras_type == "LSTM" else None
        model = load_model(
            self._path(model_type, arch, lookback, iteration, keras_type),
            compile=False,
            custom_objects=custom_objects,
        )
        with self._lock:
            self._cache.setdefault(key, model)
        return self._cache[key]

    def clear(self):
        self._cache.clear()


# ============================================================
# Main forecaster
# ============================================================
class F10Forecaster:
    VALID_TYPES = ["MLP_dynamic", "LSTM_dynamic", "MLP_static", "LSTM_static"]

    def __init__(
        self,
        base_dir: str = "",
        data_file: str = "Data/noaa_radio_flux_fullset.csv",
        scaling_mean_path: str = "Data/Scaling_mean.npy",
        scaling_std_path: str = "Data/Scaling_std.npy",
        model_save_location: str = "Trained Models",
        max_lookback: int = 22,
        features: int = 2,
    ):
        self.base_dir = base_dir
        self.mean = np.load(os.path.join(base_dir, scaling_mean_path))
        self.std = np.load(os.path.join(base_dir, scaling_std_path))
        self.max_lookback = max_lookback
        self.features = features

        f10 = get_data(os.path.join(base_dir, data_file))
        self.f10_hist = back_average(f10)  # has 'f10' and 'backward_avg' columns

        self._cache = _ModelCache(base_dir, model_save_location)
        self._loaded_types: List[str] = []
        self._batched_groups: Dict[tuple, BatchedDenseGroup] = {}
        self._batched_rnn_groups: Dict[tuple, BatchedRNNGroup] = {}

    def _get_batched_group(self, mt: str, lookback: int, arch: int, keras_type: str) -> List[BatchedDenseGroup]:
        key = (mt, lookback, arch)
        if key not in self._batched_groups:
            models = [self._cache.get(mt, arch, lookback, it, keras_type) for it in range(10)]
            self._batched_groups[key] = [BatchedDenseGroup(c) for c in _cluster_models_by_shape(models)]
        return self._batched_groups[key]

    def _get_batched_rnn_group(self, mt: str, lookback: int, arch: int, keras_type: str) -> List[BatchedRNNGroup]:
        key = (mt, lookback, arch)
        if key not in self._batched_rnn_groups:
            models = [self._cache.get(mt, arch, lookback, it, keras_type) for it in range(10)]
            self._batched_rnn_groups[key] = [BatchedRNNGroup(c) for c in _cluster_models_by_shape(models)]
        return self._batched_rnn_groups[key]

    # -------------------------------------------------- model loading ----
    def load_models(self, model_types: Optional[List[str]] = None, max_workers: int = 8):
        """
        Loads and caches every (architecture x lookback x iteration) model
        for each requested type, using a thread pool since this step is
        I/O-bound (disk reads + per-file deserialization) -- this is the
        step that actually benefits from real concurrency, unlike the
        vectorized forward pass. Static/dynamic each = 6 lookbacks x 3
        architectures x 10 iterations = 180 models.
        """
        model_types = model_types or ["MLP_dynamic", "LSTM_dynamic"]
        for mt in model_types:
            if mt not in self.VALID_TYPES:
                raise ValueError(f"Unknown model_type '{mt}'. Choose from {self.VALID_TYPES}")

        jobs = []
        for mt in model_types:
            keras_type = "MLP" if mt.startswith("MLP") else "LSTM"
            for lookback in range(7, 23, 3):
                for arch in range(1, 4):
                    for it in range(0, 10):
                        jobs.append((mt, arch, lookback, it, keras_type))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(lambda j: self._cache.get(*j), jobs))

        for mt in model_types:
            if mt not in self._loaded_types:
                self._loaded_types.append(mt)
        print(f"Loaded model types: {self._loaded_types} "
              f"({len(self._loaded_types) * 180} models cached, {max_workers} loader threads)")

    # -------------------------------------------------- input building ---
    def _build_lookback_window(self, start_date: str) -> np.ndarray:
        """
        Returns the normalized (max_lookback*features,) input row using the
        max_lookback days of history immediately preceding start_date.
        """
        start_dt = pd.to_datetime(start_date)
        window = self.f10_hist.loc[:start_dt - pd.Timedelta(days=1)].tail(self.max_lookback)
        if len(window) < self.max_lookback:
            raise ValueError(
                f"Not enough history before {start_date}: need {self.max_lookback} days, "
                f"found {len(window)}. Earliest usable start_date is "
                f"{self.f10_hist.index[self.max_lookback]}."
            )
        flat = window[["f10", "backward_avg"]].to_numpy().flatten()  # (lookback*2,)
        return (flat - self.mean) / self.std

    # -------------------------------------------------------- forecast --
    def forecast(self, start_date: str, horizon_days: int,
                 model_types: Optional[List[str]] = None,
                 hourly_method: str = "linear") -> Dict[str, Dict[str, np.ndarray]]:
        model_types = model_types or self._loaded_types
        missing = [mt for mt in model_types if mt not in self._loaded_types]
        if missing:
            raise RuntimeError(f"Call load_models({missing}) before forecasting with them.")

        x_row_flat = self._build_lookback_window(start_date)
        forecast_dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")

        results = {}
        for mt in model_types:
            if mt in ("MLP_static", "LSTM_static") and horizon_days > 6:
                warnings.warn(
                    f"{mt} is a fixed 6-day static model; horizon_days={horizon_days} "
                    f"will be truncated to 6. Use the *_dynamic variant for longer horizons."
                )

            keras_type = "MLP" if mt.startswith("MLP") else "LSTM"
            is_dynamic = "dynamic" in mt
            eff_horizon = horizon_days if is_dynamic else min(horizon_days, 6)

            preds = []
            for lookback in range(7, 23, 3):
                x_lb = x_row_flat[-lookback * self.features:]
                for arch in range(1, 4):
                    if keras_type == "MLP":
                        groups = self._get_batched_group(mt, lookback, arch, keras_type)
                        for group in groups:
                            if is_dynamic:
                                preds.append(dynamic_pred_batched(group, x_lb, eff_horizon))
                            else:
                                preds.append(static_pred_batched(group, x_lb, eff_horizon))
                    else:
                        groups = self._get_batched_rnn_group(mt, lookback, arch, keras_type)
                        for group in groups:
                            if is_dynamic:
                                preds.append(dynamic_pred_lstm_batched(group, x_lb, eff_horizon))
                            else:
                                preds.append(static_pred_lstm_batched(group, x_lb, eff_horizon))

            ensemble = np.vstack(preds)  # (180, eff_horizon) normalized
            ensemble_unscaled = ensemble * self.std + self.mean

            dates_used = forecast_dates[:eff_horizon]
            mean = ensemble_unscaled.mean(axis=0)
            std = ensemble_unscaled.std(axis=0)

            results[mt] = {
                "dates": dates_used,
                "daily": ensemble_unscaled,
                "mean": mean,
                "std": std,
                "hourly_mean": HourlyInterpolator.to_hourly(dates_used, mean, hourly_method),
                "hourly_std": HourlyInterpolator.to_hourly(dates_used, std, hourly_method),
            }

        return results


if __name__ == "__main__":
    BASE_DIR = ""  # <- set this

    fc = F10Forecaster(base_dir=BASE_DIR)
    fc.load_models(model_types=["MLP_dynamic", "LSTM_dynamic"])

    out = fc.forecast(start_date="2000-01-15", horizon_days=30)

    for mt, res in out.items():
        print(mt, "daily mean (first 5 days):", res["mean"][:5])
        print(mt, "hourly series length:", len(res["hourly_mean"]))