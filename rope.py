"""
rope.py  —  ROPE for TIE-GCM  (f10, f10_41day_avg, ap)
------------------------------------------------------
Two switches, both off by default (= the old ROPE pipeline: IC table + F10 from the CSV):

    rope = ROPE()                                            # old behaviour
    rope = ROPE(use_da=True)                                 # IC from data assimilation
    rope = ROPE(use_f10_forecaster=True)                     # F10 after start from the forecaster
    rope = ROPE(use_da=True, use_f10_forecaster=True)        # both

    out = rope.run("2024-05-10 18:00", horizon=72)                     # constructor settings
    out = rope.run("2024-05-10 18:00", horizon=72, use_da=False)       # per-call override

Warm-up (off by default): ROPE(warmup_hours=48) or run(..., warmup_hours=48) starts the run that
many hours earlier from the IC table on observed drivers and returns only the requested window,
so the IC-table error has faded by the start (as rope-framework's warmup_time_hours). It applies
to the IC-table start only; with DA (or an explicit IC array) the IC is already the state at
the start, so warm-up is skipped.

uncertainty=True gives ROPE's model (ensemble) uncertainty, plus the input uncertainty of
whatever is switched on, through one unscented transform over the active inputs:
    use_da             -> 10 dims (mean of the 3 DA states, DA covariance)
    use_f10_forecaster -> 1 dim   (offset on the forecast F10, sigma from the forecaster ensemble)
    both               -> 11 dims, 2(10+1)+1 = 23 sigma points
density_std is then sqrt(input_var + model_var); the parts come back as input_std, ic_std,
f10_std, model_std.

drivers="celestrak" (default, data/sw_celestrack_1957_ap.csv.gz, 1957..2026) or "tiegcm"
(SW_TIEGCM.csv, 1996..2009; with both switches off this reproduces the original rope.py exactly).
DA ICs exist for 2024-03-01 .. 2024-07-31 (starts on 00/03/06/... UT).

Output keys: window_df, mean_density, density_std (if uncertainty), ic_latents, ic_source,
             f10_source, and when switched on: f10_forecast, f10_sigma, input_std, ic_std,
             f10_std, model_std, model_mean_density
Driver set:  f10, f10_41day_avg, ap, t1, t2, t3, t4
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata

import tensorflow as tf
from tensorflow import keras

from ts_utils.custom_layers import PositionalEncoding
from ae_utils import utils as utils_cae
from ae_utils.attn_models import COAE
from rope_inputs import (
    F10_DIR, F10_HISTORY, UnscentedTransform, mean_state_cov,
    get_forecaster, f10_daily_forecast, forecast_driver_df,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")

# Let TensorFlow take GPU memory as needed instead of all of it, so torch (COAE decoder) and
# other processes can share the GPU.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:          # GPUs already initialized by an earlier import
        pass


# ── Driver columns (fixed for TIE-GCM) ────────────────────────────────────────
TIEGCM_DRIVER_COLS: List[str] = ["f10", "f10_41day_avg", "ap", "t1", "t2", "t3", "t4"]
F10_AVG_ROWS = 984   # 41 days of hourly rows

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER_CSVS = {
    "celestrak": os.path.join(THIS_DIR, "data", "sw_celestrack_1957_ap.csv.gz"),
    "tiegcm":    os.path.join(THIS_DIR, "data", "SW_TIEGCM.csv"),
}


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ROPEConfig:
    latent_dim:        int = 10
    seq_len:           int = 3
    decode_batch_size: int = 4

    @property
    def total_dim(self) -> int:
        return self.latent_dim + len(TIEGCM_DRIVER_COLS)   # 17


# ============================================================
# Small utils
# ============================================================
def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)


def _safe_device(device: str) -> str:
    return device if not (device.startswith("cuda") and not torch.cuda.is_available()) else "cpu"


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Feature normalizer
# ============================================================
class FeatureNormalizer:
    def __init__(self, stats_ts: Dict[str, Any], latent_dim: int = 10):
        self.K     = latent_dim
        self.mu    = _to_numpy(stats_ts["mu"]).astype(np.float32)
        self.sigma = _to_numpy(stats_ts["sigma"]).astype(np.float32)

        if len(self.mu) != len(self.sigma):
            raise ValueError(f"stats_ts mu/sigma length mismatch: {len(self.mu)} vs {len(self.sigma)}")

        self.total_dim  = len(self.mu)
        self.driver_dim = self.total_dim - self.K

        if self.driver_dim != len(TIEGCM_DRIVER_COLS):
            raise ValueError(
                f"stats_ts driver_dim={self.driver_dim} but TIE-GCM expects {len(TIEGCM_DRIVER_COLS)}. "
                f"Check you loaded the correct stats_ts file."
            )

    def norm_full(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sigma).astype(np.float32)

    def norm_driver(self, drv: np.ndarray) -> np.ndarray:
        return ((drv - self.mu[self.K:]) / self.sigma[self.K:]).astype(np.float32)

    def denorm_latents(self, lat_norm: np.ndarray) -> np.ndarray:
        return (lat_norm * self.sigma[: self.K] + self.mu[: self.K]).astype(np.float32)


# ============================================================
# IC-table interpolator
# ============================================================
class ICTableInterpolator:
    def __init__(self, ic_table: pd.DataFrame):
        self.pts  = ic_table[["F10", "Ap"]].values
        self.vals = ic_table.drop(columns=["F10", "Ap"]).values

    def get_latent_coeffs(self, f10: float, ap: float) -> np.ndarray:
        query = np.array([[f10, ap]])
        pred  = griddata(self.pts, self.vals, query, method="linear")
        if np.isnan(pred).any():
            pred = griddata(self.pts, self.vals, query, method="nearest")
        return pred.flatten().astype(np.float32)


# ============================================================
# Driver window builder
# ============================================================
class DriverWindowBuilder:
    def build(
        self,
        driver_df:      pd.DataFrame,
        start_datetime,
        horizon:        int,
        seq_len:        int,
    ) -> pd.DataFrame:
        start_dt   = pd.Timestamp(start_datetime)
        hist_start = pd.Timestamp(start_dt - timedelta(hours=(seq_len - 1)))
        end_dt     = pd.Timestamp(start_dt + timedelta(hours=(horizon - 1)))
        timeline   = pd.date_range(hist_start, end_dt, freq="h")

        df = (
            driver_df.copy()
            .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .set_index("datetime")
            .reindex(timeline)
            .reset_index()
            .rename(columns={"index": "datetime"})
        )

        hour       = df["datetime"].dt.hour
        doy        = df["datetime"].dt.dayofyear.astype(float)
        df["t1"]   = np.sin(2 * np.pi * hour / 24)
        df["t2"]   = np.cos(2 * np.pi * hour / 24)
        df["t3"]   = np.sin(2 * np.pi * doy / 365.25)
        df["t4"]   = np.cos(2 * np.pi * doy / 365.25)

        # f10_41day_avg at dt = mean of the last 984 driver rows at or before dt
        # (same as drv.loc[:dt, "f10"].tail(984).mean(), computed on a slice for speed)
        drv = (
            driver_df.assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .sort_values("datetime")
        )
        t   = drv["datetime"].to_numpy()
        lo  = max(int(np.searchsorted(t, hist_start.to_datetime64(), side="left")) - F10_AVG_ROWS, 0)
        hi  = int(np.searchsorted(t, end_dt.to_datetime64(), side="right"))
        f10 = drv["f10"].iloc[lo:hi].reset_index(drop=True)
        roll = pd.Series(
            f10.rolling(F10_AVG_ROWS, min_periods=1).mean().to_numpy(),
            index=pd.DatetimeIndex(t[lo:hi]),
        )
        df["f10_41day_avg"] = roll.reindex(timeline, method="ffill").to_numpy()

        missing = df[["f10", "ap"]].isna().any(axis=1)
        if missing.any():
            bad = df.loc[missing, ["datetime", "f10", "ap"]].head(10)
            raise ValueError(f"Driver CSV has gaps near {start_datetime}.\n{bad}")

        return df


# ============================================================
# Sequence builder
# ============================================================
class SequenceBuilder:
    def __init__(
        self,
        normalizer: FeatureNormalizer,
        ic_interp:  ICTableInterpolator,
        latent_dim: int = 10,
        seq_len:    int = 3,
    ):
        self.norm = normalizer
        self.ic   = ic_interp
        self.K    = latent_dim
        self.S    = seq_len
        self.D    = normalizer.total_dim

    def build_X_init_norm(self, window_df: pd.DataFrame, latents=None) -> np.ndarray:
        """latents: optional (S, K) IC in ROPE latent units; None -> IC-table lookup."""
        coeff_rows, drv_rows = [], []
        for i in range(self.S):
            row = window_df.iloc[i]
            coeff_rows.append(np.asarray(latents[i], dtype=np.float32) if latents is not None
                              else self.ic.get_latent_coeffs(row["f10"], row["ap"]))
            drv_rows.append(row[TIEGCM_DRIVER_COLS].to_numpy(dtype=np.float32))
        X = np.hstack([np.vstack(coeff_rows), np.vstack(drv_rows)])
        return self.norm.norm_full(X)

    def build_x_chunk(
        self, X_init_norm: np.ndarray, forecast_df: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        x_chunk    = np.zeros((horizon, self.S, self.D), dtype=np.float32)
        x_chunk[0] = X_init_norm
        raw_drv    = forecast_df[TIEGCM_DRIVER_COLS].to_numpy(dtype=np.float32)
        for t in range(1, horizon):
            drv_norm     = self.norm.norm_driver(raw_drv[t])
            row          = np.zeros(self.D, dtype=np.float32)
            row[self.K:] = drv_norm
            x_chunk[t]   = np.vstack([x_chunk[t - 1][1:], row])
        return x_chunk


# ============================================================
# Keras helpers
# ============================================================
def _load_keras_ensemble(
    model_dir:      str,
    n_models:       int  = 5,
    custom_objects: Optional[dict] = None,
) -> List[keras.Model]:
    return [
        keras.models.load_model(
            os.path.join(model_dir, f"best_model_{i}.keras"),
            compile=False,
            custom_objects=custom_objects,
        )
        for i in range(1, n_models + 1)
    ]


def _make_infer_fn(model: keras.Model):
    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)
    return infer


# ============================================================
# Dynamic rollout
# ============================================================
class DynamicRollout:
    def __init__(self, latent_dim: int = 10):
        self.K = latent_dim

    def run(self, infer_fn, x_chunk_np: np.ndarray, horizon: int) -> np.ndarray:
        return self.run_batch(infer_fn, x_chunk_np[None], horizon)[0]

    def run_batch(self, infer_fn, x_chunks: np.ndarray, horizon: int) -> np.ndarray:
        """x_chunks: (B, H, S, D) -> (B, H-1, K); B independent rollouts in one pass."""
        inp   = x_chunks[:, 0].copy()
        preds = np.zeros((x_chunks.shape[0], horizon - 1, self.K), dtype=np.float32)
        for t in range(1, horizon):
            p               = infer_fn(tf.constant(inp)).numpy()
            preds[:, t - 1] = p.astype(np.float32)
            if (t + 1) < horizon:
                # the new last row pairs the latent just predicted (hour t0+t) with the drivers of
                # the same hour, as in training (and in rope-framework). x_chunks[:, t] ends with the
                # drivers of hour t0+t; the original rope.py used t+1 here (drivers one hour ahead).
                inp[:, :-1]          = inp[:, 1:]
                inp[:, -1, : self.K] = p
                inp[:, -1, self.K:]  = x_chunks[:, t, -1, self.K:]
        return preds


# ============================================================
# Meta fusion
# ============================================================
class MetaFusion:
    def __init__(self, coeff_level: bool = True):
        self.coeff_level = coeff_level

    def fuse(
        self,
        meta_infer_fn,
        x_chunk_np: np.ndarray,
        all_preds:  np.ndarray,
    ) -> np.ndarray:
        W         = meta_infer_fn(tf.constant(x_chunk_np)).numpy()
        preds_TMK = all_preds.transpose(1, 0, 2)
        if self.coeff_level:
            return np.sum(W * preds_TMK, axis=1).astype(np.float32)
        return np.sum(W[:, :, None] * preds_TMK, axis=1).astype(np.float32)


# ============================================================
# Latent decoder
# ============================================================
class LatentDecoder:
    def __init__(self, cae: COAE, stats_cae: Any, device: str, batch_size: int = 4):
        self.cae        = cae
        self.stats_cae  = stats_cae
        self.device     = device
        self.batch_size = batch_size

    @torch.no_grad()
    def decode(self, latent_series: np.ndarray) -> np.ndarray:
        loader = DataLoader(
            torch.tensor(latent_series, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False,
        )
        dens = [
            np.power(
                10,
                utils_cae.denormalize(
                    self.cae.decoder(batch.to(self.device)).detach().cpu().numpy(),
                    self.stats_cae,
                ),
            )
            for batch in loader
        ]
        return np.concatenate(dens, axis=0).squeeze(1)


# ============================================================
# Unscented Transform
# ============================================================
def _unscented_transform_density(
    base_latents_norm: np.ndarray,
    mu_latents_full:   np.ndarray,
    init_lat_phys:     np.ndarray,
    normalizer:        FeatureNormalizer,
    decoder:           LatentDecoder,
) -> tuple[np.ndarray, np.ndarray]:
    M, _, K = base_latents_norm.shape
    H       = mu_latents_full.shape[0]

    base_phys    = normalizer.denorm_latents(
        base_latents_norm.reshape(-1, K)
    ).reshape(M, H - 1, K).astype(np.float32)
    init_rep     = np.repeat(init_lat_phys[None, None, :], M, axis=0)
    lat_full_all = np.concatenate([init_rep, base_phys], axis=1)

    alpha, beta, kappa = 1.0, 2.0, 0.0
    lam = alpha ** 2 * (K + kappa) - K
    c   = K + lam
    Wm    = np.full(2 * K + 1, 1.0 / (2.0 * c), dtype=np.float32)
    Wc    = np.full(2 * K + 1, 1.0 / (2.0 * c), dtype=np.float32)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha ** 2 + beta)

    sigma_lat          = np.zeros((H, 2 * K + 1, K), dtype=np.float32)
    sigma_lat[:, 0, :] = mu_latents_full

    for t in range(H):
        Xt = lat_full_all[:, t, :].astype(np.float32)
        mt = mu_latents_full[t].astype(np.float32)
        d  = Xt - mt[None, :]
        Pt = (d.T @ d) / float(max(M - 1, 1))
        Pt = 0.5 * (Pt + Pt.T) + 1e-6 * np.eye(K, dtype=np.float32)
        try:
            S = np.linalg.cholesky(c * Pt).astype(np.float32)
        except np.linalg.LinAlgError:
            Pt += 1e-3 * np.eye(K, dtype=np.float32)
            S   = np.linalg.cholesky(c * Pt).astype(np.float32)
        for i in range(K):
            sigma_lat[t, 1 + i,     :] = mt + S[:, i]
            sigma_lat[t, 1 + K + i, :] = mt - S[:, i]

    lat_flat  = sigma_lat.reshape(H * (2 * K + 1), K)
    dens_flat = decoder.decode(lat_flat)
    dens_sig  = dens_flat.reshape(H, 2 * K + 1, *dens_flat.shape[1:]).astype(np.float32)

    ut_mean = np.tensordot(Wm, dens_sig, axes=(0, 1)).astype(np.float32)
    diff    = dens_sig - ut_mean[:, None, ...]
    ut_var  = np.tensordot(Wc, diff * diff, axes=(0, 1)).astype(np.float32)

    return ut_mean, np.sqrt(np.maximum(ut_var, 0.0)).astype(np.float32)


# ============================================================
# ROPE  (IC-table mode, meta fusion only)
# ============================================================
class ROPE:
    """
    ROPE for TIE-GCM, meta-model fusion.

        rope = ROPE()                                        # IC table + F10 from CSV (old ROPE)
        rope = ROPE(use_da=True, use_f10_forecaster=True)    # DA IC + forecast F10
        out  = rope.run("2024-05-10 18:00", horizon=72)
    """

    def __init__(
        self,
        device:            str   = "cuda",
        decode_batch_size: int   = 4,
        use_xla:           bool  = False,

        # ── Stats / COAE ──────────────────────────────────────
        stats_ts_path:    str = os.path.join(THIS_DIR, "data", "stats_ts.pt"),
        stats_cae_path:   str = os.path.join(THIS_DIR, "data", "stats_cae.pt"),
        coae_config_yaml: str = os.path.join(THIS_DIR, "weights", "finetuned_coae", "config.yaml"),
        coae_weights_pth: str = os.path.join(THIS_DIR, "weights", "finetuned_coae", "best_weights_1gpu.pth"),

        # ── Ensemble model dirs ───────────────────────────────
        lstm_dir:        str = os.path.join(THIS_DIR, "Models", "StormTuned", "LSTM MODELS"),
        gru_dir:         str = os.path.join(THIS_DIR, "Models", "StormTuned", "GRU MODELS"),
        transformer_dir: str = os.path.join(THIS_DIR, "Models", "StormTuned", "TRANSFORMER MODELS"),

        # ── Meta model ────────────────────────────────────────
        meta_model_path: str = os.path.join(THIS_DIR, "Meta Models", "meta_model_41df10_ap_stormtuned.keras"),

        # ── Drivers + IC table ────────────────────────────────
        drivers:      str = "celestrak",          # "celestrak" | "tiegcm" (ignored if driver_csv given)
        driver_csv:   Optional[str] = None,
        ic_table_csv: str = os.path.join(THIS_DIR, "data", "IC_Table.csv"),

        # ── Switches (both False = old ROPE) ──────────────────
        use_da:             bool = False,         # IC from data assimilation (rope_da.DAIC)
        use_f10_forecaster: bool = False,         # F10 after start from the F10 forecaster

        # ── DA / F10-forecaster settings ──────────────────────
        da_h5:          Optional[str] = None,     # DA run file (default: da/out/coae_20240301_20240731.h5)
        ic_cov_mode:    str   = "hour_avg",       # 3-hour DA cov -> 10x10: "hour_avg" | "mean" | "last"
        ic_cov_scale:   float = 1.0,              # multiplies DA sigma (filter sigma is ~70x too small)
        f10_model_type: str   = "LSTM_dynamic",   # "LSTM_dynamic" | "MLP_dynamic"
        f10_sigma:      Optional[float] = None,   # fixed F10 sigma (sfu); None = forecaster ensemble
        warmup_hours:   int   = 0,                # spin-up hours before the start (IC-table start only)
    ):
        self.device = _safe_device(device)
        self.cfg    = ROPEConfig(decode_batch_size=decode_batch_size)

        tf.config.optimizer.set_jit(bool(use_xla))

        # ── Stats ─────────────────────────────────────────────
        self.stats_ts = torch.load(stats_ts_path, map_location="cpu")
        try:
            self.stats_cae = torch.load(stats_cae_path, weights_only=True, map_location="cpu")
        except TypeError:
            self.stats_cae = torch.load(stats_cae_path, map_location="cpu")

        self.normalizer = FeatureNormalizer(self.stats_ts, latent_dim=self.cfg.latent_dim)

        # ── IC + driver pipeline (IC-table mode) ──────────────
        driver_csv = driver_csv or DRIVER_CSVS[drivers]
        df = pd.read_csv(driver_csv)
        df["datetime"]  = pd.to_datetime(df["datetime"])
        self._driver_df = df.sort_values("datetime").reset_index(drop=True)
        self.driver_csv = driver_csv
        self.da         = None    # set to rope_da.DAIC(self) to enable ic="da"

        ic_table        = pd.read_csv(ic_table_csv)
        self._ic_interp = ICTableInterpolator(ic_table)

        self._window_builder = DriverWindowBuilder()
        self._seq_builder    = SequenceBuilder(
            normalizer=self.normalizer,
            ic_interp=self._ic_interp,
            latent_dim=self.cfg.latent_dim,
            seq_len=self.cfg.seq_len,
        )

        # ── COAE ──────────────────────────────────────────────
        coae_cfg = _load_yaml(coae_config_yaml)
        self.cae = COAE(config=coae_cfg.get("model"))
        try:
            sd = torch.load(coae_weights_pth, weights_only=True, map_location="cpu")
        except TypeError:
            sd = torch.load(coae_weights_pth, map_location="cpu")
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.cae.load_state_dict(sd)
        self.cae.to(self.device).eval()

        # ── Ensemble models ─────────────────────────────────────
        self._all_models = (
            _load_keras_ensemble(lstm_dir, n_models=5)
            + _load_keras_ensemble(gru_dir, n_models=5)
            + _load_keras_ensemble(
                transformer_dir, n_models=5,
                custom_objects={"PositionalEncoding": PositionalEncoding},
            )
        )
        self._infer_fns = [_make_infer_fn(m) for m in self._all_models]

        # ── Meta model ──────────────────────────────────────────
        self._meta_infer_fn = _make_infer_fn(
            keras.models.load_model(meta_model_path, compile=False)
        )

        # ── Shared pipeline components ────────────────────────
        self._rollout = DynamicRollout(latent_dim=self.cfg.latent_dim)
        self._fuser   = MetaFusion(coeff_level=True)
        self._decoder = LatentDecoder(
            self.cae, self.stats_cae,
            device=self.device,
            batch_size=self.cfg.decode_batch_size,
        )

        self.driver_df = self._driver_df

        # ── Switches ──────────────────────────────────────────
        self.use_da, self.use_f10_forecaster = bool(use_da), bool(use_f10_forecaster)
        self._da_h5 = da_h5
        self.ic_cov_mode, self.ic_cov_scale = ic_cov_mode, float(ic_cov_scale)
        self.f10_model_type, self.f10_sigma = f10_model_type, f10_sigma
        if int(warmup_hours) < 0:
            raise ValueError("warmup_hours must be >= 0")
        self.warmup_hours = int(warmup_hours)
        self._f10_forecaster = None
        if self.use_da:
            self._ensure_da()
        if self.use_f10_forecaster:
            _ = self.f10_forecaster

        print(f"\nROPE ready  [drivers={os.path.basename(driver_csv)}  use_da={self.use_da}  "
              f"use_f10_forecaster={self.use_f10_forecaster}  device={self.device}]\n")

    # ----------------------------------------------------------
    def _ensure_da(self):
        """Load the DA IC provider on first use (also if rope.da was set by hand)."""
        if self.da is None:
            import rope_da
            self.da = rope_da.DAIC(self, h5_path=self._da_h5) if self._da_h5 else rope_da.DAIC(self)
        return self.da

    @property
    def f10_forecaster(self):
        """F10 forecaster, loaded on first use and cached."""
        if self._f10_forecaster is None:
            self._f10_forecaster = get_forecaster(self.f10_model_type, F10_DIR, F10_HISTORY)
        return self._f10_forecaster

    # ----------------------------------------------------------
    def resolve_ic(self, start_datetime, ic: Any = "table") -> Optional[np.ndarray]:
        """(seq_len, K) IC latents in ROPE units, or None for the IC-table lookup."""
        if isinstance(ic, str):
            if ic == "table":
                return None
            if ic == "da":
                return self._ensure_da().initial_latents(start_datetime, n=self.cfg.seq_len)
            raise ValueError(f"ic must be 'table', 'da' or an array, got {ic!r}")
        lat = np.asarray(ic, dtype=np.float32)
        if lat.shape != (self.cfg.seq_len, self.cfg.latent_dim):
            raise ValueError(f"ic array must be ({self.cfg.seq_len}, {self.cfg.latent_dim}), got {lat.shape}")
        return lat

    # ----------------------------------------------------------
    def _prepare(
        self, start_datetime, horizon: int, latents=None, driver_df: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        internal_df = self._window_builder.build(
            self._driver_df if driver_df is None else driver_df, start_datetime,
            horizon=horizon, seq_len=self.cfg.seq_len,
        )
        start_idx   = self.cfg.seq_len - 1
        forecast_df = internal_df.iloc[start_idx: start_idx + horizon].reset_index(drop=True)
        X_init_norm = self._seq_builder.build_X_init_norm(internal_df.iloc[: self.cfg.seq_len], latents)
        x_chunk_np  = self._seq_builder.build_x_chunk(X_init_norm, forecast_df, horizon)
        return x_chunk_np, X_init_norm, forecast_df

    # ----------------------------------------------------------
    @staticmethod
    def _window_df(forecast_df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["datetime", "f10", "f10_41day_avg", "ap"] if c in forecast_df.columns]
        return forecast_df[cols].copy()

    # ----------------------------------------------------------
    def forecast_latents(self, x_chunks: np.ndarray, X_inits: np.ndarray):
        """
        Batched core: x_chunks (B, H, S, D), X_inits (B, S, D) normalized.
        Returns fused physical latents (B, H, K), base normalized latents (M, B, H-1, K),
        and t0 physical latents (B, K).
        """
        B, H = x_chunks.shape[:2]
        K    = self.cfg.latent_dim
        base = np.stack([self._rollout.run_batch(fn, x_chunks, H) for fn in self._infer_fns],
                        axis=0).astype(np.float32)                         # (M, B, H-1, K)
        M    = base.shape[0]
        W    = self._meta_infer_fn(tf.constant(
            x_chunks[:, : H - 1].reshape(B * (H - 1), *x_chunks.shape[2:]))).numpy()
        W    = W.reshape(B, H - 1, *W.shape[1:])                           # (B, H-1, M, K)
        fused_norm = np.einsum("btmk,mbtk->btk", W, base).astype(np.float32)
        init_phys  = self.normalizer.denorm_latents(X_inits[:, -1, :K])    # (B, K)
        fused_full = np.concatenate(
            [init_phys[:, None, :], self.normalizer.denorm_latents(fused_norm)], axis=1)
        return fused_full, base, init_phys

    # ----------------------------------------------------------
    def run(
        self,
        start_datetime,
        horizon:            int  = 120,
        uncertainty:        bool = True,
        use_da:             Optional[bool] = None,
        use_f10_forecaster: Optional[bool] = None,
        ic:                 Any  = None,
        driver_df:          Optional[pd.DataFrame] = None,
        f10_sigma:          Optional[float] = None,
        warmup_hours:       Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a ROPE forecast (meta fusion).

        use_da / use_f10_forecaster : None = constructor setting, True/False = this call only
        uncertainty : model (ensemble) UT; plus input UT over the switched-on inputs
        ic          : advanced, overrides use_da: "table" | "da" | array (seq_len, K) ROPE latents
        driver_df   : driver override for this call (datetime, f10, ap)
        f10_sigma   : F10 sigma (sfu) for this call; default constructor value, else forecaster spread
        warmup_hours: spin-up hours before the start for this call (None = constructor value);
                      IC-table start only, skipped for DA / an explicit IC array

        Returns dict: window_df, mean_density, density_std (if uncertainty), ic_latents,
        ic_source, f10_source, warmup_hours, and (when switched on) f10_forecast, f10_sigma,
        input_std, ic_std, f10_std, model_std, model_mean_density. All time series cover the
        requested window only (start .. start+horizon-1), with or without warm-up.
        """
        H     = int(horizon)
        K, S  = self.cfg.latent_dim, self.cfg.seq_len
        start = pd.Timestamp(start_datetime)
        use_da = self.use_da if use_da is None else bool(use_da)
        use_fc = self.use_f10_forecaster if use_f10_forecaster is None else bool(use_f10_forecaster)
        if ic is None:
            ic = "da" if use_da else "table"
        da_ic = isinstance(ic, str) and ic == "da"
        base_df = self._driver_df if driver_df is None else driver_df

        # ── warm-up: IC-table start only (DA / explicit IC = the state at the start itself)
        W = self.warmup_hours if warmup_hours is None else int(warmup_hours)
        if W < 0:
            raise ValueError("warmup_hours must be >= 0")
        warmup_note = None
        if W and not (isinstance(ic, str) and ic == "table"):
            warmup_note = f"warm-up ({W} h) skipped: the {'DA' if da_ic else 'given'} IC is the state at the start"
            W = 0
        run_start = start - pd.Timedelta(hours=W)          # == start without warm-up
        H_run     = H + W

        # ── F10: CSV, or forecaster mean after the start hour ──
        f10_dates = f10_mean = f10_table = None
        if use_fc:
            f10_dates, f10_mean, f10_std, n_cover = f10_daily_forecast(
                self.f10_forecaster, start, H, self.f10_model_type)
            f10_table = pd.DataFrame({"date": f10_dates, "f10_mean": f10_mean, "f10_std": f10_std})
            if f10_sigma is None:
                f10_sigma = self.f10_sigma if self.f10_sigma is not None else float(np.mean(f10_std[:n_cover]))
            base_df = forecast_driver_df(base_df, start, H, S, f10_dates, f10_mean, first=run_start)

        meta = {
            "ic_source":    ic if isinstance(ic, str) else "array",
            "f10_source":   f"forecaster ({self.f10_model_type})" if use_fc else "csv",
            "warmup_hours": W,
        }
        if warmup_note:
            meta["warmup_note"] = warmup_note
        if use_fc:
            meta.update(f10_forecast=f10_table, f10_sigma=f10_sigma)

        # ── input UT when an uncertain input is switched on ───
        if uncertainty and (da_ic or use_fc):
            if da_ic:
                st = self._ensure_da().initial_state(start, n=S)
                z_rows = st["z"].astype(np.float64)
                _, S_ic = mean_state_cov(z_rows, st["Sigma_joint"], self.ic_cov_mode)
                S_ic = S_ic * self.ic_cov_scale ** 2
            else:
                z_rows = self.resolve_ic(start, ic)
                if z_rows is None:                              # IC table: not treated as uncertain
                    w = self._window_builder.build(base_df, run_start, horizon=1, seq_len=S)
                    z_rows = np.stack([self._ic_interp.get_latent_coeffs(r["f10"], r["ap"])
                                       for _, r in w.iloc[:S].iterrows()])
                S_ic = None
            out = self.run_input_ut(run_start, H_run, z_rows, S_ic, base_df,
                                    f10_sigma if use_fc else None, f10_origin=start)
            out.update(meta)
            return self._trim_warmup(out, W)

        # ── deterministic run (+ model UT): the old ROPE path ─
        latents = self.resolve_ic(start, ic)
        x_chunk_np, X_init_norm, forecast_df = self._prepare(run_start, H_run, latents, base_df)
        self._last_forecast_df = forecast_df

        fused, base, init_phys = self.forecast_latents(x_chunk_np[None], X_init_norm[None])
        fused_full, init_lat_phys = fused[0], init_phys[0]
        base_latents_norm = base[:, 0]

        density = self._decoder.decode(fused_full)

        out: Dict[str, Any] = {
            "window_df":    self._window_df(forecast_df),
            "mean_density": density,
            "ic_latents":   self.normalizer.denorm_latents(X_init_norm[:, :K]),
            **meta,
        }

        if uncertainty:
            ut_mean, ut_std = _unscented_transform_density(
                base_latents_norm=base_latents_norm,
                mu_latents_full=fused_full,
                init_lat_phys=init_lat_phys,
                normalizer=self.normalizer,
                decoder=self._decoder,
            )
            out["mean_density"] = ut_mean
            out["density_std"]  = ut_std

        return self._trim_warmup(out, W)

    # ----------------------------------------------------------
    @staticmethod
    def _trim_warmup(out: Dict[str, Any], W: int) -> Dict[str, Any]:
        """Drop the first W (warm-up) hours from every time series in a run() result."""
        if not W:
            return out
        for k in ("mean_density", "density_std", "central_density", "input_std", "ic_std",
                  "f10_std", "model_std", "total_std", "model_mean_density"):
            if k in out and out[k] is not None:
                out[k] = out[k][W:]
        for k in ("fused_latents", "sigma_densities"):
            if k in out:
                out[k] = out[k][:, W:]
        out["window_df"] = out["window_df"].iloc[W:].reset_index(drop=True)
        return out

    # ----------------------------------------------------------
    def run_input_ut(
        self,
        start_datetime,
        horizon:      int,
        z_rows:       np.ndarray,
        ic_cov:       Optional[np.ndarray],
        driver_df:    pd.DataFrame,
        f10_sigma:    Optional[float],
        model_uncertainty:    bool = True,
        keep_sigma_densities: bool = False,
        f10_origin=None,
    ) -> Dict[str, Any]:
        """
        Unscented transform over the active uncertain inputs, all sigma points in one batched pass.

        z_rows    : (seq_len, K) IC latents (ROPE units); the UT works on their mean, and a sigma
                    point shifts all rows by the same delta (central point = z_rows exactly)
        ic_cov    : (K, K) covariance of that shift, or None (IC not uncertain)
        driver_df : drivers for the central run (F10 already forecast if wanted)
        f10_sigma : sfu; one offset on f10 after the forecast origin, or None (F10 not uncertain)
        f10_origin: forecast origin for the F10 offset (default: start_datetime; later when the run
                    starts early for warm-up)

        n = K*(ic_cov given) + (f10_sigma given); 2n+1 sigma points. Model (ensemble) UT at the
        central point; density_std = sqrt(input_var + model_var). ic_std / f10_std come from
        central differences of each +/- sigma-point pair (they add up to input_std for a linear
        response).
        """
        K, S, H = self.cfg.latent_dim, self.cfg.seq_len, int(horizon)
        start = pd.Timestamp(start_datetime)
        origin = start if f10_origin is None else pd.Timestamp(f10_origin)
        H_from_origin = H - int((origin - start) / pd.Timedelta(hours=1))
        z_rows = np.asarray(z_rows, dtype=np.float64)
        z_bar = z_rows.mean(axis=0)
        n_ic = K if ic_cov is not None else 0
        n_f  = 1 if f10_sigma is not None else 0
        n = n_ic + n_f
        if n == 0:
            raise ValueError("run_input_ut needs ic_cov and/or f10_sigma")

        mean = np.zeros(n)
        cov  = np.zeros((n, n))
        if n_ic:
            mean[:K], cov[:K, :K] = z_bar, ic_cov
        if n_f:
            cov[n_ic, n_ic] = float(f10_sigma) ** 2
        ut  = UnscentedTransform(n)
        pts = ut.sigma_points(mean, cov)
        B   = len(pts)

        x_chunks, X_inits, central_fdf = [], [], None
        for b, p in enumerate(pts):
            lat = (z_rows + (p[:K] - z_bar)).astype(np.float32) if n_ic else z_rows.astype(np.float32)
            drv = (forecast_driver_df(driver_df, origin, H_from_origin, S, offset=float(p[n_ic]), first=start)
                   if n_f else driver_df)
            xc, Xi, fdf = self._prepare(start, H, latents=lat, driver_df=drv)
            x_chunks.append(xc)
            X_inits.append(Xi)
            if b == 0:
                central_fdf = fdf
        fused, base, init_phys = self.forecast_latents(np.stack(x_chunks), np.stack(X_inits))
        dens = np.stack([self._decoder.decode(fused[b]) for b in range(B)])     # (B, H, ...)

        mean_d = np.tensordot(ut.Wm, dens.astype(np.float64), axes=(0, 0))
        input_var = np.zeros_like(mean_d)
        for i in range(B):
            input_var += ut.Wc[i] * (dens[i] - mean_d) ** 2

        def pair_var(dirs):
            v = np.zeros_like(mean_d)
            for j in dirs:
                v += (0.5 * (dens[1 + j].astype(np.float64) - dens[1 + n + j])) ** 2
            return v / ut.c

        sq = lambda v: np.sqrt(np.maximum(v, 0)).astype(np.float32)
        self._last_forecast_df = central_fdf
        out: Dict[str, Any] = {
            "window_df":       self._window_df(central_fdf),
            "mean_density":    mean_d.astype(np.float32),   # input-UT mean
            "central_density": dens[0],                      # deterministic run at the mean inputs
            "input_std":       sq(input_var),
            "ic_std":          sq(pair_var(range(n_ic))) if n_ic else np.zeros_like(dens[0]),
            "f10_std":         sq(pair_var([n_ic])) if n_f else np.zeros_like(dens[0]),
            "ic_latents":      z_rows.astype(np.float32),
            "ic_mean":         z_bar,
            "ic_cov":          ic_cov,
            "sigma_points":    pts,
            "fused_latents":   fused,
        }
        if model_uncertainty:
            m_mean, m_std = _unscented_transform_density(
                base_latents_norm=base[:, 0], mu_latents_full=fused[0], init_lat_phys=init_phys[0],
                normalizer=self.normalizer, decoder=self._decoder,
            )
            out["model_mean_density"] = m_mean
            out["model_std"]   = m_std
            out["total_std"]   = sq(input_var + m_std.astype(np.float64) ** 2)
            out["density_std"] = out["total_std"]
        else:
            out["density_std"] = out["input_std"]
        if keep_sigma_densities:
            out["sigma_densities"] = dens
        return out
