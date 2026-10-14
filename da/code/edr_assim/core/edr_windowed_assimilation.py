"""
Windowed information-filter Kalman assimilation of EDR density observations.

The information filter (dual KF) is used because N_obs per window can be
O(10^4–10^5) while the ROM dimension k is only 10. This reduces the update
from O(N^3) to O(N*k^2).

State: z (k,) — ROM coefficients in log10-density space.
Observation model: the EDR retrieval (Picone 2005 Eq. 4.6, see
TLE_density_class.EDREstimatorMidpoint) reports one density per arc that is
physically the v_rel³-weighted time-average of the true density over the arc:
ρ_eff = ∫ρ v_rel³ dt / ∫v_rel³ dt. The consistent prediction is the matching
weighted linear-space arc average
    h(z) = log10( Σ_i wv_i · 10^{mu0_i + H_i z} ),   Σ_i wv_i = 1
over the arc's geolocated sample points (pts_lat/pts_lst/pts_alt). The weights
wv_i track the ∫v_rel³ kernel; absent stored per-point v_rel they default to
uniform (1/N), exact for the near-circular orbits the pipeline admits
(eccentricity ≤ 0.02). This is
nonlinear in z, so it is linearized about z_pred (EKF observation step):
    h(z_pred) = log10( mean_i ρ_i ),   ρ_i = 10^{mu0_i + H_i z_pred}
    H_eff     = Σ_i w_i H_i,           w_i = ρ_i / Σ_j ρ_j
yielding one effective row + predicted value per arc — the O(N·k^2) update is
unchanged. Evaluating only at the arc centroid (the previous behaviour) is
inconsistent: arcs span the full TLE interval (up to 96 h, all lat/LST), and
log10(centroid ρ) ≠ log10(arc-mean ρ).

Information filter update:
  Omega_post = Omega_prior + H^T R^{-1} H
  xi_post    = xi_prior    + H^T R^{-1} dy
  P_post = Omega_post^{-1}
  z_post = z_prior + P_post @ xi_post

Dynamics: when a model_object (full ROM .mat dict) is provided, the predict
step uses sindy_step() from propagator_utilities_V4 (3 hourly calls per
3-hour window) and z0 is initialised from the ic_array lookup table.
The open-loop trajectory is pre-computed via pu.propagate().

Output is written to HDF5 (one group per window) with checkpoint-resume support.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from .observation_error_model import compute_R_diag  # noqa: F401 (used below)

# ── Propagator path (mirrors rom_loader setup) ────────────────────────────────
_PROPAGATOR_DIR = Path(__file__).resolve().parents[3] / "PyDensityEstimation" / "2026-01-09_v2"
if str(_PROPAGATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_PROPAGATOR_DIR))

import propagator_utilities_V4 as pu  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────────

_FILTER_HASH_FIELDS = (
    "Q_diag", "P0_diag",
    "use_npz_sigma", "alt_bins_km", "sigma_by_bin",
    "sigma_floor", "sigma_cap",
    "min_alt_km", "max_alt_km",
    "prefer_calibrated", "log10_clip_lo", "log10_clip_hi",
    "rom_lat_max",
)


def _filter_config_hash(cfg) -> str:
    """Stable SHA-1 hash of filter/obs-error config fields.

    Used by ``init_hdf5`` to detect silent config drift between resume runs:
    if the YAML changes Q_diag, sigma_*, or any other field that affects the
    posterior, the HDF5 is invalidated rather than mixing two configurations
    across windows. Tuple fields are coerced via repr so list/tuple round-trips
    don't spuriously trigger invalidation.
    """
    payload = {f: repr(getattr(cfg, f, None)) for f in _FILTER_HASH_FIELDS}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()


def init_hdf5(hdf5_path: str, cfg) -> None:
    """Create (or open) the HDF5 output file and write top-level attributes.

    If an existing file has a different start_date, end_date, n_components,
    filter_mode, window_hours, or filter-config hash than the current config,
    it is removed and recreated from scratch so stale windows from a previous
    run don't pollute the time-series datasets. The checkpoint JSON is removed
    alongside it; otherwise it would still mark windows "ok" against an empty
    file and crash the resume path.
    """
    p = Path(hdf5_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg_hash = _filter_config_hash(cfg)
    if p.exists():
        try:
            with h5py.File(hdf5_path, "r") as f:
                # window_hours is structural: it sets the time grid, the end-of-window
                # z_post semantics, and the open-loop sampling index. Changing it must
                # invalidate the file or windows of different lengths get mixed.
                stale = (
                    f.attrs.get("start_date")    != cfg.start_date
                    or f.attrs.get("end_date")   != cfg.end_date
                    or f.attrs.get("n_components") != cfg.n_components
                    or f.attrs.get("filter_mode") != cfg.filter_mode
                    or float(f.attrs.get("window_hours", -1.0)) != float(cfg.window_hours)
                    or f.attrs.get("filter_config_hash") != cfg_hash
                )
        except Exception:
            stale = True
        if stale:
            print(f"[init_hdf5] Config changed — removing stale HDF5: {hdf5_path}")
            p.unlink()
            ck = Path(cfg.checkpoint_json)
            if ck.exists():
                print(f"[init_hdf5] Removing stale checkpoint: {ck}")
                ck.unlink()
    with h5py.File(hdf5_path, "a") as f:
        f.attrs.setdefault("start_date",    cfg.start_date)
        f.attrs.setdefault("end_date",      cfg.end_date)
        f.attrs.setdefault("window_hours",  cfg.window_hours)
        f.attrs.setdefault("n_components",  cfg.n_components)
        f.attrs.setdefault("filter_mode",   cfg.filter_mode)
        f.attrs.setdefault("rom_mat_path",  cfg.rom_mat_path)
        # Always overwrite the config hash so a successful re-init records the
        # current YAML state (setdefault would keep a stale value if the file
        # somehow survived a partial reset).
        f.attrs["filter_config_hash"] = cfg_hash
        # Resizable time-series datasets — created individually so new datasets
        # are added to existing files without requiring a full reset.
        ts = f.require_group("time_series")
        k  = cfg.n_components
        if "window_starts" not in ts:
            ts.create_dataset("window_starts", shape=(0,),   maxshape=(None,),
                              dtype=h5py.string_dtype(), chunks=True)
        # NIS_per_obs is dyᵀ S⁻¹ dy / N_obs — the chi-squared statistic
        # *normalized by observation count* (consistency target ≈ 1, not N).
        # Mislabeling it as raw NIS in earlier versions led to consumers
        # mistakenly running chi2.cdf(NIS, df=N), which rejects every
        # window. The dataset is still named "NIS" too for back-compat (it
        # holds the same per-obs value).
        for ds_name, shape, dtype, opts in [
            ("z_post",       (0, k), "f8", {"compression": "gzip", "compression_opts": 4}),
            ("trace_P",      (0,),   "f8", {"compression": "gzip"}),
            ("NIS",          (0,),   "f8", {"compression": "gzip"}),
            ("NIS_per_obs",  (0,),   "f8", {"compression": "gzip"}),
            ("innov_rms",    (0,),   "f8", {"compression": "gzip"}),
            ("n_valid",      (0,),   "i4", {"compression": "gzip"}),
            ("diag_P",       (0, k), "f8", {"compression": "gzip", "compression_opts": 4}),
            ("z_openloop",   (0, k), "f8", {"compression": "gzip", "compression_opts": 4}),
        ]:
            if ds_name not in ts:
                maxshape = tuple(None if d == 0 else d for d in shape)
                ts.create_dataset(ds_name, shape=shape, maxshape=maxshape,
                                  dtype=dtype, chunks=True, **opts)


def append_window_to_hdf5(hdf5_path: str, result: dict) -> None:
    """Append one window result to the HDF5 file."""
    wkey = result["window_start"].strftime("%Y-%m-%dT%H:%M:%S")
    with h5py.File(hdf5_path, "a") as f:
        # Per-window group (full state for checkpoint resumption)
        grp = f.require_group(f"windows/{wkey}")
        for name in ("z_prior", "z_post", "P_post", "diag_P"):
            if name not in grp:
                grp.create_dataset(name, data=result[name],
                                   compression="gzip", compression_opts=4)
        # Augmented (companion-form) posterior — persisted so a resume can
        # reconstruct lag history and cross-block covariances exactly. Absent
        # in static-mode windows; the resume path handles that.
        for name in ("_X_aug_post", "_P_aug_post", "_z_buf_post"):
            if name in result and name not in grp:
                grp.create_dataset(
                    name.lstrip("_"), data=result[name],
                    compression="gzip", compression_opts=4,
                )
        for name in ("n_obs", "n_valid", "NIS", "trace_P", "innov_rms"):
            grp.attrs[name] = result[name]
        # Document the per-obs normalization so consumers don't chi²-test the
        # NIS attr directly. Same value, explicit name.
        grp.attrs["NIS_per_obs"] = result["NIS"]
        grp.attrs["window_start"] = wkey
        grp.attrs["window_end"]   = result["window_end"].strftime("%Y-%m-%dT%H:%M:%S")

        # Append to resizable time-series datasets — but make it idempotent.
        # On a crash between this HDF5 write and the checkpoint save, the window is
        # reprocessed on resume; appending unconditionally would duplicate (and
        # corrupt, since the resumed lag covariance is reseeded) the row. If wkey is
        # already present, overwrite that row in place instead of growing the dataset.
        ts = f["time_series"]
        existing = [s.decode() if isinstance(s, bytes) else s
                    for s in ts["window_starts"][:]]
        try:
            idx = existing.index(wkey)
            grow = False
        except ValueError:
            idx = ts["window_starts"].shape[0]
            grow = True

        if grow:
            ts["window_starts"].resize(idx + 1, axis=0)
        ts["window_starts"][idx] = wkey

        for ds_name, val in [
            ("z_post",       result["z_post"]),
            ("z_openloop",   result["z_openloop"]),
            ("diag_P",       result["diag_P"]),
            ("trace_P",      result["trace_P"]),
            ("NIS",          result["NIS"]),          # back-compat alias
            ("NIS_per_obs",  result["NIS"]),
            ("innov_rms",    result["innov_rms"]),
            ("n_valid",      result["n_valid"]),
        ]:
            if grow:
                ts[ds_name].resize(idx + 1, axis=0)
            ts[ds_name][idx] = val


def load_last_state_from_hdf5(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Recover the most recent z_post and P_post from the HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        if "windows" not in f:
            raise RuntimeError("No windows group found in HDF5 file.")
        # Find the latest window by ISO timestamp (lexicographic sort works)
        keys = sorted(f["windows"].keys())
        if not keys:
            raise RuntimeError("No window data found in HDF5 file.")
        last = f[f"windows/{keys[-1]}"]
        z = np.asarray(last["z_post"])
        P = np.asarray(last["P_post"])
    return z, P


def load_last_augmented_state_from_hdf5(
    hdf5_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Recover the augmented (X_aug, P_aug, z_buf) state if persisted.

    Returns None if the latest window predates the augmented-state persistence
    (e.g., resuming from a checkpoint written by an older version), so callers
    can fall back to the approximate seed-from-current-z path.
    """
    with h5py.File(hdf5_path, "r") as f:
        if "windows" not in f:
            return None
        keys = sorted(f["windows"].keys())
        if not keys:
            return None
        last = f[f"windows/{keys[-1]}"]
        if not all(n in last for n in ("X_aug_post", "P_aug_post", "z_buf_post")):
            return None
        return (
            np.asarray(last["X_aug_post"]),
            np.asarray(last["P_aug_post"]),
            np.asarray(last["z_buf_post"]),
        )


def load_checkpoint(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Information filter update
# ─────────────────────────────────────────────────────────────────────────────

_OMEGA_RIDGE = 1e-12     # ridge added to Omega_post before inversion (PD guard)
# Hard floor on observation variance before 1/R inversion. compute_R_diag
# already clips sigma to sigma_floor**2, but a degenerate upstream row
# (sigma_floor=0 or stale per-arc sigma=0) would otherwise produce w=inf and
# corrupt every downstream window. Set well below any plausible σ²_log10.
_R_FLOOR = 1e-12
# Loose correlation-preserving overflow guard on the posterior covariance
# (normalized z units). Matches the reference ark_filter_windowed_Nsat.py value;
# it only fires to prevent unbounded growth during data gaps, NOT as a tight cap
# at the P0 level (a value near 1.0 would force the posterior to never be more
# uncertain than the initial guess, making the filter overconfident).
_P_MAX_VAR   = 1e6
_P_AUG_PREDICT_MAX_VAR = 1e6  # loose overflow guard on the augmented prior after predict
_DEBUG_PSD_CHECK = False  # set True to assert PSD after each window (expensive)


def _information_filter_update(
    z_prior: np.ndarray,
    P_prior: np.ndarray,
    H_obs: np.ndarray,
    dy: np.ndarray,
    R_diag: np.ndarray,
    p_max_var: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the information-filter KF update.

    Uses the correction-space formulation: the prior mean in correction space
    is zero (since dy already has H @ z_prior subtracted), so the information
    vector prior is xi_prior = 0.

    Complexity O(N*k^2) — avoids forming the N×N innovation covariance matrix.

    Parameters
    ----------
    z_prior : (k,)   prior ROM coefficients
    P_prior : (k,k)  prior covariance (symmetric PD)
    H_obs     : (N,k)  point-wise observation matrix
    dy        : (N,)   innovations = y_obs - mu0 - H @ z_prior
    R_diag    : (N,)   observation variances (sigma^2 in log10 units)
    p_max_var : (k,) or scalar  per-mode variance cap; defaults to _P_MAX_VAR

    Returns
    -------
    z_post  : (k,)
    P_post  : (k,k)  symmetric PSD, variance-capped
    NIS     : float  per-obs NIS = dyᵀ S⁻¹ dy / N_obs (consistency target ≈ 1).
                     NOT raw χ²; do not feed to chi2.cdf with df=N.
    """
    k = len(z_prior)
    eye_k = np.eye(k, dtype=np.float64)
    if p_max_var is None:
        p_max_var_vec = np.full(k, _P_MAX_VAR, dtype=np.float64)
    else:
        p_max_var_vec = np.broadcast_to(np.asarray(p_max_var, dtype=np.float64), (k,)).copy()

    # Information matrix of the prior (Omega = P^{-1})
    try:
        Omega_prior = cho_solve(cho_factor(P_prior), eye_k)
    except np.linalg.LinAlgError:
        Omega_prior = np.linalg.inv(P_prior + 1e-10 * eye_k)

    # Vectorised information update — O(N*k^2). Floor R to a small positive
    # value so a degenerate sigma=0 row from upstream cannot blow up w → inf.
    w = 1.0 / np.maximum(R_diag, _R_FLOOR)                 # (N,)
    Omega_post = Omega_prior + (H_obs.T * w) @ H_obs       # (k,k)
    Omega_post += _OMEGA_RIDGE * eye_k                     # ridge for PD guarantee
    xi_post    = H_obs.T @ (w * dy)                        # (k,)

    # Recover posterior covariance and cache Cholesky for NIS reuse
    try:
        c_and_lower = cho_factor(Omega_post)
        P_post = cho_solve(c_and_lower, eye_k)
        chol_ok = True
    except np.linalg.LinAlgError:
        P_post = np.linalg.inv(Omega_post)
        chol_ok = False

    # Symmetrize to eliminate floating-point asymmetry
    P_post = 0.5 * (P_post + P_post.T)

    # Correlation-preserving variance cap (mirrors reference ark_filter line 478-482).
    # Trigger per-mode: with a per-mode cap vector a single mode can exceed its own
    # cap without the global max being exceeded.
    pd_diag = np.diag(P_post)
    if np.any(pd_diag > p_max_var_vec):
        s = np.sqrt(np.minimum(p_max_var_vec / np.maximum(pd_diag, 1e-300), 1.0))
        P_post = s[:, None] * P_post * s[None, :]
        P_post = 0.5 * (P_post + P_post.T)

    if _DEBUG_PSD_CHECK:
        assert np.linalg.eigvalsh(P_post).min() > 0, "P_post not PSD after update"

    z_post = z_prior + P_post @ xi_post

    # NIS via matrix-inversion lemma — avoids forming N×N matrix S = H P H^T + R
    # S^{-1} dy = R^{-1} dy - R^{-1} H Sigma H^T R^{-1} dy,  Sigma = Omega_post^{-1}
    if chol_ok:
        Rinv_dy   = w * dy                                        # (N,)
        HtRinv_dy = H_obs.T @ Rinv_dy                            # (k,)
        Sigma_v   = cho_solve(c_and_lower, HtRinv_dy)            # (k,)
        S_inv_dy  = Rinv_dy - w * (H_obs @ Sigma_v)              # (N,)
        NIS = float(dy @ S_inv_dy / max(len(dy), 1))
    else:
        NIS = float("nan")

    return z_post, P_post, NIS


# ─────────────────────────────────────────────────────────────────────────────
# SINDy dynamics helpers  (only used when model_object is provided)
# ─────────────────────────────────────────────────────────────────────────────

# Reference MATLAB datenum for 2000-01-01 (verified against propagator_utilities_V4)
_MATLAB_DN_2000 = 730486.0
_PY_DT_2000     = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _to_matlab_datenum(dt: datetime) -> float:
    """Convert an aware UTC datetime to a MATLAB datenum (days since 0-Jan-0000)."""
    return _MATLAB_DN_2000 + (dt - _PY_DT_2000).total_seconds() / 86400.0


_DN_TOL = 1.0 / 1440.0   # one minute, in MATLAB datenum days


def _dn_span(dt_ds: np.ndarray) -> str:
    """First/last date of a MATLAB-datenum array, for error messages."""
    def _fmt(dn):
        return (_PY_DT_2000 + timedelta(days=float(dn) - _MATLAB_DN_2000)).strftime(
            "%Y-%m-%d %H:%M")
    return f"{_fmt(dt_ds.min())} .. {_fmt(dt_ds.max())}"


def _init_z0_from_ic(model_object: dict, start_date: datetime, k: int) -> np.ndarray:
    """Return z0 (k,) from ic_array lookup on (F10, Kp) at start_date.

    Raises if start_date is not in the driver dataset: a nearest-index clamp would
    silently seed the filter from a different epoch's F10/Kp.
    """
    ic_raw = model_object["ic_array"]
    if hasattr(ic_raw, "ic_array"):
        ic_raw = ic_raw.ic_array
    ic_arr    = np.asarray(ic_raw,                          dtype=np.float64)  # (N_ic, 2+K)
    u_ar      = np.asarray(model_object["u_ar_ds"],         dtype=np.float64)  # (74, T)
    dt_ds     = np.asarray(model_object["datetimes_ds"],    dtype=np.float64).ravel()  # (T,) MATLAB dn

    target_dn = _to_matlab_datenum(start_date)
    t_idx     = int(np.argmin(np.abs(dt_ds - target_dn)))
    if abs(dt_ds[t_idx] - target_dn) > _DN_TOL:
        raise ValueError(
            f"[init_z0] start_date {start_date:%Y-%m-%d %H:%M} is not in the ROM's "
            f"driver data ({_dn_span(dt_ds)}). Rebuild the model with drivers "
            "covering this date.")

    f10 = float(u_ar[5, t_idx])   # row 5 = F10
    kp  = float(u_ar[8, t_idx])   # row 8 = Kp (74-row layout)

    # Nearest-neighbour lookup in (F10, Kp) space
    diff = np.abs(ic_arr[:, 0] - f10) + np.abs(ic_arr[:, 1] - kp)
    best = int(np.argmin(diff))
    z0   = ic_arr[best, 2 : 2 + k].copy()

    print(f"[init_z0] F10={f10:.1f}  Kp={kp:.2f}  "
          f"(driver idx={t_idx})  z0[:3]={z0[:3].round(3)}")
    return z0


def _build_x_drivers_interp(model_object: dict,
                             start_dt: datetime,
                             end_dt: datetime,
                             warmup_hours: int = 0) -> np.ndarray:
    """Slice and return the hourly driver matrix for [start_dt - warmup_hours, end_dt].

    The warmup_hours prefix provides real driver history for the AR lag terms in
    sindy_step so that past-lag lookups (idx = interp_idx - i) hit actual data
    instead of being clamped to index 0. Callers must offset interp_idx by
    warmup_hours so column 0 corresponds to start_dt - warmup_hours.

    Returns x_drivers_interp (74, T_hours) where T_hours = warmup_hours + run_hours.
    """
    u_ar  = np.asarray(model_object["u_ar_ds"],       dtype=np.float64)   # (74, T_full)
    dt_ds = np.asarray(model_object["datetimes_ds"],  dtype=np.float64).ravel()   # (T_full,) MATLAB dn

    extended_start = start_dt - timedelta(hours=int(warmup_hours))
    start_dn = _to_matlab_datenum(extended_start)
    end_dn   = _to_matlab_datenum(end_dt)

    mask = (dt_ds >= start_dn) & (dt_ds <= end_dn)
    # Downstream code indexes these columns as consecutive hours, so the span must be
    # fully covered and gap-free. The old nearest-index fallback ran the filter on
    # another epoch's space weather without any visible symptom.
    sel = dt_ds[mask]
    if (sel.size == 0 or sel[0] - start_dn > _DN_TOL
            or end_dn - sel[-1] > 1.0 / 24.0 + _DN_TOL
            or np.any(np.abs(np.diff(sel) - 1.0 / 24.0) > _DN_TOL)):
        raise ValueError(
            f"[x_drivers_interp] ROM driver data ({_dn_span(dt_ds)}) does not cover "
            f"{extended_start:%Y-%m-%d %H:%M} -> {end_dt:%Y-%m-%d %H:%M} hourly without "
            "gaps. Rebuild the model with drivers covering this span.")

    x_interp = pu.interpolate_matrix_rows(u_ar[:, mask], time_conversion_ratio=1.0)
    n_cols = np.atleast_2d(x_interp).shape[1]
    print(f"[x_drivers_interp] warmup={warmup_hours}h  cols={n_cols}"
          f"  span=[{extended_start.strftime('%Y-%m-%dT%H:%M')} → "
          f"{end_dt.strftime('%Y-%m-%dT%H:%M')}]")
    return np.atleast_2d(x_interp)


def _precompute_openloop(model_object: dict,
                         start_dt: datetime,
                         end_dt: datetime,
                         n_components: int,
                         ar_lags: list) -> np.ndarray:
    """Run pu.propagate() to obtain the ROM open-loop forecast.

    Returns z_ol (T_hours, k) — one row per hourly step from start_dt.
    """
    # propagate() expects naive datetimes (its internal data is naive)
    start_naive = start_dt.replace(tzinfo=None)
    end_naive   = end_dt.replace(tzinfo=None)

    z_ol_full, _, _, _ = pu.propagate(
        start_date=start_naive,
        stop_date=end_naive,
        model_object=model_object,
        ar_lags=ar_lags,
        T1=3600,
        T2=3600,
        time_conversion_factor=1.0,
        n_components=n_components,
        f10_idx=5,
        model_name="sindy-c",
        drivers="cls",
        kp_idx=8,
    )
    # z_ol_full: (k, T_hourly_steps) → transpose to (T_hourly_steps, k)
    return np.asarray(z_ol_full, dtype=np.float64).T


def _sindy_predict_step(
    z_buf_norm: np.ndarray,
    dyn_params: dict,
    interp_idx: int,
    x_drivers_interp: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single 1-hour SINDy predict step.

    Replicates pu.sindy_loop_optimized (the validated forecast path used by
    pu.propagate) rather than pu.sindy_step. pu.sindy_step normalizes the AR
    drivers with the wrong X-regressor columns and misaligns the AR-state lags,
    which makes the forecast diverge by orders of magnitude; sindy_loop_optimized
    does neither, so we mirror it here.

    Buffer convention: z_buf_norm holds normalized z with the CURRENT state z_k
    in the newest column (-1) and z_{k-lag} at column -1-lag. The step reads z_k
    and its lags from the buffer, advances one hour, then slides the buffer so
    the new state z_{k+1} becomes the newest column. This keeps the buffer (and
    hence the augmented mean) aligned in time with the returned physical state —
    pushing the *input* instead would leave the augmented mean one step behind
    the propagated mean used for the innovation.

    Returns (z_next_phys, z_buf_norm_updated).
    """
    nm      = dyn_params["normalization_method"]
    ar_lags = dyn_params["ar_lags"]
    n_lags  = len(ar_lags)
    ads     = dyn_params["ar_drivers_start"]
    f10_idx = dyn_params["f10_idx"]
    kp_idx  = dyn_params["kp_idx"]
    A_d     = dyn_params["A_d"]
    B_state = dyn_params["B_ar_state_stack"]      # (n_lags, k, k)
    B_drv   = dyn_params["B_ar_drivers_stack"]    # (n_lags, k, 3)

    # Current normalized state and its AR lags read straight from the buffer.
    z_k = z_buf_norm[:, -1:]                                  # (k,1) current state

    # AR state lags: column -1-ar_lags[j] is the state ar_lags[j] steps in the past.
    z_ar = z_buf_norm[:, [-1 - int(lag) for lag in ar_lags]]  # (k, n_lags)
    ar_state = np.einsum("lij,jl->i", B_state, z_ar).reshape(-1, 1)

    # AR drivers from the dedicated driver rows (3 drivers x n_lags), normalized
    # with the offset slice of X_norm (see _AR_DRIVERS_NORM_OFFSET in rom_loader).
    ii = min(int(interp_idx), x_drivers_interp.shape[1] - 1)
    ar_drv_raw = x_drivers_interp[kp_idx + 1 : kp_idx + 1 + 3 * n_lags, ii].reshape(-1, 1)
    ar_drv_norm = pu.normalize_with_dict(ar_drv_raw, dyn_params["X_ar_drivers_norm"], nm)
    ar_drv_norm = ar_drv_norm.reshape(n_lags, 3).T           # (3, n_lags)
    ar_drv = np.einsum("lik,kl->i", B_drv, ar_drv_norm).reshape(-1, 1)

    # Nonlinear SINDy library terms (evaluated at the physical current state).
    z_k_denorm = pu.denormalize(z_k, dyn_params["Y_norm"], nm)
    sindy_inputs = pu.build_dyn_sindy_inputs_optimized(
        z_k_denorm,
        x_drivers_interp,
        dyn_params["X_norm"],
        f10_idx,
        dyn_params["model_params"],
        nm,
        dyn_params["input_features"],
        ii,
        kp_idx=kp_idx,
    )
    nl = dyn_params["B_nonlinear"] @ sindy_inputs[:ads]

    z_next_norm = (A_d @ z_k + ar_state + ar_drv + nl).ravel()
    z_next_phys = y_std * z_next_norm + y_mean

    # Slide buffer: the new state becomes the newest column, oldest is dropped.
    z_buf_new = np.column_stack([z_buf_norm[:, 1:], z_next_norm])  # (k, max_lag+1)

    return z_next_phys, z_buf_new


# ─────────────────────────────────────────────────────────────────────────────
# Companion-form (augmented) covariance — mirrors ark_filter_windowed_Nsat.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_companion_matrix(dyn_params: dict, k: int) -> np.ndarray:
    """Companion state-transition matrix for the augmented covariance.

    The augmented state is X_aug = [z_k; z_{k-1}; …; z_{k-max_lag}] (one block per
    referenced lag), so n_aug = k * (max_lag + 1). The block rows encode the SINDy
    AR recursion exactly as our mean dynamics (sindy_loop_optimized):

        z_{k+1} = A_d z_k + Σ_j B_ar[j] z_{k - ar_lags[j]} + drivers + nonlinear(z_k)

    Top block row: [A_d, …, B_ar[j] at column ar_lags[j], …]; remaining rows are an
    identity shift register. The nonlinear contribution to the Jacobian is added
    per-step by pu.sindy_process_jacobian. This is the companion-form propagation
    used in ark_filter_windowed_Nsat.py; the block placement follows our corrected
    lag convention so the linearization matches the propagated mean.
    """
    ar_lags = dyn_params["ar_lags"]
    max_lag = dyn_params["max_lag"]
    p       = max_lag + 1
    n_aug   = k * p

    A_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
    A_aug[:k, :k] = dyn_params["A_d"]                       # current-state linear term
    for j, lag in enumerate(ar_lags):
        A_aug[:k, lag * k : (lag + 1) * k] = dyn_params["B_ar_state_lst"][j]
    for i in range(1, p):                                   # identity shift register
        A_aug[i * k : (i + 1) * k, (i - 1) * k : i * k] = np.eye(k)
    return A_aug


def _augmented_information_update(
    X_aug_pred: np.ndarray,
    P_aug: np.ndarray,
    H_norm: np.ndarray,
    dy: np.ndarray,
    R_diag: np.ndarray,
    k: int,
    p_max_var: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Information-form Kalman update on the augmented state, observing only z_k.

    The measurement touches only the top (current-state) block, so the information
    contribution H^T R^{-1} H lands in the top-left k×k block of the augmented
    information matrix; the full posterior covariance (incl. cross-lag terms) is
    recovered by inverting it. Equivalent to the Joseph-form update in
    ark_filter_windowed_Nsat.py but O(N·k^2) via the information form (N can be ~10^4).

    Parameters
    ----------
    X_aug_pred : (n_aug,)   prior augmented mean (normalized z space)
    P_aug      : (n_aug,n_aug) prior augmented covariance (normalized)
    H_norm     : (N,k)      observation matrix in normalized z space (= U·y_std)
    dy         : (N,)       innovations = y_obs - mu0 - H @ z_pred
    R_diag     : (N,)       observation variances (log10 units)

    Returns
    -------
    X_aug_post : (n_aug,)
    P_aug_post : (n_aug,n_aug)  symmetric, variance-capped
    NIS        : float
    """
    n_aug = P_aug.shape[0]
    eye   = np.eye(n_aug, dtype=np.float64)
    w     = 1.0 / np.maximum(R_diag, _R_FLOOR)

    try:
        Omega = cho_solve(cho_factor(P_aug), eye)
    except np.linalg.LinAlgError:
        Omega = np.linalg.inv(P_aug + 1e-10 * eye)

    HtRH = (H_norm.T * w) @ H_norm                          # (k,k)
    Omega[:k, :k] += HtRH
    Omega += _OMEGA_RIDGE * eye

    try:
        P_post = cho_solve(cho_factor(Omega), eye)
    except np.linalg.LinAlgError:
        P_post = np.linalg.inv(Omega)
    P_post = 0.5 * (P_post + P_post.T)

    # Correlation-preserving per-mode variance cap. The cap is replicated across
    # the (max_lag+1) lag blocks so each lag-k history slot uses the same per-mode
    # bound (current state and lag state share the same physical interpretation).
    if p_max_var is None:
        p_max_var_aug = np.full(n_aug, _P_MAX_VAR, dtype=np.float64)
    else:
        p_max_var_vec = np.broadcast_to(np.asarray(p_max_var, dtype=np.float64), (k,))
        p_max_var_aug = np.tile(p_max_var_vec, n_aug // k)
    pd_diag = np.diag(P_post)
    if np.any(pd_diag > p_max_var_aug):
        s = np.sqrt(np.minimum(p_max_var_aug / np.maximum(pd_diag, 1e-300), 1.0))
        P_post = s[:, None] * P_post * s[None, :]
        P_post = 0.5 * (P_post + P_post.T)

    xi = np.zeros(n_aug, dtype=np.float64)
    xi[:k] = H_norm.T @ (w * dy)
    X_aug_post = X_aug_pred + P_post @ xi

    # NIS via Woodbury on the prior top-block covariance C = P_aug[:k,:k].
    # S = R + H_norm C H_norm^T ;  S^{-1} dy = R^{-1}dy - R^{-1}H_norm M^{-1} H_norm^T R^{-1}dy
    C = P_aug[:k, :k]
    try:
        Cinv = cho_solve(cho_factor(C), np.eye(k))
    except np.linalg.LinAlgError:
        Cinv = np.linalg.inv(C + 1e-10 * np.eye(k))
    M         = Cinv + HtRH
    Rinv_dy   = w * dy
    HtRinv_dy = H_norm.T @ Rinv_dy
    S_inv_dy  = Rinv_dy - w * (H_norm @ np.linalg.solve(M, HtRinv_dy))
    NIS = float(dy @ S_inv_dy / max(len(dy), 1))

    return X_aug_post, P_post, NIS


# ─────────────────────────────────────────────────────────────────────────────
# Per-window processing
# ─────────────────────────────────────────────────────────────────────────────

def _is_nonlinear_rom(meas_model) -> bool:
    """True when the measurement operator is not the affine PCA basis.

    ``RopeDataInterpolator`` decodes as log10_rho = mu0(x) + U(x) @ z, so its Jacobian
    does not depend on z and ``get_H`` alone is enough.  ``CAEMeasurementModel`` wraps a
    conv-net decoder, log10_rho = D(z), so both the value and the Jacobian have to be
    evaluated at a specific z via ``h_and_H``.  Detect rather than require the caller to
    declare it, so every existing call site keeps working untouched.
    """
    return not isinstance(meas_model, pu.RopeDataInterpolator)


def _eval_logrho_H_seam(
    lat: np.ndarray, lst: np.ndarray, alt: np.ndarray, meas_model,
    z: np.ndarray, is_nonlinear: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seam-aware point-wise ROM evaluation at z: returns (log10_rho (M,), H (M,k)).

    This is the ONLY place the reduced-order model is queried per arc sample point, and
    the only place that knows whether the decoder is affine or not.  Both branches
    return the same pair -- the density at z and its gradient with respect to z -- so
    everything downstream (the arc average, the EKF linearisation, the information
    update) is identical for the PCA basis and the COAE decoder.

        pca  : log10_rho = mu0 + H @ z,   H = get_H(lla)        (H independent of z)
        coae : (log10_rho, H) = meas_model.h_and_H(z, lla)      (both depend on z)

    LST is wrapped cyclically.  The ROM grid spans [0, lt_grid[-1]] where
    lt_grid[-1] ~= 23.6667 h (one cell short of 24 h).  Points whose wrapped LST falls
    in the seam (lt_grid[-1], 24) are blended between the two edge cells (lt_grid[-1]
    and lt_grid[0]) by the fractional position through the gap, rather than snapped onto
    a single edge cell (which would drop the LST gradient there).

    The two branches blend DIFFERENT quantities, deliberately:

      * affine -- blend (H, mu) and only then form mu + H @ z.  Algebraically the same
        as blending the decoded value, but not the same in floating point, and the PCA
        results this repo has already published were produced in this order.  Keeping it
        makes the PCA path bit-identical across this refactor, which is the regression
        gate for the whole change.
      * nonlinear -- there is no mu to blend, so blend the decoded log10 density and its
        Jacobian directly.  This is the natural generalisation: for the affine case the
        two blends coincide.

    ``lat`` is assumed already clipped to the ROM range.
    """
    if is_nonlinear is None:
        is_nonlinear = _is_nonlinear_rom(meas_model)

    lst_raw = np.mod(lst, 24.0)
    lt_lo = float(meas_model.lt_grid[0])      # 0.0
    lt_hi = float(meas_model.lt_grid[-1])     # e.g. 23.6667
    seam_width = (24.0 - lt_hi) + lt_lo       # length of the wrap gap
    in_seam = lst_raw > lt_hi                 # points in the cyclic gap

    if not np.any(in_seam):
        if is_nonlinear:
            h, H = meas_model.h_and_H(z, np.column_stack([lat, lst_raw, alt]))
            return (np.asarray(h, dtype=np.float64).ravel(),
                    np.asarray(H, dtype=np.float64))
        H, mu = meas_model.get_H(np.column_stack([lat, lst_raw, alt]))
        H  = np.asarray(H,  dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64).ravel()
        return mu + H @ z, H

    # Two-pass query: in-grid points use lst_raw unchanged (alpha=0 -> value = pass a);
    # seam points are blended between lt_hi (alpha=0) and lt_lo (alpha=1), where alpha
    # is the fractional distance through the wrap gap starting at lt_hi.
    lst_eval_a = lst_raw.copy()
    lst_eval_a[in_seam] = lt_hi
    lst_eval_b = lst_raw.copy()
    lst_eval_b[in_seam] = lt_lo

    alpha = np.zeros_like(lst_raw)
    alpha[in_seam] = (lst_raw[in_seam] - lt_hi) / seam_width
    a = alpha[:, None]

    if is_nonlinear:
        # Only the seam points need the second evaluation -- pass b is identical to
        # pass a everywhere else, and alpha is 0 there, so blending them is a no-op.
        # The seam is a 0.33 h gap in 24 h, so this is typically 0-2 points out of ~60.
        # It matters because on CUDA the Jacobian is one backward pass PER ROW (functorch
        # cannot vmap the TorchScript decoder there), so evaluating all M rows twice
        # would double the dominant cost of the whole filter to no purpose.
        h, H = meas_model.h_and_H(z, np.column_stack([lat, lst_eval_a, alt]))
        h = np.asarray(h, dtype=np.float64).ravel().copy()
        H = np.asarray(H, dtype=np.float64).copy()

        sel = np.flatnonzero(in_seam)
        h_b, H_b = meas_model.h_and_H(
            z, np.column_stack([lat[sel], lst_eval_b[sel], alt[sel]]))
        h_b = np.asarray(h_b, dtype=np.float64).ravel()
        H_b = np.asarray(H_b, dtype=np.float64)

        al = alpha[sel]
        h[sel] = (1.0 - al) * h[sel] + al * h_b
        H[sel] = (1.0 - al[:, None]) * H[sel] + al[:, None] * H_b
        return h, H

    H_a, mu_a = meas_model.get_H(np.column_stack([lat, lst_eval_a, alt]))
    H_a  = np.asarray(H_a,  dtype=np.float64)
    mu_a = np.asarray(mu_a, dtype=np.float64).ravel()

    H_b, mu_b = meas_model.get_H(np.column_stack([lat, lst_eval_b, alt]))
    H_b  = np.asarray(H_b,  dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64).ravel()

    H  = (1.0 - a) * H_a + a * H_b
    mu = (1.0 - alpha) * mu_a + alpha * mu_b
    return mu + H @ z, H


def _arc_observation_operator(
    win_df: pd.DataFrame, interpolator, cfg, z_pred: np.ndarray,
    is_nonlinear: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear-space arc-averaged observation operator (operator C, EKF-linearized).

    The EDR retrieval (Picone 2005 Eq. 4.6, TLE_density_class.EDREstimatorMidpoint)
    assumes ρ constant over the arc and reports ρ = c·Δn / ∫v_rel³ dt, which is
    physically the v_rel³-weighted time-average of the true density along the arc.
    The N=n_midpoint_arc_points geolocated samples (pts_lat/pts_lst/pts_alt) are the
    exact nodes the pipeline uses for that ∫v_rel³ trapz integral. The consistent
    model prediction is therefore the matching weighted linear-space average of the
    ROM density over those nodes:

        h(z) = log10( Σ_i wv_i · ρ_i(z) ),   Σ_i wv_i = 1

    Weights wv_i track the ∫v_rel³ kernel. Per-point v_rel is not stored in the NPZ,
    so wv_i defaults to uniform (1/N) — exact for the near-circular orbits the
    pipeline admits (eccentricity ≤ 0.02), where v_rel is ~constant over the arc. If
    a future NPZ stores pts_vrel, swap the uniform weights for v_rel_i³-normalized
    ones (single line below) for an exact match.

    This is nonlinear in z, so it is linearized about ``z_pred``. With per-point
    densities ρ_i = 10^{mu0_i + H_i z_pred} and (here uniform) node weights, the
    EKF effective weights become w_i = wv_i ρ_i / Σ_j wv_j ρ_j:

        h(z_pred) = log10( mean_i ρ_i )          (predicted observation)
        H_eff     = ∂h/∂z = Σ_i w_i H_i          (effective basis row; ln10 cancels)

    so the linear model around z_pred is  y ≈ h(z_pred) + H_eff @ (z − z_pred),
    i.e. the innovation is simply  dy = y_obs − h(z_pred)  and the information
    update uses H_eff exactly as a one-row-per-arc operator (still O(N·k^2)).

    This replaces centroid evaluation, which discarded the arc geometry the
    observation integrates over: arcs span the full TLE interval (up to 96 h,
    all lat/LST), so log10(centroid ρ) ≠ log10(arc-mean ρ).

    Returns (H_eff (N_arcs, k), h_pred (N_arcs,)).
    """
    n_arcs = len(win_df)
    k = len(z_pred)

    # Flatten all arc sample points into one batch, tagged by arc row index.
    pts_lat = win_df["pts_lat"].to_numpy()
    pts_lst = win_df["pts_lst"].to_numpy()
    pts_alt = win_df["pts_alt"].to_numpy()
    n_pts   = np.fromiter((len(a) for a in pts_lat), dtype=np.int64, count=n_arcs)
    arc_id  = np.repeat(np.arange(n_arcs), n_pts)            # (M,) point → arc

    lat = np.clip(np.concatenate(pts_lat).astype(np.float64),
                  -cfg.rom_lat_max, cfg.rom_lat_max)
    lst = np.concatenate(pts_lst).astype(np.float64)
    alt = np.concatenate(pts_alt).astype(np.float64)

    # Per-point density and its gradient at the linearisation point.  For the affine
    # PCA basis this is mu0_i + H_i z with H independent of z; for the COAE decoder both
    # come out of one h_and_H call.  Everything below is written in terms of rho_i and
    # d(log10 rho_i)/dz only, so it is the same maths either way -- the arc-average chain
    # rule d/dz log10(Sigma wv_i rho_i) = Sigma w_i * d(log10 rho_i)/dz holds for ANY
    # per-point operator, linear or not.
    log10rho_pts, H_pts = _eval_logrho_H_seam(
        lat, lst, alt, interpolator, z_pred, is_nonlinear)      # (M,), (M,k)

    # Node weights wv_i for the ∫v_rel³ kernel, normalized to sum to 1 within each
    # arc. Uniform (1/n_arc_pts) here — exact for the near-circular orbits the
    # pipeline admits. To match the retrieval kernel exactly when the NPZ carries
    # per-point v_rel, replace this line with:
    #     wv = np.concatenate(win_df["pts_vrel"]).astype(float) ** 3
    cnt = np.bincount(arc_id, minlength=n_arcs).astype(np.float64)
    wv  = 1.0 / cnt[arc_id]                                  # (M,)  Σ over arc = 1
    wv /= np.bincount(arc_id, weights=wv, minlength=n_arcs)[arc_id]  # renormalize

    # Per-point ROM density at z_pred. log10 ρ stays in [~-15, -7] (clip range),
    # so 10**(·) is finite — no log-sum-exp stabilization needed.
    rho   = np.power(10.0, log10rho_pts)                    # (M,)
    wrho  = wv * rho                                         # (M,)
    sum_wrho = np.bincount(arc_id, weights=wrho, minlength=n_arcs)  # (N_arcs,) = Σ wv ρ
    h_pred   = np.log10(sum_wrho)                           # (N_arcs,)

    # EKF effective row: ∂h/∂z = Σ_i (wv_i ρ_i / Σ_j wv_j ρ_j) H_i (ln10 cancels).
    w = wrho / sum_wrho[arc_id]                             # (M,)  Σ over arc = 1
    H_eff = np.empty((n_arcs, k), dtype=np.float64)
    for c in range(k):
        H_eff[:, c] = np.bincount(arc_id, weights=w * H_pts[:, c], minlength=n_arcs)

    return H_eff, h_pred


def process_window(
    window_start: datetime,
    window_end: datetime,
    win_df: pd.DataFrame,
    z_prior: np.ndarray,
    P_prior: np.ndarray,
    interpolator,
    cfg,
    z_pred: Optional[np.ndarray] = None,
    P_pred: Optional[np.ndarray] = None,
    p_max_var: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    is_nonlinear: Optional[bool] = None,
) -> dict:
    """Run one 3-hour window of the EDR information-filter assimilation.

    The filter state z_prior/z_pred must be in PHYSICAL ROM coefficient space:
        log10_rho = mu0 + U @ z_phys

    z_pred / P_pred : if provided (from sindy dynamics predict step), the
                      internal static predict step is skipped.
    p_max_var       : (k,) per-mode variance cap for the information filter
                      update; should be in the same units as P (physical).

    Returns a dict with all state arrays in physical z space.
    """
    k = len(z_prior)
    t0 = time.perf_counter()

    # ── Predict step ──────────────────────────────────────────────────────────
    if z_pred is None:
        # Static / identity predict. cfg.Q_diag is normalized-space variance
        # (per the contract documented in run_windowed_assimilation); scale to
        # physical units using y_std**2 so P_pred matches the physical-z P.
        q_norm = np.asarray(cfg.Q_diag[:k], dtype=np.float64)
        if y_std is None:
            Q_mat = np.diag(q_norm)        # legacy fallback (no y_std available)
        else:
            Q_mat = np.diag(q_norm * y_std**2)
        P_pred = P_prior + Q_mat
        z_pred = z_prior.copy()

    n_obs = len(win_df)
    result_base = {
        "window_start": window_start,
        "window_end":   window_end,
        "z_prior":      z_pred.copy(),
        "n_obs":        n_obs,
    }

    # ── No observations → coast ──────────────────────────────────────────────
    if n_obs == 0:
        return {
            **result_base,
            "z_post":    z_pred,
            "P_post":    P_pred,
            "diag_P":    P_pred.diagonal(),
            "n_valid":   0,
            "NIS":       float("nan"),
            "trace_P":   float(np.trace(P_pred)),
            "innov_rms": float("nan"),
        }

    # ── Linear-space arc-averaged observation (operator C, linearized about z_pred) ──
    # win_df is already altitude- and finite-filtered by the loader; n_valid == n_obs.
    # H_obs is the EKF effective row Σ w_i H_i; h_pred = log10(mean_i ρ_i(z_pred)).
    H_obs, h_pred = _arc_observation_operator(
        win_df, interpolator, cfg, z_pred, is_nonlinear)                 # (N,k),(N,)

    # Innovation about the linearization point: dy = y_obs - h(z_pred).
    dy     = win_df["log10_rho"].values - h_pred
    n_valid = n_obs

    # ── Observation noise ────────────────────────────────────────────────────
    R_diag = compute_R_diag(
        df=win_df,
        use_npz_sigma=cfg.use_npz_sigma,
        alt_bins_km=cfg.alt_bins_km,
        sigma_by_bin=cfg.sigma_by_bin,
        sigma_floor=cfg.sigma_floor,
        sigma_cap=cfg.sigma_cap,
    )

    # ── Information filter update ────────────────────────────────────────────
    z_post, P_post, NIS = _information_filter_update(
        z_pred, P_pred, H_obs, dy, R_diag, p_max_var=p_max_var
    )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    innov_post = dy - H_obs @ (z_post - z_pred)
    innov_rms  = float(np.sqrt(np.mean(innov_post**2)))
    trace_P    = float(np.trace(P_post))

    elapsed = time.perf_counter() - t0
    print(
        f"  [{window_start.strftime('%Y-%m-%dT%H:%M')}]"
        f"  n_obs={n_obs}  n_valid={n_valid}"
        f"  NIS={NIS:.3f}  trace_P={trace_P:.4f}"
        f"  innov_rms={innov_rms:.3f}"
        f"  ({elapsed:.1f}s)"
    )

    return {
        **result_base,
        "z_post":    z_post,
        "P_post":    P_post,
        "diag_P":    P_post.diagonal(),
        "n_valid":   n_valid,
        "NIS":       NIS,
        "trace_P":   trace_P,
        "innov_rms": innov_rms,
    }


def _process_window_augmented(
    window_start: datetime,
    window_end: datetime,
    win_df: pd.DataFrame,
    z_pred_phys: np.ndarray,
    X_aug_pred: np.ndarray,
    P_aug: np.ndarray,
    interpolator,
    cfg,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    p_max_var_norm: Optional[np.ndarray] = None,
    is_nonlinear: Optional[bool] = None,
) -> dict:
    """One window of the augmented (companion-form) information-filter update.

    The mean is propagated in physical z space (z_pred_phys); the covariance is the
    augmented normalized-space P_aug from the companion predict. The measurement
    update observes only the current-state block. Output state arrays are physical
    (top block); the full augmented posterior is returned under private keys for the
    caller to carry forward.
    """
    k = len(z_pred_phys)
    t0 = time.perf_counter()
    n_obs = len(win_df)

    P_top_phys = (y_std[:, None] * P_aug[:k, :k] * y_std[None, :])
    result_base = {
        "window_start": window_start,
        "window_end":   window_end,
        "z_prior":      z_pred_phys.copy(),
        "n_obs":        n_obs,
    }

    # ── No observations → coast on the prior ─────────────────────────────────
    if n_obs == 0:
        return {
            **result_base,
            "z_post":      z_pred_phys.copy(),
            "P_post":      P_top_phys,
            "diag_P":      np.diag(P_top_phys),
            "n_valid":     0,
            "NIS":         float("nan"),
            "trace_P":     float(np.trace(P_top_phys)),
            "innov_rms":   float("nan"),
            "_P_aug_post": P_aug,
            "_X_aug_post": X_aug_pred.copy(),
        }

    # ── Linear-space arc-averaged observation (operator C, linearized about z_pred) ──
    # H_obs is the EKF effective row Σ w_i H_i; h_pred = log10(mean_i ρ_i(z_pred_phys)).
    H_obs, h_pred = _arc_observation_operator(
        win_df, interpolator, cfg, z_pred_phys, is_nonlinear)   # (N,k), (N,)

    # Innovation about the linearization point: dy = y_obs - h(z_pred).
    # H_eff maps physical-z corrections; normalized-space H is H_eff · y_std.
    dy      = win_df["log10_rho"].values - h_pred
    H_norm  = H_obs * y_std[None, :]                       # (N, k) normalized-space H
    n_valid = n_obs

    R_diag = compute_R_diag(
        df=win_df,
        use_npz_sigma=cfg.use_npz_sigma,
        alt_bins_km=cfg.alt_bins_km,
        sigma_by_bin=cfg.sigma_by_bin,
        sigma_floor=cfg.sigma_floor,
        sigma_cap=cfg.sigma_cap,
    )

    X_aug_post, P_aug_post, NIS = _augmented_information_update(
        X_aug_pred, P_aug, H_norm, dy, R_diag, k, p_max_var=p_max_var_norm,
    )

    z_post     = y_std * X_aug_post[:k] + y_mean
    P_post     = (y_std[:, None] * P_aug_post[:k, :k] * y_std[None, :])
    innov_post = dy - H_obs @ (z_post - z_pred_phys)
    innov_rms  = float(np.sqrt(np.mean(innov_post**2)))
    trace_P    = float(np.trace(P_post))

    elapsed = time.perf_counter() - t0
    print(
        f"  [{window_start.strftime('%Y-%m-%dT%H:%M')}]"
        f"  n_obs={n_obs}  n_valid={n_valid}"
        f"  NIS={NIS:.3f}  trace_P={trace_P:.4f}"
        f"  innov_rms={innov_rms:.3f}"
        f"  ({elapsed:.1f}s)"
    )

    return {
        **result_base,
        "z_post":      z_post,
        "P_post":      P_post,
        "diag_P":      np.diag(P_post),
        "n_valid":     n_valid,
        "NIS":         NIS,
        "trace_P":     trace_P,
        "innov_rms":   innov_rms,
        "_P_aug_post": P_aug_post,
        "_X_aug_post": X_aug_post,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main assimilation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_windowed_assimilation(
    obs_df: pd.DataFrame,
    interpolator,
    cfg,
    model_object: Optional[dict] = None,
) -> None:
    """Main loop: iterate 3-hour windows with checkpoint-resume support.

    State (z, P) is propagated across windows. After each window result is
    written to HDF5, the checkpoint JSON is updated. Re-running the same
    command skips completed windows automatically.

    Parameters
    ----------
    obs_df        : full DataFrame of EDR observations (all windows)
    interpolator  : the measurement operator -- RopeDataInterpolator (affine PCA
                    basis) or CAEMeasurementModel (COAE decoder).  Detected once
                    below; the two differ only inside _eval_logrho_H_seam.
    cfg           : EDRAssimConfig
    model_object  : full loadmat dict (optional); when provided enables:
                    - ic_array z0 initialisation (replaces z=0)
                    - sindy_step predict dynamics  (replaces identity)
                    - pu.propagate() open-loop reference (replaces flat zero)
    """
    k = cfg.n_components

    # Which decoder are we driving?  Resolved once here rather than per window, so a
    # 24-window run does one isinstance check instead of 24.
    is_nonlinear = _is_nonlinear_rom(interpolator)
    print(f"[assimilation] measurement operator: {type(interpolator).__name__}"
          f"  ({'nonlinear -- Jacobian re-evaluated at each window' if is_nonlinear else 'affine'})")

    # ── Config sanity checks ─────────────────────────────────────────────────
    if len(cfg.Q_diag) < k or len(cfg.P0_diag) < k:
        raise ValueError(
            f"Q_diag (len {len(cfg.Q_diag)}) and P0_diag (len {len(cfg.P0_diag)}) "
            f"must each have at least n_components={k} entries."
        )

    # ── Initialise output ────────────────────────────────────────────────────
    init_hdf5(cfg.hdf5_path, cfg)
    checkpoint = load_checkpoint(cfg.checkpoint_json)

    start_dt  = datetime.strptime(cfg.start_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_dt    = datetime.strptime(cfg.end_date,   "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    dt_window = timedelta(hours=cfg.window_hours)
    hours_per_window = int(round(cfg.window_hours))
    if abs(cfg.window_hours - hours_per_window) > 1e-9:
        raise ValueError(
            f"window_hours={cfg.window_hours} must be a whole number of hours; "
            "the SINDy predict advances the state in fixed 1-hour steps, so a "
            "fractional window desyncs the step count and driver indexing."
        )

    # ── SINDy dynamics setup (if model_object provided) ──────────────────────
    use_sindy        = model_object is not None
    dyn_params       = None
    x_drivers_interp = None
    z_ol_hourly      = None   # (T_hours, k) open-loop from propagate() — physical z
    z_buf            = None   # (k, max_lag+1) normalized-z history buffer for sindy_step

    # Normalization constants — used only by dynamics (_sindy_predict_step),
    # NOT by the observation model (which runs directly in physical z space).
    y_mean: Optional[np.ndarray] = None
    y_std:  Optional[np.ndarray] = None

    if use_sindy:
        from ..loaders.rom_loader import extract_dyn_params
        dyn_params = extract_dyn_params(model_object, k)
        _max_lag_warmup = dyn_params["max_lag"]  # warmup period for driver history
        x_drivers_interp = _build_x_drivers_interp(
            model_object, start_dt, end_dt, warmup_hours=_max_lag_warmup
        )

        y_mean = dyn_params["Y_norm"]["x_mean"].ravel()  # (k,)
        y_std  = dyn_params["Y_norm"]["x_std"].ravel()   # (k,)

        print("[assimilation] Pre-computing open-loop trajectory via pu.propagate()…")
        z_ol_hourly = _precompute_openloop(
            model_object, start_dt, end_dt, k, dyn_params["ar_lags"]
        )
        print(f"[assimilation] Open-loop shape: {z_ol_hourly.shape}")

    # ── Initial state — filter runs in PHYSICAL z space ──────────────────────
    # log10_rho = mu0 + U @ z_phys  (no normalization transform in H needed)
    # CONTRACT: cfg.P0_diag / cfg.Q_diag are ALWAYS specified in NORMALIZED
    # z-space units (per-mode variance after subtracting Y mean and dividing
    # by Y std). They are scaled by y_std**2 here to obtain the physical-z
    # covariance the filter operates on. This contract is enforced uniformly
    # across SINDy and static modes: in static mode the ROM .mat is still
    # consulted to recover y_std (a tiny extra load), eliminating the
    # historical unit ambiguity where the same YAML key meant different
    # things in different modes.
    if use_sindy:
        z_init = _init_z0_from_ic(model_object, start_dt, k)   # physical z
        print(f"[assimilation] z_init_phys[:3] = {z_init[:3].round(4)}")
    else:
        # Static (identity-predict) mode still loads y_std from the ROM .mat
        # so the YAML covariance units are unambiguous. Falls back to all-ones
        # only if Y_sindy_norm_dict is genuinely absent (legacy ROM file).
        if y_std is None:
            from ..loaders.rom_loader import load_model_object
            try:
                _mo = load_model_object(cfg.rom_mat_path)
                y_mean = np.asarray(_mo["Y_sindy_norm_dict"].x_mean,
                                    dtype=np.float64).ravel()[:k]
                y_std  = np.asarray(_mo["Y_sindy_norm_dict"].x_std,
                                    dtype=np.float64).ravel()[:k]
                print(f"[assimilation] static mode: y_std loaded from ROM "
                      f"(min/median/max = {y_std.min():.3g} / "
                      f"{np.median(y_std):.3g} / {y_std.max():.3g})")
            except Exception as e:
                print(f"[assimilation] WARNING: could not load y_std from ROM "
                      f"({e!r}); using unit normalization.")
                y_mean = np.zeros(k, dtype=np.float64)
                y_std  = np.ones(k, dtype=np.float64)
        z_init = np.zeros(k, dtype=np.float64)

    # Everything a reader needs to turn the stored normalized augmented posterior
    # (windows/*/X_aug_post, P_aug_post) into physical hourly z and covariance,
    # without loading the ROM. Metadata only; see drivers/query_state.py.
    with h5py.File(cfg.hdf5_path, "a") as f:
        f.attrs["y_mean"]  = y_mean
        f.attrs["y_std"]   = y_std
        f.attrs["max_lag"] = int(dyn_params["max_lag"]) if use_sindy else 0
        f.attrs["rom_kind"] = "coae" if is_nonlinear else "pca"

    # Scale normalized-space cfg variances to physical units once.
    p0_norm = np.asarray(cfg.P0_diag[:k], dtype=np.float64)
    P = np.diag(p0_norm * y_std**2)
    p_max_var_phys = _P_MAX_VAR * y_std**2
    # Normalized-space caps for the augmented (companion-form) path.
    p_max_var_norm = np.full(k, _P_MAX_VAR, dtype=np.float64)
    p_max_var_predict_aug = np.tile(
        np.full(k, _P_AUG_PREDICT_MAX_VAR, dtype=np.float64),
        (dyn_params["max_lag"] + 1) if use_sindy else 1,
    )

    z = z_init.copy()

    # Companion-form (augmented) covariance setup — mirrors ark_filter_windowed_Nsat.py.
    # The covariance lives in NORMALIZED z space; the physical top-block P (above) is
    # derived from it each window for output. n_aug = k * (max_lag + 1).
    A_aug = Q_aug = P_aug = jac_cache = None
    if use_sindy:
        max_lag = dyn_params["max_lag"]  # == _max_lag_warmup
        # Buffer stores NORMALIZED z for the dynamics; initialize from physical z_init
        z_init_norm = (z_init - y_mean) / y_std
        z_buf = np.tile(z_init_norm[:, None], (1, max_lag + 1))  # (k, max_lag+1)

        p_blocks = max_lag + 1
        n_aug    = k * p_blocks
        A_aug    = _build_companion_matrix(dyn_params, k)
        p0_norm  = np.asarray(cfg.P0_diag[:k], dtype=np.float64)
        # Block-Toeplitz init: cross-blocks equal the diagonal blocks because
        # z_buf is tiled with the same z_init_norm at every lag — the mean
        # encodes 100% correlation across lags, so the covariance must too. A
        # zero-cross-block init would lie about lag-vs-current independence
        # and bias the EKF predict for the first ~max_lag windows.
        P0_block = np.diag(p0_norm)
        P_aug    = np.zeros((n_aug, n_aug), dtype=np.float64)
        for i in range(p_blocks):
            for j in range(p_blocks):
                P_aug[i * k:(i + 1) * k, j * k:(j + 1) * k] = P0_block
        Q_aug = np.zeros((n_aug, n_aug), dtype=np.float64)   # process noise: top block only
        Q_aug[:k, :k] = np.diag(np.asarray(cfg.Q_diag[:k], dtype=np.float64))
        jac_cache = pu.sindy_jacobian_cache_build(
            model_params=dyn_params["model_params"],
            f10_idx=dyn_params["f10_idx"],
            kp_idx=dyn_params["kp_idx"],
            n_components=k,
            ar_drivers_start=dyn_params["ar_drivers_start"],
            X_reg_norm_dict=dyn_params["X_norm"],
            Y_reg_norm_dict=dyn_params["Y_norm"],
        )

    # ── Window iteration ─────────────────────────────────────────────────────
    window_start    = start_dt
    n_windows_total = int(np.ceil((end_dt - start_dt) / dt_window))
    n_done          = 0
    window_idx      = 0   # counts windows from 0 for driver indexing

    print(f"\n{'='*70}")
    print(f"EDR Assimilation: {cfg.start_date} → {cfg.end_date}")
    print(f"Windows: {n_windows_total}  |  window_hours={cfg.window_hours}"
          f"  |  mode={'sindy' if use_sindy else 'static'}  |  k={k}")
    print(f"{'='*70}\n")

    while window_start < end_dt:
        window_end = min(window_start + dt_window, end_dt)
        wkey = window_start.strftime("%Y-%m-%dT%H:%M:%S")

        # ── Checkpoint skip ──────────────────────────────────────────────────
        if checkpoint.get(wkey, {}).get("status") == "ok":
            if n_done == 0:
                # HDF5 stores physical z; load directly into filter state
                z, P = load_last_state_from_hdf5(cfg.hdf5_path)
                if use_sindy:
                    # Try to restore the augmented state exactly. If the HDF5
                    # file predates augmented-state persistence (older runs),
                    # fall back to seeding lag blocks from P0 — biased for ~max_lag
                    # windows after resume but better than crashing.
                    aug = load_last_augmented_state_from_hdf5(cfg.hdf5_path)
                    if aug is not None:
                        # X_aug_ck is derivable from z_buf (top block = current
                        # state); we only need P_aug and z_buf for the next
                        # predict step's mean/covariance propagation.
                        _, P_aug_ck, z_buf_ck = aug
                        P_aug = P_aug_ck.copy()
                        z_buf = z_buf_ck.copy()
                        print(f"[assimilation] resume: restored augmented state "
                              f"(P_aug {P_aug.shape}, z_buf {z_buf.shape})")
                    else:
                        print("[assimilation] resume: HDF5 has no augmented "
                              "state — falling back to P0-seeded lag blocks.")
                        z_norm_ck = (z - y_mean) / y_std
                        z_buf = np.tile(z_norm_ck[:, None],
                                        (1, dyn_params["max_lag"] + 1))
                        P_aug = np.zeros_like(P_aug)
                        p0_norm_resume = np.asarray(cfg.P0_diag[:k],
                                                    dtype=np.float64)
                        P0_block_resume = np.diag(p0_norm_resume)
                        for i in range(1, dyn_params["max_lag"] + 1):
                            for j in range(1, dyn_params["max_lag"] + 1):
                                P_aug[i*k:(i+1)*k, j*k:(j+1)*k] = P0_block_resume
                        P_aug[:k, :k] = (P / y_std[:, None]) / y_std[None, :]
            window_start = window_end
            n_done       += 1
            window_idx   += 1
            continue

        # ── Slice observations for this window ───────────────────────────────
        mask   = (obs_df["epoch_mid"] >= window_start) & (obs_df["epoch_mid"] < window_end)
        win_df = obs_df[mask]

        if use_sindy:
            # ── SINDy predict: 3 hourly steps; mean via sindy_loop dynamics,
            #    covariance via the companion-form EKF (matches ark_filter_Nsat). ──
            z_cur    = z.copy()   # physical z (mean)
            base_idx = window_idx * hours_per_window
            for step in range(hours_per_window):
                # +max_lag offsets into the warmup-extended x_drivers_interp so AR-driver
                # lookups map to real history rather than being clamped to column 0.
                interp_idx = base_idx + step + max_lag
                # EKF Jacobian linearized at the current (pre-step) normalized state.
                z_k_norm = ((z_cur - y_mean) / y_std).reshape(-1, 1)
                F_jac = pu.sindy_process_jacobian(
                    z_k_norm=z_k_norm,
                    x_drivers_interp=x_drivers_interp,
                    interp_idx=interp_idx,
                    A_aug=A_aug,
                    B_nonlinear=dyn_params["B_nonlinear"],
                    ar_drivers_start=dyn_params["ar_drivers_start"],
                    n_components=k,
                    f10_idx=dyn_params["f10_idx"],
                    kp_idx=dyn_params["kp_idx"],
                    model_params=dyn_params["model_params"],
                    X_reg_norm_dict=dyn_params["X_norm"],
                    Y_reg_norm_dict=dyn_params["Y_norm"],
                    normalization_method=dyn_params["normalization_method"],
                    jac_cache=jac_cache,
                )
                z_cur, z_buf = _sindy_predict_step(
                    z_buf, dyn_params, interp_idx, x_drivers_interp,
                    y_mean, y_std,
                )
                P_aug = F_jac @ P_aug @ F_jac.T + Q_aug
                # Loose per-mode overflow guard (open-loop modes can have
                # spectral radius > 1). Cap is replicated across the lag blocks.
                _pd = np.diag(P_aug)
                if np.any(_pd > p_max_var_predict_aug):
                    _s = np.sqrt(np.minimum(
                        p_max_var_predict_aug / np.maximum(_pd, 1e-300), 1.0))
                    P_aug = _s[:, None] * P_aug * _s[None, :]
                    P_aug = 0.5 * (P_aug + P_aug.T)

            z_pred_phys = z_cur
            # Augmented prior mean from the (normalized) history buffer: block i = z_{k-i}.
            X_aug_pred = np.concatenate([z_buf[:, -1 - i] for i in range(max_lag + 1)])

            result = _process_window_augmented(
                window_start, window_end, win_df, z_pred_phys, X_aug_pred,
                P_aug, interpolator, cfg, y_mean, y_std,
                p_max_var_norm=p_max_var_norm,
                is_nonlinear=is_nonlinear,
            )
            P_aug = result["_P_aug_post"]
            X_aug_post = result["_X_aug_post"]
            # Sync the mean history buffer with the posterior augmented lags so the
            # next predict steps from the corrected state.
            for i in range(max_lag + 1):
                z_buf[:, -1 - i] = X_aug_post[i * k:(i + 1) * k]
            # Persist the buffer alongside the augmented posterior so a resume
            # can reconstruct lag history exactly. _z_buf_post is the post-sync
            # buffer (matches X_aug_post by construction).
            result["_z_buf_post"] = z_buf.copy()
        else:
            # ── Static (non-sindy) path: identity predict + (k,k) update. ─────
            result = process_window(
                window_start=window_start,
                window_end=window_end,
                win_df=win_df,
                z_prior=z,
                P_prior=P,
                interpolator=interpolator,
                cfg=cfg,
                p_max_var=p_max_var_phys,
                y_std=y_std,
                is_nonlinear=is_nonlinear,
            )

        # ── Open-loop trajectory for this window ─────────────────────────────
        # pu.propagate() returns physical z; store as-is in HDF5. z_post is the
        # end-of-window state (predict advances hours_per_window steps from
        # window_start), so sample the open-loop at window_end — one window ahead
        # of window_start — to compare like-for-like in time.
        if use_sindy and z_ol_hourly is not None:
            ol_idx = min((window_idx + 1) * hours_per_window, z_ol_hourly.shape[0] - 1)
            result["z_openloop"] = z_ol_hourly[ol_idx].copy()
        else:
            result["z_openloop"] = np.zeros(k, dtype=np.float64)

        # ── Update filter state (z and P are in physical space) ──────────────
        z = result["z_post"]
        P = result["P_post"]

        # ── Write to HDF5 (z is physical — no conversion needed) ─────────────
        append_window_to_hdf5(cfg.hdf5_path, result)

        # ── Update checkpoint ────────────────────────────────────────────────
        checkpoint[wkey] = {
            "status":  "ok",
            "n_obs":   result["n_obs"],
            "n_valid": result["n_valid"],
        }
        save_checkpoint(cfg.checkpoint_json, checkpoint)

        n_done       += 1
        window_idx   += 1
        window_start  = window_end

    print(f"\nDone. {n_done} windows processed. Output: {cfg.hdf5_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Density reconstruction utility
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_density_grid(
    z_post: np.ndarray,
    interpolator,
) -> np.ndarray:
    """Reconstruct the full 3D log10-density field from ROM coefficients.

    Returns log10_rho_grid (N_grid,), flattened C-order over (n_lt, n_lat, n_alt).

        pca  : mu0_flat + U0_flat @ z_post
        coae : decode_full_field(z_post), one decoder pass over the whole grid

    Both reduced-order models are defined on the same 72 x 36 x 45 grid, so the
    flattened output is directly comparable between them.
    """
    if _is_nonlinear_rom(interpolator):
        # decode_full_field wants a batch dimension: a bare (k,) vector reaches the
        # TorchScript decoder's first Linear as (k, 1) and fails with
        # "input and weight.T shapes cannot be multiplied".  Reshape here rather than
        # relying on the caller.
        field = interpolator.decode_full_field(
            np.asarray(z_post, dtype=np.float64).reshape(1, -1))
        return np.asarray(field, dtype=np.float64).reshape(-1)

    n_lt, n_lat, n_alt, k = interpolator.U0.shape
    N = n_lt * n_lat * n_alt

    U0_flat  = interpolator.U0.reshape(N, k)         # (N_grid, k)
    mu0_flat = interpolator.mu0.reshape(N)            # (N_grid,)

    return mu0_flat + U0_flat @ z_post                # (N_grid,)
