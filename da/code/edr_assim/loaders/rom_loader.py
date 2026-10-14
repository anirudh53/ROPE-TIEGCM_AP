"""
Load the ROM model from a .mat file and instantiate a RopeDataInterpolator.

Mirrors the setup_rom_interpolator pattern from
PyDensityEstimation/2026-01-09_v2/ark_filter_windowed_Nsat.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio

# ── Locate RopeDataInterpolator from the existing codebase ───────────────────
_PROPAGATOR_DIR = Path(__file__).resolve().parents[3] / "PyDensityEstimation" / "2026-01-09_v2"
if str(_PROPAGATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_PROPAGATOR_DIR))

import propagator_utilities_V4 as pu  # noqa: E402  (late import after path setup)
import scipy.io.matlab as _sio_mat


def _mat_struct_to_dict(obj):
    """Convert a scipy mat_struct to a plain dict (top-level only, non-recursive)."""
    if isinstance(obj, _sio_mat._mio5_params.mat_struct):
        return {f: getattr(obj, f) for f in obj._fieldnames}
    return obj


def _build_model_params_dict(mp):
    """Convert model_params mat_struct to a dict, rebuilding 'functions' as callables."""
    d = _mat_struct_to_dict(mp)
    # 'functions' is a mat_struct of MATLAB function handles; reconstruct as lambda dict
    if "functions" in d and isinstance(d["functions"], _sio_mat._mio5_params.mat_struct):
        d["functions"] = pu.mat_to_lambda_dict(d["functions"])
    return d


# Grid dimensions — fixed by the ROM training procedure
_N_LT  = 72
_N_LAT = 36
_N_ALT = 45

# Driver row indices (74-row u_ar_ds layout: kpstd inserted at row 7)
_F10_IDX = 5
_KP_IDX  = 8

# Offset between ar_drivers_start (B-matrix column space) and the AR-driver columns
# of the X-regressor normalization dict. Matches `delta_idx` in
# propagator_utilities_V4.sindy_loop_optimized — the validated forecast path.
_AR_DRIVERS_NORM_OFFSET = 5


def load_rom(
    mat_path: str,
    n_components: int = 10,
) -> Tuple[pu.RopeDataInterpolator, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load ROM model from .mat file, return (interpolator, k, A_dyn, Q_dyn).

    Parameters
    ----------
    mat_path     : path to the aligned .mat ROM file
    n_components : number of ROM modes to use; must not exceed modes in the file

    Returns
    -------
    interpolator : RopeDataInterpolator  — get_H(lla) where lla = [lat, lst, alt]
    k            : int  actual number of modes loaded (== n_components)
    A_dyn        : None  (dynamic mode not supported)
    Q_dyn        : None  (dynamic mode not supported)
    """
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # U: (N_grid, K_full) → truncate to n_components → reshape to (n_lt, n_lat, n_alt, k)
    K_full = mat["U"].shape[1]
    if n_components > K_full:
        raise ValueError(
            f"[rom_loader] Requested n_components={n_components} exceeds "
            f"available modes K_full={K_full} in {mat_path!r}."
        )
    k = n_components
    U_flat = np.asarray(mat["U"][:, :k], dtype=np.float64)          # (N_grid, k)
    mu_flat = np.asarray(mat["mu"], dtype=np.float64).reshape(-1)    # (N_grid,)

    U0  = U_flat.reshape((_N_LT, _N_LAT, _N_ALT, k), order="C")
    mu0 = mu_flat.reshape((_N_LT, _N_LAT, _N_ALT), order="C")

    interpolator = pu.RopeDataInterpolator(U0=U0, mu0=mu0)

    print(
        f"[rom_loader] Loaded ROM: k={k}/{K_full} modes"
        f" | U0 {U0.shape} | mu0 {mu0.shape}"
    )

    return interpolator, k, None, None


def load_measurement_model(cfg):
    """Build the measurement operator named by ``cfg``, PCA or COAE.

    Returns (meas_model, k, is_nonlinear).

    ``cfg.rom_config_path`` unset  -> the historical path: read U / mu out of
    ``cfg.rom_mat_path`` and return a ``pu.RopeDataInterpolator``.  Bit-identical to
    what this module has always done, so no PCA result can move.

    ``cfg.rom_config_path`` set    -> delegate to PyDensityEstimation's ``rom_config``,
    which is already the single place that knows how to turn a YAML into either a
    ``RopeDataInterpolator`` (reduction_method: pca) or a ``CAEMeasurementModel``
    (reduction_method: coae).  Reusing it means the decoder path, the stats_cae
    denormalisation and the CUDA precision settings are maintained in exactly one
    place rather than forked into this repo.

    Both objects expose ``h_and_H(z, lla) -> (log10_rho (n,), H (n, k))``; the PCA one
    additionally exposes the affine ``get_H``.  ``is_nonlinear`` says which contract
    the caller should use, and is what selects the branch in
    ``_eval_logrho_H_seam`` / ``reconstruct_density_grid``.
    """
    if not getattr(cfg, "rom_config_path", None):
        interp, k, _, _ = load_rom(mat_path=cfg.rom_mat_path,
                                   n_components=cfg.n_components)
        return interp, k, False

    import rom_config  # from PyDensityEstimation/2026-01-09_v2, already on sys.path

    rc = rom_config.load_config(cfg.rom_config_path)
    k = int(min(int(cfg.n_components), int(rc["n_components"])))
    rc["n_components"] = k

    # The .mat named by the EDR config is authoritative for the dynamics; the one
    # named inside the rom_config YAML must be the same model, or the decoder and the
    # SINDy operator would come from different trainings.  Fail loudly rather than
    # silently mixing them.
    import os
    a = os.path.realpath(rc["model_path"])
    b = os.path.realpath(cfg.rom_mat_path)
    if a != b:
        raise ValueError(
            f"[rom_loader] rom_config model_path and EDR rom_mat_path disagree:\n"
            f"    rom_config : {a}\n"
            f"    edr config : {b}\n"
            "The decoder and the SINDy dynamics must come from the same .mat.")

    model_object = load_model_object(cfg.rom_mat_path)
    mm = rom_config.build_measurement_model(rc, model_object)
    is_nonlinear = (rc["reduction_method"] != "pca")

    print(f"[rom_loader] Loaded ROM via rom_config: "
          f"reduction_method={rc['reduction_method']}  k={k}"
          + (f"  device={getattr(mm, 'device', 'n/a')}" if is_nonlinear else ""))
    return mm, k, is_nonlinear


def model_n_components(model_object: dict) -> int:
    """Modes available in a .mat, for either reduced-order model.

    The PCA models carry the basis as ``U (N_grid, K)``; the COAE models carry no
    basis at all -- the decoder is a separate TorchScript file -- so ``U`` is absent
    and the mode count has to come from the ``n_components`` field, which both write.
    """
    if "n_components" in model_object:
        return int(np.asarray(model_object["n_components"]).ravel()[0])
    return int(np.asarray(model_object["U"]).shape[1])


def load_model_object(mat_path: str) -> dict:
    """Return the full loadmat dict from the ROM .mat file.

    Needed by pu.propagate() and sindy_step() for dynamics propagation.
    """
    return sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)


def extract_dyn_params(model_object: dict, n_components: int) -> Dict:
    """Extract sindy_step / propagate parameters from the model_object dict.

    Uses T1=T2=3600 s (hourly model steps) so change_resolution is a no-op
    and sindy_step is called once per hour with direct driver indexing.

    Returns a flat dict consumed by _sindy_predict_step() in the assimilation.
    """
    k   = n_components
    mp  = model_object["model_params"]

    B_full = np.asarray(model_object["B_sindy"], dtype=np.float64)  # (k, n_cols)
    A_d    = B_full[:, -k:].copy()    # (k, k) — 1-hour state transition
    B_d    = B_full[:, :-k].copy()    # (k, n_cols-k) — 1-hour driver matrix

    # change_resolution with T1=T2=3600 is identity (A^1), kept for API symmetry
    A_d2, B_d2 = pu.change_resolution(A_d, B_d, T1=3600.0, T2=3600.0)

    ar_lags   = list(np.asarray(mp.ar_lags).astype(int))
    n_lags    = len(ar_lags)
    max_lag   = max(ar_lags)

    # B_full column layout (from ark_filter_Nsat.py lines 202-210):
    #   [0 : ar_drivers_start)           → SINDy nonlinear library
    #   [ar_drivers_start : ar_state_start) → AR drivers (3 * max_lag cols)
    #   [ar_state_start : -k)             → AR state lags (max_lag * k cols)
    #   [-k :]                            → A_d (linear state)
    ar_drivers_start = B_full.shape[1] - k - max_lag * k - 3 * max_lag
    ar_state_start   = B_full.shape[1] - k - n_lags * k

    B_nonlinear      = B_d2[:, :ar_drivers_start]
    B_ar_drivers_lst = [
        B_d2[:, ar_drivers_start + 3 * i : ar_drivers_start + 3 * (i + 1)]
        for i in range(n_lags)
    ]
    B_ar_state_lst = [
        B_d2[:, ar_state_start + i * k : ar_state_start + (i + 1) * k]
        for i in range(n_lags)
    ]

    def _norm_dict(obj):
        return {
            "x_mean": np.asarray(obj.x_mean, dtype=np.float64).reshape(-1, 1),
            "x_std":  np.asarray(obj.x_std,  dtype=np.float64).reshape(-1, 1),
        }

    Y_norm = _norm_dict(model_object["Y_sindy_norm_dict"])
    X_norm = _norm_dict(model_object["X_sindy_norm_dict"])

    # Stacked AR matrices for the einsum-based predict step (mirrors
    # sindy_loop_optimized in propagator_utilities_V4, the validated forecast path).
    B_ar_state_stack   = np.stack(B_ar_state_lst,   axis=0)   # (n_lags, k, k)
    B_ar_drivers_stack = np.stack(B_ar_drivers_lst, axis=0)   # (n_lags, k, 3)

    # AR-driver normalization columns. sindy_loop_optimized offsets the
    # X-regressor normalization slice by delta_idx=5 relative to ar_drivers_start;
    # omitting this offset (as pu.sindy_step does) normalizes F10/F10A/Kp with the
    # wrong mean/std and makes the SINDy forecast diverge.
    sl0 = ar_drivers_start - _AR_DRIVERS_NORM_OFFSET
    sl1 = ar_drivers_start + 3 * max_lag - _AR_DRIVERS_NORM_OFFSET
    X_ar_drivers_norm = {
        "x_mean": X_norm["x_mean"][sl0:sl1],
        "x_std":  X_norm["x_std"][sl0:sl1],
    }

    print(
        f"[rom_loader] Dynamics params: k={k}  ar_lags={ar_lags}"
        f"  ar_drivers_start={ar_drivers_start}  ar_state_start={ar_state_start}"
    )

    return {
        "A_d":               A_d2,                  # (k, k) 1-hour linear state matrix
        "B_nonlinear":       B_nonlinear,
        "B_ar_drivers_lst":  B_ar_drivers_lst,
        "B_ar_state_lst":    B_ar_state_lst,
        "B_ar_state_stack":     B_ar_state_stack,
        "B_ar_drivers_stack":   B_ar_drivers_stack,
        "X_ar_drivers_norm":    X_ar_drivers_norm,
        "ar_lags":           ar_lags,
        "max_lag":           max_lag,
        "f10_idx":           _F10_IDX,
        "kp_idx":            _KP_IDX,
        "Y_norm":            Y_norm,
        "X_norm":            X_norm,
        "ar_drivers_start":  ar_drivers_start,
        "normalization_method": str(model_object["normalization_method"]),
        "input_features":    model_object["input_features"],
        "model_params":      _build_model_params_dict(model_object["model_params"]),
    }
