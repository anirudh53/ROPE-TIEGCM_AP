"""
Per-observation measurement noise variance for the EDR assimilation.

Priority:
  1. Use sigma_log_rho from the calibrated NPZ when available and finite.
  2. Fall back to an altitude-binned empirical lookup table.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_R_diag(
    df: pd.DataFrame,
    use_npz_sigma: bool,
    alt_bins_km: Tuple[float, ...],
    sigma_by_bin: Tuple[float, ...],
    sigma_floor: float = 0.10,
    sigma_cap: float = 1.00,
) -> np.ndarray:
    """Compute per-observation variance R_i = sigma_i^2 in log10(rho) units.

    Parameters
    ----------
    df            : DataFrame with columns alt_km, sigma_log_rho
    use_npz_sigma : if True, use df["sigma_log_rho"] where finite
    alt_bins_km   : bin edges (ascending), length B+1
    sigma_by_bin  : sigma (log10 units) for each of B bins, length B
    sigma_floor   : minimum allowed sigma
    sigma_cap     : maximum allowed sigma

    Returns
    -------
    R_diag : (N,) variance values (sigma^2), one per row of df
    """
    n = len(df)
    sigma = np.empty(n, dtype=np.float64)

    # ── Start with empirical fallback for all rows ───────────────────────────
    bins_arr = np.asarray(alt_bins_km, dtype=np.float64)
    sig_arr  = np.asarray(sigma_by_bin, dtype=np.float64)
    n_bins   = len(sig_arr)

    bin_idx = np.digitize(df["alt_km"].values, bins_arr) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sigma[:] = sig_arr[bin_idx]

    # ── Override with NPZ sigma where available ──────────────────────────────
    if use_npz_sigma and "sigma_log_rho" in df.columns:
        npz_sigma = df["sigma_log_rho"].values.astype(np.float64)
        good = np.isfinite(npz_sigma) & (npz_sigma > 0)
        sigma[good] = npz_sigma[good]

    # ── Apply floor and cap ──────────────────────────────────────────────────
    sigma = np.clip(sigma, sigma_floor, sigma_cap)

    return sigma ** 2
