"""
Load EDR density estimates from NPZ files produced by the TLE density pipeline.

Three input formats are supported and auto-detected:

  * Midpoint+arcpts NPZ   — one row per arc; arc points embedded with pts_
                             prefix (pts_arc_idx, pts_lat, pts_lst, pts_alt).
                             Detected by presence of ``pts_arc_idx``.
  * Legacy arcpts NPZ     — separate two-table file with ``arc_idx`` key.
  * Legacy midpoint NPZ   — one row per arc, no arc-point arrays.

Either way the returned DataFrame has one row per *arc* and carries the arc's
sample positions in object columns `pts_lat`, `pts_lst`, `pts_alt`. For the
legacy midpoint file each arc is a single point, so those arrays have length 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


_SIGMA_UNITS_VALID = ("log10", "ln", "auto")
_LN10 = float(np.log(10.0))


def _coerce_sigma(sigma_raw: np.ndarray, units: str = "auto") -> np.ndarray:
    """Convert sigma to log10 units.

    units : one of "log10" (no conversion), "ln" (divide by ln 10),
            or "auto" (median-based heuristic, kept for backward compatibility
            but emits a notice; explicit "log10"/"ln" should be preferred).

    The historical "auto" heuristic compared the median to 2.0 (≈ ln 10); files
    whose median sat just under the threshold but with a tail of ln-unit values
    would leak un-converted ln tails into the clipping stage. Explicit units
    eliminate the ambiguity. The heuristic now operates on positive *finite*
    values only and prints a notice with the decision boundary so callers can
    detect borderline files.
    """
    if units not in _SIGMA_UNITS_VALID:
        raise ValueError(
            f"sigma_units={units!r} must be one of {_SIGMA_UNITS_VALID}"
        )
    sigma_raw = np.asarray(sigma_raw, dtype=np.float64)
    if units == "log10":
        return sigma_raw
    if units == "ln":
        return sigma_raw / _LN10
    pos = sigma_raw[np.isfinite(sigma_raw) & (sigma_raw > 0)]
    if pos.size == 0:
        return sigma_raw
    med = float(np.median(pos))
    looks_ln = med > 2.0   # ln(10) ≈ 2.303 → values that big are almost
                            # certainly ln-units (log10 sigmas rarely exceed 1).
    if looks_ln:
        print(f"[edr_loader] sigma_units=auto: median(σ⁺)={med:.3g} > 2.0 → "
              "treating as ln units, dividing by ln(10).")
        return sigma_raw / _LN10
    if med > 1.5:
        print(f"[edr_loader] sigma_units=auto: median(σ⁺)={med:.3g} is borderline "
              "(close to ln(10)). If this file's sigmas are in ln units, pass "
              "sigma_units='ln' explicitly to avoid an under-conversion.")
    return sigma_raw


def _parse_epoch(epoch_raw: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime([str(e).rstrip("Z") for e in epoch_raw], utc=True)


def load_edr_npz(
    npz_path: str,
    prefer_calibrated: bool = True,
    log10_clip_lo: float = -15.0,
    log10_clip_hi: float = -7.0,
    min_alt_km: float = 150.0,
    max_alt_km: float = 980.0,
    sigma_units: str = "auto",
) -> pd.DataFrame:
    """Load an EDR NPZ (midpoint or arcpts) into a per-arc DataFrame.

    Returns
    -------
    DataFrame with columns:
        norad_id        : int32
        epoch_mid       : pd.Timestamp (UTC, tz-aware)
        rho_use         : float64 [kg/m3]
        log10_rho       : float64 = log10(rho_use), clipped
        alt_km          : float64  (arc-mean altitude; used for the R-bin lookup)
        sigma_log_rho   : float64 (NaN if not in NPZ)
        n_arc_pts       : int32
        pts_lat         : object  (1-D np.ndarray of arc-point latitudes [deg])
        pts_lst         : object  (1-D np.ndarray of arc-point LST [hours])
        pts_alt         : object  (1-D np.ndarray of arc-point altitudes [km])
        cal_available   : bool
    """
    d = np.load(npz_path, allow_pickle=True)
    if "pts_arc_idx" in d.files:
        return _load_midpoint_arcpts(d, npz_path, prefer_calibrated,
                                     log10_clip_lo, log10_clip_hi,
                                     min_alt_km, max_alt_km, sigma_units)
    if "arc_idx" in d.files:
        return _load_arcpts(d, npz_path, log10_clip_lo, log10_clip_hi,
                            min_alt_km, max_alt_km, sigma_units)
    return _load_midpoint(d, npz_path, prefer_calibrated, log10_clip_lo,
                          log10_clip_hi, min_alt_km, max_alt_km, sigma_units)


def _load_midpoint(
    d, npz_path, prefer_calibrated, log10_clip_lo, log10_clip_hi,
    min_alt_km, max_alt_km, sigma_units,
) -> pd.DataFrame:
    """One row per arc, geolocated at the midpoint (legacy format)."""
    # ── Weight guard (midpoint only) ─────────────────────────────────────────
    # The midpoint NPZ has weight == 1 for every row. The arc-mean sibling has
    # weight = 1/n_arc_pts; loading it here would silently inflate the
    # information matrix. Reject anything that isn't the midpoint file.
    if "weight" in d.files:
        weight = np.asarray(d["weight"], dtype=np.float64)
        if not np.all(weight == 1):
            raise ValueError(
                f"[edr_loader] 'weight' column in {npz_path!r} contains values != 1 "
                f"(min={weight.min():.4g}, max={weight.max():.4g}). "
                "Use edr_densities_geolocated_midpoint.npz or the arcpts variant."
            )

    # ── Density selection ────────────────────────────────────────────────────
    cal_available = False
    if prefer_calibrated and "rho_cal" in d.files:
        rho_use = np.asarray(d["rho_cal"], dtype=np.float64)
        cal_available = True
    elif "rho_kg_m3" in d.files:
        rho_use = np.asarray(d["rho_kg_m3"], dtype=np.float64)
    elif "rho" in d.files:
        rho_use = np.asarray(d["rho"], dtype=np.float64)
    else:
        raise KeyError(f"No density field found in {npz_path}. Available: {d.files}")

    if "sigma_log_rho" in d.files:
        sigma_raw = _coerce_sigma(d["sigma_log_rho"], units=sigma_units)
    else:
        sigma_raw = np.full(len(rho_use), np.nan)

    df = pd.DataFrame({
        "norad_id":      np.asarray(d["norad_id"], dtype=np.int32),
        "epoch_mid":     _parse_epoch(np.asarray(d["epoch_mid"])),
        "rho_use":       rho_use,
        "alt_km":        np.asarray(d["alt_km"], dtype=np.float64),
        "lat_deg":       np.asarray(d["lat_deg"], dtype=np.float64),
        "lon_deg":       np.asarray(d["lon_deg"], dtype=np.float64),
        "lst_hours":     np.asarray(d["lst_hours"], dtype=np.float64),
        "sigma_log_rho": sigma_raw,
        "cal_available": cal_available,
    })

    valid = (
        np.isfinite(df["rho_use"])
        & (df["rho_use"] > 0)
        & (df["alt_km"] >= min_alt_km)
        & (df["alt_km"] <= max_alt_km)
        & np.isfinite(df["lat_deg"])
        & np.isfinite(df["lst_hours"])
    )
    df = df[valid].reset_index(drop=True)
    df["log10_rho"] = np.clip(np.log10(df["rho_use"]), log10_clip_lo, log10_clip_hi)
    df = df.sort_values("epoch_mid").reset_index(drop=True)

    # Wrap each midpoint as a length-1 arc so the assimilation path is uniform.
    df["pts_lat"]   = [np.array([v], dtype=np.float64) for v in df["lat_deg"].values]
    df["pts_lst"]   = [np.array([v], dtype=np.float64) for v in df["lst_hours"].values]
    df["pts_alt"]   = [np.array([v], dtype=np.float64) for v in df["alt_km"].values]
    df["n_arc_pts"] = np.ones(len(df), dtype=np.int32)

    n_total, n_kept = len(rho_use), len(df)
    print(
        f"[edr_loader] Loaded {n_kept}/{n_total} midpoint arcs"
        f" | calibrated={cal_available}"
        f" | date range: {df['epoch_mid'].min()} → {df['epoch_mid'].max()}"
        f" | unique NORADs: {df['norad_id'].nunique()}"
    )
    return df


def _load_midpoint_arcpts(
    d, npz_path, prefer_calibrated, log10_clip_lo, log10_clip_hi,
    min_alt_km, max_alt_km, sigma_units,
) -> pd.DataFrame:
    """Midpoint NPZ with embedded arc points (pts_arc_idx / pts_lat / pts_lst / pts_alt).

    Arc-level scalars (lat_deg, lon_deg, lst_hours, alt_km) are preserved under
    their original names for backward compatibility with downstream code.
    """
    cal_available = False
    if prefer_calibrated and "rho_cal" in d.files:
        rho_use = np.asarray(d["rho_cal"], dtype=np.float64)
        cal_available = True
    elif "rho_kg_m3" in d.files:
        rho_use = np.asarray(d["rho_kg_m3"], dtype=np.float64)
    elif "rho" in d.files:
        rho_use = np.asarray(d["rho"], dtype=np.float64)
    else:
        raise KeyError(f"No density field found in {npz_path}. Available: {d.files}")

    sigma_raw = (_coerce_sigma(d["sigma_log_rho"], units=sigma_units)
                 if "sigma_log_rho" in d.files
                 else np.full(len(rho_use), np.nan))

    df = pd.DataFrame({
        "norad_id":      np.asarray(d["norad_id"],   dtype=np.int32),
        "epoch_mid":     _parse_epoch(np.asarray(d["epoch_mid"])),
        "rho_use":       rho_use,
        "alt_km":        np.asarray(d["alt_km"],     dtype=np.float64),
        "lat_deg":       np.asarray(d["lat_deg"],    dtype=np.float64),
        "lon_deg":       np.asarray(d["lon_deg"],    dtype=np.float64),
        "lst_hours":     np.asarray(d["lst_hours"],  dtype=np.float64),
        "sigma_log_rho": sigma_raw,
        "cal_available": cal_available,
    })

    # Point-level arc trajectory
    p_arc = np.asarray(d["pts_arc_idx"], dtype=np.int64)
    p_lat = np.asarray(d["pts_lat"],     dtype=np.float64)
    p_lst = np.asarray(d["pts_lst"],     dtype=np.float64)
    p_alt = np.asarray(d["pts_alt"],     dtype=np.float64)

    pmask = (
        np.isfinite(p_lat) & np.isfinite(p_lst) & np.isfinite(p_alt)
        & (p_alt >= min_alt_km) & (p_alt <= max_alt_km)
    )
    pts = pd.DataFrame({
        "arc_idx": p_arc[pmask],
        "lat":     p_lat[pmask],
        "lst":     p_lst[pmask],
        "alt":     p_alt[pmask],
    })
    grp     = pts.groupby("arc_idx", sort=True)
    pts_lat = grp["lat"].apply(lambda s: s.to_numpy())
    pts_lst = grp["lst"].apply(lambda s: s.to_numpy())
    pts_alt = grp["alt"].apply(lambda s: s.to_numpy())

    # Attach grouped arc-point arrays before filtering (index = sequential arc_idx)
    n_arcs = len(df)
    df["pts_lat"] = [pts_lat.get(i, np.array([df["lat_deg"].iat[i]])) for i in range(n_arcs)]
    df["pts_lst"] = [pts_lst.get(i, np.array([df["lst_hours"].iat[i]])) for i in range(n_arcs)]
    df["pts_alt"] = [pts_alt.get(i, np.array([df["alt_km"].iat[i]]))   for i in range(n_arcs)]

    valid = (
        np.isfinite(df["rho_use"])
        & (df["rho_use"] > 0)
        & (df["alt_km"] >= min_alt_km)
        & (df["alt_km"] <= max_alt_km)
        & np.isfinite(df["lat_deg"])
        & np.isfinite(df["lst_hours"])
    )
    df = df[valid].reset_index(drop=True)
    df["n_arc_pts"] = np.array([len(a) for a in df["pts_lat"]], dtype=np.int32)
    df["log10_rho"] = np.clip(np.log10(df["rho_use"]), log10_clip_lo, log10_clip_hi)
    df = df.sort_values("epoch_mid").reset_index(drop=True)

    n_total = len(rho_use)
    print(
        f"[edr_loader] Loaded {len(df)}/{n_total} midpoint+arcpts arcs"
        f" | calibrated={cal_available}"
        f" | mean pts/arc: {df['n_arc_pts'].mean():.1f}"
        f" | date range: {df['epoch_mid'].min()} → {df['epoch_mid'].max()}"
        f" | unique NORADs: {df['norad_id'].nunique()}"
    )
    return df


def _load_arcpts(
    d, npz_path, log10_clip_lo, log10_clip_hi, min_alt_km, max_alt_km, sigma_units,
) -> pd.DataFrame:
    """One row per arc with attached per-arc-point geolocations."""
    # ── Point-level table ────────────────────────────────────────────────────
    p_arc = np.asarray(d["arc_idx"], dtype=np.int64)
    p_lat = np.asarray(d["lat_deg"], dtype=np.float64)
    p_lst = np.asarray(d["lst_hours"], dtype=np.float64)
    p_alt = np.asarray(d["alt_km"], dtype=np.float64)

    pmask = (
        np.isfinite(p_lat) & np.isfinite(p_lst) & np.isfinite(p_alt)
        & (p_alt >= min_alt_km) & (p_alt <= max_alt_km)
    )
    pts = pd.DataFrame({
        "arc_idx": p_arc[pmask],
        "lat":     p_lat[pmask],
        "lst":     p_lst[pmask],
        "alt":     p_alt[pmask],
    })
    grp = pts.groupby("arc_idx", sort=True)
    pts_lat  = grp["lat"].apply(lambda s: s.to_numpy())
    pts_lst  = grp["lst"].apply(lambda s: s.to_numpy())
    pts_alt  = grp["alt"].apply(lambda s: s.to_numpy())
    alt_mean = grp["alt"].mean()
    lat_mean = grp["lat"].mean()
    lst_mean = grp["lst"].mean()
    surviving = pts_lat.index.to_numpy()    # arc_idx values with ≥1 valid point

    # ── Arc-level table (indexed by arc_idx) ─────────────────────────────────
    rho = np.asarray(d["rho"], dtype=np.float64)
    sigma = (_coerce_sigma(d["sigma_log_rho"], units=sigma_units)
             if "sigma_log_rho" in d.files
             else np.full(len(rho), np.nan))
    arc = pd.DataFrame({
        "norad_id":      np.asarray(d["norad_id"], dtype=np.int32),
        "epoch_mid":     _parse_epoch(np.asarray(d["epoch_mid"])),
        "rho_use":       rho,
        "sigma_log_rho": sigma,
    })

    # Keep arcs that have surviving points and a valid density.
    arc = arc.iloc[surviving].copy()
    arc["pts_lat"]  = pts_lat.to_numpy()
    arc["pts_lst"]  = pts_lst.to_numpy()
    arc["pts_alt"]  = pts_alt.to_numpy()
    arc["alt_km"]   = alt_mean.to_numpy()
    arc["lat_deg"]  = lat_mean.to_numpy()
    arc["lst_hours"] = lst_mean.to_numpy()
    arc["n_arc_pts"] = np.array([len(a) for a in arc["pts_lat"]], dtype=np.int32)

    valid = np.isfinite(arc["rho_use"]) & (arc["rho_use"] > 0)
    arc = arc[valid].reset_index(drop=True)
    arc["log10_rho"] = np.clip(np.log10(arc["rho_use"]), log10_clip_lo, log10_clip_hi)
    arc["cal_available"] = False
    arc = arc.sort_values("epoch_mid").reset_index(drop=True)

    n_arcs_total = len(rho)
    n_pts_kept = int(pmask.sum())
    print(
        f"[edr_loader] Loaded {len(arc)}/{n_arcs_total} arcs"
        f" | {n_pts_kept} arc points ({pmask.size} before alt filter)"
        f" | mean pts/arc: {arc['n_arc_pts'].mean():.1f}"
        f" | date range: {arc['epoch_mid'].min()} → {arc['epoch_mid'].max()}"
        f" | unique NORADs: {arc['norad_id'].nunique()}"
    )
    return arc
