"""
EDRAssimConfig — configuration dataclass and YAML loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass(frozen=True)
class EDRAssimConfig:
    # ── ROM ──────────────────────────────────────────────────────────────────
    rom_mat_path: str
    n_components: int
    rom_lat_max: float

    # ── EDR input ────────────────────────────────────────────────────────────
    edr_npz_path: str
    prefer_calibrated: bool
    log10_clip_lo: float
    log10_clip_hi: float

    # ── Observation error ────────────────────────────────────────────────────
    use_npz_sigma: bool
    alt_bins_km: Tuple[float, ...]
    sigma_by_bin: Tuple[float, ...]
    sigma_floor: float
    sigma_cap: float

    # ── Altitude validity range (ROM grid bounds) ─────────────────────────────
    min_alt_km: float
    max_alt_km: float

    # ── Kalman filter ────────────────────────────────────────────────────────
    window_hours: float
    filter_mode: str           # "sindy" (SINDy-c dynamics) or "static" (identity predict)
    Q_diag: Tuple[float, ...]
    P0_diag: Tuple[float, ...]

    # ── Output ───────────────────────────────────────────────────────────────
    hdf5_path: str
    checkpoint_json: str

    # ── Runtime ──────────────────────────────────────────────────────────────
    start_date: str
    end_date: str

    # ── Sigma unit handling (default preserves the historical heuristic) ─────
    # Explicit unit of sigma_log_rho in the NPZ. "log10" / "ln" disable the
    # median-based heuristic that historically misclassified borderline files;
    # prefer one of those when the upstream pipeline knows its own units.
    sigma_units: str = "auto"

    # ── Reduced-order model selection (optional; default = the linear PCA path) ──
    # Path to a PyDensityEstimation rom_config YAML (COAE_version/configs/*.yaml).
    # When set, the measurement operator is built through rom_config, so the
    # reduction_method key in THAT file decides whether the observation operator is
    # the affine PCA basis or the nonlinear COAE decoder.  When unset (None) the
    # loader keeps the historical path that reads U / mu straight out of
    # rom_mat_path, so every PCA result stays bit-identical.
    #
    # rom_mat_path is still required either way: the SINDy-c dynamics, the
    # normalisation dicts and ic_array all come from the .mat, and only the decoder
    # changes between the two reduced-order models.
    rom_config_path: Optional[str] = None


def load_config(
    path: str,
    overrides: Optional[dict] = None,
) -> EDRAssimConfig:
    """Load config.yaml and return an EDRAssimConfig.

    overrides: flat dict of attribute-name → value to override after loading.
    """
    with open(path) as f:
        y = yaml.safe_load(f)

    cfg = EDRAssimConfig(
        # ROM
        rom_mat_path    = y["rom"]["mat_path"],
        n_components    = int(y["rom"]["n_components"]),
        rom_lat_max     = float(y["rom"]["lat_max"]),
        # EDR
        edr_npz_path        = y["edr"]["npz_path"],
        prefer_calibrated   = bool(y["edr"]["prefer_calibrated"]),
        log10_clip_lo       = float(y["edr"]["log10_clip_lo"]),
        log10_clip_hi       = float(y["edr"]["log10_clip_hi"]),
        # Obs error
        use_npz_sigma   = bool(y["obs_error"]["use_npz_sigma"]),
        alt_bins_km     = tuple(float(v) for v in y["obs_error"]["alt_bins_km"]),
        sigma_by_bin    = tuple(float(v) for v in y["obs_error"]["sigma_by_bin"]),
        sigma_floor     = float(y["obs_error"]["sigma_floor"]),
        sigma_cap       = float(y["obs_error"]["sigma_cap"]),
        # Altitude range
        min_alt_km      = float(y["arc"]["min_alt_km"]),
        max_alt_km      = float(y["arc"]["max_alt_km"]),
        # Filter
        window_hours    = float(y["filter"]["window_hours"]),
        filter_mode     = str(y["filter"]["mode"]),
        Q_diag          = tuple(float(v) for v in y["filter"]["Q_diag"]),
        P0_diag         = tuple(float(v) for v in y["filter"]["P0_diag"]),
        # Output
        hdf5_path           = str(y["output"]["hdf5_path"]),
        checkpoint_json     = str(y["output"]["checkpoint_json"]),
        # Runtime
        start_date  = str(y["runtime"]["start_date"]),
        end_date    = str(y["runtime"]["end_date"]),
        # Sigma units (optional; default "auto" preserves historical heuristic)
        sigma_units = str(y.get("edr", {}).get("sigma_units", "auto")),
        # ROM selection (optional)
        rom_config_path = (str(y["rom"]["config_path"])
                           if y.get("rom", {}).get("config_path") else None),
    )

    if overrides:
        # frozen dataclass → use object.__setattr__ trick via a copy
        d = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
        d.update(overrides)
        cfg = EDRAssimConfig(**d)

    return cfg
