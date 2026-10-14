#!/usr/bin/env python3
"""
CLI entry point for the ROM-based EDR density assimilation.

Usage
-----
    conda run -n hasdm_env python run_edr_assimilation.py \\
        --config ../edr_assim/config/config.yaml

    # Override date range and output:
    conda run -n hasdm_env python run_edr_assimilation.py \\
        --config ../edr_assim/config/config.yaml \\
        --start-date "2003-10-28 00:00:00" \\
        --end-date   "2003-10-29 00:00:00" \\
        --output     ../results/test_run.h5 \\
        --n-components 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edr_assim.config.config import load_config
from edr_assim.loaders.edr_loader import load_edr_npz
from edr_assim.loaders.rom_loader import load_measurement_model, load_model_object
from edr_assim.core.edr_windowed_assimilation import run_windowed_assimilation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROM-based EDR density assimilation using information-filter KF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True,
                   help="Path to config.yaml")
    p.add_argument("--start-date", default=None,
                   help="Override start_date (format: 'YYYY-MM-DD HH:MM:SS')")
    p.add_argument("--end-date", default=None,
                   help="Override end_date")
    p.add_argument("--edr-npz", default=None,
                   help="Override edr.npz_path in config")
    p.add_argument("--output", default=None,
                   help="Override output.hdf5_path")
    p.add_argument("--n-components", type=int, default=None,
                   help="Override rom.n_components")
    p.add_argument("--mode", choices=["sindy", "static"], default=None,
                   help="Override filter.mode ('sindy' = SINDy-c dynamics, "
                        "'static' = identity predict)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config with CLI overrides ───────────────────────────────────────
    # Path overrides come from the shell, so resolve them against the CWD here.
    # Paths from the YAML are resolved against the config dir later via _abs();
    # making these absolute now means _abs() leaves them untouched (pathlib drops
    # the left operand when joining with an absolute path).
    overrides: dict = {}
    if args.start_date:
        overrides["start_date"] = args.start_date
    if args.end_date:
        overrides["end_date"] = args.end_date
    if args.edr_npz:
        overrides["edr_npz_path"] = str(Path(args.edr_npz).resolve())
    if args.output:
        out_abs = Path(args.output).resolve()
        overrides["hdf5_path"] = str(out_abs)
        overrides["checkpoint_json"] = str(out_abs.with_suffix(".checkpoint.json"))
    if args.n_components:
        overrides["n_components"] = args.n_components
    if args.mode:
        overrides["filter_mode"] = args.mode

    # Resolve config path relative to CWD
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path), overrides=overrides)

    # Resolve relative paths against the config file's directory
    cfg_dir = cfg_path.parent

    def _abs(p: str) -> str:
        return str((cfg_dir / p).resolve())

    rom_path = _abs(cfg.rom_mat_path)
    romcfg_path = _abs(cfg.rom_config_path) if cfg.rom_config_path else None
    edr_path = _abs(cfg.edr_npz_path)
    h5_path  = _abs(cfg.hdf5_path)
    ck_path  = _abs(cfg.checkpoint_json)

    # Patch resolved paths back into cfg
    cfg = load_config(str(cfg_path), overrides={
        **overrides,
        "rom_mat_path":   rom_path,
        "rom_config_path": romcfg_path,
        "edr_npz_path":   edr_path,
        "hdf5_path":      h5_path,
        "checkpoint_json": ck_path,
    })

    print("\n" + "="*70)
    print("EDR Assimilation Configuration")
    print("="*70)
    print(f"  ROM model   : {cfg.rom_mat_path}")
    print(f"  ROM config  : {cfg.rom_config_path or '(none -- affine PCA basis from the .mat)'}")
    print(f"  EDR input   : {cfg.edr_npz_path}")
    print(f"  Output HDF5 : {cfg.hdf5_path}")
    print(f"  Date range  : {cfg.start_date} → {cfg.end_date}")
    print(f"  Filter mode : {cfg.filter_mode}")
    print(f"  ROM modes   : {cfg.n_components}")
    print("="*70 + "\n")

    # ── Load ROM ─────────────────────────────────────────────────────────────
    # Dispatches on cfg.rom_config_path: unset -> the affine PCA basis read straight
    # out of the .mat (unchanged); set -> whatever rom_config's reduction_method says,
    # i.e. the COAE decoder.
    interpolator, k_actual, _is_nonlinear = load_measurement_model(cfg)
    # The dynamics model is only needed for SINDy mode; static mode uses an
    # identity predict, so leave model_object unset to select that path.
    model_object = (load_model_object(cfg.rom_mat_path)
                    if cfg.filter_mode != "static" else None)

    if k_actual != cfg.n_components:
        cfg = load_config(str(cfg_path), overrides={
            **overrides,
            "rom_mat_path":    rom_path,
            "rom_config_path": romcfg_path,
            "edr_npz_path":    edr_path,
            "hdf5_path":       h5_path,
            "checkpoint_json": ck_path,
            "n_components":    k_actual,
        })

    # ── Load EDR observations ─────────────────────────────────────────────────
    obs_df = load_edr_npz(
        npz_path=cfg.edr_npz_path,
        prefer_calibrated=cfg.prefer_calibrated,
        log10_clip_lo=cfg.log10_clip_lo,
        log10_clip_hi=cfg.log10_clip_hi,
        min_alt_km=cfg.min_alt_km,
        max_alt_km=cfg.max_alt_km,
        sigma_units=cfg.sigma_units,
    )

    # ── Clip end date to last observation + one window ────────────────────────
    # Prevents the filter from coasting through empty windows when the data
    # does not cover the full config date range.
    if len(obs_df) > 0:
        from datetime import datetime, timedelta, timezone as _tz
        last_obs = obs_df["epoch_mid"].max()
        if last_obs.tzinfo is None:
            last_obs = last_obs.replace(tzinfo=_tz.utc)
        data_end = last_obs + timedelta(hours=cfg.window_hours)
        config_end = datetime.strptime(cfg.end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz.utc)
        if data_end < config_end:
            effective_end = data_end.strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [note] Data ends {last_obs.strftime('%Y-%m-%d %H:%M')} UTC; "
                  f"clamping end_date to {effective_end}")
            cfg = load_config(str(cfg_path), overrides={
                **overrides,
                "rom_mat_path":    rom_path,
                "rom_config_path": romcfg_path,
                "edr_npz_path":    edr_path,
                "hdf5_path":       h5_path,
                "checkpoint_json": ck_path,
                "n_components":    k_actual,
                "end_date":        effective_end,
            })

    # ── Run assimilation ──────────────────────────────────────────────────────
    run_windowed_assimilation(
        obs_df=obs_df,
        interpolator=interpolator,
        cfg=cfg,
        model_object=model_object,
    )


if __name__ == "__main__":
    main()
