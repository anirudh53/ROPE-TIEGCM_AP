"""
Config-driven selection of the reduced-order model (PCA or COAE).

The Kalman filter, the propagator and the results/plotting tooling are all identical
between the two architectures -- the only thing that differs is the decoder, i.e. how a
reduced state z becomes a density. This module picks that decoder from a YAML config so
a run can be switched with one file rather than an edited notebook.

Usage
-----
    import rom_config
    cfg = rom_config.load_config('../COAE_version/configs/coae_poly2.yaml')
    model_object = rom_config.load_model_object(cfg)
    meas_model = rom_config.build_measurement_model(cfg, model_object)

    kf = pu.kalman_filter_logspace_ar(..., **rom_config.kf_measurement_kwargs(meas_model))

`meas_model` exposes the same interface either way:

    h_and_H(z, lla)          -> (h (n,), H (n, n_components))   log10 density
    interpolate_density(z, lla) -> rho (n, 1)

Both PCA and COAE reduced states are the *raw* coefficients (the scale of the model's
z1_ds), not the regression-normalised ones.
"""

import os

import numpy as np
import scipy.io as sio
import yaml


def load_config(path):
    """Read a YAML config and resolve relative paths against the config's own directory."""
    path = os.path.abspath(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    base = os.path.dirname(path)
    for key in ('model_path', 'decoder_path', 'filter_noise_path'):
        if cfg.get(key):
            cfg[key] = os.path.normpath(os.path.join(base, cfg[key]))

    method = cfg.get('reduction_method')
    if method not in ('pca', 'coae'):
        raise ValueError(f"reduction_method must be 'pca' or 'coae', got {method!r}")
    cfg.setdefault('n_components', 10)
    cfg.setdefault('grid', {})
    cfg['_config_path'] = path
    return cfg


def load_model_object(cfg):
    """Load the .mat reduced-order model named by the config."""
    path = cfg['model_path']
    if not os.path.exists(path) and os.path.exists(path + '.mat'):
        path = path + '.mat'
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def build_measurement_model(cfg, model_object):
    """
    Construct the measurement operator for this config.

    pca  -> propagator_utilities_V4.RopeDataInterpolator over the stored U / mu
    coae -> coae_decoder.CAEMeasurementModel over the TorchScript decoder, with the
            stats_cae normalisation taken from the .mat (see coae_assets/README.md for
            why not from stats_cae.bin)
    """
    method = cfg['reduction_method']
    n_components = int(cfg['n_components'])
    grid = cfg.get('grid') or {}

    if method == 'pca':
        import propagator_utilities_V4 as pu

        U = np.asarray(model_object['U'])[:, :n_components]
        mu = np.asarray(model_object['mu']).reshape(-1)
        n_lt = int(grid.get('n_lt', 72))
        n_lat = int(grid.get('n_lat', 36))
        n_alt = int(grid.get('n_alt', 45))
        if U.shape[0] != n_lt * n_lat * n_alt:
            raise ValueError(f'U has {U.shape[0]} rows, expected '
                             f'{n_lt}*{n_lat}*{n_alt} = {n_lt * n_lat * n_alt}')
        U0 = U.reshape((n_lt, n_lat, n_alt, n_components), order='C')
        mu0 = mu.reshape((n_lt, n_lat, n_alt), order='C')
        return pu.RopeDataInterpolator(U0=U0, mu0=mu0, **_grid_bounds(grid))

    from coae_decoder import CAEMeasurementModel

    stats = model_object['stats_cae']
    return CAEMeasurementModel(
        cfg['decoder_path'], stats.mu, stats.sigma,
        n_components=n_components,
        device=cfg.get('device', 'cpu'),
        jacobian_mode=cfg.get('jacobian_mode', 'auto'),
        allow_tf32=bool(cfg.get('allow_tf32', True)),
        deterministic=bool(cfg.get('deterministic', False)),
        **_grid_bounds(grid),
    )


def _grid_bounds(grid):
    """Grid extent kwargs shared by both operators (defaults match RopeDataInterpolator)."""
    return {
        'lt_low': float(grid.get('lt_low', 0.0)),
        'lt_high': float(grid.get('lt_high', 23.6667)),
        'lat_low': float(grid.get('lat_low', -87.5)),
        'lat_high': float(grid.get('lat_high', 87.5)),
        'alt_low': float(grid.get('alt_low', 100.0)),
        'alt_high': float(grid.get('alt_high', 980.0)),
    }


def kf_measurement_kwargs(meas_model):
    """
    Route the operator to the right kalman_filter_logspace_ar keyword.

    Both keywords accept an object with h_and_H; this only keeps the call site readable
    and preserves the original argument names.
    """
    from propagator_utilities_V4 import RopeDataInterpolator

    if isinstance(meas_model, RopeDataInterpolator):
        return {'interpolator': meas_model, 'nonlinear_meas': None}
    return {'interpolator': None, 'nonlinear_meas': meas_model}


def load_filter_noise(cfg, n_components=None, q_scale=None):
    """
    Load P0 and Q derived by COAE_version/derive_filter_noise.py.

    Q is multiplied by the config's `q_scale` (default 1.0), so the process-noise setting
    travels with the config rather than living in each driver's command line. That matters
    because the two reduced-order models are tuned very differently -- PCA to NIS ~ 1 at
    q of order 100-1000, COAE deliberately at q = 1 -- and a multi-hour run picking up the
    wrong one silently produces a plausible-looking but untunable result. Pass `q_scale`
    explicitly to override the config, e.g. for a sweep.

    Falls back to the config's explicit `p0_scale` / `q_diag` if no .npz is named, so a
    config can still pin hand-chosen values. Returns (P0, Q).
    """
    n = int(n_components or cfg['n_components'])
    qs = float(cfg.get('q_scale', 1.0)) if q_scale is None else float(q_scale)
    path = cfg.get('filter_noise_path')
    if path and os.path.exists(path):
        d = np.load(path)
        return d['P0'][:n, :n], d['Q_step'][:n, :n] * qs

    p0_scale = float(cfg.get('p0_scale', 1e1))
    q_diag = cfg.get('q_diag')
    if q_diag is None:
        q_diag = [1e-2, 1e-2] + [1e-3] * (n - 2)
    return np.eye(n) * p0_scale, np.diag(np.asarray(q_diag, dtype=float)[:n]) * qs


def describe(cfg, model_object=None):
    """One-line summary for notebook output, so a run records which ROM it used."""
    parts = [f"reduction_method={cfg['reduction_method']}",
             f"n_components={cfg['n_components']}",
             f"model={os.path.basename(cfg['model_path'])}"]
    if cfg['reduction_method'] == 'coae':
        parts.append(f"decoder={os.path.basename(cfg['decoder_path'])}")
    if model_object is not None and 'model_params' in model_object:
        mp = model_object['model_params']
        parts.append(f"poly_order={getattr(mp, 'poly_order', '?')}")
    return '  '.join(parts)
