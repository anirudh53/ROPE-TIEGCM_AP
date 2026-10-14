"""
rope_da.py -- data-assimilation initial conditions for ROPE (TIE-GCM f10/f10avg/ap version).

Used by ROPE(use_da=True). initial_state() also returns the filter covariance carried into
ROPE's latent space, which ROPE uses for the IC part of its input uncertainty.

Replaces ROPE's (F10, Kp) IC-table lookup with latent states estimated by the EDR
assimilation filter (vendored under ./da). Use:

    from rope import ROPE
    import rope_da
    rope = ROPE(device="cuda")
    rope.da = rope_da.DAIC(rope)

    rope.run("2024-06-15 03:00", horizon=72, uncertainty=False)               # table IC (stock)
    rope.run("2024-06-15 03:00", horizon=72, uncertainty=False, ic="da")      # assimilated IC
    rope.da.initial_state("2024-06-15 03:00")                                 # z and Sigma z, ROPE units

Why a translation step exists
-----------------------------
The filter's COAE decoder has the same weights as ROPE's, but its stats_cae (the per-cell
mean/std that turn raw decoder output into log10 density) differ, so the same z decodes to
different fields. DAIC finds, for each assimilated state, the ROPE-space z whose ROPE-decoded
field matches the field the filter believes. Typical fit error is ~0.03 in log10 (about 7%
in density); see fit rms printed on every new fit.

Covariance in ROPE space
------------------------
Filter field  f_F(z)  = D(z) * sigma_F + mu_F   (log10, filter stats_cae)
ROPE field    f_R(z') = D(z') * sigma_R + mu_R  (log10, ROPE stats_cae), same decoder D.
The fit z' = argmin ||f_R(z') - f_F(z)||^2 has derivative G = pinv(J_R(z')) J_F(z), with J the
decoder Jacobians at the two points, so Sigma_R = G Sigma_F G^T (per hour, G block-diagonal over
the joint 3-hour covariance). G is deterministic, so optimizer noise in the fit does not leak in.

The assimilated state is a smoothed estimate: for hour T it uses the first 3-hourly filter
update at or after T, so keep T on 00/03/06/... UTC for a clean, update-time state.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

DA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "da")
_CODE = os.path.join(DA_DIR, "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

DEFAULT_H5 = os.path.join(DA_DIR, "out", "coae_20240301_20240731.h5")


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _key(T, n):
    return f"{pd.Timestamp(T):%Y%m%dT%H}_n{n}"


class DAIC:
    """Assimilated initial latents in ROPE's latent space, for any hour inside the filter run."""

    def __init__(self, rope, h5_path=DEFAULT_H5, cache_path=None, iters=400, lr=2e-3,
                 chunk=4, seed_from=None, verbose=True):
        from query_state import get_state, run_span          # vendored, imported lazily
        from coae_decoder import CAEMeasurementModel

        self._get_state = get_state
        self.rope, self.h5 = rope, h5_path
        self.iters, self.lr, self.chunk, self.verbose = iters, lr, chunk, verbose
        self.cache_path = cache_path or os.path.join(DA_DIR, "out", "ic_rope_space_cache.npz")

        st = sio.loadmat(os.path.join(DA_DIR, "data", "coae_model.mat"), squeeze_me=True,
                         struct_as_record=False, variable_names=["stats_cae"])["stats_cae"]
        self._bundle = CAEMeasurementModel(os.path.join(DA_DIR, "data", "coae_decoder.pt"),
                                           st.mu, st.sigma, device="cpu")

        cae = rope.cae
        self._dev = next(cae.decoder.parameters()).device
        self._mu = torch.tensor(_np(rope.stats_cae["mu"]), dtype=torch.float32, device=self._dev)
        self._sg = torch.tensor(_np(rope.stats_cae["sigma"]), dtype=torch.float32, device=self._dev)
        for p in cae.decoder.parameters():
            p.requires_grad_(False)

        self._G = {}                                             # per-key (n, K, K) transport
        self._mu_f = self._bundle.mu.to(self._dev)
        self._sg_f = self._bundle.sigma.to(self._dev)
        self._cache = {}
        if os.path.exists(self.cache_path):
            d = np.load(self.cache_path)
            self._cache = {k: d[k] for k in d.files}
        if seed_from and os.path.exists(seed_from):            # npz with 'starts' and 'z' arrays
            d = np.load(seed_from)
            for t, z in zip(pd.to_datetime(d["starts"]), d["z"]):
                self._cache.setdefault(_key(t, 3), np.asarray(z, np.float32))
        self.span = run_span(self.h5)

    # ------------------------------------------------------------------ public
    def coverage(self, n=3):
        """(first, last) hour T for which an n-hour IC can be returned."""
        return self.span[0] + pd.Timedelta(hours=n - 1), self.span[1]

    def prefit(self, times, n=3):
        """Fit and cache ICs for many start times at once (batched; much faster than one by one)."""
        todo = [pd.Timestamp(t) for t in times if _key(t, n) not in self._cache]
        for i in range(0, len(todo), self.chunk):
            part = todo[i:i + self.chunk]
            zb = np.stack([self._get_state(self.h5, f"{t:%Y-%m-%d %H:%M}", n_hours=n)["z"]
                           for t in part])                      # (m, n, 10) filter latent space
            z, rms = self._fit(zb)
            for t, zz in zip(part, z):
                self._cache[_key(t, n)] = zz.astype(np.float32)
            if self.verbose:
                print(f"[rope_da] fitted {part[0]:%Y-%m-%d %H:%M} .. {part[-1]:%Y-%m-%d %H:%M}"
                      f"   fit rms log10 = {rms.mean():.3f}")
            if self._dev.type == "cuda":
                torch.cuda.empty_cache()
        if todo:
            self._save()

    def initial_latents(self, T, n=3):
        """(n, 10) latents at T-(n-1)h .. T in ROPE's physical latent units, ready for ROPE's IC slot."""
        if not 1 <= n <= 4:
            raise ValueError("the filter keeps at most 4 usable hourly states per update; need 1 <= n <= 4")
        self.prefit([T], n)
        return self._cache[_key(T, n)].copy()

    def initial_state(self, T, n=3):
        """
        Assimilated IC with covariance, in ROPE latent units.

        Returns dict:
          times        : the n hours, oldest first
          z            : (n, K)    ROPE-space latents (same as initial_latents)
          Sigma_joint  : (nK, nK)  joint covariance of the n hours, ROPE space
          Sigma        : (n, K, K) per-hour blocks
          z_filter, Sigma_joint_filter : the filter's own values (filter latent space)
          G            : (n, K, K) filter -> ROPE linear transport per hour
          n_obs_span   : EDR arcs that informed these hours
        """
        z = self.initial_latents(T, n)
        s = self._get_state(self.h5, f"{pd.Timestamp(T):%Y-%m-%d %H:%M}", n_hours=n)
        key = _key(T, n)
        if key not in self._G:
            self._G[key] = np.stack([self._transport(zf, zr) for zf, zr in zip(s["z"], z)])
        G = self._G[key]
        k = z.shape[1]
        Gb = np.zeros((n * k, n * k))
        for i in range(n):
            Gb[i * k:(i + 1) * k, i * k:(i + 1) * k] = G[i]
        Sj = Gb @ s["Sigma_joint"] @ Gb.T
        Sj = 0.5 * (Sj + Sj.T)
        return {
            "times": s["times"], "z": z, "Sigma_joint": Sj,
            "Sigma": np.stack([Sj[i * k:(i + 1) * k, i * k:(i + 1) * k] for i in range(n)]),
            "z_filter": s["z"], "Sigma_joint_filter": s["Sigma_joint"], "G": G,
            "n_obs_span": s["n_obs_span"],
        }

    # ----------------------------------------------------------------- internals
    def _jac(self, z, sg):
        """d(D(z) * sg) / dz, flattened to (n_cells, K), float64; TF32 off for accuracy."""
        dec = self.rope.cae.decoder
        zt = torch.tensor(np.asarray(z, np.float32)[None], device=self._dev)
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            J = torch.autograd.functional.jacobian(
                lambda q: (dec(q).squeeze(1) * sg).reshape(-1), zt,
                vectorize=True, strategy="forward-mode")             # 10 inputs: 10 JVPs
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        return J.reshape(J.shape[0], -1).double().cpu().numpy()

    def _transport(self, z_filter, z_rope):
        """(K, K) G with dz_rope = G dz_filter (least squares in log10 density, like the fit)."""
        Jf = self._jac(z_filter, self._sg_f)
        Jr = self._jac(z_rope, self._sg)
        return np.linalg.lstsq(Jr, Jf, rcond=None)[0]

    def _log10(self, z):
        return self._bundle.decode_full_field(np.asarray(z, dtype=np.float32)[:, None],
                                              log_space=True)[0]

    def _fit(self, zb):
        m, h, k = zb.shape
        flat = zb.reshape(-1, k)
        tgt = torch.tensor(np.stack([self._log10(z) for z in flat]),
                           dtype=torch.float32, device=self._dev).unsqueeze(1)
        zr = torch.tensor(flat, dtype=torch.float32, device=self._dev, requires_grad=True)
        opt = torch.optim.Adam([zr], lr=self.lr)
        dec = self.rope.cae.decoder
        for _ in range(self.iters):
            opt.zero_grad()
            pred = (dec(zr).squeeze(1) * self._sg + self._mu).unsqueeze(1)
            mse = ((pred - tgt) ** 2).mean(dim=(1, 2, 3, 4))     # per row: chunk size doesn't change the step
            mse.sum().backward()
            opt.step()
        return (zr.detach().cpu().numpy().reshape(m, h, k),
                mse.detach().sqrt().cpu().numpy().reshape(m, h))

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(self.cache_path, **self._cache)


def run_filter(start, end, edr_npz=None, out=None, device="cpu", wait=True):
    """
    Run the COAE EDR assimilation filter over [start, end] using the vendored code.

    Resumes automatically if `out` and its .checkpoint.json already exist (delete both to start
    fresh). Arcs must cover the range; the filter starts one day before `start` to settle, so pass
    start = first day wanted minus one day, as the bundle does (e.g. 2024-02-29 for 2024-03-01).
    """
    cfgdir, outdir = os.path.join(DA_DIR, "configs"), os.path.join(DA_DIR, "out")
    edr_npz = edr_npz or os.path.join(DA_DIR, "data", "edr", "edr_20240301_20240731.npz")
    out = out or DEFAULT_H5
    os.makedirs(outdir, exist_ok=True)

    rom = open(os.path.join(cfgdir, "coae_poly2_th001_cuda.yaml")).read()
    rom = re.sub(r"(?m)^device: .*$", f"device: {device}", rom)
    cfg = open(os.path.join(cfgdir, "config_coae.yaml")).read()
    cfg2 = cfg.replace('config_path: "coae_poly2_th001_cuda.yaml"',
                       f'config_path: "../out/rom_coae_{device}.yaml"')
    if cfg2 == cfg:
        raise RuntimeError("config_coae.yaml no longer has the expected config_path line")
    open(os.path.join(outdir, f"rom_coae_{device}.yaml"), "w").write(rom)
    cfg_path = os.path.join(outdir, f"config_coae_{device}.yaml")
    open(cfg_path, "w").write(cfg2)

    fmt = lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    cmd = [sys.executable, os.path.join(_CODE, "run_edr_assimilation.py"), "--config", cfg_path,
           "--start-date", fmt(start), "--end-date", fmt(end),
           "--edr-npz", os.path.abspath(edr_npz), "--output", os.path.abspath(out)]
    print(" ".join(cmd))
    if wait:
        subprocess.run(cmd, cwd=DA_DIR, check=True)
        return None
    return subprocess.Popen(cmd, cwd=DA_DIR)
