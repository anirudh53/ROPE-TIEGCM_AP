"""
COAE (convolutional autoencoder) decoder as a nonlinear measurement operator.

The PCA architecture uses an affine decoder, log10 rho(x) = U(x) @ z + mu(x), so the
measurement operator H = d(log10 rho)/dz is the constant matrix U(x) and
`RopeDataInterpolator` can interpolate the *basis functions* once and reuse them for
every filter step.

The COAE decoder is a conv net, log10 rho = D(z), so H = d/dz interp(D(z)) depends on the
current state and must be recomputed at each measurement epoch. `CAEMeasurementModel`
below provides that, with an API deliberately close to `RopeDataInterpolator` so the
surrounding filter code changes as little as possible.

Requires torch.
"""

import os

import numpy as np
import torch

# Above this many simultaneous measurements, forward-mode beats reverse-mode.
# Benchmarked on the vendored decoder: n_meas=1 jacrev 137 ms vs jacfwd 533 ms;
# n_meas=3 343 vs 489; n_meas=8 855 vs 484. Crossover sits just under 5.
_JACREV_MAX_MEAS = 5

# Same defaults as RopeDataInterpolator (propagator_utilities_V4.py:1525-1528).
_LT_LOW, _LT_HIGH = 0.0, 23.6667
_LAT_LOW, _LAT_HIGH = -87.5, 87.5
_ALT_LOW, _ALT_HIGH = 100.0, 980.0


def trilinear_torch(field, points, grids):
    """
    Differentiable trilinear interpolation on a regular 3-D grid.

    Mirrors scipy's RegularGridInterpolator with bounds_error=False and fill_value=None,
    which *extrapolates* linearly outside the grid. That is reproduced here by clamping
    the cell index to the valid range while leaving the interpolation weight unclamped,
    so it runs outside [0, 1] beyond the edges.

    field:  (n0, n1, n2) tensor
    points: (n, 3) tensor, columns matching the order of `grids`
    grids:  3-tuple of 1-D coordinate tensors
    Returns: (n,) tensor
    """
    idx, wgt = [], []
    for axis, coord in zip(grids, points.T):
        i = torch.clamp(
            torch.searchsorted(axis.contiguous(), coord.contiguous()) - 1,
            0, axis.numel() - 2,
        )
        idx.append(i)
        wgt.append((coord - axis[i]) / (axis[i + 1] - axis[i]))

    (i0, i1, i2), (w0, w1, w2) = idx, wgt
    out = 0.0
    for a in (0, 1):
        fa = w0 if a else (1.0 - w0)
        for b in (0, 1):
            fb = w1 if b else (1.0 - w1)
            for c in (0, 1):
                fc = w2 if c else (1.0 - w2)
                out = out + fa * fb * fc * field[i0 + a, i1 + b, i2 + c]
    return out


def _trilinear_batch(fields, points, grids):
    """
    Trilinear interpolation where each sample has its own field and its own point.

    fields: (B, n0, n1, n2)
    points: (B, 3), columns matching the order of `grids`
    Returns: (B,)

    Same extrapolating convention as trilinear_torch.
    """
    b = torch.arange(fields.shape[0], device=fields.device)
    idx, wgt = [], []
    for axis, coord in zip(grids, points.T):
        i = torch.clamp(
            torch.searchsorted(axis.contiguous(), coord.contiguous()) - 1,
            0, axis.numel() - 2,
        )
        idx.append(i)
        wgt.append((coord - axis[i]) / (axis[i + 1] - axis[i]))

    (i0, i1, i2), (w0, w1, w2) = idx, wgt
    out = 0.0
    for a in (0, 1):
        fa = w0 if a else (1.0 - w0)
        for c1 in (0, 1):
            fb = w1 if c1 else (1.0 - w1)
            for c2 in (0, 1):
                fc = w2 if c2 else (1.0 - w2)
                out = out + fa * fb * fc * fields[b, i0 + a, i1 + c1, i2 + c2]
    return out


def resolve_device(device="auto"):
    """
    Pick the compute device, falling back rather than failing.

    "auto" prefers Metal (MPS) on Apple silicon, then CUDA, then CPU. Measured on an
    M4 Max against this decoder, MPS is worth roughly an order of magnitude -- the
    n_meas=1 `jacrev` the filter actually uses runs 251 ms on CPU against 18 ms on MPS,
    and the decoder forward 69 ms/state against 5.7 ms/state. The whole graph is
    supported: conv3d, searchsorted and torch.func.jacrev all run without a CPU
    fallback, and end-to-end density agrees with the CPU path to 7e-4 %.

    An explicit "mps"/"cuda" that is unavailable degrades to CPU with a warning, so a
    config committed on one machine still runs on another.
    """
    want = str(device or "auto").lower()
    have_mps = torch.backends.mps.is_available()
    have_cuda = torch.cuda.is_available()

    if want == "auto":
        return torch.device("mps" if have_mps else "cuda" if have_cuda else "cpu")
    if want.startswith("mps") and not have_mps:
        print("[coae_decoder] MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if want.startswith("cuda") and not have_cuda:
        print("[coae_decoder] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(want)


def configure_cuda_precision(allow_tf32=True, deterministic=False):
    """
    Set the CUDA backend up for this decoder. Defaults are tuned for speed.

    allow_tf32: TF32 in convs and matmuls. Torch enables it by default on sm_80+. It
        truncates the mantissa to 10 bits -- 1.2e-3 in decoded log10 density against
        3.8e-6 without it, i.e. 0.28% in density, well under the 5% sensor error the
        filter's R already models.
    deterministic: pin cuDNN to deterministic algorithms and enable
        torch.use_deterministic_algorithms. Off by default -- it costs speed, forbids
        cudnn.benchmark, and the reverse-mode Jacobian path this uses on CUDA has
        atomic accumulation in both the conv and the trilinear backward. Turn it on
        only for a run that has to be bitwise reproducible.

    cudnn.benchmark is enabled whenever deterministic is off: the decoder is called with
    a small set of fixed shapes, so autotuning pays for itself immediately.
    """
    if not torch.cuda.is_available():
        return None
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[coae_decoder] strict determinism unavailable ({exc}); "
                  f"falling back to warn_only. Repeat runs may not be bitwise.")
            torch.use_deterministic_algorithms(True, warn_only=True)
    return bool(allow_tf32)


def default_batch_size(device, cpu_default=8, gpu_default=32):
    """
    Decoder batch size for the device.

    The CPU limit of 8 is a *memory* constraint, not a speed one -- the decoder's
    intermediate activations are (batch, channels, 72, 36, 45) and a batch of 32 drove
    resident memory to ~20 GB on a 120k-point track. On MPS the same activations live in
    unified memory and are freed per batch, and per-state cost is flat from 8 to 64
    (5.7 -> 5.4 ms), so a larger batch is safe and marginally faster.
    """
    return gpu_default if torch.device(device).type in ("mps", "cuda") else cpu_default


def load_coae_decoder(pt_path, device="cpu"):
    """Load the TorchScript COAE decoder. Signature: (B, 10) -> (B, 1, 72, 36, 45)."""
    dec = torch.jit.load(str(pt_path), map_location=device)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


class CAEMeasurementModel:
    """
    Nonlinear measurement operator backed by the COAE decoder.

    Drop-in counterpart to `propagator_utilities_V4.RopeDataInterpolator`: it exposes the
    same `interpolate_density(z, lla)` and the same [lat, local_time, alt] column
    convention for `lla`. It cannot expose `get_H(lla)`, because H depends on z — use
    `h_and_H(z, lla)` instead.
    """

    def __init__(self, pt_path, stats_cae_mu, stats_cae_sigma,
                 n_components=10, device="cpu", jacobian_mode="auto",
                 allow_tf32=True, deterministic=False,
                 lt_low=_LT_LOW, lt_high=_LT_HIGH,
                 lat_low=_LAT_LOW, lat_high=_LAT_HIGH,
                 alt_low=_ALT_LOW, alt_high=_ALT_HIGH):
        """
        stats_cae_mu, stats_cae_sigma: (72, 36, 45) arrays from the .mat `stats_cae`
            struct. The decoder emits normalised values; these put it back in log10
            density. Do not source them from stats_cae.bin -- see coae_assets/README.md.
        """
        self.device = resolve_device(device)
        # Must run before the first conv: TF32 would otherwise cost three digits of
        # agreement on the density field, and this operator feeds an EKF.
        if self.device.type == "cuda":
            configure_cuda_precision(allow_tf32, deterministic)
        self.decoder = load_coae_decoder(pt_path, self.device)
        self.n_components = n_components
        self.batch_size = default_batch_size(self.device)

        # Reverse-mode autodiff is NOT reproducible on MPS: its backward pass accumulates
        # with GPU atomics whose ordering varies between otherwise identical calls. Ten
        # jacrev calls on one input drift by 1.3e-6; ten jacfwd calls are bitwise equal.
        # That matters far more than it sounds, because the EKF amplifies it: a 1.7e-9
        # perturbation at step 0 grew to 5.2e-1 -- larger than |z| itself -- within 100
        # steps, and two identical runs finished 0.9 MAPE points apart. So on GPU default
        # to forward mode and buy reproducibility for ~2.6x on the Jacobian; on CPU keep
        # reverse mode, which is both deterministic and faster at n_meas=1.
        # CUDA cannot use either functorch mode against this TorchScript decoder --
        # both go through vmap, which the scripted interpreter rejects -- so it gets
        # plain autograd instead. That is reverse mode, so cuDNN's nondeterministic
        # conv backward applies; configure_cuda_precision also pins
        # cudnn.deterministic to keep repeat runs bitwise.
        #   'auto' -> autograd on cuda, jacfwd on mps, per-call crossover on cpu
        #   'jacrev' / 'jacfwd' / 'autograd' -> force one
        if jacobian_mode not in ("auto", "jacrev", "jacfwd", "autograd"):
            raise ValueError(f"jacobian_mode must be auto/jacrev/jacfwd/autograd, got "
                             f"{jacobian_mode!r}")
        self.jacobian_mode = jacobian_mode

        self.mu = torch.as_tensor(np.asarray(stats_cae_mu), dtype=torch.float32,
                                  device=self.device)
        self.sigma = torch.as_tensor(np.asarray(stats_cae_sigma), dtype=torch.float32,
                                     device=self.device)
        if self.mu.shape != self.sigma.shape:
            raise ValueError(f"stats_cae mu {tuple(self.mu.shape)} and sigma "
                             f"{tuple(self.sigma.shape)} must match")

        n_lt, n_lat, n_alt = self.mu.shape
        self.grids = (
            torch.linspace(lt_low, lt_high, n_lt, device=self.device),
            torch.linspace(lat_low, lat_high, n_lat, device=self.device),
            torch.linspace(alt_low, alt_high, n_alt, device=self.device),
        )

    # Grid accessors matching RopeDataInterpolator's attribute surface.  Both operators
    # are defined on the same 72 x 36 x 45 grid and rom_config already treats them as
    # interchangeable, so anything that needs the axes -- the local-time wrap seam in the
    # EDR arc operator, for one -- should not have to know which decoder it holds.
    # numpy, not torch, because every consumer of these is numpy-side.
    @property
    def lt_grid(self):
        return self.grids[0].detach().cpu().numpy().astype(np.float64)

    @property
    def lat_grid(self):
        return self.grids[1].detach().cpu().numpy().astype(np.float64)

    @property
    def alt_grid(self):
        return self.grids[2].detach().cpu().numpy().astype(np.float64)

    # ---------------------------------------------------------------- internals

    def _decode_one(self, z):
        """(n_components,) tensor -> (n_lt, n_lat, n_alt) log10 density."""
        return self.decoder(z.reshape(1, -1)).squeeze(0).squeeze(0) * self.sigma + self.mu

    def _points(self, lla):
        """lla columns [lat, local_time, alt] -> grid order (lt, lat, alt)."""
        lla = np.atleast_2d(np.asarray(lla, dtype=np.float64))
        pts = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])
        return torch.as_tensor(pts, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------- public

    def h_and_H(self, z, lla):
        """
        Measurement prediction and its Jacobian, both in log10 density space.

        z:   (n_components,) or (n_components, 1)
        lla: (n_samples, 3), columns [latitude, local_time, altitude]

        Returns (h, H) with h of shape (n_samples,) and H of shape
        (n_samples, n_components).

        All rows of `lla` are evaluated in a single call -- they share `z`, so batching
        them costs one decoder pass instead of one per row.
        """
        pts = self._points(lla)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float64).reshape(-1),
                              dtype=torch.float32, device=self.device)

        def f(q):
            return trilinear_torch(self._decode_one(q), pts, self.grids)

        mode = self.jacobian_mode
        if mode == "auto":
            # CUDA cannot use functorch here at all: jacrev and jacfwd both build the
            # Jacobian with vmap, and vmap through the TorchScript interpreter raises
            # "Cannot access data pointer of Tensor that doesn't have storage" -- the
            # scripted conv path wants a real storage pointer and a BatchedTensor has
            # none. MPS happens to take a path that tolerates it. So on CUDA use plain
            # autograd, which never batches the forward pass.
            if self.device.type == "cuda":
                mode = "autograd"
            elif self.device.type == "mps":
                mode = "jacfwd"                          # reproducibility, see __init__
            else:
                mode = ("jacrev" if pts.shape[0] < _JACREV_MAX_MEAS else "jacfwd")

        if mode == "autograd":
            # One ordinary backward per measurement row. The filter calls this with a
            # single row, so that is one backward for the whole Jacobian -- no vmap, no
            # functorch, and nothing that needs the decoder to be an eager module.
            #
            # MEASURED DEAD END, 2026-09-11: collapsing this loop into one call with
            #     H, = torch.autograd.grad(h_t, z_g, grad_outputs=torch.eye(h_t.numel()),
            #                              is_grads_batched=True)
            # is numerically correct here -- it agrees with the loop to 2e-7 relative on
            # both cpu and mps, unlike jacrev/jacfwd it does NOT trip the TorchScript
            # vmap failure, because only the backward graph is batched and the scripted
            # forward still runs once un-batched. But it buys nothing: timed at 1/2/3/4/7
            # rows it is 0.97-0.99x, and cost still grows linearly in rows on both
            # devices (mps 9.2/12.5/15.7/19.0/28.6 ms). vmap has no batching rule for the
            # conv-transpose backward, so it decomposes into the same sequential loop.
            # Batching rules are dispatcher-level, not per-device, so CUDA will do the
            # same. Do not re-try this.
            #
            # Nor is there a sparse shortcut. The obvious idea -- evaluate only the output
            # cells the trilinear sample touches instead of all 72*36*45 -- is ruled out by
            # the architecture: the decoder contains four squeeze-excitation blocks
            # (adaptive_avg_pool3d -> linear -> sigmoid -> mul), and that pooling is over
            # the whole spatial extent, so every output cell depends on every spatial
            # position of the intermediate feature maps. There is no local cone to
            # evaluate. The full-field decode is irreducible for this decoder.
            z_g = z_t.detach().requires_grad_(True)
            h_t = f(z_g)
            rows = []
            for i in range(h_t.numel()):
                g, = torch.autograd.grad(h_t[i], z_g,
                                         retain_graph=(i + 1 < h_t.numel()))
                rows.append(g)
            H = torch.stack(rows, dim=0)
            h = h_t.detach()
        else:
            jac = torch.func.jacrev if mode == "jacrev" else torch.func.jacfwd
            with torch.no_grad():
                h = f(z_t)
            H = jac(f)(z_t)

        return (h.detach().cpu().numpy().astype(np.float64),
                H.detach().cpu().numpy().astype(np.float64))

    def h_only(self, z, lla):
        """log10 density at `lla` for a single state, without the Jacobian."""
        pts = self._points(lla)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float64).reshape(-1),
                              dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h = trilinear_torch(self._decode_one(z_t), pts, self.grids)
        return h.cpu().numpy().astype(np.float64)

    def h_batch(self, z_seq, lla_seq, batch_size=None):
        """
        log10 density for a *sequence* of (state, location) pairs -- one location per
        state, each decoded from its own z.

        z_seq:   (n_components, T)
        lla_seq: (T, 3), columns [latitude, local_time, altitude]
        Returns: (T,)

        Post-processing along a satellite track needs one evaluation per measurement
        (tens of thousands over a two-week window). Calling h_and_H that many times would
        compute an unwanted Jacobian each time; decoding in mini-batches instead is
        roughly an order of magnitude faster.

        `batch_size` is deliberately small. The decoder's intermediate activations, not
        its (B, 72, 36, 45) output, dominate memory: a batch of 32 drove resident memory
        to ~20 GB on a 120k-point track. 8 keeps the peak to a few GB at no meaningful
        cost in speed.
        """
        batch_size = batch_size or self.batch_size
        z_seq = np.atleast_2d(np.asarray(z_seq, dtype=np.float32))
        pts = self._points(lla_seq)
        T = z_seq.shape[1]
        if pts.shape[0] != T:
            raise ValueError(f'{T} states but {pts.shape[0]} locations')

        out = np.empty(T, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, T, batch_size):
                zb = torch.as_tensor(z_seq[:, i:i + batch_size].T, dtype=torch.float32,
                                     device=self.device)
                fields = self.decoder(zb).squeeze(1) * self.sigma + self.mu
                out[i:i + batch_size] = _trilinear_batch(
                    fields, pts[i:i + batch_size], self.grids).cpu().numpy()
        return out

    def h_batch_grouped(self, z_unique, lla_seq, group, batch_size=None):
        """
        log10 density where many measurements share the same filter state.

        z_unique: (n_components, G) -- one state per distinct filter step
        lla_seq:  (T, 3) measurement locations
        group:    (T,) index into z_unique for each measurement, non-decreasing
        Returns:  (T,)

        Along a satellite track the measurement cadence is finer than the filter step
        (10 s against 60 s for GRACE-FO), so runs of consecutive measurements share one
        state. Decoding once per distinct state and interpolating every measurement in
        that run against the same field is ~6x less decoder work than h_batch, which
        decodes once per measurement.
        """
        batch_size = batch_size or self.batch_size
        z_unique = np.atleast_2d(np.asarray(z_unique, dtype=np.float32))
        group = np.asarray(group, dtype=np.int64)
        pts = self._points(lla_seq)
        G = z_unique.shape[1]
        if group.size != pts.shape[0]:
            raise ValueError(f'{group.size} group ids but {pts.shape[0]} locations')

        # Run boundaries, so a batch of states maps to a contiguous slice of measurements.
        starts = np.searchsorted(group, np.arange(G), side='left')
        stops = np.searchsorted(group, np.arange(G), side='right')

        out = np.empty(pts.shape[0], dtype=np.float64)
        with torch.no_grad():
            for g0 in range(0, G, batch_size):
                g1 = min(g0 + batch_size, G)
                lo, hi = starts[g0], stops[g1 - 1]
                if hi <= lo:
                    continue
                zb = torch.as_tensor(z_unique[:, g0:g1].T, dtype=torch.float32,
                                     device=self.device)
                fields = self.decoder(zb).squeeze(1) * self.sigma + self.mu
                local = torch.as_tensor(group[lo:hi] - g0, dtype=torch.long,
                                        device=self.device)
                out[lo:hi] = _trilinear_batch(
                    fields[local], pts[lo:hi], self.grids).cpu().numpy()
        return out

    def interpolate_density(self, z, lla):
        """
        Physical density at spacecraft locations. Matches
        RopeDataInterpolator.interpolate_density, including the (n_samples, 1) shape.
        """
        return (10.0 ** self.h_only(z, lla)).reshape(-1, 1)

    def decode_full_field(self, z, batch_size=None, log_space=False):
        """
        Grid-wide reconstruction, for reconstruction-error metrics and field plots.

        z: (n_components, T)
        Returns (T, n_lt, n_lat, n_alt) -- log10 density if log_space, else density.

        Same computation as sindy_atr_utilities.decode_latent_to_density's 'cae' branch,
        but with a much smaller default batch. That function uses 256, which is not
        survivable on CPU here: the decoder's intermediate feature maps are sized
        (batch, channels, 72, 36, 45), so a batch of 256 runs to tens of GB and gets the
        process OOM-killed. 32 keeps the peak modest at no meaningful speed cost.
        """
        batch_size = batch_size or max(self.batch_size, 32)
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        out = []
        with torch.no_grad():
            for i in range(0, z.shape[1], batch_size):
                batch = torch.as_tensor(z[:, i:i + batch_size].T, dtype=torch.float32,
                                        device=self.device)
                dec = self.decoder(batch).squeeze(1) * self.sigma + self.mu
                out.append(dec.cpu().numpy().astype(np.float64))
        field = np.concatenate(out, axis=0)
        return field if log_space else 10.0 ** field


def check_jacobian(model, z, lla, eps=3e-4):
    """
    Central-difference validation of h_and_H's analytic Jacobian.

    Returns (H_analytic, H_numeric, max_rel_err). Used as a correctness gate before
    trusting filter output.

    Do not expect agreement much better than ~1e-2 here. The decoder is float32, so the
    central difference is roundoff-limited: measured max relative error runs 8.3e-1 at
    eps=1e-5, 1.5e-2 at eps=3e-4 (the optimum, hence the default), and 7.6e-1 at
    eps=1e-2. Promoting the arithmetic to float64 moves the floor to 1.4e-4 at eps=1e-5
    but no further, because the exported weights are themselves float32. The U-shaped
    dependence on eps is the signature of a *correct* Jacobian against a noisy finite
    difference; a genuinely wrong Jacobian shows an error floor that does not move with
    eps. Filter accuracy is not sensitive at this level -- measurement noise dominates.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    _, H = model.h_and_H(z, lla)
    H_num = np.zeros_like(H)
    for i in range(z.size):
        dz = np.zeros_like(z)
        dz[i] = eps
        hp, _ = model.h_and_H(z + dz, lla)
        hm, _ = model.h_and_H(z - dz, lla)
        H_num[:, i] = (hp - hm) / (2 * eps)
    denom = np.maximum(np.abs(H_num), 1e-12)
    return H, H_num, float(np.max(np.abs(H - H_num) / denom))
