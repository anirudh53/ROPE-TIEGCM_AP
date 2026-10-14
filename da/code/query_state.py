#!/usr/bin/env python3
"""
Get the hourly model state z and its covariance (Sigma z) from a finished run.

You give an hour T. The script gives z and Sigma z at T-2 h, T-1 h and T. It also
gives the joint covariance of the three hours together.

How the script gets hourly values
---------------------------------
The filter does one update every 3 hours. Its state keeps the last 6 hourly values
of z (T, T-1 h, ... T-5 h), and each update corrects all 6 values. The run file
stores the result of each update (windows/<start>/X_aug_post and P_aug_post).

For hour T, the script uses the first update at or after T. That update is at most
2 hours after T. So each value is a smoothed estimate: it uses all EDR arcs until
that update. All hours come from the same update, so the joint covariance is
correct. At an update time, the newest hour is the same as the stored z_post and
P_post. The script makes sure of this each time.

Command line
------------
    python code/query_state.py --time "2024-06-15 03:00"               # COAE
    python code/query_state.py --time "2024-06-15 03:00" --rom both
    python code/query_state.py --time "2024-06-15 03:00" --out s.json  # or .npz

Without --h5, the script finds the run file in out/ from the date.

Python
------
    from query_state import get_state
    s = get_state("out/coae.h5", "2024-05-11 03:00")
    s["times"]        # the hours, oldest first
    s["z"]            # (3, 10)       z at each hour
    s["Sigma"]        # (3, 10, 10)   Sigma z at each hour
    s["Sigma_joint"]  # (30, 30)      joint Sigma z, in blocks of one hour
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT = _HERE.parent / "out"
_FMT = "%Y-%m-%dT%H:%M:%S"
_HOUR = timedelta(hours=1)


def _parse_time(t) -> datetime:
    if isinstance(t, datetime):
        dt = t.replace(tzinfo=None)
    else:
        s = str(t).strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"cannot read the time {t!r}. Use the format 'YYYY-MM-DD HH:MM'")
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError(f"{dt:%Y-%m-%d %H:%M:%S} is not a full hour. "
                         "The model state is hourly. Use a full hour, for example 03:00")
    return dt


def _norm_constants(f: h5py.File, rom_mat: str | None):
    """(y_mean, y_std) of the ROM's latent normalization."""
    if "y_mean" in f.attrs and "y_std" in f.attrs:
        return (np.asarray(f.attrs["y_mean"], dtype=np.float64),
                np.asarray(f.attrs["y_std"], dtype=np.float64))
    # Runs written before these attrs existed: recover them from the ROM .mat.
    if rom_mat is None:
        raise ValueError(f"{f.filename} predates the y_mean/y_std attributes; "
                         "pass --rom-mat with the model it was run with")
    import scipy.io as sio
    mo = sio.loadmat(rom_mat, squeeze_me=True, struct_as_record=False,
                     variable_names=["Y_sindy_norm_dict"])
    k = int(f.attrs["n_components"])
    nd = mo["Y_sindy_norm_dict"]
    return (np.asarray(nd.x_mean, dtype=np.float64).ravel()[:k],
            np.asarray(nd.x_std, dtype=np.float64).ravel()[:k])


def run_span(h5_path) -> tuple[datetime, datetime]:
    """(first window start, last window end) actually present in a run file."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["windows"].keys()) if "windows" in f else []
        if not keys:
            raise ValueError(f"{h5_path} contains no windows")
        return (datetime.strptime(f[f"windows/{keys[0]}"].attrs["window_start"], _FMT),
                datetime.strptime(f[f"windows/{keys[-1]}"].attrs["window_end"], _FMT))


def get_state(h5_path, t, n_hours: int = 3, rom_mat: str | None = None) -> dict:
    """Smoothed z and covariance at hours t-(n_hours-1) .. t from a finished run.

    Raises ValueError rather than returning anything approximate: off-hour times,
    hours before the run's start or after its last window, and lags deeper than the
    filter keeps are all errors.
    """
    T = _parse_time(t)
    if not 1 <= n_hours <= 4:
        # 4 is the most that every hour allows: T can be 2 h before an update, and
        # the filter keeps 5 h of lags behind the update.
        raise ValueError("the number of hours must be 1 to 4")
    times = [T - (n_hours - 1 - j) * _HOUR for j in range(n_hours)]   # oldest first

    with h5py.File(h5_path, "r") as f:
        if "windows" not in f or not len(f["windows"]):
            raise ValueError(f"{h5_path} contains no windows")
        k = int(f.attrs["n_components"])
        y_mean, y_std = _norm_constants(f, rom_mat)

        keys = sorted(f["windows"].keys())
        spans = [(datetime.strptime(f[f"windows/{w}"].attrs["window_start"], _FMT),
                  datetime.strptime(f[f"windows/{w}"].attrs["window_end"], _FMT))
                 for w in keys]
        run_start, run_end = spans[0][0], spans[-1][1]
        # Before run_start the lag blocks hold copies of the seeded initial state, not
        # estimates, so they are refused rather than returned.
        if times[0] < run_start or T > run_end:
            raise ValueError(
                f"{times[0]:%Y-%m-%d %H:%M} .. {T:%Y-%m-%d %H:%M} is not in this run "
                f"({run_start:%Y-%m-%d %H:%M} .. {run_end:%Y-%m-%d %H:%M}, {h5_path})")

        wi = next(i for i, (_, we) in enumerate(spans) if we >= T)
        g = f[f"windows/{keys[wi]}"]
        if "X_aug_post" not in g:
            raise ValueError(f"{h5_path} was run in static mode; it has no hourly "
                             "lag states (rerun with filter.mode: sindy)")
        X = np.asarray(g["X_aug_post"], dtype=np.float64)
        P = np.asarray(g["P_aug_post"], dtype=np.float64)
        max_lag = X.size // k - 1
        W = spans[wi][1]
        lags = [int((W - tt) / _HOUR) for tt in times]
        if max(lags) > max_lag:
            raise ValueError(f"{n_hours} hours back from {T:%Y-%m-%d %H:%M} needs lag "
                             f"{max(lags)} h; the filter keeps {max_lag}")

        # Same operation order as the filter's own normalized -> physical conversion
        # (_process_window_augmented), so the newest hour matches z_post exactly.
        blk = [slice(L * k, (L + 1) * k) for L in lags]
        z = np.stack([y_std * X[b] + y_mean for b in blk])
        Sigma_joint = np.block([[y_std[:, None] * P[bi, bj] * y_std[None, :]
                                 for bj in blk] for bi in blk])
        Sigma = np.stack([Sigma_joint[i * k:(i + 1) * k, i * k:(i + 1) * k]
                          for i in range(n_hours)])

        if lags[-1] == 0:
            # Guards the lag convention: block 0 IS the stored end-of-window posterior.
            if not (np.allclose(z[-1], g["z_post"][()], rtol=1e-12, atol=0)
                    and np.allclose(Sigma[-1], g["P_post"][()], rtol=1e-12, atol=0)):
                raise RuntimeError(f"lag-0 block of window {keys[wi]} does not match its "
                                   "stored z_post/P_post; the file is inconsistent")

        # Arcs that informed these hours: every window from the one containing the
        # oldest hour through the one used.
        w0 = next(i for i, (ws, we) in enumerate(spans) if ws <= times[0] < we)
        n_obs_span = int(sum(int(f[f"windows/{keys[i]}"].attrs["n_obs"])
                             for i in range(w0, wi + 1)))
        out = {
            "times": [tt.strftime(_FMT) for tt in times],
            "z": z,
            "Sigma": Sigma,
            "sigma": np.sqrt(np.stack([np.diag(S) for S in Sigma])),
            "Sigma_joint": Sigma_joint,
            "lags_h": np.asarray(lags),
            "window_start": spans[wi][0].strftime(_FMT),
            "window_end": W.strftime(_FMT),
            "n_obs_window": int(g.attrs["n_obs"]),
            "n_obs_span": n_obs_span,
            "constrained": n_obs_span > 0,
            "rom_kind": str(f.attrs.get("rom_kind", "")),
            "h5": str(h5_path),
        }
    return out


def find_run(rom: str, T: datetime, n_hours: int, out_dir: Path = _OUT) -> str:
    """The run file in out_dir for this ROM whose span covers the requested hours."""
    first = T - (n_hours - 1) * _HOUR
    cands = sorted(set(glob.glob(str(out_dir / f"{rom}.h5")))
                   | set(glob.glob(str(out_dir / f"{rom}_*.h5"))))
    seen, covering = [], []
    for p in cands:
        try:
            s, e = run_span(p)
        except (OSError, ValueError, KeyError):
            continue
        seen.append(f"  {os.path.basename(p)}: {s:%Y-%m-%d %H:%M} .. {e:%Y-%m-%d %H:%M}")
        if s <= first and T <= e:
            covering.append((s, p))
    if covering:
        # Several runs can cover one hour (the Gannon run sits inside the long range).
        # The earliest-starting run has been assimilating longest, so it is furthest
        # from its initial-condition transient.
        return min(covering)[1]
    raise ValueError(f"no {rom} run in {out_dir} contains {first:%Y-%m-%d %H:%M} .. "
                     f"{T:%Y-%m-%d %H:%M}. The runs in out/ are:\n" + ("\n".join(seen) or "  none"))


def _print(s: dict, label: str) -> None:
    k = s["z"].shape[1]
    print(f"\n{label}  ({s['h5']})")
    print(f"  filter update used: {s['window_start']} -> {s['window_end']}   "
          f"EDR arcs used for these hours: {s['n_obs_span']}"
          + ("" if s["constrained"] else "   ** no EDR arcs: model prediction only **"))
    head = "  " + "mode".ljust(6) + "".join(f"{t.replace('T', ' ')[:16]:>26s}"
                                           for t in s["times"])
    print(head)
    for m in range(k):
        print(f"  z{m + 1:<5d}" + "".join(f"{s['z'][i, m]:>14.6g} ± {s['sigma'][i, m]:<9.3g}"
                                        for i in range(len(s["times"]))))


def _to_jsonable(s: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time", required=True, help="the hour in UTC, for example '2024-06-15 03:00'")
    ap.add_argument("--rom", choices=["pca", "coae", "both"], default="coae")
    ap.add_argument("--hours", type=int, default=3, help="number of hours to give, 1 to 4 (default 3). The last hour is --time")
    ap.add_argument("--h5", default=None, help="the run file. Without this option, the script finds it in out/")
    ap.add_argument("--rom-mat", default=None,
                    help="the model .mat file. Only necessary for old run files")
    ap.add_argument("--out", default=None,
                    help="also write the result to a .json or .npz file. With --rom both, "
                         "the script adds the model name to the file name")
    a = ap.parse_args()

    T = _parse_time(a.time)
    roms = ["pca", "coae"] if a.rom == "both" else [a.rom]
    for rom in roms:
        h5 = a.h5 or find_run(rom, T, a.hours)
        s = get_state(h5, T, a.hours, rom_mat=a.rom_mat)
        _print(s, rom.upper())
        if a.out:
            root, ext = os.path.splitext(a.out)
            path = f"{root}_{rom}{ext}" if len(roms) > 1 else a.out
            if ext == ".json":
                with open(path, "w") as fh:
                    json.dump(_to_jsonable(s), fh, indent=1)
            elif ext == ".npz":
                np.savez(path, **{k: np.asarray(v) for k, v in s.items()})
            else:
                raise SystemExit("--out must end in .json or .npz")
            print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
