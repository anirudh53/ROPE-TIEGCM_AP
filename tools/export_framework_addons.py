"""
export_framework_addons.py -- write the rope-framework add-on files for this model.

    python tools/export_framework_addons.py da  --model-dir <dir>/tiegcm-f10ap-v1
    python tools/export_framework_addons.py f10 --model-dir <dir>/tiegcm-f10ap-v1 [--type LSTM_dynamic]

da   -> <model-dir>/addons/da_states_<first>_<last>.dabin
        Assimilated initial conditions for every hour of the DA run, already in this model's latent
        space (rope_da.DAIC: fitted latents + filter covariance carried over with the decoder
        Jacobians). The framework reads it when [data_assimilation] enabled = true.
f10  -> <model-dir>/addons/f10_<type>.f10bin
        The 180-member F10.7 forecaster (Daniell & Mehta 2023) as plain layer weights, for the
        framework's built-in forward pass when [f10_forecaster] enabled = true.

Both also register themselves in <model-dir>/model_manifest.json under "addons" (ignored by
rope-framework builds without add-on support).

.dabin (little-endian):
    u32 magic 0x52504441 "RPDA", u32 version 1, u32 n_records, u32 seq_len S, u32 latent_dim K
    u32 len + bytes   source description (utf-8)
    n_records x { i64 tp (hour T, unix s), i32 n_obs_span,
                  f32 z[S*K] (hours T-S+1..T, oldest first), f32 Sigma_joint[(S*K)^2] (row-major) }

.f10bin (little-endian):
    u32 magic 0x52504631 "RPF1", u32 version 1
    f64 scale_mean, f64 scale_std, u32 max_lookback (days), u32 n_features (2: f10, backward avg)
    u32 history_avg_days (12), u32 n_models
    per model: u32 lookback (days), u32 input_kind (0 = sequence (2*lookback, 1), 1 = flat 2*lookback),
               u32 n_layers, per layer:
                 u32 type (0 LSTM, 1 Dense), u32 act, u32 rec_act, u32 return_sequences,
                 u32 in_dim, u32 units,
                 LSTM : f32 kernel[in_dim*4u], f32 recurrent[u*4u], f32 bias[4u]   (gates i, f, c, o)
                 Dense: f32 kernel[in_dim*u], f32 bias[u]
    activation ids: 0 linear, 1 tanh, 2 sigmoid, 3 relu
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ACT = {"linear": 0, "tanh": 1, "sigmoid": 2, "relu": 3}


def _register(model_dir, key, entry):
    path = os.path.join(model_dir, "model_manifest.json")
    m = json.load(open(path))
    m.setdefault("addons", {})[key] = entry
    json.dump(m, open(path, "w"), indent=2)


def _bytes(s):
    b = s.encode()
    return struct.pack("<I", len(b)) + b


# --------------------------------------------------------------------------- DA
def export_da(model_dir, h5=None, chunk=64, every=1):
    from rope import ROPE
    import rope_da
    rope = ROPE()
    da = rope_da.DAIC(rope, **({"h5_path": h5} if h5 else {}), chunk=chunk, verbose=True)
    first, last = (pd.Timestamp(t) for t in da.coverage(3))
    hours = pd.date_range(first, last, freq=f"{every}h")
    print(f"[da] {len(hours)} hours {first} .. {last}")
    da.prefit(hours)                                      # batched fits, cached in da/out/

    S, K = rope.cfg.seq_len, rope.cfg.latent_dim
    name = f"da_states_{first:%Y%m%d}_{last:%Y%m%d}.dabin"
    out = os.path.join(model_dir, "addons", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    src = (f"EDR-arc COAE assimilation run {os.path.basename(da.h5)}; latents fitted to this model's decoder "
           f"(rope_da.DAIC); covariance carried with decoder Jacobians")
    with open(out, "wb") as f:
        f.write(struct.pack("<IIIII", 0x52504441, 1, len(hours), S, K))
        f.write(_bytes(src))
        for i, T in enumerate(hours):
            st = da.initial_state(T, n=S)
            f.write(struct.pack("<qi", int(T.timestamp()), int(st["n_obs_span"])))
            f.write(np.asarray(st["z"], "<f4").reshape(-1).tobytes())
            f.write(np.asarray(st["Sigma_joint"], "<f4").reshape(-1).tobytes())
            if i % 500 == 0:
                print(f"[da] wrote {i}/{len(hours)}  {T}")
    _register(model_dir, "data_assimilation", {
        "kind": "da_state_table", "file": f"addons/{name}",
        "coverage": [f"{first:%Y-%m-%dT%H:%M:%S}", f"{last:%Y-%m-%dT%H:%M:%S}"],
        "seq_len": S, "latent_dim": K,
        "description": "Assimilated initial latent states and joint covariance (hours T-2..T) per hour T.",
    })
    print(f"[da] -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- F10
def export_f10(model_dir, model_type="LSTM_dynamic", f10_dir=None):
    f10_dir = f10_dir or os.path.join(ROOT, "F10_Forecasting")
    sys.path.insert(0, f10_dir)
    import f10_forecaster as F
    from keras.layers import LSTM, Dense
    fc = F.F10Forecaster(base_dir=f10_dir, data_file="Data/noaa_radio_flux_celestrak_ext.csv")
    keras_type = "MLP" if model_type.startswith("MLP") else "LSTM"
    models = [(lb, fc._cache.get(model_type, arch, lb, it, keras_type))
              for lb in range(7, 23, 3) for arch in range(1, 4) for it in range(10)]   # forecaster order

    name = f"f10_{model_type}.f10bin"
    out = os.path.join(model_dir, "addons", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<II", 0x52504631, 1))
        f.write(struct.pack("<dd", float(np.asarray(fc.mean)), float(np.asarray(fc.std))))
        f.write(struct.pack("<IIII", fc.max_lookback, fc.features, 12, len(models)))
        for lb, m in models:
            f.write(struct.pack("<III", lb, 0 if keras_type == "LSTM" else 1, len(m.layers)))
            for layer in m.layers:
                w = layer.get_weights()
                act = ACT[layer.activation.__name__]
                if isinstance(layer, LSTM):
                    rec = ACT[layer.recurrent_activation.__name__]
                    units = w[1].shape[0]
                    f.write(struct.pack("<IIIIII", 0, act, rec, int(layer.return_sequences), w[0].shape[0], units))
                    for a in w:
                        f.write(np.asarray(a, "<f4").reshape(-1).tobytes())
                elif isinstance(layer, Dense):
                    f.write(struct.pack("<IIIIII", 1, act, 0, 0, w[0].shape[0], w[0].shape[1]))
                    for a in w:
                        f.write(np.asarray(a, "<f4").reshape(-1).tobytes())
                else:
                    raise TypeError(f"unsupported layer {type(layer).__name__}")
    entry = {"kind": "f10_ensemble", "file": f"addons/{name}", "model_type": model_type,
             "n_members": len(models),
             "description": "Daniell & Mehta (2023) F10.7 forecaster, dynamic lookback ensemble (7-22 d).",
             "reference": "https://arxiv.org/abs/2306.02169"}
    path = os.path.join(model_dir, "model_manifest.json")
    m = json.load(open(path))
    ad = m.setdefault("addons", {})
    fcd = ad.setdefault("f10_forecaster", {"default": model_type, "ensembles": {}})
    fcd["ensembles"][model_type] = entry
    json.dump(m, open(path, "w"), indent=2)
    print(f"[f10] {len(models)} {model_type} members -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("da"); d.add_argument("--model-dir", required=True); d.add_argument("--h5")
    d.add_argument("--chunk", type=int, default=64); d.add_argument("--every", type=int, default=1)
    g = sub.add_parser("f10"); g.add_argument("--model-dir", required=True)
    g.add_argument("--type", default="LSTM_dynamic", choices=["LSTM_dynamic", "MLP_dynamic"])
    a = p.parse_args()
    if a.cmd == "da":
        export_da(a.model_dir, a.h5, a.chunk, a.every)
    else:
        export_f10(a.model_dir, a.type)


if __name__ == "__main__":
    main()
