"""
export_rope_framework.py -- export this ROPE-TIEGCM model as a rope-framework (v0.6) model directory.

    conda activate rope_export          # rope_env + tf2onnx
    python tools/export_rope_framework.py --out <dir>/tiegcm-f10ap-v1 \
        --decoder-from <rope-framework data>/models/tiegcm-aurora-v1

Writes the stacked_ensemble artifact set the framework loads:
    base_model_00..14.onnx   15 Keras base models (LSTM 0-4, GRU 5-9, Transformer 10-14), input (B, 3, 17)
    meta_model.onnx          Keras meta model, input (T, 3, 17) -> fusion weights (T, 15, 10)
    meta_model_out_shape.bin (ndim, T-placeholder, M, K) as in tiegcm-aurora-v1
    stats_ts.bin             feature normalizer, [10 latent | 7 driver]
    stats_cae.bin, coae_decoder.{onnx,onnx.data,pt}
                             copied from --decoder-from: the shipped TIE-GCM COAE decoder is the same network
                             with identical stats_cae (checked here, refuses to copy otherwise)
    ic_table.icbin           IC table on (f10, ap), filled to the full regular grid (see regular_ic_table)
    model_manifest.json      drivers f10, f10_41day_avg, ap, t1..t4; grid 72x36x45, lat +-87.5, 100..980 km

Each ONNX graph is checked against Keras on random inputs before it is written.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

K, S = 10, 3
DRIVER_COLS = ["f10", "f10_41day_avg", "ap", "t1", "t2", "t3", "t4"]
DESCRIPTIONS = {
    "f10": "F10.7 cm solar radio flux, in solar flux units (SFU). Must come from the model's raw driver data source (CSV column, .swbin field, or explicit override).",
    "f10_41day_avg": "Trailing 41-day (984-hour) mean of the raw 'f10' column, ending at and including the current row; averaged over fewer hours near the start of the source's coverage. Always computed by the consuming framework from raw f10 history; never read from a raw data source itself.",
    "ap": "Ap planetary geomagnetic equivalent amplitude index (linear scale, ~0-400). Must come from the model's raw driver data source (CSV column, .swbin field, or explicit override).",
    "t1": "sin(2*pi*hour/24) - diurnal phase harmonic. Always computed from the row's timestamp; never read from a raw data source.",
    "t2": "cos(2*pi*hour/24) - diurnal phase harmonic. Always computed from the row's timestamp; never read from a raw data source.",
    "t3": "sin(2*pi*day_of_year/365.25) - annual phase harmonic. Always computed from the row's timestamp; never read from a raw data source.",
    "t4": "cos(2*pi*day_of_year/365.25) - annual phase harmonic. Always computed from the row's timestamp; never read from a raw data source.",
}


def write_stats_bin(path, mu, sigma, shape):
    mu = np.asarray(mu, "<f4").reshape(-1)
    sigma = np.asarray(sigma, "<f4").reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(shape)))
        f.write(struct.pack("<" + "I" * len(shape), *shape))
        f.write(mu.tobytes())
        f.write(sigma.tobytes())


def read_stats_bin(path):
    b = open(path, "rb").read()
    nd = struct.unpack_from("<I", b)[0]
    sh = struct.unpack_from("<" + "I" * nd, b, 4)
    n = int(np.prod(sh))
    off = 4 + 4 * nd
    return np.frombuffer(b, "<f4", n, off), np.frombuffer(b, "<f4", n, off + 4 * n), sh


def regular_ic_table(ic_csv):
    """
    The IC table is scattered (1131 of the 42 x 74 (F10, Ap) nodes). The framework interpolates
    bilinearly and, if any of the 4 corners is missing, falls back to the nearest point in raw
    (F10, Ap) units, which is far from what Python ROPE does. Fill every node of the full grid with
    Python ROPE's own lookup (scipy linear on the triangulation, nearest outside the hull): values
    at existing nodes are unchanged, and bilinear between nodes then tracks the Python result.
    """
    import pandas as pd
    from scipy.interpolate import griddata
    t = pd.read_csv(ic_csv)
    ycols = [f"y{i}" for i in range(1, K + 1)]
    pts, vals = t[["F10", "Ap"]].to_numpy(), t[ycols].to_numpy()
    f_ax, a_ax = np.unique(t["F10"]), np.unique(t["Ap"])
    F, A = np.meshgrid(f_ax, a_ax, indexing="ij")
    q = np.column_stack([F.ravel(), A.ravel()])
    out = griddata(pts, vals, q, method="linear")
    bad = np.isnan(out).any(axis=1)
    out[bad] = griddata(pts, vals, q[bad], method="nearest")
    g = pd.DataFrame(out, columns=ycols)
    g.insert(0, "Ap", q[:, 1]); g.insert(0, "F10", q[:, 0])
    return g, len(t), int(bad.sum())


def write_icbin(path, ic_csv):
    t, n_orig, n_nearest = regular_ic_table(ic_csv)
    print(f"IC table: {n_orig} scattered rows -> {len(t)} regular nodes ({n_nearest} outside the hull, nearest)")
    ycols = [f"y{i}" for i in range(1, K + 1)]
    axes = [("f10", t["F10"].to_numpy()), ("ap", t["Ap"].to_numpy())]
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", 0x52504943, 2, len(t), K))
        for name, _ in axes:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
        rec = np.column_stack([axes[0][1], axes[1][1], t[ycols].to_numpy()]).astype("<f4")
        f.write(rec.tobytes())
    return len(t)


def _erf_gelu(x):
    import tensorflow as tf
    return 0.5 * x * (1.0 + tf.math.erf(x * 0.7071067811865476))


def _onnx_safe_activations(model):
    """Exact GELU traces to Erfc, which ONNX lacks; swap in the identical erf form (ONNX has Erf)."""
    n = 0
    for layer in model._flatten_layers(include_self=False, recursive=True):
        act = getattr(layer, "activation", None)
        if act is not None and getattr(act, "__name__", "") == "gelu":
            layer.activation = _erf_gelu
            n += 1
        # PositionalEncoding keeps an eager tensor, which tf2onnx turns into an extra graph input;
        # as a numpy array it is embedded as a constant (same values)
        if hasattr(layer, "pos_enc") and hasattr(layer.pos_enc, "numpy"):
            layer.pos_enc = layer.pos_enc.numpy()
    return n


def export_onnx(model, d_in, out_path, n_check=8, seed=0):
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort

    x = np.random.default_rng(seed).normal(0, 1, (n_check, S, d_in)).astype(np.float32)
    ref = model(x, training=False).numpy()          # original model, before any activation swap
    _onnx_safe_activations(model)

    spec = (tf.TensorSpec((None, S, d_in), tf.float32, name="x"),)

    @tf.function(input_signature=spec)
    def fn(x):
        return model(x, training=False)

    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=out_path)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))
    if err > 1e-3:          # float32 kernel differences (GRU ~2e-4); a real conversion error is O(1)
        raise RuntimeError(f"{os.path.basename(out_path)}: ONNX vs Keras relative error {err:.2e}")
    return err, tuple(got.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--decoder-from", required=True, help="framework model dir with the TIE-GCM COAE decoder")
    p.add_argument("--models", default=os.path.join(ROOT, "Models", "StormTuned"))
    p.add_argument("--meta", default=os.path.join(ROOT, "Meta Models", "meta_model_41df10_ap_stormtuned.keras"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    from tensorflow import keras
    from ts_utils.custom_layers import PositionalEncoding

    # ---- stats
    ts = torch.load(os.path.join(ROOT, "data", "stats_ts.pt"), map_location="cpu")
    mu, sg = ts["mu"].numpy(), ts["sigma"].numpy()
    assert len(mu) == K + len(DRIVER_COLS)
    write_stats_bin(os.path.join(a.out, "stats_ts.bin"), mu, sg, (len(mu),))
    cae = torch.load(os.path.join(ROOT, "data", "stats_cae.pt"), map_location="cpu", weights_only=True)
    fmu, fsg, fsh = read_stats_bin(os.path.join(a.decoder_from, "stats_cae.bin"))
    if not (np.array_equal(fmu, cae["mu"].numpy().reshape(-1)) and np.array_equal(fsg, cae["sigma"].numpy().reshape(-1))):
        raise RuntimeError("stats_cae differs from --decoder-from; the decoder cannot be reused")
    for f in ("stats_cae.bin", "coae_decoder.onnx", "coae_decoder.onnx.data", "coae_decoder.pt"):
        if os.path.exists(os.path.join(a.decoder_from, f)):
            shutil.copy2(os.path.join(a.decoder_from, f), os.path.join(a.out, f))
    print("stats_ts.bin, stats_cae.bin + decoder copied (identical stats_cae)")

    # ---- IC table
    n = write_icbin(os.path.join(a.out, "ic_table.icbin"), os.path.join(ROOT, "data", "IC_Table.csv"))
    print(f"ic_table.icbin: {n} rows, axes (f10, ap)")

    # ---- base models
    d_in = len(mu)
    base = []
    for arch, sub, co in (("lstm", "LSTM MODELS", None), ("gru", "GRU MODELS", None),
                          ("transformer", "TRANSFORMER MODELS", {"PositionalEncoding": PositionalEncoding})):
        for i in range(1, 6):
            idx = len(base)
            m = keras.models.load_model(os.path.join(a.models, sub, f"best_model_{i}.keras"),
                                        compile=False, custom_objects=co)
            fname = f"base_model_{idx:02d}.onnx"
            err, shp = export_onnx(m, d_in, os.path.join(a.out, fname), seed=idx)
            base.append({"file": fname, "backend": "onnx", "architecture": arch,
                         "inter_op_threads": 2 if arch == "transformer" else 1})
            print(f"{fname}  {arch:11s} {sub}/best_model_{i}.keras  out {shp}  rel err {err:.1e}")

    # ---- meta model
    meta = keras.models.load_model(a.meta, compile=False)
    err, shp = export_onnx(meta, d_in, os.path.join(a.out, "meta_model.onnx"), seed=99)
    if shp[1:] != (len(base), K):
        raise RuntimeError(f"meta model output {shp}, expected (T, {len(base)}, {K})")
    with open(os.path.join(a.out, "meta_model_out_shape.bin"), "wb") as f:
        f.write(struct.pack("<iii", 2, len(base), K))
    print(f"meta_model.onnx  out {shp}  rel err {err:.1e}")

    # ---- manifest (same schema as the shipped tiegcm-aurora-v1 / wam-borealis-v1)
    manifest = {
        "schema_version": 1,
        "kind": "stacked_ensemble",
        "runtime_requirements": {"onnxruntime": "1.25", "libtorch": "2.7"},
        "latent_dim": K,
        "validated": False,
        "grid": {"n_lst": 72, "n_lat": 36, "n_alt": 45, "lat_min_deg": -87.5, "lat_max_deg": 87.5,
                 "alt_min_km": 100.0, "alt_max_km": 980.0},
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "ap"], "file": "ic_table.icbin"}},
        "decoder": {"kind": "coae", "params": {"decode_batch_size": 120, "stages": [
            {"backends": {"onnx": "coae_decoder.onnx", "libtorch": "coae_decoder.pt"},
             "stats": "stats_cae.bin", "alt_start": 0, "alt_end": 45}]}},
        "stacked_ensemble": {"seq_len": S, "base_models": base,
                             "meta_model": {"file": "meta_model.onnx", "backend": "onnx"}},
        "drivers": {"source": "celestrak_sw",
                    "columns": [{"name": c, "description": DESCRIPTIONS[c]} for c in DRIVER_COLS]},
    }
    with open(os.path.join(a.out, "model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("model_manifest.json written ->", a.out)


if __name__ == "__main__":
    main()
