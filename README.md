# ROPE-TIEGCM

**Reduced-Order Probabilistic Emulator** of TIE-GCM thermospheric density, driven by
F10.7, its 41-day average and ap. It forecasts global 3-D density (72 LST × 36 lat × 45 alt,
100–980 km, hourly) with uncertainty estimates.

* **Latent dynamics:** 15-model ensemble (5 LSTM, 5 GRU, 5 Transformer) fused by a meta-model, in the
  10-dim latent space of a convolutional orthogonal autoencoder (COAE)
* **Optional data-assimilation initial condition** (`use_da`): latent state from an EDR-arc
  assimilation filter
* **Optional F10.7 forecast** (`use_f10_forecaster`): neural-network ensemble forecast of F10.7
* **Uncertainty:** model (ensemble) spread, plus input uncertainty (IC and/or F10) through one
  unscented transform

See **[demo.ipynb](demo.ipynb)**.

## Quick start

```python
from rope import ROPE

rope = ROPE()                                            # IC table + F10 from the driver CSV
out  = rope.run("2024-05-10 18:00", horizon=72)          # uncertainty=True by default

out["mean_density"]   # (72, 72, 36, 45)  kg/m³   (hour, LST, lat, alt)
out["density_std"]    # (72, 72, 36, 45)
out["window_df"]      # datetime, f10, f10_41day_avg, ap used

rope = ROPE(use_da=True, use_f10_forecaster=True)        # DA IC + forecast F10
out  = rope.run("2024-05-10 18:00", horizon=72)
out  = rope.run("2024-05-10 18:00", horizon=72, use_da=False)   # per-call override
```

| `use_da` | `use_f10_forecaster` | Initial condition | F10.7 | `uncertainty=True` gives |
|---|---|---|---|---|
| False | False | IC table (F10, ap lookup) | driver CSV | model spread |
| True | False | data assimilation | driver CSV | model + IC (10-dim UT, 21 points) |
| False | True | IC table | forecaster after the start hour | model + F10 (1-dim UT, 3 points) |
| True | True | data assimilation | forecaster after the start hour | model + IC + F10 (11-dim UT, 23 points) |

`density_std` is always the total of what is switched on, √(input² + model²). With an input
switched on, the parts are also returned: `ic_std`, `f10_std`, `model_std`, `input_std`.

Density at a point (time, LST, lat, alt), with σ:

```python
from interpolator import DensityInterpolator
DensityInterpolator(out).query("2024-05-11 06:30", lst=14.0, lat=30.0, alt_km=450.0, time_mode="interp_time")
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `drivers` | `"celestrak"` | `"celestrak"`: `data/sw_celestrack_1957_ap.csv.gz` (1957-10 → 2026-10); `"tiegcm"`: TIE-GCM training drivers (1996–2009) |
| `driver_csv` | – | your own hourly CSV with `datetime, f10, ap` |
| `device` | `"cuda"` | falls back to CPU if no GPU |
| `ic_cov_mode` | `"hour_avg"` | how the 3-hour DA covariance becomes the 10×10 IC covariance: `"hour_avg"`, `"mean"`, `"last"` |
| `ic_cov_scale` | `1.0` | multiplies the DA σ (the filter's own σ is known to be too small, see `da/README_DA.md`) |
| `f10_model_type` | `"LSTM_dynamic"` | or `"MLP_dynamic"` |
| `f10_sigma` | – | fixed F10 σ (sfu) instead of the forecaster's ensemble spread |

`run()` also takes `uncertainty`, `use_da`, `use_f10_forecaster`, `f10_sigma`, `driver_df`
(per-call driver override), and `ic` (advanced: `"table"`, `"da"`, or a `(3, 10)` latent array).

## Installation

```bash
conda create -n rope python=3.11 -y
conda activate rope
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
```

Run from the repository root. A GPU is optional; one 72-hour run takes a few seconds on a GPU.

## How it works

1. **Drivers.** Hourly F10.7, F10.7 41-day trailing mean and ap, plus diurnal/annual harmonics.
   The CelesTrak file is built by `tools/build_celestrak_drivers.py` from CelesTrak `SW-All.csv`:
   F10.7 (observed) anchored at 00 UT and Kp/ap on 3-hour nodes, linear in between.
2. **Initial condition.** Three hourly latent states (T-2h … T). Either interpolated from an IC table on
   (F10, ap), or the assimilated state (`use_da`).
3. **Forecast.** 15 ensemble members roll out autoregressively in latent space; a meta-model
   weights them per step and per latent coefficient.
4. **Decode.** The COAE decoder maps latents to log10 density.
5. **Uncertainty.**
   * Model: unscented transform over the ensemble's latent covariance at each hour.
   * Input: unscented transform over the switched-on inputs: the mean of the 3 DA states
     (10 dims, DA covariance) and/or one F10 offset over the forecast hours (σ from the F10
     ensemble). All sigma points run in one batched pass.

### Data assimilation (`use_da`)

The data-assimilation work (the EDR-arc filter and its COAE run) is by **Sriram Narayanan**
(West Virginia University).

`da/` contains an EDR-arc assimilation filter run (COAE latent state, 15 debris objects) for
**2024-03-01 → 2024-07-31**. Start times should fall on 00/03/06/… UT (filter update times).
The filter's decoder has the same weights as ROPE's but different normalization, so
`rope_da.DAIC` fits the ROPE-space latent whose decoded field matches the filter's, and carries
the filter covariance across with the decoder Jacobians. Fits are cached in `da/out/`.
To assimilate another period, `rope_da.run_filter(start, end, edr_npz=...)` needs EDR arcs for
it (see `da/README_DA.md`).

### F10.7 forecaster (`use_f10_forecaster`)

`F10_Forecasting/`: dynamic-lookback LSTM and MLP ensembles (180 members each) from
Daniell & Mehta (2023), *Probabilistic Solar Proxy Forecasting with Neural Network Ensembles*,
Space Weather ([arXiv:2306.02169](https://arxiv.org/abs/2306.02169)). Its history file is NOAA
F10.7 to 2021-12-10 extended with CelesTrak observations (`tools/extend_f10_history.py`).
Observed F10 is kept up to the start hour; the forecast mean is used after it.

### Notes on the drivers

* **ap.** The training ap was built by converting 3-hourly Kp to ap and interpolating ap linearly
  between the 3-hour nodes. CelesTrak's official 3-hourly ap, interpolated the same way, matches it to about 1 nT
  (correlation 0.99). Flat stretches in ap are real: two consecutive 3-hour intervals with the same value.
* **F10.7.** ROPE was trained on the F10.7 that drove the TIE-GCM runs. That series is smoother than observed
  daily F10.7: about 13 sfu day-to-day variability vs 23 sfu observed. ROPE is run with observed F10.7, as was the
  original ROPE. Against GRACE-FO and Swarm-B (March–July 2024), smoothing the observed F10.7 (3–7 day or 81 day)
  changed the error by at most ~2 percentage points, so it is not applied.
* **Known bias.** Over March–July 2024 ROPE runs about 1.25–1.35× above GRACE-FO/Swarm-B densities with any of
  these driver choices, so the bias is not driver-related.

## Repository layout

```
rope.py               ROPE class (switches, forecast, uncertainty)
rope_inputs.py        unscented transform, DA covariance reduction, F10 forecaster helpers
rope_da.py            DA initial conditions (DAIC) and filter runner
interpolator.py       point queries on ROPE output
demo.ipynb            demo
ae_utils/ ts_utils/   COAE and custom-layer definitions
weights/              COAE weights
Models/StormTuned/    15 ensemble members        Meta Models/   meta-model
data/                 normalization stats, IC table, driver CSVs
da/                   assimilation filter code, run file, IC cache
F10_Forecasting/      F10.7 forecaster code, models, history
tools/                driver / F10-history builders; export to rope-framework (C++): export_rope_framework.py, export_framework_addons.py
```

## Updating the drivers

```bash
python tools/build_celestrak_drivers.py                    # downloads SW-All.csv, rewrites data/sw_celestrack_1957_ap.csv.gz
python tools/extend_f10_history.py --sw-all data/SW-All.csv
```

## Credits

* **ROPE-TIEGCM** (models, DA and F10.7 coupling, uncertainty): Anirudh Tapedia, West Virginia University
* **Data assimilation** (EDR-arc assimilation filter, `da/`): Sriram Narayanan, West Virginia University,
  sriram.narayanan@mail.wvu.edu
* **F10.7 forecaster** (`F10_Forecasting/`): Joshua D. Daniell and Piyush M. Mehta,
  *Probabilistic Solar Proxy Forecasting with Neural Network Ensembles*, Space Weather (2023)

## Contact

Anirudh Tapedia, West Virginia University, anirudh.tapedia@mail.wvu.edu
