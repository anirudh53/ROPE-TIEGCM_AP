# Density assimilation demo: PCA and COAE

This bundle puts thermosphere density data from 15 debris objects into a
reduced-order density model. It compares two versions of the model:

- **PCA**: a linear model.
- **COAE**: a nonlinear model (a convolutional autoencoder).

Both versions use the same filter code. Only the model is different.

The data is **EDR arcs**. Each arc is the mean density along the orbit of one object
between two of its TLEs (two-line element sets). We get it from how fast the orbit
decays. The 15 objects are not random. An optimizer selected them from the
Space-Track catalogue because they give the most information about the model state.

**HASDM** is the density model that we use as the truth when we score the results.

## Step 1. Install

    conda create -n coae python=3.11 -y
    conda activate coae
    pip install -r requirements.txt

Then install torch. If your machine has a different CUDA version, change `cu124` in
this command:

    pip install torch --index-url https://download.pytorch.org/whl/cu124

A GPU is not necessary. All steps run on a CPU.

## Step 2. Make sure that your machine gives the correct results

Run the test:

    bash run_demo.sh

The test uses the Gannon storm, 2024-05-10 to 2024-05-13. It does these steps:

1. It makes sure that the COAE decoder gives the same densities as our CPU.
2. It runs both filters.
3. It compares both results with HASDM density.
4. It compares your results with our results in `reference/`.

The last line tells you the result. If the last line says `DISAGREES`, do not use the
results from this machine. Send us the output.

The test is necessary because two errors in the COAE decoder occur only on CUDA
GPUs. One is TF32 in the convolutions. The other is a failure of `vmap` with the
TorchScript decoder. The RMSE values are too coarse to show these errors. Part 1 of
the test finds them.

Our CPU results:

| | PCA | COAE |
|---|---|---|
| RMSE, 15 objects, with the filter | 0.3420 | 0.3306 |
| RMSE, 15 objects, model only | 0.4880 | 0.4286 |
| RMSE, 12 held-out orbits, with the filter | 0.2833 | 0.2549 |
| RMSE, 12 held-out orbits, model only | 0.3996 | 0.3360 |
| Filter NIS (mean) | 1.0363 | 0.5781 |

The RMSE unit is log10 kg/m³. "Held-out" means 12 orbits that the filter did not use,
so they give the fairest test. "Model only" means the model without the filter. NIS
(normalized innovation squared) is about 1 when the filter's uncertainty matches its
real errors.

The PCA column is the same as the published results to six decimals. The test
stops with an error if this is not true.

## Step 3. Run the filters for the long date range

This bundle has these date ranges:

| Label | First day | Last day | EDR arcs | Objects with arcs |
|---|---|---|---|---|
| `20240301_20240731` | 2024-03-01 | 2024-07-31 | 2995 | 15 of 15 |

Run both filters for a range:

    bash run_demo.sh --range 20240301_20240731

Add `--device cpu` to use the CPU only.

This step has no pass/fail result, because there is no HASDM truth for these dates. It makes the files that Step 4 reads.

For the 5-month range, the COAE filter takes about 10 minutes on a laptop CPU. The PCA filter is faster. Each filter writes one file of about 80 MB, `out/pca_20240301_20240731.h5` and `out/coae_20240301_20240731.h5`.

Each run starts one day before the first day. The filter needs this day to settle. Step 4 accepts hours from 00:00 on the first day to 23:00 on the last day.

An object without arcs was not in orbit, or was too high, on those dates. `data/ranges.json` gives the number of arcs for each object.

## Step 4. Get z and Σz for an hour

Give an hour. The script gives z and Σz for that hour and for the two hours before
it:

    python code/query_state.py --time "2024-06-15 03:00"

For the example above, you get these values:

- z at 01:00, 02:00 and 03:00. z has 10 values (one for each mode).
- Σz at each of these hours (a 10 × 10 matrix).
- The joint Σz for the three hours together (a 30 × 30 matrix).

Options:

| Option | What it does |
|---|---|
| `--rom coae` | Uses the COAE model. This is the default. |
| `--rom pca` | Uses the PCA model. |
| `--rom both` | Gives the results for the two models. |
| `--hours 4` | Gives 4 hours, not 3. The maximum is 4. |
| `--out result.npz` | Also writes the result to a file. Use `.npz` or `.json`. |
| `--h5 FILE` | Uses this run file. Without this option, the script finds the file in `out/`. |

You can also use the script from Python:

    import sys
    sys.path.insert(0, "code")
    from query_state import get_state

    s = get_state("out/coae_LABEL.h5", "2024-06-15 03:00")
    s["times"]        # the 3 hours, oldest first
    s["z"]            # z, shape (3, 10)
    s["Sigma"]        # Σz for each hour, shape (3, 10, 10)
    s["Sigma_joint"]  # joint Σz, shape (30, 30)

Replace LABEL with the label from the table in Step 3.

Rules for the time:

- Use UTC.
- Use a full hour. The script does not accept 03:30.
- The hours must be inside a run that you did in Step 2 or Step 3. If they are not,
  the script stops and shows the dates of your runs.

If two runs contain the hour, the script uses the run that started first. That run
has used more data.

## How the script gets hourly values

The filter does one update every 3 hours. But its state keeps the last 6 hourly
values of z, and each update corrects all 6 values.

For hour T, the script uses the first 3-hour update at or after T. So each value
uses all EDR arcs until that update. This is at most 2 hours after T. This type of
result is a "smoothed" estimate. It is not a real-time estimate.

All the hours come from the same update, so the joint Σz is correct for the three
hours together. At 03:00, 06:00, 09:00 and the other update times, the newest hour
is the same as the value that the filter stored. The script makes sure of this
each time.

z uses the units of its model. You cannot compare PCA z with COAE z. To get density
from z, use the same model (`data/pca_model.mat`, or `data/coae_model.mat` with
`data/coae_decoder.pt`).

## Limits of the results

- **Σz is too small.** The filter's uncertainty is about 70 times too
  small when we compare it with HASDM. We do not know the cause. Use Σz as the
  filter's own estimate, not as a calibrated error.
- **The two models have different tuning.** We tuned PCA so that NIS is about 1.
  We did not do this for COAE. In our tuning tests, it made the COAE error on
  held-out satellites about 12 MAPE points worse. So do not compare the two models
  by NIS.
- **High altitudes.** Above about 600 km, the densities are near the top of
  the range that the models learned. Both models reach the measured values here.
  COAE has less margin at the high-density end.
- **HASDM data.** `data/truth_n15_20240510_20240513.npz` contains HASDM densities
  along 27 orbits. HASDM belongs to Space Environment Technologies. Do not give this
  file to other persons. All steps except the RMSE comparison work without it.

## What is different between the two models

Both models have 10 modes and the same grid: 72 local times, 36 latitudes and 45
altitudes. Both use the same SINDy-c dynamics, the same EDR arcs and the same
filter. They differ only in how z becomes density:

    PCA   log10 rho(x) = mu0(x) + U(x) . z     linear. The Jacobian does not change with z.
    COAE  log10 rho(x) = D(z)(x)               neural network. The filter calculates
                                               the Jacobian again at each update.

An EDR arc is a mean density along the orbit, not the density at one point. The
mean uses weights of v_rel³. So the filter calculates log10(Σ w_i ρ_i) from about 10
points on each arc. This calculation is correct for any model. That is why one
filter works with both models.

A GPU helps with larger work than this bundle: the full catalogue, many runs, or the
full density field. On an RTX 6000 Ada, the
COAE decoder is 27 to 39 times faster than on a CPU.

## Files in this bundle

| Folder or file | Contents |
|---|---|
| `code/` | The filter, the decoder, the propagator and the scripts |
| `configs/` | The filter settings for PCA and COAE, and the COAE model settings |
| `data/` | The two models, the COAE decoder, the EDR arcs for the storm, the list of the 15 objects, and the HASDM data |
| `data/edr/` | The EDR arcs for the long date range |
| `data/ranges.json` | The date ranges, with the number of arcs for each object |
| `reference/` | Our CPU results, for Step 2 |
| `out/` | The results of your runs |
| `MANIFEST.txt` | The size and SHA-256 of each file |
