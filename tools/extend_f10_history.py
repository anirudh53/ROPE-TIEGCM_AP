"""
extend_f10_history.py -- extend the F10 forecaster's daily history with CelesTrak observations.

F10_Forecasting/Data/noaa_radio_flux_fullset.csv stops at 2021-12-10, so the forecaster cannot
build its 22-day lookback for later start dates (e.g. the 2024 DA period). This keeps every
NOAA row as-is and appends CelesTrak F10.7_OBS for the days after the last NOAA day, written in
the same "time,f10" / M/D/YYYY layout.

    python tools/extend_f10_history.py --sw-all data/SW-All_celestrak_20261013.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
F10_DATA = os.path.join(ROOT, "F10_Forecasting", "Data")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sw-all", required=True)
    p.add_argument("--noaa", default=os.path.join(F10_DATA, "noaa_radio_flux_fullset.csv"))
    p.add_argument("--out", default=os.path.join(F10_DATA, "noaa_radio_flux_celestrak_ext.csv"))
    a = p.parse_args()

    noaa = pd.read_csv(a.noaa)
    noaa_t = pd.to_datetime(noaa["time"], format="%m/%d/%Y")
    sw = pd.read_csv(a.sw_all, parse_dates=["DATE"])
    sw = sw[sw["F10.7_DATA_TYPE"].isin(["OBS", "INT"]) & (sw["DATE"] > noaa_t.max())]
    sw = sw.dropna(subset=["F10.7_OBS"])

    ext = pd.DataFrame({"time": [f"{d.month}/{d.day}/{d.year}" for d in sw["DATE"]],
                        "f10": sw["F10.7_OBS"].to_numpy()})
    out = pd.concat([noaa[["time", "f10"]], ext], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: NOAA {noaa_t.min():%Y-%m-%d}..{noaa_t.max():%Y-%m-%d} "
          f"+ CelesTrak {sw['DATE'].min():%Y-%m-%d}..{sw['DATE'].max():%Y-%m-%d} ({len(ext)} days)")


if __name__ == "__main__":
    main()
