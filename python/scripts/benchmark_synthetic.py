#!/usr/bin/env python3
"""
Synthetic benchmark behind the "preliminary results" section of the paper.

One protocol throughout:

* site 46.52 N, 6.63 E, 500 m (Swiss plateau, open horizon) unless stated;
* truth: tilt 30 deg, azimuth -20 deg (EU), 100 kWp;
* window 2023-05-01 .. 2023-06-20, hourly, UTC;
* measurements produced at 1-minute resolution and averaged over each hour,
  labelled by the interval's end -- i.e. what a meter records -- then given
  multiplicative Gaussian noise;
* the dictionary is integrated over the same interval, with the same label.

Every number quoted in the paper comes from a run of this script.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from pv_orientation_estimator import (
    build_pu_power_matrix,
    run_estimation,
    DAYTIME_GHI_THRESHOLD,
    LAYOUTS,
)
from pv_orientation_estimator.demo import synthetic_plant
from pv_orientation_estimator.horizon import download_horizon

SITE = (46.52, 6.63, 500.0)
WIMMIS = (46.6777, 7.6409, 639.0)
TILT, AZ, KWP = 30, -20, 100.0
START, END = "2023-05-01", "2023-06-20"
TILTS = np.array([t for t, _ in LAYOUTS])
AZS = np.array([a for _, a in LAYOUTS])


def centroid(alpha):
    """Capacity-weighted tilt and azimuth — sub-grid, and robust to ridges."""
    total = alpha.sum()
    return (alpha * TILTS).sum() / total, (alpha * AZS).sum() / total


def row(label, result, truth=(TILT, AZ, KWP)):
    if result["alpha"] is None:
        return f"  {label:<32} solver: {result['status']}"
    tilt_c, az_c = centroid(result["alpha"])
    return (f"  {label:<32} {result['best_tilt']:>4}° {result['best_az_eu']:>5}° "
            f"{tilt_c:>7.1f} {az_c:>7.1f} {result['effective_kWp']:>9.1f} "
            f"{100*(result['effective_kWp']/truth[2]-1):>+7.1f}% {result['r2']:>8.5f}")


HEAD = (f"  {'':<32} {'tilt':>5} {'az':>6} {'tilt_c':>7} {'az_c':>7} "
        f"{'kWp':>9} {'err':>8} {'R²':>8}")


def noisy(power, pct, seed=0):
    if not pct:
        return power
    rng = np.random.default_rng(seed)
    return np.clip(power * (1 + pct / 100 * rng.standard_normal(len(power))), 0.0, None)


def make(site, stamps, tilt=TILT, az=AZ, kwp=KWP, **kwargs):
    return synthetic_plant(*site, stamps, tilt, az, kwp,
                           interval="1h", label="end", **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-horizon", action="store_true",
                        help="skip the experiment that downloads a PVGIS horizon")
    args = parser.parse_args(argv)

    stamps = pd.date_range(START, END, freq="1h", tz="UTC")
    matrix, ghi = build_pu_power_matrix(*SITE, stamps, interval="1h", label="end")
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    print(f"protocol: {len(stamps)} hourly samples, {daytime.sum()} daytime, "
          f"{len(LAYOUTS)} dictionary columns\n")

    print("E1  baseline, open horizon, no clipping"); print(HEAD)
    clean = make(SITE, stamps)
    for pct in (0.0, 1.0, 2.0, 4.0, 8.0):
        print(row(f"noise {pct:.0f} %", run_estimation(matrix, noisy(clean, pct), daytime)))

    print("\nE2  interval integration vs sampling the model at the timestamp"); print(HEAD)
    plain, plain_ghi = build_pu_power_matrix(*SITE, stamps)
    plain_day = plain_ghi > DAYTIME_GHI_THRESHOLD
    for label in ("centre", "start", "end"):
        measured = synthetic_plant(*SITE, stamps, TILT, AZ, KWP,
                                   interval="1h", label=label, noise_pct=2.0)
        exact, exact_ghi = build_pu_power_matrix(*SITE, stamps, interval="1h", label=label)
        print(row(f"{label:<6} model at timestamp",
                  run_estimation(plain, measured, plain_day)))
        print(row(f"{label:<6} model integrated",
                  run_estimation(exact, measured, exact_ghi > DAYTIME_GHI_THRESHOLD)))

    print("\nE3  window length (2 % noise)"); print(HEAD)
    for weeks in (1, 2, 4, 7):
        short = pd.date_range(START, periods=weeks * 7 * 24, freq="1h", tz="UTC")
        sub_matrix, sub_ghi = build_pu_power_matrix(*SITE, short, interval="1h", label="end")
        sub_day = sub_ghi > DAYTIME_GHI_THRESHOLD
        print(row(f"{weeks} week(s), {sub_day.sum():>3} daytime",
                  run_estimation(sub_matrix, noisy(make(SITE, short), 2.0), sub_day)))

    print("\nE4  two arrays behind one meter: 60 kWp @ (30,-45) + 40 kWp @ (30,+45)")
    print(HEAD)
    mixed = (make(SITE, stamps, tilt=30, az=-45, kwp=60.0)
             + make(SITE, stamps, tilt=30, az=45, kwp=40.0))
    for pct in (0.0, 2.0):
        result = run_estimation(matrix, noisy(mixed, pct), daytime)
        print(row(f"noise {pct:.0f} %", result, truth=(30, 0, 100.0)))
        alpha = result["alpha"]
        print(f"      east of -20°: {alpha[AZS < -20].sum():5.1f} kWp (true 60), "
              f"west of +20°: {alpha[AZS > 20].sum():5.1f} kWp (true 40)")

    if not args.skip_horizon:
        print("\nE5  terrain horizon, Wimmis BE, 2023-02-01..03-25 (2 % noise)"); print(HEAD)
        winter = pd.date_range("2023-02-01", "2023-03-25", freq="1h", tz="UTC")
        horizon = download_horizon(WIMMIS[0], WIMMIS[1])
        shaded = noisy(synthetic_plant(*WIMMIS, winter, TILT, AZ, KWP, interval="1h",
                                       label="end", horizon=horizon), 2.0)
        aware, aware_ghi = build_pu_power_matrix(*WIMMIS, winter, interval="1h",
                                                 label="end", horizon=horizon)
        blind, blind_ghi = build_pu_power_matrix(*WIMMIS, winter, interval="1h", label="end")
        print(row("horizon in the dictionary",
                  run_estimation(aware, shaded, aware_ghi > DAYTIME_GHI_THRESHOLD)))
        print(row("horizon ignored",
                  run_estimation(blind, shaded, blind_ghi > DAYTIME_GHI_THRESHOLD)))

    print("\nE7  temperature: a plant that heats up, under an ambient series")
    print("    (9-27 °C, mean 18 °C), 2 % noise"); print(HEAD)
    hours = np.asarray(stamps.hour, dtype=float)
    day_of_year = np.asarray(stamps.dayofyear, dtype=float)
    ambient = (14.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
               + 7.0 * np.sin(2 * np.pi * (hours - 9) / 24))
    hot = noisy(make(SITE, stamps, air_temp=ambient), 2.0)
    for name, assumed in (("no temperature model", None),
                          ("constant 25 °C", 25.0),
                          ("constant 20 °C", 20.0),
                          ("the measured series", ambient)):
        warm, warm_ghi = build_pu_power_matrix(*SITE, stamps, interval="1h",
                                               label="end", air_temp=assumed)
        print(row(name, run_estimation(warm, hot, warm_ghi > DAYTIME_GHI_THRESHOLD)))

    print("\nE6  inverter clipping, 100 kWp behind 75 kVA (DC/AC 1.33)")
    print("    A estimates the rating; A1 and B are given it"); print(HEAD)
    for pct in (0.0, 1.0, 2.0, 4.0):
        clipped = noisy(make(SITE, stamps, ac_rating=75.0), pct)
        print(row(f"noise {pct:.0f} %, ignored", run_estimation(matrix, clipped, daytime)))
        print(row(f"noise {pct:.0f} %, method A",
                  run_estimation(matrix, clipped, daytime, method="A")))
        print(row(f"noise {pct:.0f} %, method A1",
                  run_estimation(matrix, clipped, daytime, method="A1", ac_rating=75.0)))
        print(row(f"noise {pct:.0f} %, method B",
                  run_estimation(matrix, clipped, daytime, method="B", ac_rating=75.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
