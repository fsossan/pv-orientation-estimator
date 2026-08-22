"""
The synthetic demo — no DB, no network, no figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator.demo import demo_estimate, synthetic_plant

LAT, LON, ELEV = 46.52, 6.63, 500.0
STAMPS = pd.date_range("2023-06-01", periods=48, freq="1h", tz="UTC")


def test_synthetic_plant_is_physical():
    power = synthetic_plant(LAT, LON, ELEV, STAMPS, tilt=30, az_eu=0, capacity_kwp=100.0)

    assert power.shape == (len(STAMPS),)
    assert (power >= 0).all()
    assert power.max() < 100.0            # per-unit POA never exceeds 1 kW/kWp here
    assert power[STAMPS.hour == 0].max() == 0.0     # dark at midnight UTC


def test_synthetic_plant_scales_with_capacity():
    small = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 10.0)
    large = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 100.0)
    assert large == pytest.approx(small * 10.0)


def test_interval_averaging_smooths_the_edges():
    """Averaging over the interval must differ from sampling at the timestamp,
    most at sunrise/sunset where the curve bends hardest."""
    instant = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 100.0)
    averaged = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 100.0,
                               interval="1h", label="end", subsample="5min")

    assert not np.allclose(instant, averaged)
    assert np.abs(instant - averaged).max() > 1.0        # kW


def test_noise_is_reproducible_and_daylight_only():
    clean = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 100.0)
    noisy = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 100.0, noise_pct=5.0, seed=1)
    again = synthetic_plant(LAT, LON, ELEV, STAMPS, 30, 0, 100.0, noise_pct=5.0, seed=1)

    assert noisy == pytest.approx(again)                 # seeded
    assert not np.allclose(noisy, clean)
    assert noisy[clean == 0.0] == pytest.approx(0.0)     # noise scales with power


def test_demo_recovers_the_synthetic_plant():
    result = demo_estimate(tilt=30, az_eu=-20, capacity_kwp=100.0,
                           start="2023-05-01", end="2023-06-20",
                           interval="1h", label="end", noise_pct=2.0, plot=False)

    assert result["alpha"] is not None
    assert result["best_tilt"] == 30
    assert result["best_az_eu"] == -20
    assert result["effective_kWp"] == pytest.approx(100.0, rel=0.05)
    assert result["r2"] > 0.99
