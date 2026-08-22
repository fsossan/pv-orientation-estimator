"""
The three temperature modes — no DB, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import (
    build_pu_power_matrix,
    run_estimation,
    DAYTIME_GHI_THRESHOLD,
    GAMMA_DEFAULT,
    LAYOUTS,
    TEMP_AIR_STC_C,
    TEMP_RISE_COEFF_DEFAULT,
)
from pv_orientation_estimator.demo import synthetic_plant

LAT, LON, ELEV = 46.52, 6.63, 500.0
TILT, AZ, KWP = 30, -20, 100.0
STAMPS = pd.date_range("2023-05-01", "2023-06-20", freq="1h", tz="UTC")


def ambient_series(stamps=STAMPS):
    """A plausible ambient: seasonal ramp plus diurnal swing, 9–27 °C."""
    hours = np.asarray(stamps.hour, dtype=float)
    day_of_year = np.asarray(stamps.dayofyear, dtype=float)
    return (14.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
            + 7.0 * np.sin(2 * np.pi * (hours - 9) / 24))


def centroid(alpha):
    tilts = np.array([t for t, _ in LAYOUTS])
    azimuths = np.array([a for _, a in LAYOUTS])
    total = alpha.sum()
    return (alpha * tilts).sum() / total, (alpha * azimuths).sum() / total


def test_constant_ambient_derates_with_irradiance():
    """With a constant ambient the factor is 1 + γ(T_air−25) + γκ·POA."""
    plain, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)
    warm, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, air_temp=TEMP_AIR_STC_C)

    lit = plain > 0
    ratio = warm[lit] / plain[lit]
    poa = plain[lit] * 1000.0                       # per-unit back to W/m²
    expected = 1.0 + GAMMA_DEFAULT * TEMP_RISE_COEFF_DEFAULT * poa
    assert ratio == pytest.approx(expected, rel=1e-6)

    # ~16 % down at STC irradiance, and nothing at all in the dark
    near_stc = np.abs(poa - 1000.0) < 20.0
    if near_stc.any():
        assert ratio[near_stc].mean() == pytest.approx(0.837, abs=0.01)


def test_the_three_modes_are_distinct():
    none, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)
    constant, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, air_temp=TEMP_AIR_STC_C)
    series, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, air_temp=ambient_series())

    assert not np.allclose(none, constant)
    assert not np.allclose(constant, series)
    assert (constant <= none + 1e-12).all()          # 25 °C only ever derates


def test_a_constant_ambient_recovers_the_geometry_of_a_hot_plant():
    """
    The point of the middle mode: a plant that heats up is fitted with the
    right orientation even though no temperature was measured.
    """
    measured = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                               interval="1h", label="end",
                               air_temp=ambient_series(), noise_pct=2.0)

    cold, cold_ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS,
                                           interval="1h", label="end")
    warm, warm_ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h",
                                           label="end", air_temp=TEMP_AIR_STC_C)

    ignored = run_estimation(cold, measured, cold_ghi > DAYTIME_GHI_THRESHOLD)
    modelled = run_estimation(warm, measured, warm_ghi > DAYTIME_GHI_THRESHOLD)

    _, az_ignored = centroid(ignored["alpha"])
    _, az_modelled = centroid(modelled["alpha"])

    assert abs(az_modelled - AZ) < 2.0               # within 2° of the truth
    assert abs(az_ignored - AZ) > 5.0                # biased without the model
    assert abs(modelled["effective_kWp"] - KWP) < abs(ignored["effective_kWp"] - KWP)


def test_the_assumed_constant_moves_capacity_not_geometry():
    """The constant part of the factor is a scaling, which alpha absorbs."""
    measured = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                               interval="1h", label="end",
                               air_temp=ambient_series(), noise_pct=2.0)

    capacities, azimuths = [], []
    for assumed in (15.0, 20.0, 25.0):
        matrix, ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h",
                                            label="end", air_temp=assumed)
        result = run_estimation(matrix, measured, ghi > DAYTIME_GHI_THRESHOLD)
        capacities.append(result["effective_kWp"])
        azimuths.append(centroid(result["alpha"])[1])

    # ~0.43 %/°C on the capacity, and the azimuth barely moves
    assert capacities[2] > capacities[1] > capacities[0]
    assert max(azimuths) - min(azimuths) < 1.0


def test_matched_models_round_trip_exactly():
    for air in (None, TEMP_AIR_STC_C, ambient_series()):
        matrix, ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS,
                                            interval="1h", label="end", air_temp=air)
        measured = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                                   interval="1h", label="end", air_temp=air)
        result = run_estimation(matrix, measured, ghi > DAYTIME_GHI_THRESHOLD)

        assert result["best_tilt"] == TILT
        assert result["best_az_eu"] == AZ
        assert result["effective_kWp"] == pytest.approx(KWP, rel=2e-3)
