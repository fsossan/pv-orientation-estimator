"""
Terrain shading of the per-unit power dictionary — no DB, no network.

The horizon profiles here are synthetic, so nothing reaches out to PVGIS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import (
    build_pu_power_matrix,
    run_estimation,
    shaded_fraction,
    DAYTIME_GHI_THRESHOLD,
    LAYOUTS,
)
from pv_orientation_estimator.demo import synthetic_plant
from pv_orientation_estimator.horizon import (
    beam_blocked,
    HorizonProfile,
    sky_view_factor,
    sky_view_factors,
)

LAT, LON, ELEV = 46.52, 6.63, 500.0
STAMPS = pd.date_range("2023-02-01", periods=48, freq="1h", tz="UTC")


def uniform_horizon(elevation_deg: float) -> HorizonProfile:
    """A skyline at the same elevation in every direction."""
    azimuth = np.arange(-180.0, 180.1, 7.5)
    return HorizonProfile(
        lat=LAT, lon=LON, site_elevation_m=ELEV, source="synthetic",
        azimuth_eu=azimuth,
        elevation_deg=np.full(azimuth.shape, float(elevation_deg)),
    )


def ridge_horizon(peak_deg: float = 25.0, centre_eu: float = -40.0) -> HorizonProfile:
    """A single ridge, tapering away from `centre_eu`."""
    azimuth = np.arange(-180.0, 180.1, 7.5)
    elevation = peak_deg * np.exp(-((azimuth - centre_eu) / 35.0) ** 2)
    return HorizonProfile(
        lat=LAT, lon=LON, site_elevation_m=ELEV, source="synthetic",
        azimuth_eu=azimuth, elevation_deg=elevation,
    )


# --------------------------------------------------------------- geometry


@pytest.mark.parametrize("skyline", [0.0, 10.0, 30.0, 45.0, 60.0, 90.0])
def test_view_factor_of_a_horizontal_plane_is_cos_squared(skyline):
    """
    For a horizontal plane under a uniform skyline at elevation h, the visible
    isotropic sky integrates to cos²h — an exact result to check the numerical
    dome integration against.
    """
    expected = np.cos(np.radians(skyline)) ** 2
    assert sky_view_factor(uniform_horizon(skyline), tilt=0, az_eu=0) == \
        pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("tilt", [0, 15, 30, 45, 60, 75])
def test_open_horizon_sees_the_whole_sky(tilt):
    assert sky_view_factor(uniform_horizon(0.0), tilt, az_eu=-20) == \
        pytest.approx(1.0, abs=1e-3)


def test_view_factor_falls_as_the_skyline_rises():
    profile_low, profile_high = uniform_horizon(10.0), uniform_horizon(40.0)
    assert sky_view_factor(profile_high, 30, 0) < sky_view_factor(profile_low, 30, 0) < 1.0


def test_batch_view_factors_match_the_single_one():
    profile = ridge_horizon()
    layouts = [(0, 0), (30, -20), (75, 45)]
    batch = sky_view_factors(profile, layouts)
    singles = [sky_view_factor(profile, t, a) for t, a in layouts]
    assert batch == pytest.approx(singles)


def test_beam_blocked_compares_against_the_skyline():
    profile = uniform_horizon(20.0)
    assert beam_blocked(profile, 0.0, 10.0)          # sun below the ridge
    assert not beam_blocked(profile, 0.0, 30.0)      # sun above it
    assert beam_blocked(profile, 0.0, -5.0)          # night counts as blocked


# --------------------------------------------------------- the dictionary


def test_open_horizon_leaves_the_dictionary_alone():
    plain, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)
    with_open_sky, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS,
                                             horizon=uniform_horizon(0.0))
    assert with_open_sky == pytest.approx(plain, rel=1e-3, abs=1e-6)


def test_terrain_only_ever_removes_production():
    plain, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)
    shadowed, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, horizon=ridge_horizon())

    assert (shadowed <= plain + 1e-12).all()
    assert shadowed.sum() < 0.95 * plain.sum()


def test_shaded_fraction_is_a_fraction_of_daylight():
    profile = ridge_horizon()
    instant = shaded_fraction(LAT, LON, ELEV, STAMPS, profile)
    averaged = shaded_fraction(LAT, LON, ELEV, STAMPS, profile, interval="1h", label="end")

    assert set(np.unique(instant)) <= {0.0, 1.0}          # a binary test at an instant
    assert ((averaged >= 0.0) & (averaged <= 1.0)).all()
    assert ((averaged > 0.0) & (averaged < 1.0)).any()    # partly shaded intervals exist

    # Deep night carries no shading.  The test deliberately avoids the hours
    # around sunrise: visibility is judged on the *apparent* elevation, which
    # refraction lifts ~0.6° above the true one, so the sun can count as up
    # while the clear-sky model still reports no irradiance.
    deep_night = (STAMPS.hour >= 22) | (STAMPS.hour <= 3)   # February, 46.5°N
    assert averaged[deep_night] == pytest.approx(0.0)


def test_a_wall_shades_every_daylight_sample():
    _, ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)
    walled = shaded_fraction(LAT, LON, ELEV, STAMPS, uniform_horizon(89.0))
    assert walled[ghi > DAYTIME_GHI_THRESHOLD] == pytest.approx(1.0)


# ------------------------------------------------------------- round trip


def test_modelling_the_horizon_beats_ignoring_it():
    """A shaded plant is recovered when the dictionary knows the terrain, and
    mis-estimated when it does not."""
    profile = ridge_horizon(peak_deg=25.0, centre_eu=-40.0)
    stamps = pd.date_range("2023-02-01", "2023-03-25", freq="1h", tz="UTC")
    tilt, az_eu, capacity = 30, -20, 100.0

    measured = synthetic_plant(LAT, LON, ELEV, stamps, tilt, az_eu, capacity,
                               interval="1h", label="end", subsample="5min",
                               horizon=profile)

    aware, ghi = build_pu_power_matrix(LAT, LON, ELEV, stamps, interval="1h",
                                       label="end", subsample="5min", horizon=profile)
    blind, _ = build_pu_power_matrix(LAT, LON, ELEV, stamps, interval="1h",
                                     label="end", subsample="5min")
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    assert daytime.sum() > len(LAYOUTS)          # over-determined

    with_terrain = run_estimation(aware, measured, daytime)
    without = run_estimation(blind, measured, daytime)

    assert with_terrain["best_tilt"] == tilt
    assert with_terrain["best_az_eu"] == az_eu
    assert with_terrain["effective_kWp"] == pytest.approx(capacity, rel=1e-3)
    assert with_terrain["r2"] > without["r2"]
    assert abs(without["effective_kWp"] - capacity) > \
           abs(with_terrain["effective_kWp"] - capacity)
