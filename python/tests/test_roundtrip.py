"""
Synthetic round-trip test (no DB, no network).

Generate "measured" power from a single known orientation using the same
clearsky reference matrix, then verify the estimator recovers that orientation
and capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import (
    LAYOUTS,
    DAYTIME_GHI_THRESHOLD,
    build_pu_power_matrix,
    run_estimation,
    format_results_table,
    alpha_to_heatmap_grid,
)

TIMESTAMPS = pd.date_range("2023-05-01", "2023-07-15", freq="1h", tz="UTC")

# Mid-latitude site (Lausanne, CH) and a long-ish summer window so the daytime
# sample count comfortably exceeds the 320-column grid (identifiability).
LAT, LON, ELEV = 46.52, 6.63, 500.0
TRUE_TILT, TRUE_AZ = 30, 0          # must exist in the grid
TRUE_CAPACITY = 100.0               # kWp


@pytest.fixture(scope="module")
def reference():
    P_pu, ghi = build_pu_power_matrix(LAT, LON, ELEV, TIMESTAMPS)
    return P_pu, ghi


def _true_index() -> int:
    return LAYOUTS.index((TRUE_TILT, TRUE_AZ))


def test_recovers_known_orientation_noiseless(reference):
    P_pu, ghi = reference
    idx = _true_index()
    P_measured = P_pu[:, idx] * TRUE_CAPACITY
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    assert daytime.sum() > 320  # over-determined -> orientation is identifiable

    result = run_estimation(P_pu, P_measured, daytime)

    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["alpha"] is not None
    assert result["r2"] == pytest.approx(1.0, abs=1e-4)
    assert result["best_tilt"] == TRUE_TILT
    assert result["best_az_eu"] == TRUE_AZ
    assert result["effective_kWp"] == pytest.approx(TRUE_CAPACITY, rel=1e-3)


def test_recovers_known_orientation_with_noise(reference):
    P_pu, ghi = reference
    idx = _true_index()
    daytime = ghi > DAYTIME_GHI_THRESHOLD

    clean = P_pu[:, idx] * TRUE_CAPACITY
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.02 * clean.max(), size=clean.shape)
    P_measured = np.clip(clean + noise, 0.0, None)

    result = run_estimation(P_pu, P_measured, daytime)

    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["r2"] > 0.98
    # Orientation recovered within one grid step (±5°) under light noise.
    assert abs(result["best_tilt"] - TRUE_TILT) <= 5
    assert abs(result["best_az_eu"] - TRUE_AZ) <= 5
    assert result["effective_kWp"] == pytest.approx(TRUE_CAPACITY, rel=0.1)


def test_postprocess_shapes(reference):
    P_pu, ghi = reference
    idx = _true_index()
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    result = run_estimation(P_pu, P_pu[:, idx] * TRUE_CAPACITY, daytime)

    df = format_results_table(result["alpha"])
    assert {"tilt_deg", "azimuth_eu", "capacity_kwp", "share_pct"} <= set(df.columns)
    assert (df["share_pct"] <= 100.0).all()

    grid, tilts, azs = alpha_to_heatmap_grid(result["alpha"])
    assert grid.shape == (len(tilts), len(azs))


def test_temperature_correction_derates_reference(reference):
    """With a warm ambient temperature the corrected reference is strictly
    below the uncorrected one wherever there is irradiance (gamma < 0)."""
    P_pu, _ = reference
    P_pu_t, _ = build_pu_power_matrix(LAT, LON, ELEV, TIMESTAMPS, air_temp=25.0)

    lit = P_pu > 0
    # At T_air = 25 °C, T_cell = 25 + 0.038*POA >= 25, so factor <= 1.
    assert np.all(P_pu_t[lit] < P_pu[lit])
    assert np.all(P_pu_t[lit] > 0)
    # Dark samples are unaffected (factor multiplies a zero column).
    assert np.all(P_pu_t[~lit] == 0.0)


def test_recovers_orientation_with_temperature_model():
    """If both the synthetic 'measured' power and the reference use the same
    temperature model, the estimator still recovers orientation and STC kWp."""
    air_temp = 20.0  # constant ambient [°C]
    P_pu, ghi = build_pu_power_matrix(LAT, LON, ELEV, TIMESTAMPS, air_temp=air_temp)

    idx = _true_index()
    P_measured = P_pu[:, idx] * TRUE_CAPACITY  # generated with the temp model
    daytime = ghi > DAYTIME_GHI_THRESHOLD

    result = run_estimation(P_pu, P_measured, daytime)

    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["r2"] == pytest.approx(1.0, abs=1e-4)
    assert result["best_tilt"] == TRUE_TILT
    assert result["best_az_eu"] == TRUE_AZ
    # alpha keeps its meaning as the STC nominal capacity P_nom.
    assert result["effective_kWp"] == pytest.approx(TRUE_CAPACITY, rel=1e-3)
