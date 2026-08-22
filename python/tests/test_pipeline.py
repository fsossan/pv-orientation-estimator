"""
The one-stop entry point — no DB, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import (
    build_pu_power_matrix,
    estimate_orientation,
    run_estimation,
    DAYTIME_GHI_THRESHOLD,
)
from pv_orientation_estimator.demo import synthetic_plant

LAT, LON, ELEV = 46.52, 6.63, 500.0
TILT, AZ, KWP = 30, -20, 100.0
STAMPS = pd.date_range("2023-05-01", "2023-06-20", freq="1h", tz="UTC")


@pytest.fixture(scope="module")
def measurements():
    power = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                            interval="1h", label="end", noise_pct=2.0)
    return pd.Series(power, index=STAMPS)


def test_matches_the_two_stage_path(measurements):
    """The wrapper must be exactly the three calls it replaces."""
    one_stop = estimate_orientation(LAT, LON, ELEV, measurements,
                                    interval="1h", label="end")

    matrix, ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS,
                                        interval="1h", label="end")
    staged = run_estimation(matrix, measurements.to_numpy(),
                            ghi > DAYTIME_GHI_THRESHOLD)

    assert one_stop["alpha"] == pytest.approx(staged["alpha"])
    assert one_stop["best_tilt"] == staged["best_tilt"] == TILT
    assert one_stop["best_az_eu"] == staged["best_az_eu"] == AZ


def test_accepts_a_series_or_an_array(measurements):
    from_series = estimate_orientation(LAT, LON, ELEV, measurements,
                                       interval="1h", label="end")
    from_array = estimate_orientation(LAT, LON, ELEV, measurements.to_numpy(), STAMPS,
                                      interval="1h", label="end")
    assert from_series["alpha"] == pytest.approx(from_array["alpha"])


def test_clipping_arguments_pass_through():
    clipped = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                              interval="1h", label="end", ac_rating=75.0)
    series = pd.Series(clipped, index=STAMPS)

    method_a = estimate_orientation(LAT, LON, ELEV, series, interval="1h",
                                    label="end", method="A")
    method_b = estimate_orientation(LAT, LON, ELEV, series, interval="1h",
                                    label="end", method="B", ac_rating=75.0)

    assert method_a["method"] == "A"
    assert method_b["method"] == "B"
    assert method_a["effective_kWp"] == pytest.approx(KWP, rel=1e-3)
    assert method_b["ac_rating_kw"] == pytest.approx(75.0)


@pytest.mark.parametrize("power, stamps, kwargs, exc, message", [
    (np.zeros(5), None, {}, TypeError, "timestamps are required"),
    (np.zeros(5), STAMPS, {}, ValueError, "5 samples but there are"),
    (np.zeros(len(STAMPS)), STAMPS.tz_localize(None), {}, ValueError, "timezone-aware"),
    (np.zeros(len(STAMPS)), STAMPS, {"max_shaded_fraction": 0.5}, ValueError,
     "needs a horizon"),
])
def test_rejects_inconsistent_input(power, stamps, kwargs, exc, message):
    with pytest.raises(exc, match=message):
        estimate_orientation(LAT, LON, ELEV, power, stamps, **kwargs)
