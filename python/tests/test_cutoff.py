"""
Inverter cut-in: the power below which the system does not start.

Mirror image of clipping — a zero reading is left-censored.  No DB, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import (
    build_pu_power_matrix,
    run_estimation,
    DAYTIME_GHI_THRESHOLD,
    LAYOUTS,
)
from pv_orientation_estimator.demo import synthetic_plant

LAT, LON, ELEV = 46.52, 6.63, 500.0

from pv_orientation_estimator.grid import LAYOUTS as _LAYOUTS

_ON_GRID = next(l for l in _LAYOUTS if l[1] == 0.0 and 20 < l[0] < 50)

# The planted orientation must be one the grid actually contains: the
# dictionary samples the sphere, so round degrees are not on it. Taken
# from LAYOUTS rather than written out, so it follows the grid if the
# sampling changes.
TILT, AZ = _ON_GRID
CUTOFF = 0.5
STAMPS = pd.date_range("2023-05-01", "2023-06-20", freq="1h", tz="UTC")


def centroid(alpha):
    tilts = np.array([t for t, _ in LAYOUTS])
    azimuths = np.array([a for _, a in LAYOUTS])
    total = alpha.sum()
    return (alpha * tilts).sum() / total, (alpha * azimuths).sum() / total


@pytest.fixture(scope="module")
def dictionary():
    matrix, ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS,
                                        interval="1h", label="end")
    return matrix, ghi > DAYTIME_GHI_THRESHOLD


def small_plant(capacity_kwp, with_cutoff=True):
    """Hourly averages of a plant whose inverter sleeps below the cut-in."""
    return synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, capacity_kwp,
                           interval="1h", label="end", subsample="1min",
                           cutoff_kw=CUTOFF if with_cutoff else None)


def test_the_cut_in_truncates_the_shoulders():
    full = small_plant(3.0, with_cutoff=False)
    cut = small_plant(3.0)

    assert (cut <= full + 1e-12).all()          # it can only remove production
    assert cut.sum() < full.sum()
    assert (full[cut == 0.0] > 0.0).any()       # some lit intervals now read zero


def test_a_large_plant_is_untouched(dictionary):
    """With the standard daytime mask, a 0.5 kW cut-in is invisible at 100 kWp."""
    matrix, daytime = dictionary
    result = run_estimation(matrix, small_plant(100.0), daytime)

    tilt_c, az_c = centroid(result["alpha"])
    assert tilt_c == pytest.approx(TILT, abs=0.5)
    assert az_c == pytest.approx(AZ, abs=0.5)


def test_ignoring_it_makes_a_small_plant_look_steeper(dictionary):
    matrix, daytime = dictionary
    ignored = run_estimation(matrix, small_plant(2.0), daytime)

    tilt_c, _ = centroid(ignored["alpha"])
    assert tilt_c > TILT + 10           # truncated shoulders read as a steep array
    assert ignored["effective_kWp"] > 2.0 * 1.05


@pytest.mark.parametrize("capacity", [5.0, 3.0, 2.0])
def test_declaring_the_cut_in_recovers_the_plant(dictionary, capacity):
    matrix, daytime = dictionary
    measured = small_plant(capacity)

    result = run_estimation(matrix, measured, daytime, cutoff_kw=CUTOFF)
    tilt_c, az_c = centroid(result["alpha"])

    assert tilt_c == pytest.approx(TILT, abs=1.0)
    assert az_c == pytest.approx(AZ, abs=1.0)
    assert result["effective_kWp"] == pytest.approx(capacity, rel=0.02)
    assert result["cutoff_kw"] == CUTOFF
    assert 0.0 < result["cutoff_share"] < 1.0


def test_the_band_matters(dictionary):
    """Censoring only the bare cut-in leaves about half the bias: an interval
    straddling it averages above it while still being truncated."""
    matrix, daytime = dictionary
    measured = small_plant(3.0)

    bare = run_estimation(matrix, measured, daytime, cutoff_kw=CUTOFF, cutoff_band=1.0)
    banded = run_estimation(matrix, measured, daytime, cutoff_kw=CUTOFF)

    assert abs(centroid(banded["alpha"])[0] - TILT) < abs(centroid(bare["alpha"])[0] - TILT)


def test_censoring_everything_is_an_error(dictionary):
    matrix, daytime = dictionary
    with pytest.raises(ValueError, match="every daytime sample was censored"):
        run_estimation(matrix, small_plant(2.0), daytime, cutoff_kw=100.0)
