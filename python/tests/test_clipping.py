"""
Inverter clipping: a plant whose DC capacity exceeds its AC rating.

No DB, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import (
    build_pu_power_matrix,
    run_estimation,
    DAYTIME_GHI_THRESHOLD,
)
from pv_orientation_estimator.demo import synthetic_plant
from pv_orientation_estimator.estimate import detect_ac_rating

LAT, LON, ELEV = 46.52, 6.63, 500.0

from pv_orientation_estimator.grid import LAYOUTS as _LAYOUTS

_ON_GRID = next(l for l in _LAYOUTS if l[1] == 0.0 and 20 < l[0] < 50)

# The planted orientation must be one the grid actually contains: the
# dictionary samples the sphere, so round degrees are not on it. Taken
# from LAYOUTS rather than written out, so it follows the grid if the
# sampling changes.
TILT, AZ = _ON_GRID
KWP, KVA = 100.0, 75.0            # DC/AC = 1.33
STAMPS = pd.date_range("2023-05-01", "2023-06-20", freq="1h", tz="UTC")


@pytest.fixture(scope="module")
def clipped_plant():
    matrix, ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h", label="end")
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    dc = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                         interval="1h", label="end")
    measured = synthetic_plant(LAT, LON, ELEV, STAMPS, TILT, AZ, KWP,
                               interval="1h", label="end", ac_rating=KVA)
    return matrix, ghi, daytime, dc, measured


def test_the_inverter_actually_clips(clipped_plant):
    _, _, daytime, dc, measured = clipped_plant
    assert measured.max() == pytest.approx(KVA)
    assert (measured < dc - 1e-9)[daytime].mean() > 0.3     # a third of the day


@pytest.mark.xfail(reason=
    "the claim was measured on the 5 deg Cartesian grid; the default grid samples the sphere at ~22 deg azimuth / ~11 deg tilt, which is coarser than the effect, so it cannot be demonstrated here. Passes when LAYOUTS is set to RECTANGULAR_LAYOUTS. See doc/algorithm.tex, Why not a Cartesian grid.",
    strict=False)
def test_ignoring_the_clipping_biases_the_fit(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant
    naive = run_estimation(matrix, measured, daytime)

    # The flat top reads as a shallower, more easterly array with less capacity.
    assert naive["best_tilt"] < TILT
    assert naive["effective_kWp"] < KWP


def test_method_a_covers_every_measurement(clipped_plant):
    """Method A's fit is above every sample — that is its defining property."""
    matrix, _, daytime, _, measured = clipped_plant
    result = run_estimation(matrix, measured, daytime, method="A")

    modelled = matrix[daytime] @ result["alpha"]
    assert (modelled >= measured[daytime] - 1e-6).all()
    assert result["method"] == "A"
    assert result["ac_rating_kw"] == pytest.approx(KVA)      # read off, not supplied


def test_method_a_is_exact_on_clean_data(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant     # noiseless by construction
    result = run_estimation(matrix, measured, daytime, method="A")

    assert result["best_tilt"] == TILT
    assert result["best_az_eu"] == AZ
    assert result["effective_kWp"] == pytest.approx(KWP, rel=1e-3)


def test_method_a_inflates_with_noise(clipped_plant):
    """The covering criterion is driven by the extremes, so noise pushes it up."""
    matrix, _, daytime, _, measured = clipped_plant
    rng = np.random.default_rng(0)
    noisy = np.clip(measured * (1 + 0.02 * rng.standard_normal(len(measured))), 0, None)

    clean_fit = run_estimation(matrix, measured, daytime, method="A")
    noisy_fit = run_estimation(matrix, noisy, daytime, method="A")

    assert noisy_fit["effective_kWp"] > clean_fit["effective_kWp"]
    assert noisy_fit["ac_rating_kw"] > KVA          # S becomes the largest reading


def test_method_b_is_steadier_under_noise(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant
    rng = np.random.default_rng(0)
    noisy = np.clip(measured * (1 + 0.02 * rng.standard_normal(len(measured))), 0, None)

    method_a = run_estimation(matrix, noisy, daytime, method="A")
    method_b = run_estimation(matrix, noisy, daytime, method="B", ac_rating=KVA)

    assert abs(method_b["effective_kWp"] - KWP) < abs(method_a["effective_kWp"] - KWP)
    assert method_b["best_tilt"] == TILT


def test_method_a1_uses_the_given_rating(clipped_plant):
    """A1 is Method A with the rating supplied — the fair comparison to B."""
    matrix, _, daytime, _, measured = clipped_plant
    rng = np.random.default_rng(0)
    noisy = np.clip(measured * (1 + 0.02 * rng.standard_normal(len(measured))), 0, None)

    method_a = run_estimation(matrix, noisy, daytime, method="A")
    method_a1 = run_estimation(matrix, noisy, daytime, method="A1", ac_rating=KVA)

    # A has to cover the noisy plateau and reports an inflated rating; A1 is
    # told the rating, so the plateau stops pushing the capacity up.
    assert method_a["ac_rating_kw"] > KVA
    assert method_a1["ac_rating_kw"] == pytest.approx(KVA)
    assert method_a1["effective_kWp"] < method_a["effective_kWp"]

    # Its covering target is the measurement clipped at the rating.
    modelled = matrix[daytime] @ method_a1["alpha"]
    assert (modelled >= np.minimum(noisy[daytime], KVA) - 1e-6).all()


def test_method_validation(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant

    with pytest.raises(ValueError, match="must be 'A', 'A1', 'B' or None"):
        run_estimation(matrix, measured, daytime, method="C")
    with pytest.raises(ValueError, match="needs an ac_rating"):
        run_estimation(matrix, measured, daytime, method="B")
    with pytest.raises(ValueError, match="needs an ac_rating"):
        run_estimation(matrix, measured, daytime, method="A1")

    # A bare ac_rating keeps meaning Method B, as it did before the methods
    # were named.
    assert run_estimation(matrix, measured, daytime, ac_rating=KVA)["method"] == "B"
    assert run_estimation(matrix, measured, daytime)["method"] is None


def test_a_known_rating_recovers_the_dc_capacity(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant
    result = run_estimation(matrix, measured, daytime, ac_rating=KVA)

    assert result["best_tilt"] == TILT
    assert result["best_az_eu"] == AZ
    # Production above the cap is unobservable, so capacity is bounded only
    # weakly from above and lands slightly high.
    assert result["effective_kWp"] == pytest.approx(KWP, rel=0.05)
    assert result["effective_kWp"] > result["ac_rating_kw"]      # kWp > kVA
    assert result["dc_ac_ratio"] == pytest.approx(KWP / KVA, rel=0.05)
    assert result["clipped_share"] > 0.3


def test_clipping_model_beats_ignoring_it(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant
    naive = run_estimation(matrix, measured, daytime)
    aware = run_estimation(matrix, measured, daytime, ac_rating=KVA)

    assert abs(aware["effective_kWp"] - KWP) < abs(naive["effective_kWp"] - KWP)
    assert aware["r2"] > naive["r2"]


def test_no_rating_leaves_the_old_behaviour(clipped_plant):
    """Without ac_rating the result is the plain NNLS, and says so."""
    matrix, _, daytime, _, measured = clipped_plant
    result = run_estimation(matrix, measured, daytime)

    assert result["ac_rating_kw"] is None
    assert result["dc_ac_ratio"] is None
    assert result["clipped_share"] == 0.0


def test_detects_a_clipped_plateau(clipped_plant):
    _, _, daytime, _, measured = clipped_plant
    assert detect_ac_rating(measured[daytime]) == pytest.approx(KVA, rel=0.05)


def test_finds_no_plateau_on_an_unclipped_plant(clipped_plant):
    """An unclipped plant must not have a rating invented for it — that would
    censor its real peak."""
    _, _, daytime, dc, _ = clipped_plant
    rng = np.random.default_rng(0)
    noisy = dc * (1.0 + 0.02 * rng.standard_normal(len(dc)))
    assert detect_ac_rating(noisy[daytime]) is None


def test_auto_matches_a_known_rating(clipped_plant):
    matrix, _, daytime, _, measured = clipped_plant
    auto = run_estimation(matrix, measured, daytime, ac_rating="auto")
    known = run_estimation(matrix, measured, daytime, ac_rating=KVA)

    assert auto["ac_rating_kw"] == pytest.approx(KVA, rel=0.05)
    assert auto["best_tilt"] == known["best_tilt"]
    assert auto["best_az_eu"] == known["best_az_eu"]
