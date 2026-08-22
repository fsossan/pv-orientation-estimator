"""
Interval integration of the reference matrix — no DB, no network.

Metered power is an average over an interval; evaluating the model at a single
instant instead is a bias, not noise.  These tests cover the machinery that
integrates the model over the measurement interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import build_pu_power_matrix
from pv_orientation_estimator.reference import (
    SUBSAMPLES_PER_INTERVAL,
    _reference_at_instants,
    _subsample_offsets,
)

LAT, LON, ELEV = 46.52, 6.63, 500.0
STAMPS = pd.date_range("2023-06-01 04:00", periods=12, freq="1h", tz="UTC")


def _minutes(offsets):
    return [o.total_seconds() / 60.0 for o in offsets]


def test_offsets_are_subinterval_midpoints():
    # An hour in four 15-minute sub-samples: midpoints at 7.5, 22.5, 37.5, 52.5.
    assert _minutes(_subsample_offsets("1h", "start", "15min")) == [7.5, 22.5, 37.5, 52.5]
    assert _minutes(_subsample_offsets("1h", "centre", "15min")) == [-22.5, -7.5, 7.5, 22.5]
    assert _minutes(_subsample_offsets("1h", "end", "15min")) == [-52.5, -37.5, -22.5, -7.5]
    assert _minutes(_subsample_offsets("1h", "center", "15min")) == \
           _minutes(_subsample_offsets("1h", "centre", "15min"))


def test_default_subsample_splits_the_interval():
    assert len(_subsample_offsets("1h", "centre", None)) == SUBSAMPLES_PER_INTERVAL

    default_matrix, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h")
    explicit_matrix, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h",
                                                subsample="5min")
    assert np.array_equal(default_matrix, explicit_matrix)


def test_instantaneous_build_is_unchanged():
    """interval=None must go through the original code path untouched."""
    new, new_ghi = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)
    old, old_ghi = _reference_at_instants(LAT, LON, ELEV, STAMPS, None, -0.0043, 0.038, 25.0)

    assert np.array_equal(new, old)
    assert np.array_equal(new_ghi, old_ghi)


def test_interval_build_is_the_mean_over_sub_samples():
    offsets = _subsample_offsets("1h", "centre", "15min")
    parts = [build_pu_power_matrix(LAT, LON, ELEV, STAMPS + o) for o in offsets]

    averaged, averaged_ghi = build_pu_power_matrix(
        LAT, LON, ELEV, STAMPS, interval="1h", label="centre", subsample="15min"
    )
    assert averaged == pytest.approx(np.mean([p[0] for p in parts], axis=0))
    assert averaged_ghi == pytest.approx(np.mean([p[1] for p in parts], axis=0))


def test_label_places_the_window():
    """An end-labelled interval covers the same window as start-labelled, one back."""
    ending, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS,
                                       interval="1h", label="end", subsample="15min")
    starting, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS - pd.Timedelta("1h"),
                                         interval="1h", label="start", subsample="15min")
    assert ending == pytest.approx(starting)


def test_interval_build_tracks_the_true_interval_average():
    """
    Against a finely resolved interval average, the interval build must beat
    sampling the model at the timestamp — that gap is the bias this feature
    removes.

    The reference averages 1-minute *midpoints* (:00:30, :01:30, …).  Sampling
    the minute marks instead would itself be a left-endpoint sum, biased by
    half a minute of slope — enough to swamp what is being measured here.
    """
    fine = pd.date_range(STAMPS[0] + pd.Timedelta("30s"),
                         periods=len(STAMPS) * 60, freq="1min", tz="UTC")
    fine_matrix, _ = build_pu_power_matrix(LAT, LON, ELEV, fine)
    truth = fine_matrix.reshape(len(STAMPS), 60, -1).mean(axis=1)

    integrated, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h",
                                           label="start", subsample="5min")
    instantaneous, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS)

    error_integrated = np.abs(integrated - truth).max()
    error_instantaneous = np.abs(instantaneous - truth).max()

    assert error_integrated < 1e-3                     # per-unit: < 1 W/m² of POA
    assert error_instantaneous > 100 * error_integrated


def test_finer_sub_sampling_converges():
    """Midpoint-rule error falls with the sub-step, and is exact when the
    sub-samples land on the reference points."""
    fine = pd.date_range(STAMPS[0] + pd.Timedelta("30s"),
                         periods=len(STAMPS) * 60, freq="1min", tz="UTC")
    fine_matrix, _ = build_pu_power_matrix(LAT, LON, ELEV, fine)
    truth = fine_matrix.reshape(len(STAMPS), 60, -1).mean(axis=1)

    errors = []
    for step in ("15min", "5min", "1min"):
        built, _ = build_pu_power_matrix(LAT, LON, ELEV, STAMPS, interval="1h",
                                          label="start", subsample=step)
        errors.append(np.abs(built - truth).max())

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kwargs, exc, message", [
    ({"interval": "1h", "subsample": "7min"}, ValueError, "divide interval"),
    ({"interval": "1h", "label": "middle"}, ValueError, "label must be one of"),
    ({"interval": "-1h"}, ValueError, "must be positive"),
    ({"interval": "1h", "subsample": "0min"}, ValueError, "must be positive"),
    ({"interval": 60}, TypeError, "must carry a unit"),
    ({"interval": "1h", "subsample": 5}, TypeError, "must carry a unit"),
])
def test_rejects_bad_interval_arguments(kwargs, exc, message):
    with pytest.raises(exc, match=message):
        build_pu_power_matrix(LAT, LON, ELEV, STAMPS, **kwargs)
