"""
Offline tests for the PVGIS horizon parser — no network access required.
"""

import numpy as np
import pytest

from pv_orientation_estimator.horizon import parse_horizon_response


PAYLOAD = {
    "inputs": {
        "location": {"latitude": 46.52, "longitude": 6.63, "elevation": 487.0},
        "horizon_db": "DEM-calculated",
    },
    "outputs": {
        "horizon_profile": [
            {"A": -180.0, "H_hor": 4.0},
            {"A": -90.0, "H_hor": 2.0},
            {"A": 0.0, "H_hor": 6.0},
            {"A": 90.0, "H_hor": 3.0},
            {"A": 180.0, "H_hor": 4.0},
        ],
        "winter_solstice": [
            {"A_sun(w)": -90.0, "H_sun(w)": 0.0},
            {"A_sun(w)": 0.0, "H_sun(w)": 20.0},
        ],
    },
}


def test_parse_horizon_response():
    profile = parse_horizon_response(PAYLOAD, lat=46.52, lon=6.63)

    assert profile.lat == pytest.approx(46.52)
    assert profile.site_elevation_m == pytest.approx(487.0)
    assert profile.source == "DEM-calculated"
    assert len(profile.azimuth_eu) == 5
    assert profile.elevation_deg.max() == pytest.approx(6.0)
    assert profile.summer_solstice is None
    assert profile.winter_solstice[1][-1] == pytest.approx(20.0)


def test_dataframe_columns():
    frame = parse_horizon_response(PAYLOAD, lat=46.52, lon=6.63).to_dataframe()
    assert list(frame.columns) == ["azimuth_eu", "elevation_deg"]
    assert len(frame) == 5


def test_elevation_at_interpolates_and_wraps():
    profile = parse_horizon_response(PAYLOAD, lat=46.52, lon=6.63)

    assert profile.elevation_at(0.0) == pytest.approx(6.0)      # on a sample
    assert profile.elevation_at(-45.0) == pytest.approx(4.0)    # midway -90 -> 0
    assert profile.elevation_at(270.0) == pytest.approx(2.0)    # wraps to -90
    assert profile.elevation_at(-180.0) == pytest.approx(
        profile.elevation_at(180.0)                             # duplicated sample
    )

    queried = profile.elevation_at(np.array([0.0, 90.0]))
    assert queried == pytest.approx([6.0, 3.0])


def test_missing_profile_raises():
    with pytest.raises(ValueError, match="horizon_profile"):
        parse_horizon_response({"outputs": {}}, lat=0.0, lon=0.0)
