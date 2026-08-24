"""The grid exists to be well conditioned; these pin that it still is.

The Cartesian tilt x azimuth product that preceded it is a lat/lon grid on a
sphere. It clusters at the pole, and the pole is a horizontal panel whose plane
does not depend on azimuth at all, so every azimuth at tilt 0 names the same
plane. The resulting dictionary was singular to double precision, which made the
recovered alpha a property of the solver rather than of the data. These tests
fail if that regresses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_orientation_estimator import build_pu_power_matrix
from pv_orientation_estimator.grid import (
    AZIMUTH_RANGE,
    LAYOUTS,
    N_LAYOUTS,
    RECTANGULAR_LAYOUTS,
    TILT_RANGE,
    deserno_layouts,
)

LAT, LON, ELEV = 46.27, 7.87, 1291.0


@pytest.fixture(scope="module")
def timestamps() -> pd.DatetimeIndex:
    # A fortnight around the equinox: enough sun geometry to separate planes.
    return pd.date_range("2026-03-15", "2026-03-29", freq="15min", tz="UTC")


def _dictionary(layouts, timestamps):
    import pvlib

    site = pvlib.location.Location(LAT, LON, altitude=ELEV)
    clearsky = site.get_clearsky(timestamps)
    solar = site.get_solarposition(timestamps)
    columns = []
    for tilt, azimuth in layouts:
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=float(tilt),
            surface_azimuth=float(180 + azimuth),
            solar_zenith=solar["apparent_zenith"].values,
            solar_azimuth=solar["azimuth"].values,
            dni=clearsky["dni"].values,
            ghi=clearsky["ghi"].values,
            dhi=clearsky["dhi"].values,
        )["poa_global"]
        columns.append(np.clip(np.asarray(poa, dtype=float), 0.0, None) / 1000.0)
    matrix = np.column_stack(columns)
    return matrix[matrix.max(axis=1) > 0.05]        # daytime rows only


def test_no_two_layouts_name_the_same_plane():
    assert len(set(LAYOUTS)) == N_LAYOUTS


def test_exactly_one_horizontal_candidate():
    """At tilt 0 the azimuth is meaningless, so more than one is a duplicate."""
    flat = [layout for layout in LAYOUTS if layout[0] == 0.0]
    assert flat == [(0.0, 0.0)]

    # the grid this replaced had twenty of them
    rectangular_flat = [l for l in RECTANGULAR_LAYOUTS if l[0] == 0.0]
    assert len(rectangular_flat) == 20


def test_rectangular_flat_columns_really_are_identical(timestamps):
    matrix = _dictionary([l for l in RECTANGULAR_LAYOUTS if l[0] == 0.0], timestamps)
    assert np.array_equal(matrix, np.repeat(matrix[:, :1], matrix.shape[1], axis=1))


def test_conditioning_beats_the_rectangular_grid_by_orders_of_magnitude(timestamps):
    deserno = np.linalg.cond(_dictionary(LAYOUTS, timestamps))
    rectangular = np.linalg.cond(_dictionary(RECTANGULAR_LAYOUTS, timestamps))

    # Measured at ~1e7 against ~1e23. The margin asked for is deliberately loose:
    # the point is the sixteen orders of magnitude between them, not the digits.
    # The absolute bound tracks DEFAULT_TARGET — a finer sampling conditions
    # worse, which is the trade DEFAULT_TARGET is chosen to sit at the edge of.
    assert deserno < 1e9
    assert rectangular / deserno > 1e12


def test_dictionary_has_full_column_rank(timestamps):
    matrix = _dictionary(LAYOUTS, timestamps)
    assert np.linalg.matrix_rank(matrix) == N_LAYOUTS


def test_ranges_describe_what_is_searched():
    assert TILT_RANGE == (min(t for t, _ in LAYOUTS), max(t for t, _ in LAYOUTS))
    assert AZIMUTH_RANGE == (min(a for _, a in LAYOUTS), max(a for _, a in LAYOUTS))


def test_rings_thin_towards_horizontal():
    """Points per ring should follow sin(tilt) — that is the whole mechanism."""
    per_ring: dict[float, int] = {}
    for tilt, _ in LAYOUTS:
        per_ring[tilt] = per_ring.get(tilt, 0) + 1
    counts = [per_ring[t] for t in sorted(per_ring)]
    assert counts == sorted(counts)          # monotonically non-decreasing
    assert counts[0] == 1                    # the horizontal plate


def test_target_count_is_approximately_honoured():
    for target in (10, 21, 40, 80):
        layouts = deserno_layouts(target)
        assert target * 0.7 <= len(layouts) <= target * 1.4


def test_build_pu_power_matrix_follows_the_grid(timestamps):
    P_pu, ghi = build_pu_power_matrix(LAT, LON, ELEV, timestamps)
    assert P_pu.shape == (len(timestamps), N_LAYOUTS)
    assert ghi.shape == (len(timestamps),)
