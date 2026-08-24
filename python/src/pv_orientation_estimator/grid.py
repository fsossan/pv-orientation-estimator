"""
Orientation grid and shared constants.

Defines the (tilt, azimuth) layout grid the estimator searches over, the
ordering convention used to flatten it into the columns of the reference
matrix, and the pvlib azimuth convention used downstream.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Orientation grid
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The rectangular grid — kept for comparison, and for heatmap axes
# ---------------------------------------------------------------------------
# The Cartesian product of a tilt list and an azimuth list is the obvious way to
# enumerate orientations and the wrong one. It is a lat/lon grid on a sphere: it
# clusters at the pole, and the pole here is a horizontal panel, whose plane does
# not depend on azimuth at all. Every azimuth at tilt 0 is *the same plane*, so
# the 320 columns contain 20 bit-identical copies of the flat plate and, at every
# low tilt, near-copies of each other. Measured on a real plant the resulting
# dictionary has a condition number of order 1e23 — singular to double precision
# — and an effective rank near 20.

TILTS_DEG: list[int] = list(range(0, 80, 5))          # 0 … 75  (16 values)
AZIMUTHS_EU_DEG: list[int] = list(range(-50, 50, 5))  # -50 … 45  (20 values)

RECTANGULAR_LAYOUTS: list[tuple[float, float]] = [
    (float(tilt), float(az)) for az in AZIMUTHS_EU_DEG for tilt in TILTS_DEG
]


# ---------------------------------------------------------------------------
# The default grid — equidistributed over the south-facing hemisphere
# ---------------------------------------------------------------------------

#: Target number of planes. 80 (giving 82) sits at the identifiability ceiling:
#: it is the largest sampling that still yields a full-column-rank dictionary on
#: real data. Asking for more does not buy resolution, it buys rank deficiency —
#: at 161 planes the rank is 151, at 321 it is 272. The data supports about
#: \ang{22} of azimuth and \ang{11} of tilt, and no sampling scheme changes that.
DEFAULT_TARGET = 80


def deserno_layouts(
    n_target: int = DEFAULT_TARGET,
    tilt_max: float = 90.0,
    azimuth_half_span: float = 90.0,
    include_flat: bool = True,
) -> list[tuple[float, float]]:
    """
    Orientations spread evenly over the south-facing part of the sphere.

    Follows Deserno (2004), *How to generate equidistributed points on the
    surface of a sphere*: lay out rings of constant polar angle and give each
    ring a number of points proportional to ``sin`` of that angle, so the
    spacing between neighbours is roughly the same everywhere.

    Here the polar angle is the surface tilt, so the rings thin out towards
    horizontal exactly where azimuth stops meaning anything — a flat panel gets
    one candidate, not one per azimuth. That is the whole point: the redundancy
    the rectangular grid has to be cleaned up after is never built.

    ``include_flat`` prepends an explicit ``(0, 0)``. Deserno's ring centres sit
    at half-steps and so skip the pole, and a horizontal array is a real
    configuration — on the plant this was developed against it carries most of
    the fitted capacity.

    The result is the sampling the source paper prescribes (Sossan et al., 2019,
    §III-B: "J = 21 tilted planes with tilt and azimuth values equally spaced on
    a south-facing semi-sphere"), rather than the Cartesian product above.
    """
    tilt_max_rad = math.radians(tilt_max)
    span_rad = math.radians(2.0 * azimuth_half_span)

    # Solid angle of the region, and the area each point should own.
    omega = span_rad * (1.0 - math.cos(tilt_max_rad))
    area = omega / n_target
    spacing = math.sqrt(area)

    n_rings = max(1, round(tilt_max_rad / spacing))
    d_tilt = tilt_max_rad / n_rings
    d_azimuth = area / d_tilt

    layouts: list[tuple[float, float]] = [(0.0, 0.0)] if include_flat else []
    for ring in range(n_rings):
        tilt = tilt_max_rad * (ring + 0.5) / n_rings
        per_ring = max(1, round(span_rad * math.sin(tilt) / d_azimuth))
        for k in range(per_ring):
            azimuth = -azimuth_half_span + 2.0 * azimuth_half_span * (k + 0.5) / per_ring
            layouts.append((round(math.degrees(tilt), 1), round(azimuth, 1)))
    return layouts


#: Candidate orientations the estimator searches. 23 planes: one horizontal,
#: then rings at roughly 11, 34, 56 and 79 degrees carrying 2, 5, 7 and 8
#: azimuths. Fits as well as the 320-column rectangular grid on real data
#: (marginally better, in fact) at a condition number twenty orders of
#: magnitude smaller, which is what makes the recovered alpha reproducible
#: instead of a property of whichever solver ran.
LAYOUTS: list[tuple[float, float]] = deserno_layouts(DEFAULT_TARGET)

LAYOUT_LABELS: list[str] = [f"{t:g},{a:g}" for t, a in LAYOUTS]
N_LAYOUTS = len(LAYOUTS)

# pvlib uses North-clockwise azimuth: az_pvlib = 180 + az_eu
AZIMUTHS_PVLIB: list[float] = [float(180 + a) for _, a in LAYOUTS]
TILTS_FLOAT: list[float] = [float(t) for t, _ in LAYOUTS]

#: Extent of the search, read off LAYOUTS rather than assumed — a caller
#: checking whether a recorded orientation is reachable must not consult the
#: rectangular lists, which no longer describe what is searched.
TILT_RANGE: tuple[float, float] = (min(t for t, _ in LAYOUTS), max(t for t, _ in LAYOUTS))
AZIMUTH_RANGE: tuple[float, float] = (min(a for _, a in LAYOUTS), max(a for _, a in LAYOUTS))

# Minimum clearsky GHI (W/m²) to consider a timestep "daytime".
DAYTIME_GHI_THRESHOLD = 50.0
