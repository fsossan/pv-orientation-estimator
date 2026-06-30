"""
Orientation grid and shared constants.

Defines the (tilt, azimuth) layout grid the estimator searches over, the
ordering convention used to flatten it into the columns of the reference
matrix, and the pvlib azimuth convention used downstream.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Orientation grid — matches MATLAB clabels convention
# ---------------------------------------------------------------------------

TILTS_DEG: list[int] = list(range(0, 80, 5))          # 0 … 75  (16 values)
AZIMUTHS_EU_DEG: list[int] = list(range(-50, 50, 5))  # -50 … 45  (20 values)

# Flat list of (tilt, azimuth_eu) pairs, same ordering as MATLAB clabels
# (azimuth is the outer loop, tilt the inner loop).
LAYOUTS: list[tuple[int, int]] = [
    (tilt, az) for az in AZIMUTHS_EU_DEG for tilt in TILTS_DEG
]
LAYOUT_LABELS: list[str] = [f"{t},{a}" for t, a in LAYOUTS]
N_LAYOUTS = len(LAYOUTS)  # 320

# pvlib uses North-clockwise azimuth: az_pvlib = 180 + az_eu
AZIMUTHS_PVLIB: list[float] = [float(180 + a) for _, a in LAYOUTS]
TILTS_FLOAT: list[float] = [float(t) for t, _ in LAYOUTS]

# Minimum clearsky GHI (W/m²) to consider a timestep "daytime".
DAYTIME_GHI_THRESHOLD = 50.0
