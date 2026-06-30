"""
pv_orientation_estimator
=================

Estimate the tilt and azimuth (and effective capacity) of a PV array from its
measured AC power.

The method fits the measured AC power to a grid of clearsky plane-of-array
reference profiles (computed with pvlib) and solves a non-negative
least-squares problem (via cvxpy) for the per-orientation capacity ``alpha``.
The dominant peak in ``alpha`` identifies the installed orientation.

Typical use::

    from pv_orientation_estimator import (
        build_reference_matrix, run_estimation, DAYTIME_GHI_THRESHOLD,
    )

    P_pu, clearsky_ghi = build_reference_matrix(lat, lon, elev, timestamps)
    daytime = clearsky_ghi > DAYTIME_GHI_THRESHOLD
    result = run_estimation(P_pu, P_measured, daytime)
    print(result["best_tilt"], result["best_az_eu"], result["effective_kWp"])
"""

from __future__ import annotations

from pv_orientation_estimator.grid import (
    AZIMUTHS_EU_DEG,
    DAYTIME_GHI_THRESHOLD,
    LAYOUT_LABELS,
    LAYOUTS,
    N_LAYOUTS,
    TILTS_DEG,
)
from pv_orientation_estimator.reference import (
    build_reference_matrix,
    GAMMA_DEFAULT,
    TEMP_RISE_COEFF_DEFAULT,
    TEMP_REF_C,
)
from pv_orientation_estimator.estimate import EstimationResult, run_estimation
from pv_orientation_estimator.postprocess import alpha_to_heatmap_grid, format_results_table

__version__ = "0.1.0"

__all__ = [
    # grid
    "TILTS_DEG",
    "AZIMUTHS_EU_DEG",
    "LAYOUTS",
    "LAYOUT_LABELS",
    "N_LAYOUTS",
    "DAYTIME_GHI_THRESHOLD",
    # reference matrix
    "build_reference_matrix",
    "GAMMA_DEFAULT",
    "TEMP_RISE_COEFF_DEFAULT",
    "TEMP_REF_C",
    # estimation
    "run_estimation",
    "EstimationResult",
    # post-processing
    "format_results_table",
    "alpha_to_heatmap_grid",
]
