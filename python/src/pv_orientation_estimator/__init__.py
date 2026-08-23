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
        build_pu_power_matrix, run_estimation, DAYTIME_GHI_THRESHOLD,
    )

    # One stop: measurements in, orientation out.
    result = estimate_orientation(lat, lon, elev, power_series,
                                  interval="1h", label="end")
    print(result["best_tilt"], result["best_az_eu"], result["effective_kWp"])

Or in two stages, when the dictionary is worth reusing::

    P_pu, clearsky_ghi = build_pu_power_matrix(lat, lon, elev, timestamps)
    daytime = clearsky_ghi > DAYTIME_GHI_THRESHOLD
    result = run_estimation(P_pu, P_measured, daytime)
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
    build_pu_power_matrix,
    TEMP_AIR_STC_C,
    build_reference_matrix,
    shaded_fraction,
    GAMMA_DEFAULT,
    TEMP_RISE_COEFF_DEFAULT,
    TEMP_REF_C,
)
from pv_orientation_estimator.estimate import (
    CLIP_BAND,
    CUTOFF_BAND,
    EstimationResult,
    run_estimation,
)
from pv_orientation_estimator.pipeline import estimate_orientation
from pv_orientation_estimator.demo import demo_estimate, plot_estimate, synthetic_plant
from pv_orientation_estimator.geocode import (
    geocode,
    geocode_one,
    Place,
    reverse_geocode,
)
from pv_orientation_estimator.horizon import (
    beam_blocked,
    demo_horizon,
    download_horizon,
    HorizonProfile,
    plot_horizon,
    sky_view_factor,
    sky_view_factors,
)
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
    # per-unit power dictionary
    "build_pu_power_matrix",
    "build_reference_matrix",   # deprecated alias
    "shaded_fraction",
    "GAMMA_DEFAULT",
    "TEMP_AIR_STC_C",
    "TEMP_RISE_COEFF_DEFAULT",
    "TEMP_REF_C",
    # estimation
    "estimate_orientation",
    "run_estimation",
    "EstimationResult",
    "CLIP_BAND",
    "CUTOFF_BAND",
    # post-processing
    "format_results_table",
    "alpha_to_heatmap_grid",
    # synthetic demo
    "demo_estimate",
    "plot_estimate",
    "synthetic_plant",
    # geocoding (swisstopo GeoAdmin)
    "geocode",
    "geocode_one",
    "reverse_geocode",
    "Place",
    # terrain horizon (PVGIS)
    "download_horizon",
    "HorizonProfile",
    "plot_horizon",
    "demo_horizon",
    "beam_blocked",
    "sky_view_factor",
    "sky_view_factors",
]
