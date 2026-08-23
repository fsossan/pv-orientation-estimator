"""
One-stop estimation: measurements in, orientation out.

:func:`estimate_orientation` builds the per-unit power dictionary for the site,
derives the daytime mask, and solves the fit — the three steps that every
caller otherwise repeats:

    P_pu, ghi = build_pu_power_matrix(lat, lon, elev, timestamps, ...)
    daytime   = ghi > DAYTIME_GHI_THRESHOLD
    result    = run_estimation(P_pu, P_measured, daytime, ...)

Beyond saving the boilerplate it removes a real trap: ``interval`` and
``label`` describe *the measurements*, and the dictionary has to be integrated
the same way.  Split across two calls, nothing checks that they agree, and a
mismatch is a bias of tens of degrees that the fit statistics do not reveal
(see :mod:`pv_orientation_estimator.reference`).  Given once, they cannot
disagree.

The two-stage API stays available and is the right one when a dictionary is
reused — fitting several plants at one site, or the same plant under several
clipping models — since building it is much more expensive than solving.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from pv_orientation_estimator.estimate import (
    CLIP_BAND,
    CUTOFF_BAND,
    EstimationResult,
    run_estimation,
)
from pv_orientation_estimator.grid import DAYTIME_GHI_THRESHOLD
from pv_orientation_estimator.horizon import HorizonProfile
from pv_orientation_estimator.reference import (
    GAMMA_DEFAULT,
    TEMP_REF_C,
    TEMP_RISE_COEFF_DEFAULT,
    build_pu_power_matrix,
    shaded_fraction,
)


def estimate_orientation(
    lat: float,
    lon: float,
    elev: float,
    power: Union[pd.Series, np.ndarray],
    timestamps: Optional[pd.DatetimeIndex] = None,
    *,
    # how the measurements are stamped — applied to the dictionary as well
    interval: Optional[Union[str, pd.Timedelta]] = None,
    label: str = "centre",
    subsample: Optional[Union[str, pd.Timedelta]] = None,
    # site and plant model
    horizon: Optional[HorizonProfile] = None,
    air_temp: Optional[Union[float, np.ndarray]] = None,
    gamma: float = GAMMA_DEFAULT,
    temp_rise_coeff: float = TEMP_RISE_COEFF_DEFAULT,
    temp_ref: float = TEMP_REF_C,
    # which samples to fit
    daytime_threshold: float = DAYTIME_GHI_THRESHOLD,
    max_shaded_fraction: Optional[float] = None,
    # inverter clipping
    method: Optional[str] = None,
    ac_rating: Optional[Union[float, str]] = None,
    clip_band: float = CLIP_BAND,
    cutoff_kw: Optional[float] = None,
    cutoff_band: float = CUTOFF_BAND,
) -> EstimationResult:
    """
    Estimate a plant's tilt, azimuth and capacity from its measured AC power.

    ``power`` is the measured AC power in kW: a pandas Series, whose index
    supplies the timestamps, or an array together with ``timestamps``.

    ``interval`` and ``label`` say what the timestamps mean — see
    :func:`~pv_orientation_estimator.reference.build_pu_power_matrix`.  Metered
    data is an average over an interval, so passing them is the normal case;
    omitting them declares the readings to be instantaneous.

    ``air_temp`` selects the temperature mode: ``None`` for no modelling, a
    scalar °C for cells heating over a constant ambient (pass
    :data:`~pv_orientation_estimator.reference.TEMP_AIR_STC_C` when nothing is
    known — it is much better than leaving it out), or a measured series.

    ``horizon`` accounts for the terrain skyline, and ``max_shaded_fraction``
    additionally drops samples the terrain shaded for more than that fraction
    of their interval — the alternative to modelling them, useful where the
    plant model is weak at the low irradiance shade leaves behind.

    ``method``, ``ac_rating`` and ``clip_band`` handle a plant whose DC
    capacity exceeds its inverter rating, and ``cutoff_kw`` the power below
    which the inverter does not start; see
    :func:`~pv_orientation_estimator.estimate.run_estimation`.

    Returns the same :class:`~pv_orientation_estimator.estimate.EstimationResult`
    as the two-stage path, so every post-processing helper applies unchanged.
    """
    power, timestamps = _as_series(power, timestamps)

    P_pu, clearsky_ghi = build_pu_power_matrix(
        lat, lon, elev, timestamps,
        air_temp=air_temp, gamma=gamma, temp_rise_coeff=temp_rise_coeff,
        temp_ref=temp_ref, interval=interval, label=label, subsample=subsample,
        horizon=horizon,
    )

    usable = clearsky_ghi > daytime_threshold
    if max_shaded_fraction is not None:
        if horizon is None:
            raise ValueError("max_shaded_fraction needs a horizon to be given")
        shaded = shaded_fraction(lat, lon, elev, timestamps, horizon,
                                 interval=interval, label=label, subsample=subsample)
        usable = usable & (shaded <= max_shaded_fraction)

    return run_estimation(P_pu, power, usable, method=method,
                          ac_rating=ac_rating, clip_band=clip_band,
                          cutoff_kw=cutoff_kw, cutoff_band=cutoff_band)


def _as_series(
    power: Union[pd.Series, np.ndarray],
    timestamps: Optional[pd.DatetimeIndex],
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Accept a Series (index = timestamps) or an array plus timestamps."""
    if isinstance(power, pd.Series):
        if timestamps is None:
            timestamps = pd.DatetimeIndex(power.index)
        power = power.to_numpy()

    if timestamps is None:
        raise TypeError(
            "timestamps are required unless power is a pandas Series indexed by them"
        )

    power = np.asarray(power, dtype=float)
    timestamps = pd.DatetimeIndex(timestamps)
    if len(power) != len(timestamps):
        raise ValueError(
            f"power has {len(power)} samples but there are {len(timestamps)} timestamps"
        )
    if timestamps.tz is None:
        raise ValueError(
            "timestamps must be timezone-aware (UTC), otherwise the solar "
            "position — and with it the estimated azimuth — is undefined"
        )
    return power, timestamps
