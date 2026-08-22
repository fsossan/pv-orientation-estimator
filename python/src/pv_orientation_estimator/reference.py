"""
Clear-sky per-unit power matrix.

Builds the (T, N_LAYOUTS) matrix whose column j is the per-unit production
[kW/kWp] a unit-capacity array of layout j would deliver at the site under
clear sky — the *dictionary* the estimator decomposes the measurements over.
Per-unit power is the plane-of-array (POA) irradiance over the STC irradiance
of 1000 W/m², computed with pvlib's clear-sky and transposition models.

Optionally applies the empirical temperature derating of Sossan et al.,
"Solar Irradiance Estimations for Modeling the Variability of Photovoltaic
Generation …", Eqs. (6)-(7):

    T_cell,t = T_air,t + kappa * I_t                                   (7)
    P_t      = (I_t / 1000) * [1 + gamma * (T_cell,t - T_ref)] * P_nom (6)

where ``I`` is the POA irradiance [W/m²].  The correction multiplies each
per-unit column by ``[1 + gamma * (T_cell - T_ref)]``, so the fitted ``alpha``
keeps its meaning as the STC nominal capacity ``P_nom`` [kWp].
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import pvlib

from pv_orientation_estimator.grid import AZIMUTHS_PVLIB, LAYOUTS, TILTS_FLOAT
from pv_orientation_estimator.horizon import (
    beam_blocked,
    HorizonProfile,
    sky_view_factors,
)

# Default temperature-model parameters from Sossan et al. (Eqs. 6-7).
GAMMA_DEFAULT = -0.0043       # power temperature coefficient [1/°C], polycrystalline
TEMP_RISE_COEFF_DEFAULT = 0.038  # cell-temperature rise per POA irradiance [°C·m²/W]
TEMP_REF_C = 25.0             # reference (STC) cell temperature [°C]

# Stand-in ambient temperature for sites with no measured series.  Assuming a
# constant ambient still captures the part of the derating that carries
# information — the cells heating with the irradiance itself — see
# build_pu_power_matrix.
TEMP_AIR_STC_C = 25.0

# Sub-samples used to integrate the model over a measurement interval when no
# explicit step is given.  The midpoint rule's error scales with the sub-step,
# so 12 of them puts an hourly interval at 5-minute steps.
SUBSAMPLES_PER_INTERVAL = 12

# With a terrain horizon the integrand contains a step — the beam switches off
# the moment the sun goes behind the skyline — and a step is only resolved as
# finely as it is sampled.  The sun takes about two minutes to clear its own
# diameter, so the interval is then cut into ~1-minute pieces instead.  On
# hourly data, 5-minute steps misplace the shaded fraction by up to 7 % of the
# interval; 1-minute steps by 1 %.
SHADING_SUBSAMPLE = pd.Timedelta(minutes=1)

# Accepted meanings of a timestamp that labels an averaging interval.
INTERVAL_LABELS = ("centre", "center", "start", "end")


def build_pu_power_matrix(
    lat: float,
    lon: float,
    elev: float,
    timestamps: pd.DatetimeIndex,
    air_temp: Optional[Union[float, np.ndarray]] = None,
    gamma: float = GAMMA_DEFAULT,
    temp_rise_coeff: float = TEMP_RISE_COEFF_DEFAULT,
    temp_ref: float = TEMP_REF_C,
    interval: Optional[Union[str, pd.Timedelta]] = None,
    label: str = "centre",
    subsample: Optional[Union[str, pd.Timedelta]] = None,
    horizon: Optional[HorizonProfile] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (P_pu, clearsky_ghi) for the given timestamps and site.

    P_pu shape: (T, N_LAYOUTS)  — per-unit POA irradiance (clearsky / 1000 W/m²)
    clearsky_ghi shape: (T,)    — used to create a daytime mask

    Temperature correction (optional)
    ---------------------------------
    ``air_temp`` selects one of three modes:

    ==========================  ===============================================
    ``air_temp``                what is modelled
    ==========================  ===============================================
    ``None`` (default)          nothing; plain per-unit POA irradiance
    a scalar °C                 cells heating with the irradiance, over a
                                constant ambient — :data:`TEMP_AIR_STC_C` is
                                the stand-in when nothing is known
    an array of length T        the above, over a measured ambient series
    ==========================  ===============================================

    Each column is scaled by the empirical factor of Sossan et al. (Eqs. 6-7)::

        T_cell = air_temp + temp_rise_coeff * POA          # POA in W/m²
        factor = 1 + gamma * (T_cell - temp_ref)

    Defaults (``gamma=-0.0043``, ``temp_rise_coeff=0.038``, ``temp_ref=25``)
    reproduce the paper's polycrystalline / mixed-mounting values.

    The middle mode is worth more than it looks.  With a constant ambient the
    factor becomes ``1 + gamma*(air_temp - temp_ref) + gamma*temp_rise_coeff*POA``:
    a constant plus a term in POA.  The POA term is the one that carries the
    geometry — it flattens the midday peak relative to the shoulders, worth
    −16 % at 1000 W/m² — and it needs no temperature data at all.  Leaving it
    out biases the fitted azimuth by several degrees and the capacity low by
    ~6 % on a plant that really does heat up.

    The constant part is a scaling, which ``alpha`` absorbs, so the assumed
    ambient moves the reported capacity by about 0.43 %/°C and leaves the
    orientation alone.  If the site's mean daytime ambient is known even
    roughly, using it instead of 25 °C removes that offset.

    Interval integration (optional)
    -------------------------------
    By default the model is evaluated **at** each timestamp, which is right for
    instantaneous measurements.  Metered data is almost always an *average over
    an interval* instead, and comparing an interval average against a
    point-in-time model is a bias, not noise — it does not average out with
    more data.

    Pass ``interval`` (e.g. ``"1h"``, ``"15min"``) to integrate the model over
    the same interval the measurements cover, and ``label`` to say what the
    timestamps mean:

    =========== ==========================================
    ``label``   interval covered by a timestamp ``t``
    =========== ==========================================
    ``"centre"`` ``[t - interval/2, t + interval/2)``
    ``"start"``  ``[t, t + interval)``
    ``"end"``    ``(t - interval, t]``
    =========== ==========================================

    The interval is integrated with the midpoint rule over
    ``interval / subsample`` sub-samples (default:
    :data:`SUBSAMPLES_PER_INTERVAL`, or ~1-minute steps once a ``horizon`` is
    given — see :data:`SHADING_SUBSAMPLE`), so the cost is that many times the
    instantaneous build.  An explicit ``subsample`` must divide ``interval``
    exactly.

    Getting ``label`` wrong is the expensive mistake: labelling hourly averages
    with the interval's end, then fitting them, moves the recovered azimuth by
    ~30° while the fit still reports R² > 0.999.  ``air_temp`` is taken as
    constant across an interval — it is not interpolated.

    Terrain shading (optional)
    --------------------------
    Pass a :class:`~pv_orientation_estimator.horizon.HorizonProfile` as
    ``horizon`` to account for the skyline around the site.  Each column then
    becomes::

        POA = (1 - blocked) · POA_beam + SVF · POA_sky_diffuse + POA_ground

    where ``blocked`` is 1 while the terrain hides the sun (see
    :func:`~pv_orientation_estimator.horizon.beam_blocked`) and ``SVF`` is the
    fraction of the sky dome the plane still sees
    (:func:`~pv_orientation_estimator.horizon.sky_view_factor`, one constant
    per orientation).  Ground-reflected irradiance is left alone: the terrain
    in front of the array is shaded too, but that is a small term on a small
    term.

    Blocking is a step in time, so an instantaneous evaluation can only ever
    answer 0 or 1 — and sampling that finer does not help, the error at each
    transition stays O(1).  Combined with ``interval`` it becomes the
    *fraction* of the interval the beam is lost for, which is the quantity
    that actually belongs in an interval average.  **Use the two together.**

    The sky model here is isotropic, so the diffuse has no separate circumsolar
    part to remove along with the beam; under a Perez-type sky it would have to
    follow the beam, since it comes from within ~25° of the solar disc.

    :func:`shaded_fraction` reports how much of each interval was shaded, for
    callers who would rather drop those samples than model them.
    """
    # The view factors are pure geometry: one number per orientation, reused
    # for every timestamp and every sub-sample.
    view_factors = None if horizon is None else sky_view_factors(horizon, LAYOUTS)

    if interval is None:
        return _reference_at_instants(
            lat, lon, elev, timestamps, air_temp, gamma, temp_rise_coeff, temp_ref,
            horizon, view_factors,
        )

    offsets = _subsample_offsets(interval, label, subsample, fine=horizon is not None)
    P_sum: Optional[np.ndarray] = None
    ghi_sum: Optional[np.ndarray] = None
    for offset in offsets:
        P_pu, ghi = _reference_at_instants(
            lat, lon, elev, timestamps + offset,
            air_temp, gamma, temp_rise_coeff, temp_ref,
            horizon, view_factors,
        )
        if P_sum is None:
            P_sum, ghi_sum = P_pu, ghi
        else:
            P_sum += P_pu
            ghi_sum += ghi

    return P_sum / len(offsets), ghi_sum / len(offsets)


def _reference_at_instants(
    lat: float,
    lon: float,
    elev: float,
    timestamps: pd.DatetimeIndex,
    air_temp: Optional[Union[float, np.ndarray]],
    gamma: float,
    temp_rise_coeff: float,
    temp_ref: float,
    horizon: Optional[HorizonProfile] = None,
    view_factors: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit power and clearsky GHI evaluated *at* the given instants."""
    location = pvlib.location.Location(
        latitude=lat, longitude=lon, altitude=elev
    )
    clearsky = location.get_clearsky(timestamps)          # GHI, DNI, DHI  [W/m²]
    solar_pos = location.get_solarposition(timestamps)

    ghi = clearsky["ghi"].values
    dni = clearsky["dni"].values
    dhi = clearsky["dhi"].values
    zen = solar_pos["apparent_zenith"].values
    az  = solar_pos["azimuth"].values

    apply_temp = air_temp is not None
    if apply_temp:
        t_air = np.asarray(air_temp, dtype=float)
        if t_air.ndim == 0:
            t_air = np.full(len(timestamps), float(t_air))
        elif t_air.shape != (len(timestamps),):
            raise ValueError(
                f"air_temp must be a scalar or have length {len(timestamps)}, "
                f"got shape {t_air.shape}"
            )

    blocked = None
    if horizon is not None:
        # EU azimuth (0 = south) is what the horizon profile is indexed by.
        blocked = beam_blocked(horizon, az - 180.0, solar_pos["apparent_elevation"].values)

    columns = []
    for index, (tilt, az_pvlib) in enumerate(zip(TILTS_FLOAT, AZIMUTHS_PVLIB)):
        irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=az_pvlib,
            solar_zenith=zen,
            solar_azimuth=az,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
        )
        if horizon is None:
            poa = irradiance["poa_global"]
        else:
            poa = (
                np.where(blocked, 0.0, irradiance["poa_direct"])       # beam: on or off
                + view_factors[index] * irradiance["poa_sky_diffuse"]  # sky: partly hidden
                + irradiance["poa_ground_diffuse"]                     # ground: untouched
            )
        poa = np.clip(np.asarray(poa, dtype=float), 0.0, None)
        col = poa / 1000.0
        if apply_temp:
            t_cell = t_air + temp_rise_coeff * poa            # Eq. (7)
            col = col * (1.0 + gamma * (t_cell - temp_ref))   # Eq. (6)
        columns.append(col)

    P_pu = np.column_stack(columns)  # (T, N_LAYOUTS)
    return P_pu, ghi


def shaded_fraction(
    lat: float,
    lon: float,
    elev: float,
    timestamps: pd.DatetimeIndex,
    horizon: HorizonProfile,
    interval: Optional[Union[str, pd.Timedelta]] = None,
    label: str = "centre",
    subsample: Optional[Union[str, pd.Timedelta]] = None,
) -> np.ndarray:
    """
    Fraction of each timestamp's interval spent in the terrain's shadow.

    Counts only daylight: a sample is shaded when the sun is above the
    astronomical horizon but behind the skyline.  Night is 0, not 1.

    Without ``interval`` the answer is the 0-or-1 test at each timestamp.  With
    it, the interval is sub-sampled exactly as
    :func:`build_pu_power_matrix` does, so ``0.4`` means the beam was lost for
    40 % of that interval.

    Useful for dropping shaded samples rather than modelling them::

        daytime = ghi > DAYTIME_GHI_THRESHOLD
        usable  = daytime & (shaded_fraction(...) < 0.5)

    which is the cheaper option when the plant model is weak at the low
    irradiance levels that terrain shade leaves behind.
    """
    offsets = ([pd.Timedelta(0)] if interval is None
               else _subsample_offsets(interval, label, subsample, fine=True))
    location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=elev)

    shaded = np.zeros(len(timestamps), dtype=float)
    for offset in offsets:
        solar_pos = location.get_solarposition(timestamps + offset)
        elevation = solar_pos["apparent_elevation"].values
        up = elevation > 0.0
        shaded += up & beam_blocked(horizon, solar_pos["azimuth"].values - 180.0, elevation)

    return shaded / len(offsets)


def build_reference_matrix(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Deprecated alias for :func:`build_pu_power_matrix`.

    The old name said "reference" without saying reference *of what*; the
    matrix holds per-unit power in kW/kWp.
    """
    warnings.warn(
        "build_reference_matrix() is deprecated, use build_pu_power_matrix()",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_pu_power_matrix(*args, **kwargs)


def _subsample_offsets(
    interval: Union[str, pd.Timedelta],
    label: str,
    subsample: Optional[Union[str, pd.Timedelta]],
    fine: bool = False,
) -> list[pd.Timedelta]:
    """
    Offsets from a timestamp to the midpoints of its averaging sub-intervals.

    Midpoints (rather than edges) keep the rule second-order accurate and stop
    interval boundaries from being counted twice.

    ``fine`` picks the default step for a step-shaped integrand — see
    :data:`SHADING_SUBSAMPLE`.  The count is chosen so the step always divides
    the interval exactly, and never falls below
    :data:`SUBSAMPLES_PER_INTERVAL`.
    """
    width = _as_timedelta(interval, "interval")
    if width <= pd.Timedelta(0):
        raise ValueError(f"interval must be positive, got {interval!r}")

    if subsample is None:
        pieces = SUBSAMPLES_PER_INTERVAL
        if fine:
            pieces = max(pieces, round(width / SHADING_SUBSAMPLE))
        step = width / pieces
    else:
        step = _as_timedelta(subsample, "subsample")
    if step <= pd.Timedelta(0):
        raise ValueError(f"subsample must be positive, got {subsample!r}")

    count = width / step
    if abs(count - round(count)) > 1e-9:
        raise ValueError(
            f"subsample ({step}) must divide interval ({width}) exactly, "
            f"got {count:.6g} sub-samples"
        )
    count = round(count)

    if label in ("centre", "center"):
        start = -width / 2
    elif label == "start":
        start = pd.Timedelta(0)
    elif label == "end":
        start = -width
    else:
        raise ValueError(
            f"label must be one of {INTERVAL_LABELS}, got {label!r}"
        )

    return [start + (k + 0.5) * step for k in range(count)]


def _as_timedelta(value: Union[str, pd.Timedelta], name: str) -> pd.Timedelta:
    """
    Parse a duration, rejecting bare numbers.

    ``pd.Timedelta(15)`` is 15 *nanoseconds*, which would silently produce a
    meaningless interval — so a number is an error, not an implied unit.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise TypeError(
            f"{name} must carry a unit, e.g. '15min' or pd.Timedelta(minutes=15); "
            f"got the bare number {value!r}"
        )
    try:
        return pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{name}={value!r} is not a valid duration") from exc
