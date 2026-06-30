"""
Clearsky reference matrix.

Builds the (T, N_LAYOUTS) matrix of per-unit plane-of-array (POA) irradiance
profiles using pvlib's clearsky and transposition models — one column per
(tilt, azimuth) layout in the grid.

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

from typing import Optional, Union

import numpy as np
import pandas as pd
import pvlib

from pv_orientation_estimator.grid import AZIMUTHS_PVLIB, TILTS_FLOAT

# Default temperature-model parameters from Sossan et al. (Eqs. 6-7).
GAMMA_DEFAULT = -0.0043       # power temperature coefficient [1/°C], polycrystalline
TEMP_RISE_COEFF_DEFAULT = 0.038  # cell-temperature rise per POA irradiance [°C·m²/W]
TEMP_REF_C = 25.0             # reference (STC) cell temperature [°C]


def build_reference_matrix(
    lat: float,
    lon: float,
    elev: float,
    timestamps: pd.DatetimeIndex,
    air_temp: Optional[Union[float, np.ndarray]] = None,
    gamma: float = GAMMA_DEFAULT,
    temp_rise_coeff: float = TEMP_RISE_COEFF_DEFAULT,
    temp_ref: float = TEMP_REF_C,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (P_pu, clearsky_ghi) for the given timestamps and site.

    P_pu shape: (T, N_LAYOUTS)  — per-unit POA irradiance (clearsky / 1000 W/m²)
    clearsky_ghi shape: (T,)    — used to create a daytime mask

    Temperature correction (optional)
    ---------------------------------
    If ``air_temp`` is given (a scalar °C, or an array of length T), each column
    is additionally scaled by the empirical temperature factor of Sossan et al.
    (Eqs. 6-7)::

        T_cell = air_temp + temp_rise_coeff * POA          # POA in W/m²
        factor = 1 + gamma * (T_cell - temp_ref)

    Defaults (``gamma=-0.0043``, ``temp_rise_coeff=0.038``, ``temp_ref=25``)
    reproduce the paper's polycrystalline / mixed-mounting values.  When
    ``air_temp`` is ``None`` (default) no correction is applied and the matrix
    is the plain per-unit POA irradiance.
    """
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

    columns = []
    for tilt, az_pvlib in zip(TILTS_FLOAT, AZIMUTHS_PVLIB):
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=az_pvlib,
            solar_zenith=zen,
            solar_azimuth=az,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
        )["poa_global"]
        poa = np.clip(np.asarray(poa, dtype=float), 0.0, None)
        col = poa / 1000.0
        if apply_temp:
            t_cell = t_air + temp_rise_coeff * poa            # Eq. (7)
            col = col * (1.0 + gamma * (t_cell - temp_ref))   # Eq. (6)
        columns.append(col)

    P_pu = np.column_stack(columns)  # (T, N_LAYOUTS)
    return P_pu, ghi
