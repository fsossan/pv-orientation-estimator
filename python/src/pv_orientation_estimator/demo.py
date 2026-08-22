"""
End-to-end demo of the orientation estimator on synthetic data.

Synthesises the AC power of a plant whose geometry we know, hands it to the
estimator, and checks what comes back.  The synthetic plant is deliberately
*not* generated from the reference matrix the estimator fits:

* its power is integrated over each metering interval from 1-minute samples,
  the way a real meter averages, rather than sampled at the timestamp;
* it carries measurement noise.

So the demo exercises the same mismatch real data has, instead of the
self-fulfilling round trip of ``tests/test_roundtrip.py``.

Pass a ``horizon`` and the synthetic plant is shaded by the terrain, exactly as
the estimator's dictionary is — plus the demo then refits with a horizon-blind
dictionary, to show what ignoring the skyline costs.

Typical use::

    from pv_orientation_estimator.demo import demo_estimate

    result = demo_estimate()                      # print + plot
    result = demo_estimate(tilt=15, az_eu=30)     # a different truth

    from pv_orientation_estimator.horizon import download_horizon
    demo_estimate(lat=46.6777, lon=7.6409, elev=639.0,      # Wimmis BE
                  horizon=download_horizon(46.6777, 7.6409))
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import pvlib

from pv_orientation_estimator.estimate import EstimationResult, run_estimation
from pv_orientation_estimator.grid import (
    AZIMUTHS_EU_DEG,
    DAYTIME_GHI_THRESHOLD,
    TILTS_DEG,
)
from pv_orientation_estimator.horizon import (
    beam_blocked,
    HorizonProfile,
    sky_view_factor,
)
from pv_orientation_estimator.postprocess import alpha_to_heatmap_grid, format_results_table
from pv_orientation_estimator.reference import (
    GAMMA_DEFAULT,
    TEMP_REF_C,
    TEMP_RISE_COEFF_DEFAULT,
    _subsample_offsets,
    build_pu_power_matrix,
    shaded_fraction,
)

# Open-horizon site on the Swiss plateau — same one the round-trip test uses.
DEMO_LAT, DEMO_LON, DEMO_ELEV = 46.52, 6.63, 500.0


def synthetic_plant(
    lat: float,
    lon: float,
    elev: float,
    timestamps: pd.DatetimeIndex,
    tilt: float,
    az_eu: float,
    capacity_kwp: float,
    interval: Optional[str] = None,
    label: str = "centre",
    subsample: str = "1min",
    noise_pct: float = 0.0,
    seed: int = 0,
    horizon: Optional[HorizonProfile] = None,
    ac_rating: Optional[float] = None,
    air_temp: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """
    Clear-sky AC power [kW] of one plant, as a meter would record it.

    With ``interval`` the plant is sampled every ``subsample`` across each
    interval and averaged, which is what an energy meter reports; without it
    the power is evaluated at the timestamps themselves.

    ``noise_pct`` adds zero-mean Gaussian noise proportional to the
    instantaneous power, so it vanishes at night instead of making the dark
    hours informative.

    With a ``horizon``, the terrain blocks the beam and hides part of the sky
    dome.  Because the shading is resolved at ``subsample`` and then averaged,
    an interval the sun leaves mid-way through comes out partly shaded, the way
    a real one does.

    ``ac_rating`` [kW] puts an inverter in front of the meter: DC production
    above it is cut off, so a plant with ``capacity_kwp > ac_rating`` shows the
    flat top of a clipped array.  Clipping is applied before the noise, since
    the meter sees the clipped signal.

    ``air_temp`` [°C] makes the cells heat up and derate, following the same
    Eqs. (6)-(7) the dictionary uses — a scalar or a series of length T.  Real
    arrays always do this, so it is the way to generate measurements the plain
    dictionary does *not* explain.
    """
    view_factor = None if horizon is None else sky_view_factor(horizon, tilt, az_eu)
    args = (tilt, az_eu, capacity_kwp, horizon, view_factor, air_temp)

    if interval is None:
        power = _plant_power(lat, lon, elev, timestamps, *args)
    else:
        offsets = _subsample_offsets(interval, label, subsample)
        power = np.mean(
            [_plant_power(lat, lon, elev, timestamps + o, *args) for o in offsets],
            axis=0,
        )

    if ac_rating is not None:
        power = np.minimum(power, float(ac_rating))

    if noise_pct:
        rng = np.random.default_rng(seed)
        power = power * (1.0 + noise_pct / 100.0 * rng.standard_normal(len(power)))

    return np.clip(power, 0.0, None)


def demo_estimate(
    lat: float = DEMO_LAT,
    lon: float = DEMO_LON,
    elev: float = DEMO_ELEV,
    tilt: int = 30,
    az_eu: int = -20,
    capacity_kwp: float = 100.0,
    start: str = "2023-05-01",
    end: str = "2023-06-20",
    interval: str = "1h",
    label: str = "end",
    noise_pct: float = 2.0,
    seed: int = 0,
    horizon: Optional[HorizonProfile] = None,
    ac_rating: Optional[float] = None,
    method: Optional[str] = None,
    plot: bool = True,
    show: bool = True,
    savefig: Optional[str] = None,
) -> EstimationResult:
    """
    Synthesise a plant of known geometry, estimate it back, and report.

    Defaults recreate the awkward real-world case: hourly energy readings
    stamped with the *end* of each interval, 2 % noise.  Returns the
    :class:`~pv_orientation_estimator.estimate.EstimationResult`.

    With a ``horizon``, the plant is shaded by the terrain and so is the
    dictionary — and the same measurements are additionally fitted with a
    horizon-blind dictionary, to price the omission.

    ``ac_rating`` [kW] puts the plant behind a smaller inverter, so its DC
    capacity exceeds its AC rating and the measurements are clipped.  Both
    clipping methods are then reported side by side against a fit that ignores
    the inverter; ``method`` ("A", "A1" or "B", default "B") picks which one
    the returned result and the plot come from.
    """
    timestamps = pd.date_range(start, end, freq=interval, tz="UTC")

    measured = synthetic_plant(
        lat, lon, elev, timestamps, tilt, az_eu, capacity_kwp,
        interval=interval, label=label, noise_pct=noise_pct, seed=seed,
        horizon=horizon, ac_rating=ac_rating,
    )
    P_pu, ghi = build_pu_power_matrix(
        lat, lon, elev, timestamps, interval=interval, label=label, horizon=horizon,
    )
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    result = run_estimation(P_pu, measured, daytime, method=method,
                            ac_rating=ac_rating)

    shaded = (None if horizon is None else
              shaded_fraction(lat, lon, elev, timestamps, horizon,
                              interval=interval, label=label))

    print(f"site       : {lat:.4f}, {lon:.4f} ({elev:.0f} m)")
    print(f"window     : {start} .. {end}, {interval} intervals labelled by {label}")
    print(f"samples    : {len(timestamps)} ({daytime.sum()} daytime, grid has "
          f"{P_pu.shape[1]} columns)")
    print(f"noise      : {noise_pct:.1f} %")
    if ac_rating is not None:
        print(f"inverter   : {ac_rating:.1f} kVA for {capacity_kwp:.1f} kWp "
              f"(DC/AC {capacity_kwp/ac_rating:.2f}), "
              f"{100*result.get('clipped_share', 0.0):.0f} % of daytime samples clipped")
    if shaded is not None:
        print(f"terrain    : {100*shaded[daytime].mean():.0f} % of daytime shaded, "
              f"peak skyline {horizon.elevation_deg.max():.0f}°")
    print(f"\ntruth      : tilt {tilt}°, azimuth {az_eu}° (EU), {capacity_kwp:.1f} kWp")
    if result["alpha"] is None:
        print(f"estimate   : solver returned {result['status']!r}")
        return result

    print(f"estimate   : tilt {result['best_tilt']}°, azimuth {result['best_az_eu']}° "
          f"(EU), {result['effective_kWp']:.1f} kWp")
    print(f"error      : tilt {result['best_tilt']-tilt:+d}°, "
          f"azimuth {result['best_az_eu']-az_eu:+d}°, "
          f"capacity {100*(result['effective_kWp']/capacity_kwp - 1):+.1f} %")
    print(f"fit        : R² {result['r2']:.5f}, RMSE {result['rmse_kw']:.3f} kW")
    print("\ncapacity by orientation (> 1 % of total):")
    print(format_results_table(result["alpha"]).to_string(index=False))

    if ac_rating is not None:
        print("\nclipping models on the same data:")
        print(f"  {'':<34} {'tilt':>6} {'azimuth':>8} {'kWp':>9} {'kVA':>7} {'R²':>8}")
        variants = (
            ("A  covering, rating estimated", {"method": "A"}),
            ("A1 covering, rating given", {"method": "A1", "ac_rating": ac_rating}),
            ("B  censored least squares", {"method": "B", "ac_rating": ac_rating}),
            ("   clipping ignored", {}),
        )
        for name, kwargs in variants:
            other = run_estimation(P_pu, measured, daytime, **kwargs)
            if other["alpha"] is None:
                print(f"  {name:<34} solver returned {other['status']!r}")
                continue
            found = other["ac_rating_kw"]
            print(f"  {name:<34} {other['best_tilt']:>5}° "
                  f"{other['best_az_eu']:>7}° {other['effective_kWp']:>9.1f} "
                  f"{(f'{found:.1f}' if found else '-'):>7} {other['r2']:>8.5f}")
        print(f"  {'   truth':<34} {tilt:>5}° {az_eu:>7}° {capacity_kwp:>9.1f} "
              f"{ac_rating:>7.1f}")

    if horizon is not None:
        blind_matrix, _ = build_pu_power_matrix(
            lat, lon, elev, timestamps, interval=interval, label=label
        )
        blind = run_estimation(blind_matrix, measured, daytime)
        if blind["alpha"] is not None:
            print(f"\nsame data, horizon ignored in the dictionary:")
            print(f"  tilt {blind['best_tilt']}° ({blind['best_tilt']-tilt:+d}), "
                  f"azimuth {blind['best_az_eu']}° ({blind['best_az_eu']-az_eu:+d}), "
                  f"{blind['effective_kWp']:.1f} kWp "
                  f"({100*(blind['effective_kWp']/capacity_kwp - 1):+.1f} %), "
                  f"R² {blind['r2']:.5f}")

    if plot:
        import matplotlib.pyplot as plt

        plot_estimate(result, timestamps, measured, P_pu, (tilt, az_eu), shaded=shaded)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=150)
            print(f"\nfigure written to {savefig}")
        if show:
            plt.show()

    return result


def plot_estimate(
    result: EstimationResult,
    timestamps: pd.DatetimeIndex,
    measured: np.ndarray,
    P_pu: np.ndarray,
    truth: Optional[tuple[int, int]] = None,
    days: int = 3,
    shaded: Optional[np.ndarray] = None,
):
    """
    Two panels: the fit against the measurements over a few days, and the
    capacity the solver placed on each orientation.

    ``shaded`` (from
    :func:`~pv_orientation_estimator.reference.shaded_fraction`) marks the
    intervals the terrain took the beam away for.

    Returns the two Axes.  Needs matplotlib (``pip install ".[plot]"``).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    MEASURED = "#1f6feb"
    FITTED = "#d97706"
    INK_MUTED = "#64748b"
    GRID = "#e2e8f0"

    fitted = P_pu @ result["alpha"]
    rating = result.get("ac_rating_kw")
    if rating is not None:
        fitted = np.minimum(fitted, rating)      # what the meter would record
    fig, (ax_fit, ax_grid) = plt.subplots(
        1, 2, figsize=(13.0, 4.4), gridspec_kw={"width_ratios": [1.45, 1]}
    )

    # --- left: measured vs fitted over the first `days` days ---------------
    window = timestamps < timestamps[0] + pd.Timedelta(days=days)
    if shaded is not None:
        ax_fit.fill_between(timestamps[window], 0.0, 1.0, where=shaded[window] > 0.0,
                            transform=ax_fit.get_xaxis_transform(), step="mid",
                            color=INK_MUTED, alpha=0.15, linewidth=0, zorder=1,
                            label="Terrain-shaded")
    ax_fit.plot(timestamps[window], measured[window], "o", markersize=3.5,
                color=MEASURED, label="Measured", zorder=3)
    ax_fit.plot(timestamps[window], fitted[window], "-", linewidth=2.0,
                color=FITTED, label="Fitted", zorder=4)
    if rating is not None:
        ax_fit.axhline(rating, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)
        ax_fit.annotate(f"{rating:.0f} kVA", xy=(0.99, rating),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=8, color=INK_MUTED)
    ax_fit.set_ylabel("AC power [kW]")
    ax_fit.set_title(f"Fit over the first {days} days", loc="left", fontsize=11)
    ax_fit.legend(frameon=False, fontsize=9, loc="upper left")
    ax_fit.grid(True, color=GRID, linewidth=0.6)
    ax_fit.set_axisbelow(True)
    for side in ("top", "right"):
        ax_fit.spines[side].set_visible(False)
    for tick in ax_fit.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")

    # --- right: where the solver put the capacity -------------------------
    grid, tilts, azimuths = alpha_to_heatmap_grid(result["alpha"])
    # alpha is sparse by construction — nearly all capacity lands in one cell.
    # A linear ramp would render every secondary lobe as white, so compress it.
    mesh = ax_grid.pcolormesh(azimuths, tilts, grid, cmap="Blues", shading="nearest",
                              norm=mcolors.PowerNorm(gamma=0.4, vmin=0.0,
                                                     vmax=max(grid.max(), 1e-9)))
    bar = fig.colorbar(mesh, ax=ax_grid, label="attributed capacity [kWp]")
    bar.outline.set_visible(False)

    ax_grid.plot(result["best_az_eu"], result["best_tilt"], "s", markersize=11,
                 markerfacecolor="none", markeredgecolor=FITTED, markeredgewidth=2.0,
                 label="Estimated")
    if truth is not None:
        ax_grid.plot(truth[1], truth[0], "+", markersize=13, color=MEASURED,
                     markeredgewidth=2.0, label="True")
    ax_grid.set_xlabel("Azimuth [°] (0 = south, − = east)")
    ax_grid.set_ylabel("Tilt [°]")
    ax_grid.set_title("Capacity per candidate orientation", loc="left", fontsize=11)
    ax_grid.legend(frameon=False, fontsize=9, loc="upper left")
    ax_grid.set_xticks(AZIMUTHS_EU_DEG[::4])
    ax_grid.set_yticks(TILTS_DEG[::3])

    for ax in (ax_fit, ax_grid):
        ax.tick_params(colors=INK_MUTED, labelcolor="#334155")

    return ax_fit, ax_grid


def _plant_power(
    lat: float,
    lon: float,
    elev: float,
    timestamps: pd.DatetimeIndex,
    tilt: float,
    az_eu: float,
    capacity_kwp: float,
    horizon: Optional[HorizonProfile] = None,
    view_factor: Optional[float] = None,
    air_temp: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Instantaneous clear-sky AC power [kW] of one orientation."""
    location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=elev)
    clearsky = location.get_clearsky(timestamps)
    solar_pos = location.get_solarposition(timestamps)

    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt),
        surface_azimuth=float(180 + az_eu),
        solar_zenith=solar_pos["apparent_zenith"].values,
        solar_azimuth=solar_pos["azimuth"].values,
        dni=clearsky["dni"].values,
        ghi=clearsky["ghi"].values,
        dhi=clearsky["dhi"].values,
    )

    if horizon is None:
        poa = irradiance["poa_global"]
    else:
        blocked = beam_blocked(horizon, solar_pos["azimuth"].values - 180.0,
                               solar_pos["apparent_elevation"].values)
        poa = (np.where(blocked, 0.0, irradiance["poa_direct"])
               + view_factor * irradiance["poa_sky_diffuse"]
               + irradiance["poa_ground_diffuse"])

    poa = np.clip(np.asarray(poa, dtype=float), 0.0, None)
    per_unit = poa / 1000.0
    if air_temp is not None:                                   # Eqs. (6)-(7)
        cell = np.asarray(air_temp, dtype=float) + TEMP_RISE_COEFF_DEFAULT * poa
        per_unit = per_unit * (1.0 + GAMMA_DEFAULT * (cell - TEMP_REF_C))
    return per_unit * capacity_kwp


if __name__ == "__main__":
    demo_estimate()
