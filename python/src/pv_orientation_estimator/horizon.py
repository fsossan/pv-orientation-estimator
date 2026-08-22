"""
Topographic (terrain) horizon profiles from PVGIS.

Downloads the horizon profile computed by the JRC PVGIS service from its
digital elevation model: for a set of azimuth directions around the site, the
elevation angle of the skyline (hills, mountains) above the astronomical
horizon.

    https://re.jrc.ec.europa.eu/api/v5_3/printhorizon?lat=..&lon=..&outputformat=json

The service returns 49 samples spanning -180°..180° in 7.5° steps, plus the
sun paths at the winter and summer solstice, which are handy for judging when
the terrain actually shades the array.

Conventions
-----------
PVGIS reports horizon azimuth as ``0 = south, -90 = east, +90 = west``, which
is exactly the European convention used elsewhere in this package (see
``grid.AZIMUTHS_EU_DEG``), so the values are kept as-is.  pvlib's own helper,
``pvlib.iotools.get_pvgis_horizon``, instead rotates them to the
north-clockwise convention (``south = 180``).

Typical use::

    from pv_orientation_estimator.horizon import download_horizon, plot_horizon

    profile = download_horizon(lat=46.52, lon=6.63)
    profile.to_dataframe().to_csv("horizon.csv", index=False)
    plot_horizon(profile)

    # Skyline elevation in the direction the array faces (EU azimuth):
    profile.elevation_at(-20.0)

Two quantities feed the production model (see
:mod:`pv_orientation_estimator.reference`):

* :func:`beam_blocked` — is the direct beam behind the terrain right now?
* :func:`sky_view_factor` — how much of the isotropic sky-diffuse does a tilted
  plane still see, terrain included?

For Swiss sites, :mod:`pv_orientation_estimator.geocode` resolves the
coordinates to a commune, which ``plot_horizon`` puts in the title.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests

from pv_orientation_estimator.geocode import Place, reverse_geocode

# PVGIS non-interactive service (v5.3) for the horizon profile.
PVGIS_HORIZON_URL = "https://re.jrc.ec.europa.eu/api/v5_3/printhorizon"
DEFAULT_TIMEOUT_S = 30.0

# Sky-dome cell size for the view-factor integration [deg].  The PVGIS profile
# itself is sampled every 7.5°, so 1° costs little and resolves it fully.
SVF_STEP_DEG = 1.0


@dataclass(frozen=True)
class HorizonProfile:
    """
    Terrain horizon around a site, as returned by PVGIS.

    Azimuths follow the European convention: 0 = south, negative toward east,
    positive toward west.
    """
    lat: float                              # site latitude [deg N]
    lon: float                              # site longitude [deg E]
    site_elevation_m: float                 # ground elevation reported by PVGIS [m]
    source: str                             # PVGIS horizon database, e.g. "DEM-calculated"
    azimuth_eu: np.ndarray                  # (K,) azimuth [deg], 0 = S, - = E, + = W
    elevation_deg: np.ndarray               # (K,) skyline elevation angle [deg]
    winter_solstice: Optional[tuple[np.ndarray, np.ndarray]] = None  # (azimuth, sun elevation) Dec 21
    summer_solstice: Optional[tuple[np.ndarray, np.ndarray]] = None  # (azimuth, sun elevation) Jun 21

    def to_dataframe(self) -> pd.DataFrame:
        """Horizon profile as a two-column DataFrame (azimuth_eu, elevation_deg)."""
        return pd.DataFrame(
            {"azimuth_eu": self.azimuth_eu, "elevation_deg": self.elevation_deg}
        )

    def elevation_at(
        self, azimuth_eu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Skyline elevation [deg] at arbitrary azimuth(s), linearly interpolated
        and periodic over 360°.  The sun is behind the terrain whenever its
        elevation is below this value.
        """
        query = _wrap_deg(np.asarray(azimuth_eu, dtype=float))

        xp = _wrap_deg(self.azimuth_eu)
        order = np.argsort(xp, kind="stable")
        xp, fp = xp[order], self.elevation_deg[order]
        xp, first = np.unique(xp, return_index=True)   # drop the duplicated ±180° sample
        fp = fp[first]

        return np.interp(query, xp, fp, period=360.0)


def download_horizon(
    lat: float,
    lon: float,
    url: str = PVGIS_HORIZON_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> HorizonProfile:
    """
    Download the terrain horizon at (lat, lon) from PVGIS.

    Requires network access.  Raises ``requests.HTTPError`` if PVGIS rejects
    the request (e.g. a location outside its DEM coverage) and ``ValueError``
    if the payload does not contain a horizon profile.
    """
    response = requests.get(
        url, params={"lat": lat, "lon": lon, "outputformat": "json"}, timeout=timeout
    )
    response.raise_for_status()
    return parse_horizon_response(response.json(), lat=lat, lon=lon)


def parse_horizon_response(
    payload: dict[str, Any], lat: float, lon: float
) -> HorizonProfile:
    """Turn a PVGIS ``printhorizon`` JSON payload into a :class:`HorizonProfile`."""
    outputs = payload.get("outputs", {})
    rows = outputs.get("horizon_profile")
    if not rows:
        raise ValueError("PVGIS response contains no 'horizon_profile' section")

    location = payload.get("inputs", {}).get("location", {})

    return HorizonProfile(
        lat=float(location.get("latitude", lat)),
        lon=float(location.get("longitude", lon)),
        site_elevation_m=float(location.get("elevation", float("nan"))),
        source=str(payload.get("inputs", {}).get("horizon_db", "unknown")),
        azimuth_eu=np.array([float(r["A"]) for r in rows]),
        elevation_deg=np.array([float(r["H_hor"]) for r in rows]),
        winter_solstice=_sun_path(outputs.get("winter_solstice"), "A_sun(w)", "H_sun(w)"),
        summer_solstice=_sun_path(outputs.get("summer_solstice"), "A_sun(s)", "H_sun(s)"),
    )


def beam_blocked(
    profile: HorizonProfile,
    solar_azimuth_eu: Union[float, np.ndarray],
    solar_elevation: Union[float, np.ndarray],
) -> np.ndarray:
    """
    True where the terrain hides the sun, i.e. the direct beam is lost.

    Both angles are in degrees, the azimuth in the EU convention (0 = south).
    Pass the *apparent* (refraction-corrected) elevation: refraction is what
    decides visibility near the skyline.

    The test is binary at one instant.  Averaging it over a measurement
    interval — which :func:`~pv_orientation_estimator.reference.shaded_fraction`
    does — is what turns it into the fraction of the interval that the beam is
    actually lost for.
    """
    return np.asarray(solar_elevation) < profile.elevation_at(solar_azimuth_eu)


def sky_view_factor(
    profile: HorizonProfile,
    tilt: float,
    az_eu: float,
    step: float = SVF_STEP_DEG,
) -> float:
    """
    Fraction of the isotropic sky-diffuse irradiance a plane still receives.

    ``1.0`` is an unobstructed horizon; a plane hemmed in by mountains sees
    less.  The sky dome is discretised in ``step``-degree cells and each cell
    counted when it clears both the terrain and the plane's own horizon::

        SVF = ∫∫_visible cos θ_inc dΩ  /  [π (1 + cos β) / 2]

    with the denominator the same integral over an open horizon.  Dozier and
    Frew (1990) give a closed form; the numerical sum is used here because it
    handles the plane's own cut-off without special cases.

    The factor depends only on the geometry, not on time, so it costs one
    evaluation per candidate orientation for a whole time series.
    """
    return float(sky_view_factors(profile, [(tilt, az_eu)], step)[0])


def sky_view_factors(
    profile: HorizonProfile,
    layouts: Sequence[tuple[float, float]],
    step: float = SVF_STEP_DEG,
) -> np.ndarray:
    """
    :func:`sky_view_factor` for many ``(tilt, azimuth_eu)`` layouts at once.

    The sky-dome discretisation and the terrain visibility mask are built once
    and shared, which is what makes the whole 320-orientation grid cheap.
    """
    azimuth = np.arange(-180.0, 180.0, step) + step / 2.0     # cell midpoints
    elevation = np.arange(0.0, 90.0, step) + step / 2.0
    phi, elev = np.meshgrid(np.radians(azimuth), np.radians(elevation), indexing="ij")

    visible = elev > np.radians(profile.elevation_at(azimuth))[:, None]
    solid_angle = np.cos(elev) * np.radians(step) ** 2        # dΩ = cos(elev) dElev dφ
    weight = np.where(visible, solid_angle, 0.0)

    factors = []
    for tilt, az_eu in layouts:
        beta, gamma = np.radians(float(tilt)), np.radians(float(az_eu))
        cos_inc = np.cos(beta) * np.sin(elev) + np.sin(beta) * np.cos(elev) * np.cos(phi - gamma)
        seen = np.sum(np.clip(cos_inc, 0.0, None) * weight)
        open_sky = np.pi * (1.0 + np.cos(beta)) / 2.0
        factors.append(seen / open_sky)

    return np.array(factors)


def plot_horizon(
    profile: HorizonProfile,
    ax=None,
    show_solstices: bool = True,
    title: Optional[str] = None,
    place: Optional[Union[Place, str]] = None,
):
    """
    Plot the horizon profile (elevation vs azimuth), with the solstice sun
    paths overlaid when PVGIS provided them.

    ``place`` — a :class:`~pv_orientation_estimator.geocode.Place` or a plain
    string — names the site in the title; the coordinates, elevation and full
    place label go in the subtitle underneath.

    Needs matplotlib (``pip install ".[plot]"``).  Returns the Axes, so the
    caller decides whether to ``show()`` or ``savefig()``.
    """
    import matplotlib.pyplot as plt

    TITLE_PAD_PT = 24.0          # room for the subtitle line under the title
    TERRAIN_FILL = "#94a3b8"
    TERRAIN_EDGE = "#475569"
    WINTER = "#1f6feb"
    SUMMER = "#d97706"
    INK_MUTED = "#64748b"

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.0))

    ax.fill_between(
        profile.azimuth_eu, 0.0, profile.elevation_deg,
        color=TERRAIN_FILL, alpha=0.55, linewidth=0, zorder=2,
    )
    ax.plot(
        profile.azimuth_eu, profile.elevation_deg,
        color=TERRAIN_EDGE, linewidth=2.0, zorder=3, label="Terrain horizon",
    )

    peak = float(np.max(profile.elevation_deg))
    if show_solstices:
        for path, color, label in (
            (profile.winter_solstice, WINTER, "Sun path, winter solstice"),
            (profile.summer_solstice, SUMMER, "Sun path, summer solstice"),
        ):
            if path is None:
                continue
            sun_az, sun_el = path
            visible = np.where(sun_el > 0.0, sun_el, np.nan)  # hide the night-time zeros
            ax.plot(sun_az, visible, color=color, linewidth=2.0, zorder=4, label=label)
            if np.any(np.isfinite(visible)):
                peak = max(peak, float(np.nanmax(visible)))

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(0.0, peak * 1.15 + 1.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 45.0))
    ax.set_xticklabels(["N\n−180°", "−135°", "E\n−90°", "−45°",
                        "S\n0°", "45°", "W\n90°", "135°", "N\n180°"])
    ax.set_xlabel("Azimuth (0° = south, negative = east)")
    ax.set_ylabel("Elevation angle [°]")

    site_name = place.short_label if isinstance(place, Place) else place
    if title is None:
        title = f"Terrain horizon at {site_name}" if site_name else "Terrain horizon"

    subtitle = [f"{profile.lat:.4f}°, {profile.lon:.4f}°",
                f"{profile.site_elevation_m:.0f} m a.s.l."]
    if isinstance(place, Place) and place.label != site_name:
        subtitle.insert(0, place.label)     # e.g. "1015 Lausanne, Ecublens (VD)"

    # matplotlib shares one pad between the left/centre/right title slots, and
    # each set_title() resets it — so the padded call has to come last.
    ax.set_title(f"PVGIS, {profile.source}", loc="right", fontsize=9,
                 color=INK_MUTED, pad=TITLE_PAD_PT)
    ax.set_title(title, loc="left", fontsize=12, pad=TITLE_PAD_PT)
    ax.annotate(
        "  ·  ".join(subtitle), xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(0.0, 7.0), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, color=INK_MUTED,
    )

    ax.grid(True, color="#e2e8f0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd5e1")
    ax.tick_params(colors=INK_MUTED, labelcolor="#334155")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    return ax


def demo_horizon(
    lat: float = 46.6777,
    lon: float = 7.6409,
    show: bool = True,
    savefig: Optional[str] = None,
    name_site: bool = True,
) -> HorizonProfile:
    """
    Small end-to-end demo: download the horizon for one location (Wimmis BE
    by default, in the Simme valley), name the commune it sits in, and plot
    it.  Returns the downloaded profile.

    With ``name_site`` the coordinates are reverse-geocoded through the
    swisstopo API; the plot simply goes untitled if that lookup fails or the
    site is outside Switzerland.
    """
    import matplotlib.pyplot as plt

    profile = download_horizon(lat, lon)
    place = locate_site(profile.lat, profile.lon) if name_site else None

    print(f"PVGIS horizon at {profile.lat:.4f}, {profile.lon:.4f} "
          f"({profile.site_elevation_m:.0f} m a.s.l., {profile.source})")
    if place is not None:
        print(f"  commune:   {place.label}")
    print(f"  {len(profile.azimuth_eu)} samples, "
          f"peak {profile.elevation_deg.max():.1f}° at "
          f"{profile.azimuth_eu[np.argmax(profile.elevation_deg)]:.1f}° azimuth")
    print(f"  due south: {profile.elevation_at(0.0):.1f}°")

    plot_horizon(profile, place=place)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
        print(f"  figure written to {savefig}")
    if show:
        plt.show()

    return profile


def locate_site(lat: float, lon: float) -> Optional[Place]:
    """
    Best-effort reverse geocoding for labelling a plot: the commune containing
    the point, or ``None`` outside Switzerland or when the service is
    unreachable.  Naming a site is cosmetic, so a failed lookup must never
    take the plot down with it.
    """
    try:
        return reverse_geocode(lat, lon)
    except requests.RequestException:
        return None


def _wrap_deg(angles: np.ndarray) -> np.ndarray:
    """Wrap angles into [-180, 180)."""
    return np.mod(angles + 180.0, 360.0) - 180.0


def _sun_path(
    rows: Optional[list[dict[str, Any]]], az_key: str, el_key: str
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extract a solstice sun path, or None when PVGIS did not return it."""
    if not rows:
        return None
    return (
        np.array([float(r[az_key]) for r in rows]),
        np.array([float(r[el_key]) for r in rows]),
    )


if __name__ == "__main__":
    demo_horizon()