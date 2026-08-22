"""
Geocoding for Swiss sites, via the swisstopo GeoAdmin API.

Two directions, both against the free, key-less federal service
(https://api3.geo.admin.ch):

* **reverse** — coordinates to place: :func:`reverse_geocode` asks the
  ``identify`` endpoint which locality and which commune polygon contain the
  point.
* **forward** — address to coordinates: :func:`geocode` / :func:`geocode_one`
  query the ``SearchServer`` endpoint, which does a fuzzy search over the
  official address register, the commune list and the gazetteer.

Coverage is **Switzerland (and Liechtenstein) only** — outside it, the service
returns nothing and these functions return ``None`` / an empty list.  For sites
elsewhere, an OSM-based service (Nominatim, Photon) or GeoNames is the usual
substitute.

Typical use::

    from pv_orientation_estimator.geocode import geocode_one, reverse_geocode

    reverse_geocode(46.0207, 7.7491).label      # '3920 Zermatt (VS)'
    geocode_one("Bahnhofstrasse 1, 3920 Zermatt")  # Place(lat=46.023.., lon=7.748..)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import requests

# GeoAdmin endpoints.
IDENTIFY_URL = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
SEARCH_URL = "https://api3.geo.admin.ch/rest/services/ech/SearchServer"

# Official locality/postcode register: point-in-polygon over inhabited places.
LOCALITY_LAYER = "ch.swisstopo-vd.ortschaftenverzeichnis_plz"
# swissBOUNDARIES3D communes: tiles the whole country, so it also resolves
# points on alps, forests and glaciers, where no locality is defined.
COMMUNE_LAYER = "ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill"

# Search sources: official addresses, commune list (gg25), place names.
DEFAULT_ORIGINS = ("address", "gg25", "gazetteer")
DEFAULT_TIMEOUT_S = 30.0

# Gazetteer entries for large regions ("Grossregion Mittelland"): they span
# dozens of communes, are never a usable site location, and their labels list
# every member commune, which makes any text match meaningless.
EXCLUDED_OBJECT_CLASSES = frozenset({"TLM_GEBIETSNAME"})

# A query token and a result token count as related when their first
# MIN_PREFIX_MATCH characters agree — see _looks_like_match.
MIN_PREFIX_MATCH = 3

_HTML_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Place:
    """A located Swiss place.  Fields not resolved by the query stay ``None``."""
    lat: float                         # [deg N], WGS84
    lon: float                         # [deg E], WGS84
    locality: Optional[str] = None     # postal locality ("Ortschaft"), e.g. "Zermatt"
    postcode: Optional[int] = None     # PLZ / NPA
    commune: Optional[str] = None      # political commune ("Gemeinde"), e.g. "Zermatt"
    canton: Optional[str] = None       # canton abbreviation, e.g. "VS"
    bfs_number: Optional[int] = None   # federal commune number (BFS/OFS)
    address: Optional[str] = None      # raw label from a forward search, if any

    @property
    def label(self) -> str:
        """Full one-line description, e.g. ``'1015 Lausanne, Ecublens (VD)'``."""
        if self.locality and self.postcode:
            head = f"{self.postcode} {self.locality}"
        else:
            head = self.locality or self.commune or ""

        if not head:
            return self.address or f"{self.lat:.4f}, {self.lon:.4f}"

        if self.commune and self.commune != self.locality:
            head = f"{head}, {self.commune}"
        return _with_canton(head, self.canton)

    @property
    def short_label(self) -> str:
        """Compact name for titles, e.g. ``'Ecublens (VD)'`` — commune first."""
        name = self.commune or self.locality
        if not name:
            return self.address or f"{self.lat:.4f}, {self.lon:.4f}"
        return _with_canton(name, self.canton)

    def __str__(self) -> str:
        return self.label


def reverse_geocode(
    lat: float,
    lon: float,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Optional[Place]:
    """
    Resolve coordinates to the Swiss locality and commune containing them.

    Returns ``None`` for points outside Switzerland (and for the few spots
    covered by neither layer, such as the middle of a large lake).
    """
    locality = _identify(lat, lon, LOCALITY_LAYER, timeout)
    commune = _current_commune(_identify_all(lat, lon, COMMUNE_LAYER, timeout))

    if locality is None and commune is None:
        return None

    locality = locality or {}
    commune = commune or {}
    postcode = locality.get("plz")

    return Place(
        lat=float(lat),
        lon=float(lon),
        locality=locality.get("langtext"),
        postcode=int(postcode) if postcode is not None else None,
        commune=commune.get("gemname"),
        canton=commune.get("kanton"),
        bfs_number=_as_int(commune.get("gde_nr")),
    )


def geocode(
    query: str,
    limit: int = 5,
    origins: Sequence[str] = DEFAULT_ORIGINS,
    timeout: float = DEFAULT_TIMEOUT_S,
    verify: bool = True,
) -> list[Place]:
    """
    Forward geocoding: search an address or place name, best match first.

    The GeoAdmin search is **fuzzy and never reports "no match"** — it always
    returns its best-ranked guess, so ``"xyzzy plugh"`` comes back as
    ``Bülach (ZH)``.  With ``verify=True`` (default) results that share no
    word stem with the query are dropped, which turns a nonsense query into an
    empty list; pass ``verify=False`` to see the service's raw ranking.

    Fuzziness within a real match is kept: a street number that does not exist
    still returns the closest entries in that street, so check the ``address``
    field rather than assuming an exact hit.  Only ``lat``, ``lon`` and
    ``address`` are filled in; use :func:`geocode_one` with ``enrich=True``
    (or call :func:`reverse_geocode`) for commune, canton and postcode.
    """
    response = requests.get(
        SEARCH_URL,
        params={
            "searchText": query,
            "type": "locations",
            "origins": ",".join(origins),
            "limit": limit,
            "sr": 4326,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    matches = parse_search_response(response.json())
    if verify:
        matches = [m for m in matches if _looks_like_match(query, m.address or "")]
    return matches


def geocode_one(
    query: str,
    enrich: bool = True,
    origins: Sequence[str] = DEFAULT_ORIGINS,
    timeout: float = DEFAULT_TIMEOUT_S,
    verify: bool = True,
) -> Optional[Place]:
    """
    Best match for ``query``, or ``None`` when nothing plausible is found.

    With ``enrich=True`` (default) the coordinates are reverse-geocoded to also
    fill in locality, postcode, commune and canton — one extra request.  See
    :func:`geocode` for what ``verify`` rejects.
    """
    matches = geocode(query, limit=1, origins=origins, timeout=timeout, verify=verify)
    if not matches:
        return None

    best = matches[0]
    if not enrich:
        return best

    detailed = reverse_geocode(best.lat, best.lon, timeout=timeout)
    if detailed is None:
        return best

    # Keep the searched coordinates and the address label from the search.
    return Place(
        lat=best.lat,
        lon=best.lon,
        locality=detailed.locality,
        postcode=detailed.postcode,
        commune=detailed.commune,
        canton=detailed.canton,
        bfs_number=detailed.bfs_number,
        address=best.address,
    )


def parse_search_response(payload: dict[str, Any]) -> list[Place]:
    """
    Turn a GeoAdmin ``SearchServer`` payload into a list of :class:`Place`.

    Entries of an excluded object class (see :data:`EXCLUDED_OBJECT_CLASSES`)
    are skipped.
    """
    places = []
    for result in payload.get("results", []):
        attrs = result.get("attrs", {})
        if attrs.get("lat") is None or attrs.get("lon") is None:
            continue
        if attrs.get("objectclass") in EXCLUDED_OBJECT_CLASSES:
            continue
        places.append(
            Place(
                lat=float(attrs["lat"]),
                lon=float(attrs["lon"]),
                address=_strip_html(attrs.get("label", "")) or None,
            )
        )
    return places


def _identify(
    lat: float, lon: float, layer: str, timeout: float
) -> Optional[dict[str, Any]]:
    """First feature of ``layer`` containing the point, or None."""
    features = _identify_all(lat, lon, layer, timeout)
    return features[0] if features else None


def _identify_all(
    lat: float, lon: float, layer: str, timeout: float
) -> list[dict[str, Any]]:
    """
    Properties of every feature of ``layer`` containing the point.

    ``tolerance=0`` makes this a plain point-in-polygon test.  (The parameter
    is in screen pixels, not metres, so any non-zero value would need a
    matching ``mapExtent``/``imageDisplay`` pair to mean anything.)
    """
    response = requests.get(
        IDENTIFY_URL,
        params={
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "layers": f"all:{layer}",
            "sr": 4326,
            "tolerance": 0,
            "returnGeometry": False,
            "geometryFormat": "geojson",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return [r["properties"] for r in response.json().get("results", [])]


def _current_commune(features: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    Pick today's commune out of the boundary layer.

    swissBOUNDARIES3D carries one row per historical year (the 2026 layer
    returns ~177 of them per point), of which exactly one is flagged current.
    """
    for properties in features:
        if properties.get("is_current_jahr"):
            return properties
    return None


def _with_canton(name: str, canton: Optional[str]) -> str:
    """
    Append the canton, unless the name already carries it.

    Commune names are disambiguated by the source itself where they are not
    unique — swissBOUNDARIES3D returns ``"Ecublens (VD)"``, not ``"Ecublens"``.
    """
    if not canton or f"({canton})" in name:
        return name
    return f"{name} ({canton})"


def _looks_like_match(query: str, label: str) -> bool:
    """
    Cheap plausibility test between a query and a search hit.

    True when any word of the query and any word of the label start with the
    same ``MIN_PREFIX_MATCH`` characters, comparing accent-folded lowercase.
    The prefix (rather than equality) keeps language variants and small typos
    alive — ``"Geneva"`` still matches ``"Genève"`` — while rejecting hits
    that have nothing to do with the query.
    """
    query_words = _word_stems(query)
    if not query_words:
        return True
    label_words = _word_stems(label)
    return any(q == l for q in query_words for l in label_words)


def _word_stems(text: str) -> set[str]:
    """Accent-folded, lowercase word prefixes of at least MIN_PREFIX_MATCH chars."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return {
        word[:MIN_PREFIX_MATCH]
        for word in re.split(r"[^a-z0-9]+", folded)
        if len(word) >= MIN_PREFIX_MATCH
    }


def _strip_html(text: str) -> str:
    """GeoAdmin marks matched substrings with <b> tags; drop them."""
    return _HTML_TAG.sub("", text).strip()


def _as_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)
