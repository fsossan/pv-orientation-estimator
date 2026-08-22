"""
Offline tests for the GeoAdmin geocoding helpers — no network access required.
"""

import pytest

from pv_orientation_estimator.geocode import (
    Place,
    _looks_like_match,
    parse_search_response,
)


SEARCH_PAYLOAD = {
    "results": [
        {"attrs": {"label": "<b>Bahnhofstrasse 1 3920 Zermatt</b>",
                   "lat": 46.02328, "lon": 7.74807, "origin": "address"}},
        {"attrs": {"label": "<b>Zermatt (VS)</b>",
                   "lat": 45.99043, "lon": 7.74208, "origin": "gg25"}},
        # A region entry: dropped, its label lists every member commune.
        {"attrs": {"label": "<i>Grossregion</i> <b>Mittelland</b> - Zug,Olten,Suhr",
                   "lat": 47.0, "lon": 7.5, "objectclass": "TLM_GEBIETSNAME"}},
        # No coordinates: unusable.
        {"attrs": {"label": "<b>Nowhere</b>", "lat": None, "lon": None}},
    ]
}


def test_parse_search_response_strips_html_and_filters():
    places = parse_search_response(SEARCH_PAYLOAD)

    assert len(places) == 2
    assert places[0].address == "Bahnhofstrasse 1 3920 Zermatt"   # tags gone
    assert places[0].lat == pytest.approx(46.02328)
    assert places[1].address == "Zermatt (VS)"


def test_label_combines_locality_commune_and_canton():
    assert Place(lat=0, lon=0, locality="Zermatt", postcode=3920,
                 commune="Zermatt", canton="VS").label == "3920 Zermatt (VS)"

    assert Place(lat=0, lon=0, locality="Lausanne", postcode=1015,
                 commune="Ecublens (VD)", canton="VD").label == (
        "1015 Lausanne, Ecublens (VD)")   # canton not repeated


def test_short_label_prefers_the_commune():
    place = Place(lat=0, lon=0, locality="Lausanne", postcode=1015,
                  commune="Ecublens (VD)", canton="VD")
    assert place.short_label == "Ecublens (VD)"

    # Nothing resolved: fall back to the raw search label, then to coordinates.
    assert Place(lat=0, lon=0, address="Somewhere").short_label == "Somewhere"
    assert Place(lat=46.5, lon=6.6).short_label == "46.5000, 6.6000"


def test_looks_like_match_accepts_variants_and_rejects_nonsense():
    assert _looks_like_match("Bahnhofstrasse 1, 3920 Zermatt",
                             "Bahnhofstrasse 1 3920 Zermatt")
    assert _looks_like_match("Geneva", "Genève")        # accent-folded prefix
    assert _looks_like_match("1015 Ecublens", "Chemin Erna Hamburger 1b 1015 Lausanne")

    assert not _looks_like_match("xyzzy plugh", "Bülach (ZH)")
    assert not _looks_like_match("qqqzzz", "Willisau (LU)")

    assert _looks_like_match("", "anything")            # nothing to check against
