#!/usr/bin/env python3
"""
Geocode a Swiss site, in either direction, through the swisstopo GeoAdmin API.

Examples
--------
    # address -> coordinates
    python scripts/geocode.py "Bahnhofstrasse 1, 3920 Zermatt"

    # ... with the alternative matches the search returned
    python scripts/geocode.py "Route Cantonale, Ecublens" --candidates

    # coordinates -> commune
    python scripts/geocode.py --lat 46.0207 --lon 7.7491
"""

from __future__ import annotations

import argparse
import sys

from pv_orientation_estimator.geocode import geocode, geocode_one, reverse_geocode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Swiss geocoding: address to coordinates, or coordinates to commune.",
    )
    parser.add_argument("query", nargs="?", help="address or place name to look up")
    parser.add_argument("--lat", type=float, help="latitude [deg N], for a reverse lookup")
    parser.add_argument("--lon", type=float, help="longitude [deg E], for a reverse lookup")
    parser.add_argument("--candidates", action="store_true",
                        help="list all matches of a forward search, not just the best")
    parser.add_argument("--limit", type=int, default=5,
                        help="how many candidates to list (default: 5)")
    args = parser.parse_args(argv)

    if args.query:
        return _forward(args)
    if args.lat is not None and args.lon is not None:
        return _reverse(args.lat, args.lon)

    parser.error("give an address to look up, or --lat and --lon for a reverse lookup")


def _forward(args: argparse.Namespace) -> int:
    if args.candidates:
        matches = geocode(args.query, limit=args.limit)
        if not matches:
            print(f"no match for {args.query!r}")
            return 1
        for match in matches:
            print(f"{match.lat:10.5f}, {match.lon:9.5f}   {match.address}")
        return 0

    place = geocode_one(args.query)
    if place is None:
        print(f"no match for {args.query!r}")
        return 1

    print(f"{args.query!r}")
    print(f"  coordinates: {place.lat:.5f}, {place.lon:.5f}")
    print(f"  matched:     {place.address}")
    print(f"  place:       {place.label}")
    if place.bfs_number:
        print(f"  commune no.: {place.bfs_number}")
    return 0


def _reverse(lat: float, lon: float) -> int:
    place = reverse_geocode(lat, lon)
    if place is None:
        print(f"{lat}, {lon} is not in Switzerland (or in no locality/commune)")
        return 1

    print(f"{lat}, {lon}")
    print(f"  place:       {place.label}")
    print(f"  locality:    {place.locality}  ({place.postcode})")
    print(f"  commune:     {place.commune} ({place.canton}), BFS {place.bfs_number}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
