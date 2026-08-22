#!/usr/bin/env python3
"""
Download the topographic horizon profile of a site from PVGIS.

Examples
--------
    # print the profile for a site
    python scripts/download_horizon.py --lat 46.6777 --lon 7.6409

    # save it and show the plot
    python scripts/download_horizon.py --lat 46.6777 --lon 7.6409 \
        --csv horizon.csv --plot

    # built-in demo: one location, downloaded and plotted
    python scripts/download_horizon.py --demo
"""

from __future__ import annotations

import argparse
import sys

from pv_orientation_estimator.horizon import (
    demo_horizon,
    download_horizon,
    locate_site,
    plot_horizon,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a topographic horizon profile from PVGIS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat", type=float, help="latitude [deg N]")
    parser.add_argument("--lon", type=float, help="longitude [deg E]")
    parser.add_argument("--csv", help="write the profile to this CSV file")
    parser.add_argument("--png", help="write the plot to this image file")
    parser.add_argument("--plot", action="store_true", help="show the plot in a window")
    parser.add_argument(
        "--demo", action="store_true",
        help="run demo_horizon(): download one location (Wimmis BE) and plot it",
    )
    parser.add_argument(
        "--no-geocode", action="store_true",
        help="skip the swisstopo lookup that names the commune on the plot",
    )
    args = parser.parse_args(argv)

    if args.demo:
        demo_horizon(show=True, savefig=args.png, name_site=not args.no_geocode)
        return 0

    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon are required (or use --demo)")

    profile = download_horizon(args.lat, args.lon)
    frame = profile.to_dataframe()
    place = None if args.no_geocode else locate_site(profile.lat, profile.lon)

    print(f"PVGIS horizon at {profile.lat:.4f}, {profile.lon:.4f} "
          f"({profile.site_elevation_m:.0f} m a.s.l., {profile.source})")
    if place is not None:
        print(f"commune: {place.label}")
    print(frame.to_string(index=False))

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"written to {args.csv}")

    if args.png or args.plot:
        import matplotlib.pyplot as plt

        plot_horizon(profile, place=place)
        plt.tight_layout()
        if args.png:
            plt.savefig(args.png, dpi=150)
            print(f"figure written to {args.png}")
        if args.plot:
            plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
