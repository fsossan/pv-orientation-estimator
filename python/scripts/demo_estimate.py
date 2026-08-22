#!/usr/bin/env python3
"""
Run the orientation estimator end-to-end on a synthetic plant.

Examples
--------
    # defaults: 100 kWp at tilt 30, azimuth -20, hourly end-labelled, 2 % noise
    python scripts/demo_estimate.py

    # a different plant, saved to a file instead of shown
    python scripts/demo_estimate.py --tilt 15 --azimuth 30 --png fit.png --no-plot

    # what happens on 15-minute data with more noise
    python scripts/demo_estimate.py --interval 15min --noise 5

    # 100 kWp behind a 75 kVA inverter (DC/AC 1.33), clipped; both methods reported
    python scripts/demo_estimate.py --kva 75
    python scripts/demo_estimate.py --kva 75 --method A --noise 0

    # a shaded valley site (Wimmis BE), with and without the terrain horizon
    python scripts/demo_estimate.py --lat 46.6777 --lon 7.6409 --elev 639 --horizon
"""

from __future__ import annotations

import argparse
import sys

from pv_orientation_estimator.demo import DEMO_LAT, DEMO_LON, DEMO_ELEV, demo_estimate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic end-to-end demo of the PV orientation estimator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat", type=float, default=DEMO_LAT, help="latitude [deg N]")
    parser.add_argument("--lon", type=float, default=DEMO_LON, help="longitude [deg E]")
    parser.add_argument("--elev", type=float, default=DEMO_ELEV, help="elevation [m]")
    parser.add_argument("--tilt", type=int, default=30, help="true tilt [deg]")
    parser.add_argument("--azimuth", type=int, default=-20,
                        help="true azimuth [deg, EU: 0 = south]")
    parser.add_argument("--capacity", type=float, default=100.0, help="true capacity [kWp]")
    parser.add_argument("--start", default="2023-05-01", help="first day of the window")
    parser.add_argument("--end", default="2023-06-20", help="last day of the window")
    parser.add_argument("--interval", default="1h", help="metering interval, e.g. 15min")
    parser.add_argument("--label", default="end", choices=["centre", "start", "end"],
                        help="what the timestamps mark")
    parser.add_argument("--noise", type=float, default=2.0, help="measurement noise [%%]")
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    parser.add_argument("--kva", type=float, default=None,
                        help="AC rating [kW]; below --capacity it clips the plant")
    parser.add_argument("--method", choices=["A", "A1", "B"], default=None,
                        help="clipping model: A = min capacity covering the data "
                             "(rating estimated), A1 = same with the rating given, "
                             "B = censored least squares (default with --kva)")
    parser.add_argument("--horizon", action="store_true",
                        help="download the site's terrain horizon from PVGIS and shade "
                             "both the plant and the dictionary with it")
    parser.add_argument("--png", help="write the figure to this file")
    parser.add_argument("--no-plot", action="store_true", help="print only, no figure")
    args = parser.parse_args(argv)

    horizon = None
    if args.horizon:
        from pv_orientation_estimator.horizon import download_horizon
        horizon = download_horizon(args.lat, args.lon)

    result = demo_estimate(
        lat=args.lat, lon=args.lon, elev=args.elev,
        tilt=args.tilt, az_eu=args.azimuth, capacity_kwp=args.capacity,
        start=args.start, end=args.end,
        interval=args.interval, label=args.label,
        noise_pct=args.noise, seed=args.seed, horizon=horizon, ac_rating=args.kva, method=args.method,
        plot=not args.no_plot or bool(args.png),
        show=not args.no_plot,
        savefig=args.png,
    )
    return 0 if result["alpha"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
