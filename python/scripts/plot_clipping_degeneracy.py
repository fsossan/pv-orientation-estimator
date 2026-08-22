#!/usr/bin/env python3
"""
Why the one-sided clipping relaxation cannot identify the capacity.

Draws two panels:

  left  — the feasible set for the modelled power X.  The constraints
          X <= P_pu @ alpha and X <= S bound X from above only, so the whole
          region under the ceiling is admissible and X = P sits inside it
          however large alpha is.
  right — the residual as a function of the assumed capacity, along the ray
          alpha = c * e_true.  The relaxation's residual falls to zero and
          stays there: a flat valley reaching to infinity, i.e. no minimiser.
          The censored objective has a proper minimum at the true capacity.

Also draws how the two estimators differ once noise is present: Method A rides
the top of the measurements, Method B runs through them.

Writes doc/figures/clipping_degeneracy.{pdf,png} and
doc/figures/clipping_methods.{pdf,png}.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from pv_orientation_estimator import (
    build_pu_power_matrix,
    DAYTIME_GHI_THRESHOLD,
    LAYOUTS,
)
from pv_orientation_estimator.demo import synthetic_plant
from pv_orientation_estimator.estimate import CLIP_BAND

LAT, LON, ELEV = 46.52, 6.63, 500.0
TILT, AZ, KWP, KVA = 30, -20, 100.0, 75.0        # DC/AC = 1.33
START, END = "2023-05-01", "2023-06-20"

MEASURED = "#1f6feb"
TRUE_CAP = "#d97706"
ABSURD_CAP = "#7c3aed"
INK_MUTED = "#64748b"
GRID = "#e2e8f0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../doc/figures", help="output directory")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    timestamps = pd.date_range(START, END, freq="1h", tz="UTC")
    matrix, ghi = build_pu_power_matrix(LAT, LON, ELEV, timestamps,
                                        interval="1h", label="end")
    daytime = ghi > DAYTIME_GHI_THRESHOLD
    measured = synthetic_plant(LAT, LON, ELEV, timestamps, TILT, AZ, KWP,
                               interval="1h", label="end", ac_rating=KVA)

    profile = matrix[:, LAYOUTS.index((TILT, AZ))]      # per-unit, the true shape
    fig, (ax_set, ax_obj) = plt.subplots(1, 2, figsize=(13.0, 4.4),
                                         gridspec_kw={"width_ratios": [1.25, 1]})

    # ---------------- left: the feasible set is one-sided -------------------
    window = timestamps < timestamps[0] + pd.Timedelta(days=2)
    hours = timestamps[window]
    for capacity, colour, style in ((KWP, TRUE_CAP, "-"), (3 * KWP, ABSURD_CAP, "--")):
        ceiling = np.minimum(profile * capacity, KVA)[window]
        ax_set.fill_between(hours, 0.0, ceiling, color=colour, alpha=0.10, linewidth=0)
        ax_set.plot(hours, ceiling, style, color=colour, linewidth=2.0,
                    label=f"ceiling min(Hα, S), α = {capacity:.0f} kWp")

    ax_set.plot(hours, measured[window], "o", markersize=4.0, color=MEASURED,
                zorder=5, label="measured P  (= X at the optimum, both cases)")
    ax_set.set_ylabel("AC power [kW]")
    ax_set.set_title("X is bounded from above only — so X = P is feasible either way",
                     loc="left", fontsize=11)
    ax_set.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax_set.set_ylim(0, 1.35 * KVA)
    for tick in ax_set.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")

    # ---------------- right: residual along the capacity ray ----------------
    lit_profile, lit_measured = profile[daytime], measured[daytime]
    clipped = lit_measured >= KVA * (1.0 - CLIP_BAND)
    capacities = np.linspace(20.0, 320.0, 601)

    def rms(values):
        return np.sqrt(np.mean(values ** 2))

    relaxed, plain, censored = [], [], []
    for capacity in capacities:
        modelled = lit_profile * capacity
        # relaxation: X may duck below the ceiling for free, so only the part
        # of P that the ceiling cannot reach is ever penalised
        relaxed.append(rms(np.maximum(0.0, lit_measured - np.minimum(modelled, KVA))))
        plain.append(rms(lit_measured - modelled))
        censored.append(rms(np.concatenate([
            lit_measured[~clipped] - modelled[~clipped],
            np.maximum(0.0, KVA - modelled[clipped]),
        ])))

    relaxed, plain, censored = map(np.asarray, (relaxed, plain, censored))
    ax_obj.plot(capacities, relaxed, "-", color=ABSURD_CAP, linewidth=2.2,
                label="proposed relaxation")
    ax_obj.plot(capacities, plain, "-", color=MEASURED, linewidth=2.0,
                label="plain NNLS (clipping ignored)")
    ax_obj.plot(capacities, censored, "-", color=TRUE_CAP, linewidth=2.0,
                label="censored least squares")

    flat_from = capacities[np.argmax(relaxed < 1e-9)]
    ax_obj.axvspan(flat_from, capacities[-1], color=ABSURD_CAP, alpha=0.06, linewidth=0)
    ax_obj.annotate(f"flat at zero for every α ≥ {flat_from:.0f} kWp —\n"
                    "each one is a global optimum,\nso nothing picks out the capacity",
                    xy=(flat_from + 12, 0.62), xycoords=("data", "axes fraction"),
                    fontsize=8.5, color=ABSURD_CAP, va="top")

    ax_obj.axvline(KWP, color=INK_MUTED, linewidth=1.0, linestyle=":")
    ax_obj.annotate("true capacity", xy=(KWP, 0.30), xycoords=("data", "axes fraction"),
                    xytext=(-5, 0), textcoords="offset points", fontsize=8.5,
                    color=INK_MUTED, va="center", ha="right", rotation=90)

    for series, colour, offset, align in ((plain, MEASURED, (-8, 10), "right"),
                                          (censored, TRUE_CAP, (8, -16), "left")):
        best = capacities[int(np.argmin(series))]
        ax_obj.plot([best], [series.min()], "v", color=colour, markersize=8, zorder=5)
        ax_obj.annotate(f"min at {best:.0f} kWp", xy=(best, series.min()), xytext=offset,
                        textcoords="offset points", ha=align, fontsize=8.5, color=colour)
    ax_obj.set_xlabel("assumed capacity along α = c · e$_{true}$  [kWp]")
    ax_obj.set_ylabel("residual  RMS [kW]")
    ax_obj.set_title("Only the censored objective has a minimum", loc="left", fontsize=11)
    ax_obj.legend(frameon=False, fontsize=8.5, loc="upper right")
    # Zoom on the region that decides identifiability; the two rising branches
    # leave the frame, which is exactly the point about them.
    ax_obj.set_ylim(-1.2, 32.0)
    ax_obj.set_xlim(capacities[0], capacities[-1])

    for ax in (ax_set, ax_obj):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelcolor="#334155")

    fig.suptitle(f"{KWP:.0f} kWp at tilt {TILT}°, azimuth {AZ}°, behind a {KVA:.0f} kVA "
                 f"inverter — {100*clipped.mean():.0f} % of daytime samples clipped",
                 x=0.005, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out, "clipping_degeneracy")

    _save(methods_figure(matrix, ghi, daytime), out, "clipping_methods")

    if args.show:
        plt.show()
    return 0


def methods_figure(matrix, ghi, daytime):
    """Method A rides the top of the data; Method B runs through it."""
    import matplotlib.pyplot as plt

    from pv_orientation_estimator import run_estimation

    timestamps = pd.date_range(START, END, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    clean = synthetic_plant(LAT, LON, ELEV, timestamps, TILT, AZ, KWP,
                            interval="1h", label="end", ac_rating=KVA)
    measured = np.clip(clean * (1 + 0.02 * rng.standard_normal(len(clean))), 0.0, None)

    method_a = run_estimation(matrix, measured, daytime, method="A")
    method_b = run_estimation(matrix, measured, daytime, method="B", ac_rating=KVA)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    window = timestamps < timestamps[0] + pd.Timedelta(days=2)
    for result, colour, name in ((method_a, ABSURD_CAP, "A"), (method_b, TRUE_CAP, "B")):
        curve = np.minimum(matrix @ result["alpha"], result["ac_rating_kw"])
        above = (curve[daytime] >= measured[daytime] - 1e-6).mean()
        ax.plot(timestamps[window], curve[window], "-", color=colour, linewidth=2.1,
                label=f"Method {name} — above {100*above:.0f} % of samples, "
                      f"{result['effective_kWp']:.1f} kWp, "
                      f"S = {result['ac_rating_kw']:.1f} kW")
    ax.plot(timestamps[window], measured[window], "o", markersize=4.5, color=MEASURED,
            zorder=5, label=f"measured, {KWP:.0f} kWp / {KVA:.0f} kVA, 2 % noise")

    ax.set_ylabel("AC power [kW]")
    ax.set_ylim(0, 95)
    ax.set_title("Method A covers the measurements; Method B averages through them",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelcolor="#334155")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    return fig


def _save(fig, out: pathlib.Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, dpi=150)
        print(f"written {path}")


if __name__ == "__main__":
    sys.exit(main())
