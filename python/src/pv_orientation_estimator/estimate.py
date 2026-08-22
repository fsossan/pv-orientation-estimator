"""
NNLS orientation estimation.

Solves a non-negative least-squares fit of measured AC power against the
clear-sky per-unit power dictionary:

    minimize  ‖ P_measured − P_pu @ alpha ‖²    s.t. alpha >= 0

alpha_i is the effective kWp attributed to layout i.  The dominant peak in
alpha reveals the estimated tilt and azimuth.

Inverter clipping
-----------------
A plant with more DC capacity than AC rating (kWp > kVA) has its production
cut off at the rating for part of the day.  Those samples are *right-censored*:
they say the array produced at least the rating, not how much more.  Fitting
them as if they were plain measurements drags the estimate towards a flatter,
lower array.

Two estimators handle it, selected with ``method``:

**Method A** — the smallest capacity whose production still covers every
measurement::

    minimize  1' alpha        s.t.  P_pu @ alpha >= P_measured,  alpha >= 0

A linear program.  It needs no rating and no censoring threshold, and returns
the rating as a by-product, S = max(P).  Being a covering criterion it fits the
*upper envelope* of the data, so it is exact on clean measurements and inflates
with noise --- every upward excursion has to be cleared.

**Method A1** — Method A with the rating supplied instead of estimated, which
is what makes it comparable with Method B::

    minimize  1' alpha   s.t.  P_pu @ alpha >= min(P_measured, S),  alpha >= 0

A reading above S cannot be covered by any array (the inverter caps the model
too), so the binding requirement there is only that the DC production reach S;
the covering target is the measurement clipped at the rating.  This removes the
plateau's contribution to the inflation but not the shoulders', so it recovers
the capacity better than Method A and the tilt no better.

**Method B** — censored least squares: the free samples are fitted as usual,
the saturated ones only one-sidedly::

    minimize  Σ_free (P - P_pu @ alpha)²  +  Σ_clipped max(0, S - P_pu @ alpha)²

which needs the rating ``S`` but averages over noise instead of chasing its
extremes.

Measured on a 100 kWp / 75 kVA synthetic plant, capacity-weighted centroid of
the tilt and total capacity against a truth of 30° and 100 kWp:

=======  =================  =================  =================
noise    Method A           Method A1          Method B
=======  =================  =================  =================
0 %      30.0°, 100.0 kWp   30.0°, 100.0 kWp   33.6°, 103.1 kWp
1 %      26.3°, 101.6 kWp   25.0°, 101.0 kWp   33.0°, 102.5 kWp
2 %      22.7°, 103.4 kWp   21.9°, 102.3 kWp   32.5°, 102.1 kWp
4 %      15.7°, 108.4 kWp   15.0°, 106.1 kWp   23.6°,  97.2 kWp
=======  =================  =================  =================

so the covering methods below roughly 1 % total error, Method B above it.
"Error" here means everything the model does not capture --- clear-sky
mismatch, soiling, temperature --- not just meter noise.
"""

from __future__ import annotations

from typing import Optional, TypedDict, Union

import numpy as np
import cvxpy as cp

from pv_orientation_estimator.grid import LAYOUTS, LAYOUT_LABELS, N_LAYOUTS

# A sample counts as sitting on the inverter's plateau when it is within this
# fraction of it.  It has to exceed the measurement noise, or a noisy plateau
# is not recognised as one.
CLIP_TOLERANCE = 0.02

# Width of the band below the rating whose samples are treated as clipped.
# Noise scatters a clipped reading either side of the rating, and a reading
# that is really clipped but lands below the band is fitted as if the plant had
# genuinely produced that little, which drags the whole fit down.  Measured on
# a 100 kWp / 75 kVA plant: the band has to be about three times the noise
# (2 % noise needs 6 %), while an over-wide band costs only ~1 % of capacity.
CLIP_BAND = 0.06

# --- constants of the plateau detector (see detect_ac_rating) --------------
# A clipped plant piles samples up at one power level.  The detector looks for
# that atom in the histogram of daytime power, then checks the level really is
# a ceiling.  The thresholds below were calibrated on synthetic plants at DC/AC
# ratios of 1.0 to 1.6 and 0-4 % measurement noise.
CLIP_HIST_BINS = 40
CLIP_SPIKE_RATIO = 5.0      # tallest upper bin, relative to a typical upper bin
CLIP_CEILING_GUARD = 3.0    # samples this many tolerances above the level ...
CLIP_MAX_ABOVE_SHARE = 0.15 # ... must stay below this share of the plateau


class EstimationResult(TypedDict, total=False):
    """
    Result of :func:`run_estimation`.

    On failure only ``status`` and ``alpha`` (None) are populated.  On success
    all keys are present.
    """
    status: str                  # solver status, e.g. "optimal"
    alpha: Optional[np.ndarray]  # (N_LAYOUTS,) attributed kWp per layout, or None
    effective_kWp: float         # sum(alpha)
    best_idx: int                # argmax(alpha)
    best_label: str              # "tilt,azimuth" label of the dominant layout
    best_tilt: int               # tilt of the dominant layout (deg)
    best_az_eu: int              # azimuth of the dominant layout (deg, EU convention)
    r2: float                    # coefficient of determination on daytime points
    rmse_kw: float               # RMSE on daytime points (kW)
    method: Optional[str]        # clipping model used: None, "A", "A1" or "B"
    ac_rating_kw: Optional[float]  # AC rating the fit clipped at, or None
    clipped_share: float         # share of daytime samples treated as clipped
    dc_ac_ratio: Optional[float]  # effective_kWp / ac_rating_kw, or None


def detect_ac_rating(
    P_daytime: np.ndarray,
    tolerance: float = CLIP_TOLERANCE,
) -> Optional[float]:
    """
    Read an inverter's AC rating off the measurements, or return ``None``.

    **Prefer passing the nameplate rating.**  It is registry data, and this
    detector is a fallback for when it is genuinely unavailable.

    How it works: clipping puts an atom in the distribution of power, a pile of
    samples at one level.  The tallest bin in the upper half of the histogram
    is taken as a candidate when it stands
    :data:`CLIP_SPIKE_RATIO` times above a typical bin there, and is accepted
    only if it behaves like a ceiling — almost nothing above it.  The rating is
    then the median of the samples on the plateau, which the noise smears
    around it.

    Measured behaviour on synthetic plants (7 weeks of hourly clear-sky data,
    tilt ​30°, azimuth −20°):

    ==========  =====================================================
    DC/AC       result
    ==========  =====================================================
    ≤ 1.05      not detected — and not needed, the bias is negligible
    1.1 – 1.6   detected up to ~2 % noise, level within 4 %
    any         missed at 4 % noise below DC/AC 1.33
    ==========  =====================================================

    It errs towards ``None``: a missed detection leaves the plain fit, an
    invented one would censor a real peak.
    """
    P_daytime = np.asarray(P_daytime, dtype=float)
    if len(P_daytime) == 0 or P_daytime.max() <= 0.0:
        return None

    # Ignore the near-dark tail: it dominates the histogram otherwise.
    lit = P_daytime[P_daytime > 0.05 * P_daytime.max()]
    if len(lit) == 0:
        return None

    counts, edges = np.histogram(lit, bins=CLIP_HIST_BINS)
    centres = 0.5 * (edges[:-1] + edges[1:])
    upper = centres > 0.5 * lit.max()
    if not upper.any():
        return None

    typical = max(float(np.median(counts[upper])), 1.0)
    if counts[upper].max() / typical < CLIP_SPIKE_RATIO:
        return None                                   # no atom: nothing clipped

    level = float(centres[upper][np.argmax(counts[upper])])
    plateau = np.abs(lit - level) <= tolerance * level
    above = lit > level * (1.0 + CLIP_CEILING_GUARD * tolerance)
    if above.sum() > CLIP_MAX_ABOVE_SHARE * plateau.sum():
        return None                                   # a busy peak, not a ceiling

    return float(np.median(lit[plateau]))


def _resolve_method(method: Optional[str], ac_rating) -> Optional[str]:
    """
    Normalise ``method``, keeping the older ``ac_rating``-only call working.

    Passing a rating and no method means Method B, which is what that call did
    before the methods were named.
    """
    if method is None:
        return "B" if ac_rating is not None else None

    normalised = str(method).strip().upper()
    if normalised not in ("A", "A1", "B"):
        raise ValueError(f"method must be 'A', 'A1', 'B' or None, got {method!r}")
    if normalised in ("A1", "B") and ac_rating is None:
        raise ValueError(
            f"method={normalised!r} needs an ac_rating: the nameplate value in kW, "
            "or 'auto' to read it off the data.  method='A' needs no rating."
        )
    return normalised


def run_estimation(
    P_pu: np.ndarray,
    P_measured: np.ndarray,
    daytime_mask: np.ndarray,
    method: Optional[str] = None,
    ac_rating: Optional[Union[float, str]] = None,
    clip_band: float = CLIP_BAND,
) -> EstimationResult:
    """
    Solve NNLS on daytime-filtered data:
        min ‖ P_meas[day] − P_pu[day] @ alpha ‖²   s.t. alpha >= 0

    Inverter clipping
    -----------------
    ``method`` selects how a plant whose DC capacity exceeds its AC rating is
    handled (see the module docstring for the comparison):

    * ``None`` (default) — no clipping model, the plain NNLS above.  Passing an
      ``ac_rating`` without a method selects Method B.
    * ``"A"`` — smallest capacity covering every measurement::

          min 1'α   s.t.  P_pu α >= P_measured,  α >= 0

      No rating needed: it comes back in ``ac_rating_kw`` as the highest
      measurement.  Exact on clean data, inflates with noise.

    * ``"A1"`` — Method A with ``ac_rating`` supplied rather than estimated,
      which is what makes it comparable with Method B::

          min 1'α   s.t.  P_pu α >= min(P_measured, S),  α >= 0

      No reading above ``S`` is coverable, so the requirement there is only
      that the DC production reach the rating.

    * ``"B"`` — censored least squares, needs ``ac_rating``::

          min Σ_free (P_t − (P_pu α)_t)² + Σ_clipped max(0, S − (P_pu α)_t)²
          s.t. α >= 0

      The censored term penalises the model only for predicting *less* than the
      rating where the meter saturated, which is all such a sample says.

    Both are convex, and in both ``alpha`` keeps its meaning: the **DC**
    capacity in kWp, which may exceed the rating.

    ``ac_rating`` (Method B) is a number in kW, or ``"auto"`` to read it off
    the data with :func:`detect_ac_rating`.  ``clip_band`` sets how far below
    the rating a sample still counts as clipped and has to cover the noise
    (see :data:`CLIP_BAND`).  Because production above the cap is never
    observed, the DC capacity is only weakly bounded from above and comes out
    1-3 % high.

    Returns an :class:`EstimationResult`.
    """
    method = _resolve_method(method, ac_rating)
    P_pu_dt   = P_pu[daytime_mask]
    P_meas_dt = P_measured[daytime_mask]

    alpha = cp.Variable(N_LAYOUTS, nonneg=True)

    if method in ("A", "A1"):
        if method == "A":
            # The rating is not an input: at the optimum the ceiling touches
            # the data, so the highest measurement is the plateau.
            rating = float(np.max(P_meas_dt)) if len(P_meas_dt) else None
            target = P_meas_dt
        else:
            rating = (detect_ac_rating(P_meas_dt) if ac_rating == "auto"
                      else float(ac_rating))
            # A reading above the rating is not coverable by any array — the
            # inverter caps the model too — so ask only that the DC production
            # reach the rating there.
            target = P_meas_dt if rating is None else np.minimum(P_meas_dt, rating)
        prob = cp.Problem(cp.Minimize(cp.sum(alpha)), [P_pu_dt @ alpha >= target])
    else:
        rating = (detect_ac_rating(P_meas_dt) if ac_rating == "auto"
                  else None if ac_rating is None else float(ac_rating))
        clipped = (np.zeros(len(P_meas_dt), dtype=bool) if rating is None
                   else P_meas_dt >= rating * (1.0 - clip_band))

        residual = cp.sum_squares(P_meas_dt[~clipped] - P_pu_dt[~clipped] @ alpha)
        if clipped.any():
            # One-sided: only under-prediction at a censored sample is an error.
            residual = residual + cp.sum_squares(
                cp.pos(rating - P_pu_dt[clipped] @ alpha))
        prob = cp.Problem(cp.Minimize(residual))

    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate") or alpha.value is None:
        return {"status": prob.status, "alpha": None}

    a = np.clip(alpha.value, 0.0, None)

    # Fit quality on daytime points, against what the meter would have seen:
    # the modelled DC production after the inverter cuts it at the rating.
    P_fitted = P_pu_dt @ a if rating is None else np.minimum(P_pu_dt @ a, rating)
    ss_res   = float(np.sum((P_meas_dt - P_fitted) ** 2))
    ss_tot   = float(np.sum((P_meas_dt - P_meas_dt.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse     = float(np.sqrt(ss_res / len(P_meas_dt)))

    best_idx = int(np.argmax(a))
    best_tilt, best_az_eu = LAYOUTS[best_idx]

    return {
        "status":        prob.status,
        "alpha":         a,
        "effective_kWp": float(a.sum()),
        "best_idx":      best_idx,
        "best_label":    LAYOUT_LABELS[best_idx],
        "best_tilt":     best_tilt,
        "best_az_eu":    best_az_eu,
        "r2":            r2,
        "rmse_kw":       rmse,
        "method":        method,
        "ac_rating_kw":  rating,
        "clipped_share": (0.0 if rating is None else
                          float((P_meas_dt >= rating * (1.0 - clip_band)).mean())),
        "dc_ac_ratio":   None if rating in (None, 0.0) else float(a.sum() / rating),
    }
