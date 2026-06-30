"""
NNLS orientation estimation.

Solves a non-negative least-squares fit of measured AC power against the
clearsky reference matrix:

    minimize  ‖ P_measured − P_pu @ alpha ‖²    s.t. alpha >= 0

alpha_i is the effective kWp attributed to layout i.  The dominant peak in
alpha reveals the estimated tilt and azimuth.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np
import cvxpy as cp

from pv_orientation_estimator.grid import LAYOUTS, LAYOUT_LABELS, N_LAYOUTS


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


def run_estimation(
    P_pu: np.ndarray,
    P_measured: np.ndarray,
    daytime_mask: np.ndarray,
) -> EstimationResult:
    """
    Solve NNLS on daytime-filtered data:
        min ‖ P_meas[day] − P_pu[day] @ alpha ‖²   s.t. alpha >= 0

    Returns an :class:`EstimationResult`.
    """
    P_pu_dt   = P_pu[daytime_mask]
    P_meas_dt = P_measured[daytime_mask]

    alpha = cp.Variable(N_LAYOUTS, nonneg=True)
    obj   = cp.Minimize(cp.sum_squares(P_meas_dt - P_pu_dt @ alpha))
    prob  = cp.Problem(obj)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate") or alpha.value is None:
        return {"status": prob.status, "alpha": None}

    a = np.clip(alpha.value, 0.0, None)

    # Fit quality on daytime points
    P_fitted = P_pu_dt @ a
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
    }
