"""
Post-processing helpers for the NNLS solution vector ``alpha``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_orientation_estimator.grid import AZIMUTHS_EU_DEG, LAYOUTS, LAYOUT_LABELS, TILTS_DEG


def format_results_table(alpha: np.ndarray) -> pd.DataFrame:
    """
    Return a DataFrame of orientations sorted by attributed kWp.
    Only includes rows with > 1 % of total capacity.
    """
    total = alpha.sum()
    threshold = 0.01 * total if total > 0 else 0.0
    rows = []
    for kWp, label, (tilt, az) in zip(alpha, LAYOUT_LABELS, LAYOUTS):
        if kWp > threshold:
            rows.append({
                "tilt_deg":     tilt,
                "azimuth_eu":   az,
                "label":        label,
                "capacity_kwp": round(float(kWp), 2),
                "share_pct":    round(100.0 * float(kWp) / total, 1) if total > 0 else 0.0,
            })
    return pd.DataFrame(rows).sort_values("capacity_kwp", ascending=False).reset_index(drop=True)


def alpha_to_heatmap_grid(alpha: np.ndarray) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Arrange alpha (N_LAYOUTS,) on a tilt x azimuth grid for a heatmap.

    Returns ``(grid, tilts, azimuths_eu)`` with the axes taken from the layouts
    actually searched. The default grid is equidistributed over the hemisphere
    rather than a Cartesian product, so most cells of the rectangle it spans
    have no layout behind them: those come back as ``NaN``, which a heatmap
    should draw as a gap (``hoverongaps=False``) rather than as zero capacity.
    Reshaping instead of scattering would silently mis-place every value the
    moment the grid stopped being rectangular.
    """
    tilts = sorted({t for t, _ in LAYOUTS})
    azimuths = sorted({a for _, a in LAYOUTS})
    row = {t: i for i, t in enumerate(tilts)}
    col = {a: k for k, a in enumerate(azimuths)}

    grid = np.full((len(tilts), len(azimuths)), np.nan)
    for value, (tilt, azimuth) in zip(np.asarray(alpha, dtype=float), LAYOUTS):
        grid[row[tilt], col[azimuth]] = value
    return grid, tilts, azimuths
