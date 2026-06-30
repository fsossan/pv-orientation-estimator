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


def alpha_to_heatmap_grid(alpha: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Reshape alpha (N_LAYOUTS,) into a 2D grid (n_tilts × n_azimuths_eu)
    for a heatmap.  Returns (grid, tilts, azimuths_eu).
    """
    n_tilts = len(TILTS_DEG)
    n_azs   = len(AZIMUTHS_EU_DEG)
    # LAYOUTS ordering: az is outer loop, tilt is inner loop
    grid = alpha.reshape(n_azs, n_tilts).T  # (tilts, azimuth_eu)
    return grid, TILTS_DEG, AZIMUTHS_EU_DEG
