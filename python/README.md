# pv-orientation-estimator

Estimate the **tilt** and **azimuth** (and effective capacity) of a PV array
from its measured AC power.

It fits the measured AC power of a plant to a grid of clearsky plane-of-array
(POA) reference profiles and finds which `(tilt, azimuth)` orientation best
explains the data.

## Method

A non-negative least-squares (NNLS) problem is solved:

```
minimize  ‖ P_measured − P_pu @ alpha ‖²    s.t. alpha >= 0
```

`P_pu` is a `(T × 320)` matrix of per-unit POA irradiance profiles computed with
[**pvlib**](https://pvlib-python.readthedocs.io/) (Ineichen clearsky model) for
every combination of:

- **Tilt:** 0°, 5°, …, 75° (16 values)
- **Azimuth:** −50°, …, +45° relative to south (20 values, European convention)

The solution `alpha` (one value per orientation, in kWp) is naturally sparse;
its dominant peak identifies the estimated installation geometry. The NNLS
program is solved with [**cvxpy**](https://www.cvxpy.org/) (CLARABEL solver).

This package is **self-contained**: it depends only on `numpy`, `pandas`,
`pvlib`, and `cvxpy`, and returns a plain result dict — no project-specific or
domain-model dependencies.

## Install

```bash
pip install -e .          # editable, for development
# or
pip install .
```

## Usage

```python
import pandas as pd
from pv_orientation_estimator import (
    build_reference_matrix, run_estimation,
    DAYTIME_GHI_THRESHOLD, format_results_table,
)

# Hourly, timezone-aware (UTC) timestamps of your measured series.
timestamps = pd.DatetimeIndex(...)        # the index of P_measured
P_measured = ...                          # 1-D np.ndarray of AC power (kW), same length

# Build per-unit clearsky reference profiles for the site.
P_pu, clearsky_ghi = build_reference_matrix(lat=46.52, lon=6.63, elev=500.0,
                                            timestamps=timestamps)

# Restrict the fit to daytime samples.
daytime = clearsky_ghi > DAYTIME_GHI_THRESHOLD

result = run_estimation(P_pu, P_measured, daytime)
print(result["best_tilt"], result["best_az_eu"], result["effective_kWp"], result["r2"])

# Orientations holding > 1 % of the inferred capacity:
print(format_results_table(result["alpha"]))
```

### Optional temperature correction

By default the reference profiles are pure per-unit POA irradiance. You can
optionally apply the empirical temperature derating of Sossan et al.
(*Solar Irradiance Estimations for Modeling the Variability of Photovoltaic
Generation…*, Eqs. 6–7):

```
T_cell = air_temp + temp_rise_coeff · POA          # POA in W/m²        (7)
P      = (POA / 1000) · [1 + gamma · (T_cell − 25)] · P_nom            (6)
```

Pass an ambient temperature (a scalar °C or an array of length `T`):

```python
P_pu, ghi = build_reference_matrix(
    lat, lon, elev, timestamps,
    air_temp=t_ambient,        # °C, scalar or length-T array — enables the model
    gamma=-0.0043,             # power temp. coeff. [1/°C] (polycrystalline)
    temp_rise_coeff=0.038,     # cell-temp rise [°C·m²/W] (mixed mounting)
    temp_ref=25.0,             # STC reference [°C]
)
```

Defaults (`GAMMA_DEFAULT`, `TEMP_RISE_COEFF_DEFAULT`, `TEMP_REF_C`, all
exported) reproduce the paper's values. The correction only rescales each
column, so `alpha` still comes out as the **STC nominal capacity** `P_nom`
(kWp). With `air_temp=None` (default) no correction is applied.

### Result schema (`EstimationResult`)

| key             | meaning                                              |
|-----------------|------------------------------------------------------|
| `status`        | solver status (e.g. `"optimal"`)                     |
| `alpha`         | `(320,)` kWp attributed to each orientation, or `None` |
| `effective_kWp` | `sum(alpha)` — inferred capacity                     |
| `best_idx`      | `argmax(alpha)`                                      |
| `best_label`    | `"tilt,azimuth"` of the dominant orientation         |
| `best_tilt`     | tilt of the dominant orientation (°)                 |
| `best_az_eu`    | azimuth of the dominant orientation (°, EU)          |
| `r2`            | R² of the fit on daytime points                      |
| `rmse_kw`       | RMSE of the fit on daytime points (kW)               |

On solver failure only `status` and `alpha` (`None`) are populated.

## Conventions

- **Azimuth (EU):** degrees relative to **south**, negative toward east,
  positive toward west. pvlib's north-clockwise azimuth is `180 + az_eu`.
- **Per-unit:** POA irradiance is divided by 1000 W/m², so `alpha` comes out in
  kWp when `P_measured` is in kW.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite performs a synthetic round-trip (generate power from a known
orientation, recover it) and needs no database or network access.
