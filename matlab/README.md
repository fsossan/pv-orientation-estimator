# pv-orientation-estimator — MATLAB

MATLAB port of the PV tilt/azimuth estimator. It fits measured AC power to a
grid of clearsky plane-of-array (POA) reference profiles and solves a
non-negative least-squares (NNLS) problem with **[CVX](https://cvxr.com/cvx/)**:

```
minimize  ‖ P_measured − P_pu * alpha ‖²    s.t. alpha >= 0
```

The dominant peak in `alpha` identifies the installed `(tilt, azimuth)`.

## Requirements

- **MATLAB** R2016b or newer (uses `datetime`, `table`, `round(x,n)`).
- **CVX** toolbox — the only external dependency, used by `runEstimation`.
  Download from <https://cvxr.com/cvx/> and run its `cvx_setup`.

The irradiance model is **self-contained**: clearsky (Ineichen), solar position
(NOAA), and the isotropic transposition are reimplemented in base MATLAB, so no
PV/solar toolbox (e.g. Sandia PV_LIB) is required.

## Install

Add this folder to the MATLAB path so the `+pvorient` package is visible:

```matlab
addpath('/path/to/pv-orientation-estimator/matlab');
```

(`demo_roundtrip.m` does this automatically for its own folder.)

## Usage

```matlab
% Hourly UTC timestamps matching your measured series:
times = (datetime(2023,1,1,'TimeZone','UTC') : hours(1) : ...
         datetime(2023,12,31,'TimeZone','UTC')).';
Pmeasured = ...;   % T x 1 AC power [kW], aligned to `times`

% Per-unit clearsky reference profiles for the site:
[Ppu, ghi] = pvorient.buildReferenceMatrix(lat, lon, elev, times);

% Daytime mask, then estimate:
g       = pvorient.orientationGrid();
daytime = ghi > g.DAYTIME_GHI_THRESHOLD;
result  = pvorient.runEstimation(Ppu, Pmeasured, daytime);

fprintf('tilt=%d  az=%d  kWp=%.1f  r2=%.3f\n', ...
        result.best_tilt, result.best_az_eu, result.effective_kWp, result.r2);

disp(pvorient.formatResultsTable(result.alpha));
[grid, tilts, azs] = pvorient.alphaToHeatmapGrid(result.alpha);
```

Run the synthetic round-trip demo:

```matlab
>> demo_roundtrip
```

## Package contents (`+pvorient`)

| Function | Purpose |
|----------|---------|
| `orientationGrid`     | Tilt/azimuth grid (16×20 = 320 layouts) and shared constants |
| `solarPosition`       | Apparent zenith + azimuth (NOAA), toolbox-free |
| `ineichenClearsky`    | GHI/DNI/DHI clearsky (Ineichen), constant Linke turbidity |
| `buildReferenceMatrix`| `(T×320)` per-unit POA matrix + clearsky GHI |
| `runEstimation`       | NNLS fit via CVX → result struct |
| `formatResultsTable`  | Orientations with > 1 % of inferred capacity (as a `table`) |
| `alphaToHeatmapGrid`  | Reshape `alpha` into a `(tilt × azimuth)` grid |

## Conventions & notes

- **Azimuth (EU):** degrees relative to **south**, negative east / positive
  west; pvlib north-clockwise azimuth is `180 + azimuth_eu`.
- **Per-unit:** POA is divided by 1000 W/m², so `alpha` is in kWp when
  `Pmeasured` is in kW.
- **`best_idx` is 1-based** (MATLAB), unlike the 0-based `best_idx` in the
  Python package.
- **Linke turbidity** defaults to `3.0`; override via
  `buildReferenceMatrix(..., 'LinkeTurbidity', TL)`. Ground albedo defaults to
  `0.25` (`'Albedo'`).
- **Optional temperature correction** (Sossan et al., Eqs. 6–7): pass
  `'AirTemp'` (a scalar degC or a `T x 1` vector) to rescale each column by
  `1 + Gamma*(Tcell - TempRef)` with `Tcell = AirTemp + TempRiseCoeff*POA`.
  Defaults `Gamma = -0.0043`, `TempRiseCoeff = 0.038`, `TempRef = 25`. The
  correction only rescales the columns, so `alpha` stays the STC nominal
  capacity `P_nom`. Example:
  `buildReferenceMatrix(..., 'AirTemp', Tair)`. Without `'AirTemp'` the matrix
  is the plain per-unit POA irradiance.
- This is an **independent reimplementation** of the irradiance physics, not a
  binding to pvlib. Absolute values may differ slightly from the Python
  package, but the implementation is internally self-consistent (the
  round-trip demo recovers the planted orientation exactly).
