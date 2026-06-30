# pv-orientation-estimator

This library estimates the **tilt**, **azimuth**, and effective capacity (in
kWp) of a PV plant from historical, timestamped AC power measurements. For best
results, the measurements should come from clear-sky days. The repository
provides both Python and MATLAB implementations. The method is loosely based on
[1].

The algorithm works as follows:

- It creates model-based clear-sky global horizontal irradiance (GHI) time
  series for the target timestamps.
- It transposes the GHI time series across a predefined grid of tilt and
  azimuth configurations to compute plane-of-array (POA) irradiance.
- It converts POA irradiance into per-unit PV production with a simple model,
  optionally including the effect of temperature. This produces an exploratory
  signal for each tilt/azimuth configuration: a matrix whose rows are
  timestamps and whose columns are candidate configurations. The unit of this matrix is kW per kWp installed.
- It then solves the following least squares estimation problem:

```
minimize  ‖ P_measured − P_pu @ alpha ‖²    s.t. alpha >= 0
```

where `P_measured` is the measured power (kW), `P_pu` is the matrix described above (kW/kWp), and `alpha` (kWp) is a vector that contains the installed capacity assigned to each explored tilt/azimuth configuration.

**Notes**

- It is assumed that the PV converter is sized for the full DC capacity, so peak power is not
  clipped and the full DC power capacity is reflected in the AC power curve, thus becoming clearly identifiable.
- Owing that `alpha` is a vector, the algorithm can estimate multiple strings at different tilt and azimuth in case of converters with multiple MPPT entries.

**Required inputs** 

The required inputs for the model are:

- timestamped PV generation measurements (kW)
- longitude and latitude of the installation
- optionally, a temperature time series (deg C)



The method fits measured AC power to a grid of clear-sky POA reference profiles
and solves a non-negative least-squares (NNLS) problem for the per-orientation
capacity. The dominant peak identifies the installed orientation.

## Implementation

This repo ships **two independent, self-contained implementations** of the same
method, in separate folders. Both MATLAB and Python have external requirements,
listed below.

| Folder | Language | Requirements | Convex solver | Irradiance model | Notes |
|--------|----------|--------------|---------------|------------------|-------|
| [`python/`](python/) | Python | Python >= 3.10; `numpy`, `pandas`, `pvlib`, `cvxpy` | [cvxpy](https://www.cvxpy.org/) | [pvlib](https://pvlib-python.readthedocs.io/) | Installable package `pv_orientation_estimator`; consumed by the enso service-portal. |
| [`matlab/`](matlab/) | MATLAB | MATLAB R2016b or newer; [CVX](https://cvxr.com/cvx/) | [CVX](https://cvxr.com/cvx/) | self-contained (Ineichen clearsky + NOAA solar position + isotropic transposition) | `+pvorient` package; run `demo_roundtrip.m`. |

Both estimate over the same orientation grid — tilt `0:5:75°` (16 values) and
azimuth `−50:5:45°` relative to south, EU convention (20 values) — i.e. 320
candidate layouts, and expose the same conceptual API
(`buildReferenceMatrix` → `runEstimation` → `formatResultsTable` /
`alphaToHeatmapGrid`).

## Optional temperature correction

By default, the reference profiles are pure per-unit POA irradiance. You can
optionally enable the empirical temperature derating of Sossan et al. [1]
(Eqs. 6–7), which scales the POA irradiance by a cell-temperature factor:

```
T_cell = T_air + κ · POA                          (POA in W/m²)        (7)
P      = (POA / 1000) · [1 + γ · (T_cell − 25)] · P_nom               (6)
```

It is **off by default**: pass an ambient temperature (a scalar °C or a
per-timestamp series) to `buildReferenceMatrix` to turn it on; omit it and you
get the original model unchanged.

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| ambient temperature | `T_air` | — (off) | air temperature [°C]; scalar or length-`T` series |
| power temp. coefficient | `γ` | `−0.0043` 1/°C | module derating per °C (polycrystalline) |
| cell-temp. rise coefficient | `κ` | `0.038` °C·m²/W | POA-to-cell-temperature rise (mixed mounting) |
| reference temperature | `T_ref` | `25` °C | STC reference |

The correction only rescales each column, so the fitted `alpha` keeps its
meaning as the **STC nominal capacity** `P_nom` (kWp).

```python
# Python
P_pu, ghi = build_reference_matrix(lat, lon, elev, timestamps, air_temp=t_ambient)
```
```matlab
% MATLAB
[Ppu, ghi] = pvorient.buildReferenceMatrix(lat, lon, elev, times, 'AirTemp', Tair);
```

See each folder's `README.md` for install and usage details.

[1]: F. Sossan et al. (2018). Unsupervised disaggregation of photovoltaic production from composite power flow measurements of heterogeneous prosumers. IEEE Transactions on Industrial Informatics, 14(9), 3904-3913.

[2]: F. Sossan, E. Scolari,  et al. (2019). Solar irradiance estimations for modeling the variability of photovoltaic generation and assessing violations of grid constraints: A comparison between satellite and pyranometers measurements with load flow simulations. Journal of Renewable and Sustainable Energy, 11(5).
