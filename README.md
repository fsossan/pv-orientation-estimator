# pv-orientation-estimator

This library estimates the **tilt**, **azimuth**, and effective capacity (in
kWp) of a PV plant from historical, timestamped AC power measurements. The measurements should come from clear-sky days. The repository provides both Python and MATLAB implementations. The method is loosely based on [1].

The algorithm works as follows:

- It creates model-based clear-sky global horizontal irradiance (GHI) time
  series for the target timestamps.
- It transposes the GHI time series across a predefined grid of tilt and
  azimuth configurations to compute plane-of-array (POA) irradiance.
- It converts POA irradiance into per-unit PV production with a simple model,
  optionally including the effect of temperature. This produces an exploratory
  signal for each tilt/azimuth configuration, which has the shape of a matrix (denoted in the following as `P_pu`) whose rows are timestamps and whose columns are candidate configurations. The unit of this matrix is kW per kWp installed.
- It then solves the following least-squares estimation problem:

```
minimize  ‖ P_measured − P_pu @ alpha ‖²    s.t. alpha >= 0
```

where `P_measured` is the measured power of the installation (in kW), `P_pu` is the matrix described above (kW/kWp), and `alpha` (kWp) is a vector that contains the installed capacity assigned to each explored tilt/azimuth configuration.

The output of the problem is `alpha` and will tell how much capacity (kWp) was found at each candidate configuration.

**Notes**

- It is assumed that the PV converter is sized for the full DC capacity, so peak power is not
  clipped and the full DC power capacity is reflected in the AC power curve, thus becoming clearly identifiable.
- It is assumed that the converter operates with a constant efficiency from zero till the converter capacity. 
- Owing that `alpha` is a vector, the algorithm can estimate multiple strings at different tilt and azimuth in case of converters with multiple MPPT entries.

**Usage in one call (Python)**

```python
from pv_orientation_estimator import estimate_orientation

result = estimate_orientation(lat=46.52, lon=6.63, elev=500.0, power=measured_kw,
                              interval="15min", label="end")
```

where `power` is a pandas Series of measured AC power indexed by UTC timestamps.
The dictionary is built and integrated to match how the measurements are
stamped; the two-stage API underneath stays available for reusing a dictionary.

**Required inputs** 

The required inputs for the model are:

- timestamped PV generation measurements (kW)
- longitude and latitude of the installation
- optionally, a temperature time series (deg C)



The method fits measured AC power to a grid of clear-sky POA reference profiles
and solves a non-negative least-squares (NNLS) problem for the per-orientation
capacity. The dominant peak identifies the installed orientation.

## Documentation

The algorithm is written up in LaTeX in [`doc/`](doc/) — the clear-sky
reference dictionary, the optimization problem (formulation, convexity, KKT
conditions and the sparsity they induce, solvers), the post-processing of
`alpha`, and the identifiability caveats.

```bash
cd doc && make      # -> doc/algorithm.pdf
```

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
optionally enable an empirical temperature derating (Sossan et al. [1] Eqs. 6–7 and associated references)
, which scales the POA irradiance by a cell-temperature factor:

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
P_pu, ghi = build_pu_power_matrix(lat, lon, elev, timestamps, air_temp=t_ambient)
```
```matlab
% MATLAB
[Ppu, ghi] = pvorient.buildReferenceMatrix(lat, lon, elev, times, 'AirTemp', Tair);
```

## Inverter clipping (kWp > kVA)

When a plant has more DC capacity than its inverter passes, the measurements are
cut off at the AC rating. Those samples are right-censored — they say the array
produced *at least* the rating — and fitting them as equalities makes a flat top
look like a shallower, smaller array. Pass the rating and they enter the
objective one-sidedly instead:

```python
result = run_estimation(P_pu, P_measured, daytime, ac_rating=75.0)
```

Two estimators are available. **Method A** asks for the smallest plant whose
production still covers every measurement — a linear program needing no rating
at all, exact on clean data. **Method B** keeps least squares and writes the
censoring into the residual, which averages over noise rather than chasing its
extremes. The crossover is around 1 % of unmodelled error, so B is the default:

```python
result = run_estimation(P_pu, P_measured, daytime, method="A")                  # no rating needed
result = run_estimation(P_pu, P_measured, daytime, method="B", ac_rating=75.0)
```

On a 100 kWp plant behind a 75 kVA inverter, ignoring the clipping costs 15° of
tilt, 30° of azimuth and 6.6 % of capacity. See
[`python/README.md`](python/README.md#inverter-clipping-kwp--kva).

## Interval-averaged measurements

Metered power is an average over an interval, not a reading at an instant.
Comparing it against a model evaluated at the timestamp is a bias that does not
average out: on hourly data, an undeclared end-of-interval convention moves the
recovered azimuth by **~30°** while the fit still reports R² > 0.999.

The Python implementation therefore lets the measurement interval be declared,
and integrates the reference profiles over it:

```python
P_pu, ghi = build_pu_power_matrix(lat, lon, elev, timestamps,
                                   interval="15min", label="end")
```

Correctly declared hourly data then recovers the orientation as accurately as
5-minute data — the sampling convention matters, the resolution barely does.
See [`python/README.md`](python/README.md#interval-averaged-measurements).

## Terrain horizon (PVGIS)

The Python package also ships a small standalone utility to download the
**topographic horizon** of a site from [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/)
— the skyline elevation angle in every direction, from PVGIS' digital
elevation model — and plot it together with the solstice sun paths.

The estimator accounts for it: the direct beam is switched off while the terrain
hides the sun, and the sky-diffuse is scaled by the sky view factor of each
candidate plane. Combined with the interval integration above, the blocking
becomes the *fraction* of each interval the beam is lost for. On a shaded valley
site, ignoring the terrain costs 10° of tilt and 6 % of capacity.

```python
P_pu, ghi = build_pu_power_matrix(lat, lon, elev, timestamps,
                                  interval="1h", label="end",
                                  horizon=download_horizon(lat, lon))
```

```bash
cd python
python scripts/download_horizon.py --demo                       # one location (Wimmis BE), downloaded and plotted
python scripts/download_horizon.py --lat 46.02 --lon 7.75 --png horizon.png
```

Azimuths follow the same EU convention as the orientation grid (0° = south).
Swiss sites are additionally reverse-geocoded, so the plot is titled with the
commune. See [`python/README.md`](python/README.md#terrain-horizon-pvgis) for
the API.

## Geocoding (swisstopo)

A companion utility resolves Swiss sites in both directions through the free
[GeoAdmin API](https://api3.geo.admin.ch/) — coordinates to commune, and an
address to coordinates:

```bash
cd python
python scripts/geocode.py "Bahnhofstrasse 1, 3920 Zermatt"   # -> 46.02328, 7.74807
python scripts/geocode.py --lat 46.0207 --lon 7.7491         # -> 3920 Zermatt (VS)
```

Switzerland only; see [`python/README.md`](python/README.md#geocoding) for the
caveats.

A synthetic end-to-end demo runs the whole chain — generate a plant of known
geometry, estimate it back, plot the fit:

```bash
cd python && python scripts/demo_estimate.py     # Python
```
```matlab
cd matlab, demo_roundtrip                        % MATLAB
```

See each folder's `README.md` for install and usage details.

[1]: F. Sossan et al. (2018). Unsupervised disaggregation of photovoltaic production from composite power flow measurements of heterogeneous prosumers. IEEE Transactions on Industrial Informatics, 14(9), 3904-3913.

[2]: F. Sossan, E. Scolari,  et al. (2019). Solar irradiance estimations for modeling the variability of photovoltaic generation and assessing violations of grid constraints: A comparison between satellite and pyranometers measurements with load flow simulations. Journal of Renewable and Sustainable Energy, 11(5).
