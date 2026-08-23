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

One call — measurements in, orientation out:

```python
import pandas as pd
from pv_orientation_estimator import estimate_orientation, format_results_table

power = pd.Series(...)          # measured AC power [kW], DatetimeIndex in UTC

result = estimate_orientation(
    lat=46.52, lon=6.63, elev=500.0, power=power,
    interval="1h", label="end",     # what the timestamps mean — see below
)
print(result["best_tilt"], result["best_az_eu"], result["effective_kWp"], result["r2"])
print(format_results_table(result["alpha"]))
```

`interval` and `label` describe *the measurements*, and the dictionary is
integrated to match — which is the point of the one-stop form. Split across two
calls the two can silently disagree, and that mismatch is a bias of tens of
degrees (see [Interval-averaged measurements](#interval-averaged-measurements)).
`horizon=`, `method=`/`ac_rating=` and `air_temp=` pass straight through, so the
terrain, clipping and temperature options are all available here too.

### In two stages

Building the dictionary is far more expensive than solving, so when one is
reused — several plants at the same site, or one plant under several clipping
models — build it once:

```python
import pandas as pd
from pv_orientation_estimator import (
    build_pu_power_matrix, run_estimation,
    DAYTIME_GHI_THRESHOLD, format_results_table,
)

# Hourly, timezone-aware (UTC) timestamps of your measured series.
timestamps = pd.DatetimeIndex(...)        # the index of P_measured
P_measured = ...                          # 1-D np.ndarray of AC power (kW), same length

# Build per-unit clearsky reference profiles for the site.
P_pu, clearsky_ghi = build_pu_power_matrix(lat=46.52, lon=6.63, elev=500.0,
                                            timestamps=timestamps)

# Restrict the fit to daytime samples.
daytime = clearsky_ghi > DAYTIME_GHI_THRESHOLD

result = run_estimation(P_pu, P_measured, daytime)
print(result["best_tilt"], result["best_az_eu"], result["effective_kWp"], result["r2"])

# Orientations holding > 1 % of the inferred capacity:
print(format_results_table(result["alpha"]))
```

### Temperature: three modes

`air_temp` selects how the cells' temperature is handled:

| `air_temp` | what is modelled |
|---|---|
| `None` (default) | nothing — plain per-unit POA irradiance |
| a scalar °C | cells heating with the irradiance, over a constant ambient |
| an array of length T | the same, over a measured ambient series |

```python
from pv_orientation_estimator import TEMP_AIR_STC_C   # 25.0

P_pu, ghi = build_pu_power_matrix(lat, lon, elev, timestamps,
                                  air_temp=TEMP_AIR_STC_C)     # no data needed
```

Each column is scaled by the empirical factor of Sossan et al. (Eqs. 6–7):

```
T_cell = air_temp + temp_rise_coeff · POA          # POA in W/m²        (7)
factor = 1 + gamma · (T_cell − temp_ref)                                (6)
```

**The middle mode is worth more than it looks.** With a constant ambient the
factor splits into

```
1 + gamma·(air_temp − 25)   +   gamma·temp_rise_coeff·POA
      a constant scaling            needs no data at all
```

and the second term is the one carrying information: it flattens the midday
peak relative to the shoulders (−16 % at 1000 W/m²), which is the feature the
azimuth is read from. The first term is a scaling that `alpha` absorbs, so the
assumed ambient moves the reported *capacity* by ~0.43 %/°C and leaves the
orientation alone.

On a plant that really does heat up (ambient 9–27 °C, mean 18 °C, 2 % noise):

| Dictionary assumes | tilt | azimuth | capacity |
|---|---|---|---|
| no temperature model | 31.1° | **−28.6°** | **−6.4 %** |
| constant 25 °C | 33.5° | **−20.9°** | +3.7 % |
| constant 20 °C | 33.3° | −21.1° | **+1.2 %** |
| the measured series | 31.0° | −24.6° | +1.6 % |

(centroids; planted 30° / −20° / 100 kWp)

So: **if you have no temperature data, pass a constant rather than `None`** —
it recovers the azimuth to within 1° where omitting it is off by 8.6°. Any
constant does that; the value only sets the capacity offset, so use the site's
mean daytime ambient if you know it even roughly. Defaults `GAMMA_DEFAULT`,
`TEMP_RISE_COEFF_DEFAULT`, `TEMP_REF_C` reproduce the paper's polycrystalline /
mixed-mounting values, and `alpha` remains the **STC nominal capacity** in all
three modes.

The measured series does best on capacity but not on azimuth — the temperature
factor is one more way for tilt and capacity to trade against each other, so
the extra freedom costs conditioning. With matched models and no noise all
three modes recover the plant exactly.

### Interval-averaged measurements

By default the model is evaluated **at** each timestamp, which is right for
instantaneous readings. Metered data is almost always an *average over an
interval* — and comparing an interval average against a point-in-time model is
a **bias, not noise**: it does not average out with more data.

Declare the interval and what the timestamps mean, and the model is integrated
over the same window:

```python
P_pu, ghi = build_pu_power_matrix(
    lat, lon, elev, timestamps,
    interval="1h",        # width of the averaging interval
    label="end",          # "centre" (default) | "start" | "end"
    subsample="5min",     # integration step; default interval/12
)
```

| `label`    | interval covered by a timestamp `t` |
|------------|-------------------------------------|
| `"centre"` | `[t - interval/2, t + interval/2)`  |
| `"start"`  | `[t, t + interval)`                 |
| `"end"`    | `(t - interval, t]`                 |

**Why it matters.** Recovering a known 100 kWp array at tilt 30°, azimuth −20°
from hourly averages of a clear-sky year:

| Timestamps | Model | recovered | azimuth error | capacity |
|------------|-------|-----------|---------------|----------|
| hourly, end-labelled | at the timestamp | (20, +40) | **+30°** | +5.4 % |
| hourly, start-labelled | at the timestamp | (30, −50) | **−27°** | +3.5 % |
| hourly, any labelling | interval-integrated | (30, −20) | +0.5° | +0.05 % |
| 15 min, end-labelled | at the timestamp | (30, −15) | +9.5° | +1.3 % |
| 15 min, end-labelled | interval-integrated | (30, −20) | +0.6° | +0.08 % |

The mislabelled fits still report **R² > 0.999** — they are confidently wrong,
which is why the convention has to be declared rather than guessed. Resolution
itself is not the problem: correctly declared hourly data recovers the
orientation as well as 5-minute data does.

The cost is `interval / subsample` model evaluations (12 by default), and
`subsample` must divide `interval` exactly. `air_temp` is held constant across
an interval — it is not interpolated.

### Terrain shading

Pass a horizon profile and each column of the dictionary is assembled from its
components instead of being taken whole:

```
POA = (1 − blocked) · POA_beam  +  SVF · POA_sky_diffuse  +  POA_ground
```

```python
from pv_orientation_estimator import build_pu_power_matrix, shaded_fraction
from pv_orientation_estimator.horizon import download_horizon

horizon = download_horizon(lat, lon)
P_pu, ghi = build_pu_power_matrix(lat, lon, elev, timestamps,
                                  interval="1h", label="end", horizon=horizon)
```

- **`blocked`** is 1 while the terrain hides the sun (`beam_blocked()`), tested
  on the *apparent* elevation — refraction is what decides visibility at the
  skyline.
- **`SVF`** is the sky view factor (`sky_view_factor()`): the fraction of the
  sky dome the plane still sees. Pure geometry, so it costs one number per
  orientation for a whole time series. For a horizontal plane under a uniform
  skyline at elevation *h* it reduces to `cos²h`, which the tests check against.
- **Ground-reflected irradiance is left alone.** The terrain in front of the
  array is shaded too, but that is a small correction on a small term.
- The sky model is isotropic, so there is no separate circumsolar part to switch
  off with the beam. Under a Perez-type sky there would be — it comes from
  within ~25° of the solar disc.

**Use `horizon` together with `interval`.** Blocking is a step in time: an
instantaneous evaluation can only answer 0 or 1, and sampling it more finely
does not help — the error at each transition stays O(1). Averaged over the
interval it becomes the *fraction* of the interval the beam is lost for. The
sub-sampling refines itself to ~1-minute steps when a horizon is given, because
the sun crosses its own diameter in about two minutes: on hourly data, 5-minute
steps misplace the shaded fraction by up to 7 % of the interval, 1-minute steps
by 1 %.

To drop shaded samples rather than model them:

```python
usable = (ghi > DAYTIME_GHI_THRESHOLD) & (
    shaded_fraction(lat, lon, elev, timestamps, horizon,
                    interval="1h", label="end") < 0.5)
```

Modelling keeps the samples but leans on a constant-efficiency plant model at
the low irradiance terrain shade leaves behind; excluding them costs sample
count and assumes nothing.

**What it is worth.** A synthetic 100 kWp plant at Wimmis BE (peak skyline 26°,
14 % of daytime samples shaded), true tilt 30° / azimuth −20°:

| Dictionary | tilt | azimuth | capacity | R² |
|---|---|---|---|---|
| terrain ignored | 20° (−10) | −25° (−5) | +5.9 % | 0.974 |
| terrain modelled | **30°** (0) | −25° (−5) | **+1.0 %** | **0.998** |

```bash
python scripts/demo_estimate.py --lat 46.6777 --lon 7.6409 --elev 639 --horizon \
    --start 2023-02-01 --end 2023-03-25
```

### Inverter clipping (kWp > kVA)

A plant built with more DC capacity than its inverter passes has its output cut
off at the AC rating. Those samples are **right-censored**: they say the array
would have produced *at least* the rating, not how much more. Fitting them as
equalities is not neutral — a flat top is best explained by a shallower, more
easterly array with less capacity (15° instead of 30°, −6.6 % capacity, on the
plant below).

Two estimators handle it, selected with `method`:

#### Method A — the smallest plant covering the measurements

```python
result = run_estimation(P_pu, P_measured, daytime, method="A")
result["ac_rating_kw"]    # 75.0 — an output, not an input
```

```
min 1ᵀα   s.t.   P_pu α ≥ P_measured,   α ≥ 0
```

A linear program. It needs no rating, no censoring band and no partition of the
samples, and the rating falls out as `max(P)`. It is the limit λ→0 of adding
`λ·1ᵀα + μ·S` to the one-sided relaxation — and the weights are not a delicate
choice, the solution is unchanged for λ between 0.001 and 1.

Because it is a *covering* criterion it fits the **upper envelope** of the
data, not its centre. On clean measurements that is exactly right; with noise,
every upward excursion has to be cleared, so both capacity and rating inflate.

#### Method A1 — Method A with the rating given

```python
result = run_estimation(P_pu, P_measured, daytime, method="A1", ac_rating=75.0)
```

```
min 1ᵀα   s.t.   P_pu α ≥ min(P_measured, S),   α ≥ 0
```

Method A estimates the rating; Method B is handed it — so comparing them
directly is not like-for-like. A1 is the fair comparison: a reading above the
rating cannot be covered by any array (the inverter caps the model too), so the
covering target is the measurement clipped at the rating.

Fixing the rating stops the noisy *plateau* from inflating the envelope, but the
morning and evening *shoulders* still have to be covered, and it is those that
set the shape. So A1 recovers the capacity better than A and the tilt no better.

#### Method B — censored least squares

```python
result = run_estimation(P_pu, P_measured, daytime, method="B", ac_rating=75.0)
```

```
min  Σ_free (P_t − (P_pu α)_t)²  +  Σ_clipped max(0, S − (P_pu α)_t)²    s.t. α ≥ 0
```

Keeps least squares and writes the censoring into the residual: the model is
penalised only for predicting *less* than the rating where the meter was
saturated. Needs the rating (a number in kW, or `"auto"`), but averages over
noise instead of chasing its extremes.

#### Choosing

Recovery of 100 kWp at tilt 30° / azimuth −20° behind a 75 kVA inverter (tilt is
the capacity-weighted centroid):

A1 and B are given the true 75 kW rating; A estimates it (last column):

| noise | A: tilt / kWp | A1: tilt / kWp | B: tilt / kWp | A's S |
|---|---|---|---|---|
| 0 % | **30.0° / 100.0** | **30.0° / 100.0** | 33.6° / 103.1 | 75.0 |
| 1 % | 26.3° / 101.6 | 25.0° / 101.0 | **33.0° / 102.5** | 77.1 |
| 2 % | 22.7° / 103.4 | 21.9° / 102.3 | **32.5° / 102.1** | 79.3 |
| 4 % | 15.7° / 108.4 | 15.0° / 106.1 | **23.6° / 97.2** | 83.5 |
| 8 % | 15.0° / 119.2 | 15.0° / 114.7 | **19.0° / 94.8** | 92.1 |

The crossover is around **1 % of unmodelled error** — and that means total model
error (clear-sky mismatch, soiling, temperature), not meter accuracy. Since that
error is largely one-sided, it is the direction a covering constraint absorbs
worst, which is why **B is the default** and A is for well-conditioned data.

```bash
python scripts/demo_estimate.py --kva 75              # both methods, side by side
python scripts/demo_estimate.py --kva 75 --method A --noise 0
```

Whichever you use: capacity is only weakly bounded from above, since production
above the cap is never observed. Report the pair (capacity, rating) rather than
capacity alone. `clip_band` (Method B, default 6 %) must cover the measurement
noise — measured rule of thumb, band ≈ 3 × noise.

`"auto"` rating detection is a fallback, not a default: it finds the plateau for
DC/AC ≥ 1.1 up to ~2 % noise (level within 4 %), misses milder or noisier
clipping, and errs towards "no clipping" — a missed detection leaves the plain
fit, an invented one would censor a real peak.

### Inverter cut-in (the system does not start below X kW)

An inverter needs a minimum input to start. A zero reading is then
**left-censored** — the mirror of a clipped one: it says the array produced
*less* than the cut-in, not that it produced nothing. Those samples enter the
objective one-sidedly too, penalising the model only for predicting *more*
than the cut-in:

```
Σ_below max(0, (P_pu α)_t − cutoff)²
```

```python
result = run_estimation(P_pu, P_measured, daytime, cutoff_kw=0.5)
result["cutoff_share"]     # fraction of daytime samples treated this way
```

**It only matters when the cut-in is a noticeable share of the capacity.** With
a 0.5 kW cut-in and the standard 50 W/m² daytime mask, planted 30° tilt:

| capacity | cut-in ÷ capacity | samples touched | tilt if ignored | with `cutoff_kw` |
|---|---|---|---|---|
| 100 kWp | 0.005 | 0.0 % | 30.0° | 30.0° |
| 5 kWp | 0.10 | 17.7 % | 32.8° | **30.0°** |
| 3 kWp | 0.17 | 23.1 % | 39.3° | **30.0°** |
| 2 kWp | 0.25 | 25.8 % | **52.4°** (+9.7 % capacity) | **30.0°** (+0.02 %) |

So it is safe to ignore on commercial plants and necessary on small residential
ones. The bias has a consistent direction: truncating the shoulders makes the
daily profile look narrower, which reads as a *steeper* array.

`cutoff_band` (default 2.0) widens the censored set to `2 × cutoff_kw`, because
an interval straddling the cut-in averages above it while still being partly
truncated — censoring the bare cut-in alone leaves about half the bias.

**Outages look identical and must be removed first.** A zero in the middle of a
sunny day is a fault, curtailment or snow, not a cut-in, and fitting it as one
tells the model the array is small. Watch `cutoff_share`: on a plant where the
cut-in is a small fraction of capacity it should be a few percent at most.

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
| `method`        | clipping model used: `None`, `"A"` or `"B"`         |
| `ac_rating_kw`  | AC rating the fit clipped at, or `None`              |
| `clipped_share` | share of daytime samples treated as censored          |
| `dc_ac_ratio`   | `effective_kWp / ac_rating_kw`, or `None`           |
| `cutoff_kw`     | inverter cut-in power, or `None`                     |
| `cutoff_share`  | share of daytime samples below the cut-in            |

On solver failure only `status` and `alpha` (`None`) are populated.

## Demo

A synthetic end-to-end run — generate a plant of known geometry, estimate it
back, and see the fit:

```bash
python scripts/demo_estimate.py                          # 100 kWp, tilt 30, az -20
python scripts/demo_estimate.py --tilt 15 --azimuth 30 --interval 15min --noise 5
```

```python
from pv_orientation_estimator import demo_estimate

result = demo_estimate()               # prints the comparison, plots the fit
```

It prints the recovered orientation against the truth and plots two panels: the
fit over the first days, and the capacity the solver placed on each candidate
orientation.

The synthetic plant is **not** generated from the reference matrix the
estimator fits — its power is integrated over each metering interval from
1-minute samples, the way a meter averages, and carries measurement noise. So
the demo reproduces the mismatch real data has, unlike the self-fulfilling
round trip in `tests/test_roundtrip.py`. Add `--horizon` and the terrain is
modelled on both sides, and the demo additionally refits with a horizon-blind
dictionary to price the omission.

## Conventions

- **Azimuth (EU):** degrees relative to **south**, negative toward east,
  positive toward west. pvlib's north-clockwise azimuth is `180 + az_eu`.
- **Per-unit:** POA irradiance is divided by 1000 W/m², so `alpha` comes out in
  kWp when `P_measured` is in kW.

## Terrain horizon (PVGIS)

`horizon.py` downloads the **topographic horizon** of a site from the JRC
[PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) service (`printhorizon`
endpoint): the elevation angle of the skyline — hills, mountains — in every
direction around the site, computed from PVGIS' digital elevation model. It is
fed to `build_pu_power_matrix` (see *Terrain shading* above) it makes the
dictionary account for the skyline; on its own it tells you whether terrain
shading is worth worrying about at a site at all.

```bash
# one location, downloaded and plotted (Wimmis BE)
python scripts/download_horizon.py --demo

# any site, saved to CSV and PNG
python scripts/download_horizon.py --lat 46.0207 --lon 7.7491 \
    --csv horizon.csv --png horizon.png
```

```python
from pv_orientation_estimator import download_horizon, plot_horizon, demo_horizon

profile = download_horizon(lat=46.6777, lon=7.6409)
profile.to_dataframe()            # columns: azimuth_eu, elevation_deg
profile.elevation_at(-20.0)       # skyline elevation at an arbitrary azimuth [deg]
plot_horizon(profile)             # matplotlib Axes; call plt.show() yourself

demo_horizon()                    # download + plot in one call
```

PVGIS returns 49 samples over −180°…180° in 7.5° steps, plus the sun paths at
the winter and summer solstice, which `plot_horizon` overlays — where a sun
path dips below the terrain line, the site is shaded. For Swiss sites the plot
is titled with the commune the point falls in (see *Geocoding* below); pass
`--no-geocode`, or `name_site=False`, to skip that lookup.

Azimuths are kept in the **same EU convention as the orientation grid**
(0° = south, negative = east, positive = west). Note that pvlib's own
`pvlib.iotools.get_pvgis_horizon` instead returns north-clockwise azimuths.

Plotting requires matplotlib, an optional extra:

```bash
pip install -e ".[plot]"
```

## Geocoding

`geocode.py` resolves Swiss sites in both directions through the free, key-less
[swisstopo GeoAdmin API](https://api3.geo.admin.ch/). It is what names the
commune on a horizon plot, and is useful on its own to attach a place to a set
of coordinates, or coordinates to a plant address.

```bash
python scripts/geocode.py "Bahnhofstrasse 1, 3920 Zermatt"   # address -> coordinates
python scripts/geocode.py --lat 46.0207 --lon 7.7491         # coordinates -> commune
python scripts/geocode.py "Route Cantonale, Ecublens" --candidates
```

```python
from pv_orientation_estimator import reverse_geocode, geocode, geocode_one

place = reverse_geocode(46.5191, 6.5668)
place.label          # '1015 Lausanne, Ecublens (VD)'
place.short_label    # 'Ecublens (VD)'    — commune, for titles
place.commune, place.canton, place.postcode, place.bfs_number

geocode_one("Bahnhofstrasse 1, 3920 Zermatt")   # -> Place(lat=46.02328, lon=7.74807, …)
geocode("Route Cantonale, Ecublens", limit=5)   # ranked candidates
```

| Direction | Endpoint | Data |
|-----------|----------|------|
| coordinates → place | `identify` | `ch.swisstopo-vd.ortschaftenverzeichnis_plz` (locality + PLZ), `ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill` (commune, canton, BFS number) |
| address → coordinates | `SearchServer` | official address register, commune list, gazetteer |

Things worth knowing:

- **Switzerland only.** Outside it, `reverse_geocode` returns `None` and
  `geocode` returns `[]`. Large water bodies are in neither layer, so a point
  in the middle of Lake Geneva also comes back `None`; land is fully covered,
  including alps and glaciers (46.30, 7.60 → *Susten, Leuk (VS)*).
- **The search never says "no match".** GeoAdmin always returns its
  best-ranked guess — `"xyzzy plugh"` comes back as *Bülach (ZH)*. `geocode`
  therefore drops results sharing no word stem with the query (accent-folded,
  so *Geneva* still finds *Genève*); pass `verify=False` for the raw ranking.
- **Fuzzy within a match:** a street number that does not exist returns the
  nearest ones in that street, so check `place.address` before trusting an
  exact hit.
- Commune names already carry a canton where they are ambiguous
  (`"Ecublens (VD)"`), and the boundary layer returns one row per historical
  year — the current one is selected via `is_current_jahr`.

## Benchmark

`scripts/benchmark_synthetic.py` runs the synthetic validation behind
[§7 of the paper](../doc/algorithm.tex): noise sweep, timestamp conventions,
window length, a two-orientation plant, terrain shading and inverter clipping,
all under one protocol.

```bash
python scripts/benchmark_synthetic.py                  # ~25 s
python scripts/benchmark_synthetic.py --skip-horizon   # no network
```

Every number quoted in that section comes from a run of it. Note the caveat it
carries: the measurements are generated with the same clear-sky and
transposition model that builds the dictionary, so the benchmark validates the
*estimator*, not the irradiance model.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite performs a synthetic round-trip (generate power from a known
orientation, recover it) and needs no database or network access.
