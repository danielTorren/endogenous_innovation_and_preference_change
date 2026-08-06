"""
=============================================================================
 FAKE DATA GENERATOR -- PLACEHOLDER ONLY, NOT REAL CALIFORNIA DATA
=============================================================================

Generates three CSVs with EXACTLY the schema that the real zip-level data
must have, so that the whole pipeline (synthetic_population.py ->
calibration -> validation) can be built and tested before the real data
arrives. When the real data arrives, write it out with the same column
names into package/calibration_data/zip_data/ and delete nothing else:
every downstream module reads the schema, never these numbers.

Run:
    python -m package.generating_data.fake_zip_data_gen

Outputs (into package/calibration_data/FAKE_zip_data/):
    FAKE_synthetic_population.csv   one row per household
    FAKE_zip_ev_stock.csv           one row per (zip, year)
    FAKE_zip_ev_sales.csv           one row per (zip, year)
    FAKE_DATA_README.md             the column contract

DELIBERATE PROPERTIES OF THE FAKE DATA (so the pipeline is actually
exercised rather than trivially satisfied):
  - income, VMT, ruralness, political leaning are CORRELATED, both within
    zip (income-VMT) and across zips (rural zips are poorer and more
    conservative). Independent fake covariates would make the calibration
    look better identified than it is.
  - the zip EV share has a real gradient in all three zip-level
    covariates, and that gradient GROWS over time. That is what makes the
    dispersion moments in summary_stats.py informative.
  - EV counts are drawn as integers from a Binomial, so the observation
    noise that validation/bootstrap_targets.py estimates is genuinely
    there rather than assumed.
  - the state-level aggregate is pinned to the REAL California series
    already in package/calibration_data/calibration_data_output.pkl, so
    swapping fake -> real should not move the aggregate target much.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Where the data goes. The real data should land in a sibling directory
# ("zip_data") so the fake stuff can be deleted wholesale without touching it.
# ---------------------------------------------------------------------------
OUT_DIR = "package/calibration_data/FAKE_zip_data"

# ---------------------------------------------------------------------------
# Shape of the fake dataset. Real California has ~1,760 ZCTAs; nothing
# downstream depends on this number, it is only here to keep the fake files
# small and fast to read.
# ---------------------------------------------------------------------------
NUM_ZIPS = 250
HOUSEHOLD_ROWS_PER_ZIP = (120, 380)      # rows sampled per zip, uniform in this range

# ---------------------------------------------------------------------------
# MARGINAL MATCHING. These two spreads are not arbitrary. The model's existing
# parameters were all calibrated against the parametric draws in
# controller.gen_beta / gen_distance, whose POOLED marginals are:
#   income  log-sd 0.927   (income_sigma in base_params)
#   d_vec   log-sd 0.819   (Poisson(1.5428) mapped onto the six survey bins)
# A fake population that is more HOMOGENEOUS than that puts the model in a
# degenerate regime -- with no low-mileage agents, nobody stays on ICE and EV
# share runs to ~85% instead of the observed ~4%, which would make every
# validation result below meaningless. So the within-zip spreads are chosen to
# reproduce those two pooled log-sds once the across-zip variation is added in
# quadrature:
#   income:  sqrt(0.45^2 + 0.81^2) = 0.93
#   vmt:     sqrt(0.75^2 + var(0.45*ruralness)) = 0.76
# The real population will have whatever spread it has; the point of matching
# here is only that the placeholder exercises the same regime.
# package/validation/compare_populations.py prints this comparison, and is the
# first thing to run when the real file arrives.
# ---------------------------------------------------------------------------
WITHIN_ZIP_INCOME_LOG_SD = 0.81
WITHIN_ZIP_VMT_LOG_SD = 0.75
WITHIN_ZIP_INCOME_LOG_SD_ACROSS = 0.45      # across-zip component of income spread

VEHICLES_PER_HOUSEHOLD = 1.8
MEAN_HOUSEHOLDS_PER_ZIP = 9_400          # after applying `weight`

YEARS = np.arange(2016, 2024)            # 2016..2023, matches the current calibration window

# State-level EV STOCK share, 2016..2023. Taken from the real series in
# calibration_data_output.pkl ("EV Prop"[6:]) so the fake zip data aggregates
# up to something realistic.
STATE_STOCK_SHARE = np.array([
    4.56782135e-03, 5.61984590e-03, 8.09717263e-03, 1.06251899e-02,
    1.28851200e-02, 1.74482660e-02, 2.60592757e-02, 3.80204262e-02,
])

# State-level EV SALES share, 2016..2023. The real pickle only carries four
# values ("EV Sales Prop" = 2020..2023); the earlier four here are a plausible
# back-extension. FAKE. The summary-stat spec (`sales_years`) is what decides
# which of these actually get used as a calibration target, so a shorter real
# series needs no code change, only a shorter `sales_years`.
STATE_SALES_SHARE = np.array([0.021, 0.031, 0.049, 0.052,
                              0.05680672, 0.09369294, 0.16569547, 0.21243213])

# Annual new-vehicle sales as a fraction of the stock (fleet turnover).
SALES_TURNOVER = 0.06

# True gradients used to GENERATE the fake zip shares (log-odds per 1 SD of
# the z-scored zip covariate) in the first and last year. Growing over time =
# the signature of diffusion rather than static sorting.
GRAD_INCOME    = (0.55, 0.95)
GRAD_RURALNESS = (-0.45, -0.80)
GRAD_POLITICAL = (0.30, 0.60)
ZIP_RESIDUAL_SD = 0.35                   # unexplained zip log-odds spread


def _gauss_copula(rs, n, rho):
    """
    Two standard normals with correlation rho. Used everywhere a marginal
    distribution must be preserved exactly while its DEPENDENCE on something
    else is controlled -- the same device synthetic_population.py uses for
    rho_pol and rho_age, so the fake data is generated by the same logic the
    model uses to read it.
    """
    z1 = rs.normal(size=n)
    z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rs.normal(size=n)
    return z1, z2


def _zscore(x, w=None):
    """Weighted z-score. w=None means unweighted."""
    if w is None:
        m, s = np.mean(x), np.std(x)
    else:
        m = np.average(x, weights=w)
        s = np.sqrt(np.average((x - m) ** 2, weights=w))
    return (x - m) / (s if s > 0 else 1.0)


def gen_zip_covariates(rs):
    """
    Zip-level covariates, GENERATED FROM GEOGRAPHY.

    Coordinates come first and ruralness is derived from distance to the nearest
    urban centre; income and political leaning then follow ruralness. That
    ordering is deliberate and it is what makes this dataset able to test the
    homophily parameters at all:

      - income similarity and spatial proximity become CORRELATED but NOT
        identical. If they were independent, `homophily_spatial_weight` would be
        trivially identified; if they were identical it would be unidentifiable.
        The realistic and hard case is in between, and that is what the
        distance-to-centre construction produces.
      - rural zips are poorer and more conservative, which is the confounding
        that forces summary_stats.py to use a multivariate weighted OLS gradient
        rather than three separate high-minus-low contrasts.

    Coordinates are a plausible California bounding box. They are FAKE: no zip
    code here corresponds to its real location.
    """
    from scipy.stats import norm

    # California-ish bounding box
    lat = rs.uniform(32.6, 41.9, size=NUM_ZIPS)
    lon = rs.uniform(-124.2, -114.4, size=NUM_ZIPS)

    # three urban centres, roughly LA / Bay Area / San Diego-Sacramento scale.
    # Population is concentrated near them, so draw most zips close to a centre
    # and leave a minority scattered.
    centres = np.array([[34.05, -118.25], [37.77, -122.42], [32.72, -117.16]])
    n_urban = int(0.75 * NUM_ZIPS)
    which = rs.randint(0, len(centres), size=n_urban)
    spread = 0.55
    lat[:n_urban] = centres[which, 0] + rs.normal(0, spread, size=n_urban)
    lon[:n_urban] = centres[which, 1] + rs.normal(0, spread, size=n_urban)
    lat = np.clip(lat, 32.6, 41.9)
    lon = np.clip(lon, -124.2, -114.4)

    # ruralness from distance to the nearest centre, mapped to [0,1] by rank so
    # the marginal stays the Beta(1.5,3) shape used before (mostly urban)
    d = np.min(np.sqrt((lat[:, None] - centres[None, :, 0]) ** 2
                       + (lon[:, None] - centres[None, :, 1]) ** 2), axis=1)
    d_rank = np.argsort(np.argsort(d)) / (NUM_ZIPS - 1)
    ruralness = _beta_ppf(d_rank, 1.5, 3.0)

    # political leaning in [0,1] (0 = most conservative, 1 = most liberal),
    # negatively correlated with ruralness via a rank copula
    z_r = np.clip(np.argsort(np.argsort(ruralness)) / (NUM_ZIPS - 1), 1e-6, 1 - 1e-6)
    z_pol = -0.55 * norm.ppf(z_r) + np.sqrt(1 - 0.55 ** 2) * rs.normal(size=NUM_ZIPS)
    political = norm.cdf(z_pol)

    # zip median household income, negatively correlated with ruralness
    z_inc = -0.35 * norm.ppf(z_r) + np.sqrt(1 - 0.35 ** 2) * rs.normal(size=NUM_ZIPS)
    zip_median_income = np.exp(11.225 + WITHIN_ZIP_INCOME_LOG_SD_ACROSS * z_inc)

    # ZIP codes assigned in spatial order, which is roughly how real US ZIPs
    # work. synthetic_population falls back to this ordering as a crude spatial
    # proxy when latitude/longitude are absent, so the fake file should have the
    # property the fallback assumes.
    order = np.lexsort((lat, np.round(lon, 0)))
    zips = np.empty(NUM_ZIPS, dtype=np.int64)
    zips[order] = 90000 + np.arange(NUM_ZIPS) * 3

    return pd.DataFrame({
        "zip": zips,
        "latitude": lat,
        "longitude": lon,
        "ruralness": ruralness,
        "political_leaning": political,
        "_zip_median_income": zip_median_income,
    })


def _beta_ppf(u, a, b):
    from scipy.stats import beta as beta_dist
    return beta_dist.ppf(np.clip(u, 1e-9, 1 - 1e-9), a, b)


def gen_households(rs, zip_df):
    """
    Household rows. The key structural feature is that income and VMT are
    JOINTLY drawn (rho ~ 0.35) and that VMT also rises with ruralness. In the
    current model these two are independent draws (controller.gen_distance vs
    controller.gen_beta), which is exactly the thing this dataset fixes.
    """
    frames = []
    for _, z in zip_df.iterrows():
        n = rs.randint(HOUSEHOLD_ROWS_PER_ZIP[0], HOUSEHOLD_ROWS_PER_ZIP[1] + 1)

        z_inc, z_vmt = _gauss_copula(rs, n, rho=0.35)

        # WITHIN_ZIP_INCOME_LOG_SD and WITHIN_ZIP_VMT_LOG_SD are set so the
        # POOLED marginals match the parametric distributions the model
        # currently uses -- see the note on marginal matching at the top of this
        # file, and package/validation/compare_populations.py which checks it.
        income = z["_zip_median_income"] * np.exp(WITHIN_ZIP_INCOME_LOG_SD * z_inc)

        # annual PER-VEHICLE VMT: rises with income and with ruralness.
        # Per-vehicle, not per-household, because each model agent owns exactly
        # one car. Centred at 13,000 mi/yr so that after the /12 conversion in
        # SyntheticPopulation.gen_d_vec it lands near the parametric default's
        # monthly median of 1,274.
        log_vmt = (np.log(13_000.0)
                   + WITHIN_ZIP_VMT_LOG_SD * z_vmt
                   + 0.45 * z["ruralness"])
        vmt = np.exp(log_vmt)

        age = np.clip(rs.normal(51.0, 15.0, size=n), 20.0, 90.0)

        # expansion weight: households represented by this row
        weight = np.full(n, MEAN_HOUSEHOLDS_PER_ZIP / n)

        frames.append(pd.DataFrame({
            "zip": np.full(n, z["zip"], dtype=np.int64),
            "weight": weight,
            "income": income,
            "vmt": vmt,
            "age": age,
            "political_leaning": np.full(n, z["political_leaning"]),
            "ruralness": np.full(n, z["ruralness"]),
            "latitude": np.full(n, z["latitude"]),
            "longitude": np.full(n, z["longitude"]),
        }))

    return pd.concat(frames, ignore_index=True)


def gen_zip_ev_series(rs, zip_df, pop_df):
    """
    Zip x year EV stock and sales counts.

    Build a log-odds surface with a growing covariate gradient, solve for the
    per-year intercept that reproduces the state aggregate exactly, then draw
    integer counts from a Binomial so the files carry realistic small-count
    observation noise.
    """
    # population-weighted zip aggregates, then z-score across zips weighted by
    # vehicle stock -- same convention summary_stats.py uses, so the "true"
    # gradients below are directly comparable to the recovered ones.
    agg = pop_df.groupby("zip").apply(
        lambda g: pd.Series({
            "households": g["weight"].sum(),
            "median_income": np.median(np.repeat(g["income"].values,
                                                 np.maximum(1, (g["weight"] * 10).astype(int)))),
        }),
        include_groups=False,
    ).reset_index()
    zip_df = zip_df.merge(agg, on="zip")

    zip_df["total_stock"] = np.round(
        zip_df["households"] * VEHICLES_PER_HOUSEHOLD
    ).astype(np.int64)

    w = zip_df["total_stock"].values.astype(float)
    z_income = _zscore(np.log(zip_df["median_income"].values), w)
    z_rural = _zscore(zip_df["ruralness"].values, w)
    z_pol = _zscore(zip_df["political_leaning"].values, w)

    resid = rs.normal(0.0, ZIP_RESIDUAL_SD, size=len(zip_df))

    stock_rows, sales_rows = [], []
    n_yr = len(YEARS)
    for k, year in enumerate(YEARS):
        frac = k / (n_yr - 1)
        b_inc = GRAD_INCOME[0] + frac * (GRAD_INCOME[1] - GRAD_INCOME[0])
        b_rur = GRAD_RURALNESS[0] + frac * (GRAD_RURALNESS[1] - GRAD_RURALNESS[0])
        b_pol = GRAD_POLITICAL[0] + frac * (GRAD_POLITICAL[1] - GRAD_POLITICAL[0])

        lin = b_inc * z_income + b_rur * z_rural + b_pol * z_pol + resid

        # --- STOCK ---
        a = _solve_intercept(lin, w, STATE_STOCK_SHARE[k])
        p_stock = _sigmoid(a + lin)
        ev_stock = rs.binomial(zip_df["total_stock"].values, p_stock)

        # --- SALES --- same surface, sales gradient is steeper than stock
        # because stock is a slow-moving accumulation of past sales
        total_sales = np.maximum(
            1, np.round(zip_df["total_stock"].values * SALES_TURNOVER).astype(np.int64)
        )
        a_s = _solve_intercept(1.35 * lin, total_sales.astype(float), STATE_SALES_SHARE[k])
        p_sales = _sigmoid(a_s + 1.35 * lin)
        ev_sales = rs.binomial(total_sales, p_sales)

        stock_rows.append(pd.DataFrame({
            "zip": zip_df["zip"].values,
            "year": year,
            "ev_stock": ev_stock,
            "total_stock": zip_df["total_stock"].values,
        }))
        sales_rows.append(pd.DataFrame({
            "zip": zip_df["zip"].values,
            "year": year,
            "ev_sales": ev_sales,
            "total_sales": total_sales,
        }))

    return (pd.concat(stock_rows, ignore_index=True),
            pd.concat(sales_rows, ignore_index=True))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(lin, w, target_share, iters=80):
    """Bisect for the intercept `a` making the w-weighted mean of sigmoid(a+lin) equal target_share."""
    lo, hi = -30.0, 30.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = np.average(_sigmoid(mid + lin), weights=w)
        if got < target_share:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


README = """\
# ZIP-LEVEL DATA: COLUMN CONTRACT

Everything in `FAKE_zip_data/` is **synthetic placeholder data generated by
`package/generating_data/fake_zip_data_gen.py`**. It is NOT real California
data. It exists so the calibration and validation pipeline can be written and
tested before the real data arrives.

## To plug in the real data

1. Write the real files into `package/calibration_data/zip_data/` with the
   filenames and columns below.
2. In your base-params JSON set
   `parameters_synthetic_population.population_path` and
   `parameters_zip_data.{stock_path,sales_path}` to the real paths.
3. Run `python -m package.calibration.calibration_data_outputs_zip` to rebuild
   the target vector.
4. Nothing else changes. No module reads any of the numbers below; they only
   read these column names.

## `synthetic_population.csv` -- one row per household

| column              | type  | meaning                                                        |
|---------------------|-------|----------------------------------------------------------------|
| `zip`               | int   | 5-digit ZIP / ZCTA. Must match the zip column in the two files below. |
| `weight`            | float | expansion weight: households in the real population represented by this row. Use 1.0 if the population is already a full enumeration. |
| `income`            | float | annual household income, USD, nominal in the base year          |
| `vmt`               | float | annual **per-vehicle** vehicle miles travelled                   |
| `age`               | float | age of household head, years                                    |
| `political_leaning` | float | 0 = most conservative, 1 = most liberal. Zip-level, repeated on every row of that zip. Any monotone index works: only the RANKS are used. |
| `ruralness`         | float | 0 = most urban, 1 = most rural. Zip-level, repeated on every row. Must be on [0,1] because it enters `nu_i = nu * (1 + a_rural * ruralness_i)` multiplicatively. |
| `latitude`          | float | zip centroid latitude, decimal degrees. Optional but strongly recommended: see below. |
| `longitude`         | float | zip centroid longitude, decimal degrees. Optional but strongly recommended. |

### `latitude` / `longitude` drive spatial homophily

These are the only source of geography in the model. They are used for two
things:

1. **Network placement.** `homophily_spatial_weight` decides how much an agent's
   position in the small-world ring is determined by physical proximity rather
   than income similarity. Proximity is measured by a Hilbert space-filling
   curve index over the zip centroids, which is what maps 2-D position to the
   1-D ring while preserving locality.
2. **Moran's I**, the spatial-autocorrelation summary statistic that separates
   local social contagion from preference sorting. Its k-nearest-neighbour
   spatial weight matrix is built from these coordinates.

If they are absent, `synthetic_population` falls back to ordering zips by ZIP
CODE NUMBER, on the basis that US ZIP codes are assigned roughly
geographically. It warns loudly when it does this. The fallback is crude:
numerically adjacent ZIPs are usually but not always physically adjacent, and
the failure is worst exactly where it matters most, at metro boundaries. Zip
centroids are a free join from Census ZCTA gazetteer files, so supply them.

Extra columns are ignored, so a richer real file can be dropped in as is.

### Two unit traps in `vmt`, both silent if you get them wrong

1. **Period.** The model steps monthly and `d_vec` is a per-timestep distance
   (the Poisson it replaces has median 1,274 monthly miles). An annual column
   is divided by 12; set
   `parameters_synthetic_population.vmt_period = "monthly"` if the real file is
   already monthly. A 12x too large `d_vec` makes lifetime fuel cost swamp the
   utility and every agent adopts an EV within a year, with no error raised.
2. **Per vehicle, not per household.** Each model agent owns exactly one car,
   so `vmt` must be miles per vehicle. If the real population reports household
   VMT, divide by that household's vehicle count before writing this column.

`SyntheticPopulation.gen_d_vec` prints a warning when the resulting monthly
median is more than 3x away from the parametric default, which catches both of
these plus a kilometres-vs-miles mix-up. Do not ignore that warning: every
other parameter in the model was calibrated against the parametric distance
scale.

## `zip_ev_stock.csv` -- one row per (zip, year)

| column        | type | meaning                                          |
|---------------|------|--------------------------------------------------|
| `zip`         | int  |                                                  |
| `year`        | int  | calendar year                                    |
| `ev_stock`    | int  | EVs registered in that zip at that year's snapshot |
| `total_stock` | int  | ALL vehicles registered in that zip at that snapshot |

Counts, not shares. The counts are what let
`package/validation/bootstrap_targets.py` estimate observation noise; if you
only have shares, supply `total_stock` anyway so the noise can be sized.

## `zip_ev_sales.csv` -- one row per (zip, year)

| column        | type | meaning                                       |
|---------------|------|-----------------------------------------------|
| `zip`         | int  |                                               |
| `year`        | int  | calendar year                                 |
| `ev_sales`    | int  | new EV sales/registrations in that zip that year |
| `total_sales` | int  | ALL new vehicle sales/registrations that year  |

If the real "sales" series is actually all registrations including used
vehicles, set `parameters_zip_data.sales_include_used = true` so the model
side counts second-hand purchases too (see
`socialNetworkUsers.history_zip_used_sales_EV`).

The year coverage of the two files does NOT have to match, and neither has to
cover all of 2016-2023. `package/calibration/summary_stats.py` takes the years
it uses from the spec (`stock_years`, `sales_years`), so a shorter real series
only needs a shorter list there.

## What the fake numbers were built to contain

Written down so you can tell whether a pipeline result is real or an artefact
of the placeholder:

- income, VMT, ruralness and political leaning are all correlated. Rural zips
  are poorer and more conservative; within a zip income and VMT correlate at
  about 0.35; VMT also rises with ruralness.
- the zip log-odds of EV share has true gradients per SD of the z-scored zip
  covariate that GROW from 2016 to 2023: income +0.55 -> +0.95, ruralness
  -0.45 -> -0.80, political +0.30 -> +0.60, plus a zip residual of SD 0.35.
- the state aggregate is pinned to the real California EV stock series and to
  the real 2020-2023 sales series.
- counts are Binomial draws, so small zips are genuinely noisy.
"""


def main(seed=12345):
    rs = np.random.RandomState(seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    zip_df = gen_zip_covariates(rs)
    pop_df = gen_households(rs, zip_df)
    stock_df, sales_df = gen_zip_ev_series(rs, zip_df, pop_df)

    pop_path = os.path.join(OUT_DIR, "FAKE_synthetic_population.csv")
    pop_df.to_csv(pop_path, index=False)
    stock_df.to_csv(os.path.join(OUT_DIR, "FAKE_zip_ev_stock.csv"), index=False)
    sales_df.to_csv(os.path.join(OUT_DIR, "FAKE_zip_ev_sales.csv"), index=False)
    with open(os.path.join(OUT_DIR, "FAKE_DATA_README.md"), "w") as f:
        f.write(README)

    print(f"wrote FAKE data to {OUT_DIR}/")
    print(f"  households      : {len(pop_df):,} rows, {pop_df['zip'].nunique()} zips")
    print(f"  represented hh  : {pop_df['weight'].sum():,.0f}")
    print(f"  zip-year stock  : {len(stock_df):,} rows, years {stock_df['year'].min()}-{stock_df['year'].max()}")
    print(f"  state stock share by year:")
    chk = stock_df.groupby("year").apply(
        lambda g: g["ev_stock"].sum() / g["total_stock"].sum(), include_groups=False)
    print(chk.to_string())
    print(f"  state sales share by year:")
    chk2 = sales_df.groupby("year").apply(
        lambda g: g["ev_sales"].sum() / g["total_sales"].sum(), include_groups=False)
    print(chk2.to_string())
    print("\n  income-VMT correlation (should be > 0, this is the point): "
          f"{np.corrcoef(np.log(pop_df['income']), np.log(pop_df['vmt']))[0,1]:.3f}")


if __name__ == "__main__":
    main()
