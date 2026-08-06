"""
The summary-statistic vector that the calibration targets, computed
identically from the model and from the observed zip-level data.

ONE function (`_compute`) produces the vector; `model_summary` and
`data_summary` differ only in how they assemble the per-zip counts they hand
it. That is deliberate and is the single most important property of this
module: if the model side and data side each had their own statistic code,
any drift between them would show up as a calibration bias that no amount of
posterior checking would reveal.

WHAT IS IN THE VECTOR (28 dims with the default spec)

  8  state EV stock share, 2016..2023, April snapshot
       -- unchanged from the current calibration, the aggregate anchor
  8  state EV sales share, per year
       -- already loaded by calibration_data_outputs.py and then never used
  9  cross-sectional GRADIENTS: 3 covariates x 3 years
       -- weighted OLS of zip EV stock share on the z-scored zip covariates
          (log median income, ruralness, political leaning). This is what
          identifies eps_beta, a_rural and rho_pol. Zip-level rather than
          stratum-level: pooling ~250 zips averages away the per-zip Monte
          Carlo noise (12 agents/zip at num_individuals=3000) far more
          effectively than binning does, and it needs no cut points that the
          model and data sides would have to agree on.
  3  between-stratum DISPERSION of EV share, 3 years
       -- stock-weighted SD across the coarse strata. Partly redundant with
          the gradients by construction; it is here because it is the moment
          whose GROWTH over time separates diffusion from static sorting.
          Run package/validation/noise_screen.py: if these three dims carry
          no theta signal above seed noise, drop them from the spec rather
          than let the density estimator spend capacity on them.

  3  MORAN'S I of zip EV share, 3 years
       -- spatial autocorrelation, on a k-nearest-neighbour weight matrix built
          from the zip centroids. THIS IS WHAT MAKES THE HOMOPHILY PARAMETERS
          IDENTIFIABLE, and it was added in the same change that introduced
          them, because the two cannot be separated afterwards.

          The problem it solves: once agents are placed in the network by
          similarity, spatially clustered EV adoption has TWO explanations that
          look identical in a single cross-section. Either similar people were
          placed together and they happen to share preferences (sorting, i.e.
          homophily_strength and homophily_spatial_weight), or adoption spread
          locally through the network (contagion, i.e. a_chi/b_chi). Fitting
          both to cross-sectional gradients alone would leave them trading off
          against each other with no way to tell which is doing the work.

          What separates them is TIME. Pure sorting produces a spatial pattern
          whose shape is essentially fixed from t=0: the clusters are wherever
          the covariates are, and they scale up together. Contagion produces
          spatial autocorrelation that GROWS, because adoption has to propagate
          outward from wherever it started. So Moran's I is taken at three
          separated years and it is the TRAJECTORY, not the level, that carries
          the identifying information. Read the three disp_* dims the same way.

          Caveat worth keeping in view: with homophily_spatial_weight near 0 the
          network is income-assortative but not spatial, so contagion through it
          still produces spatial autocorrelation via the income-geography
          correlation, just weaker. The separation is therefore a matter of
          degree rather than a clean exclusion restriction. Check it empirically
          with noise_screen.py rather than assuming it.

Every statistic accepts a `zip_subset`, which is what makes the zip-holdout
validation splits work: the same code computes "the target over training zips"
and "the target over held-out zips".
"""

import numpy as np

# ---------------------------------------------------------------------------
# The default spec. Editing this changes the calibration target, so it is
# saved alongside every calibration run and re-read by validation rather than
# re-imported, in case it changes later.
#
# `sales_years` is short because the real EV_Sales.xlsx only carries four
# annual values (2020-2023). The FAKE zip data covers 2016-2023, so widen this
# once the real zip sales series is longer. Nothing else needs to change.
# ---------------------------------------------------------------------------
DEFAULT_SPEC = {
    "stock_years":      [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "sales_years":      [2020, 2021, 2022, 2023],
    "gradient_years":   [2018, 2021, 2023],
    "dispersion_years": [2018, 2021, 2023],
    "moran_years":      [2018, 2021, 2023],
    "moran_k":          8,          # nearest-neighbour count for the spatial weights
    "snapshot_month":   4,          # April, matching the existing convert_data()
    "sales_include_used": False,    # True if the observed "sales" are all registrations
    "base_year":        2001,       # calendar year of controller timestep 0 (post burn-in)
}

GRADIENT_COVARIATES = ("z_income", "z_ruralness", "z_political")


# ===========================================================================
# Dimension naming. Masks in package/validation/splits.py are built by name,
# never by position, so reordering or extending the spec cannot silently
# invalidate a saved split.
# ===========================================================================

def dim_names(spec):
    names = []
    names += [f"stock_{y}" for y in spec["stock_years"]]
    names += [f"sales_{y}" for y in spec["sales_years"]]
    for y in spec["gradient_years"]:
        for c in GRADIENT_COVARIATES:
            names.append(f"grad_{c[2:]}_{y}")
    names += [f"disp_{y}" for y in spec["dispersion_years"]]
    names += [f"moran_{y}" for y in spec.get("moran_years", [])]
    return names


def dim_index(spec):
    return {n: i for i, n in enumerate(dim_names(spec))}


# ===========================================================================
# Time indexing. Kept here so the model side and data side cannot disagree
# about which month "2023" is.
# ===========================================================================

def month_index(year, month, base_params, spec):
    """
    Controller timestep for a given calendar year/month.

    Same convention as the existing convert_data() in
    NN_multi_round_calibration_multi_gen.py: timestep `duration_burn_in`
    is January of `base_year`, and the burn-in period sits before it.
    """
    return ((year - spec["base_year"]) * 12
            + base_params["duration_burn_in"]
            + (month - 1))


def _check_range(idx, n_hist, label):
    if idx < 0 or idx >= n_hist:
        raise IndexError(
            f"{label}: timestep {idx} outside the simulated history of length {n_hist}. "
            f"Either duration_calibration is too short for the years in the summary spec, "
            f"or duration_burn_in / base_year is inconsistent with them."
        )


# ===========================================================================
# The shared statistic computation
# ===========================================================================

def _compute(spec, zip_table, zip_subset, stock_ev, stock_tot, sales_ev, sales_tot):
    """
    Build the summary vector from per-zip counts.

    Args:
        zip_table: output of synthetic_population.build_zip_table -- supplies the
            canonical zip ordering, the z-scored covariates and the strata.
        zip_subset: boolean mask over zips (len == number of zips), or None for
            all zips. Applied to EVERY statistic including the state
            aggregates, so a held-out zip leaks nothing into the fitted target.
        stock_ev / stock_tot: dict year -> (num_zips,) counts.
        sales_ev / sales_tot: dict year -> (num_zips,) counts.

    Returns:
        (n_dims,) float64 vector, ordered as dim_names(spec).
    """
    nz = len(zip_table["zip"])
    if zip_subset is None:
        sub = np.ones(nz, dtype=bool)
    else:
        sub = np.asarray(zip_subset, dtype=bool)
        if sub.shape != (nz,):
            raise ValueError(f"zip_subset must have shape ({nz},), got {sub.shape}")

    out = []

    # ---- state aggregates -------------------------------------------------
    for y in spec["stock_years"]:
        tot = stock_tot[y][sub].sum()
        out.append(stock_ev[y][sub].sum() / tot if tot > 0 else np.nan)

    for y in spec["sales_years"]:
        tot = sales_tot[y][sub].sum()
        out.append(sales_ev[y][sub].sum() / tot if tot > 0 else np.nan)

    # ---- cross-sectional gradients ---------------------------------------
    # Weighted OLS, weights = that side's own vehicle/agent count per zip.
    # Multivariate rather than three high-minus-low contrasts because the three
    # covariates are genuinely correlated (rural zips are poorer and more
    # conservative), so a marginal contrast on income would silently carry the
    # ruralness effect with it.
    X = np.column_stack([np.ones(nz)] + [zip_table[c] for c in GRADIENT_COVARIATES])
    for y in spec["gradient_years"]:
        share, w = _share_and_weight(stock_ev[y], stock_tot[y], sub)
        out.extend(_wls(X, share, w)[1:])          # drop the intercept

    # ---- between-stratum dispersion --------------------------------------
    strata = zip_table["strata_index"]
    n_strata = int(strata.max()) + 1
    for y in spec["dispersion_years"]:
        out.append(_strata_dispersion(stock_ev[y], stock_tot[y], sub, strata, n_strata))

    # ---- spatial autocorrelation -----------------------------------------
    moran_years = spec.get("moran_years", [])
    if moran_years:
        W, keep_idx = _moran_weights(zip_table, spec.get("moran_k", 8), sub)
        for y in moran_years:
            out.append(_morans_i(stock_ev[y], stock_tot[y], W, keep_idx))

    return np.asarray(out, dtype=np.float64)


# Cache the kNN weight matrix per (zip_table identity, k, subset). It is the same
# every year and every simulation, and building it is O(nz^2) which would
# otherwise be paid three times per simulation across thousands of runs.
_W_CACHE = {}


def _moran_weights(zip_table, k, sub):
    key = (id(zip_table), int(k), sub.tobytes())
    if key not in _W_CACHE:
        from package.model.synthetic_population import zip_knn_weights
        _W_CACHE[key] = zip_knn_weights(zip_table, k=k, zip_subset=sub)
    return _W_CACHE[key]


def _morans_i(ev, tot, W, keep_idx):
    """
    Moran's I of the zip EV share over a row-standardised weight matrix.

        I = (n / sum(w)) * sum_ij w_ij (x_i - xbar)(x_j - xbar) / sum_i (x_i - xbar)^2

    Deliberately UNWEIGHTED by zip vehicle count, unlike every other statistic
    in this module. Those measure population aggregates, so they must be
    population-weighted; this one measures a spatial PATTERN, and weighting it by
    population would turn it into a statement about where people live rather than
    about how adoption is arranged in space.

    Zips with a zero denominator are dropped rather than treated as zero share,
    which would otherwise manufacture spatial structure out of missing data. If
    that leaves too few zips (a small validation holdout, say) the statistic is
    NaN and the caller sanitises it -- a Moran's I on a dozen scattered zips
    would be noise pretending to be a moment.
    """
    if W is None:
        return np.nan
    tot = np.asarray(tot, dtype=np.float64)[keep_idx]
    ev = np.asarray(ev, dtype=np.float64)[keep_idx]
    ok = tot > 0
    if ok.sum() < 10:
        return np.nan

    x = np.divide(ev, tot, out=np.zeros_like(tot), where=tot > 0)
    Wm = W[np.ix_(ok, ok)]
    x = x[ok]

    z = x - x.mean()
    denom = (z ** 2).sum()
    s_w = Wm.sum()
    if denom <= 0 or s_w <= 0:
        return np.nan
    return float((len(z) / s_w) * (z @ (Wm @ z)) / denom)


def _share_and_weight(ev, tot, sub):
    """Per-zip share with a zero-safe denominator, and the regression weight."""
    tot = np.asarray(tot, dtype=np.float64)
    ev = np.asarray(ev, dtype=np.float64)
    w = np.where(sub, tot, 0.0)
    share = np.divide(ev, tot, out=np.zeros_like(tot), where=tot > 0)
    return share, w


def _wls(X, y, w):
    """
    Weighted least squares via the normal equations.

    lstsq on the whitened design rather than an explicit inverse: zips with
    w == 0 (held out, or empty of model agents) contribute exactly nothing, and
    if a validation split leaves too few zips to identify all four columns the
    rank-deficient least-squares solution is returned instead of a blown-up
    inverse. Such a split is a design error, and run_validation.py reports the
    zip count per side so it is visible.
    """
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef


def _strata_dispersion(ev, tot, sub, strata, n_strata):
    """Stock-weighted SD of EV share across the coarse strata."""
    tot = np.where(sub, np.asarray(tot, dtype=np.float64), 0.0)
    ev = np.where(sub, np.asarray(ev, dtype=np.float64), 0.0)
    s_tot = np.bincount(strata, weights=tot, minlength=n_strata)
    s_ev = np.bincount(strata, weights=ev, minlength=n_strata)
    keep = s_tot > 0
    if keep.sum() < 2:
        return np.nan
    share = s_ev[keep] / s_tot[keep]
    w = s_tot[keep]
    m = np.average(share, weights=w)
    return float(np.sqrt(np.average((share - m) ** 2, weights=w)))


# ===========================================================================
# Model side
# ===========================================================================

def model_summary(controller, base_params, spec, zip_table, zip_subset=None):
    """
    Summary vector from a finished simulation.

    Reads the always-on per-zip histories added by
    socialNetworkUsers._init_zip_tracking. Stock is a snapshot in
    `snapshot_month`; sales are SUMMED over the twelve months of the calendar
    year, because the observed sales series is annual and a single month of
    model sales at num_individuals=3000 would be almost pure noise.
    """
    sn = controller.social_network
    if getattr(sn, "zip_index", None) is None:
        raise RuntimeError(
            "model_summary needs per-zip tracking. Set parameters_synthetic_population "
            "in the base params so controller.setup_synthetic_population() attaches a zip_index."
        )

    stock_hist = sn.history_zip_EV_stock
    n_hist = len(stock_hist)
    zip_counts = sn.zip_agent_counts.astype(np.float64)

    stock_ev, stock_tot = {}, {}
    for y in spec["stock_years"]:
        idx = month_index(y, spec["snapshot_month"], base_params, spec)
        _check_range(idx, n_hist, f"stock snapshot {y}")
        stock_ev[y] = stock_hist[idx].astype(np.float64)
        # every agent owns exactly one vehicle at all times in this model, so
        # the per-zip stock denominator is just the agent count
        stock_tot[y] = zip_counts

    if spec["sales_include_used"]:
        sales_ev_hist = [a + b for a, b in zip(sn.history_zip_new_sales_EV,
                                              sn.history_zip_used_sales_EV)]
        sales_tot_hist = [a + b for a, b in zip(sn.history_zip_new_sales,
                                               sn.history_zip_used_sales)]
    else:
        sales_ev_hist = sn.history_zip_new_sales_EV
        sales_tot_hist = sn.history_zip_new_sales

    sales_ev, sales_tot = {}, {}
    for y in spec["sales_years"]:
        start = month_index(y, 1, base_params, spec)
        _check_range(start, n_hist, f"sales year {y} start")
        _check_range(start + 11, n_hist, f"sales year {y} end")
        sales_ev[y] = np.sum(sales_ev_hist[start:start + 12], axis=0).astype(np.float64)
        sales_tot[y] = np.sum(sales_tot_hist[start:start + 12], axis=0).astype(np.float64)

    # gradient/dispersion years must be covered by the stock arrays above
    _require_years(spec, stock_ev)

    return _compute(spec, zip_table, zip_subset, stock_ev, stock_tot, sales_ev, sales_tot)


# ===========================================================================
# Data side
# ===========================================================================

def data_summary(stock_arrays, sales_arrays, spec, zip_table, zip_subset=None):
    """
    Summary vector from the observed zip-level data.

    `stock_arrays` / `sales_arrays` are the dicts returned by
    load_zip_observations(), keyed by year, each value a (num_zips,) array in
    the canonical zip order.
    """
    stock_ev, stock_tot = stock_arrays["ev"], stock_arrays["total"]
    sales_ev, sales_tot = sales_arrays["ev"], sales_arrays["total"]
    _require_years(spec, stock_ev)
    for y in spec["sales_years"]:
        if y not in sales_ev:
            raise KeyError(
                f"sales year {y} is in the summary spec but not in the observed sales data "
                f"(available: {sorted(sales_ev)}). Shorten spec['sales_years']."
            )
    return _compute(spec, zip_table, zip_subset, stock_ev, stock_tot, sales_ev, sales_tot)


def _require_years(spec, stock_ev):
    needed = (set(spec["gradient_years"]) | set(spec["dispersion_years"])
              | set(spec.get("moran_years", [])))
    missing = needed - set(spec["stock_years"])
    if missing:
        raise ValueError(
            f"gradient_years/dispersion_years/moran_years {sorted(missing)} must also appear "
            f"in stock_years, since all three are computed from the stock snapshot."
        )
    absent = needed - set(stock_ev)
    if absent:
        raise KeyError(f"stock data missing for years {sorted(absent)}")


# ===========================================================================
# Loading the observed zip tables into canonical-zip-order arrays
# ===========================================================================

def load_zip_observations(stock_path, sales_path, zip_table):
    """
    Read the two observed zip x year CSVs and reindex them onto the canonical
    zip order from `zip_table` (i.e. the ascending unique zips of the
    synthetic population file).

    A zip present in the observed data but absent from the population file is
    dropped WITH A WARNING, and vice versa a population zip with no observation
    gets a zero denominator, which every statistic here already treats as
    "no information" rather than as a zero share. Silent misalignment between
    the two zip universes would be the single easiest way to produce a
    confidently wrong calibration, so both are reported.
    """
    import pandas as pd

    zips = zip_table["zip"]
    nz = len(zips)
    pos = {int(z): i for i, z in enumerate(zips)}

    def read(path, ev_col, tot_col):
        df = pd.read_csv(path)
        for c in ("zip", "year", ev_col, tot_col):
            if c not in df.columns:
                raise ValueError(f"{path} is missing required column '{c}'")
        unknown = sorted(set(df["zip"]) - set(pos))
        if unknown:
            print(f"  WARNING {path}: {len(unknown)} zip(s) not in the population file, "
                  f"dropped (first few: {unknown[:5]})")
            df = df[df["zip"].isin(pos)]
        ev, tot = {}, {}
        for year, g in df.groupby("year"):
            e = np.zeros(nz); t = np.zeros(nz)
            j = g["zip"].map(pos).to_numpy()
            e[j] = g[ev_col].to_numpy(float)
            t[j] = g[tot_col].to_numpy(float)
            ev[int(year)] = e
            tot[int(year)] = t
        return {"ev": ev, "total": tot}

    stock = read(stock_path, "ev_stock", "total_stock")
    sales = read(sales_path, "ev_sales", "total_sales")

    any_year = next(iter(stock["total"].values()))
    n_missing = int(np.sum(any_year == 0))
    if n_missing:
        print(f"  WARNING {n_missing} of {nz} population zips have no stock observation; "
              f"they carry zero weight in every statistic.")

    return stock, sales
