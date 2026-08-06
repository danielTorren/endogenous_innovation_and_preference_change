"""
Zip-level synthetic population: agent draw and preference-parameter mapping.

This module takes over the five per-agent attribute draws that
controller.py used to do from independent parametric distributions
(gen_distance / gen_chi / gen_gamma / gen_nu / gen_beta) and instead
derives them from an empirical household-level synthetic population
carrying income, VMT, age, political leaning, ruralness and a zip code.

WHY THIS EXISTS (beyond "more realistic inputs"):

1. In the old code d_vec (VMT) and beta_vec (income) are INDEPENDENT draws.
   Empirically they are not: higher-income households drive more, so they get
   both low price sensitivity AND large absolute fuel savings from an EV. That
   correlation is a first-order driver of the adoption gradient the model is
   asked to reproduce, and it is simply absent from the parametric version.

2. It opens four new free parameters, each attaching a preference to an
   OBSERVED covariate rather than to a fitted spread:

     eps_beta  income elasticity of price sensitivity
               beta_i = beta_med * (median_income/income_i) ** eps_beta
               eps_beta = 1 is the old hard-coded value.

     a_rural   range anxiety by ruralness
               nu_i = nu * (1 + a_rural * ruralness_i)
               a_rural = 0 is the old homogeneous nu.

     rho_pol   rank correlation between WTP_E and political leaning
               rho_pol = 0 is the old independent normal draw.

     rho_age   rank correlation between chi (innovativeness) and -age
               rho_age = 0 is the old independent beta draw.

   The two correlations are imposed with a GAUSSIAN COPULA ON RANKS, not a
   regression. That leaves the marginal distributions of WTP_E and chi exactly
   as they are today, so rho = 0 nests the current model and the existing
   a_chi / b_chi posterior stays a valid starting point. One new parameter per
   channel, no re-fitting of a whole distribution.

3. Every agent carries a zip, so model output can be aggregated to zip and
   compared against zip-level EV stock and sales. That cross-sectional
   variation is where the identification of the four parameters above comes
   from; eight aggregate state-level numbers cannot do it.

SEEDING. All population draws use their own RandomState, seeded by
`seed_population`, NOT the shared `random_state_inputs`. Two reasons: the
existing NK-landscape / network / firm draws keep their exact RNG stream so
the fallback path is bit-for-bit unchanged, and population-sample uncertainty
becomes a separately reportable quantity (vary seed_population, hold
seed_inputs) rather than something hidden inside one fixed draw.

See package/calibration_data/FAKE_zip_data/FAKE_DATA_README.md for the input
column contract.
"""

import os
import numpy as np
from scipy.stats import norm, beta as beta_dist

# ---------------------------------------------------------------------------
# Columns required in the population file. Extra columns are ignored, so a
# richer real file drops in unchanged.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ("zip", "weight", "income", "vmt", "age",
                    "political_leaning", "ruralness")

# Optional. Absent => the spatial ordering falls back to ZIP CODE NUMBER, which
# is only roughly geographic. See _spatial_rank.
OPTIONAL_COLUMNS = ("latitude", "longitude")

# Resolution of the Hilbert grid used to turn 2-D zip centroids into a 1-D
# locality-preserving ordering. 2^10 = 1024 cells per axis is far finer than
# ~1,760 California ZCTAs need, so no two zips share a cell in practice.
HILBERT_ORDER = 10

# Coarse groups used ONLY for the between-group dispersion summary statistic
# (summary_stats.py). The gradients are computed at zip level and need no
# binning at all. Terciles x terciles = 9 groups, ~28 zips each in a
# 250-zip file, which keeps the group-level Monte Carlo noise well below the
# real between-group spread.
DEFAULT_STRATA_SPEC = {"income": 3, "ruralness": 3}

# The model's timestep is one month and d_vec is a PER-TIMESTEP distance: the
# Poisson that controller.gen_distance fits to the CA vehicle-survey bins has
# mean 1783 and median 1274, i.e. monthly miles per vehicle. A synthetic
# population will almost always carry ANNUAL VMT, so it has to be divided by
# 12. Getting this wrong is the single easiest way to break the whole model
# without anything erroring: a 12x too large d_vec makes lifetime fuel cost
# dominate the utility and every agent adopts an EV immediately. Hence the
# explicit `vmt_period` setting and the loud range check in gen_d_vec.
VMT_PERIOD_DIVISOR = {"annual": 12.0, "monthly": 1.0}

# Median monthly per-vehicle distance implied by the parametric default, and
# the factor either side of it that gen_d_vec will accept without warning.
PARAMETRIC_MEDIAN_D = 1274.0
D_VEC_WARN_FACTOR = 3.0

# Module-level cache. The calibration runs thousands of simulations; under
# joblib's loky backend each worker process is reused across tasks, so this
# turns "read the population file once per simulation" into "once per worker".
_POP_CACHE = {}


# ===========================================================================
# Loading
# ===========================================================================

def load_population(path, use_npz_cache=True):
    """
    Load the household-level synthetic population as a dict of numpy arrays.

    A real California household synthetic population is millions of rows; a
    CSV parse per simulation would dominate calibration runtime. So on first
    read the arrays are also written to a sibling `.npz`, and later reads
    prefer that (~50x faster). The npz is regenerated whenever the CSV is
    newer, so editing the CSV never silently serves stale data.
    """
    key = os.path.abspath(path)
    if key in _POP_CACHE:
        return _POP_CACHE[key]

    npz_path = os.path.splitext(path)[0] + ".cache.npz"
    if (use_npz_cache and os.path.exists(npz_path)
            and os.path.getmtime(npz_path) >= os.path.getmtime(path)):
        with np.load(npz_path) as z:
            pop = {k: z[k] for k in z.files}
    else:
        import pandas as pd
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"population file {path} is missing required column(s) {missing}. "
                f"Required: {list(REQUIRED_COLUMNS)}"
            )
        pop = {
            "zip": df["zip"].to_numpy(np.int64),
            "weight": df["weight"].to_numpy(np.float64),
            "income": df["income"].to_numpy(np.float64),
            "vmt": df["vmt"].to_numpy(np.float64),
            "age": df["age"].to_numpy(np.float64),
            "political_leaning": df["political_leaning"].to_numpy(np.float64),
            "ruralness": df["ruralness"].to_numpy(np.float64),
        }
        for c in OPTIONAL_COLUMNS:
            if c in df.columns:
                pop[c] = df[c].to_numpy(np.float64)
        _validate(pop, path)
        if use_npz_cache:
            # atomic: several joblib workers may hit this simultaneously.
            # The tmp name must end in .npz -- np.savez silently appends the
            # extension otherwise and the replace would look for a file that
            # was never written.
            tmp = npz_path + f".tmp{os.getpid()}.npz"
            np.savez(tmp, **pop)
            os.replace(tmp, npz_path)

    _POP_CACHE[key] = pop
    return pop


def _validate(pop, path):
    """Fail loudly on the input errors that would otherwise show up as a silently wrong calibration."""
    if np.any(pop["weight"] <= 0):
        raise ValueError(f"{path}: weight must be strictly positive")
    if np.any(pop["income"] <= 0):
        raise ValueError(f"{path}: income must be strictly positive (beta_i divides by it)")
    if np.any(pop["vmt"] <= 0):
        raise ValueError(f"{path}: vmt must be strictly positive (gamma_i divides by it)")
    for col in ("political_leaning", "ruralness"):
        v = pop[col]
        if np.nanmin(v) < 0.0 or np.nanmax(v) > 1.0:
            raise ValueError(
                f"{path}: {col} must be on [0,1] (got [{np.nanmin(v):.3g}, {np.nanmax(v):.3g}]). "
                f"ruralness enters nu_i = nu*(1 + a_rural*ruralness_i) multiplicatively, so its "
                f"scale is not free; political_leaning is rank-transformed but is checked for symmetry."
            )
    for col in REQUIRED_COLUMNS:
        if np.any(~np.isfinite(pop[col].astype(np.float64))):
            raise ValueError(f"{path}: {col} contains NaN/inf")
    if "latitude" in pop:
        if np.any(~np.isfinite(pop["latitude"])) or np.any(~np.isfinite(pop["longitude"])):
            raise ValueError(f"{path}: latitude/longitude contain NaN/inf. Drop the columns "
                             f"entirely rather than leaving gaps, so the ZIP-number fallback "
                             f"is used consistently instead of for an arbitrary subset.")
        if np.nanmax(np.abs(pop["latitude"])) > 90 or np.nanmax(np.abs(pop["longitude"])) > 180:
            raise ValueError(f"{path}: latitude/longitude out of range; expected decimal degrees")


# ===========================================================================
# Zip-level aggregation (shared by the model side and the data side, so the
# two agree on the canonical zip ordering and on the covariate z-scores)
# ===========================================================================

def weighted_quantile(values, quantiles, weights):
    """Weighted quantiles of `values`. quantiles in [0,1]."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    # midpoint convention, standard for weighted quantiles
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(np.atleast_1d(quantiles), cw, v)


def build_zip_table(pop, strata_spec=None):
    """
    Collapse the household population to one row per zip.

    Returns a dict of arrays with `zip` in ASCENDING order. That ordering is
    the canonical zip index used by everything downstream: the per-agent
    `zip_index`, the model's per-zip output histories, and the observed
    zip_ev_stock / zip_ev_sales tables. Both sides of the comparison call this
    same function, so there is exactly one definition of "zip 37".

    The covariates are z-scored HERE, weighted by household count, and the
    z-scores are what the gradient statistics regress on. Doing it once, from
    the population file alone, means the model side and data side use
    identical regressors -- if each side z-scored its own subset the recovered
    gradients would not be comparable across a zip holdout split.
    """
    if strata_spec is None:
        strata_spec = DEFAULT_STRATA_SPEC

    zips, inv = np.unique(pop["zip"], return_inverse=True)
    nz = len(zips)
    w = pop["weight"]

    households = np.bincount(inv, weights=w, minlength=nz)

    def wmean(x):
        return np.bincount(inv, weights=w * x, minlength=nz) / households

    # income uses a weighted MEDIAN (household income is right-skewed and the
    # mean is dragged around by a handful of high earners in small zips)
    median_income = np.empty(nz)
    for z in range(nz):
        m = inv == z
        median_income[z] = weighted_quantile(pop["income"][m], 0.5, w[m])[0]

    table = {
        "zip": zips,
        "households": households,
        "median_income": median_income,
        "mean_vmt": wmean(pop["vmt"]),
        "mean_age": wmean(pop["age"]),
        # ruralness / political_leaning are zip-level in the input; the
        # weighted mean recovers the constant exactly and tolerates a real
        # file that happens to vary them slightly within zip
        "ruralness": wmean(pop["ruralness"]),
        "political_leaning": wmean(pop["political_leaning"]),
    }

    # regressors for the gradient statistics: z-scored across zips, weighted
    # by household count. log income because the covariate that matters for
    # beta_i is log income (beta_i is a power of an income ratio).
    table["z_income"] = _wzscore(np.log(table["median_income"]), households)
    table["z_ruralness"] = _wzscore(table["ruralness"], households)
    table["z_political"] = _wzscore(table["political_leaning"], households)

    table["strata_index"], table["strata_labels"] = _build_strata(
        table, households, strata_spec
    )

    # geography: coordinates (if supplied) and the 1-D locality-preserving order
    has_coords = "latitude" in pop
    if has_coords:
        table["latitude"] = wmean(pop["latitude"])
        table["longitude"] = wmean(pop["longitude"])
    table["has_coords"] = has_coords
    table["spatial_rank"] = _spatial_rank(table, has_coords)
    return table


# ---------------------------------------------------------------------------
# Turning 2-D geography into a 1-D ring order
# ---------------------------------------------------------------------------

def _hilbert_index(x, y, order=HILBERT_ORDER):
    """
    Hilbert curve index of integer grid coordinates (x, y) on a 2^order grid.

    A Hilbert curve is used rather than sorting by latitude, or a Morton/Z-order
    curve, because the ring position IS the network neighbourhood: two zips that
    are close on the curve must be close in space, or "spatial homophily" would
    connect agents who merely share a latitude. Sorting by one axis fails
    completely in the other axis. Morton order preserves locality on average but
    has periodic long jumps at bit boundaries, which would wire a handful of
    distant zip pairs together and, because K = 150 neighbours here, each such
    jump contaminates many edges. Hilbert has no such jumps: consecutive indices
    are always adjacent cells.

    Standard iterative xy->d algorithm. Vectorised over the input arrays.
    """
    x = np.asarray(x, dtype=np.int64).copy()
    y = np.asarray(y, dtype=np.int64).copy()
    n = 1 << order
    d = np.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rotate/reflect the quadrant
        flip = (ry == 0)
        xf = np.where(flip & (rx == 1), s - 1 - x, x)
        yf = np.where(flip & (rx == 1), s - 1 - y, y)
        x = np.where(flip, yf, xf)
        y = np.where(flip, xf, yf)
        s >>= 1
    return d


def _spatial_rank(table, has_coords):
    """
    Rank of each zip along a 1-D spatial ordering, 0 = one end of the curve.

    With coordinates: the Hilbert index of the zip centroid on a 2^HILBERT_ORDER
    grid scaled to the bounding box of the data.

    Without coordinates: ZIP CODE NUMBER, on the basis that US ZIP codes are
    assigned roughly geographically. This is a real property but a crude one --
    numerically adjacent ZIPs are usually physically adjacent, and the failures
    cluster at metro boundaries, which is exactly where a spatial homophily
    parameter is doing its most interesting work. Warned about at call time.
    """
    nz = len(table["zip"])
    if not has_coords:
        print("  WARNING no latitude/longitude in the population file. Spatial proximity "
              "is falling back to ZIP CODE NUMBER order, which is only roughly geographic "
              "and is worst at metro boundaries. This affects homophily_spatial_weight and "
              "Moran's I. Zip centroids are a free join from the Census ZCTA gazetteer.")
        return np.argsort(np.argsort(table["zip"])).astype(np.int64)

    lat, lon = table["latitude"], table["longitude"]
    n = (1 << HILBERT_ORDER) - 1
    gx = np.round(_unit_scale(lon) * n).astype(np.int64)
    gy = np.round(_unit_scale(lat) * n).astype(np.int64)
    h = _hilbert_index(gx, gy)
    # rank, so the scale matches the income rank it gets blended with
    return np.argsort(np.argsort(h)).astype(np.int64)


def _unit_scale(v):
    lo, hi = np.min(v), np.max(v)
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def zip_knn_weights(table, k=8, zip_subset=None, row_standardise=True):
    """
    Row-standardised k-nearest-neighbour spatial weight matrix over zips, for
    Moran's I in summary_stats.py.

    kNN rather than a distance band because California zip density varies by
    orders of magnitude between downtown Los Angeles and Modoc County: any fixed
    distance band that gives a rural zip neighbours at all would give an urban
    zip thousands, and Moran's I would then be dominated by the urban block. kNN
    gives every zip the same number of neighbours, so the statistic measures
    spatial pattern rather than spatial sampling density.

    When `zip_subset` is given the neighbourhoods are rebuilt AMONG the subset
    only. That matters for the zip-holdout validation splits: reusing the
    full-sample neighbours would let a held-out zip's Moran contribution depend
    on training zips.
    """
    idx = np.arange(len(table["zip"])) if zip_subset is None else np.where(zip_subset)[0]
    m = len(idx)
    if m < k + 1:
        return None, idx

    if table["has_coords"]:
        # equirectangular approximation, fine at the scale of one US state
        lat = table["latitude"][idx]
        lon = table["longitude"][idx]
        y = lat
        x = lon * np.cos(np.deg2rad(np.mean(lat)))
        pts = np.column_stack([x, y])
    else:
        pts = table["spatial_rank"][idx].astype(float)[:, None]

    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]

    W = np.zeros((m, m))
    rows = np.repeat(np.arange(m), k)
    W[rows, nn.ravel()] = 1.0
    if row_standardise:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
    return W, idx


def _wzscore(x, w):
    m = np.average(x, weights=w)
    s = np.sqrt(np.average((x - m) ** 2, weights=w))
    return (x - m) / (s if s > 0 else 1.0)


def _build_strata(table, households, strata_spec):
    """
    Assign every zip to one coarse stratum, as the full cross of
    weighted-quantile bins of the covariates named in `strata_spec`.

    Used ONLY for the between-group dispersion summary statistic. Bin edges
    come from the weighted quantiles of the whole population, so they are a
    property of the population file, identical on the model and data sides,
    and unaffected by which zips a validation split happens to hold out.
    """
    codes = np.zeros(len(table["zip"]), dtype=np.int64)
    labels_per_dim = []
    mult = 1
    for name, nbins in sorted(strata_spec.items()):
        col = {"income": "median_income", "ruralness": "ruralness",
               "political": "political_leaning", "age": "mean_age",
               "vmt": "mean_vmt"}[name]
        v = table[col]
        edges = weighted_quantile(v, np.linspace(0, 1, nbins + 1)[1:-1], households)
        b = np.searchsorted(edges, v, side="right")
        codes += mult * b
        mult *= nbins
        labels_per_dim.append([f"{name}{i}" for i in range(nbins)])

    labels = []
    for code in range(mult):
        parts, c = [], code
        for dim_labels in labels_per_dim:
            parts.append(dim_labels[c % len(dim_labels)])
            c //= len(dim_labels)
        labels.append("_".join(parts))
    return codes, labels


# ===========================================================================
# The agent draw + preference mapping
# ===========================================================================

class SyntheticPopulation:
    """
    Draws `num_individuals` model agents from the household population and
    maps their covariates onto the model's per-agent preference vectors.

    Instantiated once per simulation from controller.gen_users_parameters().
    The controller still owns the ORDER in which the vectors are built
    (distance -> chi -> gamma -> nu -> beta, because calc_beta_median needs
    the medians of gamma, nu and d), so that dependency stays in one place;
    this class only owns HOW each vector is drawn.
    """

    def __init__(self, params, num_individuals, seed_population,
                 homophily_strength=0.0, homophily_spatial_weight=0.5):
        self.params = params
        self.num_individuals = int(round(num_individuals))
        self.rs = np.random.RandomState(seed_population)

        self.homophily_strength = float(homophily_strength)
        self.homophily_spatial_weight = float(homophily_spatial_weight)
        if not 0.0 <= self.homophily_strength <= 1.0:
            raise ValueError(f"homophily_strength must be on [0,1], got {homophily_strength}")
        if not 0.0 <= self.homophily_spatial_weight <= 1.0:
            raise ValueError(f"homophily_spatial_weight must be on [0,1], "
                             f"got {homophily_spatial_weight}")

        self.strata_spec = params.get("strata_spec", DEFAULT_STRATA_SPEC)
        self.pop = load_population(params["population_path"])
        self.zip_table = build_zip_table(self.pop, self.strata_spec)

        self._draw_agents()

    # ---------------------------------------------------------------- draw
    def _draw_agents(self):
        """
        Draw agents with probability proportional to their expansion weight,
        with replacement.

        Proportional (rather than stratified-equal) draw means no per-agent
        weights are needed anywhere downstream: an unweighted mean over agents
        is already an unbiased estimate of the population mean, and per-zip
        agent counts automatically track real zip populations, which is
        exactly the weighting the zip-level gradient regression wants.

        AGENT ORDER IS NETWORK STRUCTURE. The social network is a
        Watts-Strogatz ring lattice over agent INDEX (socialNetworkUsers.
        create_network), so index adjacency IS network adjacency. Which
        household sits at which index therefore decides who talks to whom, and
        that is precisely the lever homophily pulls -- see _homophilous_order.
        At homophily_strength = 0 the order is uniformly random, reproducing the
        original behaviour.
        """
        w = self.pop["weight"]
        p = w / w.sum()
        idx = self.rs.choice(len(w), size=self.num_individuals, replace=True, p=p)
        idx = self._homophilous_order(idx)
        self.row_index = idx

        self.income = self.pop["income"][idx]
        self.vmt = self.pop["vmt"][idx]
        self.age = self.pop["age"][idx]
        self.ruralness = self.pop["ruralness"][idx]
        self.political_leaning = self.pop["political_leaning"][idx]

        # map each agent to its position in the canonical ascending zip list
        agent_zip = self.pop["zip"][idx]
        self.zip_index = np.searchsorted(self.zip_table["zip"], agent_zip)
        self.num_zips = len(self.zip_table["zip"])
        self.zip_agent_counts = np.bincount(self.zip_index, minlength=self.num_zips)

        # stratum of each agent, via its zip
        self.strata_index = self.zip_table["strata_index"][self.zip_index]
        self.num_strata = len(self.zip_table["strata_labels"])

    # --------------------------------------------------- network placement
    def _homophilous_order(self, idx):
        """
        Order the drawn households along the small-world ring so that
        neighbours are similar, with two parameters.

        homophily_strength h in [0,1]
            0 = uniformly random placement (the original behaviour, and the ring
                carries no information about who anyone is)
            1 = placement determined entirely by the similarity index

        homophily_spatial_weight w in [0,1]
            0 = similarity is INCOME only: an agent's ring neighbours are the
                agents with the most similar income, wherever they live
            1 = similarity is PHYSICAL PROXIMITY only: an agent's ring
                neighbours are the agents in the nearest zips, whatever they earn

        HOW h INTERPOLATES. The placement key is

            key_i = h * z_similarity_i + sqrt(1 - h^2) * e_i,     e ~ N(0,1)

        and agents are sorted by it. This is the same Gaussian-copula device used
        for rho_pol and rho_age, and it is chosen over ad-hoc alternatives (such
        as "randomly reshuffle a fraction 1-h of agents") for one concrete
        reason: h is then exactly the CORRELATION between an agent's ring
        position and its similarity index. That makes h scale-free, monotone,
        and comparable across different similarity definitions -- so a fitted
        h = 0.4 means the same thing at w = 0 and w = 1, which it would not if h
        were a reshuffled fraction.

        HOW w BLENDS. z_similarity = (1-w) * z_income + w * z_spatial, on
        rank-normal transforms of income and of the zip's Hilbert-curve position.
        Rank-normal on both sides so the blend is not dominated by whichever
        input happens to have the heavier tail: raw income is lognormal with
        log-sd ~0.93 while a Hilbert rank is uniform, and blending those
        directly would make w = 0.5 behave almost like w = 0.

        Note the two are CORRELATED in reality (rich zips cluster), so w is
        identified by the residual: how much of the observed spatial clustering
        of EV adoption survives once income clustering is accounted for. That is
        exactly what Moran's I in the summary vector measures, and it is why
        adding this parameter without that statistic would have left w and the
        chi contagion parameters mutually unidentifiable.

        WHAT h = 1 DOES NOT DO. The Watts-Strogatz rewiring probability
        (SW_prob_rewire, 0.1) is applied AFTER placement and is untouched, so
        about 10% of every agent's edges remain uniformly random long-range
        bridges no matter how high h is. Homophily therefore saturates: h = 1 is
        strongly assortative local structure, not a disconnected set of cliques.
        That is deliberate -- it keeps the small-world property that motivates
        the network in the first place -- but it means the achievable range of
        network assortativity is bounded by prob_rewire. Lower prob_rewire if you
        need segregation to bite harder, and report realised_homophily() rather
        than h alone.
        """
        n = len(idx)
        h = self.homophily_strength
        if h == 0.0:
            # uniformly random: identical in distribution to the original
            # unconditional shuffle, and skips building the similarity index
            self.rs.shuffle(idx)
            self._placement_key = None
            return idx

        w = self.homophily_spatial_weight
        agent_zip_pos = np.searchsorted(self.zip_table["zip"], self.pop["zip"][idx])

        z_income = self._rank_normal_of(self.pop["income"][idx])
        # ties within a zip are broken at random by _rank_normal_of's jitter, so
        # same-zip agents are contiguous but internally unordered
        z_spatial = self._rank_normal_of(
            self.zip_table["spatial_rank"][agent_zip_pos].astype(np.float64))

        z_sim = (1.0 - w) * z_income + w * z_spatial
        # renormalise: a blend of two correlated unit-variance terms does not
        # have unit variance, and h must remain the correlation with the key
        sd = z_sim.std()
        if sd > 0:
            z_sim = z_sim / sd

        key = h * z_sim + np.sqrt(1.0 - h ** 2) * self.rs.normal(size=n)
        order = np.argsort(key)
        self._placement_key = key[order]
        return idx[order]

    def _rank_normal_of(self, v):
        """Rank-normal transform of an arbitrary vector, random tie-breaking."""
        n = len(v)
        jitter = self.rs.rand(n)
        order = np.lexsort((jitter, v))
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        return norm.ppf(ranks / (n + 1.0))

    def realised_homophily(self, adjacency=None, k_ring=None, prob_rewire=None):
        """
        What the network ACTUALLY ended up assorting on. Report this, not just
        the input parameters.

        Returns Pearson correlations across connected pairs for log income and
        for spatial position, plus the mean great-circle-ish distance between
        connected agents. Because prob_rewire caps the achievable assortativity
        (see _homophilous_order), the input h and the realised correlation are
        not the same number and the gap grows with prob_rewire.

        Pass the model's adjacency matrix for exact figures. Called without one,
        it evaluates the RING that placement produced (before rewiring), using
        k_ring neighbours either side, which is the cheap version and an upper
        bound on the realised value.
        """
        n = self.num_individuals
        li = np.log(self.income)
        zp = self.zip_table["spatial_rank"][self.zip_index].astype(float)

        if adjacency is not None:
            r, c = np.nonzero(adjacency)
            keep = r < c
            r, c = r[keep], c[keep]
        else:
            k = int(k_ring if k_ring is not None else 1)
            offs = np.arange(1, k + 1)
            r = np.repeat(np.arange(n), k)
            c = (r + np.tile(offs, n)) % n

        out = {
            "income_assortativity": float(np.corrcoef(li[r], li[c])[0, 1]),
            "spatial_assortativity": float(np.corrcoef(zp[r], zp[c])[0, 1]),
            "n_pairs": int(len(r)),
        }
        if self.zip_table["has_coords"]:
            la = self.zip_table["latitude"][self.zip_index]
            lo = self.zip_table["longitude"][self.zip_index]
            dx = (lo[r] - lo[c]) * np.cos(np.deg2rad(np.mean(la)))
            dy = la[r] - la[c]
            out["mean_pair_distance_km"] = float(np.mean(np.sqrt(dx ** 2 + dy ** 2)) * 111.0)
            out["same_zip_share"] = float(np.mean(self.zip_index[r] == self.zip_index[c]))
        return out

    # ------------------------------------------------------- rank helper
    def _rank_normal(self, x, descending=False):
        """
        Map a covariate to a standard normal by its RANK, i.e. the Gaussian
        copula's marginal transform. Rank-based on purpose: it is invariant to
        any monotone rescaling of the covariate, so the recovered rho_pol /
        rho_age do not depend on whether political leaning was supplied as a
        vote share, a z-score or a 1-7 Likert index.

        Ties (e.g. political_leaning is constant within a zip, so hundreds of
        agents tie) are broken at random rather than given the same rank. Equal
        ranks would collapse those agents onto identical copula draws and put a
        spike in the marginal; random tie-breaking keeps the marginal exactly
        uniform, which is what makes rho = 0 reproduce the old draw.
        """
        n = len(x)
        jitter = self.rs.rand(n)
        key = -x if descending else x
        order = np.lexsort((jitter, key))
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        return norm.ppf(ranks / (n + 1.0))

    def _copula(self, z_cov, rho):
        """Standard normal correlated with z_cov at exactly `rho`."""
        if rho == 0.0:
            return self.rs.normal(size=self.num_individuals)
        return rho * z_cov + np.sqrt(1.0 - rho ** 2) * self.rs.normal(size=self.num_individuals)

    # ------------------------------------------------------------ vectors
    def gen_d_vec(self, warn=True):
        """
        Per-timestep (monthly) driving distance from the population's VMT column.

        Replaces the Poisson fitted to six aggregate CA-survey bins
        (controller.gen_distance, fitted_lambda = 1.5428...). Two gains: the
        within-population VMT distribution is the empirical one rather than a
        six-atom lattice, and VMT is now jointly distributed with income
        instead of independent of it.

        UNITS. The model steps monthly and d_vec is a per-timestep distance, so
        an annual VMT column is divided by 12. Set
        parameters_synthetic_population.vmt_period to "monthly" if the real file
        already supplies monthly VMT. This is not a detail: a 12x too large
        d_vec makes the lifetime fuel-cost term swamp everything else in the
        utility and drives EV share to 1.0 within a year, with no error raised
        anywhere. The range check below is the tripwire for that, and for a VMT
        column that turns out to be in kilometres.
        """
        period = self.params.get("vmt_period", "annual")
        if period not in VMT_PERIOD_DIVISOR:
            raise ValueError(
                f"vmt_period must be one of {list(VMT_PERIOD_DIVISOR)}, got {period!r}")
        d = self.vmt / VMT_PERIOD_DIVISOR[period]

        med = float(np.median(d))
        lo = PARAMETRIC_MEDIAN_D / D_VEC_WARN_FACTOR
        hi = PARAMETRIC_MEDIAN_D * D_VEC_WARN_FACTOR
        if warn and not (lo < med < hi):
            print(
                f"  WARNING d_vec median is {med:,.0f} monthly miles/vehicle after "
                f"treating the vmt column as {period}. The parametric default is "
                f"{PARAMETRIC_MEDIAN_D:,.0f} (mean 1,783), so this is off by "
                f"{med / PARAMETRIC_MEDIAN_D:.1f}x. Check vmt_period, and check "
                f"whether the column is per-household or per-vehicle and in miles "
                f"or kilometres. Every model parameter was calibrated against the "
                f"parametric scale, so a wrong scale here silently invalidates all of them."
            )
        return d

    def gen_chi_vec(self, a_chi, b_chi, chi_max, proportion_zero_target, rho_age):
        """
        Innovativeness. Marginal is still Beta(a_chi, b_chi) scaled by chi_max
        with a proportion of exact zeros, EXACTLY as before; rho_age only
        controls how that marginal is coupled to -age (younger = more
        innovative). rho_age = 0 reproduces the old independent draw.

        Implemented as inverse-CDF of the Beta at a copula-correlated uniform,
        which is why the marginal is preserved exactly rather than
        approximately.
        """
        z_age = self._rank_normal(self.age, descending=True)   # young -> high z
        u = norm.cdf(self._copula(z_age, rho_age))
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        chi_continuous = beta_dist.ppf(u, a_chi, b_chi)

        num_zeros = int(proportion_zero_target * self.num_individuals)
        if num_zeros > 0:
            zero_idx = self.rs.choice(self.num_individuals, size=num_zeros, replace=False)
            chi_continuous[zero_idx] = 0.0

        return chi_continuous * chi_max

    def gen_WTP_E_vec(self, WTP_E_mean, WTP_E_sd, gamma_epsilon, rho_pol):
        """
        Willingness to pay for avoided emissions. Marginal is still
        Normal(WTP_E_mean, WTP_E_sd) clipped below at gamma_epsilon, exactly as
        controller.gen_gamma does; rho_pol couples it to political leaning
        (more liberal = higher WTP_E). rho_pol = 0 reproduces the old draw.

        Worth noting what this buys: the current WTP_E_sd/WTP_E_mean ratio is
        0.84, so most of the model's gamma spread is a free fitted quantity.
        rho_pol moves part of that variance onto an observed covariate, making
        it predictive instead of fitted. The caller still divides by d_vec and
        applies the discounting constant, so gamma_i inherits the VMT
        correlation too.
        """
        z_pol = self._rank_normal(self.political_leaning)
        z = self._copula(z_pol, rho_pol)
        wtp = WTP_E_mean + WTP_E_sd * z
        return np.clip(wtp, a_min=gamma_epsilon, a_max=np.inf)

    def gen_nu_vec(self, nu, a_rural, nu_epsilon):
        """
        Range preference, scaled by ruralness:  nu_i = nu * (1 + a_rural * ruralness_i)

        nu multiplies +(B * Eff_omega)^zeta in the utility (socialNetworkUsers
        line ~981), i.e. the value placed on vehicle RANGE. ICE range in this
        model (fuel_tank 469 x efficiency ~1.5) exceeds EV range (battery ~75 x
        efficiency ~6), so a higher nu penalises EVs. Rural agents therefore
        adopt less, which is the range-anxiety / charging-access channel and is
        directly what a charging-infrastructure counterfactual acts on.

        a_rural = 0 reproduces the old homogeneous nu exactly.
        """
        nu_vec = nu * (1.0 + a_rural * self.ruralness)
        return np.clip(nu_vec, a_min=nu_epsilon, a_max=np.inf)

    def gen_beta_vec(self, median_beta, eps_beta):
        """
        Price/quality sensitivity:  beta_i = median_beta * (median_income/income_i) ** eps_beta

        eps_beta = 1 is the currently hard-coded exponent
        (controller.gen_beta), so eps_beta is a strict generalisation.

        The median of beta_vec stays exactly median_beta for any eps_beta > 0:
        median_income/income_i has median 1, and x -> x**eps_beta is monotone,
        so the median maps to 1**eps_beta = 1. That matters because
        calc_beta_median() is derived analytically to place the median agent at
        a sensible point in the utility, and eps_beta must not drag that anchor
        around while it changes the SPREAD.

        NO SHUFFLE, unlike controller.gen_beta. There, shuffling was harmless
        (the incomes were an i.i.d. draw anyway). Here agent order carries the
        zip mapping, so shuffling beta alone would decouple it from the agent's
        own income, VMT and zip and destroy the entire point of the dataset.
        """
        median_income = np.median(self.income)
        return median_beta * (median_income / self.income) ** eps_beta

    # ------------------------------------------------------------ reporting
    def summary(self):
        """Diagnostics for a single run; printed by the smoke test, not used in the model."""
        return {
            "num_individuals": self.num_individuals,
            "num_zips": self.num_zips,
            "num_strata": self.num_strata,
            "agents_per_zip_min": int(self.zip_agent_counts.min()),
            "agents_per_zip_median": float(np.median(self.zip_agent_counts)),
            "empty_zips": int(np.sum(self.zip_agent_counts == 0)),
            "corr_log_income_log_vmt": float(
                np.corrcoef(np.log(self.income), np.log(self.vmt))[0, 1]),
        }
