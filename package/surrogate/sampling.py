"""
sampling.py — LHS design generation and ABM evaluation interface.

The ABM is expensive, so we front-load a Latin Hypercube Sample (LHS) to
cover the 5D policy space efficiently, then refine via active BO in optimisation.py.

Output columns (Y):
  0: ev_uptake        — mean final EV share across seeds (fraction, e.g. 0.94)
                        Always absolute — it's used as a constraint target
                        (94-96% EV share), not an objective, so it's never
                        made relative to BAU.
  1: utility          — mean cumulative utility across seeds (£)
  2: emissions        — mean cumulative emissions across seeds (kg CO2)
  3: net_cost         — mean cumulative net policy cost across seeds (£)

Columns 1-3 are BAU-relative deltas (this policy vs. doing nothing) whenever
a bau_baseline is passed to run_policy_combination()/evaluate_lhs() — see
compute_bau_baseline(). Paired per seed (common random numbers) rather than
subtracting one overall BAU mean, which cancels seed-specific noise. Without
a bau_baseline, all four columns are absolute (legacy behaviour).

Keeping outputs in raw (£, kg) units — whether absolute or BAU-relative — is
intentional: the surrogate standardises internally so the GP length-scales
are comparable.
"""

import json
import pickle
import numpy as np
from copy import deepcopy
from scipy.stats.qmc import LatinHypercube, scale
from joblib import Parallel, delayed, load
import multiprocessing

OUTPUT_NAMES = ["ev_uptake", "utility", "emissions", "net_cost"]

# ---------------------------------------------------------------------------
# Policy space definition
# ---------------------------------------------------------------------------

class PolicyBounds:
    """
    Defines the search range for each policy instrument.

    Adjust the bounds to match your domain knowledge — the surrogate will
    only be accurate within this region, so don't set them too wide.
    Setting lower=0 allows a policy to be switched off entirely.
    """

    def __init__(self, bounds: dict):
        """
        bounds: dict mapping policy name -> (lower, upper)
        """
        self.bounds = bounds

    @property
    def names(self):
        return list(self.bounds.keys())

    @property
    def lower(self):
        return np.array([v[0] for v in self.bounds.values()])

    @property
    def upper(self):
        return np.array([v[1] for v in self.bounds.values()])

    @property
    def n(self):
        return len(self.bounds)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)


def load_policy_bounds(path: str) -> PolicyBounds:
    """
    Load policy bounds from a JSON file.

    Expected format:
        {"Carbon_price": [0, 0.91], "Adoption_subsidy": [0, 36875.57], ...}

    Each value is a [lower, upper] list. Zero lower bound means the policy
    can be fully switched off.
    """
    with open(path) as f:
        raw = json.load(f)
    bounds = {name: tuple(limits) for name, limits in raw.items()}
    return PolicyBounds(bounds)


# ---------------------------------------------------------------------------
# LHS design
# ---------------------------------------------------------------------------

def generate_lhs(bounds: PolicyBounds, n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Latin Hypercube Sample over the policy space.

    LHS ensures every row and column of the unit hypercube is sampled exactly
    once, giving much better coverage than random sampling for the same budget.

    Returns: (n_samples, n_policies) array in original (un-normalised) units.
    """
    sampler = LatinHypercube(d=bounds.n, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    return scale(unit_samples, bounds.lower, bounds.upper)


# ---------------------------------------------------------------------------
# ABM interface
# ---------------------------------------------------------------------------

def _update_policy_intensity(params: dict, policy_name: str, intensity: float) -> dict:
    """Mirror of the existing update_policy_intensity in endogenous_policy_intensity_single_gen.py."""
    params["parameters_policies"]["States"][policy_name] = 1
    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity
    return params


def _reset_policies(params: dict) -> dict:
    """Zero out all policy states before applying a new combination."""
    for key in params["parameters_policies"]["States"]:
        params["parameters_policies"]["States"][key] = 0
    return params


def _single_seed_run(params: dict, controller_file: str) -> tuple:
    """Run one seed and return scalar outputs."""
    controller = load(controller_file)
    from package.resources.run import load_in_controller
    data = load_in_controller(controller, params)
    return (
        data.calc_EV_prop(),
        data.social_network.utility_cumulative,
        data.social_network.emissions_cumulative,
        data.calc_net_policy_distortion(),
    )


def compute_bau_baseline(base_params: dict, controller_files: list) -> dict:
    """
    Run BAU (all policies off) once across all seeds, keeping PER-SEED values
    (not means) so run_policy_combination() can subtract a paired baseline —
    seed i's policy result is compared against seed i's own BAU result, which
    cancels seed-specific noise (common random numbers) rather than just
    subtracting one overall BAU average from every point.

    Returns dict of per-seed arrays: {"ev_uptake", "utility", "emissions", "net_cost"},
    each shape (n_seeds,), aligned by index to controller_files.
    """
    params = deepcopy(base_params)
    params = _reset_policies(params)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(_single_seed_run)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )
    ev_arr, util_arr, emis_arr, cost_arr = (np.array(a) for a in zip(*results))
    return {
        "ev_uptake": ev_arr,
        "utility": util_arr,
        "emissions": emis_arr,
        "net_cost": cost_arr,
    }


def get_or_create_bau_baseline(
    base_params: dict, controller_files: list, cache_path: str = None,
) -> dict:
    """Cached wrapper around compute_bau_baseline() — same seeds, so it only needs computing once per calibration."""
    if cache_path:
        import os
        if os.path.exists(cache_path):
            print(f"Loading cached BAU baseline from {cache_path}")
            data = np.load(cache_path)
            return {k: data[k] for k in ("ev_uptake", "utility", "emissions", "net_cost")}

    baseline = compute_bau_baseline(base_params, controller_files)

    if cache_path:
        np.savez(cache_path, **baseline)
        print(f"BAU baseline saved to {cache_path}")

    return baseline


def run_policy_combination(
    base_params: dict,
    policy_dict: dict,
    controller_files: list,
    bau_baseline: dict = None,
) -> np.ndarray:
    """
    Run one policy combination across all pre-saved controller seeds in parallel.

    policy_dict: {policy_name: intensity_value, ...}
                 Policies with intensity=0 are left inactive.

    bau_baseline : per-seed BAU arrays from compute_bau_baseline()/get_or_create_bau_baseline().
                   If given, utility/emissions/net_cost are returned as deltas relative to
                   BAU (paired per seed) — i.e. "how much better/worse is this policy than
                   doing nothing". ev_uptake is never made relative: it's used as an absolute
                   constraint target (94-96% EV share), not an objective to optimise.
                   If None, all four outputs are absolute (legacy behaviour).

    Returns: shape (4,) array — [mean_ev_uptake, mean_utility, mean_emissions, mean_net_cost]
             (utility/emissions/net_cost are BAU-relative deltas when bau_baseline is given)
    """
    params = deepcopy(base_params)
    params = _reset_policies(params)
    for name, intensity in policy_dict.items():
        if intensity > 0:
            params = _update_policy_intensity(params, name, intensity)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(_single_seed_run)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )

    ev_arr, util_arr, emis_arr, cost_arr = (np.array(a) for a in zip(*results))

    if bau_baseline is not None:
        util_arr = util_arr - bau_baseline["utility"]
        emis_arr = emis_arr - bau_baseline["emissions"]
        cost_arr = cost_arr - bau_baseline["net_cost"]

    return np.array([
        np.mean(ev_arr),
        np.mean(util_arr),
        np.mean(emis_arr),
        np.mean(cost_arr),
    ])


# ---------------------------------------------------------------------------
# Batch LHS evaluation
# ---------------------------------------------------------------------------

def evaluate_lhs(
    bounds: PolicyBounds,
    n_samples: int,
    base_params: dict,
    controller_files: list,
    bau_baseline: dict = None,
    seed: int = 42,
    cache_path: str = None,
) -> tuple:
    """
    Generate an LHS design and evaluate each point with the ABM.

    If cache_path is given and the file exists, loads from cache instead of
    re-running — useful since each ABM evaluation takes ~seconds.

    bau_baseline : passed through to run_policy_combination() — if given,
                   utility/emissions/net_cost columns of Y are BAU-relative
                   deltas rather than absolute values (see run_policy_combination).

    Returns: X (n_samples, n_policies), Y (n_samples, 4)
    """
    if cache_path:
        import os
        if os.path.exists(cache_path):
            print(f"Loading cached LHS data from {cache_path}")
            data = np.load(cache_path)
            return data["X"], data["Y"]

    X = generate_lhs(bounds, n_samples, seed=seed)
    Y = np.zeros((n_samples, 4))

    for i, x in enumerate(X):
        policy_dict = dict(zip(bounds.names, x))
        print(f"  LHS {i+1}/{n_samples}: {policy_dict}")
        Y[i] = run_policy_combination(base_params, policy_dict, controller_files, bau_baseline=bau_baseline)

    if cache_path:
        np.savez(cache_path, X=X, Y=Y)
        print(f"LHS data saved to {cache_path}")

    return X, Y


# ---------------------------------------------------------------------------
# Pairwise warmstart
# ---------------------------------------------------------------------------

def load_pairwise_warmstart(path: str, bounds: PolicyBounds) -> tuple:
    """
    Convert pre-computed pairwise ABM outcomes into surrogate training data.

    The pkl file contains results for every pair of policies run at several
    intensity levels, with all other policies held at zero.  These are real
    ABM evaluations (same seed ensemble as the main workflow) and can be used
    directly as X_init / Y_init for the BO loop, bypassing or reducing the
    LHS phase.

    WHY THIS HELPS
    --------------
    - 100 real evaluations already available (20 pairs × 5 intensities)
    - 53 of those already fall inside the 94–96 % EV feasibility window
    - Points cover all 10 pairwise edges of the 5D space, giving the GP
      strong boundary structure information before the first BO iteration
    - The BO can then focus on the interior (3-, 4-, 5-policy combinations)

    LIMITATIONS
    -----------
    - Points are sparse 2D "faces" of the 5D space (other 3 policies = 0)
    - Adding a small LHS (20–30 points) before BO is still recommended so
      the GP sees some interior points before it starts proposing combinations
      with all 5 policies active

    Returns
    -------
    X : (n, n_policies) — policy intensity matrix; zeros for inactive policies
    Y : (n, 4)          — [ev_uptake, utility, emissions, net_cost]
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    policy_idx = {name: i for i, name in enumerate(bounds.names)}
    n_policies = bounds.n

    rows_X, rows_Y = [], []
    seen = set()

    for (p1_name, p2_name), items in raw.items():
        # Skip pairs where either policy is not in our bounds
        if p1_name not in policy_idx or p2_name not in policy_idx:
            continue
        i1, i2 = policy_idx[p1_name], policy_idx[p2_name]

        for item in items:
            x = np.zeros(n_policies)
            x[i1] = item["policy1_value"]
            x[i2] = item["policy2_value"]

            # Clip to bounds (some values may slightly exceed due to search)
            x = bounds.clip(x)

            # Deduplicate on rounded key (avoids double-counting (A,B)/(B,A) runs
            # that happened to land on the same intensity combination)
            key = tuple(np.round(x, 4))
            if key in seen:
                continue
            seen.add(key)

            y = np.array([
                item["mean_ev_uptake"],
                item["mean_utility_cumulative"],
                item["mean_emissions_cumulative"],
                item["mean_net_cost"],
            ])
            rows_X.append(x)
            rows_Y.append(y)

    X = np.array(rows_X)
    Y = np.array(rows_Y)

    n_feas = np.sum((Y[:, 0] >= 0.94) & (Y[:, 0] <= 0.96))
    print(f"Loaded {len(X)} pairwise warmstart points  "
          f"({n_feas} already feasible, {len(X) - n_feas} outside EV window)")
    return X, Y
