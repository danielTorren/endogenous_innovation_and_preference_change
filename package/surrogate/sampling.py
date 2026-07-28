"""
sampling.py — LHS design generation and ABM evaluation interface.

The ABM is expensive, so we front-load a Latin Hypercube Sample (LHS) to
cover the 5D policy space efficiently, then refine via active BO in optimisation.py.

Output columns (Y) — all four ABSOLUTE (not BAU-relative):
  0: ev_uptake   — mean final EV share across seeds (fraction). Diagnostic
                   only; not used as a constraint or objective any more.
  1: log_utility — mean, across seeds, of a one-time post-simulation
                   calculation: shift every (timestep, individual) raw
                   utility value to be positive, take log(), sum everything.
                   See compute_log_utility_metric(). Concave in each
                   individual's utility, so it penalises inequality — a
                   policy that concentrates gains in a few people scores
                   worse than one that spreads the same total more evenly.
  2: emissions   — mean cumulative emissions across seeds (kg CO2)
  3: net_cost    — mean cumulative net policy cost across seeds (£)

The optimisation constraints (emissions <= X% of BAU, log_utility >= Y% of
BAU, net_cost >= 0) are evaluated against these absolute values in
optimisation.py, using a separately-computed BAU reference (see
compute_bau_baseline()) — Y itself is never BAU-relative here, so the GP
surrogate is trained on the same absolute units for every point.
"""

import json
import pickle
import numpy as np
from copy import deepcopy
from scipy.stats.qmc import LatinHypercube, scale
from joblib import Parallel, delayed, load
import multiprocessing
from package.resources.utility import get_num_workers

OUTPUT_NAMES = ["ev_uptake", "log_utility", "emissions", "net_cost"]

# Empirically probed worst-case raw (timestep, individual) utility across BAU
# + a spread of extreme single/all-policy-max scenarios: min ~= -457,798.
# 1,000,000 gives >2x safety margin. If compute_log_utility_metric() ever
# raises because this proves insufficient, raise this value — don't silently
# clip or floor the utility values themselves.
LOG_UTILITY_SHIFT = 1_000_000.0


def compute_log_utility_metric(step_utility_array: np.ndarray, shift: float = LOG_UTILITY_SHIFT) -> float:
    """
    One-time post-simulation calculation (NOT done inside the simulation loop,
    which only accumulates raw utility — see socialNetworkUsers.history_utility_individual_always).

    step_utility_array : shape (n_timesteps, num_individuals) — raw utility per
                         person per timestep for the whole policy period.

    Returns: float — sum over every (timestep, individual) value of log(utility + shift).
    """
    shifted = step_utility_array + shift
    min_shifted = shifted.min()
    if min_shifted <= 0:
        raise ValueError(
            f"LOG_UTILITY_SHIFT={shift:.6g} insufficient: shifted min={min_shifted:.6g} <= 0. "
            "This policy combination pushed raw utility lower than anything seen in the "
            "empirical probe — increase LOG_UTILITY_SHIFT in sampling.py rather than clipping."
        )
    return float(np.sum(np.log(shifted)))

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
    log_utility = compute_log_utility_metric(
        np.stack(data.social_network.history_utility_individual_always)
    )
    return (
        data.calc_EV_prop(),
        log_utility,
        data.social_network.emissions_cumulative,
        data.calc_net_policy_distortion(),
    )


def compute_bau_baseline(base_params: dict, controller_files: list) -> dict:
    """
    Run BAU (all policies off) once across all seeds, keeping PER-SEED values
    (not means). Used to derive the scalar BAU reference points (mean across
    seeds) that optimisation.py's constraints are evaluated against — e.g.
    emissions_bau_ref = compute_bau_baseline(...)["emissions"].mean().

    Returns dict of per-seed arrays: {"ev_uptake", "log_utility", "emissions",
    "net_cost"}, each shape (n_seeds,), aligned by index to controller_files.
    """
    params = deepcopy(base_params)
    params = _reset_policies(params)

    num_cores = get_num_workers()
    results = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(_single_seed_run)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )
    ev_arr, logutil_arr, emis_arr, cost_arr = (np.array(a) for a in zip(*results))
    return {
        "ev_uptake": ev_arr,
        "log_utility": logutil_arr,
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
            return {k: data[k] for k in ("ev_uptake", "log_utility", "emissions", "net_cost")}

    baseline = compute_bau_baseline(base_params, controller_files)

    if cache_path:
        np.savez(cache_path, **baseline)
        print(f"BAU baseline saved to {cache_path}")

    return baseline


def run_policy_combination(
    base_params: dict,
    policy_dict: dict,
    controller_files: list,
) -> np.ndarray:
    """
    Run one policy combination across all pre-saved controller seeds in parallel.

    policy_dict: {policy_name: intensity_value, ...}
                 Policies with intensity=0 are left inactive.

    Returns: shape (4,) array — [mean_ev_uptake, mean_log_utility, mean_emissions, mean_net_cost]
             All absolute — see module docstring for why BAU-relativity is
             handled downstream (in optimisation.py), not here.
    """
    params = deepcopy(base_params)
    params = _reset_policies(params)
    for name, intensity in policy_dict.items():
        if intensity > 0:
            params = _update_policy_intensity(params, name, intensity)

    num_cores = get_num_workers()
    results = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(_single_seed_run)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )

    ev_arr, logutil_arr, emis_arr, cost_arr = (np.array(a) for a in zip(*results))

    return np.array([
        np.mean(ev_arr),
        np.mean(logutil_arr),
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
    seed: int = 42,
    cache_path: str = None,
) -> tuple:
    """
    Generate an LHS design and evaluate each point with the ABM.

    If cache_path is given and the file exists, loads from cache instead of
    re-running — useful since each ABM evaluation takes ~seconds.

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
        Y[i] = run_policy_combination(base_params, policy_dict, controller_files)

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

    STALE: this pkl predates both the log_utility metric and the constraint-
    based objective — its "mean_utility_cumulative" is the old switchers-only
    utility_cumulative, not log_utility. Do not use until pairwise_outcomes.pkl
    is regenerated with the new metric.

    Returns
    -------
    X : (n, n_policies) — policy intensity matrix; zeros for inactive policies
    Y : (n, 4)          — [ev_uptake, utility, emissions, net_cost] (OLD units, see above)
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
