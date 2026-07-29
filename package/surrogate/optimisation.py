"""
optimisation.py — Active Bayesian optimisation loop and best-policy search.

OVERVIEW OF THE BO LOOP
------------------------
Each iteration:
  1. Fit the GP surrogate on all data collected so far
  2. Maximise an acquisition function to propose the next policy to evaluate
  3. Run the ABM at that policy combination
  4. Append the result and repeat

This is a CONSTRAINED SINGLE-OBJECTIVE search, not a multi-objective trade-off:
  objective:   minimise net_cost
  constraints: emissions   <= emissions_frac  * emissions_bau_ref   (e.g. <= 70% of BAU)
               log_utility >= utility_frac    * log_utility_bau_ref (e.g. >= 95% of BAU)
               net_cost    >= cost_floor                            (default 0: policy can't
                                                                      be net income — see
                                                                      constants/optimisation_config.json)

emissions_bau_ref / log_utility_bau_ref are scalar references (mean across
seeds of the BAU run) computed once via sampling.compute_bau_baseline() —
see run.py. The EV-uptake band that used to be a hard constraint is gone;
ev_uptake is still tracked in Y as a diagnostic, nothing more.

The acquisition function is:
  EI(x) × P(emissions feasible | x) × P(log_utility feasible | x) × P(net_cost >= cost_floor | x)

  EI = Expected Improvement for MINIMISING net_cost, relative to the lowest
       net_cost seen among currently-feasible points.

Early iterations (no feasible point yet): the acquisition reduces to the
feasibility probability alone — the loop first learns where the constraints
are satisfied, then switches to minimising cost within that region.

BEST POLICY SEARCH
--------------------
After the loop, find_best_policy() returns the SINGLE lowest-cost feasible
policy — not a Pareto front. There's only one true objective (cost) here;
the old Pareto-front / diverse-weight-vector machinery doesn't apply once
utility and emissions are constraints rather than competing objectives.
It combines:
  (a) the best observed feasible point from the LHS + BO data
  (b) one surrogate-based local refinement (constrained differential
      evolution on the fitted GP) that may find a better point the
      LHS/BO search didn't happen to land on
and returns whichever of the two is cheaper (and actually feasible).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from .sampling import PolicyBounds, run_policy_combination

DEFAULT_EMISSIONS_FRAC = 0.70   # emissions <= 70% of BAU (>= 30% reduction)
DEFAULT_UTILITY_FRAC   = 0.95   # log_utility >= 95% of BAU (<= 5% decrease)
DEFAULT_COST_FLOOR     = 0.0    # net_cost >= 0 (a policy can't be net income)


def load_optimisation_config(path: str) -> dict:
    """
    Load the constraint fractions/floor and LHS/BO search budget from JSON.

    Returns a dict with:
      emissions_frac — emissions constraint: policy emissions <= emissions_frac * BAU emissions
      utility_frac   — utility constraint: policy log_utility >= utility_frac * BAU log_utility
      cost_floor     — net_cost constraint: policy net_cost >= cost_floor (default 0: can't be net income)
      n_lhs          — number of LHS evaluations (Phase 2 only)
      n_bo           — number of active BO iterations (Phase 2 only)
    """
    with open(path) as f:
        cfg = json.load(f)
    c = cfg["constraints"]
    s = cfg["search"]
    return {
        "emissions_frac": c["emissions_max_fraction_of_bau"],
        "utility_frac": c["utility_min_fraction_of_bau"],
        "cost_floor": c.get("net_cost_floor", DEFAULT_COST_FLOOR),
        "n_lhs": s["n_lhs"],
        "n_bo": s["n_bo"],
    }


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------

def _feasibility_prob(
    mu: np.ndarray, sigma: np.ndarray,
    emissions_bau_ref: float, log_utility_bau_ref: float,
    emissions_frac: float, utility_frac: float,
    cost_floor: float = DEFAULT_COST_FLOOR,
) -> float:
    """
    P(emissions constraint satisfied) x P(log_utility constraint satisfied)
    x P(net_cost >= cost_floor), each under the GP's Gaussian posterior for
    that output. mu/sigma indices: 0=ev_uptake, 1=log_utility, 2=emissions, 3=net_cost.
    """
    emissions_threshold = emissions_frac * emissions_bau_ref
    p_emissions = norm.cdf(emissions_threshold, mu[2], sigma[2] + 1e-8)

    utility_threshold = utility_frac * log_utility_bau_ref
    p_utility = 1.0 - norm.cdf(utility_threshold, mu[1], sigma[1] + 1e-8)

    p_cost_floor = 1.0 - norm.cdf(cost_floor, mu[3], sigma[3] + 1e-8)

    return p_emissions * p_utility * p_cost_floor


def _is_feasible(y: np.ndarray, emissions_bau_ref: float, log_utility_bau_ref: float,
                  emissions_frac: float, utility_frac: float,
                  cost_floor: float = DEFAULT_COST_FLOOR) -> bool:
    """Same three constraints, evaluated on an observed (not predicted) Y row."""
    return (
        y[2] <= emissions_frac * emissions_bau_ref
        and y[1] >= utility_frac * log_utility_bau_ref
        and y[3] >= cost_floor
    )


def _feasible_mask(Y: np.ndarray, emissions_bau_ref: float, log_utility_bau_ref: float,
                    emissions_frac: float, utility_frac: float,
                    cost_floor: float = DEFAULT_COST_FLOOR) -> np.ndarray:
    return (
        (Y[:, 2] <= emissions_frac * emissions_bau_ref)
        & (Y[:, 1] >= utility_frac * log_utility_bau_ref)
        & (Y[:, 3] >= cost_floor)
    )


def _acquisition(
    x: np.ndarray, surrogate,
    y_best_cost: float,
    emissions_bau_ref: float, log_utility_bau_ref: float,
    emissions_frac: float, utility_frac: float,
    cost_floor: float = DEFAULT_COST_FLOOR,
) -> float:
    """
    Constrained Expected Improvement for MINIMISING net_cost (to minimise — negated).

    EI(-net_cost) × P(feasible)
    """
    mu, sigma = surrogate.predict(x[None])
    mu, sigma = mu[0], sigma[0]

    p_feas = _feasibility_prob(mu, sigma, emissions_bau_ref, log_utility_bau_ref,
                                emissions_frac, utility_frac, cost_floor)

    if y_best_cost is None:
        # No feasible point yet — just explore toward the feasible region
        return -p_feas

    # EI for minimisation: improvement = y_best_cost - predicted cost
    sigma_cost = sigma[3] + 1e-8
    z = (y_best_cost - mu[3]) / sigma_cost
    ei = sigma_cost * (z * norm.cdf(z) + norm.pdf(z))
    return -(ei * p_feas)


def _propose_next(
    surrogate, bounds: PolicyBounds, Y_all: np.ndarray,
    emissions_bau_ref: float, log_utility_bau_ref: float,
    emissions_frac: float, utility_frac: float,
    n_restarts: int = 20,
    cost_floor: float = DEFAULT_COST_FLOOR,
) -> np.ndarray:
    """
    Maximise the acquisition function via multi-start L-BFGS-B.
    Returns the proposed next policy vector.
    """
    feasible_mask = _feasible_mask(Y_all, emissions_bau_ref, log_utility_bau_ref,
                                    emissions_frac, utility_frac, cost_floor)
    y_best_cost = Y_all[feasible_mask, 3].min() if feasible_mask.sum() > 0 else None

    bounds_list = list(zip(bounds.lower, bounds.upper))
    best_val, best_x = np.inf, None

    rng = np.random.default_rng(seed=None)
    for _ in range(n_restarts):
        x0 = rng.uniform(bounds.lower, bounds.upper)
        result = minimize(
            _acquisition,
            x0,
            args=(surrogate, y_best_cost, emissions_bau_ref, log_utility_bau_ref,
                  emissions_frac, utility_frac, cost_floor),
            bounds=bounds_list,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if result.fun < best_val:
            best_val, best_x = result.fun, result.x

    return bounds.clip(best_x)


# ---------------------------------------------------------------------------
# Active BO loop
# ---------------------------------------------------------------------------

def active_bo_loop(
    X_init: np.ndarray,
    Y_init: np.ndarray,
    bounds: PolicyBounds,
    base_params: dict,
    controller_files: list,
    n_iterations: int = 50,
    emissions_bau_ref: float = None,
    log_utility_bau_ref: float = None,
    emissions_frac: float = DEFAULT_EMISSIONS_FRAC,
    utility_frac: float = DEFAULT_UTILITY_FRAC,
    cost_floor: float = DEFAULT_COST_FLOOR,
    cache_path: str = None,
) -> tuple:
    """
    Active Bayesian optimisation loop — constrained search for minimum-cost
    feasible policy (see module docstring).

    emissions_bau_ref, log_utility_bau_ref : scalar BAU references (mean across
        seeds) from sampling.compute_bau_baseline() — required.

    Returns: (X_all, Y_all) — all evaluated points including the initial LHS.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached BO data from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["Y"]

    if emissions_bau_ref is None or log_utility_bau_ref is None:
        raise ValueError("active_bo_loop requires emissions_bau_ref and log_utility_bau_ref")

    X_all = X_init.copy()
    Y_all = Y_init.copy()

    for iteration in range(n_iterations):
        print(f"\n--- BO iteration {iteration + 1}/{n_iterations} "
              f"(dataset: {len(X_all)} pts) ---")

        # Fit surrogate on all data so far
        surrogate = _fit_surrogate(bounds, X_all, Y_all)

        # Propose next point
        x_next = _propose_next(surrogate, bounds, Y_all, emissions_bau_ref, log_utility_bau_ref,
                                emissions_frac, utility_frac, cost_floor=cost_floor)
        policy_dict = dict(zip(bounds.names, x_next))
        print(f"  Proposed: { {k: round(v, 4) for k, v in policy_dict.items()} }")

        # Evaluate ABM
        y_next = run_policy_combination(base_params, policy_dict, controller_files)
        print(f"  Result: ev={y_next[0]:.3f}  log_utility={y_next[1]:.4g}  "
              f"emis={y_next[2]:.4g}  cost={y_next[3]:.4g}")

        X_all = np.vstack([X_all, x_next[None]])
        Y_all = np.vstack([Y_all, y_next[None]])

        # Progress summary
        feas_mask = _feasible_mask(Y_all, emissions_bau_ref, log_utility_bau_ref,
                                    emissions_frac, utility_frac, cost_floor)
        n_feas = feas_mask.sum()
        best_cost = Y_all[feas_mask, 3].min() if n_feas > 0 else np.nan
        print(f"  Feasible: {n_feas}/{len(Y_all)}   Best feasible net_cost so far: {best_cost:.4g}")

    if cache_path:
        np.savez(cache_path, X=X_all, Y=Y_all)
        print(f"\nBO data saved to {cache_path}")

    return X_all, Y_all


def _fit_surrogate(bounds: PolicyBounds, X: np.ndarray, Y: np.ndarray):
    from .surrogate import SurrogateGP
    surrogate = SurrogateGP(bounds.lower, bounds.upper)
    surrogate.fit(X, Y)
    return surrogate


# ---------------------------------------------------------------------------
# Best policy search (replaces the old Pareto front)
# ---------------------------------------------------------------------------

def find_best_policy(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    surrogate,
    bounds: PolicyBounds,
    emissions_bau_ref: float,
    log_utility_bau_ref: float,
    emissions_frac: float = DEFAULT_EMISSIONS_FRAC,
    utility_frac: float = DEFAULT_UTILITY_FRAC,
    cost_floor: float = DEFAULT_COST_FLOOR,
) -> tuple:
    """
    Find the single minimum-net_cost policy satisfying:
      emissions   <= emissions_frac * emissions_bau_ref
      log_utility >= utility_frac   * log_utility_bau_ref
      net_cost    >= cost_floor

    Combines the best observed feasible point with one surrogate-based local
    refinement (constrained differential evolution on the fitted GP) — the
    refinement can find a policy the LHS/BO search never happened to land on,
    at no extra ABM cost, but is only trusted if the surrogate itself predicts
    it to be feasible.

    Returns: (best_x, best_y, source) — source is "observed" or "surrogate-refined".
             (None, None, None) if nothing feasible was found either way.
    """
    obs_feasible = _feasible_mask(Y_all, emissions_bau_ref, log_utility_bau_ref,
                                   emissions_frac, utility_frac, cost_floor)
    candidates = []

    if obs_feasible.sum() > 0:
        idx = np.argmin(Y_all[obs_feasible, 3])
        candidates.append((X_all[obs_feasible][idx], Y_all[obs_feasible][idx], "observed"))
    else:
        print("WARNING: no observed feasible point among LHS + BO evaluations.")

    # Surrogate-based refinement: minimise predicted net_cost subject to a
    # large penalty (normalised by the BAU reference scale, so it's
    # comparable across the very different emissions/utility/cost units)
    # for violating either constraint.
    def neg_obj(x):
        mu, _ = surrogate.predict(x[None])
        mu = mu[0]
        emissions_viol = max(0.0, mu[2] - emissions_frac * emissions_bau_ref) / abs(emissions_bau_ref)
        utility_viol   = max(0.0, utility_frac * log_utility_bau_ref - mu[1]) / abs(log_utility_bau_ref)
        cost_viol      = max(0.0, cost_floor - mu[3])
        penalty = 1e9 * (emissions_viol**2 + utility_viol**2) + 10 * cost_viol
        return mu[3] + penalty

    bounds_de = list(zip(bounds.lower, bounds.upper))
    result = differential_evolution(
        neg_obj, bounds_de, seed=42, maxiter=300, tol=1e-8,
        popsize=15, mutation=(0.5, 1.5), recombination=0.7,
    )
    x_refined = bounds.clip(result.x)
    mu_refined, _ = surrogate.predict(x_refined[None])
    mu_refined = mu_refined[0]

    if _is_feasible(mu_refined, emissions_bau_ref, log_utility_bau_ref, emissions_frac, utility_frac, cost_floor):
        candidates.append((x_refined, mu_refined, "surrogate-refined"))
    else:
        print("Surrogate-refined candidate not feasible under the GP's own prediction — discarded.")

    if not candidates:
        print("WARNING: no feasible candidate found (neither observed nor surrogate-refined). "
              "Run more BO iterations or loosen the constraints.")
        return None, None, None

    candidates.sort(key=lambda c: c[1][3])  # ascending net_cost
    best_x, best_y, source = candidates[0]
    print(f"Best policy found ({source}): net_cost={best_y[3]:.4g}  "
          f"log_utility={best_y[1]:.4g} (need >= {utility_frac * log_utility_bau_ref:.4g})  "
          f"emissions={best_y[2]:.4g} (need <= {emissions_frac * emissions_bau_ref:.4g})  "
          f"ev_uptake={best_y[0]:.3f}")
    return best_x, best_y, source


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bo_convergence(
    Y_all: np.ndarray, n_init: int,
    emissions_bau_ref: float, log_utility_bau_ref: float,
    emissions_frac: float = DEFAULT_EMISSIONS_FRAC,
    utility_frac: float = DEFAULT_UTILITY_FRAC,
    cost_floor: float = DEFAULT_COST_FLOOR,
    save_dir: str = None,
):
    """
    Shows how the number of feasible points and the best (lowest) feasible
    net_cost evolved over BO iterations. Use this to check whether more BO
    iterations are needed: if both curves have plateaued, the loop has converged.
    """
    n_total = len(Y_all)
    feas_mask_all = _feasible_mask(Y_all, emissions_bau_ref, log_utility_bau_ref,
                                    emissions_frac, utility_frac, cost_floor)
    cumulative_feasible = np.cumsum(feas_mask_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, n_total + 1), cumulative_feasible, color='steelblue')
    ax1.axvline(n_init, color='grey', linestyle='--', label='LHS → BO boundary')
    ax1.set_xlabel("Evaluation number")
    ax1.set_ylabel("Cumulative feasible points")
    ax1.set_title("Feasible point accumulation")
    ax1.legend()

    # Best (lowest) feasible net_cost among points seen so far
    best_cost = []
    best_so_far = np.inf
    for i in range(n_total):
        if feas_mask_all[i]:
            best_so_far = min(best_so_far, Y_all[i, 3])
        best_cost.append(best_so_far if best_so_far < np.inf else np.nan)

    ax2.plot(range(1, n_total + 1), best_cost, color='darkorange')
    ax2.axvline(n_init, color='grey', linestyle='--', label='LHS → BO boundary')
    ax2.set_xlabel("Evaluation number")
    ax2.set_ylabel("Best feasible net_cost so far")
    ax2.set_title("BO convergence — cost (lower is better)")
    ax2.legend()

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/bo_convergence.png", dpi=150, bbox_inches='tight')
    try:
        # Raises on headless cluster nodes with no display backend; the
        # figure is already saved above, so it's safe to skip silently.
        plt.show()
    except Exception:
        pass
