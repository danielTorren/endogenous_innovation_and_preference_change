"""
optimisation.py — Active Bayesian optimisation loop and Pareto front extraction.

OVERVIEW OF THE BO LOOP
------------------------
Each iteration:
  1. Fit the GP surrogate on all data collected so far
  2. Maximise an acquisition function to propose the next policy to evaluate
  3. Run the ABM at that policy combination
  4. Append the result and repeat

The acquisition function is:
  EI(x) × P(EV uptake ∈ [ev_lo, ev_hi] | x)

  EI = Expected Improvement on the scalarized objective (utility - λ*emissions - λ*cost)
       relative to the best feasible point seen so far.

  P(feasible) = probability the EV constraint is satisfied, from the GP's
                posterior over EV uptake.

Early iterations (no feasible point yet): the acquisition reduces to
  P(feasible) only — the loop first learns where the constraint is satisfied,
  then switches to optimising within it.

PARETO FRONT
------------
After the loop, we have ~170 evaluated policies (120 LHS + 50 BO). We:
  1. Filter for feasible points (EV uptake ∈ [ev_lo, ev_hi])
  2. Run a further 100 surrogate optimisations with random weight vectors
     to find points the BO may have missed
  3. Return the non-dominated set = the Pareto front

The Pareto front is a set of policies where no other policy is better on ALL
three objectives simultaneously. Choosing among them is a value judgement:
  high utility + high cost = generous subsidy regime
  low emissions + lower utility = strict carbon price regime
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from copy import deepcopy

from .sampling import PolicyBounds, run_policy_combination, OUTPUT_NAMES
from .surrogate import SurrogateGP

EV_LO_DEFAULT = 0.94
EV_HI_DEFAULT = 0.96


def load_optimisation_config(path: str) -> dict:
    """
    Load EV constraint bounds, fixed reference scales, and BO weights from JSON.

    Returns a dict with:
      ev_lo, ev_hi  — EV uptake constraint (fraction, e.g. 0.94 / 0.96)
      y_refs        — shape (4,) array [dummy, utility_ref, emissions_ref, net_cost_ref]
                      used to normalise objectives to comparable [0,1] scale
      weights       — shape (3,) array [w_utility, w_emissions, w_net_cost]

    WHY FIXED REFERENCES?
    Using the observed data range to normalise (the old approach) means the
    effective weight of each objective shifts every BO iteration as new points
    arrive.  A fixed reference computed once from the pairwise data gives
    stable, interpretable weights throughout the optimisation.
    """
    with open(path) as f:
        cfg = json.load(f)
    ev_lo = cfg["ev_constraint"]["ev_lo"]
    ev_hi = cfg["ev_constraint"]["ev_hi"]
    refs  = cfg["objective_reference_scales"]
    # Index 0 is ev_uptake — not in the objective, so set to 1.0 as a safe dummy
    y_refs = np.array([1.0, refs["utility"], refs["emissions"], refs["net_cost"]])
    w = cfg["bo_weights"]
    weights = np.array([w["utility"], w["emissions"], w["net_cost"]], dtype=float)
    return {"ev_lo": ev_lo, "ev_hi": ev_hi, "y_refs": y_refs, "weights": weights}


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------

def _scalarize(mu: np.ndarray, weights: np.ndarray, y_refs: np.ndarray) -> float:
    """
    Weighted scalarized objective (to be maximised).
    Normalises each output by a fixed reference scale so weights are stable and
    interpretable across all BO iterations.
      weights[0] × utility/ref - weights[1] × emissions/ref - weights[2] × cost/ref
    """
    return (weights[0] * mu[1] / y_refs[1]
            - weights[1] * mu[2] / y_refs[2]
            - weights[2] * mu[3] / y_refs[3])


def _acquisition(x: np.ndarray, surrogate: SurrogateGP,
                 y_best_scalar: float, weights: np.ndarray, y_refs: np.ndarray,
                 ev_lo: float, ev_hi: float) -> float:
    """
    Constrained Expected Improvement (to minimise — negated).

    EI × P(feasible)
    """
    mu, sigma = surrogate.predict(x[None])
    mu, sigma = mu[0], sigma[0]

    # EV feasibility probability
    p_feas = (norm.cdf(ev_hi, mu[0], sigma[0] + 1e-8)
              - norm.cdf(ev_lo, mu[0], sigma[0] + 1e-8))

    # Expected Improvement on scalarized objective
    mu_scalar = _scalarize(mu, weights, y_refs)
    sigma_scalar = float(np.sqrt(
        (weights[0] * sigma[1] / y_refs[1])**2
        + (weights[1] * sigma[2] / y_refs[2])**2
        + (weights[2] * sigma[3] / y_refs[3])**2
    )) + 1e-8

    if y_best_scalar is None:
        # No feasible point yet — just explore toward the feasible region
        return -p_feas

    z = (mu_scalar - y_best_scalar) / sigma_scalar
    ei = sigma_scalar * (z * norm.cdf(z) + norm.pdf(z))
    return -(ei * p_feas)


def _propose_next(surrogate: SurrogateGP, bounds: PolicyBounds,
                  Y_all: np.ndarray, weights: np.ndarray, y_refs: np.ndarray,
                  ev_lo: float, ev_hi: float, n_restarts: int = 20) -> np.ndarray:
    """
    Maximise the acquisition function via multi-start L-BFGS-B.
    Returns the proposed next policy vector.
    """
    feasible_mask = (Y_all[:, 0] >= ev_lo) & (Y_all[:, 0] <= ev_hi)
    if feasible_mask.sum() > 0:
        scalars = [_scalarize(Y_all[i], weights, y_refs)
                   for i in np.where(feasible_mask)[0]]
        y_best = max(scalars)
    else:
        y_best = None

    bounds_list = list(zip(bounds.lower, bounds.upper))
    best_val, best_x = np.inf, None

    rng = np.random.default_rng(seed=None)
    for _ in range(n_restarts):
        x0 = rng.uniform(bounds.lower, bounds.upper)
        result = minimize(
            _acquisition,
            x0,
            args=(surrogate, y_best, weights, y_refs, ev_lo, ev_hi),
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
    ev_lo: float = EV_LO_DEFAULT,
    ev_hi: float = EV_HI_DEFAULT,
    weights: np.ndarray = None,
    y_refs: np.ndarray = None,
    cache_path: str = None,
) -> tuple:
    """
    Active Bayesian optimisation loop.

    Sequentially proposes new policy combinations to evaluate, guided by the
    GP surrogate's uncertainty and the EV constraint.

    weights : [w_utility, w_emissions, w_cost] — relative importance of each
              objective. Loaded from optimisation_config.json via run.main();
              defaults to equal weighting if not supplied.
    y_refs  : shape (4,) fixed reference scales [dummy, utility, emissions, cost]
              used to normalise objectives before weighting. Load from
              optimisation_config.json via load_optimisation_config().
              Falls back to dynamic observed-data range if not supplied.

    Returns: (X_all, Y_all) — all evaluated points including the initial LHS.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached BO data from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["Y"]

    if weights is None:
        weights = np.array([1.0, 1.0, 1.0])
    weights = weights / weights.sum()

    X_all = X_init.copy()
    Y_all = Y_init.copy()

    for iteration in range(n_iterations):
        print(f"\n--- BO iteration {iteration + 1}/{n_iterations} "
              f"(dataset: {len(X_all)} pts) ---")

        # Fit surrogate on all data so far
        surrogate = SurrogateGP(bounds.lower, bounds.upper)
        surrogate.fit(X_all, Y_all)

        # Reference scales: fixed (preferred) or fall back to observed data range
        refs = y_refs if y_refs is not None else np.maximum(
            Y_all.max(axis=0) - Y_all.min(axis=0), 1e-8
        )

        # Propose next point
        x_next = _propose_next(surrogate, bounds, Y_all, weights, refs, ev_lo, ev_hi)
        policy_dict = dict(zip(bounds.names, x_next))
        print(f"  Proposed: { {k: round(v, 4) for k, v in policy_dict.items()} }")

        # Evaluate ABM
        y_next = run_policy_combination(base_params, policy_dict, controller_files)
        print(f"  Result: ev={y_next[0]:.3f}  utility={y_next[1]:.4g}  "
              f"emis={y_next[2]:.4g}  cost={y_next[3]:.4g}")

        X_all = np.vstack([X_all, x_next[None]])
        Y_all = np.vstack([Y_all, y_next[None]])

        # Progress summary
        n_feas = ((Y_all[:, 0] >= ev_lo) & (Y_all[:, 0] <= ev_hi)).sum()
        print(f"  Feasible: {n_feas}/{len(Y_all)}")

    if cache_path:
        np.savez(cache_path, X=X_all, Y=Y_all)
        print(f"\nBO data saved to {cache_path}")

    return X_all, Y_all


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def _is_non_dominated(obj: np.ndarray) -> np.ndarray:
    """
    obj: (n, k) where higher is better for all k objectives.
    Returns boolean mask of non-dominated (Pareto-optimal) rows.
    """
    n = len(obj)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Does j dominate i?
            if np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
                dominated[i] = True
                break
    return ~dominated


def _optimise_weight_vector(
    surrogate: SurrogateGP, bounds: PolicyBounds,
    weights: np.ndarray, y_refs: np.ndarray,
    ev_lo: float, ev_hi: float,
) -> np.ndarray:
    """
    Find the policy that maximises a given weight vector on the GP surrogate,
    subject to a soft penalty for EV constraint violation.
    """
    def neg_obj(x):
        mu, _ = surrogate.predict(x[None])
        mu = mu[0]
        obj = _scalarize(mu, weights, y_refs)
        ev_penalty = 1e4 * (
            max(0.0, ev_lo - mu[0])**2 + max(0.0, mu[0] - ev_hi)**2
        )
        return -(obj - ev_penalty)

    bounds_de = [(lo, hi) for lo, hi in zip(bounds.lower, bounds.upper)]
    result = differential_evolution(
        neg_obj, bounds_de, seed=42, maxiter=300, tol=1e-6,
        popsize=10, mutation=(0.5, 1.5), recombination=0.7,
    )
    return bounds.clip(result.x)


def compute_pareto_front(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    surrogate: SurrogateGP,
    bounds: PolicyBounds,
    ev_lo: float = EV_LO_DEFAULT,
    ev_hi: float = EV_HI_DEFAULT,
    y_refs: np.ndarray = None,
    n_weight_vectors: int = 100,
) -> tuple:
    """
    Approximate the Pareto front within the EV uptake constraint.

    Combines:
      (a) All observed feasible points from the LHS + BO data
      (b) GP-based optima found by optimising the surrogate with many
          different weight vectors (Dirichlet-sampled to cover the simplex)

    (b) finds policies that the LHS/BO may never have explicitly evaluated —
    it is a "free" dense search over the surrogate, using the ABM only for
    the final validation of the best point(s).

    y_refs : fixed reference scales from load_optimisation_config(); falls back
             to observed data range if not supplied.

    Returns: (X_pareto, Y_pareto) — surrogate-predicted values for Pareto points.
    """
    refs = y_refs if y_refs is not None else np.maximum(
        Y_all.max(axis=0) - Y_all.min(axis=0), 1e-8
    )

    # (a) Observed feasible points
    obs_feas = (Y_all[:, 0] >= ev_lo) & (Y_all[:, 0] <= ev_hi)
    X_cands = list(X_all[obs_feas])
    Y_cands = list(Y_all[obs_feas])

    if len(X_cands) == 0:
        print("WARNING: no observed feasible points. Widen EV constraint or run more BO iterations.")

    # (b) Surrogate optimisation with diverse weights
    print(f"\nSurrogate Pareto search ({n_weight_vectors} weight vectors)...")
    rng = np.random.default_rng(42)
    weight_vectors = rng.dirichlet(np.ones(3), size=n_weight_vectors)

    for i, w in enumerate(weight_vectors):
        print(f"  {i+1}/{n_weight_vectors}", end="\r")
        x_opt = _optimise_weight_vector(surrogate, bounds, w, refs, ev_lo, ev_hi)
        mu, _ = surrogate.predict(x_opt[None])
        X_cands.append(x_opt)
        Y_cands.append(mu[0])
    print()

    X_cands = np.array(X_cands)
    Y_cands = np.array(Y_cands)

    # Filter to feasible (GP-predicted EV in constraint band)
    feas = (Y_cands[:, 0] >= ev_lo) & (Y_cands[:, 0] <= ev_hi)
    if feas.sum() == 0:
        print("WARNING: no feasible candidates found. Try more weight vectors or wider EV constraint.")
        return np.empty((0, X_all.shape[1])), np.empty((0, 4))

    X_feas, Y_feas = X_cands[feas], Y_cands[feas]

    # Pareto filter — maximise [utility, -emissions, -net_cost]
    obj = np.stack([Y_feas[:, 1], -Y_feas[:, 2], -Y_feas[:, 3]], axis=1)
    pareto_mask = _is_non_dominated(obj)
    print(f"Pareto front: {pareto_mask.sum()} non-dominated policies "
          f"(from {len(X_feas)} feasible candidates)")

    return X_feas[pareto_mask], Y_feas[pareto_mask]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto_front(
    X_pareto: np.ndarray,
    Y_pareto: np.ndarray,
    ev_lo: float = EV_LO_DEFAULT,
    ev_hi: float = EV_HI_DEFAULT,
    save_dir: str = None,
):
    """
    Three pairwise scatter plots of the Pareto front objectives.

    Points on this front are the set of policies where you cannot improve
    one objective without worsening another. Use these plots to decide
    which trade-off matches your priorities, then run the full ABM at
    the chosen policy to get the final published result.
    """
    if len(X_pareto) == 0:
        print("No Pareto points to plot.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        (1, 2, "Cumulative Utility", "Cumulative Emissions"),
        (1, 3, "Cumulative Utility", "Net Policy Cost"),
        (2, 3, "Cumulative Emissions", "Net Policy Cost"),
    ]
    for ax, (xi, yi, xlabel, ylabel) in zip(axs, pairs):
        ax.scatter(Y_pareto[:, xi], Y_pareto[:, yi], c="steelblue",
                   s=60, zorder=3, edgecolors='navy', linewidths=0.5)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Pareto front\nEV uptake {ev_lo:.0%}–{ev_hi:.0%}", fontsize=11)

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/pareto_front.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_bo_convergence(Y_all: np.ndarray, n_init: int,
                        ev_lo: float = EV_LO_DEFAULT,
                        ev_hi: float = EV_HI_DEFAULT,
                        save_dir: str = None):
    """
    Shows how the number of feasible points and best objective value evolved
    over BO iterations. Use this to check whether more BO iterations are needed:
    if both curves have plateaued, the loop has converged.
    """
    n_total = len(Y_all)
    cumulative_feasible = np.cumsum(
        (Y_all[:, 0] >= ev_lo) & (Y_all[:, 0] <= ev_hi)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, n_total + 1), cumulative_feasible, color='steelblue')
    ax1.axvline(n_init, color='grey', linestyle='--', label='LHS → BO boundary')
    ax1.set_xlabel("Evaluation number")
    ax1.set_ylabel("Cumulative feasible points")
    ax1.set_title("Feasible point accumulation")
    ax1.legend()

    # Best utility among feasible points so far
    best_util = []
    best_so_far = -np.inf
    for i in range(n_total):
        if Y_all[i, 0] >= ev_lo and Y_all[i, 0] <= ev_hi:
            best_so_far = max(best_so_far, Y_all[i, 1])
        best_util.append(best_so_far if best_so_far > -np.inf else np.nan)

    ax2.plot(range(1, n_total + 1), best_util, color='darkorange')
    ax2.axvline(n_init, color='grey', linestyle='--', label='LHS → BO boundary')
    ax2.set_xlabel("Evaluation number")
    ax2.set_ylabel("Best feasible utility (cumulative)")
    ax2.set_title("BO convergence — utility")
    ax2.legend()

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/bo_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()
