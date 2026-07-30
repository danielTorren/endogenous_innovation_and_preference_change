"""
run.py — Main surrogate optimisation workflow.

WHAT ACTUALLY RUNS AND WHEN
-----------------------------
The ABM has two phases. The surrogate only ever runs Phase 2:

  Phase 1 — Calibration (burn-in + historical, up to 2023)
    - Runs ONCE per surrogate session, or is reused from a previous run
    - Produces N_seeds saved controller .pkl files (the ABM state at end of 2023)
    - Compute: N_seeds × (burn_in + calibration) timesteps

  Phase 2 — Future period (policy period, 2023–2035, 144 months)
    - Runs for EVERY LHS point and every BO iteration
    - Loads the saved controller state; calibration is NOT re-run
    - Each evaluation = N_seeds parallel future-period runs
    - Compute per evaluation: N_seeds × 144 timesteps

  Total compute:
    Phase 1 (once):    N_seeds × (burn_in + calibration) timesteps
    Phase 2 (per pt):  N_seeds × 144 timesteps
    Phase 2 (total):   (N_LHS + N_BO) × N_seeds × 144 timesteps

  N_seeds is set by base_params["seed_repetitions"] — the surrogate uses
  whatever number of seeds you already used for calibration.

REUSING EXISTING CALIBRATION
------------------------------
If you already have controller files from a previous run (e.g. from
low_policy_intensity_gen.py), pass the folder path to main() via
`existing_calib_folder`. The calibration phase is then skipped entirely —
only Phase 2 runs are performed.

  Example (reusing from an existing experiment):
    X, Y = main(existing_calib_folder="results/endog_single_10_00_00__01_01_2026")

The folder must contain a Calibration_runs/ sub-directory with
controller_seed_*.pkl files, and a Data/base_params.pkl file.

FORWARD-LOOKING EXPECTATIONS (POLICY PERIOD ONLY)
----------------------------------------------------
main()'s forward_looking_expectations argument turns on forward-looking
agents for every Phase 2 evaluation (BAU baseline, every LHS point, every BO
iteration) — never for Phase 1 calibration, which always runs naive
regardless of this argument (see main()'s docstring for why: baking the flag
into base_params_surrogate.json instead would apply it to the 2001-2023
historical fit too and break it — package.command_and_control.gen's module
docstring documents hitting exactly that bug). This is the same
per-call-only pattern package.car_ban.gen and package.command_and_control.gen
use for their own forward_looking_expectations toggle, and the same
already-existing pattern in best_policies.run_final_abm().

CONSTRAINED SINGLE-OBJECTIVE SEARCH
-----------------------------------
This is NOT a multi-objective trade-off search — there's one true objective
(minimise net_cost) and two one-sided constraints, both relative to BAU
(all policies off), both cumulative over the whole policy period:

  emissions   <= emissions_frac * emissions_bau_ref    (default: <= 70% of BAU)
  log_utility >= utility_frac   * log_utility_bau_ref  (default: >= 95% of BAU)
  net_cost    >= cost_floor                            (default: 0 — a policy can't be net income)

Exceeding either bound (more decarbonisation, smaller-than-required utility
loss) is fine and unconstrained — only the stated direction is enforced.
log_utility is a one-time post-simulation calculation (shift raw per-person
utility positive, log(), sum across people and within each 12-month year,
then take the MINIMUM across years — see sampling.compute_log_utility_metric)
that penalises inequality across people (concentrating the same total
utility in fewer people scores worse) AND penalises a single brutal year
(a policy can't make up for one terrible year with several comfortable ones,
the way a straight cumulative sum would let it).

emissions_bau_ref / log_utility_bau_ref are scalars (mean across seeds of a
BAU run) computed once via sampling.compute_bau_baseline(), cached to
results_dir/Data/bau_baseline.npz. The fractions themselves come from
package/surrogate/constants/optimisation_config.json.

IMPROVING SURROGATE QUALITY
-----------------------------
  R² < 0.85 on any output  →  add more LHS points (increase N_LHS)
  Coverage << 0.95          →  increase seed_repetitions in base_params
                               (more seeds → better noise averaging)
  Large errors in one region →  narrow POLICY_BOUNDS to exclude implausible
                               combinations; more points land in the good region

CHOOSING AMONG RANKED FEASIBLE POLICIES
------------------------------------------
main() returns (X_ranked, Y_ranked) — ALL feasible policies observed (LHS +
BO), sorted by ascending net_cost, row 0 = the best (cheapest feasible)
policy found. This isn't a Pareto front (there's nothing to trade off
against once utility/emissions are constraints, not objectives) — it's just
useful to see how much net_cost changes across the next few cheapest
alternatives before committing to row 0.

Running the best policy through the real ABM (time series + plots) is done
by package/surrogate/best_policies.py, not by this module — see
run_final_abm() and main() there.
"""

import os

# Must be set before numpy/BLAS is imported: each of the seed_repetitions
# joblib workers otherwise spawns its own BLAS thread pool, oversubscribing
# the node's cores when running on a cluster. setdefault so an explicit
# value from the submission script (e.g. Slurm) still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import glob
import json
import numpy as np

from package.resources.utility import load_object, save_object
from package.analysis.endogenous_policy_intensity_single_gen import set_up_calibration_runs
from .sampling import (
    load_policy_bounds, evaluate_lhs, get_or_create_bau_baseline,
    load_pairwise_warmstart, OUTPUT_NAMES,
)
from .surrogate import SurrogateGP, validate, loo_cv, save_surrogate
from .optimisation import (
    active_bo_loop, find_best_policy, plot_bo_convergence,
    load_optimisation_config, _feasible_mask,
)

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

_CONSTANTS_DIR   = "package/surrogate/constants"
BASE_PARAMS_PATH = f"{_CONSTANTS_DIR}/base_params_surrogate.json"
BOUNDS_PATH      = f"{_CONSTANTS_DIR}/policy_bounds_surrogate.json"
PAIRWISE_PATH    = "package/surrogate/pair_wise_outcomes/pairwise_outcomes.pkl"
OPT_CONFIG_PATH  = f"{_CONSTANTS_DIR}/optimisation_config.json"
RESULTS_DIR      = "results/surrogate_optimisation"

# N_LHS/N_BO defaults live in optimisation_config.json's "search" section
# (loaded fresh inside main() via load_optimisation_config, same as the
# emissions/utility constraint fractions) rather than as constants here —
# main()'s n_lhs/n_bo parameters fall back to that config when left as None.


# ---------------------------------------------------------------------------
# Calibration management
# ---------------------------------------------------------------------------

def get_or_create_calibration(
    base_params_path: str,
    existing_calib_folder: str = None,
) -> tuple:
    """
    Returns (controller_files, base_params, calib_folder).

    If existing_calib_folder is given and contains controller_seed_*.pkl files,
    those are loaded directly — calibration (Phase 1) is NOT re-run.

    Otherwise calibration is run from scratch using base_params_path.
    The saved controllers are reusable for any future surrogate run.

    WHAT THIS RUNS:
      existing_calib_folder given  →  nothing (just glob + load base_params)
      no existing folder           →  N_seeds × full calibration ABM (once only)
    """
    if existing_calib_folder is not None:
        pattern = os.path.join(existing_calib_folder, "Calibration_runs", "controller_seed_*.pkl")
        controller_files = sorted(glob.glob(pattern))
        if not controller_files:
            raise FileNotFoundError(
                f"No controller_seed_*.pkl files found in {existing_calib_folder}/Calibration_runs/\n"
                "Either run without existing_calib_folder to create new ones, "
                "or check the folder path."
            )
        base_params = load_object(existing_calib_folder + "/Data", "base_params")
        print(f"Reusing {len(controller_files)} existing controller files from {existing_calib_folder}")
        print("  Phase 1 (calibration) skipped — only Phase 2 (future period) will run.")
        return controller_files, base_params, existing_calib_folder

    # Run calibration from scratch
    print("Running Phase 1 calibration (burn-in + historical period, once only)...")
    print("  This creates saved controller files reusable for all future surrogate runs.")
    with open(base_params_path) as f:
        base_params = json.load(f)
    controller_files, base_params, calib_folder = set_up_calibration_runs(
        base_params, "surrogate_calibration"
    )
    n = len(controller_files)
    print(f"Phase 1 done. {n} controllers saved to {calib_folder}/Calibration_runs/")
    print(f"  To skip this next time: pass existing_calib_folder='{calib_folder}'")
    return controller_files, base_params, calib_folder


def _resolve_calib_folder(results_dir: str, existing_calib_folder: str = None):
    """
    existing_calib_folder, if given, always wins. Otherwise, look for a
    calib_folder.pkl already saved under results_dir/Data from a previous
    call to main() with this same results_dir — so rerunning main() against
    the same results_dir (e.g. after an interrupted run) reuses Phase 1
    instead of silently recalibrating from scratch every time.
    """
    if existing_calib_folder is not None:
        return existing_calib_folder
    try:
        return load_object(f"{results_dir}/Data", "calib_folder")
    except FileNotFoundError:
        return None


def _resolve_forward_looking_expectations(results_dir: str, forward_looking_expectations: bool = None):
    """
    forward_looking_expectations, if given (True or False, not just truthy),
    always wins. Otherwise, look for the value main() persisted under
    results_dir/Data from whatever surrogate optimisation run produced this
    results_dir — same reuse pattern as _resolve_calib_folder(), so
    best_policies.py evaluates a policy under the SAME expectation mode it
    was optimised under without the caller having to pass it twice. Falls
    back to None (naive) if nothing was ever persisted (e.g. results_dir
    predates this option).
    """
    if forward_looking_expectations is not None:
        return forward_looking_expectations
    try:
        return load_object(f"{results_dir}/Data", "forward_looking_expectations")
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_dirs(results_dir: str):
    os.makedirs(f"{results_dir}/Data",  exist_ok=True)
    os.makedirs(f"{results_dir}/Plots", exist_ok=True)


def _train_test_split(X, Y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(10, int(test_frac * len(X)))
    return X[idx[n_test:]], Y[idx[n_test:]], X[idx[:n_test]], Y[idx[:n_test]]


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(
    base_params_path: str = BASE_PARAMS_PATH,
    bounds_path: str = BOUNDS_PATH,
    opt_config_path: str = OPT_CONFIG_PATH,
    existing_calib_folder: str = None,
    pairwise_path: str = None,
    n_lhs: int | None = None,
    n_bo: int | None = None,
    results_dir: str = RESULTS_DIR,
    forward_looking_expectations: bool = None,
) -> tuple:
    """
    Full surrogate optimisation workflow.

    Parameters
    ----------
    base_params_path       : JSON config (defaults to package/surrogate/constants/base_params_surrogate.json)
    bounds_path            : JSON policy bounds (defaults to package/surrogate/constants/policy_bounds_surrogate.json)
    opt_config_path        : JSON optimisation config — emissions/utility constraint fractions,
                             LHS/BO search budget (defaults to package/surrogate/constants/optimisation_config.json)
    existing_calib_folder  : path to a previous run's folder that has Calibration_runs/
                             controller_seed_*.pkl files — skips Phase 1 entirely.
                             None (default): Phase 1 always runs fresh — this is never
                             auto-detected from a prior call against the same results_dir,
                             so changing calibration settings between runs can't silently
                             pick up a stale calibration.
    pairwise_path          : path to pairwise_outcomes.pkl — STALE, do not use (see
                             sampling.load_pairwise_warmstart's docstring) until it's
                             regenerated with the log_utility metric.
    n_lhs                  : number of LHS evaluations (Phase 2 only, ~seconds each).
                             None (default) uses "search.n_lhs" from opt_config_path.
    n_bo                   : number of active BO iterations (Phase 2 only).
                             None (default) uses "search.n_bo" from opt_config_path.
    forward_looking_expectations : None (default) leaves base_params's own
                             setting untouched (naive — base_params_surrogate.json
                             does not set this key). Pass True to run every
                             Phase-2 (future/policy period) evaluation —
                             BAU baseline, every LHS point, every BO
                             iteration — with forward-looking agents.
                             Calibration (Phase 1, 2001-2023 historical fit)
                             is UNAFFECTED regardless of this value: it
                             always uses base_params_path/existing base_params
                             exactly as saved, since the model's calibrated
                             parameters assume naive expectations during that
                             period (see package.command_and_control.gen's
                             module docstring — baking the flag into the base
                             params file instead of passing it here broke EV
                             adoption during calibration the hard way).
                             Persisted to results_dir/Data so a later
                             package.surrogate.best_policies call against the
                             same results_dir picks up the same mode
                             automatically (see best_policies.run_top_policies).
                             CACHES ARE NOT MODE-AWARE: lhs_data.npz,
                             bo_data.npz and bau_baseline.npz under
                             results_dir/Data are keyed only by path — switch
                             results_dir (or delete those three files) when
                             changing this value for a results_dir that's
                             already been run, or you'll silently reuse
                             evaluations from the other expectation regime.

    Returns
    -------
    X_ranked, Y_ranked     : all feasible evaluated policies, sorted by ascending
                             net_cost (row 0 = cheapest feasible policy found)
    """
    _setup_dirs(results_dir)
    bounds = load_policy_bounds(bounds_path)
    cfg = load_optimisation_config(opt_config_path)
    emissions_frac = cfg["emissions_frac"]
    utility_frac = cfg["utility_frac"]
    cost_floor = cfg["cost_floor"]
    if n_lhs is None:
        n_lhs = int(cfg["n_lhs"])
    if n_bo is None:
        n_bo = int(cfg["n_bo"])

    print(f"Policy bounds loaded from {bounds_path}:")
    for name, (lo, hi) in bounds.bounds.items():
        print(f"  {name}: [{lo}, {hi}]")
    print(f"Constraints: emissions <= {emissions_frac:.0%} of BAU, "
          f"log_utility >= {utility_frac:.0%} of BAU, net_cost >= {cost_floor:.4g}")
    mode_str = "naive (base_params default)" if forward_looking_expectations is None else str(forward_looking_expectations)
    print(f"Phase 2 (future period) forward_looking_expectations: {mode_str} — calibration always runs naive")

    # ------------------------------------------------------------------
    # Step 1: Calibration — run once or reuse saved controllers
    # ------------------------------------------------------------------
    print("=== Step 1: Calibration ===")
    # existing_calib_folder is only used if the caller explicitly passed one —
    # no auto-detection from a prior call against this same results_dir, so
    # this always runs Phase 1 fresh unless you deliberately ask it not to.
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, existing_calib_folder
    )
    n_seeds = len(controller_files)
    print(f"  Seeds (parallel future-period runs per evaluation): {n_seeds}")
    print(f"  Estimated Phase 2 runs: ({n_lhs} LHS + {n_bo} BO) × {n_seeds} = "
          f"{(n_lhs + n_bo) * n_seeds} total future-period ABM runs")

    # Persisted so best_policies.py (run right after this) can reuse these
    # exact controllers without the calib_folder path having to be passed in
    # by hand — this is the one legitimate same-session reuse, not the
    # cross-invocation auto-detection removed above. Same for
    # forward_looking_expectations, so best_policies.py evaluates the
    # policies this call found under the SAME expectation mode they were
    # optimised under, without the caller having to remember to pass it twice.
    save_object(calib_folder, f"{results_dir}/Data", "calib_folder")
    save_object(forward_looking_expectations, f"{results_dir}/Data", "forward_looking_expectations")

    # BAU baseline (all policies off), per-seed — the mean across seeds gives
    # the scalar reference points the emissions/utility constraints are
    # evaluated against (see optimisation.py). Cached like everything else.
    bau_cache = f"{results_dir}/Data/bau_baseline.npz"
    bau_baseline = get_or_create_bau_baseline(
        base_params, controller_files, cache_path=bau_cache,
        forward_looking_expectations=forward_looking_expectations,
    )
    emissions_bau_ref = float(bau_baseline["emissions"].mean())
    log_utility_bau_ref = float(bau_baseline["log_utility"].mean())
    print(f"  BAU reference: log_utility={log_utility_bau_ref:.4g}  "
          f"emissions={emissions_bau_ref:.4g}  "
          f"net_cost={bau_baseline['net_cost'].mean():.4g}")
    print(f"  Constraint thresholds: emissions <= {emissions_frac * emissions_bau_ref:.4g}  "
          f"log_utility >= {utility_frac * log_utility_bau_ref:.4g}")

    # ------------------------------------------------------------------
    # Step 2: LHS sampling (Phase 2 only — future period per point)
    # ------------------------------------------------------------------
    print(f"\n=== Step 2: LHS Sampling ({n_lhs} points × {n_seeds} seeds) ===")
    lhs_cache = f"{results_dir}/Data/lhs_data.npz"
    X_lhs, Y_lhs = evaluate_lhs(
        bounds, n_lhs, base_params, controller_files,
        cache_path=lhs_cache,
        forward_looking_expectations=forward_looking_expectations,
    )
    n_init = len(X_lhs)
    print(f"LHS complete: {n_init} points")
    for j, name in enumerate(OUTPUT_NAMES):
        print(f"  {name}: [{Y_lhs[:, j].min():.4g},  {Y_lhs[:, j].max():.4g}]")

    # ------------------------------------------------------------------
    # Step 3: Surrogate validation
    #   3a: internal — LOO-CV or held-out LHS test set
    #   3b: external — pairwise holdout (if pairwise_path given)
    # ------------------------------------------------------------------
    print("\n=== Step 3: Surrogate Validation ===")
    use_loo = n_init < 60
    if use_loo:
        print("  3a: LOO-CV (dataset < 60 points)")
        surrogate_val = SurrogateGP(bounds.lower, bounds.upper)
        surrogate_val.fit(X_lhs, Y_lhs)
        val_metrics = loo_cv(
            X_lhs, Y_lhs, bounds.lower, bounds.upper,
            plot=True, save_dir=f"{results_dir}/Plots",
        )
    else:
        X_train, Y_train, X_test, Y_test = _train_test_split(X_lhs, Y_lhs)
        print(f"  3a: train/test split — train: {len(X_train)}, test: {len(X_test)}")
        surrogate_val = SurrogateGP(bounds.lower, bounds.upper)
        surrogate_val.fit(X_train, Y_train)
        val_metrics = validate(
            surrogate_val, X_test, Y_test,
            label="LHS test set",
            plot=True, save_dir=f"{results_dir}/Plots",
        )
    save_object(val_metrics, f"{results_dir}/Data", "validation_metrics")

    if pairwise_path is not None:
        print(f"\n  3b: Pairwise edge validation (held-out, not used for training)")
        X_pw, Y_pw = load_pairwise_warmstart(pairwise_path, bounds)
        surrogate_pw = SurrogateGP(bounds.lower, bounds.upper)
        surrogate_pw.fit(X_lhs, Y_lhs)
        pw_metrics = validate(
            surrogate_pw, X_pw, Y_pw,
            label="pairwise edges",
            plot=True, save_dir=f"{results_dir}/Plots",
        )
        save_object(pw_metrics, f"{results_dir}/Data", "validation_metrics_pairwise")

    poor = [k for k, v in val_metrics.items() if v["R2"] < 0.85]
    if poor:
        print(f"\n  *** WARNING: poor surrogate fit for: {poor} ***")
        print("  Consider increasing N_LHS before running BO.")
        print("  BO will still run — its uncertainty estimates help compensate —")
        print("  but the best-policy recommendation will be less reliable.\n")

    # ------------------------------------------------------------------
    # Step 4: Active BO loop (Phase 2 only, n_bo × n_seeds runs)
    # ------------------------------------------------------------------
    print(f"\n=== Step 4: Active BO ({n_bo} iterations × {n_seeds} seeds) ===")
    bo_cache = f"{results_dir}/Data/bo_data.npz"
    X_all, Y_all = active_bo_loop(
        X_lhs, Y_lhs,
        bounds=bounds,
        base_params=base_params,
        controller_files=controller_files,
        n_iterations=n_bo,
        emissions_bau_ref=emissions_bau_ref,
        log_utility_bau_ref=log_utility_bau_ref,
        emissions_frac=emissions_frac,
        utility_frac=utility_frac,
        cost_floor=cost_floor,
        cache_path=bo_cache,
        forward_looking_expectations=forward_looking_expectations,
    )
    n_feas = _feasible_mask(Y_all, emissions_bau_ref, log_utility_bau_ref,
                             emissions_frac, utility_frac, cost_floor).sum()
    print(f"\nTotal evaluations: {len(X_all)}   Feasible: {n_feas}")
    plot_bo_convergence(
        Y_all, n_init=n_init,
        emissions_bau_ref=emissions_bau_ref, log_utility_bau_ref=log_utility_bau_ref,
        emissions_frac=emissions_frac, utility_frac=utility_frac, cost_floor=cost_floor,
        save_dir=f"{results_dir}/Plots",
    )

    # ------------------------------------------------------------------
    # Step 5: Final surrogate + best policy
    # ------------------------------------------------------------------
    print("\n=== Step 5: Final Surrogate + Best Policy ===")
    surrogate_final = SurrogateGP(bounds.lower, bounds.upper)
    surrogate_final.fit(X_all, Y_all)
    save_surrogate(surrogate_final, f"{results_dir}/Data/surrogate.pkl")

    best_x, best_y, source = find_best_policy(
        X_all, Y_all, surrogate_final, bounds,
        emissions_bau_ref=emissions_bau_ref, log_utility_bau_ref=log_utility_bau_ref,
        emissions_frac=emissions_frac, utility_frac=utility_frac, cost_floor=cost_floor,
    )

    # All feasible observed policies, ranked by ascending net_cost (row 0 =
    # cheapest). If the surrogate-based refinement found something better
    # than every observed point, it's included too. Saved as pareto.npz for
    # backward compatibility with best_policies.py's array-based loading —
    # it's no longer a Pareto front (single objective now), just a ranking.
    feas_mask = _feasible_mask(Y_all, emissions_bau_ref, log_utility_bau_ref,
                                emissions_frac, utility_frac, cost_floor)
    X_ranked, Y_ranked = X_all[feas_mask].copy(), Y_all[feas_mask].copy()
    if best_x is not None and source == "surrogate-refined":
        X_ranked = np.vstack([best_x[None], X_ranked])
        Y_ranked = np.vstack([best_y[None], Y_ranked])
    if len(X_ranked) > 0:
        order = np.argsort(Y_ranked[:, 2])
        X_ranked, Y_ranked = X_ranked[order], Y_ranked[order]

    np.savez(f"{results_dir}/Data/pareto.npz", X=X_ranked, Y=Y_ranked)

    # ------------------------------------------------------------------
    # Step 6: Best policy summary
    # ------------------------------------------------------------------
    print("\n=== Step 6: Best Policy ===")
    if len(X_ranked) == 0:
        print("No feasible policy found. Run more BO iterations or loosen the "
              "constraints in optimisation_config.json.")
        return None, None

    header = f"{'#':<4} {'LogUtil':>14} {'Emissions':>14} {'NetCost':>14}  " + \
             "  ".join(f"{n[:10]:>12}" for n in bounds.names)
    print(header)
    print("-" * len(header))
    for rank in range(len(X_ranked)):
        x, y = X_ranked[rank], Y_ranked[rank]
        policy_str = "  ".join(f"{v:>12.2f}" for v in x)
        print(f"{rank:<4} {y[0]:>14.4g} {y[1]:>14.4g} {y[2]:>14.4g}  {policy_str}")

    print(f"\nBest policy ({source}): net_cost={Y_ranked[0, 2]:.4g}")
    print(f"Saved to {results_dir}/Data/pareto.npz")
    print("Next: run `python -m package.surrogate.best_policies` to run the top "
          "policies through the real ABM and produce time-series + trade-off plots.")

    return X_ranked, Y_ranked


if __name__ == "__main__":
    # Example: reuse controllers from an existing experiment
    # X_ranked, Y_ranked = main(
    #     existing_calib_folder="results/endog_single_10_00_00__01_01_2026"
    # )

    X_ranked, Y_ranked = main()
