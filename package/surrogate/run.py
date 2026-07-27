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

BAU-RELATIVE OBJECTIVES
-----------------------------
utility, emissions, and net_cost (surrogate output columns 1-3) are deltas
relative to BAU (all policies off), not absolute values — "how much better/
worse than doing nothing". Computed once per seed set (see
sampling.compute_bau_baseline) and subtracted per-seed (paired, common
random numbers) from every LHS/BO evaluation before averaging. ev_uptake
(column 0) stays absolute — it's the constraint target (94-96% EV share),
not an objective. Reference scales for weighting these deltas in the BO
acquisition/Pareto search are recomputed from each run's own LHS data
(Step 2), not read from optimisation_config.json — those static numbers
predate this change and are no longer used.

IMPROVING SURROGATE QUALITY
-----------------------------
  R² < 0.85 on any output  →  add more LHS points (increase N_LHS)
  Coverage << 0.95          →  increase seed_repetitions in base_params
                               (more seeds → better noise averaging)
  Large errors in one region →  narrow POLICY_BOUNDS to exclude implausible
                               combinations; more points land in the good region

HOW TO CHOOSE AMONG PARETO POINTS
------------------------------------
  np.argsort(Y_pareto[:, 1])[::-1]   # highest utility first
  np.argsort(Y_pareto[:, 2])         # lowest emissions first
  np.argsort(Y_pareto[:, 3])         # lowest net cost first
Running the top policies through the real ABM (time series + plots) is
done by package/surrogate/best_policies.py, not by this module — see
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
    load_pairwise_warmstart, run_policy_combination, OUTPUT_NAMES,
)
from .surrogate import SurrogateGP, validate, loo_cv, save_surrogate, load_surrogate
from .optimisation import (
    active_bo_loop, compute_pareto_front,
    plot_pareto_front, plot_bo_convergence,
    load_optimisation_config,
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

N_LHS = 100  # LHS evaluations (80% train / 20% test for validation)
N_BO  = 50   # active BO iterations after LHS

# Loaded from JSON at import time so the rest of the module can reference them
POLICY_BOUNDS = load_policy_bounds(BOUNDS_PATH)
OPT_CONFIG    = load_optimisation_config(OPT_CONFIG_PATH)


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
    n_lhs: int = N_LHS,
    n_bo: int = N_BO,
    results_dir: str = RESULTS_DIR,
) -> tuple:
    """
    Full surrogate optimisation workflow.

    Parameters
    ----------
    base_params_path       : JSON config (defaults to package/surrogate/constants/base_params_surrogate.json)
    bounds_path            : JSON policy bounds (defaults to package/surrogate/constants/policy_bounds_surrogate.json)
    opt_config_path        : JSON optimisation config — EV constraint, reference scales, weights
                             (defaults to package/surrogate/constants/optimisation_config.json)
    existing_calib_folder  : path to a previous run's folder that has Calibration_runs/
                             controller_seed_*.pkl files — skips Phase 1 entirely
    pairwise_path          : path to pairwise_outcomes.pkl — if given, these ~100 real
                             ABM evaluations are used as a held-out validation set after
                             the surrogate is fitted on the LHS data.  They are NOT
                             added to the training data (avoids biasing the GP toward
                             high-intensity 2-policy edge combinations).
                             WARNING: pairwise_outcomes.pkl stores ABSOLUTE utility/
                             emissions/net_cost, but LHS/BO now train on BAU-relative
                             deltas — do not pass this until pairwise_outcomes.pkl is
                             regenerated in the same (delta) units, or the validation
                             R²/coverage numbers will be meaningless.
    n_lhs                  : number of LHS evaluations (Phase 2 only, ~seconds each)
    n_bo                   : number of active BO iterations (Phase 2 only)

    Returns
    -------
    X_pareto, Y_pareto     : policy vectors and predicted outcomes on Pareto front
    """
    _setup_dirs(results_dir)
    bounds = load_policy_bounds(bounds_path)
    cfg    = load_optimisation_config(opt_config_path)
    ev_lo, ev_hi = cfg["ev_lo"], cfg["ev_hi"]
    weights = cfg["weights"]

    print(f"Policy bounds loaded from {bounds_path}:")
    for name, (lo, hi) in bounds.bounds.items():
        print(f"  {name}: [{lo}, {hi}]")
    print(f"EV constraint: [{ev_lo:.0%}, {ev_hi:.0%}]")
    print(f"BO weights: utility={weights[0]:.2f}  emissions={weights[1]:.2f}  "
          f"net_cost={weights[2]:.2f}")
    print("Objective reference scales: recomputed from this run's own LHS data "
          "(BAU-relative deltas) after Step 2 — optimisation_config.json's "
          "objective_reference_scales is no longer used for this.")

    # ------------------------------------------------------------------
    # Step 1: Calibration — run once or reuse saved controllers
    # ------------------------------------------------------------------
    print("=== Step 1: Calibration ===")
    existing_calib_folder = _resolve_calib_folder(results_dir, existing_calib_folder)
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, existing_calib_folder
    )
    n_seeds = len(controller_files)
    print(f"  Seeds (parallel future-period runs per evaluation): {n_seeds}")
    print(f"  Estimated Phase 2 runs: ({n_lhs} LHS + {n_bo} BO) × {n_seeds} = "
          f"{(n_lhs + n_bo) * n_seeds} total future-period ABM runs")

    # Persisted so best_policies.py can reuse these controllers without the
    # calib_folder path having to be passed in by hand.
    save_object(calib_folder, f"{results_dir}/Data", "calib_folder")

    # BAU baseline (all policies off), per-seed — used to turn utility/emissions/
    # net_cost into BAU-relative deltas throughout LHS and BO (paired per seed,
    # not just one overall BAU mean, so seed-specific noise cancels out).
    bau_cache = f"{results_dir}/Data/bau_baseline.npz"
    bau_baseline = get_or_create_bau_baseline(base_params, controller_files, cache_path=bau_cache)
    print(f"  BAU baseline: utility={bau_baseline['utility'].mean():.4g}  "
          f"emissions={bau_baseline['emissions'].mean():.4g}  "
          f"net_cost={bau_baseline['net_cost'].mean():.4g}")

    # ------------------------------------------------------------------
    # Step 2: LHS sampling (Phase 2 only — future period per point)
    # ------------------------------------------------------------------
    print(f"\n=== Step 2: LHS Sampling ({n_lhs} points × {n_seeds} seeds) ===")
    lhs_cache = f"{results_dir}/Data/lhs_data.npz"
    X_lhs, Y_lhs = evaluate_lhs(
        bounds, n_lhs, base_params, controller_files,
        bau_baseline=bau_baseline,
        cache_path=lhs_cache,
    )
    n_init = len(X_lhs)
    print(f"LHS complete: {n_init} points")
    for j, name in enumerate(OUTPUT_NAMES):
        print(f"  {name}: [{Y_lhs[:, j].min():.4g},  {Y_lhs[:, j].max():.4g}]")

    # Reference scales recomputed from THIS run's actual BAU-relative LHS data,
    # rather than trusting the static values in optimisation_config.json —
    # those were calibrated on an older, absolute-units dataset and go stale
    # the moment the objective definition, bounds, or seed count changes.
    y_refs = np.array([
        1.0,
        max(np.abs(Y_lhs[:, 1]).max(), 1e-8),
        max(np.abs(Y_lhs[:, 2]).max(), 1e-8),
        max(np.abs(Y_lhs[:, 3]).max(), 1e-8),
    ])
    print(f"Objective reference scales (BAU-relative, from LHS): "
          f"utility={y_refs[1]:.3g}  emissions={y_refs[2]:.3g}  net_cost={y_refs[3]:.3g}")

    # ------------------------------------------------------------------
    # Step 3: Surrogate validation
    #   3a: internal — LOO-CV or held-out LHS test set
    #   3b: external — pairwise holdout (if pairwise_path given)
    #       The pairwise data lives on 2-policy edges of the 5D space, so
    #       this checks whether the GP extrapolates correctly to the boundary
    #       — a meaningful sanity check that costs no new ABM runs.
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
        # Fit on the full LHS for this check (not the train split)
        surrogate_pw = SurrogateGP(bounds.lower, bounds.upper)
        surrogate_pw.fit(X_lhs, Y_lhs)
        pw_metrics = validate(
            surrogate_pw, X_pw, Y_pw,
            label="pairwise edges",
            plot=True, save_dir=f"{results_dir}/Plots",
        )
        save_object(pw_metrics, f"{results_dir}/Data", "validation_metrics_pairwise")
        print("  Good pairwise R²: GP extrapolates well to 2-policy combinations.")
        print("  Poor pairwise R²: GP struggles at edges — consider adding more LHS points")
        print("                    at low policy counts (e.g. sparse X rows with 1–2 non-zeros).")

    poor = [k for k, v in val_metrics.items() if v["R2"] < 0.85]
    if poor:
        print(f"\n  *** WARNING: poor surrogate fit for: {poor} ***")
        print("  Consider increasing N_LHS before running BO.")
        print("  BO will still run — its uncertainty estimates help compensate —")
        print("  but Pareto front recommendations will be less reliable.\n")

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
        ev_lo=ev_lo,
        ev_hi=ev_hi,
        weights=weights,
        y_refs=y_refs,
        bau_baseline=bau_baseline,
        cache_path=bo_cache,
    )
    n_feas = ((Y_all[:, 0] >= ev_lo) & (Y_all[:, 0] <= ev_hi)).sum()
    print(f"\nTotal evaluations: {len(X_all)}   "
          f"Feasible (EV {ev_lo:.0%}–{ev_hi:.0%}): {n_feas}")
    plot_bo_convergence(Y_all, n_init=n_init, ev_lo=ev_lo, ev_hi=ev_hi,
                        save_dir=f"{results_dir}/Plots")

    # ------------------------------------------------------------------
    # Step 5: Final surrogate + Pareto front
    # ------------------------------------------------------------------
    print("\n=== Step 5: Final Surrogate + Pareto Front ===")
    surrogate_final = SurrogateGP(bounds.lower, bounds.upper)
    surrogate_final.fit(X_all, Y_all)
    save_surrogate(surrogate_final, f"{results_dir}/Data/surrogate.pkl")

    X_pareto, Y_pareto = compute_pareto_front(
        X_all, Y_all, surrogate_final, bounds,
        ev_lo=ev_lo, ev_hi=ev_hi, y_refs=y_refs, n_weight_vectors=100,
    )
    np.savez(f"{results_dir}/Data/pareto.npz", X=X_pareto, Y=Y_pareto)
    plot_pareto_front(X_pareto, Y_pareto, ev_lo, ev_hi, save_dir=f"{results_dir}/Plots")

    # ------------------------------------------------------------------
    # Step 6: Pareto summary table
    # ------------------------------------------------------------------
    print("\n=== Step 6: Pareto Front ===")
    if len(X_pareto) == 0:
        print("No feasible Pareto points found. Run more BO iterations or widen EV constraint.")
        return None, None

    header = f"{'#':<4} {'EV':>6} {'ΔUtility':>14} {'ΔEmissions':>14} {'ΔNet cost':>14}  " + \
             "  ".join(f"{n[:10]:>12}" for n in bounds.names)
    print("(Δ columns are BAU-relative — vs. doing nothing — not absolute values)")
    print(header)
    print("-" * len(header))
    for rank, i in enumerate(np.argsort(Y_pareto[:, 1])[::-1]):
        x, y = X_pareto[i], Y_pareto[i]
        policy_str = "  ".join(f"{v:>12.2f}" for v in x)
        print(f"{rank:<4} {y[0]:>6.3f} {y[1]:>14.4g} {y[2]:>14.4g} {y[3]:>14.4g}  {policy_str}")

    print(f"\nSaved to {results_dir}/Data/pareto.npz")
    print("Next: run `python -m package.surrogate.best_policies` to run the top "
          "policies through the real ABM and produce time-series + trade-off plots.")

    return X_pareto, Y_pareto


if __name__ == "__main__":
    # Example: reuse controllers from an existing experiment
    # X_pareto, Y_pareto = main(
    #     existing_calib_folder="results/endog_single_10_00_00__01_01_2026"
    # )

    X_pareto, Y_pareto = main()
