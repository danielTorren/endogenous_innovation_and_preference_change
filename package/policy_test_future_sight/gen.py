"""
package/policy_test_future_sight/gen.py — naive vs. forward-looking agents,
applied to the surrogate's top-10 cheapest feasible policies.

WHY THIS EXPERIMENT
--------------------
package.surrogate.run finds the cheapest feasible policies under the
ABM's default (naive/permanent-policy) agent behaviour — see
forward_looking_expectations in controller.py/socialNetworkUsers.py/firm.py.
This experiment asks: does that ranking, and the trajectories each policy
produces, change if agents/firms instead correctly anticipate the whole
known future policy and price path? It re-runs the exact same top-10 policy
vectors (plus BAU) under both expectation_mode in {"naive",
"forward_looking"}, so the two can be compared side by side.

WHAT THIS RUNS
--------------
Phase 1 — Calibration: always run fresh here (no cross-session
auto-reuse — pass existing_calib_folder explicitly if you deliberately want
to reuse a specific prior calibration; see package.surrogate.run's
get_or_create_calibration docstring).

Phase 2 — (BAU + top-10 policies) x (naive, forward_looking) x N_seeds
future-period ABM runs, full time series (needed for the comparison plots,
not just end-of-run scalars).

INPUTS
------
base_params_path : JSON config — must match what the top-10 policies were
                    found under (defaults to the surrogate's own
                    base_params_surrogate.json, same duration_future etc.).
                    This is a static, checked-in config file, not a
                    per-run artifact, so it's fine to keep reading it
                    directly from package/surrogate/constants/.
pareto_path       : deliberately NOT read from package.surrogate's own
                    results/surrogate_optimisation/Data/pareto.npz — that
                    folder is a per-run artifact that gets rewritten/cleared
                    every time package.surrogate.run is rerun, which would
                    silently break this experiment's input if the two ever
                    ran out of sync. Instead this reads a LOCAL copy from
                    package/policy_test_future_sight/input_data/pareto.npz —
                    after each package.surrogate.run finishes, copy its
                    results/surrogate_optimisation/Data/pareto.npz to that
                    path before running this experiment.

OUTPUT
------
Saves, under a fresh results/policy_test_future_sight_<timestamp>/Data/:
  outputs_by_mode      : {expectation_mode: {rank: output_dict}}
  outputs_bau_by_mode   : {expectation_mode: output_dict}
  policy_dicts          : {rank: {policy_name: intensity}}
  Y_top                 : (10, 3) array — [log_utility, emissions, net_cost]
  base_params
Then calls plot.plot_future_sight_comparison() to produce one PNG per metric
under .../Plots/, each with two side-by-side panels (naive | forward_looking)
sharing a y-axis so the two are directly comparable.

Usage (from repo root):
    python -m package.policy_test_future_sight.gen
"""

import os

# Must be set before numpy/BLAS is imported: each of the seed_repetitions
# joblib workers otherwise spawns its own BLAS thread pool, oversubscribing
# the node's cores when running on a cluster.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from package.resources.utility import save_object, createFolder, produce_name_datetime
from package.surrogate.sampling import load_policy_bounds
from package.surrogate.run import get_or_create_calibration, BASE_PARAMS_PATH, BOUNDS_PATH
from package.surrogate.best_policies import run_final_abm, select_top_policies

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARETO_PATH = os.path.join(_THIS_DIR, "input_data", "pareto.npz")
N_BEST = 10
EXPECTATION_MODES = ("naive", "forward_looking")


def main(
    base_params_path: str = BASE_PARAMS_PATH,
    bounds_path: str = BOUNDS_PATH,
    pareto_path: str = PARETO_PATH,
    n_best: int = N_BEST,
    existing_calib_folder: str = None,
) -> str:
    """
    Full future-sight comparison workflow. Returns the output folder path.
    """
    print("=== Step 1: Calibration ===")
    # existing_calib_folder is only used if explicitly passed — always runs
    # Phase 1 fresh otherwise, same convention as package.surrogate.run /
    # package.variable_carbon_price.gen / package.car_ban.gen.
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, existing_calib_folder
    )
    n_seeds = len(controller_files)
    print(f"  Seeds: {n_seeds}")

    bounds = load_policy_bounds(bounds_path)

    if not os.path.exists(pareto_path):
        raise FileNotFoundError(
            f"{pareto_path} not found. Run package.surrogate.run, then copy its "
            f"results/surrogate_optimisation/Data/pareto.npz to {pareto_path} "
            "before running this experiment."
        )
    data = np.load(pareto_path)
    X_ranked, Y_ranked = data["X"], data["Y"]
    if len(X_ranked) == 0:
        raise ValueError(
            f"No feasible policies found in {pareto_path} — run package.surrogate.run first."
        )
    X_top, Y_top = select_top_policies(X_ranked, Y_ranked, n_best)
    print(f"  Loaded top {len(X_top)} policies from {pareto_path}")

    n_scenarios = len(X_top) + 1  # + BAU
    print(f"\n=== Step 2: {n_scenarios} scenarios (BAU + top {len(X_top)}) x "
          f"{len(EXPECTATION_MODES)} expectation modes x {n_seeds} seeds = "
          f"{n_scenarios * len(EXPECTATION_MODES) * n_seeds} future-period ABM runs ===")

    policy_dicts = {rank: dict(zip(bounds.names, x)) for rank, x in enumerate(X_top)}

    outputs_by_mode = {}
    outputs_bau_by_mode = {}

    for mode in EXPECTATION_MODES:
        forward_looking = (mode == "forward_looking")
        print(f"\n--- Expectation mode: {mode} ---")

        print("  BAU")
        outputs_bau_by_mode[mode] = run_final_abm(
            np.zeros(bounds.n), base_params, controller_files, bounds=bounds,
            save=False, forward_looking_expectations=forward_looking,
        )

        outputs_by_mode[mode] = {}
        for rank, x in enumerate(X_top):
            print(f"  Policy {rank + 1}/{len(X_top)}")
            outputs_by_mode[mode][rank] = run_final_abm(
                x, base_params, controller_files, bounds=bounds,
                save=False, forward_looking_expectations=forward_looking,
            )

    out_folder = produce_name_datetime("policy_test_future_sight")
    createFolder(out_folder)
    save_object(outputs_by_mode,     out_folder + "/Data", "outputs_by_mode")
    save_object(outputs_bau_by_mode, out_folder + "/Data", "outputs_bau_by_mode")
    save_object(policy_dicts,        out_folder + "/Data", "policy_dicts")
    save_object(Y_top,               out_folder + "/Data", "Y_top")
    save_object(base_params,         out_folder + "/Data", "base_params")
    print(f"\nSaved to {out_folder}/Data/")

    print("\n=== Step 3: Plotting ===")
    from . import plot  # deferred: keeps matplotlib off the import path for callers that only want data
    plot.plot_future_sight_comparison(base_params, out_folder, outputs_by_mode, outputs_bau_by_mode, policy_dicts, Y_top)
    print(f"Plots saved to {out_folder}/Plots/")

    return out_folder


if __name__ == "__main__":
    # Optional: --existing_calib_folder=PATH to reuse a calibration produced
    # by another package's run (e.g. package.surrogate, package.car_ban,
    # package.variable_carbon_price — base_params_surrogate.json's burn-in/
    # calibration settings must match). Without this flag, Phase 1 always
    # runs fresh.
    import sys
    _calib_arg = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--existing_calib_folder="):
            _calib_arg = _arg.split("=", 1)[1]
    main(existing_calib_folder=_calib_arg)
