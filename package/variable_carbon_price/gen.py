"""
package/variable_carbon_price/gen.py — Rising carbon price vs. flat carbon tax,
naive vs. forward-looking agents.

WHY THIS EXPERIMENT
--------------------
The ABM lets agents/firms be either naive/permanent-policy (they assume
whatever carbon price applies today will last forever — the paper's default,
see socialNetworkUsers.py / firm.py) or forward-looking (they correctly
discount the KNOWN future price path — controller.compute_discounted_indices,
gated by the "forward_looking_expectations" flag). A steadily RISING carbon
price is exactly the case where this distinction should matter most: a naive
agent buying a car in month 1 undervalues how expensive fuel/emissions will
get later in the vehicle's life, so it should under-adopt EVs relative to a
forward-looking agent facing the identical price path.

This script runs three scenario families over the 2023-2035 (144-month)
policy period:

  BAU    Carbon_price off entirely.
  RAMP   Carbon_price linear from 0 up to `end_price` at month 144 —
         N_RAMPS different end_prices spanning a plausible policy range.
  FLAT   Carbon_price constant for the whole period. A linear ramp from 0 to
         X averages X/2 over the period, so comparing a ramp to a flat tax of
         X (not X/2) would overstate how much cheaper the ramp looks. FLAT
         scenarios are therefore built at HALF a subset of the ramps'
         end_prices, giving each flat scenario the same time-averaged carbon
         price as its reference ramp — any remaining difference in outcomes
         is attributable to the ramp SHAPE (and to agents' anticipation of
         it), not to a mismatched average price level.

Every scenario is run under both expectation_mode in {"naive",
"forward_looking"} (see forward_looking_expectations in controller.py), so
the plot script can show what forward-looking expectations actually change.

CALIBRATION REUSE
-------------------
All scenarios share the exact same 2001-2023 burn-in + calibration fit — see
package.surrogate.run.get_or_create_calibration (calibration is run ONCE,
Phase 1, and saved as controller_seed_*.pkl files) and
package.resources.run.load_in_controller (re-runs only the 144-month future
period from a deep-copied, already-calibrated controller — Phase 2). Only
Phase 2 differs per scenario/expectation_mode, which is what makes testing
dozens of policy paths here computationally reasonable.

OUTPUT
--------
For every (expectation_mode, scenario) pair, this saves the full monthly time
series (not just end-of-run scalars) for five metrics, stacked across seeds:
  cost       controller.history_policy_net_cost      (cumulative net policy
             cost/revenue to date — rebates+subsidies minus carbon-tax
             receipts, negative on the "cost" axis by convention, see
             controller.calc_net_policy_distortion)
  utility    social_network.history_total_utility     (aggregate utility
             flow that month, NOT cumulative)
  emissions  social_network.history_total_emissions   (aggregate driving +
             production emissions that month, NOT cumulative)
  ev_uptake  social_network.history_ev_adoption_rate  (fraction of the
             CURRENT fleet that is EV that month — a stock share)
  sales      derived from history_new_EV_cars_bought /
             history_new_ICE_cars_bought (EV share of NEW car purchases that
             month — a flow share, distinct from ev_uptake's stock share;
             NaN in the (essentially never reached) case nobody buys a new
             car that month)

Results are pickled to <results_dir>/Data/variable_carbon_price_results.pkl
as {(expectation_mode, label): {"months": ..., "cost": ..., "utility": ...,
"emissions": ..., "ev_uptake": ..., "sales": ..., "kind": ..., "value": ...},
...}. main() also calls plot.plot_time_series() at the end, so running this
module end-to-end produces one PNG per metric under
<results_dir>/Plots/ without a separate step — re-run just
`python -m package.variable_carbon_price.plot` later if you want to
regenerate the figures from saved data without resimulating (e.g. after
tweaking plot styling).
"""

import os

# Must be set before numpy/BLAS is imported: each of the seed_repetitions
# joblib workers otherwise spawns its own BLAS thread pool, oversubscribing
# the node's cores when running on a cluster.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, load as joblib_load

from package.resources.run import load_in_controller
from package.resources.utility import save_object, get_num_workers
from package.surrogate.run import get_or_create_calibration, _resolve_calib_folder

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

_CONSTANTS_DIR = "package/variable_carbon_price/constants"
BASE_PARAMS_PATH = f"{_CONSTANTS_DIR}/base_params_variable_carbon_price.json"
RESULTS_DIR = "results/variable_carbon_price"

N_RAMPS = 3
# $/kgCO2 reached at month 144 (12 years), ramping linearly up from 0.
RAMP_END_PRICES = np.linspace(0.1, 0.9, N_RAMPS)

N_FLATS = 2
# See module docstring: half of a subset of the ramps' end_prices, so each
# flat scenario has the same time-averaged carbon price as its reference ramp.
_FLAT_REFERENCE_PRICES = np.linspace(RAMP_END_PRICES.min(), RAMP_END_PRICES.max(), N_FLATS)
FLAT_PRICES = _FLAT_REFERENCE_PRICES / 2.0

EXPECTATION_MODES = ("naive", "forward_looking")
TIME_SERIES_METRICS = ("cost", "utility", "emissions", "ev_uptake", "sales")


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def _carbon_price_scenario(base_params_future, kind, value, expectation_mode):
    """
    Deep-copies base_params_future and sets the Carbon_price policy + the
    forward_looking_expectations flag + save_timeseries_data_state=1 (we need
    the full monthly history, not just end-of-run scalars, for every run here).
    """
    params = deepcopy(base_params_future)
    params["forward_looking_expectations"] = (expectation_mode == "forward_looking")
    params["save_timeseries_data_state"] = 1

    states = params["parameters_policies"]["States"]
    values = params["parameters_policies"]["Values"]["Carbon_price"]

    if kind == "bau":
        states["Carbon_price"] = 0
        values["Carbon_price_state"] = "flat"
        values["Carbon_price_init"] = 0.0
        values["Carbon_price"] = 0.0
    elif kind == "flat":
        states["Carbon_price"] = 1
        values["Carbon_price_state"] = "flat"
        values["Carbon_price_init"] = value
        values["Carbon_price"] = value
    elif kind == "ramp":
        states["Carbon_price"] = 1
        values["Carbon_price_state"] = "linear"
        values["Carbon_price_init"] = 0.0
        values["Carbon_price"] = value
    else:
        raise ValueError(f"Unknown scenario kind: {kind!r}")

    return params


def build_scenarios(
    ramp_end_prices=RAMP_END_PRICES,
    flat_prices=FLAT_PRICES,
    expectation_modes=EXPECTATION_MODES,
):
    """Returns a flat list of {label, kind, value, expectation_mode} dicts."""
    scenarios = []
    for mode in expectation_modes:
        scenarios.append({"label": "BAU", "kind": "bau", "value": 0.0, "expectation_mode": mode})
        for v in ramp_end_prices:
            scenarios.append({"label": f"ramp_{v:.3f}", "kind": "ramp", "value": float(v), "expectation_mode": mode})
        for v in flat_prices:
            scenarios.append({"label": f"flat_{v:.3f}", "kind": "flat", "value": float(v), "expectation_mode": mode})
    return scenarios


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_one_seed(base_params_future, scenario, controller_file):
    """Load one calibrated (deep-copyable) controller and run the future period."""
    controller = joblib_load(controller_file)
    params = _carbon_price_scenario(
        base_params_future, scenario["kind"], scenario["value"], scenario["expectation_mode"]
    )
    result = load_in_controller(controller, params)
    sn = result.social_network

    months = np.asarray(result.time_series) - (result.duration_burn_in + result.duration_calibration)

    new_ev = np.asarray(sn.history_new_EV_cars_bought, dtype=float)
    new_ice = np.asarray(sn.history_new_ICE_cars_bought, dtype=float)
    new_total = new_ev + new_ice
    sales_ev_share = np.divide(
        new_ev, new_total,
        out=np.full_like(new_ev, np.nan),
        where=new_total > 0,
    )

    return {
        "months": months,
        "cost": np.asarray(result.history_policy_net_cost),
        "utility": np.asarray(sn.history_total_utility),
        "emissions": np.asarray(sn.history_total_emissions),
        "ev_uptake": np.asarray(sn.history_ev_adoption_rate),
        "sales": sales_ev_share,
    }


def _run_scenario_all_seeds(base_params_future, scenario, controller_files):
    num_cores = get_num_workers()
    per_seed = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(_run_one_seed)(base_params_future, scenario, cf) for cf in controller_files
    )
    months = per_seed[0]["months"]  # identical across seeds: same duration_future/absolute_2035
    stacked = {
        metric: np.stack([r[metric] for r in per_seed], axis=1)  # shape (n_months, n_seeds)
        for metric in TIME_SERIES_METRICS
    }
    return months, stacked


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(
    base_params_path: str = BASE_PARAMS_PATH,
    existing_calib_folder: str = None,
    results_dir: str = RESULTS_DIR,
    ramp_end_prices=RAMP_END_PRICES,
    flat_prices=FLAT_PRICES,
    expectation_modes=EXPECTATION_MODES,
) -> dict:
    os.makedirs(f"{results_dir}/Data", exist_ok=True)

    print("=== Step 1: Calibration (shared across every scenario below) ===")
    # existing_calib_folder, if passed, always wins; otherwise auto-detect a
    # calib_folder.pkl already saved under results_dir/Data from a PRIOR call
    # to this same results_dir, so simply rerunning main() again (e.g.
    # `uv run python -m package.variable_carbon_price.gen`, no args) reuses
    # that calibration instead of silently redoing Phase 1 every time.
    existing_calib_folder = _resolve_calib_folder(results_dir, existing_calib_folder)
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, existing_calib_folder
    )
    n_seeds = len(controller_files)
    save_object(calib_folder, f"{results_dir}/Data", "calib_folder")

    scenarios = build_scenarios(ramp_end_prices, flat_prices, expectation_modes)
    print(f"  Seeds: {n_seeds}")
    print(f"=== Step 2: {len(scenarios)} scenarios x {n_seeds} seeds "
          f"= {len(scenarios) * n_seeds} future-period ABM runs ===")

    results = {}
    for i, scenario in enumerate(scenarios):
        print(f"  [{i + 1}/{len(scenarios)}] {scenario['expectation_mode']:>16} | {scenario['label']}")
        months, stacked = _run_scenario_all_seeds(base_params, scenario, controller_files)
        results[(scenario["expectation_mode"], scenario["label"])] = {
            "months": months,
            "kind": scenario["kind"],
            "value": scenario["value"],
            **stacked,
        }

    save_object(results, f"{results_dir}/Data", "variable_carbon_price_results")
    save_object(scenarios, f"{results_dir}/Data", "scenarios")
    print(f"\nSaved to {results_dir}/Data/variable_carbon_price_results.pkl")

    print("\n=== Step 3: Plotting ===")
    from . import plot  # deferred: keeps matplotlib off the import path for callers that only want data
    plot.plot_time_series(results, scenarios, save_dir=f"{results_dir}/Plots")
    print(f"Saved one PNG per metric to {results_dir}/Plots/")

    return results


if __name__ == "__main__":
    # Optional: --existing_calib_folder=PATH to reuse a calibration produced
    # by ANOTHER package's run (e.g. package.car_ban), instead of only
    # auto-detecting one from this same results_dir. Valid because
    # base_params_variable_carbon_price.json and base_params_car_ban.json are
    # currently byte-for-byte identical -- a calibration from either is a
    # valid calibration for both. Falls back to the auto-detect in main() if
    # not given.
    import sys
    _calib_arg = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--existing_calib_folder="):
            _calib_arg = _arg.split("=", 1)[1]
    main(existing_calib_folder=_calib_arg)
