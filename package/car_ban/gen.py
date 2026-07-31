"""
package/car_ban/gen.py — ICE driving ban in 2030 vs. 2035 vs. 2040 vs. 2050,
naive vs. forward-looking firms and consumers.

The simulation itself only ever runs through the policy period to ~2035
(duration_future=144 months, unchanged from before) -- it is NOT extended to
actually reach 2040/2050. Those two ban years deliberately fall beyond the
simulated horizon, exercising controller.compute_discounted_indices' horizon
extrapolation (see its docstring and the "Bans beyond the simulated
horizon" section of paper/forward_looking_expectations.tex): a
forward-looking agent still partially anticipates a 2040 or 2050 ban within
the simulated 2024-2035 window, just less than it would a nearer one, while
a naive agent is (correctly) completely unaffected by either, since the
simulation ends long before the ban ever actually takes effect for them.

WHY THIS EXPERIMENT
--------------------
The ABM lets firms/agents be either naive/permanent-policy (they act as if
today's rules last forever, the paper's default) or forward-looking (they
correctly discount a KNOWN future price path — see
forward_looking_expectations in controller.py / socialNetworkUsers.py /
firm.py). A future ICE driving ban is a natural second test of this
machinery, alongside a rising carbon price (package.variable_carbon_price).

WHAT THE BAN IS AND HOW ANTICIPATION EMERGES
-----------------------------------------------
This is a DRIVING ban, not a sale ban: from ICE_driving_ban_time onward,
driving an ICE car (new OR second-hand — see controller._unpack_ice_driving_ban_parameters
docstring) is made prohibitively costly by adding ICE_DRIVING_BAN_PENALTY to
the effective per-unit ICE fuel cost for every month from that point on. This
extra cost is fed into the exact same gas-price path
(controller.compute_discounted_indices' gas_cost_effective_vec) that every
consumer and firm already reads for ordinary fuel-cost purposes — no
ICE-specific firm switch, no separately-chosen "anticipation lead"
parameter, and no changes anywhere in firm.py:
  naive             only reads TODAY's fuel cost, so it doesn't notice
                    anything until t reaches ICE_driving_ban_time, then
                    switches away from ICE via the ordinary utility-driven
                    choice mechanism (which already exists) as soon as it
                    happens to re-evaluate.
  forward_looking   discounts the WHOLE known future cost path (the same
                    Index mechanism used for carbon price), so it sees the
                    penalty coming and erodes its valuation of ICE
                    gradually as the ban approaches — governed entirely by
                    the model's own discount rate r and depreciation rate
                    delta, not by any hand-picked lead time.
Firms need no ban-specific code at all: as consumer demand for ICE erodes
(faster for forward-looking consumers, right at the deadline for naive
ones), expected profit from producing/researching ICE erodes with it, and
the EXISTING profit-maximising choose_cars_segments()/innovate() logic
naturally shifts toward EV as a direct, emergent consequence — this is
exactly "the firm just picks whichever car is currently most profitable",
which is how firms already behave; nothing new was added on the firm side.

Because the mechanism acts through the fuel-cost channel, it also reaches
second-hand ICE cars (secondHandMerchant.py already refreshes
car.fuel_cost_c from the same gas_price every step) — correct for a driving
ban, which (unlike a sale ban) makes every ICE car on the road illegal to
drive, regardless of when or from whom it was bought.

CALIBRATION REUSE
-------------------
Identical pattern to package.variable_carbon_price.gen: calibration (burn-in
+ 2001-2023 historical fit) is run ONCE via
package.surrogate.run.get_or_create_calibration and reused across every
(ban_year, expectation_mode) combination via
package.resources.run.load_in_controller, so only the 144-month future
period differs per scenario.

ONE SELF-CONTAINED, TIMESTAMPED FOLDER PER RUN
--------------------------------------------------
main() mints a fresh results/car_ban_<timestamp> folder every call
(results_dir=None, the default) — same convention as
produce_name_datetime("surrogate_calibration")/("policy_test_future_sight").
Calibration (unless existing_calib_folder is given), car_ban_results.pkl,
scenarios.pkl and every plot all live under that ONE folder — nothing is
split across a shared fixed results/car_ban path any more, so two different
runs (changed ban years, changed expectation modes, a rerun after a model
change) can never silently overwrite or cross-contaminate each other.

OUTPUT
--------
For every (expectation_mode, scenario) pair, saves the full monthly time
series (not just end-of-run scalars) for five metrics, stacked across seeds
— see package.variable_carbon_price.gen for the exact same metric
definitions (cost / utility / emissions / ev_uptake / sales). Results are
pickled to <results_dir>/Data/car_ban_results.pkl as
{(expectation_mode, label): {"months": ..., "cost": ..., "utility": ...,
"emissions": ..., "ev_uptake": ..., "sales": ..., "kind": ..., "value": ...},
...}. main() also calls plot.plot_time_series() at the end, producing one
PNG per metric under <results_dir>/Plots/.
"""

import os

# Must be set before numpy/BLAS is imported: each of the seed_repetitions
# joblib workers otherwise spawns its own BLAS thread pool, oversubscribing
# the node's cores when running on a cluster.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
import shutil
from pathlib import Path
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, load as joblib_load

from package.resources.run import load_in_controller
from package.resources.utility import save_object, get_num_workers, createFolder, produce_name_datetime
from package.surrogate.run import get_or_create_calibration

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

_CONSTANTS_DIR = "package/car_ban/constants"
BASE_PARAMS_PATH = f"{_CONSTANTS_DIR}/base_params_car_ban.json"

# Prefix for the fresh results/<prefix>_<timestamp> folder main() mints per
# call when results_dir isn't given explicitly — same convention as
# produce_name_datetime("surrogate_calibration")/("policy_test_future_sight").
# No fixed RESULTS_DIR any more: calibration + Data + Plots for a given run
# all live under that ONE self-contained folder (see main()), rather than a
# shared path that a later run could silently overwrite or reuse stale data from.
RESULTS_PREFIX = "car_ban"

# controller.t_2030 = duration_burn_in + duration_calibration + 6*12, i.e. the
# future period starts at year 2024 (see controller.unpack_controller_parameters).
_FUTURE_PERIOD_START_YEAR = 2024

BAN_YEARS = [2030, 2036, 2040, 2050]
# 2040 and 2050 fall beyond the simulated horizon (duration_future=144
# months => the future period runs 2024-~2035/36, unchanged) -- see the
# module docstring above.

# Added to the effective per-unit ICE fuel cost for every month from the ban
# onward (see controller._unpack_ice_driving_ban_parameters). Large relative
# to the calibrated gas price so that, once in effect, P(choose/keep ICE)
# collapses to ~0 via the ordinary logit choice mechanism -- tune and
# re-check via the sales/ev_uptake plots if this ever looks too soft or too
# knife-edge for a given calibration. Kept an order of magnitude below the
# MAX_LIFECYCLE_COST_TERM(_FIRM) safety clip in socialNetworkUsers.py/firm.py
# even after the model's own ~400x amplification factor and worst-case
# gamma_i (emissions willingness-to-pay) draws -- comfortable margin rather
# than relying on the clip alone.
ICE_DRIVING_BAN_PENALTY = 10.0

EXPECTATION_MODES = ("naive", "forward_looking")
TIME_SERIES_METRICS = ("cost", "utility", "emissions", "ev_uptake", "sales")


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def _ice_driving_ban_time_param(ban_year, duration_calibration):
    """
    ICE_driving_ban_time is read by controller.py relative to the END OF
    BURN-IN (the same convention as ev_research_start_time/
    ev_production_start_time — see controller._unpack_ice_driving_ban_parameters),
    i.e. duration_calibration (to reach the end of calibration) plus however
    many months into the future period are needed to reach ban_year.
    """
    return duration_calibration + (ban_year - _FUTURE_PERIOD_START_YEAR) * 12


def _car_ban_scenario(base_params_future, ban_year, expectation_mode,
                       penalty=ICE_DRIVING_BAN_PENALTY):
    """
    Deep-copies base_params_future and sets the ICE_driving_ban_time policy +
    the forward_looking_expectations flag + save_timeseries_data_state=1 (we
    need the full monthly history, not just end-of-run scalars, for every
    run here). ban_year=None means BAU (no ban at all).
    """
    params = deepcopy(base_params_future)
    params["forward_looking_expectations"] = (expectation_mode == "forward_looking")
    params["save_timeseries_data_state"] = 1

    if ban_year is None:
        params["ICE_driving_ban_time"] = None
    else:
        params["ICE_driving_ban_time"] = _ice_driving_ban_time_param(ban_year, base_params_future["duration_calibration"])
    params["ICE_driving_ban_penalty"] = penalty

    return params


def build_scenarios(ban_years=BAN_YEARS, expectation_modes=EXPECTATION_MODES):
    """Returns a flat list of {label, kind, value, expectation_mode} dicts."""
    scenarios = []
    for mode in expectation_modes:
        scenarios.append({"label": "BAU", "kind": "bau", "value": None, "expectation_mode": mode})
        for year in ban_years:
            scenarios.append({"label": f"ban_{year}", "kind": "ban", "value": int(year), "expectation_mode": mode})
    return scenarios


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_one_seed(base_params_future, scenario, controller_file):
    """Load one calibrated (deep-copyable) controller and run the future period."""
    controller = joblib_load(controller_file)
    params = _car_ban_scenario(
        base_params_future, scenario["value"], scenario["expectation_mode"]
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
    results_dir: str = None,
    ban_years=BAN_YEARS,
    expectation_modes=EXPECTATION_MODES,
) -> dict:
    """
    results_dir : None (default) mints a fresh results/car_ban_<timestamp>
        folder for THIS call — calibration (unless existing_calib_folder is
        given), car_ban_results.pkl/scenarios.pkl, and every plot all live
        under that ONE self-contained folder, so two different runs (changed
        ban years, changed expectation modes, a rerun after a model change)
        can never silently overwrite or cross-contaminate each other. Pass an
        existing path explicitly only to deliberately resume/extend that
        exact prior run.
    """
    if results_dir is None:
        results_dir = produce_name_datetime(RESULTS_PREFIX)
    createFolder(results_dir)
    print(f"Results folder for this run: {results_dir}")

    print("=== Step 1: Calibration (shared across every scenario below) ===")
    # existing_calib_folder is only used if explicitly passed (e.g. to reuse
    # another experiment's calibration on purpose) — no auto-detection from a
    # prior call against this same results_dir, so this always runs Phase 1
    # fresh unless you deliberately ask it not to. That avoids silently
    # reusing a stale calibration after changing calibration settings.
    # target_folder=results_dir: a fresh calibration is written straight into
    # results_dir/Calibration_runs (calib_folder == results_dir) instead of a
    # separate sibling folder — ignored when existing_calib_folder is given.
    ran_fresh_calibration = existing_calib_folder is None
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, existing_calib_folder, target_folder=results_dir
    )
    n_seeds = len(controller_files)

    scenarios = build_scenarios(ban_years, expectation_modes)
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

    save_object(results, f"{results_dir}/Data", "car_ban_results")
    save_object(scenarios, f"{results_dir}/Data", "scenarios")
    print(f"\nSaved to {results_dir}/Data/car_ban_results.pkl")

    print("\n=== Step 3: Plotting ===")
    from . import plot  # deferred: keeps matplotlib off the import path for callers that only want data
    plot.plot_time_series(results, scenarios, save_dir=f"{results_dir}/Plots")
    print(f"Saved one PNG per metric to {results_dir}/Plots/")

    # Nothing downstream reuses this calibration, so don't leave it on disk —
    # but only if we created it this call; an explicitly-passed
    # existing_calib_folder belongs to another experiment and must be left alone.
    # calib_folder == results_dir in the common (fresh) case, so this only
    # clears the (large) Calibration_runs subfolder, leaving car_ban_results.pkl/
    # scenarios.pkl/Plots untouched in the same self-contained folder.
    if ran_fresh_calibration:
        shutil.rmtree(Path(calib_folder) / "Calibration_runs", ignore_errors=True)

    print(f"\nEverything from this run (results, plots) is in: {results_dir}")

    return results


if __name__ == "__main__":
    # Optional: --existing_calib_folder=PATH to reuse a calibration produced
    # by ANOTHER package's run (e.g. package.variable_carbon_price). Valid
    # because base_params_car_ban.json and base_params_variable_carbon_price.json
    # are currently byte-for-byte identical -- a calibration from either is a
    # valid calibration for both. Without this flag, main() always runs
    # Phase 1 fresh.
    import sys
    _calib_arg = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--existing_calib_folder="):
            _calib_arg = _arg.split("=", 1)[1]
    main(existing_calib_folder=_calib_arg)
