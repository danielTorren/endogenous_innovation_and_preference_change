"""
package/command_and_control/gen.py — command-and-control ICE phase-out policy
stringency, compared additively: research ban vs. research+sales ban vs.
research+sales+driving ban, at a 2030 vs. a 2035 start date, simulated all
the way out to 2050 (duration_future=312 — see base_params_command_and_control.json).

WHY THIS EXPERIMENT
--------------------
package.car_ban tests a pure DRIVING ban (a cost shock on ICE fuel that
reaches every ICE car on the road, new or second-hand — see
controller._unpack_ice_driving_ban_parameters). Real-world "ICE phase-out"
policy packages are usually layered instead: first a research/R&D
restriction, then a ban on NEW sales, and only rarely (if ever) an outright
ban on driving an already-owned ICE car. This experiment compares those
three stringency levels head to head, additively:

  research_ban                  ICE_research_ban_time only
  research_sales_ban            + ICE_sales_ban_time (same date)
  research_sales_driving_ban    + ICE_driving_ban_time (same date)

All three bans, once active, stay active for the rest of the run (through
2050) -- unlike market-based instruments elsewhere in this repo (e.g. a
carbon price ramp that is explicitly coded to end at absolute_2035, see
controller.compute_discounted_indices), a command-and-control ban has no
reason to spontaneously reverse.

WHAT EACH BAN DOES AND HOW THEY DIFFER
-----------------------------------------
  ICE_research_ban_time   firm.innovate() stops generating/selecting new ICE
                           neighbouring technologies (firm.ice_research_ban_active,
                           see controller._unpack_ice_research_ban_parameters).
                           Already-researched ICE models remain in a firm's
                           memory and can still be SOLD until/unless the sales
                           ban also kicks in -- research and sales are
                           deliberately independent switches.
  ICE_sales_ban_time       firm.choose_cars_segments() drops ICE cars from its
                           candidate pool before pricing/profit are even
                           computed, and firm.next_step() immediately flushes
                           any ICE car still sitting in cars_on_sale from a
                           pre-ban period (see controller._unpack_ice_sales_ban_parameters).
                           This is a hard constraint on NEW sales only --
                           already-sold ICE cars stay legal to drive and
                           resellable on the second-hand market.
  ICE_driving_ban_time     identical mechanism to package.car_ban: a cost
                           shock added to the effective ICE fuel cost, reaching
                           every ICE car on the road (new AND second-hand),
                           see controller._unpack_ice_driving_ban_parameters.

Agents here are always forward_looking_expectations=True for the FUTURE
period only, set per-scenario in _command_and_control_scenario() below --
exactly like package.car_ban's expectation_mode toggle. base_params_command_and_control.json
deliberately does NOT set this key: calibration (Phase 1, 2001-2023 historical
fit) must stay naive regardless, since the model's calibrated parameters
assume naive/permanent expectations during that period (baking the flag into
the base params file would apply it to calibration too and visibly break EV
adoption during the historical fit -- confirmed the hard way). This
experiment is about comparing policy STRINGENCY, not naive-vs-forward-looking
anticipation (that comparison is package.car_ban's job). Forward-looking
agents still only anticipate
what the model already gives them a channel for: the driving ban's cost
shock is discounted in advance via compute_discounted_indices' gas_cost_index
path exactly as in car_ban; the sales/research bans are hard availability
constraints with no forward-discounting channel, so they bite deterministically
right at *_ban_time regardless of expectation mode -- see firm.py.

CALIBRATION REUSE
-------------------
Identical pattern to package.car_ban/package.variable_carbon_price:
calibration (burn-in + 2001-2023 historical fit) is run ONCE via
package.surrogate.run.get_or_create_calibration and reused across every
(ban_year, stringency) combination via package.resources.run.load_in_controller,
so only the 312-month future period (2024-2050) differs per scenario.

OUTPUT
--------
For every (ban_year, stringency) pair plus one BAU baseline, saves the full
monthly time series (not just end-of-run scalars) for five metrics, stacked
across seeds -- see package.car_ban.gen for the exact same metric
definitions (cost / utility / emissions / ev_uptake / sales). Results are
pickled to <results_dir>/Data/command_and_control_results.pkl as
{label: {"months": ..., "cost": ..., "utility": ..., "emissions": ...,
"ev_uptake": ..., "sales": ..., "kind": ..., "ban_year": ..., "stringency": ...},
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

import shutil
from pathlib import Path
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, load as joblib_load

from package.resources.run import load_in_controller
from package.resources.utility import save_object, get_num_workers
from package.surrogate.run import get_or_create_calibration

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

_CONSTANTS_DIR = "package/command_and_control/constants"
BASE_PARAMS_PATH = f"{_CONSTANTS_DIR}/base_params_command_and_control.json"
RESULTS_DIR = "results/command_and_control"

# controller.t_2030 = duration_burn_in + duration_calibration + 6*12, i.e. the
# future period starts at year 2024 (see controller.unpack_controller_parameters).
_FUTURE_PERIOD_START_YEAR = 2024

BAN_YEARS = [2030, 2035]

STRINGENCY_LEVELS = ("research_ban", "research_sales_ban", "research_sales_driving_ban")
STRINGENCY_LABELS = {
    "research_ban": "Research ban only",
    "research_sales_ban": "Research ban + sales ban",
    "research_sales_driving_ban": "Research ban + sales ban + driving ban",
}

# Added to the effective per-unit ICE fuel cost for every month from the
# driving ban onward (only used by the research_sales_driving_ban level —
# see controller._unpack_ice_driving_ban_parameters). Same value/rationale as
# package.car_ban.gen.ICE_DRIVING_BAN_PENALTY.
ICE_DRIVING_BAN_PENALTY = 10.0

TIME_SERIES_METRICS = ("cost", "utility", "emissions", "ev_uptake", "sales")


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def _ban_time_param(ban_year, duration_calibration):
    """
    ICE_research_ban_time/ICE_sales_ban_time/ICE_driving_ban_time are all read
    by controller.py relative to the END OF BURN-IN (the same convention as
    ev_research_start_time/ev_production_start_time — see
    controller._unpack_ice_research_ban_parameters/_unpack_ice_sales_ban_parameters/
    _unpack_ice_driving_ban_parameters), i.e. duration_calibration (to reach
    the end of calibration) plus however many months into the future period
    are needed to reach ban_year.
    """
    return duration_calibration + (ban_year - _FUTURE_PERIOD_START_YEAR) * 12


def _command_and_control_scenario(base_params_future, ban_year, stringency,
                                   penalty=ICE_DRIVING_BAN_PENALTY):
    """
    Deep-copies base_params_future and sets the three ban-time policies
    additively according to stringency + save_timeseries_data_state=1 (we
    need the full monthly history, not just end-of-run scalars, for every run
    here). ban_year=None means BAU (no bans at all).
    """
    params = deepcopy(base_params_future)
    params["forward_looking_expectations"] = True
    params["save_timeseries_data_state"] = 1

    if ban_year is None:
        params["ICE_research_ban_time"] = None
        params["ICE_sales_ban_time"] = None
        params["ICE_driving_ban_time"] = None
        params["ICE_driving_ban_penalty"] = 0.0
        return params

    ban_time = _ban_time_param(ban_year, base_params_future["duration_calibration"])

    params["ICE_research_ban_time"] = ban_time
    params["ICE_sales_ban_time"] = ban_time if stringency in ("research_sales_ban", "research_sales_driving_ban") else None
    params["ICE_driving_ban_time"] = ban_time if stringency == "research_sales_driving_ban" else None
    params["ICE_driving_ban_penalty"] = penalty if stringency == "research_sales_driving_ban" else 0.0

    return params


def build_scenarios(ban_years=BAN_YEARS, stringency_levels=STRINGENCY_LEVELS):
    """Returns a flat list of {label, kind, ban_year, stringency} dicts."""
    scenarios = [{"label": "BAU", "kind": "bau", "ban_year": None, "stringency": None}]
    for year in ban_years:
        for stringency in stringency_levels:
            scenarios.append({
                "label": f"{year}_{stringency}",
                "kind": "ban",
                "ban_year": int(year),
                "stringency": stringency,
            })
    return scenarios


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_one_seed(base_params_future, scenario, controller_file):
    """Load one calibrated (deep-copyable) controller and run the future period."""
    controller = joblib_load(controller_file)
    params = _command_and_control_scenario(
        base_params_future, scenario["ban_year"], scenario["stringency"]
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
    months = per_seed[0]["months"]  # identical across seeds: same duration_future
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
    ban_years=BAN_YEARS,
    stringency_levels=STRINGENCY_LEVELS,
) -> dict:
    os.makedirs(f"{results_dir}/Data", exist_ok=True)

    print("=== Step 1: Calibration (shared across every scenario below) ===")
    # existing_calib_folder is only used if explicitly passed (e.g. to reuse
    # another experiment's calibration on purpose) — no auto-detection from a
    # prior call against this same results_dir, so this always runs Phase 1
    # fresh unless you deliberately ask it not to. That avoids silently
    # reusing a stale calibration after changing calibration settings.
    ran_fresh_calibration = existing_calib_folder is None
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, existing_calib_folder
    )
    n_seeds = len(controller_files)

    scenarios = build_scenarios(ban_years, stringency_levels)
    print(f"  Seeds: {n_seeds}")
    print(f"=== Step 2: {len(scenarios)} scenarios x {n_seeds} seeds "
          f"= {len(scenarios) * n_seeds} future-period ABM runs ===")

    results = {}
    for i, scenario in enumerate(scenarios):
        print(f"  [{i + 1}/{len(scenarios)}] {scenario['label']}")
        months, stacked = _run_scenario_all_seeds(base_params, scenario, controller_files)
        results[scenario["label"]] = {
            "months": months,
            "kind": scenario["kind"],
            "ban_year": scenario["ban_year"],
            "stringency": scenario["stringency"],
            **stacked,
        }

    save_object(results, f"{results_dir}/Data", "command_and_control_results")
    save_object(scenarios, f"{results_dir}/Data", "scenarios")
    print(f"\nSaved to {results_dir}/Data/command_and_control_results.pkl")

    print("\n=== Step 3: Plotting ===")
    from . import plot  # deferred: keeps matplotlib off the import path for callers that only want data
    plot.plot_time_series(results, scenarios, save_dir=f"{results_dir}/Plots")
    print(f"Saved one PNG per metric to {results_dir}/Plots/")

    # Nothing downstream reuses this calibration, so don't leave it on disk —
    # but only if we created it this call; an explicitly-passed
    # existing_calib_folder belongs to another experiment and must be left alone.
    if ran_fresh_calibration:
        shutil.rmtree(Path(calib_folder) / "Calibration_runs", ignore_errors=True)

    return results


if __name__ == "__main__":
    # Optional: --existing_calib_folder=PATH to reuse a calibration produced
    # by ANOTHER package's run (e.g. package.car_ban / package.variable_carbon_price)
    # -- valid as long as its base_params_*.json shares the same burn-in/
    # calibration settings as base_params_command_and_control.json. Without
    # this flag, main() always runs Phase 1 fresh.
    import sys
    _calib_arg = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--existing_calib_folder="):
            _calib_arg = _arg.split("=", 1)[1]
    main(existing_calib_folder=_calib_arg)
