"""
Engine for a one-parameter sweep that records new-car HHI and fleet car age.

Both are calibration targets with published bands (0.11-0.18 and 10-12 years),
and most single parameters move them together rather than independently, so the
useful question is never "what does this parameter do to HHI" but "is there any
value of it that puts both series inside their bands at once". This module runs
that experiment; the thin entry scripts next to it supply the parameter.

Entry points that use it:
    package/generating_data/kappa_sweep_gen.py   parameters_vehicle_user.kappa
    package/generating_data/delta_sweep_gen.py   parameters_ICE.delta

Outputs one results/<param>_sweep_<timestamp> folder holding
  Data/param_list, Data/param_name, Data/param_label, Data/base_params,
  Data/summary
  Data/data_array_hhi          (n_values, seeds, T)   unit-share new-car HHI
  Data/data_array_age_fleet    (n_values, seeds, T)   whole-fleet mean age, MONTHS
  Data/data_array_ev_prop      (n_values, seeds, T)
plus the two figures written by package/plotting_data/sweep_hhi_age_plot.py.

The HHI series only exists when save_timeseries_data_state is 1 --
history_market_concentration is appended from
Firm_Manager.save_timeseries_data_firm_manager(), which is gated on it. The base
JSON already sets it; do not switch it off.
"""
import json
import sys

import numpy as np

from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder,
    params_list_with_seed,
    produce_name_datetime,
    save_object,
)

BASE_PARAMS_LOAD = "package/constants/base_params_calibration.json"

# Reported alongside each row, from package/plotting_data/calibration_plot.py.
TARGET_HHI = (0.11, 0.18)
TARGET_FLEET_AGE_YEARS = (10, 12)


def load_base_params():
    with open(BASE_PARAMS_LOAD) as f:
        return json.load(f)


def build_params(param_name, subdict, value, seeds):
    bp = load_base_params()
    bp[subdict][param_name] = float(value)
    bp["seed_repetitions"] = seeds
    return bp


def final_year_mean(arr_2d):
    """Mean over seeds of the last 12 recorded steps, i.e. the 2023 calendar year."""
    return float(np.nanmean(np.asarray(arr_2d, dtype=float)[:, -12:]))


def parse_seeds(argv, default):
    """`--seeds N` off the command line, so a quick check needs no file edit."""
    args = list(argv)
    seeds = default
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = int(args[i + 1])
    return seeds


def run_sweep(param_name, subdict, values, param_label, seeds, precheck=None):
    r"""
    Run one sweep and return the results folder.

    param_name  key inside `subdict` of the base params JSON, also the folder slug
    param_label how it is written on the figures, e.g. r"$\delta$"
    precheck    optional callable(base_params, values); raise from it to reject a
                grid BEFORE 640 runs are launched rather than inside a worker
    """
    values = np.asarray(values, dtype=float)
    base_params = build_params(param_name, subdict, values[0], seeds)

    if precheck is not None:
        precheck(base_params, values)

    root = produce_name_datetime(f"{param_name}_sweep")
    createFolder(root)
    burn_in = base_params["duration_burn_in"]

    header = f"{param_name:>12}{'HHI':>8}{'age yr':>9}{'EV 2023':>10}"
    print(f"fileName: {root}")
    print(f"{param_name} in {subdict}: {len(values)} values, seeds each: {seeds}, "
          f"total runs: {len(values) * seeds}\n")
    print(header)
    print(f"{'TARGET':>12}{'.11-.18':>8}{'10-12':>9}{'0.038':>10}")
    print("-" * len(header))

    hhi_all, age_all, ev_all, rows = [], [], [], []
    for value in values:
        bp = build_params(param_name, subdict, value, seeds)
        outputs = parallel_run_multi_seed(params_list_with_seed(bp))

        # Trim burn-in here, once, so every saved array shares one time origin:
        # index 0 is the end of burn-in (Jan 2001).
        hhi = np.asarray(outputs["history_market_concentration"], dtype=float)[:, burn_in:]
        age = np.asarray(outputs["history_mean_car_age_fleet"], dtype=float)[:, burn_in:]
        ev = np.asarray(outputs["history_prop_EV"], dtype=float)[:, burn_in:]

        hhi_all.append(hhi)
        age_all.append(age)
        ev_all.append(ev)

        # Dropped explicitly, not left to rebinding on the next iteration:
        # `outputs` holds every MULTI_SEED_ARRAY_KEYS series plus the cars_on_sale
        # object snapshot for every seed, and plain rebinding would build the next
        # value's dict while the previous one is still referenced, so the parent
        # would peak at two of them on top of the live workers.
        del outputs

        row = {
            param_name: float(value),
            "hhi": final_year_mean(hhi),
            "age_yr": final_year_mean(age) / 12,
            "ev_2023": final_year_mean(ev),
        }
        rows.append(row)
        print(f"{row[param_name]:>12.3e}{row['hhi']:>8.3f}{row['age_yr']:>9.1f}"
              f"{row['ev_2023']:>10.3f}", flush=True)

    save_object(np.asarray(hhi_all), root + "/Data", "data_array_hhi")
    save_object(np.asarray(age_all), root + "/Data", "data_array_age_fleet")
    save_object(np.asarray(ev_all), root + "/Data", "data_array_ev_prop")
    save_object(values, root + "/Data", "param_list")
    save_object(param_name, root + "/Data", "param_name")
    save_object(param_label, root + "/Data", "param_label")
    save_object(base_params, root + "/Data", "base_params")
    save_object(rows, root + "/Data", "summary")

    in_band = [r for r in rows
               if TARGET_HHI[0] <= r["hhi"] <= TARGET_HHI[1]
               and TARGET_FLEET_AGE_YEARS[0] <= r["age_yr"] <= TARGET_FLEET_AGE_YEARS[1]]
    print(f"\n{param_name} values inside BOTH bands: "
          + (", ".join(f"{r[param_name]:.3e}" for r in in_band) if in_band else "none"))
    print(f"wrote {root}")
    return root


def run_and_plot(param_name, subdict, values, param_label, seeds, precheck=None):
    """
    run_sweep, then the figures.

    Plotted in-process so a SLURM job leaves finished PNGs behind; MPLBACKEND=Agg
    in the submit scripts keeps it headless.
    """
    fileName = run_sweep(param_name, subdict, values, param_label, seeds,
                         precheck=precheck)
    from package.plotting_data.sweep_hhi_age_plot import main as plot_main
    plot_main(fileName)
    return fileName
