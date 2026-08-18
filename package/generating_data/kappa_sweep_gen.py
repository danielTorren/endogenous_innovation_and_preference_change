"""
Sweep parameters_vehicle_user.kappa and record HHI and fleet car age.

kappa is the logit noise scale on the vehicle choice. It is the single knob that
moves both calibration targets at once: raising it flattens the choice
probabilities, which spreads sales across firms (lower HHI) and increases churn
in the second-hand market (lower mean fleet age). The two targets therefore
cannot be tuned independently through kappa, and this sweep is what shows where
the two admissible bands overlap, if they overlap at all.

The range [1e-4, 3e-4] brackets the calibrated value in
base_params_calibration.json (kappa = 2.25e-4).

Outputs one results/kappa_sweep_<timestamp> folder holding
  Data/kappa_list, Data/base_params, Data/summary
  Data/data_array_hhi          (n_kappa, seeds, T)   unit-share new-car HHI
  Data/data_array_age_fleet    (n_kappa, seeds, T)   whole-fleet mean age, MONTHS
  Data/data_array_ev_prop      (n_kappa, seeds, T)
and the figures written by package/plotting_data/kappa_sweep_plot.py.

The two series only exist when save_timeseries_data_state is 1 --
history_market_concentration is appended from
Firm_Manager.save_timeseries_data_firm_manager(), which is gated on it. The base
JSON already sets it; do not switch it off here.

Usage:
    python -m package.generating_data.kappa_sweep_gen
    python -m package.generating_data.kappa_sweep_gen --seeds 4      # quick check
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

# 10 values, endpoints included, as asked for.
KAPPA_LIST = np.linspace(1e-4, 3e-4, 10)

# 16, not the 64 in the base JSON. Both target series are seed-noisy but the
# kappa effect is monotone and large next to that noise, so 16 seeds resolve the
# trend at a quarter of the cost of 10 x 64 = 640 runs. Raise it with --seeds
# once the trend is worth a publication figure.
SEEDS = 16

# Reported alongside each row, from package/plotting_data/calibration_plot.py.
TARGET_HHI = (0.11, 0.18)
TARGET_FLEET_AGE_YEARS = (10, 12)


def build_params(kappa, seeds):
    with open(BASE_PARAMS_LOAD) as f:
        bp = json.load(f)
    bp["parameters_vehicle_user"]["kappa"] = float(kappa)
    bp["seed_repetitions"] = seeds
    return bp


def final_year_mean(arr_2d):
    """Mean over seeds of the last 12 recorded steps, i.e. the 2023 calendar year."""
    return float(np.nanmean(np.asarray(arr_2d, dtype=float)[:, -12:]))


HEADER = f"{'kappa':>10}{'HHI':>8}{'age yr':>9}{'EV 2023':>10}"


def print_row(r):
    print(f"{r['kappa']:>10.3e}{r['hhi']:>8.3f}{r['age_yr']:>9.1f}"
          f"{r['ev_2023']:>10.3f}", flush=True)


def main(seeds=SEEDS, kappa_list=KAPPA_LIST):
    kappa_list = np.asarray(kappa_list, dtype=float)
    root = produce_name_datetime("kappa_sweep")
    createFolder(root)

    base_params = build_params(kappa_list[0], seeds)
    burn_in = base_params["duration_burn_in"]

    print(f"fileName: {root}")
    print(f"kappa values: {len(kappa_list)}, seeds each: {seeds}, "
          f"total runs: {len(kappa_list) * seeds}\n")
    print(HEADER)
    print(f"{'TARGET':>10}{'.11-.18':>8}{'10-12':>9}{'0.038':>10}")
    print("-" * len(HEADER))

    hhi_all, age_all, ev_all, rows = [], [], [], []
    for kappa in kappa_list:
        bp = build_params(kappa, seeds)
        outputs = parallel_run_multi_seed(params_list_with_seed(bp))

        # Trim burn-in here, once, so every saved array shares one time origin:
        # index 0 is the end of burn-in (Jan 2001).
        hhi = np.asarray(outputs["history_market_concentration"], dtype=float)[:, burn_in:]
        age = np.asarray(outputs["history_mean_car_age_fleet"], dtype=float)[:, burn_in:]
        ev = np.asarray(outputs["history_prop_EV"], dtype=float)[:, burn_in:]

        hhi_all.append(hhi)
        age_all.append(age)
        ev_all.append(ev)

        row = {
            "kappa": float(kappa),
            "hhi": final_year_mean(hhi),
            "age_yr": final_year_mean(age) / 12,
            "ev_2023": final_year_mean(ev),
        }
        rows.append(row)
        print_row(row)

    save_object(np.asarray(hhi_all), root + "/Data", "data_array_hhi")
    save_object(np.asarray(age_all), root + "/Data", "data_array_age_fleet")
    save_object(np.asarray(ev_all), root + "/Data", "data_array_ev_prop")
    save_object(kappa_list, root + "/Data", "kappa_list")
    save_object(base_params, root + "/Data", "base_params")
    save_object(rows, root + "/Data", "summary")

    in_band = [r for r in rows
               if TARGET_HHI[0] <= r["hhi"] <= TARGET_HHI[1]
               and TARGET_FLEET_AGE_YEARS[0] <= r["age_yr"] <= TARGET_FLEET_AGE_YEARS[1]]
    print("\nkappa values inside BOTH bands: "
          + (", ".join(f"{r['kappa']:.3e}" for r in in_band) if in_band else "none"))
    print(f"wrote {root}")
    return root


if __name__ == "__main__":
    args = sys.argv[1:]
    seeds = SEEDS
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = int(args[i + 1])
        del args[i:i + 2]

    fileName = main(seeds=seeds)

    # Plotted here so the SLURM job leaves finished figures behind; MPLBACKEND=Agg
    # in the submit script keeps it headless.
    from package.plotting_data.kappa_sweep_plot import main as plot_main
    plot_main(fileName)
