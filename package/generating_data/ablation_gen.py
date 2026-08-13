"""
Leave-one-out ablation over the landscape-dispersion changes.

Six simultaneous changes (both rho vectors, three stretch factors, min_Cost) took
EV uptake from 0.046 to 0.78 against a target of 0.038, and pushed HHI and car age
out of their ranges, while improving price dispersion. This script removes one
change at a time from the full configuration so the damage can be attributed
rather than guessed at.

The knobs are not independent -- they are three coupled pairs, so the pairs are
removed together:

  quality_stretch  stretch_Quality              (the Q=1.0 clip that lets EVs
                                                 reach the same quality ceiling
                                                 as ICE, erasing ICE's 180-month
                                                 research head start)
  battery_pair     stretch_Battery + rho[3]     (cheap small-pack EVs, which did
                                                 not exist before)
  cost_pair        stretch_Cost + rho[1] + min_Cost
                                                (the cheap end of the market, and
                                                 the quality-cost ladder that
                                                 makes the expensive end worth
                                                 buying -- neither works alone)
  efficiency_corr  rho[2]                        (measured at only 0.046 of the
                                                 overshoot, kept as a control)

Whichever row brings EV uptake back near 0.04 identifies the culprit; the rest
can be kept.

Consideration-set sampling is deliberately NOT enabled here. It is a separate
change and would confound the attribution. The used-share and P(buy) columns are
still reported, because they come for free and show what the landscape changes did
to the transaction mix.

Usage:
    python -m package.generating_data.ablation_gen                 # all configs
    python -m package.generating_data.ablation_gen full cost_pair  # named subset
    python -m package.generating_data.ablation_gen --seeds 4       # fewer seeds
"""
import json
import sys

import numpy as np

from package.generating_data.calibration_gen import SAVED_KEYS
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder,
    load_object,
    params_list_with_seed,
    produce_name_datetime,
    save_object,
)

BASE_PARAMS_LOAD = "package/constants/base_params_calibration.json"
SEEDS = 12

# The full configuration under test, i.e. what run 17_08_28 used. max_Quality is
# left at 1: raising it is provably a no-op, since with min_Quality = 0 the whole
# quality mapping is absorbed by the beta calibration, and the clip threshold sits
# in m-space and does not involve the bounds.
FULL = {
    "ICE": {"rho": [1, 0.5, 0.25], "stretch_Quality": 3.0, "stretch_Efficiency": 1.0,
            "stretch_Cost": 3.0, "min_Cost": 15000, "max_Quality": 1},
    "EV":  {"rho": [1, 0.5, 0.25, 0.75], "stretch_Efficiency": 1.0,
            "stretch_Cost": 3.0, "stretch_Battery_size": 3.0},
}
# The known-good starting point, which hit EV uptake, HHI and car age.
BASE = {
    "ICE": {"rho": [1, 0, 0], "stretch_Quality": 1.0, "stretch_Efficiency": 1.0,
            "stretch_Cost": 1.0, "min_Cost": 5000, "max_Quality": 1},
    "EV":  {"rho": [1, 0, 0, 0], "stretch_Efficiency": 1.0,
            "stretch_Cost": 1.0, "stretch_Battery_size": 1.0},
}


def _config(name):
    """Return (ICE overrides, EV overrides) for one ablation row."""
    ice, ev = dict(FULL["ICE"]), dict(FULL["EV"])
    ice["rho"], ev["rho"] = list(ice["rho"]), list(ev["rho"])

    if name == "baseline":
        return dict(BASE["ICE"]), dict(BASE["EV"])
    if name == "full":
        pass
    elif name == "no_quality_stretch":
        ice["stretch_Quality"] = 1.0
    elif name == "no_battery_pair":
        ev["stretch_Battery_size"] = 1.0
        ev["rho"][3] = 0
    elif name == "no_cost_pair":
        ice["stretch_Cost"] = ev["stretch_Cost"] = 1.0
        ice["rho"][1] = ev["rho"][1] = 0
        ice["min_Cost"] = 5000
    elif name == "no_efficiency_corr":
        ice["rho"][2] = ev["rho"][2] = 0
    else:
        raise ValueError(f"unknown ablation config: {name}")
    return ice, ev


CONFIGS = ["baseline", "full", "no_quality_stretch", "no_battery_pair",
           "no_cost_pair", "no_efficiency_corr"]


def build_params(name, seeds):
    with open(BASE_PARAMS_LOAD) as f:
        bp = json.load(f)
    ice, ev = _config(name)
    bp["parameters_ICE"].update(ice)
    bp["parameters_EV"].update(ev)
    bp["seed_repetitions"] = seeds
    # Sampling stays off so the attribution is not confounded.
    bp["parameters_social_network"].pop("num_new_considered", None)
    bp["parameters_social_network"].pop("num_second_hand_considered", None)
    return bp


def summarise(name, bp, outputs):
    """The six calibration-target metrics, final-year means over seeds."""
    b = bp["duration_burn_in"]
    A = lambda k: np.asarray(outputs[k], dtype=float)[:, b:]
    fy = lambda x: float(np.nanmean(x[..., -12:]))

    nq = np.asarray(outputs["history_new_car_price_quantiles"], dtype=float)[:, b:, :]
    uq = np.asarray(outputs["history_used_car_price_quantiles"], dtype=float)[:, b:, :]
    counts = np.asarray(outputs["history_purchase_counts"], dtype=float)[:, b:, :]
    n, u, opp = (counts[:, -12:, i].sum(axis=1) for i in range(3))

    return {
        "config": name,
        "ev_2023": fy(A("history_prop_EV")),
        "hhi": fy(A("history_market_concentration")),
        "age_yr": fy(A("history_mean_car_age_fleet")) / 12,
        "price_iqr_x": fy(nq[:, :, 2] - nq[:, :, 0]) / (57784.66 - 32359.41),
        "qual_price_ratio": fy(A("history_used_stock_quality_spread"))
                            / fy(uq[:, :, 2] - uq[:, :, 0]),
        "used_share": float(np.mean(u / (n + u))),
        "prob_buy": float(np.mean((n + u) / opp)),
    }


HEADER = (f"{'config':<20}{'EV 2023':>9}{'HHI':>7}{'age yr':>8}"
          f"{'priceIQR':>10}{'q/p':>6}{'usedsh':>8}{'P(buy)':>8}")
TARGETS = ("TARGET", 0.038, "0.11-0.18", "10-12", "1.00x", "~1", "0.67", "0.16")


def print_row(r):
    print(f"{r['config']:<20}{r['ev_2023']:>9.3f}{r['hhi']:>7.3f}{r['age_yr']:>8.1f}"
          f"{r['price_iqr_x']:>9.2f}x{r['qual_price_ratio']:>6.2f}"
          f"{r['used_share']:>8.3f}{r['prob_buy']:>8.3f}", flush=True)


def main(configs=None, seeds=SEEDS):
    configs = configs or CONFIGS
    root = produce_name_datetime("ablation_gen")
    createFolder(root)
    print(f"fileName: {root}\nseeds per config: {seeds}\n")
    print(HEADER)
    print(f"{TARGETS[0]:<20}{TARGETS[1]:>9.3f}{TARGETS[2]:>7}{TARGETS[3]:>8}"
          f"{TARGETS[4]:>10}{TARGETS[5]:>6}{TARGETS[6]:>8}{TARGETS[7]:>8}")
    print("-" * len(HEADER))

    rows = []
    for name in configs:
        bp = build_params(name, seeds)
        outputs = parallel_run_multi_seed(params_list_with_seed(bp))
        saved = {k: outputs[k] for k in SAVED_KEYS}
        # Saved per config so calibration_plot can be pointed at any single row.
        sub = f"{root}/{name}"
        createFolder(sub)
        save_object(saved, sub + "/Data", "outputs")
        save_object(bp, sub + "/Data", "base_params")
        row = summarise(name, bp, saved)
        rows.append(row)
        print_row(row)

    save_object(rows, root + "/Data", "ablation_summary")
    print(f"\nwrote {root}")
    print("plot any single row with:")
    print(f"  python -c \"from package.plotting_data.calibration_plot import main;"
          f" main('{root}/full')\"")
    return root


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    seeds = SEEDS
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = int(args[i + 1])
        del args[i:i + 2]
    main(configs=args or None, seeds=seeds)
