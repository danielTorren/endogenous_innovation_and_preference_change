"""
Grid search over b_chi for calibration.

Sweeps parameters_social_network.b_chi across the values in
constants/vary_single_b_innov.json, using constants/base_params_vary_single.json
as the base, and scores every value against the California EV stock share.

Unlike vary_single_param_gen this records EV uptake ONLY (ev_prop_parallel_run,
not ev_prop_price_emissions_parallel_run): prices and margins play no part in
picking b_chi, and dropping them keeps the per-worker memory down so more seeds
fit in one job.

Two outputs:
  1. Plots/b_chi_grid_search.png -- ONE axes, one mean line per b_chi value with
     its 95% CI, against the observed series. This is the figure to read.
  2. Data/rmse_of_mean + Data/rmse_per_seed -- the grid-search score, printed as
     a ranked table at the end of the run.

Scoring follows the NN calibration convention exactly: the model is sampled at
APRIL of each observed year (STOCK_MONTH_OFFSET on _year_start_index), so the
scores here are comparable with the EV-stock block of that calibration's x.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from package.resources.run import ev_prop_parallel_run
from package.resources.utility import (
    createFolder,
    load_object,
    produce_name_datetime,
    params_list_with_seed,
    save_object,
)

# First year of calibration_data_output["EV Prop"], set by the 2010-2023 filter
# in calibration/calibration_data_outputs.py. The array carries no year labels,
# so this is the only thing tying its entries to model time steps.
DATA_START_YEAR = 2010

# Copied from calibration/NN_multi_round_calibration_multi_gen.py rather than
# imported: that module imports torch and sbi at module scope, which this job has
# no use for. Keep the two in step if the calibration convention changes.
STOCK_MONTH_OFFSET = 3   # APRIL index, the month the EV stock (population) data refers to


def _year_start_index(year, base_params):
    """Index of January of `year` in a monthly series that starts at the burn-in."""
    return (year - 2001) * 12 + base_params["duration_burn_in"]


def produce_param_list(params: dict, property_list: list, subdict: str, property: str) -> list[dict]:
    params_list = []

    for i in property_list:
        params[subdict][property] = i
        seeds_base_params_list = params_list_with_seed(params)
        params_list.extend(seeds_base_params_list)
    return params_list


def observed_step_indices(base_params: dict, num_obs: int, series_length: int):
    """
    Model time-step index of each observed data point, and the observations it
    can actually score.

    Returns (step_indices, num_used). Years whose April falls past the end of the
    run are dropped rather than allowed to raise, so a base_params with a shorter
    duration_calibration still scores on the years it does cover.
    """
    steps = np.array([
        _year_start_index(DATA_START_YEAR + k, base_params) + STOCK_MONTH_OFFSET
        for k in range(num_obs)
    ])
    num_used = int(np.sum(steps < series_length))
    return steps[:num_used], num_used


def score_grid(data_array, real_data, base_params):
    """
    RMSE of every b_chi value against the observed EV stock share.

    rmse_of_mean is the primary score: calibration matches the seed-AVERAGED
    series, so averaging first and scoring once is the like-for-like comparison.
    rmse_per_seed is kept alongside it to show how much of any gap is seed noise
    -- a value whose rmse_of_mean is good only because seeds cancel out is not
    actually a good fit.
    """
    series_length = data_array.shape[2]
    steps, num_used = observed_step_indices(base_params, len(real_data), series_length)

    if num_used < len(real_data):
        print(
            f"WARNING: only {num_used}/{len(real_data)} observed years fit inside a "
            f"{series_length}-step run; scoring on {DATA_START_YEAR}-{DATA_START_YEAR + num_used - 1}"
        )

    real_used = np.asarray(real_data)[:num_used]

    model_at_obs = data_array[:, :, steps]                      # (values, seeds, years)
    rmse_per_seed = np.sqrt(np.mean((model_at_obs - real_used) ** 2, axis=2))
    rmse_of_mean = np.sqrt(np.mean((np.mean(model_at_obs, axis=1) - real_used) ** 2, axis=1))

    return rmse_of_mean, rmse_per_seed, steps, num_used


def plot_grid_search(base_params, data_array, property_list, name_property, real_data,
                     rmse_of_mean, steps, num_used, fileName, dpi=300):
    """One axes, one line per b_chi value, observed series in black."""
    burn_in_step = base_params["duration_burn_in"]
    num_values, num_seeds, series_length = data_array.shape

    # Calendar years on the x axis: the series is monthly and starts at the
    # burn-in, with January 2001 the first post-burn-in step (_year_start_index).
    years = 2001 + (np.arange(series_length) - burn_in_step) / 12.0
    obs_years = 2001 + (steps - burn_in_step) / 12.0

    colors = plt.cm.viridis(np.linspace(0, 1, num_values))
    fig, ax = plt.subplots(figsize=(9, 6))

    best = int(np.argmin(rmse_of_mean))

    for i, (value, color) in enumerate(zip(property_list, colors)):
        mean_data = np.mean(data_array[i], axis=0)
        ci_range = stats.sem(data_array[i], axis=0) * stats.t.ppf(0.975, num_seeds - 1)

        # The best-fitting value is drawn heavier so the figure answers the
        # grid-search question without cross-referencing the printed table.
        is_best = i == best
        ax.plot(
            years[burn_in_step:], mean_data[burn_in_step:],
            color=color, linewidth=3 if is_best else 1.8,
            label=f"{name_property} = {value} (RMSE {rmse_of_mean[i]:.4f})" + (" BEST" if is_best else ""),
        )
        ax.fill_between(
            years[burn_in_step:],
            (mean_data - ci_range)[burn_in_step:],
            (mean_data + ci_range)[burn_in_step:],
            color=color, alpha=0.2,
        )

    ax.plot(obs_years, np.asarray(real_data)[:num_used], color="black", marker="o",
            markersize=4, linestyle="dotted", label="California EV stock share")

    ax.set_xlabel("Year")
    ax.set_ylabel("EV proportion of fleet")
    ax.set_title(f"Grid search over {name_property} ({num_seeds} seeds, mean and 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{fileName}/Plots/b_chi_grid_search.png", dpi=dpi)
    return fig


def main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single.json",
        VARY_LOAD="package/constants/vary_single_b_innov.json",
        SHOW_PLOT=False,
    ) -> str:

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(VARY_LOAD) as f:
        vary_single = json.load(f)

    seed_repetitions = base_params["seed_repetitions"]

    property_varied = vary_single["property_varied"]
    subdict = vary_single["subdict"]
    property_list = vary_single["property_list"]

    root = "b_chi_grid_search"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)
    print("property_varied:", property_varied, "in", subdict)
    print("property_list:", property_list)

    params_list = produce_param_list(base_params, property_list, subdict, property_varied)
    print("TOTAL RUNS: ", len(params_list))

    data_flat_ev_prop, data_flat_price_range = ev_prop_parallel_run(params_list)

    # produce_param_list appends seeds inner, values outer, so this reshape is
    # the inverse of that ordering -- do not reorder the loops there.
    data_array_ev_prop = data_flat_ev_prop.reshape(
        len(property_list), seed_repetitions, len(data_flat_ev_prop[0])
    )
    data_array_price_range = data_flat_price_range.reshape(len(property_list), seed_repetitions)

    real_data = load_object("package/calibration_data", "calibration_data_output")["EV Prop"]

    rmse_of_mean, rmse_per_seed, steps, num_used = score_grid(
        data_array_ev_prop, real_data, base_params
    )

    createFolder(fileName)

    save_object(data_array_ev_prop, fileName + "/Data", "data_array_ev_prop")
    save_object(data_array_price_range, fileName + "/Data", "data_array_price_range")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_single, fileName + "/Data", "vary_single")
    save_object(rmse_of_mean, fileName + "/Data", "rmse_of_mean")
    save_object(rmse_per_seed, fileName + "/Data", "rmse_per_seed")

    plot_grid_search(
        base_params, data_array_ev_prop, property_list, property_varied, real_data,
        rmse_of_mean, steps, num_used, fileName,
    )

    order = np.argsort(rmse_of_mean)
    print(f"\nGRID SEARCH: {property_varied}, scored on {num_used} years from {DATA_START_YEAR}")
    print(f"{property_varied:>10} {'RMSE(mean)':>12} {'mean RMSE/seed':>16} {'sd RMSE/seed':>14}")
    for i in order:
        print(f"{property_list[i]:>10} {rmse_of_mean[i]:>12.5f} "
              f"{np.mean(rmse_per_seed[i]):>16.5f} {np.std(rmse_per_seed[i]):>14.5f}")
    print(f"\nBEST {property_varied}: {property_list[int(order[0])]} "
          f"(RMSE {rmse_of_mean[int(order[0])]:.5f})")

    if SHOW_PLOT:
        plt.show()

    print(fileName)
    return fileName


if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single.json",
        VARY_LOAD="package/constants/vary_single_b_innov.json",
    )
