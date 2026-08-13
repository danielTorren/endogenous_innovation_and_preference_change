import json
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime,
    params_list_with_seed
)
from package.plotting_data.calibration_plot import main as plotting_main

# What gets pickled out of the run. The rest of the run output (per-vehicle
# quality, efficiency and production cost histories) is not plotted, so it is
# dropped rather than written to disk.
SAVED_KEYS = (
    "history_driving_emissions",
    "history_production_emissions",
    "history_total_emissions",
    "history_prop_EV",
    "history_lower_percentile_price_ICE_EV",
    "history_upper_percentile_price_ICE_EV",
    "history_mean_price_ICE_EV",
    "history_median_price_ICE_EV",
    "history_total_utility",
    "history_market_concentration",
    "history_total_profit",
    "history_mean_profit_margins_ICE",
    "history_mean_profit_margins_EV",
    "history_mean_car_age",
    "history_past_new_bought_vehicles_prop_ev",
    "cars_on_sale",
    # Feed plot_calibration_targets: the four target variables in one figure.
    "history_mean_car_age_fleet",
    "history_new_car_price_quantiles",
    "history_used_car_price_quantiles",
    "history_used_stock_quality_spread",
)


def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    ) -> str:

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    root = "calibration_gen"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = params_list_with_seed(base_params)

    print("TOTAL RUNS: ", len(params_list))

    run_outputs = parallel_run_multi_seed(params_list)

    createFolder(fileName)

    outputs = {key: run_outputs[key] for key in SAVED_KEYS}

    save_object(outputs, fileName + "/Data", "outputs")
    save_object(base_params, fileName + "/Data", "base_params")

    print(fileName)
    return fileName

if __name__ == "__main__":
    #main(BASE_PARAMS_LOAD="package/constants/base_params_multi_seed.json")
    fileName = main(BASE_PARAMS_LOAD="package/constants/base_params_calibration.json")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1
    print("fileName",fileName)
    if RUN_PLOT:
        plotting_main(fileName = fileName)
