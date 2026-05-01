from copy import deepcopy
import json
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)
from package.plotting_data.multi_seed_plot import main as plotting_main

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    ) -> str: 

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    root = "multi_seed"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = params_list_with_seed(base_params)
    print("TOTAL RUNS: ", len(params_list))

    (
        history_driving_emissions_arr,
        history_production_emissions_arr,
        history_total_emissions_arr,
        history_prop_EV_arr, 
        history_car_age_arr, 
        history_lower_percentile_price_ICE_EV_arr,
        history_upper_percentile_price_ICE_EV_arr,
        history_mean_price_ICE_EV_arr,
        history_median_price_ICE_EV_arr, 
        history_total_utility_arr, 
        history_market_concentration_arr,
        history_total_profit_arr, 
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        history_mean_car_age,
        history_past_new_bought_vehicles_prop_ev
    ) = parallel_run_multi_seed(params_list)

    createFolder(fileName)

    outputs = {
        "history_driving_emissions": history_driving_emissions_arr,
        "history_production_emissions": history_production_emissions_arr,
        "history_total_emissions": history_total_emissions_arr,
        "history_prop_EV": history_prop_EV_arr,
        "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
        "history_median_price_ICE_EV_arr": history_median_price_ICE_EV_arr,
        "history_lower_percentile_price_ICE_EV_arr": history_lower_percentile_price_ICE_EV_arr,
        "history_upper_percentile_price_ICE_EV_arr": history_upper_percentile_price_ICE_EV_arr,
        "history_total_utility": history_total_utility_arr,
        "history_market_concentration": history_market_concentration_arr,
        "history_total_profit": history_total_profit_arr,
        "history_mean_profit_margins_ICE": history_mean_profit_margins_ICE,
        "history_mean_profit_margins_EV": history_mean_profit_margins_EV,
        "history_mean_car_age": history_mean_car_age,
        "history_past_new_bought_vehicles_prop_ev": history_past_new_bought_vehicles_prop_ev,
    }

    save_object(outputs, fileName + "/Data", "outputs")
    save_object(base_params, fileName + "/Data", "base_params")

    print(fileName)
    return fileName

if __name__ == "__main__":
    fileName = main(BASE_PARAMS_LOAD="package/constants/base_params_multi_seed.json")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1
    print("fileName",fileName)
    if RUN_PLOT:
        plotting_main(fileName = fileName)