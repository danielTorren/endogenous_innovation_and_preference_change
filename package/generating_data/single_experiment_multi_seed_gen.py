from copy import deepcopy
import json
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    root = "multi_seed_single"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = params_list_with_seed(base_params)
    
    print("TOTAL RUNS: ", len(params_list))

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    (
        history_driving_emissions_arr,#Emmissions flow
        history_production_emissions_arr,
        history_total_emissions_arr,#Emmissions flow
        history_prop_EV_arr, 
        history_car_age_arr, 
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
        history_mean_profit_margins_EV
    ) = parallel_run_multi_seed(
        params_list
    )

    createFolder(fileName)
    
    save_object(history_driving_emissions_arr, fileName + "/Data", "history_driving_emissions_arr")
    save_object(history_production_emissions_arr, fileName + "/Data", "history_production_emissions_arr")
    save_object(history_total_emissions_arr, fileName + "/Data", "history_total_emissions_arr")
    save_object(history_prop_EV_arr, fileName + "/Data", "history_prop_EV_arr")
    #save_object(history_car_age_arr, fileName + "/Data", "history_car_age_arr")
    save_object(history_mean_price_ICE_EV_arr, fileName + "/Data", "history_mean_price_ICE_EV_arr")
    save_object(history_median_price_ICE_EV_arr, fileName + "/Data", "history_median_price_ICE_EV_arr")
    save_object(history_total_utility_arr, fileName + "/Data", "history_total_utility_arr")
    save_object(history_market_concentration_arr, fileName + "/Data", "history_market_concentration_arr")
    save_object(history_total_profit_arr, fileName + "/Data", "history_total_profit_arr")
    #save_object(history_quality_ICE, fileName + "/Data", "history_quality_ICE")
    #save_object(history_quality_EV, fileName + "/Data", "history_quality_EV")
    #save_object(history_efficiency_ICE, fileName + "/Data", "history_efficiency_ICE")
    #save_object(history_efficiency_EV, fileName + "/Data", "history_efficiency_EV")
    #save_object(history_production_cost_ICE, fileName + "/Data", "history_production_cost_ICE")
    #save_object(history_production_cost_EV, fileName + "/Data", "history_production_cost_EV")
    #save_object(history_mean_profit_margins_ICE, fileName + "/Data", "history_mean_profit_margins_ICE")
    #save_object(history_mean_profit_margins_EV, fileName + "/Data", "history_mean_profit_margins_EV")
    #save_object(history_distance_individual_ICE, fileName + "/Data", "history_distance_individual_ICE")
    #save_object(history_distance_individual_EV, fileName + "/Data", "history_distance_individual_EV")
    save_object(base_params, fileName + "/Data", "base_params")

    print(fileName)
    

if __name__ == "__main__":
    main(BASE_PARAMS_LOAD="package/constants/base_params_multi_seed.json")