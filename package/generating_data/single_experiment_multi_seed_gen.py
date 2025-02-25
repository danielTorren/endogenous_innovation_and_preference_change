from copy import deepcopy
import json
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

def params_list_with_seed(base_params):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    base_params_list = []
    seed_repetitions = base_params["seed_repetitions"]

    for seed in range(1, seed_repetitions + 1):
        base_params_copy = deepcopy(base_params)
        # VARY ALL THE SEEDS
        base_params_copy["seeds"]["init_tech_seed"] = seed + seed_repetitions
        base_params_copy["seeds"]["landscape_seed_ICE"] = seed + 2 * seed_repetitions
        base_params_copy["seeds"]["social_network_seed"] = seed + 3 * seed_repetitions
        base_params_copy["seeds"]["network_structure_seed"] = seed + 4 * seed_repetitions
        base_params_copy["seeds"]["init_vals_environmental_seed"] = seed + 5 * seed_repetitions
        base_params_copy["seeds"]["init_vals_innovative_seed"] = seed + 6 * seed_repetitions
        base_params_copy["seeds"]["init_vals_price_seed"] = seed + 7 * seed_repetitions
        base_params_copy["seeds"]["innovation_seed"] = seed + 8 * seed_repetitions
        base_params_copy["seeds"]["landscape_seed_EV"] = seed + 9 * seed_repetitions
        base_params_copy["seeds"]["choice_seed"] = seed + 10 * seed_repetitions
        base_params_copy["seeds"]["remove_seed"] = seed + 11 * seed_repetitions
        base_params_copy["seeds"]["init_vals_poisson_seed"] = seed + 12 * seed_repetitions
        base_params_copy["seeds"]["init_vals_range_seed"] = seed + 13 * seed_repetitions
       
        base_params_list.append( base_params_copy)
    
    return base_params_list

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
        history_total_emissions_arr,
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
        history_profit_margins_ICE,
        history_profit_margins_EV
    ) = parallel_run_multi_seed(
        params_list
    )

    createFolder(fileName)
    
    save_object(history_driving_emissions_arr, fileName + "/Data", "history_driving_emissions_arr")
    save_object(history_production_emissions_arr, fileName + "/Data", "history_production_emissions_arr")
    save_object(history_total_emissions_arr, fileName + "/Data", "history_total_emissions_arr")
    save_object(history_prop_EV_arr, fileName + "/Data", "history_prop_EV_arr")
    save_object(history_car_age_arr, fileName + "/Data", "history_car_age_arr")
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
    #save_object(history_profit_margins_ICE, fileName + "/Data", "history_profit_margins_ICE")
    #save_object(history_profit_margins_EV, fileName + "/Data", "history_profit_margins_EV")
    #save_object(history_distance_individual_ICE, fileName + "/Data", "history_distance_individual_ICE")
    #save_object(history_distance_individual_EV, fileName + "/Data", "history_distance_individual_EV")
    save_object(base_params, fileName + "/Data", "base_params")

    print(fileName)
    

if __name__ == "__main__":
    main(BASE_PARAMS_LOAD="package/constants/base_params_multi_seed.json")