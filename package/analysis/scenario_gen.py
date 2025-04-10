from package.resources.run import load_in_controller
from package.analysis.endogenous_policy_intensity_single_gen import set_up_calibration_runs
from package.resources.utility import (
    save_object
)
from joblib import Parallel, delayed, load
import multiprocessing
import numpy as np
import shutil  # Cleanup
from pathlib import Path  # Path handling
from copy import deepcopy
import json
from package.analysis.low_policy_intensity_gen import single_policy_with_seeds

def produce_param_list(params: dict, *property_dicts) -> list[dict]:
    """
    Create parameter combinations for any number of property dictionaries.
    """
    def recursive_combinations(param_template, properties_left, current_values=()):
        if not properties_left:
            return [(deepcopy(param_template), current_values)]
        
        prop_dict = properties_left[0]
        remaining = properties_left[1:]
        results = []
        
        for value in prop_dict["property_list"]:
            updated_params = deepcopy(param_template)
            updated_params[prop_dict["subdict"]][prop_dict["property_varied"]] = value
            new_values = current_values + (value,)
            
            if not remaining:
                results.append((updated_params, new_values))
            else:
                results.extend(recursive_combinations(updated_params, remaining, new_values))
        
        return results
    
    combinations = recursive_combinations(params, property_dicts)
    params_list, intensity_tuples = zip(*combinations)
    
    return list(params_list), list(intensity_tuples)

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    *property_dicts
    ):
    
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    params_list, intensity_list = produce_param_list(base_params, *property_dicts)
    print(f"Parameter combinations: {len(params_list)}")
    print(f"Intensity list: {intensity_list}")

    print(f"TOTAL RUNS: {len(params_list) * base_params['seed_repetitions']}")

    ###############################################################################################################################

    controller_files, base_params, root_folder = set_up_calibration_runs(base_params, "scenario_gen")
    print("DONE calibration")
    
    ##############################################################################################################################
    outputs = {}

    for i, params_scenario in enumerate(params_list):
        (
            history_driving_emissions_arr,#Emmissions flow
            history_production_emissions_arr,
            history_total_emissions_arr,#Emmissions flow
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
            history_past_new_bought_vehicles_prop_ev,
            history_policy_net_cost,
            history_total_utility_bottom,
            history_ev_adoption_rate_bottom
        ) = single_policy_with_seeds(params_scenario, controller_files)

        outputs[intensity_list[i]] = {
            "history_driving_emissions": history_driving_emissions_arr,
            "history_production_emissions": history_production_emissions_arr,
            "history_total_emissions": history_total_emissions_arr,
            "history_prop_EV": history_prop_EV_arr,
            "history_total_utility": history_total_utility_arr,
            "history_market_concentration": history_market_concentration_arr,
            "history_total_profit": history_total_profit_arr,
            "history_mean_profit_margins_ICE": history_mean_profit_margins_ICE,
            "history_mean_profit_margins_EV": history_mean_profit_margins_EV,
            "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
            "history_policy_net_cost": history_policy_net_cost,
            "history_mean_car_age": history_mean_car_age,
            "history_lower_percentile_price_ICE_EV_arr": history_lower_percentile_price_ICE_EV_arr,
            "history_upper_percentile_price_ICE_EV_arr": history_upper_percentile_price_ICE_EV_arr,
            "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
            "history_median_price_ICE_EV_arr": history_median_price_ICE_EV_arr,
            "history_past_new_bought_vehicles_prop_ev": history_past_new_bought_vehicles_prop_ev,
            "history_total_utility_bottom": history_total_utility_bottom,
            "history_ev_adoption_rate_bottom": history_ev_adoption_rate_bottom
        }

    # Save all property dictionaries
    save_object(outputs, root_folder + "/Data", "outputs")
    save_object(base_params, root_folder + "/Data", "base_params")
    
    for i, prop_dict in enumerate(property_dicts, 1):
        save_object(prop_dict, root_folder + "/Data", f"property_dict_{i}")

    #######################################################################################################
    #DELETE CALIBRATION RUNS
    shutil.rmtree(Path(root_folder) / "Calibration_runs", ignore_errors=True)

if __name__ == "__main__":
    # Define the property dictionaries
    property_dict_1 = {
        "subdict": "parameters_scenarios",
        "property_varied": "Gas_price", 
        "property_list": [1, 1.5]
    }
    
    property_dict_2 = {
        "subdict": "parameters_scenarios",
        "property_varied": "Electricity_price", 
        "property_list": [0.5, 1, 1.5]
    }
    
    property_dict_3 = {
        "subdict": "parameters_scenarios",
        "property_varied": "Grid_emissions_intensity", 
        "property_list": [0.1, 0.5]
    }
    
    # Call main with the property dictionaries as separate arguments
    main(
        "package/constants/base_params_scenarios.json",
        property_dict_1,
        property_dict_2,
        property_dict_3
    )