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

def single_policy_simulation(params, controller_file):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    controller = load(controller_file)  # Load fresh controller
    data = load_in_controller(controller, params)
    return (
        data.social_network.history_driving_emissions,#Emmissions flow
        data.social_network.history_production_emissions,#Emmissions flow
        data.social_network.history_total_emissions,#Emmissions flow
        data.social_network.history_prop_EV, 
        data.social_network.history_car_age, 
        data.social_network.history_lower_percentile_price_ICE_EV,
        data.social_network.history_upper_percentile_price_ICE_EV,
        data.social_network.history_mean_price_ICE_EV,
        data.social_network.history_median_price_ICE_EV, 
        data.social_network.history_total_utility,
        data.firm_manager.history_market_concentration,
        data.firm_manager.history_total_profit, 
        data.social_network.history_quality_ICE, 
        data.social_network.history_quality_EV, 
        data.social_network.history_efficiency_ICE, 
        data.social_network.history_efficiency_EV, 
        data.social_network.history_production_cost_ICE, 
        data.social_network.history_production_cost_EV, 
        data.firm_manager.history_mean_profit_margins_ICE,
        data.firm_manager.history_mean_profit_margins_EV,
        data.history_policy_net_cost
    )

def single_policy_with_seeds(params, controller_files):
    """
    Run policy scenarios using pre-saved controllers for consistency.
    """
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(single_policy_simulation)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )
    
    (
        history_driving_emissions_arr,#Emmissions flow
        history_production_emissions_arr,
        history_total_emissions,#Emmissions flow
        history_prop_EV, 
        history_car_age, 
        history_lower_percentile_price_ICE_EV,
        history_upper_percentile_price_ICE_EV,
        history_mean_price_ICE_EV,
        history_median_price_ICE_EV, 
        history_total_utility, 
        history_market_concentration,
        history_total_profit, 
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        history_policy_net_cost
    ) = zip(*res)

        # Return results as arrays where applicable
    return (
        np.asarray(history_driving_emissions_arr),#Emmissions flow
        np.asarray(history_production_emissions_arr),
        np.asarray(history_total_emissions),#Emmissions flow
        np.asarray(history_prop_EV), 
        np.asarray(history_car_age), 
        np.asarray(history_lower_percentile_price_ICE_EV),
        np.asarray(history_upper_percentile_price_ICE_EV),
        np.asarray(history_mean_price_ICE_EV),
        np.asarray(history_median_price_ICE_EV), 
        np.asarray(history_total_utility), 
        np.asarray(history_market_concentration),
        np.asarray(history_total_profit),
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        np.asarray(history_policy_net_cost)
    )

def produce_param_list(params: dict, property_dict_1, property_dict_2) -> list[dict]:
    params_list = []
    intensity_list = []
    for i in property_dict_1["property_list"]:
        print("i", i)
        for j in  property_dict_2["property_list"]:
            print("j",j)
            params_updated = deepcopy(params)
            params_updated[property_dict_1["subdict"]][property_dict_1["property_varied"]] = i
            params_updated[property_dict_2["subdict"]][property_dict_2["property_varied"]] = j
            params_list.append(params_updated)
            intensity_list.append((i,j))
    return params_list, intensity_list

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    property_dict_1 = {"subdict": "parameters_scenarios","property_varied": "Gas_price", "property_list": [0.5, 1]}, 
    property_dict_2 = {"subdict": "parameters_scenarios","property_varied": "Electricity_price", "property_list": [1, 1.5]} 
        ):
    
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    params_list, intensity_list = produce_param_list(base_params, property_dict_1, property_dict_2)
    print(len(params_list))
    print(intensity_list)

    print("TOTAL RUNS", len(params_list)*base_params["seed_repetitions"])

    ###############################################################################################################################

    controller_files, base_params, root_folder = set_up_calibration_runs(base_params, "sceanrio_tests_gen")
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
        history_policy_net_cost
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
            "history_policy_net_cost": history_policy_net_cost
        }

    save_object(outputs, root_folder + "/Data", "outputs")
    save_object(base_params, root_folder + "/Data", "base_params")
    save_object(property_dict_1 , root_folder + "/Data", "property_dict_1")
    save_object(property_dict_2 , root_folder + "/Data", "property_dict_2")

    #######################################################################################################
    #DELETE CALIBRATION RUNS
    shutil.rmtree(Path(root_folder) / "Calibration_runs", ignore_errors=True)

if __name__ == "__main__":
    main(
    BASE_PARAMS_LOAD="package/constants/base_params_scenarios_2050_BAU.json",
    property_dict_1 = {"subdict": "parameters_scenarios","property_varied": "Gas_price", "property_list": [0.5, 1, 1.5] }, 
    property_dict_2 = {"subdict": "parameters_scenarios","property_varied": "Electricity_price", "property_list": [0.5, 1, 1.5] }, 
    property_dict_3 = {"subdict": "parameters_scenarios","property_varied": "Grid_emissions_intensity", "property_list": [0.1,0.5, 1] } 
    )