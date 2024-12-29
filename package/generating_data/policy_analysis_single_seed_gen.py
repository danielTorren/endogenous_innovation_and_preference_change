from copy import deepcopy
import json
from package.resources.run import emissions_parallel_run, ev_prop_parallel_run, generate_data, policy_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

def base_params_and_controller_list_with_policy(base_params, policy_vary):
    """
    Copy controllers changing the future policy as required
    """
    base_params_list = []
    for state, values in policy_vary.items():
        for level in values:
                base_params_copy = deepcopy(base_params)
                base_params["parameters_policies"]["States"][state] = level
                base_params_list.append( base_params_copy)

    return base_params_list

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
        VARY_POLICY_LOAD = "package/constants/vary_policy.json"
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    future_time_steps = base_params["duration_future"]#REMOVE TO RUN SINGLE RUN FOR CONSISTENCY, THEN PUT BACK IN FOR POLICY ANALYSIS
    base_params["duration_future"] = 0

    with open(VARY_POLICY_LOAD) as f:
        policy_vary = json.load(f)
    num_policies = len(policy_vary.keys())
    num_levels = len(policy_vary["Carbon_price"])

    root = "policy_analysis_single_seed"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)
    ##################################################################################################
    #RUN BURN IN + CALIBRATION PERIOD FIRST:
    controller = generate_data(base_params)  # run the simulation 

    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    ##################################################################################################

    params_list = base_params_and_controller_list_with_policy(base_params,policy_vary)


    print("TOTAL RUNS: ", len(params_list))

    data_flat_distance, data_flat_ev_prop, data_flat_age, data_flat_price , data_flat_emissions , data_flat_quality_ICE , data_flat_quality_EV , data_flat_efficiency_ICE, data_flat_efficiency_EV , data_flat_production_cost_ICE, data_flat_production_cost_EV, data_flat_distance_individual_ICE, data_flat_distance_individual_EV = policy_parallel_run(params_list, controller) 

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array_ev_prop = data_flat_ev_prop.reshape(base_params["seed_repetitions"], len(data_flat_ev_prop[0]))

    createFolder(fileName)

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array_distance = data_flat_distance.reshape(num_policies, num_levels, len(data_flat_distance[0]), base_params["parameters_social_network"]["num_individuals"])
    data_array_ev_prop = data_flat_ev_prop.reshape(num_policies, num_levels, len(data_flat_ev_prop[0]))
    data_array_age = data_flat_age.reshape(num_policies, num_levels, len(data_flat_age[0]), base_params["parameters_social_network"]["num_individuals"])
    data_array_price = data_flat_price.reshape(num_policies, num_levels, len(data_flat_price[0]), 2)
    data_array_emissions = data_flat_emissions.reshape(num_policies, num_levels, len(data_flat_emissions[0]))

    save_object(data_array_distance, fileName + "/Data", "data_array_distance")
    save_object(data_array_ev_prop , fileName + "/Data", "data_array_ev_prop")
    save_object(data_array_age , fileName + "/Data", "data_array_age")
    save_object(data_array_price , fileName + "/Data", "data_array_price")
    save_object(data_array_emissions  , fileName + "/Data", "data_array_emissions")

    save_object(data_flat_quality_ICE , fileName + "/Data", "data_flatquality_ICE")
    save_object(data_flat_quality_EV , fileName + "/Data", "data_flat_quality_EV")
    save_object(data_flat_efficiency_ICE , fileName + "/Data", "data_flat_efficiency_ICE")
    save_object(data_flat_efficiency_EV , fileName + "/Data", "data_flat_efficiency_EV")
    save_object(data_flat_production_cost_ICE , fileName + "/Data", "data_flat_production_cost_ICE")
    save_object(data_flat_production_cost_EV , fileName + "/Data", "data_flat_production_cost_EV")
    save_object(data_flat_distance_individual_ICE , fileName + "/Data", "data_flat_distance_individual_ICE")
    save_object(data_flat_distance_individual_EV , fileName + "/Data", "data_flat_distance_individual_EV")

    return params_list

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_policy_analysis_single_seed.json",
        VARY_POLICY_LOAD = "package/constants/vary_policy.json"
        )