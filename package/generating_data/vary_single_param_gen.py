from copy import deepcopy
import itertools
import json
from package.resources.run import emissions_parallel_run, distance_parallel_run, distance_ev_prop_parallel_run, distance_ev_prop_age_price_emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
import numpy as np

def update_base_params_with_seed(base_params, seed):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    seed_repetitions = base_params["seed_repetitions"]
    # VARY ALL THE SEEDS
    base_params["parameters_firm_manager"]["init_tech_seed"] = int(seed + seed_repetitions)
    base_params["parameters_ICE"]["landscape_seed"] = int(seed + 2 * seed_repetitions)
    base_params["parameters_EV"]["landscape_seed"] = int(seed + 9 * seed_repetitions)
    base_params["parameters_social_network"]["social_network_seed"] = int(seed + 3 * seed_repetitions)
    base_params["parameters_social_network"]["network_structure_seed"] = int(seed + 4 * seed_repetitions)
    base_params["parameters_social_network"]["init_vals_environmental_seed"] = int(seed + 5 * seed_repetitions)
    base_params["parameters_social_network"]["init_vals_innovative_seed"] = int(seed + 6 * seed_repetitions)
    base_params["parameters_social_network"]["init_vals_price_seed"] = int(seed + 7 * seed_repetitions)
    base_params["parameters_firm"]["innovation_seed"] = int(seed + 8 * seed_repetitions)
    return base_params


def produce_param_list(params: dict, property_list: list, subdict, property: str) -> list[dict]:
    params_list = []

    for i in property_list:
        params_updated = deepcopy(params)
        params_updated[subdict][property] = i

        for seed in np.arange(1, params_updated["seed_repetitions"] + 1):

            params_updated = update_base_params_with_seed(params_updated, seed)

            params_list.append(
                deepcopy(params_updated)
            )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict

    return params_list

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD = "package/constants/vary_single.json"
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(VARY_LOAD) as f:
        vary_single = json.load(f)


    seed_repetitions = base_params["seed_repetitions"]

    root = "single_param_vary"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    property_varied = vary_single["property_varied"]#"delta"
    subdict = vary_single["subdict"]#"parameters_ICE"
    property_values_list = vary_single["property_values_list"]#np.asarray([10e-6, 10e-5, 10e-4])
    print("property_values_list", property_values_list )
    params_list = produce_param_list(base_params, property_values_list, subdict, property_varied)
    
    print("TOTAL RUNS: ", len(params_list))
    data_flat_distance, data_flat_ev_prop, data_flat_age, data_flat_price , data_flat_emissions = distance_ev_prop_age_price_emissions_parallel_run(params_list) 

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array_distance = data_flat_distance.reshape(len(property_values_list),seed_repetitions, len(data_flat_distance[0]), base_params["parameters_social_network"]["num_individuals"])
    data_array_ev_prop = data_flat_ev_prop.reshape(len(property_values_list),seed_repetitions, len(data_flat_ev_prop[0]))
    data_array_age = data_flat_age.reshape(len(property_values_list),seed_repetitions, len(data_flat_age[0]), base_params["parameters_social_network"]["num_individuals"])
    data_array_price = data_flat_price.reshape(len(property_values_list),seed_repetitions, len(data_flat_price[0]), 2)
    data_array_emissions = data_flat_emissions.reshape(len(property_values_list),seed_repetitions, len(data_flat_emissions[0]))
    
    createFolder(fileName)

    save_object(data_array_distance, fileName + "/Data", "data_array_distance")
    save_object(data_array_ev_prop , fileName + "/Data", "data_array_ev_prop")
    save_object(data_array_age , fileName + "/Data", "data_array_age")
    save_object(data_array_price , fileName + "/Data", "data_array_price")
    save_object(data_array_emissions  , fileName + "/Data", "data_array_emissions")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_single , fileName + "/Data", "vary_single")

    return params_list

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single_delta.json",
        VARY_LOAD ="package/constants/vary_single_mu.json", #"package/constants/vary_single_delta.json"
        )