from copy import deepcopy
import json
from package.resources.run import ev_prop_parallel_run
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
    base_params["seeds"]["init_tech_seed"] = seed + seed_repetitions
    base_params["seeds"]["landscape_seed_ICE"] = seed + 2 * seed_repetitions
    base_params["seeds"]["social_network_seed"] = seed + 3 * seed_repetitions
    base_params["seeds"]["network_structure_seed"] = seed + 4 * seed_repetitions
    base_params["seeds"]["init_vals_environmental_seed"] = seed + 5 * seed_repetitions
    base_params["seeds"]["init_vals_innovative_seed"] = seed + 6 * seed_repetitions
    base_params["seeds"]["init_vals_price_seed"] = seed + 7 * seed_repetitions
    base_params["seeds"]["innovation_seed"] = seed + 8 * seed_repetitions
    base_params["seeds"]["landscape_seed_EV"] = seed + 9 * seed_repetitions
    base_params["seeds"]["choice_seed"] = seed + 10 * seed_repetitions
    base_params["seeds"]["remove_seed"] = seed + 11 * seed_repetitions
    base_params["seeds"]["init_vals_poisson_seed"] = seed + 12 * seed_repetitions
    return base_params

def produce_param_list(params: dict, property_dict_1, property_dict_2) -> list[dict]:
    params_list = []

    for i in property_dict_1["property_list"]:
        for j in  property_dict_1["property_list"]:
            params_updated = deepcopy(params)
            params_updated[property_dict_1["subdict"]][property_dict_1["property_varied"]] = i
            params_updated[property_dict_2["subdict"]][property_dict_2["property_varied"]] = j

            for seed in np.arange(1, params_updated["seed_repetitions"] + 1):

                params_updated = update_base_params_with_seed(params_updated, seed)

                params_list.append(
                    deepcopy(params_updated)
                )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict

    return params_list

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD_1 = "package/constants/vary_single.json",
        VARY_LOAD_2 = "package/constants/vary_single.json"
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(VARY_LOAD_1) as f:
        vary_1 = json.load(f)
    with open(VARY_LOAD_2) as f:
        vary_2 = json.load(f)

    vary_1["property_list"] = np.linspace( vary_1["min"],  vary_1["max"], vary_1["reps"])
    vary_2["property_list"] = np.linspace( vary_2["min"],  vary_2["max"], vary_2["reps"])

    root = "MAPE_ev_2D"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = produce_param_list(base_params, vary_1, vary_2)
    
    print("TOTAL RUNS: ", len(params_list))
    
    data_flat_ev_prop = ev_prop_parallel_run(params_list) 
    print( data_flat_ev_prop.shape)


    createFolder(fileName)
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(data_flat_ev_prop  , fileName + "/Data", "data_flat_ev_prop")
    save_object(vary_1 , fileName + "/Data", "vary_1")
    save_object(vary_2 , fileName + "/Data", "vary_2")
    
    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array_ev_prop = data_flat_ev_prop.reshape(len(vary_1["property_list"]),len(vary_2["property_list"]),base_params["seed_repetitions"], len(data_flat_ev_prop[0]))
    
    save_object(data_array_ev_prop  , fileName + "/Data", "data_array_ev_prop")
    #save_object(params_list, fileName + "/Data", "params_list_flat")
    


    return params_list

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_MAPE_2D.json",
        VARY_LOAD_1 ="package/constants/vary_single_beta_a_innov_MAPE.json", 
        VARY_LOAD_2 ="package/constants/vary_single_beta_b_innov_MAPE.json",
        )