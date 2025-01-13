from copy import deepcopy
import json
from package.resources.run import emissions_parallel_run, ev_prop_parallel_run
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

    data_flat_ev_prop = ev_prop_parallel_run(params_list) 
    print( data_flat_ev_prop.shape)

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array_ev_prop = data_flat_ev_prop.reshape(base_params["seed_repetitions"], len(data_flat_ev_prop[0]))

    createFolder(fileName)

    save_object(data_array_ev_prop  , fileName + "/Data", "data_array_ev_prop")
    save_object(params_list, fileName + "/Data", "params_list_flat")
    save_object(base_params, fileName + "/Data", "base_params")

    print("Done")
    return params_list

if __name__ == "__main__":
    results = main(BASE_PARAMS_LOAD="package/constants/base_params_multi_seed.json")