from copy import deepcopy
import json
from package.resources.run import ev_prop_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime,
    params_list_with_seed
)
import numpy as np


def produce_param_list(params: dict, property_dict_1, property_dict_2) -> list[dict]:
    params_list = []

    for i in property_dict_1["property_list"]:
        for j in  property_dict_1["property_list"]:
            params_updated = deepcopy(params)
            params_updated[property_dict_1["subdict"]][property_dict_1["property_varied"]] = i
            params_updated[property_dict_2["subdict"]][property_dict_2["property_varied"]] = j


            parames_list_pairs = params_list_with_seed(params_updated)
            params_list.extend(parames_list_pairs)

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

    # Check if 'property_list' exists in vary_1, if not, create it
    if "property_list" not in vary_1:
        vary_1["property_list"] = np.linspace(vary_1["min"], vary_1["max"], vary_1["reps"])

    # Check if 'property_list' exists in vary_2, if not, create it
    if "property_list" not in vary_2:
        vary_2["property_list"] = np.linspace(vary_2["min"], vary_2["max"], vary_2["reps"])

    root = "MAPE_ev_2D"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = produce_param_list(base_params, vary_1, vary_2)
    
    print("TOTAL RUNS: ", len(params_list))
    
    data_flat_ev_prop, data_price_range = ev_prop_parallel_run(params_list) 
    print( data_flat_ev_prop.shape)


    createFolder(fileName)
    
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_1 , fileName + "/Data", "vary_1")
    save_object(vary_2 , fileName + "/Data", "vary_2")
    
    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array_ev_prop = data_flat_ev_prop.reshape(len(vary_1["property_list"]),len(vary_2["property_list"]),base_params["seed_repetitions"], len(data_flat_ev_prop[0]))
    save_object(data_array_ev_prop  , fileName + "/Data", "data_array_ev_prop")
    
    data_price_range_arr = data_price_range.reshape(len(vary_1["property_list"]),len(vary_2["property_list"]),base_params["seed_repetitions"])

    save_object(data_price_range_arr  , fileName + "/Data", "data_price_range_arr")
    #save_object(params_list, fileName + "/Data", "params_list_flat")

    return params_list

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_MAPE_2D.json",
        VARY_LOAD_1 ="package/constants/vary_single_beta_segments_MAPE.json", 
        VARY_LOAD_2 ="package/constants/vary_single_gamma_segments_MAPE.json",
        )