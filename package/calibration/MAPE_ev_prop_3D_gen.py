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

def produce_param_list(params: dict, property_dict_1, property_dict_2, property_dict_3) -> list[dict]:
    params_list = []

    for i in property_dict_1["property_list"]:
        for j in property_dict_2["property_list"]:
            for k in property_dict_3["property_list"]:
                params_updated = deepcopy(params)
                params_updated[property_dict_1["subdict"]][property_dict_1["property_varied"]] = i
                params_updated[property_dict_2["subdict"]][property_dict_2["property_varied"]] = j
                params_updated[property_dict_3["subdict"]][property_dict_3["property_varied"]] = k

                params_list_seed = params_list_with_seed(params_updated)
                params_list.extend(params_list_seed)
    return params_list

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD_1="package/constants/vary_single.json",
        VARY_LOAD_2="package/constants/vary_single.json",
        VARY_LOAD_3="package/constants/vary_single.json"
    ) -> str: 

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(VARY_LOAD_1) as f:
        vary_1 = json.load(f)
    with open(VARY_LOAD_2) as f:
        vary_2 = json.load(f)
    with open(VARY_LOAD_3) as f:
        vary_3 = json.load(f)

    vary_1["property_list"] = np.linspace(vary_1["min"], vary_1["max"], vary_1["reps"])
    vary_2["property_list"] = np.linspace(vary_2["min"], vary_2["max"], vary_2["reps"])
    vary_3["property_list"] = np.linspace(vary_3["min"], vary_3["max"], vary_3["reps"])

    root = "MAPE_ev_3D"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = produce_param_list(base_params, vary_1, vary_2, vary_3)
    
    print("TOTAL RUNS: ", len(params_list))
    data_flat_ev_prop = ev_prop_parallel_run(params_list) 
    print(data_flat_ev_prop.shape)

    createFolder(fileName)
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(data_flat_ev_prop, fileName + "/Data", "data_flat_ev_prop")
    save_object(vary_1, fileName + "/Data", "vary_1")
    save_object(vary_2, fileName + "/Data", "vary_2")
    save_object(vary_3, fileName + "/Data", "vary_3")

    data_array_ev_prop = data_flat_ev_prop.reshape(
        len(vary_1["property_list"]),
        len(vary_2["property_list"]),
        len(vary_3["property_list"]),
        base_params["seed_repetitions"],
        len(data_flat_ev_prop[0])
    )
    
    save_object(data_array_ev_prop, fileName + "/Data", "data_array_ev_prop")

    return params_list

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_MAPE_3D.json",
        VARY_LOAD_1="package/constants/vary_single_beta_a_innov_MAPE.json", 
        VARY_LOAD_2="package/constants/vary_single_beta_b_innov_MAPE.json",
        VARY_LOAD_3="package/constants/vary_single_proportion_zero_target_MAPE.json"
    )
