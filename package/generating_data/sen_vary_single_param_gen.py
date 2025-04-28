import json
import os
from package.resources.run import ev_prop_price_emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)

def produce_param_list(params: dict, property_list: list, subdict, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[subdict][property] = i
        seeds_base_params_list = params_list_with_seed(params)
        params_list.extend(seeds_base_params_list)
    return params_list

def run_single_variation(BASE_PARAMS_LOAD, VARY_LOAD, root_folder):
    # Load parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    with open(VARY_LOAD) as f:
        vary_single = json.load(f)

    seed_repetitions = base_params["seed_repetitions"]

    # Define file name
    property_varied = vary_single["property_varied"]
    subdict = vary_single["subdict"]
    property_list = vary_single["property_list"]

    fileName = os.path.join(root_folder, produce_name_datetime(property_varied))
    print(f"Running for {property_varied}, saving in {fileName}")

    # Create parameter list
    params_list = produce_param_list(base_params, property_list, subdict, property_varied)
    print("TOTAL RUNS: ", len(params_list))

    # Run the simulation
    data_flat_ev_prop, data_flat_price, data_flat_margins = ev_prop_price_emissions_parallel_run(params_list) 

    # Reshape
    data_array_ev_prop = data_flat_ev_prop.reshape(len(property_list), seed_repetitions, len(data_flat_ev_prop[0]))
    data_array_price = data_flat_price.reshape(len(property_list), seed_repetitions, len(data_flat_price[0]), 2, 2)
    data_array_margins = data_flat_margins.reshape(len(property_list), seed_repetitions, len(data_flat_ev_prop[0]))

    # Save
    createFolder(fileName)
    save_object(data_array_ev_prop , fileName + "/Data", "data_array_ev_prop")
    save_object(data_array_price , fileName + "/Data", "data_array_price")
    save_object(data_array_margins, fileName + "/Data", "data_array_margins")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_single , fileName + "/Data", "vary_single")

    return fileName

def main(BASE_PARAMS_LOAD, VARY_LOADS):
    root_folder = "results/multi_param_run"
    createFolder(root_folder)

    output_folders = []

    for vary_load in VARY_LOADS:
        output_folder = run_single_variation(BASE_PARAMS_LOAD, vary_load, root_folder)
        output_folders.append(output_folder)

    print("All runs complete.")
    return output_folders

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single.json",
        VARY_LOADS=[
            "package/constants/vary_single_a_innov.json",
            "package/constants/vary_single_kappa.json",
            "package/constants/vary_single_landscape_K.json",
            "package/constants/vary_single_lambda.json",
            "package/constants/vary_single_delta.json",
            "package/constants/vary_single_mu.json",
        ]
    )
