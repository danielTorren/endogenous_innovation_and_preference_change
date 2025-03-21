import json
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
    property_list = vary_single["property_list"]#np.asarray([10e-6, 10e-5, 10e-4])
    print("property_list", property_list )
    params_list = produce_param_list(base_params, property_list, subdict, property_varied)
    
    print("TOTAL RUNS: ", len(params_list))
    data_flat_ev_prop, data_flat_price = ev_prop_price_emissions_parallel_run(params_list) 

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
   
    data_array_ev_prop = data_flat_ev_prop.reshape(len(property_list),seed_repetitions, len(data_flat_ev_prop[0]))
    #data_array_age = data_flat_age.reshape(len(property_list),seed_repetitions, len(data_flat_age[0]), base_params["parameters_social_network"]["num_individuals"])
    data_array_price = data_flat_price.reshape(len(property_list),seed_repetitions, len(data_flat_price[0]), 2, 2)

    createFolder(fileName)

    save_object(data_array_ev_prop , fileName + "/Data", "data_array_ev_prop")
    save_object(data_array_price , fileName + "/Data", "data_array_price")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_single , fileName + "/Data", "vary_single")

    print(fileName)
    return params_list

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single_delta.json",
        VARY_LOAD ="package/constants/vary_single_mu.json", #"package/constants/vary_single_delta.json"
        )