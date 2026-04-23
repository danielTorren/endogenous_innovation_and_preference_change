import json
import numpy as np
from package.resources.run import ev_prop_price_emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)

def produce_param_list(params: dict, property_list: list, subdict: str, prop_name: str) -> list[dict]:
    """Generates a flat list of params for parallel processing."""
    params_list = []
    for val in property_list:
        # Deep update of the specific parameter
        params[subdict][prop_name] = val
        # Generate the N seeds for this specific parameter value
        seeds_base_params_list = params_list_with_seed(params)
        params_list.extend(seeds_base_params_list)
    return params_list

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD="package/constants/vary_single.json"
    ) -> str: 

    # 1. Load Configurations
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(VARY_LOAD) as f:
        vary_single = json.load(f)

    # 2. Extract Metadata
    seed_repetitions = base_params["seed_repetitions"]
    property_varied = vary_single["property_varied"]
    subdict = vary_single["subdict"]
    property_list = vary_single["property_list"]
    
    root = f"battery_corr_{property_varied}"
    folder_name = produce_name_datetime(root)
    
    print(f"Starting simulation: {property_varied}")
    print(f"Values to test: {property_list}")
    
    # 3. Prepare Parameter List
    params_list = produce_param_list(base_params, property_list, subdict, property_varied)
    print(f"TOTAL RUNS (Scenarios x Seeds): {len(params_list)}")

    # 4. Execute Parallel Runs
    # Expected output: (EV Proportion, Electricity/Fuel Prices, Emissions/Margins)
    data_flat_ev, data_flat_prices, data_flat_emissions = ev_prop_price_emissions_parallel_run(params_list) 

    # 5. Reshape Data
    # Structure: [Scenario_Index, Seed_Index, Time_Step]
    num_scenarios = len(property_list)
    time_steps = len(data_flat_ev[0])

    data_array_ev = data_flat_ev.reshape(num_scenarios, seed_repetitions, time_steps)
    
    # Prices often have extra dimensions (e.g., matrix of price points), 
    # adjust the reshape based on your specific model output (currently 2x2 placeholder)
    try:
        data_array_price = data_flat_prices.reshape(num_scenarios, seed_repetitions, time_steps, -1)
    except ValueError:
        data_array_price = data_flat_prices.reshape(num_scenarios, seed_repetitions, time_steps)

    data_array_emissions = data_flat_emissions.reshape(num_scenarios, seed_repetitions, time_steps)

    # 6. Save Results
    createFolder(folder_name)
    data_path = folder_name + "/Data"
    createFolder(data_path)

    save_object(data_array_ev, data_path, "data_ev")
    save_object(data_array_emissions, data_path, "data_emissions")
    save_object(data_array_price, data_path, "data_price")
    save_object(base_params, data_path, "base_params")
    save_object(vary_single, data_path, "vary_metadata")

    print(f"Run Complete. Saved to: {folder_name}")
    return folder_name

if __name__ == "__main__":
    # Example: Varying Electricity Emissions Intensity
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_battery_corr.json",
        VARY_LOAD="package/constants/vary_battery_corr.json"
    )