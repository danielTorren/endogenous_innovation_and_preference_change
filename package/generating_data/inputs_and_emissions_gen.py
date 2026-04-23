import json
import numpy as np
from itertools import product
from package.resources.run import ev_prop_emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)

def produce_physical_param_list(base_params, var_configs):
    """
    Generates a parameter list for the Cartesian product of all physical variations.
    Works for any number of input variation files.
    """
    # Extract the lists of values from each config (e.g., [0.1, 0.2, 0.3])
    value_grids = [v["property_list"] for v in var_configs]
    
    final_list = []
    
    # itertools.product handles the nested looping for us
    for values in product(*value_grids):
        # Deep copy to ensure each simulation starts from a fresh base
        current_params = json.loads(json.dumps(base_params))
        
        # Apply each physical variation to the dictionary
        for i, val in enumerate(values):
            v_cfg = var_configs[i]
            sub = v_cfg["subdict"]
            prop = v_cfg["property_varied"]
            
            # Standard physical update: current_params['subdict']['property'] = value
            current_params[sub][prop] = val
                
        # Expand the specific parameter set by the number of seeds
        seeds_list = params_list_with_seed(current_params)
        final_list.extend(seeds_list)
            
    return final_list

def run_physical_trio(BASE_PARAMS_PATH, VAR_PATHS):
    # 1. Load Files
    with open(BASE_PARAMS_PATH) as f:
        base_params = json.load(f)
    
    var_configs = []
    for path in VAR_PATHS:
        with open(path) as f:
            var_configs.append(json.load(f))

    # 2. Metadata for Reshaping and Naming
    dims = [len(v["property_list"]) for v in var_configs]
    seed_reps = base_params.get("seed_repetitions", 1)
    var_names = "_vs_".join([v["property_varied"] for v in var_configs])
    
    folder_name = produce_name_datetime(f"phys_trio_{var_names}")

    # 3. Generate Parameters
    params_list = produce_physical_param_list(base_params, var_configs)
    
    print(f"Structure: {' x '.join(map(str, dims))} (total combinations: {np.prod(dims)})")
    print(f"Total individual runs (including seeds): {len(params_list)}")

    # 4. Execute Parallel Run
    # data_ev is expected to be a flat numpy array from the parallel runner
    data_ev, data_emissions= ev_prop_emissions_parallel_run(params_list)

    # 5. Reshape to Multi-Dimensional Array
    # Shape logic: (Var1, Var2, Var3, Seeds, Time)
    data_array_ev = data_ev.reshape(*dims, seed_reps, -1)
    data_array_emissions = data_emissions.reshape(*dims, seed_reps, -1)

    # 6. Save Outputs
    createFolder(folder_name)
    save_object(data_array_ev, folder_name + "/Data", "data_phys_trio_ev")
    save_object(data_array_emissions, folder_name + "/Data", "data_phys_trio_emissions")
    save_object(base_params, folder_name + "/Data", "base_params")
    save_object(var_configs, folder_name + "/Data", "vary_metadata")

    print(f"Success! Data saved to: {folder_name}")
    return folder_name

if __name__ == "__main__":
    # Update these paths to your specific physical variation JSONs
    # Example: Emissions Intensity, Fuel Price, and Electricity Price
    PHYSICAL_VARS = [
        "package/constants/vary_sen_decarb.json", 
        "package/constants/vary_sen_fuel_price.json",
        "package/constants/vary_sen_elec_price.json"
    ]

    run_physical_trio(
        BASE_PARAMS_PATH="package/constants/base_params_inputs_and_emissions.json",
        VAR_PATHS=PHYSICAL_VARS
    )