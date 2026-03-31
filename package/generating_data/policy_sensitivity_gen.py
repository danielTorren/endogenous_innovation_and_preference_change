import json
import numpy as np
from package.resources.run import ev_prop_price_emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)

def produce_nested_param_list(base_params, var_a, var_b):
    """
    Creates a flat list of params for all combinations of var_a and var_b.
    var_a: Dictionary for the 'Environment' parameter
    var_b: Dictionary for the 'Policy' parameter
    """
    final_list = []
    
    for val_a in var_a["property_list"]:
        for val_b in var_b["property_list"]:
            # Deep copy to avoid mutating the original dict in the loop
            current_params = json.loads(json.dumps(base_params))
            
            # Set Param A
            current_params[var_a["subdict"]][var_a["property_varied"]] = val_a
            # Set Param B (Policy)
            current_params[var_b["subdict"]][var_b["property_varied"]] = val_b
            
            # Expand by seeds
            seeds_list = params_list_with_seed(current_params)
            final_list.extend(seeds_list)
            
    return final_list

def run_cross_variation(BASE_PARAMS_LOAD, VAR_PHYSICAL_LOAD, VAR_POLICY_LOAD):
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    with open(VAR_PHYSICAL_LOAD) as f:
        var_phys = json.load(f)
    with open(VAR_POLICY_LOAD) as f:
        var_pol = json.load(f)

    seed_reps = base_params["seed_repetitions"]
    
    # Extract metadata for naming and reshaping
    phys_prop = var_phys["property_varied"]
    pol_prop = var_pol["property_varied"]
    phys_list = var_phys["property_list"]
    pol_list = var_pol["property_list"]

    folder_name = produce_name_datetime(f"cross_{phys_prop}_vs_{pol_prop}")

    # Generate nested list
    params_list = produce_nested_param_list(base_params, var_phys, var_pol)
    print(f"Total combinations: {len(phys_list)} x {len(pol_list)}")
    print(f"Total runs (inc. seeds): {len(params_list)}")

    # Execute Parallel Run
    data_ev, data_price, _ = ev_prop_price_emissions_parallel_run(params_list)

    # Reshape logic: (Phys_Index, Policy_Index, Seed_Index, Time_Steps)
    # This allows you to plot 'Policy' lines for each 'Physical' scenario
    data_array_ev = data_ev.reshape(
        len(phys_list), 
        len(pol_list), 
        seed_reps, 
        -1 # Automatically detect time-step length
    )

    # Save results
    createFolder(folder_name)
    save_object(data_array_ev, folder_name + "/Data", "data_cross_ev")
    save_object(base_params, folder_name + "/Data", "base_params")
    save_object({"phys": var_phys, "policy": var_pol}, folder_name + "/Data", "vary_metadata")

    return folder_name

if __name__ == "__main__":
    # Example usage
    run_cross_variation(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_joint.json",
        VAR_PHYSICAL_LOAD="package/constants/vary_policy_beta.json",
        VAR_POLICY_LOAD="package/constants/vary_policy_carbon_tax.json" # Load your policy here
    )