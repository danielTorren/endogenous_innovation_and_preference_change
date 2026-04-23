import json
import numpy as np
from package.resources.run import ev_prop_price_emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)

def update_policy_intensity(params, policy_name, intensity_level):
    """
    Update the policy intensity in the parameter dictionary.
    Ensures the policy state is active (1).
    """
    params["parameters_policies"]["States"][policy_name] = 1

    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity_level
    return params

def produce_nested_param_list(base_params, var_phys, var_policy):
    """
    Cleaned version using direct dictionary access and your policy function.
    """
    final_list = []
    
    for val_phys in var_phys["property_list"]:
        for val_policy in var_policy["property_list"]:
            # Deep copy to avoid mutating the original base_params
            current_params = json.loads(json.dumps(base_params))
            
            # 1. Update Physical Variable
            # Direct access: current_params['environment']['beta_multiplier'] = value
            sub = var_phys["subdict"]
            prop = var_phys["property_varied"]
            current_params[sub][prop] = val_phys
                
            # 2. Update Policy Variable using your new function
            current_params = update_policy_intensity(
                current_params, 
                var_policy["property_varied"], 
                val_policy
            )
            
            # 3. Expand by seeds
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

    # Metadata for labels and reshaping
    phys_list = var_phys["property_list"]
    pol_list = var_pol["property_list"]
    seed_reps = base_params["seed_repetitions"]

    folder_name = produce_name_datetime(f"cross_{var_phys['property_varied']}_vs_{var_pol['property_varied']}")

    # Generate the parameter list
    params_list = produce_nested_param_list(base_params, var_phys, var_pol)
    
    print(f"Total combinations: {len(phys_list)} x {len(pol_list)}")
    print(f"Total simulation runs (including seeds): {len(params_list)}")

    # Execute Parallel Run
    data_ev, _, _ = ev_prop_price_emissions_parallel_run(params_list)

    # Reshape logic: (Physical_Index, Policy_Index, Seed_Index, Time_Steps)
    data_array_ev = data_ev.reshape(
        len(phys_list), 
        len(pol_list), 
        seed_reps, 
        -1 
    )

    # Save everything
    createFolder(folder_name)
    save_object(data_array_ev, folder_name + "/Data", "data_cross_ev")
    save_object(base_params, folder_name + "/Data", "base_params")
    save_object({"phys": var_phys, "policy": var_pol}, folder_name + "/Data", "vary_metadata")

    print(f"Success! Data saved to: {folder_name}")
    return folder_name

if __name__ == "__main__":
    run_cross_variation(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_joint.json",
        VAR_PHYSICAL_LOAD="package/constants/vary_policy_nu.json",
        VAR_POLICY_LOAD="package/constants/vary_policy_carbon_tax.json"


        #vary_policy_a_chi
        #vary_policy_beta_multiplier
    )