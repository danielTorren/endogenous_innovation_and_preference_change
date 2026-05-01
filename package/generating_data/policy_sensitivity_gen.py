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

    phys_list = var_phys["property_list"]
    pol_list = var_pol["property_list"]
    seed_reps = base_params["seed_repetitions"]
    folder_name = produce_name_datetime(f"cross_{var_phys['property_varied']}_vs_{var_pol['property_varied']}")
    
    # Print run information
    print("\n" + "="*60)
    print("CROSS VARIATION RUN INFORMATION")
    print("="*60)
    print(f"Physical parameter varied: {var_phys['property_varied']}")
    print(f"  Values: {phys_list}")
    print(f"  Number of values: {len(phys_list)}")
    print(f"\nPolicy parameter varied: {var_pol['property_varied']}")
    print(f"  Values: {pol_list}")
    print(f"  Number of values: {len(pol_list)}")
    print(f"\nSeed repetitions: {seed_reps}")
    print("-"*60)

    # --- Policy runs (unchanged) ---
    params_list = produce_nested_param_list(base_params, var_phys, var_pol)
    
    # Calculate and print total runs for policy runs
    total_policy_runs = len(params_list)
    policy_combinations = len(phys_list) * len(pol_list)
    print(f"\nPOLICY RUNS:")
    print(f"  Parameter combinations: {policy_combinations} ({len(phys_list)} physical × {len(pol_list)} policy)")
    print(f"  Total runs (including seeds): {total_policy_runs}")
    print(f"    = {policy_combinations} combinations × {seed_reps} seeds")
    
    data_ev, _, _ = ev_prop_price_emissions_parallel_run(params_list)
    data_array_ev = data_ev.reshape(len(phys_list), len(pol_list), seed_reps, -1)

    createFolder(folder_name)
    save_object(data_array_ev, folder_name + "/Data", "data_cross_ev")
    save_object(base_params, folder_name + "/Data", "base_params")
    save_object({"phys": var_phys, "policy": var_pol}, folder_name + "/Data", "vary_metadata")

    # --- BAU runs: one per physical parameter value, policy off ---
    bau_params_list = []
    for val_phys in phys_list:
        current_params = json.loads(json.dumps(base_params))
        sub = var_phys["subdict"]
        prop = var_phys["property_varied"]
        current_params[sub][prop] = val_phys
        # Ensure all policies are off
        for pol_name in current_params["parameters_policies"]["States"]:
            current_params["parameters_policies"]["States"][pol_name] = 0
        bau_params_list.extend(params_list_with_seed(current_params))
    
    # Calculate and print total runs for BAU runs
    total_bau_runs = len(bau_params_list)
    bau_combinations = len(phys_list)
    print(f"\nBAU RUNS (all policies disabled):")
    print(f"  Parameter combinations: {bau_combinations} ({len(phys_list)} physical values)")
    print(f"  Total runs (including seeds): {total_bau_runs}")
    print(f"    = {bau_combinations} combinations × {seed_reps} seeds")

    data_ev_bau, _, _ = ev_prop_price_emissions_parallel_run(bau_params_list)
    # Shape: (Physical_Index, Seed_Index, Time_Steps)
    data_array_bau = data_ev_bau.reshape(len(phys_list), seed_reps, -1)

    save_object(data_array_bau, folder_name + "/Data", "data_cross_bau")
    
    # Print grand total
    grand_total = total_policy_runs + total_bau_runs
    print("-"*60)
    print(f"GRAND TOTAL RUNS: {grand_total}")
    print(f"  Policy runs: {total_policy_runs}")
    print(f"  BAU runs: {total_bau_runs}")
    print("="*60)
    print(f"Success! Data saved to: {folder_name}\n")
    
    return folder_name

def run_bau_only(BASE_PARAMS_LOAD, VAR_PHYSICAL_LOAD, VAR_POLICY_LOAD):
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    with open(VAR_PHYSICAL_LOAD) as f:
        var_phys = json.load(f)
    with open(VAR_POLICY_LOAD) as f:
        var_pol = json.load(f)

    phys_list = var_phys["property_list"]
    pol_list = var_pol["property_list"]
    seed_reps = base_params["seed_repetitions"]
    folder_name = produce_name_datetime(f"cross_{var_phys['property_varied']}_vs_{var_pol['property_varied']}_BAU")

    # --- BAU runs: one per physical parameter value, policy off ---
    bau_params_list = []
    for val_phys in phys_list:
        current_params = json.loads(json.dumps(base_params))
        sub = var_phys["subdict"]
        prop = var_phys["property_varied"]
        current_params[sub][prop] = val_phys
        for pol_name in current_params["parameters_policies"]["States"]:
            current_params["parameters_policies"]["States"][pol_name] = 0
        bau_params_list.extend(params_list_with_seed(current_params))

    total_runs = len(bau_params_list)
    print(f"Total BAU runs: {total_runs} ({len(phys_list)} physical values × {seed_reps} seeds)")

    data_ev_bau, _, _ = ev_prop_price_emissions_parallel_run(bau_params_list)
    data_array_bau = data_ev_bau.reshape(len(phys_list), seed_reps, -1)

    createFolder(folder_name)
    save_object(data_array_bau, folder_name + "/Data", "data_cross_bau")
    save_object(base_params, folder_name + "/Data", "base_params")
    save_object({"phys": var_phys, "policy": var_pol}, folder_name + "/Data", "vary_metadata")

    print(f"Success! BAU data saved to: {folder_name}")
    return folder_name


if __name__ == "__main__":
    #run_cross_variation(
    #    BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_joint.json",
    #    VAR_PHYSICAL_LOAD="package/constants/vary_policy_a_chi.json",
    #    VAR_POLICY_LOAD="package/constants/vary_policy_new_car_rebate.json"
    #)

    run_cross_variation(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_joint.json",
        VAR_PHYSICAL_LOAD="package/constants/vary_policy_beta_multiplier.json",
        VAR_POLICY_LOAD="package/constants/vary_policy_new_car_rebate.json"
    )
    
    #run_bau_only(
    #    BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_joint.json",
    #    VAR_PHYSICAL_LOAD="package/constants/vary_policy_a_chi.json",
    #    VAR_POLICY_LOAD="package/constants/vary_policy_carbon_tax.json"
    #)

    #run_bau_only(
    #    BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_joint.json",
    #    VAR_PHYSICAL_LOAD="package/constants/vary_policy_beta_multiplier.json",
    #    VAR_POLICY_LOAD="package/constants/vary_policy_carbon_tax.json"
    #)
        #vary_policy_a_chi
        #vary_policy_beta_multiplier
        #vary_policy_carbon_tax
