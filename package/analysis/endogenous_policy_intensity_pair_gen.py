import json
import numpy as np
import itertools
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from package.analysis.endogenous_policy_intensity_single_gen import optimize_policy_intensity_minimize
from package.resources.run import load_in_controller, parallel_run_multi_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime,
    params_list_with_seed
)

def generate_unique_policy_pairs(policy_list_all, policy_list_works):
    """
    Generate unique policy pairs (p1, p2) where p1 can be any policy,
    and p2 is restricted to a subset of policies that work well.
    """
    pairs = [(p1, p2) for p1 in policy_list_all for p2 in policy_list_works if p1 != p2]
    print("Total unique pairs:", len(pairs))
    return pairs

def optimize_second_policy_with_first_fixed(
    params,
    controller_list,
    policy1_name,
    policy2_name,
    policy1_intensity,
    policy2_init_guess,
    target_ev_uptake,
    bounds_dict,
    step_size_dict,
):
    """
    Fix policy1 intensity, then optimize policy2 to reach target EV uptake.
    """
    params_copy = deepcopy(params)
    
    if policy1_name == "Carbon_price":
        params_copy["parameters_policies"]["Values"][policy1_name]["Carbon_price"] = policy1_intensity
    else:
        params_copy["parameters_policies"]["Values"][policy1_name] = policy1_intensity

    optimized_intensity2, mean_ev_uptake2, mean_total_cost2, policy_data2 = optimize_policy_intensity_minimize(
        params_copy,
        controller_list,
        policy2_name,
        intensity_level_init=policy2_init_guess,
        target_ev_uptake=target_ev_uptake,
        bounds=bounds_dict[policy2_name],
        initial_step_size=policy2_init_guess * 0.01,
        adaptive_factor=0.5,
        step_size_bounds=step_size_dict[policy2_name]
    )
    
    return optimized_intensity2, mean_ev_uptake2, mean_total_cost2, policy_data2

def policy_pair_sweep(
    base_params,
    controller_list,
    policy1_name,
    policy2_name,
    bounds_dict,
    step_size_dict,
    target_ev_uptake=0.9,
    n_steps=10
):
    """
    Sweep policy1 intensity in steps, optimizing policy2 at each step.
    """
    p1_min, p1_max = bounds_dict[policy1_name]
    p1_values = np.linspace(p1_max, p1_min, n_steps)
    p2_guess = np.mean(bounds_dict[policy2_name])  # Start from midpoint

    results_for_pair = []
    policy_data_list = []
    
    for p1_val in p1_values:
        optimized_p2, ev_uptake2, cost2, policy_data2 = optimize_second_policy_with_first_fixed(
            base_params,
            controller_list,
            policy1_name,
            policy2_name,
            policy1_intensity=p1_val,
            policy2_init_guess=p2_guess,
            target_ev_uptake=target_ev_uptake,
            bounds_dict=bounds_dict,
            step_size_dict=step_size_dict
        )
        
        results_for_pair.append({
            "policy1_value": p1_val,
            "policy2_value": optimized_p2,
            "mean_ev_uptake": ev_uptake2,
            "mean_total_cost": cost2
        })
        policy_data_list.append(policy_data2)
        p2_guess = optimized_p2  # Use previous solution as next guess
    
    return results_for_pair, policy_data_list

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    BOUNDS_LOAD="package/analysis/policy_bounds.json", 
    policy_list_all=[
        "Carbon_price",
        "Discriminatory_corporate_tax",
        "Electricity_subsidy",
        "Adoption_subsidy",
        "Adoption_subsidy_used",
        "Production_subsidy",
        "Research_subsidy"
    ],
    policy_list_works=[
        "Carbon_price",
        "Discriminatory_corporate_tax",
        "Electricity_subsidy"
    ],
    target_ev_uptake=0.9,
    n_steps_for_sweep=5
):
    """
    Main function for running pairwise policy optimization.
    """
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        policy_params_dict = json.load(f)

    bounds_dict = policy_params_dict["bounds_dict"]
    step_size_dict = policy_params_dict["step_size_dict"]

    base_params_list = params_list_with_seed(base_params)
    controller_list = parallel_run_multi_run(base_params_list)
    
    if not controller_list:
        raise ValueError("Controller list is empty. Ensure calibration step is working.")
    
    fileName = produce_name_datetime("endogenous_policy_intensity_pair")
    createFolder(fileName)
    save_object(base_params, fileName + "/Data", "base_params_preburn")
    
    policy_pairs = generate_unique_policy_pairs(policy_list_all, policy_list_works)
    
    pairwise_outcomes = {}
    policy_data_list_dict = {}
    
    for (p1, p2) in policy_pairs:
        print(f"\n=== Optimizing Pair: ({p1}, {p2}) ===")
        results, policy_data_list = policy_pair_sweep(
            base_params,
            controller_list,
            p1,
            p2,
            bounds_dict,
            step_size_dict,
            target_ev_uptake,
            n_steps_for_sweep
        )
        pairwise_outcomes[(p1, p2)] = results
        policy_data_list_dict[(p1, p2)] = policy_data_list
    
    save_object(pairwise_outcomes, fileName + "/Data", "pairwise_outcomes")
    save_object(policy_data_list_dict, fileName + "/Data", "policy_data_list_dict")
    
    print("Optimization Complete. Data saved.")
    return "Done"

if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
        BOUNDS_LOAD="package/analysis/policy_bounds_vary_pair_policy_gen.json", 
        policy_list_all=[
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Adoption_subsidy_used",
            "Production_subsidy",
            "Research_subsidy"
        ],
        policy_list_works=["Carbon_price"],
        target_ev_uptake=0.6,
        n_steps_for_sweep=10
    )