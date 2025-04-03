import json
import numpy as np
from package.analysis.endogenous_policy_intensity_single_gen import optimize_policy_intensity_BO, set_up_calibration_runs
from package.resources.utility import (
    save_object, 
)
import shutil  # Cleanup
from pathlib import Path  # Path handling
from copy import deepcopy

def update_policy_intensity(params, policy_name, intensity_level):
    """
    Update the policy intensity in the parameter dictionary.
    """
    params["parameters_policies"]["States"][policy_name] = 1

    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity_level

    return params

def generate_unique_policy_pairs(policy_list_all, dont_work_list):
    """
    Generate unique, consistently ordered pairs where:
    - Only one pair like (A, B) is included — not (B, A).
    - No self-pairs like (A, A).
    - If one policy is in dont_work_list, it comes first in the pair.
    - Output is always in the same order for the same input.
    """
    pairs = set()

    for i, policy1 in enumerate(policy_list_all):
        for policy2 in policy_list_all[i+1:]:
            if policy1 == policy2:
                continue
            # Force policy from dont_work_list to come first, if applicable
            if policy1 in dont_work_list and policy2 not in dont_work_list:
                pair = (policy1, policy2)
            elif policy2 in dont_work_list and policy1 not in dont_work_list:
                pair = (policy2, policy1)
            else:
                pair = tuple(sorted([policy1, policy2]))

            pairs.add(pair)

    sorted_pairs = sorted(pairs)
    print("Number of unique pairs:", len(sorted_pairs))
    return sorted_pairs

def generate_all_policy_pairs(policy_list_all):

    # Generate all possible unique pairs of policies
    pairs = []
    for i, p1 in enumerate(policy_list_all):
        for j, p2 in enumerate(policy_list_all):
            if p1 != p2:
                pairs.append((p1,p2))

    sorted_pairs = sorted(pairs)
    print("Number of unique pairs:", len(sorted_pairs))
    return sorted_pairs


def policy_pair_sweep(
    base_params,
    controller_files,
    policy1_name,
    policy2_name,
    bounds_dict,
    target_ev_uptake=0.9,
    n_steps=10,
    n_calls=40, 
    noise = 0.05
):
    """
    Sweep policy1 intensity in steps, optimizing policy2 at each step.
    """
    print(bounds_dict)
    p1_min, p1_max = bounds_dict[policy1_name]
    epsilon = p1_max*0.15
    p1_values = np.linspace(p1_min + epsilon, p1_max - epsilon, n_steps)
    results_for_pair = []
    
    for p1_val in p1_values:
        print("Policy1, Val: ", policy1_name, p1_val)
        #UPDATE THE BASE PARAMS

        # Always start from a clean copy for this iteration
        params_for_this_run = deepcopy(base_params)
        params_for_this_run = update_policy_intensity(params_for_this_run, policy1_name, p1_val)

        (
            best_intensity,
            mean_ev_uptake, 
            sd_ev_uptake, 
            mean_total_cost, 
            mean_net_cost, 
            mean_emissions_cumulative, 
            mean_emissions_cumulative_driving, 
            mean_emissions_cumulative_production, 
            mean_utility_cumulative, 
            mean_utility_cumulative_30, 
            mean_profit_cumulative,
            ev_uptake,
            net_cost,
            emissions_cumulative_driving,
            emissions_cumulative_production,
            utility_cumulative,
            profit_cumulative 
        ) = optimize_policy_intensity_BO(
            params_for_this_run, 
            controller_files, 
            policy2_name, 
            target_ev_uptake=target_ev_uptake,
            bounds=bounds_dict[policy2_name], 
            n_calls=n_calls, 
            noise = noise
        )
        
        results_for_pair.append({
            "policy1_value": p1_val,
            "policy2_value": best_intensity,
            "mean_ev_uptake": mean_ev_uptake,
            "sd_ev_uptake": sd_ev_uptake,
            "mean_total_cost": mean_total_cost,
            "mean_net_cost": mean_net_cost, 
            "mean_emissions_cumulative": mean_emissions_cumulative, 
            "mean_emissions_cumulative_driving": mean_emissions_cumulative_driving, 
            "mean_emissions_cumulative_production": mean_emissions_cumulative_production, 
            "mean_utility_cumulative": mean_utility_cumulative, 
            "mean_utility_cumulative_30": mean_utility_cumulative_30,
            "mean_profit_cumulative": mean_profit_cumulative,
            "ev_uptake": ev_uptake,
            "net_cost": net_cost,
            "emissions_cumulative_driving": emissions_cumulative_driving,
            "emissions_cumulative_production": emissions_cumulative_production,
            "utility_cumulative": utility_cumulative,
            "profit_cumulative": profit_cumulative
        })

    return results_for_pair

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    BOUNDS_LOAD="package/analysis/policy_bounds.json", 
    policy_list_all=[
        "Carbon_price",
        "Electricity_subsidy",
        "Adoption_subsidy",
        "Adoption_subsidy_used",
        "Production_subsidy",
    ],
    target_ev_uptake=0.95,
    n_steps_for_sweep=5,
    n_calls=40, 
    noise = 0.01
):
    """
    Main function for running pairwise policy optimization.
    """
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        bounds_dict = json.load(f)

    policy_pairs = generate_all_policy_pairs(policy_list_all)

    policy_pairs = policy_pairs[:10]
    #policy_pairs = policy_pairs[10:]
    
    print(policy_pairs)

    
    controller_files, base_params, file_name = set_up_calibration_runs(base_params,"endog_pair")

    ###################################################################################################################

    #RUN POLICY OUTCOMES
    print("TOTAL RUNS BO: ", len(policy_pairs)*n_steps_for_sweep*n_calls*base_params["seed_repetitions"])

    pairwise_outcomes = {}
    
    for (policy1_name, policy2_name) in policy_pairs:
        print(f"\n=== Optimizing Pair: ({policy1_name}, {policy2_name}) ===")
        
        policy_base_params = deepcopy(base_params)
        results = policy_pair_sweep(
            policy_base_params,
            controller_files,
            policy1_name,
            policy2_name,
            bounds_dict,
            target_ev_uptake,
            n_steps_for_sweep,
            n_calls, 
            noise
        )
        pairwise_outcomes[(policy1_name, policy2_name)] = results
    
    save_object(pairwise_outcomes, file_name + "/Data", "pairwise_outcomes")#

    print("Optimization Complete. Data saved.")

    shutil.rmtree(Path(file_name) / "Calibration_runs", ignore_errors=True)

    conditions = {
        "policy_list_all": policy_list_all,
        "target_ev_uptake": target_ev_uptake,
        "n_steps_for_sweep": n_steps_for_sweep,
        "n_calls": n_calls,
        "noise": noise
    }
    save_object(conditions, file_name + "/Data", "conditions")

    return "Done"

if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
        BOUNDS_LOAD="package/analysis/policy_bounds_vary_pair_policy_gen.json", 
        policy_list_all=[
            "Carbon_price",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Adoption_subsidy_used",
            "Production_subsidy",
        ],
        target_ev_uptake=0.95,
        n_steps_for_sweep=10,
        n_calls=20,
        noise=0.05
    )