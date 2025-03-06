import json
import numpy as np
from package.analysis.endogenous_policy_intensity_single_gen import optimize_policy_intensity_BO, set_up_calibration_runs, update_policy_intensity
from package.resources.utility import (
    save_object, 
)
import shutil  # Cleanup
from pathlib import Path  # Path handling

def generate_unique_policy_pairs(policy_list_all, policy_list_works):
    """
    Generate unique pairs where:
    - The first policy comes from policy_list_all.
    - The second policy comes from policy_list_works.
    - No duplicates like (A, B) and (B, A) — only the lexicographically first pair is kept.
    - No self-pairs like (A, A).
    """
    pairs = set()  # Using a set to auto-handle duplicates

    for policy1 in policy_list_all:
        for policy2 in policy_list_works:
            if policy1 != policy2:
                pair = tuple(sorted([policy1, policy2]))  # Sort so (A, B) == (B, A)
                pairs.add(pair)

    pairs = list(pairs)
    print("Number of unique pairs:", len(pairs))
    return pairs



def policy_pair_sweep(
    base_params,
    controller_files,
    policy1_name,
    policy2_name,
    bounds_dict,
    target_ev_uptake=0.9,
    n_steps=10,
    n_calls=40, 
    noise = 0.05,
    epsilon = 0.01
):
    """
    Sweep policy1 intensity in steps, optimizing policy2 at each step.
    """
    p1_min, p1_max = bounds_dict[policy1_name]
    p1_values = np.linspace(p1_max*(1-epsilon), p1_min*(1+epsilon), n_steps)
    results_for_pair = []
    
    for p1_val in p1_values:
        print("Policy1, Val: ", policy1_name, p1_val)
        #UPDATE THE BASE PARAMS
        base_params = update_policy_intensity(base_params, policy1_name, p1_val)

        best_intensity, mean_ev_uptake, sd_ev_uptake, mean_total_cost, mean_net_cost, mean_emissions_cumulative, mean_emissions_cumulative_driving, mean_emissions_cumulative_production, mean_utility_cumulative, mean_profit_cumulative = optimize_policy_intensity_BO(
            base_params, controller_files, policy2_name, target_ev_uptake=target_ev_uptake,
            bounds=bounds_dict[policy2_name], n_calls=n_calls, noise = noise
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
            "mean_profit_cumulative": mean_profit_cumulative
        })
    
    return results_for_pair

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    BOUNDS_LOAD="package/analysis/policy_bounds.json", 
    policy_list_all=[
        "Carbon_price",
        "Targeted_research_subsidy",
        "Electricity_subsidy",
        "Adoption_subsidy",
        "Adoption_subsidy_used",
        "Production_subsidy",
        "Research_subsidy"
    ],
    policy_list_works=[
        "Carbon_price",
        "Targeted_research_subsidy",
        "Electricity_subsidy"
    ],
    target_ev_uptake=0.9,
    n_steps_for_sweep=5,
    n_calls=40, 
    noise = 0.01,
    epsilon = 0.01
):
    """
    Main function for running pairwise policy optimization.
    """
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        bounds_dict = json.load(f)

    policy_pairs = generate_unique_policy_pairs(policy_list_all, policy_list_works)
    print("Pairs: ", policy_pairs)
    controller_files, base_params, file_name = set_up_calibration_runs(base_params)

    ###################################################################################################################
        

    #RUN POLICY OUTCOMES
    print("TOTAL RUNS BO: ", len(policy_pairs)*n_steps_for_sweep*n_calls*base_params["seed_repetitions"])

    pairwise_outcomes = {}
    
    for (policy1_name, policy2_name) in policy_pairs:
        print(f"\n=== Optimizing Pair: ({policy1_name}, {policy2_name}) ===")
        results = policy_pair_sweep(
            base_params,
            controller_files,
            policy1_name,
            policy2_name,
            bounds_dict,
            target_ev_uptake,
            n_steps_for_sweep,
            n_calls, 
            noise,
            epsilon
        )
        pairwise_outcomes[(policy1_name, policy2_name)] = results
    
    save_object(pairwise_outcomes, file_name + "/Data", "pairwise_outcomes")#

    print("Optimization Complete. Data saved.")

    shutil.rmtree(Path(file_name) / "Calibration_runs", ignore_errors=True)

    conditions = {
        "policy_list_all": policy_list_all,
        "policy_list_works": policy_list_works,
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
            #"Targeted_research_subsidy",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Adoption_subsidy_used",
            "Production_subsidy",
            #"Research_subsidy"
        ],
        policy_list_works=[
            "Carbon_price",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy"
            ],
        target_ev_uptake=0.95,
        n_steps_for_sweep=6,
        n_calls=20,
        noise=0.05,
        epsilon = 0.03
    )