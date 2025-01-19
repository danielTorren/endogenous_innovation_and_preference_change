import json
import numpy as np
import itertools
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from package.analysis.endogenous_policy_intensity_single_gen import params_list_with_seed, optimize_policy_intensity_minimize
from package.resources.run import load_in_controller, parallel_run_multi_run

from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

def generate_unique_policy_pairs(policy_list_all, policy_list_works):
    """
    Generate pairs of policies where:
    - The first policy can be any policy from policy_list_all.
    - The second policy is only from policy_list_works.
    - Policies are not paired with themselves.
    - Pairs are unique and not reversed (e.g., (A, B) but not (B, A)).
    """
    pairs = []
    for policy1 in policy_list_all:
        for policy2 in policy_list_works:
            if policy1 != policy2 and policy1 < policy2:  # Enforce consistent ordering
                pairs.append((policy1, policy2))

    print("num pairs", len(pairs))
    return pairs
###############################################################################
# New function: For a *fixed* policy1 intensity, optimize policy2 to meet the EV target
###############################################################################
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
    Fix policy1 to 'policy1_intensity' in the params, then optimize policy2
    to achieve 'target_ev_uptake' using the existing single-policy approach.
    
    Returns:
        optimized_intensity2, mean_ev_uptake2, mean_total_cost2
    """
    # We make a copy of params so as not to overwrite the original
    params_copy = deepcopy(params)
    
    # 1) Fix policy1 in params_copy
    if policy1_name == "Carbon_price":
        # Carbon_price has a nested dict, e.g. params["Values"]["Carbon_price"]["High"]["Carbon_price"]
        params_copy["parameters_policies"]["Values"][policy1_name]["High"]["Carbon_price"] = policy1_intensity
    else:
        params_copy["parameters_policies"]["Values"][policy1_name]["High"] = policy1_intensity

    # 2) Now optimize policy2 using your existing optimize_policy_intensity_minimize
    optimized_intensity2, mean_ev_uptake2, mean_total_cost2 = optimize_policy_intensity_minimize(
        params_copy,
        controller_list,
        policy2_name,
        intensity_level_init = policy2_init_guess,
        target_ev_uptake    = target_ev_uptake,
        bounds             = bounds_dict[policy2_name],
        initial_step_size  = policy2_init_guess*0.01,  # Example step size
        adaptive_factor    = 0.5,
        step_size_bounds   = step_size_dict[policy2_name]
    )
    
    return optimized_intensity2, mean_ev_uptake2, mean_total_cost2

###############################################################################
# New function: Sweep policy1 from min->max, for each step fix policy1,
# then optimize policy2 to find the needed intensity to reach EV target.
###############################################################################
def policy_pair_sweep(
    base_params,
    controller_list,
    policy1_name,
    policy2_name,
    bounds_dict,
    step_size_dict,
    target_ev_uptake = 0.9,
    n_steps = 10
):
    """
    For the pair (policy1_name, policy2_name):
      - Use the single-policy outcomes to get initial guesses.
      - Sweep policy1 from min->max in n_steps.
      - For each step, fix policy1 at that intensity, then 
        optimize policy2 to hit the target EV uptake.
      - Use the previous solution for policy2 as the next initial guess.
      - Return all (policy1_intensity, policy2_intensity, ev_uptake, total_cost).
    """

    # min->max bounds for each policy
    p1_min, p1_max = bounds_dict[policy1_name]

    # Create the array of policy1 intensities to test.
    # Sometimes you might want a logspace or custom steps if the range is large.
    p1_values = np.linspace(p1_max,p1_min, n_steps)#START AT THE BOUNDARY P VALUE

    # Initialize the guess for policy2 from single-policy optimum
    p2_guess = bounds_dict[policy2_name][1]#start at the edge

    # We'll store the results for this pair in a list of dicts
    results_for_pair = []

    for i, p1_val in enumerate(p1_values):
        # For each step, fix policy1 = p1_val, then find best policy2
        optimized_p2, ev_uptake2, cost2 = optimize_second_policy_with_first_fixed(
            base_params,
            controller_list,
            policy1_name,
            policy2_name,
            policy1_intensity    = p1_val,
            policy2_init_guess   = p2_guess,
            target_ev_uptake     = target_ev_uptake,
            bounds_dict          = bounds_dict,
            step_size_dict       = step_size_dict
        )

        # Save the results
        results_for_pair.append({
            "policy1_value": p1_val,
            "policy2_value": optimized_p2,
            "mean_ev_uptake": ev_uptake2,
            "mean_total_cost": cost2
        })

        # Update p2_guess so next iteration uses the *new* solution for p2
        p2_guess = optimized_p2

    return results_for_pair

###############################################################################
# Example 'main' that integrates everything
###############################################################################
def main(
    BASE_PARAMS_LOAD = "package/constants/base_params_run_scenario_seeds.json",
    BOUNDS_LOAD      = "package/analysis/policy_bounds.json", 
    policy_list_all  = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Adoption_subsidy_used",
            "Production_subsidy",
            "Research_subsidy"
        ],
    policy_list_works = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy"
        ],
    target_ev_uptake = 0.9,
    n_steps_for_sweep = 5
):
    # 1) Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        policy_params_dict = json.load(f)

    bounds_dict = policy_params_dict["bounds_dict"]
    step_size_dict = policy_params_dict["step_size_dict"]

    # 2) Possibly run the calibration or "burn-in"
    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    root = "endogenous_policy_intensity_single"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

        # 4) Now do *pairwise* sweeps
    policy_pairs = generate_unique_policy_pairs(policy_list_all, policy_list_works)
    print("All unique pairs:", policy_pairs)

    # Run the burn-in with seeds
    base_params_list = params_list_with_seed(base_params)
    controller_list  = parallel_run_multi_run(base_params_list)

    createFolder(fileName)
    save_object(base_params, fileName + "/Data", "base_params_preburn")

    # Restore future duration
    print("Future_time_steps", future_time_steps)
    base_params["duration_future"] = future_time_steps

    # 4) Now do *pairwise* sweeps
    policy_pairs = generate_unique_policy_pairs(policy_list_all, policy_list_works)
    print("All unique pairs:", policy_pairs)

    pairwise_outcomes = {}
    for (p1, p2) in policy_pairs:
        print(f"\n=== Sweeping pair: ({p1}, {p2}) ===")
        results = policy_pair_sweep(
            base_params      = base_params,
            controller_list  = controller_list,
            policy1_name     = p1,
            policy2_name     = p2,
            bounds_dict      = bounds_dict,
            step_size_dict   = step_size_dict,
            target_ev_uptake = target_ev_uptake,
            n_steps          = n_steps_for_sweep
        )
        pairwise_outcomes[(p1, p2)] = results

    # 5) Save everything
    save_object(pairwise_outcomes, fileName + "/Data", "pairwise_outcomes")

    return "Done"

###############################################################################
# If you wanted to run directly:
###############################################################################
if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD = "package/constants/base_params_endogenous_policy_pair_gen.json",
        BOUNDS_LOAD = "package/analysis/policy_bounds_vary_pair_policy_gen.json", 
        policy_list_all  = [
                "Carbon_price",
                "Discriminatory_corporate_tax",
                "Electricity_subsidy",
                "Adoption_subsidy",
                "Adoption_subsidy_used",
                "Production_subsidy",
                "Research_subsidy"
            ],
        policy_list_works = [
                "Carbon_price",
                "Discriminatory_corporate_tax",
                "Electricity_subsidy"
            ],
        target_ev_uptake   = 0.9,
        n_steps_for_sweep  = 10
    )
