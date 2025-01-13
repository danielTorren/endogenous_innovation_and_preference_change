from copy import deepcopy
import json
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from package.resources.run import parallel_run_multi_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
from package.analysis.endogenous_policy_intensity_single_gen import single_policy_multi_seed_run

def params_list_with_seed(base_params):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    base_params_list = []
    seed_repetitions = base_params["seed_repetitions"]

    for seed in range(1, seed_repetitions + 1):
        base_params_copy = deepcopy(base_params)
        # VARY ALL THE SEEDS
        base_params_copy["seeds"]["init_tech_seed"] = seed + seed_repetitions
        base_params_copy["seeds"]["landscape_seed_ICE"] = seed + 2 * seed_repetitions
        base_params_copy["seeds"]["social_network_seed"] = seed + 3 * seed_repetitions
        base_params_copy["seeds"]["network_structure_seed"] = seed + 4 * seed_repetitions
        base_params_copy["seeds"]["init_vals_environmental_seed"] = seed + 5 * seed_repetitions
        base_params_copy["seeds"]["init_vals_innovative_seed"] = seed + 6 * seed_repetitions
        base_params_copy["seeds"]["init_vals_price_seed"] = seed + 7 * seed_repetitions
        base_params_copy["seeds"]["innovation_seed"] = seed + 8 * seed_repetitions
        base_params_copy["seeds"]["landscape_seed_EV"] = seed + 9 * seed_repetitions
        base_params_copy["seeds"]["choice_seed"] = seed + 10 * seed_repetitions
        base_params_copy["seeds"]["remove_seed"] = seed + 11 * seed_repetitions
       
        base_params_list.append( base_params_copy)
    
    return base_params_list

def objective_function(intensities, policy_pair, base_params, controller_list, target_ev_uptake, lambda_cost):
    """
    Objective function to minimize.
    Args:
        intensities (list): Current policy intensity levels being evaluated.
        policy_pair (tuple): Pair of policies being optimized.
        base_params (dict): Base simulation parameters.
        controller_list (list): List of controller objects.
        target_ev_uptake (float): Target EV uptake.
        lambda_cost (float): Weight for cost in the objective.

    Returns:
        float: Value of the objective function to minimize.
    """
    # Update base parameters with current intensities
    for i, policy in enumerate(policy_pair):
        if policy == "Carbon_price":
            base_params["parameters_policies"]["Values"][policy]["High"]["Carbon_price"] = intensities[i]
        else:
            base_params["parameters_policies"]["Values"][policy]["High"] = intensities[i]

    # Run simulations for the updated parameters
    EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
        params_list_with_seed(base_params),
        controller_list
    )

    # Compute mean results
    mean_ev_uptake = np.mean(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)

    # Objective: Minimize the deviation from target EV uptake + cost penalty
    return abs(target_ev_uptake - mean_ev_uptake) + lambda_cost * np.log(mean_total_cost + 1)

def optimize_policy_pair(base_params, controller_list, policy_pair, bounds, initial_guess, target_ev_uptake, lambda_cost):
    """
    Optimize a pair of policies using a numerical optimization method.
    Args:
        base_params (dict): Base simulation parameters.
        controller_list (list): List of controller objects.
        policy_pair (tuple): Pair of policies to optimize.
        bounds (dict): Bounds for each policy's intensity.
        initial_guess (list): Initial intensity levels for the policy pair.
        target_ev_uptake (float): Target EV uptake.
        lambda_cost (float): Weight for cost in the objective.

    Returns:
        dict: Optimization results containing optimal intensities and metrics.
    """
    # Define bounds for optimization
    policy_bounds = [bounds[policy] for policy in policy_pair]

    # Optimize using scipy.optimize.minimize
    result = minimize(
        objective_function,
        x0=initial_guess,
        args=(policy_pair, base_params, controller_list, target_ev_uptake, lambda_cost),
        bounds=policy_bounds,
        method="L-BFGS-B",
        options={"disp": True}
    )

    # Extract the optimized intensities
    optimal_intensities = result.x

    # Run the simulation for the optimal intensities to get final metrics
    for i, policy in enumerate(policy_pair):
        if policy == "Carbon_price":
            base_params["parameters_policies"]["Values"][policy]["High"]["Carbon_price"] = optimal_intensities[i]
        else:
            base_params["parameters_policies"]["Values"][policy]["High"] = optimal_intensities[i]

    EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
        params_list_with_seed(base_params),
        controller_list
    )

    mean_ev_uptake = np.mean(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)

    return {
        "policy_pair": policy_pair,
        "optimal_intensities": optimal_intensities,
        "mean_ev_uptake": mean_ev_uptake,
        "mean_total_cost": mean_total_cost,
        "success": result.success,
        "message": result.message
    }

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
        policy_list=[
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy",
        ],
        bounds_LOAD="package/analysis/policy_bounds.json",
        target_ev_uptake=0.9,
        lambda_cost=0.1,
        initial_guess_LOAD = "package/analysis/minimize_init_guess.json"  # Example initial intensity for all policies
):
    # Load base parameters and bounds
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(bounds_LOAD) as f:
        bounds = json.load(f)
    
    with open(initial_guess_LOAD) as f:
        initial_guess_dict = json.load(f)

    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    # Burn-in and calibration
    base_params_list = params_list_with_seed(base_params)
    controller_list = parallel_run_multi_run(base_params_list)

    # Create folder for results
    fileName = produce_name_datetime("policy_pair_optimization")
    createFolder(fileName)

    # Save initial data
    save_object(controller_list, fileName + "/Data", "controller_list")
    save_object(base_params, fileName + "/Data", "base_params")

    base_params["duration_future"] = future_time_steps

    # Optimize each pair of policies
    policy_combinations = list(combinations(policy_list, 2))
    results = []

    for policy_pair in policy_combinations:
        print(f"Optimizing policy pair: {policy_pair}")
        
        initial_values = [initial_guess_dict[policy_pair[0]],initial_guess_dict[policy_pair[1]]]
        print("INIT GUESS", initial_values)

        result = optimize_policy_pair(
            base_params, controller_list, policy_pair, bounds, initial_values, target_ev_uptake, lambda_cost
        )
        results.append(result)

    # Save results
    save_object(results, fileName + "/Data", "optimization_results")
    save_object(policy_list, fileName + "/Data", "policy_list")
    save_object(bounds, fileName + "/Data", "bounds")

    print("Optimization completed.")
    return results


if __name__ == "__main__":
    optimal = main_grid_search_with_optimal_selection(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair.json",
        policy_list=[
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy",
        ],
        repetitions=10,  # Number of intensity levels for each policy
        bounds_LOAD="package/analysis/policy_bounds.json",
        target_ev_uptake=0.9,
        lambda_cost=0.1
    )
