from copy import deepcopy
import json
from itertools import combinations, product
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
from package.resources.run import generate_data, load_in_controller, parallel_run_multi_run
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
        base_params_copy["parameters_firm_manager"]["init_tech_seed"] = seed + seed_repetitions
        base_params_copy["parameters_ICE"]["landscape_seed"] = seed + 2 * seed_repetitions
        base_params_copy["parameters_EV"]["landscape_seed"] = seed + 9 * seed_repetitions
        base_params_copy["parameters_social_network"]["social_network_seed"] = seed + 3 * seed_repetitions
        base_params_copy["parameters_social_network"]["network_structure_seed"] = seed + 4 * seed_repetitions
        base_params_copy["parameters_social_network"]["init_vals_environmental_seed"] = seed + 5 * seed_repetitions
        base_params_copy["parameters_social_network"]["init_vals_innovative_seed"] = seed + 6 * seed_repetitions
        base_params_copy["parameters_social_network"]["init_vals_price_seed"] = seed + 7 * seed_repetitions
        base_params_copy["parameters_firm"]["innovation_seed"] = seed + 8 * seed_repetitions
       
        base_params_list.append( base_params_copy)
    
    return base_params_list

def generate_grid_policy_scenarios_with_seeds(base_params, policy_list, repetitions, bounds):
    """
    Generate grid search scenarios for pairs of policies with given repetitions and seed variations.
    Args:
        base_params (dict): The base parameters for the simulation.
        policy_list (list): List of policy names to vary.
        repetitions (int): Number of intensity levels to test per policy.
        bounds (dict): Dictionary mapping policy names to (min, max) bounds.

    Returns:
        list: List of parameter sets for all grid combinations.
    """
    # Generate all combinations of policy pairs
    policy_combinations = list(combinations(policy_list, 2))

    # Generate grids for each pair
    scenarios = []
    for policy_pair in policy_combinations:
        # Create grids for each policy in the pair
        grids = []
        for policy in policy_pair:
            min_val, max_val = bounds[policy]
            grids.append(np.linspace(min_val, max_val, repetitions))  # Divide range into equal parts

        # Create the cartesian product of the grids
        for intensity_pair in product(*grids):
            base_params_copy = deepcopy(base_params)
            for i, policy in enumerate(policy_pair):
                if policy == "Carbon_price":
                    base_params_copy["parameters_policies"]["Values"][policy]["High"]["Carbon_price"] = intensity_pair[i]
                else:
                    base_params_copy["parameters_policies"]["Values"][policy]["High"] = intensity_pair[i]

            # Add seed variations to the scenarios
            seed_variations = params_list_with_seed(base_params_copy)
            for seed_params in seed_variations:
                scenarios.append((policy_pair, intensity_pair, seed_params))

    return scenarios


def grid_search_policy_optimization_with_seeds(params, controller_list, policy_list, step_sizes, bounds):
    """
    Perform grid search for pairs of policies with seed variations to optimize output.
    Args:
        params (dict): Base policy parameters.
        controller (object): Controller instance for running simulations.
        policy_list (list): List of policies to consider in pairs.
        step_sizes (dict): Dictionary of step sizes for each policy.
        bounds (dict): Dictionary of bounds for each policy.

    Returns:
        list: Results containing policy pairs, intensity levels, EV uptake, and total cost.
    """

    # Generate all grid search scenarios with seed variations
    grid_scenarios = generate_grid_policy_scenarios_with_seeds(params, policy_list, step_sizes, bounds)
    print("TOTAL RUNS",len(grid_scenarios))
    print("grid_scenarios[0]", grid_scenarios[0], len(grid_scenarios[0]))
    quit()

    results = []

    for i, values in enumerate(grid_scenarios):
        #print(f"Evaluating pair {policy_pair} with intensities {intensity_pair}...")
        (policy_pair, intensity_pair, scenario_params) = values

        controller = controller_list[i % len(controller_list)]
        # Run the ABM simulation
        EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
            [scenario_params],
            controller
        )

        # Compute mean results
        mean_ev_uptake = np.mean(EV_uptake_arr)
        mean_total_cost = np.mean(total_cost_arr)

        results.append({
            "policy_pair": policy_pair,
            "intensity_pair": intensity_pair,
            "mean_ev_uptake": mean_ev_uptake,
            "mean_total_cost": mean_total_cost,
        })

    return results

def find_optimal_policy_pair(grid_results, target_ev_uptake=0.9, lambda_cost=0.1):
    """
    Identify the optimal policy pair and their intensity levels based on grid results.
    Args:
        grid_results (list): List of results from grid search, each containing:
            - "policy_pair": Policy names (tuple).
            - "intensity_pair": Intensity levels (tuple).
            - "mean_ev_uptake": Mean EV uptake (float).
            - "mean_total_cost": Mean total cost (float).
        target_ev_uptake (float): Target EV uptake (e.g., 0.9 for 90%).
        lambda_cost (float): Weight for the cost penalty in the objective function.

    Returns:
        dict: The optimal policy pair and associated data.
    """
    optimal_result = None
    best_metric = float('inf')  # Initialize with infinity for minimization

    for result in grid_results:
        ev_uptake = result["mean_ev_uptake"]
        total_cost = result["mean_total_cost"]
        policy_pair = result["policy_pair"]
        intensity_pair = result["intensity_pair"]

        # Calculate the objective metric
        metric = abs(target_ev_uptake - ev_uptake) + lambda_cost * np.log(total_cost + 1)

        if metric < best_metric:
            best_metric = metric
            optimal_result = {
                "policy_pair": policy_pair,
                "intensity_pair": intensity_pair,
                "mean_ev_uptake": ev_uptake,
                "mean_total_cost": total_cost,
                "objective_metric": metric
            }

    return optimal_result

def main_grid_search_with_optimal_selection(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
        policy_list=[
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy",
        ],
        repetitions = 10,  # Number of intensity levels for each policy
        bounds_LOAD = "package/analysis/policy_bounds.json",
        target_ev_uptake = 0.9,
        lambda_cost = 0.1
):
    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(bounds_LOAD) as f:
        bounds = json.load(f)

    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    fileName = produce_name_datetime("grid_search_policy_intensity_pair_gen")
    print("fileName:", fileName)

    # Run burn-in and calibration
    base_params_list = params_list_with_seed(base_params)
    controller_list = parallel_run_multi_run(base_params_list)
    createFolder(fileName)

    print("FINISHED RUNS")

    save_object(controller_list, fileName + "/Data", "controller_list")
    save_object(base_params, fileName + "/Data", "base_params")

    base_params["duration_future"] = future_time_steps

    print("DONE RUNS SEED")

    # Perform grid search with seed variations
    grid_search_results = grid_search_policy_optimization_with_seeds(
        base_params, controller_list, policy_list, repetitions, bounds
    )

    # Find the optimal policy pair
    optimal_result = find_optimal_policy_pair(grid_search_results, target_ev_uptake, lambda_cost)

    # Save results
    save_object(grid_search_results, fileName + "/Data", "grid_search_results")
    save_object(optimal_result, fileName + "/Data", "optimal_result")
    save_object(policy_list, fileName + "/Data", "policy_list")
    save_object(bounds, fileName + "/Data", "bounds")
    other_params = {"repetitions": repetitions," target_ev_uptake": target_ev_uptake, "lambda_cost": lambda_cost}
    save_object(other_params, fileName + "/Data", "other_params")

    print("Grid search and optimal selection completed.")
    print("Optimal result:", optimal_result)

    return optimal_result


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
