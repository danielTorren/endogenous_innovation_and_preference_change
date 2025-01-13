from copy import deepcopy
import json
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from package.resources.run import load_in_controller, parallel_run_multi_run

from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

def generate_policy_scenarios(base_params, policy_list):
    """
    Generate a list of scenarios where policies are set to "High" for each pair of policies
    and for each individual policy.
    FOR NOW ITS 1D
    """

    # Generate parameter sets
    base_params_list = []
    for policy in policy_list:
        base_params_copy = deepcopy(base_params)

        base_params_copy["parameters_policies"]["States"][policy] = "High"
        base_params_list.append(base_params_copy)

    output = [(policy_list[i],base_params_list[i]) for i in range(len(base_params_list))]

    return output

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

def single_policy_simulation(params, controller_load):
    data = load_in_controller(controller_load, params)#FIRST RUN

    EV_uptake = data.calc_EV_prop()
    policy_distortion = data.calc_total_policy_distortion()
    return EV_uptake, policy_distortion


def single_policy_with_seeds(params, controller_list):
    """
    Perform parallel execution of all policy scenarios and seeds.
    """
    num_cores = multiprocessing.cpu_count()

    def run_scenario(scenario_params, controller):
        controller_copy = deepcopy(controller)  # Ensure a clean state for this run
        EV_uptake, total_cost = single_policy_simulation(scenario_params, controller_copy)
        return EV_uptake, total_cost


    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_scenario)(params, controller_list[i])  # No deepcopy here
        for i in range(len(controller_list))
    )

    EV_uptake_list, total_cost_list = zip(
        *res
    )
    print("EV_uptake_list, total_cost_list", EV_uptake_list, total_cost_list)
    return np.asarray(EV_uptake_list), np.asarray(total_cost_list)
###########################################################################################################################

def objective_function_wrapper_manual(intensity_level, params, controller_list, policy_name, target_ev_uptake):
    """
    Wrapper for the objective function to be minimized.
    Args:
        intensity_level (float): Current policy intensity level (single value for this optimization).
        params (dict): Policy parameters.
        controller (object): Controller instance for running simulations.
        policy_name (str): Policy being optimized.
        target_ev_uptake (float): Target EV uptake to achieve.

    Returns:
        float: Value of the objective function to minimize.
    """
    # Update the policy intensity
    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name]["High"] = intensity_level

    # Run simulation
    #base_params_seeds = params_list_with_seed(params)
    EV_uptake_arr, total_cost_arr = single_policy_with_seeds(
        params,# base_params_seeds,
        controller_list
    )

    # Compute mean values
    mean_error = np.mean(target_ev_uptake - EV_uptake_arr)
    mean_EV_uptake = np.mean(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)
    print("EV_uptake_arr", EV_uptake_arr)
    print("mean_error", abs(mean_error))
    # Compute the objective value
    return mean_error, mean_EV_uptake , mean_total_cost

def manual_optimization(params, controller_list, policy_name, intensity_level_init, target_ev_uptake, step_size=0.01, max_iter=100, adaptive_factor=0.5, min_step_size=1e-4, max_step_size=1.0):
    """
    Perform manual optimization with adaptive step size adjustment.

    Args:
        params: Parameters for the objective function.
        controller: Controller object.
        policy_name: Policy name to evaluate.
        intensity_level_init: Initial intensity level.
        target_ev_uptake: Target EV uptake.
        step_size: Initial step size for the optimization.
        max_iter: Maximum number of iterations.
        adaptive_factor: Factor to adjust the step size dynamically.
        min_step_size: Minimum allowable step size.
        max_step_size: Maximum allowable step size.

    Returns:
        Final intensity, error, EV uptake, and total cost.
    """
    intensity = intensity_level_init
    prev_error = None

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}, Intensity: {intensity}, Step Size: {step_size}")

        # Calculate error, EV uptake, and total cost
        error, ev_uptake, total_cost = objective_function_wrapper_manual(
            intensity, params, controller_list, policy_name, target_ev_uptake
        )

        # Convergence check
        if abs(error) < 1e-2:
            print("Converged successfully.")
            break

        # Adjust step size based on the error and previous error
        if prev_error is not None:
            if abs(error) < abs(prev_error):
                step_size = min(step_size * (1 + adaptive_factor), max_step_size)
            else:
                step_size = max(step_size * adaptive_factor, min_step_size)

        # Update intensity using gradient descent step
        intensity += step_size * np.sign(error)

        # Store current error for the next iteration
        prev_error = error

    return intensity, error, ev_uptake, total_cost


def optimize_policy_intensity_minimize(
        params,
        controller_list,
        policy_name,
        intensity_level_init,
        target_ev_uptake=0.9,
        bounds=(0, None),
        max_iterations=100,
        initial_step_size = 0.1,
        adaptive_factor= 0.5, 
        step_size_bounds = [0.1,100]       
):
    """
    Optimizes the intensity of a specific policy to maximize EV uptake using scipy.optimize.minimize.

    Args:
        params (dict): Initial policy parameters.
        controller (object): Controller instance to run simulations.
        policy_name (str): The name of the policy to optimize.
        intensity_level_init (float): Initial intensity level for the policy.
        target_ev_uptake (float): Target mean EV uptake (e.g., 0.9 for 90%).
        bounds (tuple): Bounds for the policy intensity.

    Returns:
        float: Optimized intensity level.
        float: Final mean EV uptake.
        float: Final mean total cost.
    """
    # Define bounds and initial guess
    bounds = [(bounds[0], bounds[1])]  # Single policy intensity bounds
    min_step_size, max_step_size= step_size_bounds
    # Optimize using scipy's minimize
    optimized_intensity, error,mean_ev_uptake, mean_total_cost = manual_optimization(params, controller_list, policy_name, intensity_level_init, target_ev_uptake, step_size=initial_step_size , max_iter=max_iterations, adaptive_factor=adaptive_factor, min_step_size=min_step_size, max_step_size=max_step_size)

    print("Optimized_intensity, error", optimized_intensity, error, error,mean_ev_uptake, mean_total_cost)

    # Extract optimized intensity

    return optimized_intensity, mean_ev_uptake, mean_total_cost

#############################################################################

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
        BOUNDS_LOAD="package/analysis/policy_bounds.json", 
        policy_list = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy"
            ]
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Load base parameters
    with open(BOUNDS_LOAD) as f:
        policy_params_dict = json.load(f)
    
    bounds_dict = policy_params_dict["bounds_dict"]
    step_size_dict  = policy_params_dict["step_size_dict"]

    future_time_steps = base_params["duration_future"]#REMOVE TO RUN SINGLE RUN FOR CONSISTENCY, THEN PUT BACK IN FOR POLICY ANALYSIS
    base_params["duration_future"] = 0   

    root = "endogenous_policy_intensity_single"

    fileName = produce_name_datetime(root)
    print("fileName:", fileName)
    ##################################################################################################
    #RUN BURN IN + CALIBRATION PERIOD FIRST:
    base_params_list = params_list_with_seed(base_params)
    controller_list = parallel_run_multi_run(base_params_list)

    createFolder(fileName)
    #save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    ##################################################################################################
    
    #ONLY NEED TO CHANGE ONE OF THE RUNS AS THE SEED DOESNT AFECT THE POLICY PART!
    base_params["duration_future"] = future_time_steps

    # Generate all pairs of policies
    policy_combinations = generate_policy_scenarios(base_params, policy_list)
    print("TOTAL SCENARIOS: ", len(policy_combinations))

    policy_outcomes = {}

    for i, policy_comb in enumerate(policy_combinations):
        policy_name, params = policy_comb
        print("policy_name",policy_name)
        if policy_name == "Carbon_price":
            intensity_level_init = 2#params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"]
        else:
            intensity_level_init = params["parameters_policies"]["Values"][policy_name]["High"]

        initial_step_size = intensity_level_init*0.01#step size of 10%

        mean_ev_uptake, mean_total_cost, intensity_level = optimize_policy_intensity_minimize(
            params,
            controller_list,
            policy_name,
            intensity_level_init,
            target_ev_uptake=0.9,
            bounds=bounds_dict[policy_name],
            initial_step_size = initial_step_size,
            adaptive_factor= 0.5,
            step_size_bounds = step_size_dict[policy_name]
        )
        
        policy_outcomes[policy_name] = [mean_ev_uptake, mean_total_cost, intensity_level]

    save_object(policy_outcomes, fileName + "/Data", "policy_outcomes")
    save_object(base_params, fileName + "/Data", "base_params")


if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_endogenous_policy_single.json",
        policy_list = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy"
            ]
        )