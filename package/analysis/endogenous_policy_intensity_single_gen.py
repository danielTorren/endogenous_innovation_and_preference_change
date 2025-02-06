from copy import deepcopy
import json
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from package.resources.run import load_in_controller, parallel_run_multi_run
from scipy.stats import norm
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
        base_params_copy["seeds"]["init_vals_poisson_seed"] = seed + 12 * seed_repetitions
        
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

def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for a given dataset and confidence level.
    """
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error of the mean
    margin = sem * norm.ppf((1 + confidence) / 2)
    return mean - margin, mean + margin

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
    conf_int = compute_confidence_interval(EV_uptake_arr, confidence=0.95)

    print("EV_uptake_arr", EV_uptake_arr)
    print("mean_error", abs(mean_error))
    print("conf_int", conf_int)
    

    # Compute the objective value
    return mean_error, mean_EV_uptake , mean_total_cost, conf_int, EV_uptake_arr, total_cost_arr

def manual_optimization(bounds, params, controller_list, policy_name, intensity_level_init, target_ev_uptake, step_size=0.01, max_iter=100, adaptive_factor=0.5, min_step_size=1e-4, max_step_size=1.0):
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
    at_boundary = False 

    data = []
    
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}, Intensity: {intensity}, Step Size: {step_size}")

        # Calculate error, EV uptake, and total cost
        error, ev_uptake, total_cost, conf_int , EV_uptake_arr, total_cost_arr= objective_function_wrapper_manual(
            intensity, params, controller_list, policy_name, target_ev_uptake
        )

        data.append([error, ev_uptake, total_cost, conf_int, EV_uptake_arr, total_cost_arr, intensity, step_size])

        # Convergence check
        if conf_int[0] <= target_ev_uptake <= conf_int[1]:
        #if abs(error) < 1e-2:
            print("Converged successfully.")
            print("Final: error, ev_uptake, total_cost, conf_int", error, ev_uptake, total_cost, conf_int)
            break

        # Adjust step size based on the error and previous error
        if prev_error is not None:
            if abs(error) < abs(prev_error):
                step_size = min(step_size * (1 + adaptive_factor), max_step_size)
            else:
                step_size = max(step_size * adaptive_factor, min_step_size)

        # Update intensity using gradient descent step 
        next_intensity = intensity + step_size * np.sign(error)
        
        # Check for boundary conditions
        if next_intensity > bounds[1]:
            intensity = bounds[1]
            if at_boundary:
                print("Reached upper boundary again. Stopping optimization.")
                break
            at_boundary = True
        elif next_intensity < bounds[0]:
            intensity = bounds[0]
            if at_boundary:
                print("Reached lower boundary again. Stopping optimization.")
                break
            at_boundary = True
        else:
            intensity = next_intensity
            at_boundary = False  # Reset boundary flag if we're within bounds

        # Store current error for the next iteration
        prev_error = error

    return intensity, error, ev_uptake, total_cost, data


def optimize_policy_intensity_minimize(
        params,
        controller_list,
        policy_name,
        intensity_level_init,
        target_ev_uptake=0.9,
        bounds=(0, None),
        max_iterations=30,
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
    min_step_size, max_step_size= step_size_bounds
    # Optimize using scipy's minimize
    optimized_intensity, error,mean_ev_uptake, mean_total_cost, data = manual_optimization(bounds, params, controller_list, policy_name, intensity_level_init, target_ev_uptake, step_size=initial_step_size , max_iter=max_iterations, adaptive_factor=adaptive_factor, min_step_size=min_step_size, max_step_size=max_step_size)

    print("Policy: Optimized_intensity, error, mean_ev_uptake, mean_total_cost: ", policy_name,optimized_intensity, error, mean_ev_uptake, mean_total_cost)

    # Extract optimized intensity

    return optimized_intensity, mean_ev_uptake, mean_total_cost, data

#############################################################################

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
        BOUNDS_LOAD="package/analysis/policy_bounds.json", 
        policy_list = [
            "Discriminatory_corporate_tax",
            "Carbon_price",
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
    runs_data = {}

    for i, policy_comb in enumerate(policy_combinations):
        policy_name, params = policy_comb
        print("policy_name",policy_name)

        initial_step_size = policy_params_dict["init_val_dict"][policy_name]*0.1#step size of 10%

        intensity_level, mean_ev_uptake, mean_total_cost, policy_data = optimize_policy_intensity_minimize(
            params,
            controller_list,
            policy_name,
            intensity_level_init = policy_params_dict["init_val_dict"][policy_name],
            target_ev_uptake=0.9,
            bounds=bounds_dict[policy_name],
            initial_step_size = initial_step_size,
            adaptive_factor= 0.5,
            step_size_bounds = step_size_dict[policy_name]
        )

        runs_data[policy_name] = policy_data
        
        policy_outcomes[policy_name] = [mean_ev_uptake, mean_total_cost, intensity_level]

    save_object(policy_outcomes, fileName + "/Data", "policy_outcomes")
    save_object(runs_data, fileName + "/Data", "runs_data")
    save_object(base_params, fileName + "/Data", "base_params")


if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_endogenous_policy_single_gen.json",
        policy_list = [
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Carbon_price",
            ],
            BOUNDS_LOAD="package/analysis/policy_bounds_vary_single_policy_gen.json"
        )