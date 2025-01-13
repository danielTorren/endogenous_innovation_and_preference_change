from copy import deepcopy
import json
from itertools import combinations
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
from package.resources.run import generate_data,load_in_controller
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
    # Generate all combinations of policy pairs
    #policy_combinations = list(combinations(policy_list, 2))
    
    # Add single-policy scenarios
    #policy_combinations += [(policy,) for policy in policy_list]

    # Generate parameter sets
    base_params_list = []
    for policy in policy_list:
        base_params_copy = deepcopy(base_params)
        #print("policy!!!!!!", policy)
        base_params_copy["parameters_policies"]["States"][policy] = "High"
        base_params_list.append(base_params_copy)
        #print("polciies",policy,base_params_copy["parameters_policies"]["States"][policy])

    output = [(policy_list[i],base_params_list[i]) for i in range(len(base_params_list))]
    #quit()
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
    #print("INSIDE, init ev prop", controller_load.calc_EV_prop() )
    #print("intensity_level",params["parameters_policies"]["Values"]["Carbon_price"]["High"]["Carbon_price"])
    data = load_in_controller(controller_load, params)#FIRST RUN
    #print("RUN POLICY")
    EV_uptake = data.calc_EV_prop()
    policy_distortion = data.calc_total_policy_distortion()
    return EV_uptake, policy_distortion

def single_policy_multi_seed_run(
        params_list,
        controller
) -> npt.NDArray:
    #res = [single_policy_simulation(params_list[i],deepcopy(controller)) for i in range(len(params_list))]
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(single_policy_simulation)(params_list[i], deepcopy(controller)) for i in range(len(params_list)))
    EV_uptake_list, total_cost_list = zip(
        *res
    )
    return np.asarray(EV_uptake_list), np.asarray(total_cost_list)

###########################################################################################################################

def objective_function_wrapper(intensity_level, params, controller, policy_name, target_ev_uptake):
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
        params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"] = intensity_level[0]
    else:
        params["parameters_policies"]["Values"][policy_name]["High"] = intensity_level[0]

    print("intensity_level", intensity_level)

    # Run simulation
    base_params_seeds = params_list_with_seed(params)
    EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
        base_params_seeds,
        controller
    )

    # Compute mean values
    mean_error = np.mean(target_ev_uptake - EV_uptake_arr)
    #mean_total_cost = np.mean(total_cost_arr)
    print("EV_uptake_arr", EV_uptake_arr)
    print("mean_error", abs(mean_error))
    # Compute the objective value
    return abs(mean_error) #+ np.log(mean_total_cost)

def objective_function_wrapper_manual(intensity_level, params, controller, policy_name, target_ev_uptake):
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
    base_params_seeds = params_list_with_seed(params)
    EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
        base_params_seeds,
        controller
    )

    # Compute mean values
    mean_error = np.mean(target_ev_uptake - EV_uptake_arr)
    mean_EV_uptake = np.mean(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)
    print("EV_uptake_arr", EV_uptake_arr)
    print("mean_error", abs(mean_error))
    # Compute the objective value
    return mean_error, mean_EV_uptake , mean_total_cost

def manual_optimization(params, controller, policy_name, intensity_level_init, target_ev_uptake, step_size=0.01, max_iter=100):
    intensity = intensity_level_init
    for _ in range(max_iter):
        print("intensity",intensity)
        error, ev_uptake, total_cost = objective_function_wrapper_manual(intensity, params, controller, policy_name, target_ev_uptake)
        new_intensity = intensity + (step_size * np.sign(error)) # Simple gradient descent update
        if abs(error) < 1e-2:  # Convergence check
            break
        intensity = new_intensity
    return intensity, error, ev_uptake, total_cost

def optimize_policy_intensity_minimize(
        params,
        controller,
        policy_name,
        intensity_level_init,
        target_ev_uptake=0.9,
        bounds=(0, None),
        max_iterations=100,
        initial_step_size = 0.1
        
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

    # Optimize using scipy's minimize
    optimized_intensity, error,mean_ev_uptake, mean_total_cost = manual_optimization(params, controller, policy_name, intensity_level_init, target_ev_uptake, step_size=initial_step_size , max_iter= max_iterations)
    print("Optimized_intensity, error", optimized_intensity, error, error,mean_ev_uptake, mean_total_cost)

    """
    result = minimize(
        fun=objective_function_wrapper,
        x0=initial_guess,
        args=(params, controller, policy_name, target_ev_uptake),
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "eps": 1e-2,      # Increase step size for numerical gradient
            "ftol": 1e-6,
            "gtol": 1e-6,
            "disp": True         # Display optimization progress
        }
    )
    optimized_intensity = result.x[0]
    print("Optimized_intensity", optimized_intensity)
    # Run simulation with optimized intensity to get final results
    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"] = optimized_intensity
    else:
        params["parameters_policies"]["Values"][policy_name]["High"] = optimized_intensity

    base_params_seeds = params_list_with_seed(params)
    
    EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
        base_params_seeds,
        controller
    )

    # Compute final mean values
    mean_ev_uptake = np.mean(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)
    """

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
        bounds_dict = json.load(f)
    
    future_time_steps = base_params["duration_future"]#REMOVE TO RUN SINGLE RUN FOR CONSISTENCY, THEN PUT BACK IN FOR POLICY ANALYSIS
    base_params["duration_future"] = 0   

    root = "endogenous_policy_intensity_single"

    fileName = produce_name_datetime(root)
    print("fileName:", fileName)
    ##################################################################################################
    #RUN BURN IN + CALIBRATION PERIOD FIRST:
    controller = generate_data(base_params)  # run the simulation 

    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    ##################################################################################################
    
    base_params["duration_future"] = future_time_steps

    # Generate all pairs of policies
    policy_combinations = generate_policy_scenarios(base_params, policy_list)
    print("TOTAL SCENARIOS: ", len(policy_combinations))

    policy_outcomes = {}

    for i, policy_comb in enumerate(policy_combinations):
        policy_name, params = policy_comb
        print("policy_name",policy_name)
        quit
        if policy_name == "Carbon_price":
            intensity_level_init = 2#params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"]
        else:
            intensity_level_init = params["parameters_policies"]["Values"][policy_name]["High"]

        initial_step_size = intensity_level_init*0.1#step size of 10%

        mean_ev_uptake, mean_total_cost, intensity_level = optimize_policy_intensity_minimize(
            params,
            controller,
            policy_name,
            intensity_level_init,
            target_ev_uptake=0.9,
            bounds=bounds_dict[policy_name],
            initial_step_size = initial_step_size 
        )
        
        policy_outcomes[policy_name] = [mean_ev_uptake, mean_total_cost, intensity_level]

    save_object(policy_outcomes, fileName + "/Data", "policy_outcomes")
    save_object(base_params, fileName + "/Data", "base_params")


if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_endogenous_policy_single.json",
        policy_list = [
            "Carbon_price",
            #"Discriminatory_corporate_tax",
            #"Electricity_subsidy",
            #"Adoption_subsidy",
            #"Production_subsidy",
            #"Research_subsidy"
            ]
        )