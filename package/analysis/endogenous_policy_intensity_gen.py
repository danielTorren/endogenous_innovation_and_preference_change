from copy import deepcopy
import json
from itertools import combinations
import numpy as np
import numpy.typing as npt
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
    policy_combinations = [(policy,) for policy in policy_list]
    
    # Generate parameter sets
    base_params_list = []
    for policy_set in policy_combinations:
        base_params_copy = deepcopy(base_params)
        # Set the level of each policy in the combination to "High"
        for policy in policy_set:
            base_params_copy["parameters_policies"]["States"][policy] = "High"
        base_params_list.append(base_params_copy)

    output = [(policy_combinations[i],base_params_list[i]) for i in range(len(base_params_list))]

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

#DO FOR A BUNCH OF RUNS IN PARALLEL FOR DIFFERNT SEEDS, IF MORE THAN X% have EV UPTAKE ABOVE threshold then decrease intensity, otherwise increase intensity
#sum of entire policy cost
#ev up take at the end of the simulation
#average of total utility for all users for all time steps
#average of total profit per time step for all time steps
#average of utility gini for all time step

def objective_function (target_EV_uptake, EV_uptake, policy_distortion):
    """Want to minimise the seperation between the ev target and simulation value and then dependent on the log of the policy cost"""
    return target_EV_uptake - EV_uptake + np.log(policy_distortion)

def single_policy_simulation(params, controller_load):
    data = load_in_controller(controller_load, params)#FIRST RUN
    EV_uptake = data.social_network.EV_users/data.social_network.social_network.num_individuals,
    policy_distortion = data.calc_total_policy_distortion()
    return EV_uptake, policy_distortion

def single_policy_multi_seed_run(
        params_list,
        controller
) -> npt.NDArray:
    #res = [single_policy_simulation(params_dict[i],deepcopy(controller)) for i in range(len(params_dict))]
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(single_policy_simulation)(i, params_list[i], deepcopy(controller) ) for i in range(len(params_list)))
    EV_uptake_list, total_cost_list = zip(
        *res
    )
    return np.asarray(EV_uptake_list), np.asarray(total_cost_list)

def optimize_policy_intensity_hill_climbing(
        params,
        controller,
        policy_name,
        intensity_level_init,
        target_ev_uptake=0.9,
        max_iterations=100,
        tol=1e-3,
        initial_step_size=0.1,
        min_step_size=1e-4
):
    """
    Optimizes the intensity of a specific policy to maximize EV uptake above a threshold
    using a hill-climbing algorithm with adaptive step size.

    Args:
        params (dict): Initial policy parameters.
        controller (object): Controller instance to run simulations.
        policy_name (str): The name of the policy to optimize.
        intensity_level_init (float): Initial intensity level for the policy.
        target_ev_uptake (float): Target mean EV uptake (e.g., 0.9 for 90%).
        max_iterations (int): Maximum number of optimization iterations.
        tol (float): Proximity tolerance for EV uptake threshold.
        initial_step_size (float): Initial multiplicative step size for adjustments.
        min_step_size (float): Minimum allowable step size for stopping fine adjustments.

    Returns:
        dict: Optimized policy parameters.
        float: Final mean EV uptake.
        float: Final mean total cost.
        float: Final policy intensity.
    """
    controller_load_copy = deepcopy(controller)
    intensity_level = intensity_level_init  # Starting policy intensity
    step_size = initial_step_size  # Start with initial step size
    base_params_seeds = params_list_with_seed(params)

    for iteration in range(max_iterations):
        # Run simulation
        EV_uptake_arr, total_cost_arr = single_policy_multi_seed_run(
            base_params_seeds,
            controller_load_copy
        )
        
        # Compute mean EV uptake and total cost
        mean_ev_uptake = np.mean(EV_uptake_arr)
        mean_total_cost = np.mean(total_cost_arr)

        print(f"Iteration {iteration}: Intensity {intensity_level:.3f}, "
              f"Step Size {step_size:.3f}, "
              f"Mean EV Uptake {mean_ev_uptake:.3f}, Mean Total Cost {mean_total_cost:.2f}")

        # Check stopping condition
        if abs(mean_ev_uptake - target_ev_uptake) <= tol and mean_ev_uptake >= target_ev_uptake:
            print("Target EV uptake threshold reached within tolerance.")
            break

        # Calculate adjustment direction
        if mean_ev_uptake < target_ev_uptake:
            intensity_level += step_size  # Increment intensity
        else:
            intensity_level -= step_size  # Decrement intensity

        # Adjust step size adaptively (reduce it as we get closer to the target)
        step_size = max(initial_step_size * abs(mean_ev_uptake - target_ev_uptake) / target_ev_uptake, min_step_size)

        # Update policy intensity in parameters
        if policy_name == "Carbon_price":
            for params in base_params_seeds:
                params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"] = intensity_level
        else:
            for params in base_params_seeds:
                params["parameters_policies"]["Values"][policy_name]["High"] = intensity_level

    # Return optimized parameters and results
    return mean_ev_uptake, mean_total_cost, intensity_level



def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
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
    
    future_time_steps = base_params["duration_future"]#REMOVE TO RUN SINGLE RUN FOR CONSISTENCY, THEN PUT BACK IN FOR POLICY ANALYSIS
    base_params["duration_future"] = 0   

    root = "endogenous_polict_intensity"

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


    for (policy_name, params) in policy_combinations:
        
        if policy_name == "Carbon_price":
            intensity_level_init = params["parameters_policies"]["Values"][policy_name]["High"]["Carbon_price"]
        else:
            intensity_level_init = params["parameters_policies"]["Values"][policy_name]["High"]

        initial_step_size = intensity_level_init*0.01#step size of 1%

        mean_ev_uptake, mean_total_cost, intensity_level = optimize_policy_intensity_hill_climbing(
            params,
            controller,
            policy_name,
            intensity_level_init,
            target_ev_uptake=0.9,
            max_iterations=100,
            tol=1e-3,
            initial_step_size=initial_step_size,
            min_step_size=1e-4
        )
        policy_outcomes[policy_name] = [mean_ev_uptake, mean_total_cost, intensity_level ]

    save_object(policy_outcomes, fileName + "/Data", "policy_outcomes")
    save_object(base_params, fileName + "/Data", "base_params")


if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_endogenous_policy.json",
        policy_list = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy"
            ]
        )