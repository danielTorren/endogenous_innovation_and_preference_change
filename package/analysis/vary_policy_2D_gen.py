from copy import deepcopy
import json
import numpy as np
from joblib import Parallel, delayed, dump, load
import multiprocessing
from package.resources.run import load_in_controller, generate_data
from package.resources.utility import createFolder, save_object, produce_name_datetime, params_list_with_seed
import shutil
from pathlib import Path
from itertools import product

def generate_two_policy_scenarios_with_seeds(base_params, policy_pairs, repetitions, bounds):
    """
    Generate grid search scenarios for pairs of policies with given repetitions and seed variations.
    """
    scenarios = []
    
    for policy1, policy2 in policy_pairs:
        min_val1, max_val1 = bounds[policy1]
        min_val2, max_val2 = bounds[policy2]
        
        intensities1 = np.linspace(min_val1, max_val1, repetitions)
        intensities2 = np.linspace(min_val2, max_val2, repetitions)
        
        # Create all combinations of intensities for the two policies
        for intensity1, intensity2 in product(intensities1, intensities2):
            base_params_copy = deepcopy(base_params)
            
            # Turn on both policies
            base_params_copy["parameters_policies"]["States"][policy1] = 1
            base_params_copy["parameters_policies"]["States"][policy2] = 1
            
            # Set policy intensities
            if policy1 == "Carbon_price":
                base_params_copy["parameters_policies"]["Values"][policy1]["Carbon_price"] = intensity1
            else:
                base_params_copy["parameters_policies"]["Values"][policy1] = intensity1
                
            if policy2 == "Carbon_price":
                base_params_copy["parameters_policies"]["Values"][policy2]["Carbon_price"] = intensity2
            else:
                base_params_copy["parameters_policies"]["Values"][policy2] = intensity2
            
            # Add seed variations
            seed_variations = params_list_with_seed(base_params_copy)
            scenarios.extend(seed_variations)

    return scenarios

def two_policy_simulation(params, controller_load):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    data = load_in_controller(controller_load, params)
    return [data.calc_EV_prop(), data.calc_total_policy_distortion(), data.calc_net_policy_distortion(), 
            data.social_network.emissions_cumulative, data.social_network.emissions_cumulative_driving, 
            data.social_network.emissions_cumulative_production, data.social_network.utility_cumulative, 
            data.firm_manager.profit_cumulative]

def grid_search_two_policies_with_seeds(grid_scenarios, controller_files):
    """
    Perform parallel execution of all policy scenarios and seeds,
    ensuring each run starts from a fresh copy of the calibrated controller.
    """
    num_cores = multiprocessing.cpu_count()

    def run_scenario(scenario_params, controller_file):
        controller = load(controller_file)  # Load a fresh copy
        return two_policy_simulation(scenario_params, controller)

    results = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_scenario)(grid_scenarios[i], controller_files[i % len(controller_files)])
        for i in range(len(grid_scenarios))
    )
    print("DONE")

    return np.asarray(results)

def parallel_multi_run(params_dict: list[dict], save_path="calibrated_controllers"):
    """
    Runs calibration for multiple seeds in parallel and saves them.
    """
    num_cores = multiprocessing.cpu_count()

    def run_and_save(param, idx):
        controller = generate_data(param)  # Run calibration
        dump(controller, f"{save_path}/Calibration_runs/controller_seed_{idx}.pkl")  # Save
        return f"{save_path}/Calibration_runs/controller_seed_{idx}.pkl"  # Return filename

    controller_files = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_and_save)(params_dict[i], i) for i in range(len(params_dict))
    )

    print("done controllers!")
    return controller_files  # Return list of file paths

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    policy_pairs=None,
    repetitions=100,
    bounds_LOAD="package/analysis/policy_bounds.json"
):
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(bounds_LOAD) as f:
        policy_info_dict = json.load(f)
        
    bounds = policy_info_dict["bounds_dict"]
    future_time_steps = base_params["duration_future"]
    
    base_params["duration_future"] = 0

    file_name = produce_name_datetime("vary_two_policies_gen")
    print("File Name:", file_name)

    base_params_list = params_list_with_seed(base_params)

    # Generate policy scenarios with different seeds
    grid_scenarios = generate_two_policy_scenarios_with_seeds(base_params, policy_pairs, repetitions, bounds)
    print("Base params list runs:", len(base_params_list))
    print("Grid scenarios runs:", len(grid_scenarios))

    # Ensure directory exists
    createFolder(file_name)
    
    # Run initial seed calibrations and save controllers
    controller_files = parallel_multi_run(base_params_list, save_path=file_name)

    print("Finished Calibration Runs")

    # Save base params
    save_object(base_params, file_name + "/Data", "base_params")

    # Restore duration
    for params in grid_scenarios:
        params["duration_future"] = future_time_steps

    # Run policy scenarios starting from saved calibration controllers
    results = grid_search_two_policies_with_seeds(grid_scenarios, controller_files)

    print("DONE ALL POLICY RUNS")
    save_object(results, file_name + "/Data", "results")
    save_object(policy_pairs, file_name + "/Data", "policy_pairs")
    save_object(policy_info_dict, file_name + "/Data", "policy_info_dict")

    # Reshape results to have dimensions: [num_policy_pairs, repetitions, repetitions, seed_repetitions, num_outputs]
    data_array = results.reshape(
        len(policy_pairs), 
        repetitions, 
        repetitions, 
        base_params["seed_repetitions"], 
        len(results[0])
    )
    
    save_object(data_array, file_name + "/Data", "data_array")

    # Cleanup: Delete the calibration data folder and all its contents
    calibration_folder = Path(file_name) / "Calibration_runs"
    if calibration_folder.exists():
        print(f"Deleting calibration data folder: {calibration_folder}")
        shutil.rmtree(calibration_folder)
    else:
        print(f"Calibration data folder not found: {calibration_folder}")

if __name__ == "__main__":
    # Define all possible policy pairs you want to test
    all_policies = [
        "Carbon_price",
        "Electricity_subsidy",
        "Adoption_subsidy",
        "Adoption_subsidy_used",
        "Production_subsidy"
    ]
    
    # Generate all possible unique pairs of policies
    policy_pairs = [(p1, p2) for i, p1 in enumerate(all_policies) 
                   for p2 in all_policies[i+1:]]
    
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_policy_2D_gen.json",
        repetitions=100, 
        policy_pairs=policy_pairs,
        bounds_LOAD="package/analysis/policy_bounds_vary_policy_2D_gen.json",
    )