from copy import deepcopy
import json
import numpy as np
from joblib import Parallel, delayed, dump, load
import multiprocessing
from package.resources.run import load_in_controller, generate_data
from package.resources.utility import createFolder, save_object, produce_name_datetime
import shutil  # Add this import at the top of your script
from pathlib import Path  # For easier path handling

def params_list_with_seed(base_params):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    base_params_list = []
    seed_repetitions = base_params["seed_repetitions"]

    for seed in range(1, seed_repetitions + 1):
        base_params_copy = deepcopy(base_params)
        # Update all the seeds
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
        base_params_list.append(base_params_copy)
    
    return base_params_list


def generate_single_policy_scenarios_with_seeds(base_params, policy_list, repetitions, bounds):
    """
    Generate grid search scenarios for single policies with given repetitions and seed variations.
    """
    scenarios = []
    for policy in policy_list:
        min_val, max_val = bounds[policy]
        intensities = np.linspace(min_val, max_val, repetitions)
        for intensity in intensities:
            base_params_copy = deepcopy(base_params)

            base_params_copy["parameters_policies"]["States"][policy] = "High"#TURN ON THE POLICY
            #SET THE POLICY INTENSITY
            if policy == "Carbon_price":
                base_params_copy["parameters_policies"]["Values"][policy]["High"]["Carbon_price"] = intensity
            else:
                base_params_copy["parameters_policies"]["Values"][policy]["High"] = intensity
            
            seed_variations = params_list_with_seed(base_params_copy)
            for seed_params in seed_variations:
                scenarios.append(seed_params)
    
    return scenarios

def single_policy_simulation(params, controller_load):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    data = load_in_controller(controller_load, params)
    EV_uptake = data.calc_EV_prop()
    policy_distortion = data.calc_total_policy_distortion()
    cum_em = data.social_network.emissions_cumulative
    return EV_uptake, policy_distortion, cum_em


def grid_search_policy_with_seeds(grid_scenarios, controller_files):
    """
    Perform parallel execution of all policy scenarios and seeds,
    ensuring each run starts from a fresh copy of the calibrated controller.
    """
    num_cores = multiprocessing.cpu_count()

    def run_scenario(scenario_params, controller_file):
        controller = load(controller_file)  # Load a fresh copy
        return single_policy_simulation(scenario_params, controller)

    results = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_scenario)(grid_scenarios[i], controller_files[i % len(controller_files)])
        for i in range(len(grid_scenarios))
    )

    return np.asarray(results)


####################################################

def parallel_multi_run(params_dict: list[dict], save_path="calibrated_controllers"):
    """
    Runs calibration for multiple seeds in parallel and saves them.
    """
    num_cores = multiprocessing.cpu_count()
    #createFolder(save_path)  # Ensure directory exists

    def run_and_save(param, idx):
        controller = generate_data(param)  # Run calibration
        dump(controller, f"{save_path}/Calibration_runs/controller_seed_{idx}.pkl")  # Save
        return f"{save_path}/Calibration_runs/controller_seed_{idx}.pkl"  # Return filename

    controller_files = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_and_save)(params_dict[i], i) for i in range(len(params_dict))
    )

    return controller_files  # Return list of file paths


####################################################

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    policy_list=None,
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

    file_name = produce_name_datetime("vary_single_policy_gen")
    print("File Name:", file_name)

    base_params_list = params_list_with_seed(base_params)

    # Generate policy scenarios with different seeds
    grid_scenarios = generate_single_policy_scenarios_with_seeds(base_params, policy_list, repetitions, bounds)
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
    results = grid_search_policy_with_seeds(grid_scenarios, controller_files)

    print("DONE ALL POLICY RUNS")
    save_object(results, file_name + "/Data", "results")
    save_object(policy_list, file_name + "/Data", "policy_list")
    save_object(policy_info_dict, file_name + "/Data", "policy_info_dict")

    data_array = results.reshape( len(policy_list), repetitions, base_params["seed_repetitions"], 3)

    save_object(data_array, file_name + "/Data", "data_array")

    # Cleanup: Delete the calibration data folder and all its contents
    calibration_folder = Path(file_name) / "Calibration_runs"
    if calibration_folder.exists():
        print(f"Deleting calibration data folder: {calibration_folder}")
        shutil.rmtree(calibration_folder)
    else:
        print(f"Calibration data folder not found: {calibration_folder}")

if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single_policy_gen.json",
        repetitions=20,
        policy_list = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            #"Electricity_subsidy",
            "Adoption_subsidy",
            "Adoption_subsidy_used",
            "Production_subsidy",
            "Research_subsidy",
        ],
        bounds_LOAD="package/analysis/policy_bounds_vary_single_policy_gen.json",
    )
