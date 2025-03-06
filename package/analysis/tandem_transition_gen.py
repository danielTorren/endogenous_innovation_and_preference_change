import json
import numpy as np
from joblib import Parallel, delayed, dump, load
import multiprocessing
from package.resources.run import load_in_controller, generate_data
from package.resources.utility import createFolder, save_object, produce_name_datetime, params_list_with_seed
import shutil  # Add this import at the top of your script
from pathlib import Path  # For easier path handling

def produce_param_list_elect_intensity(params: dict, property_list: list) -> list[dict]:
    params_list = []

    for i in property_list:
        params["parameters_scenarios"]["Grid_emissions_intensity"] = i
        seeds_base_params_list = params_list_with_seed(params)
        params_list.extend(seeds_base_params_list)
    return params_list

####################################################
#CALIBRATION
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

    print("done controlere!")
    return controller_files  # Return list of file paths

###########################################################################################
#POLICIES

def single_policy_simulation(params, controller_load):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    #print("value policy", params["parameters_policies"]["Values"]["Carbon_price"])
    #quit()
    data = load_in_controller(controller_load, params)
    EV_uptake = data.calc_EV_prop()
    policy_distortion = data.calc_total_policy_distortion()
    cum_em = data.social_network.emissions_cumulative
    net_cost = data.calc_net_policy_distortion()
    return EV_uptake, policy_distortion, cum_em, net_cost


def grid_search_policy_with_seeds(param_list, controller_files):
    """
    Perform parallel execution of all policy scenarios and seeds,
    ensuring each run starts from a fresh copy of the calibrated controller.
    """
    num_cores = multiprocessing.cpu_count()

    def run_scenario(scenario_params, controller_file):
        #print("controller_file", controller_file)
        controller = load(controller_file)  # Load a fresh copy
        return single_policy_simulation(scenario_params, controller)

    #results = [run_scenario(grid_scenarios[i], controller_files[i % len(controller_files)])   for i in range(len(grid_scenarios))]
    results = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_scenario)(param_list[i], controller_files[i % len(controller_files)])
        for i in range(len(param_list))
    )

    return np.asarray(results)


def set_up_calibration_runs(base_params):

    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    base_params_list = params_list_with_seed(base_params)
    print("NUM calibraion",  len(base_params_list))
    file_name = produce_name_datetime("vary_decarb_elec")

    createFolder(file_name)

    ########################################################################################################################
    #RUN CALIBRATION RUNS
    controller_files = parallel_multi_run(base_params_list, file_name)

    #UPDATE BASE PARAMS CORRECT NOW
    base_params["duration_future"] = future_time_steps

    save_object(base_params, file_name + "/Data", "base_params")

    print("Finished Calibration Runs")

    return controller_files, base_params, file_name

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD = "package/constants/vary_single.json",
        carbon_price = 0.1,
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(VARY_LOAD) as f:
        vary_single = json.load(f)

    
    # Run initial seed calibrations and save controllers
    controller_files, base_params, fileName = set_up_calibration_runs(base_params)
    print("Finished Calibration Runs")
    # Save base params
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_single, fileName + "/Data", "vary_single")

    ######################################################################################################
    #Policy

    electricity_decarb_vals = np.linspace(vary_single["min"],vary_single["max"], vary_single["reps"])
    
    params_list = produce_param_list_elect_intensity(base_params, electricity_decarb_vals) 
    print("TOTAL RUNS", 2*len(params_list))
    # Run policy scenarios starting from saved calibration controllers
    results = grid_search_policy_with_seeds(params_list, controller_files)

    results = results.reshape(vary_single["reps"], base_params["seed_repetitions"], len(results[0]))
    save_object(results , fileName + "/Data", "results")
    

    ##########################################################################################################
    #add carbon price
    base_params["parameters_policies"]["States"]["Carbon_price"] = 1
    base_params["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price"] = carbon_price

    params_list = produce_param_list_elect_intensity(base_params, electricity_decarb_vals) 
    
    results_with_price = grid_search_policy_with_seeds(params_list, controller_files)

    results_with_price = results_with_price.reshape(vary_single["reps"], base_params["seed_repetitions"], len(results_with_price[0]))


    save_object(results_with_price , fileName + "/Data", "results_with_price")

    ##########################################################################################################

    print(fileName)

    # Cleanup: Delete the calibration data folder and all its contents
    calibration_folder = Path(fileName) / "Calibration_runs"
    if calibration_folder.exists():
        print(f"Deleting calibration data folder: {calibration_folder}")
        shutil.rmtree(calibration_folder)
    else:
        print(f"Calibration data folder not found: {calibration_folder}")


if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_tandem_transition_gen.json",
        VARY_LOAD ="package/analysis/vary_single_decarbonize.json",
        carbon_price = 0.1
        )