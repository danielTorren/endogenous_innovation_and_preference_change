import json
import numpy as np
from joblib import Parallel, delayed, dump, load
import multiprocessing
from package.resources.run import load_in_controller, generate_data
from package.resources.utility import createFolder, save_object, produce_name_datetime, params_list_with_seed
import shutil
from pathlib import Path

def produce_param_list_grid_search(params: dict, carbon_prices: list, emissions_intensities: list) -> list[dict]:
    """
    Generate a list of parameter dictionaries for grid search over carbon price and emissions intensity.
    """
    params_list = []

    for cp in carbon_prices:
        for ei in emissions_intensities:
            # Set carbon price and emissions intensity
            params["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price"] = cp
            params["parameters_scenarios"]["Grid_emissions_intensity"] = ei

            # Generate parameter dictionaries for each seed
            seeds_base_params_list = params_list_with_seed(params)
            params_list.extend(seeds_base_params_list)

    return params_list

####################################################
# CALIBRATION
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

    print("Done calibration!")
    return controller_files  # Return list of file paths

###########################################################################################
# POLICIES

def single_policy_simulation(params, controller_load):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
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
        controller = load(controller_file)  # Load a fresh copy
        return single_policy_simulation(scenario_params, controller)

    results = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_scenario)(param_list[i], controller_files[i % len(controller_files)])
        for i in range(len(param_list))
    )

    return np.asarray(results)


def set_up_calibration_runs(base_params):
    """
    Set up and run calibration simulations.
    """
    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    base_params_list = params_list_with_seed(base_params)
    print("NUM calibration runs:", len(base_params_list))
    file_name = produce_name_datetime("grid_search_carbon_price_emissions_intensity")

    createFolder(file_name)

    # Run calibration runs
    controller_files = parallel_multi_run(base_params_list, file_name)

    # Update base_params with correct future time steps
    base_params["duration_future"] = future_time_steps

    save_object(base_params, file_name + "/Data", "base_params")

    print("Finished Calibration Runs")
    return controller_files, base_params, file_name


def main(
    BASE_PARAMS_LOAD="package/constants/base_params.json",
    carbon_price_min=0.0,
    carbon_price_max=1.0,
    emissions_intensity_min=0.0,
    emissions_intensity_max=1.0,
    grid_resolution=10,
) -> str:
    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Run initial seed calibrations and save controllers
    controller_files, base_params, fileName = set_up_calibration_runs(base_params)
    print("Finished Calibration Runs")

    # Save base params
    save_object(base_params, fileName + "/Data", "base_params")

    vary_double = {"decrb": [emissions_intensity_min, emissions_intensity_max], "carbon_price": [carbon_price_min, carbon_price_max], "reps", grid_resolution}
    save_object(vary_double, fileName + "/Data", "vary_double")

    ######################################################################################################
    # Policy Grid Search

    # Define grid values for carbon price and emissions intensity
    carbon_prices = np.linspace(carbon_price_min, carbon_price_max, grid_resolution)
    emissions_intensities = np.linspace(emissions_intensity_min, emissions_intensity_max, grid_resolution)

    # Generate parameter list for grid search
    params_list = produce_param_list_grid_search(base_params, carbon_prices, emissions_intensities)
    print("TOTAL GRID SEARCH RUNS:", len(params_list))

    # Run policy scenarios starting from saved calibration controllers
    results_serial = grid_search_policy_with_seeds(params_list, controller_files)

    # Reshape results for easier analysis
    results = results_serial.reshape(
        grid_resolution, grid_resolution, base_params["seed_repetitions"], len(results_serial[0]))
    save_object(results, fileName + "/Data", "results_grid_search")

    ##########################################################################################################

    print("Grid search results saved in:", fileName)

    # Cleanup: Delete the calibration data folder and all its contents
    calibration_folder = Path(fileName) / "Calibration_runs"
    if calibration_folder.exists():
        print(f"Deleting calibration data folder: {calibration_folder}")
        shutil.rmtree(calibration_folder)
    else:
        print(f"Calibration data folder not found: {calibration_folder}")


if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_grid_tandem_gen.json",
        carbon_price_min=0.0,
        carbon_price_max=1.0,
        emissions_intensity_min=0.0,
        emissions_intensity_max=1.0,
        grid_resolution=3,  # Number of points in the grid for each parameter
    )