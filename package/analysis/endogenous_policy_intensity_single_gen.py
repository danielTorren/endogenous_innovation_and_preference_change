import json
import numpy as np
from joblib import Parallel, delayed, dump, load
import multiprocessing
from package.resources.run import load_in_controller, generate_data
from package.resources.utility import (
    createFolder, save_object, produce_name_datetime, params_list_with_seed
)
import shutil  # Cleanup
from pathlib import Path  # Path handling
from scipy.stats import norm


def update_policy_intensity(params, policy_name, intensity_level):
    """
    Update the policy intensity in the parameter dictionary.
    """
    params["parameters_policies"]["States"][policy_name] = 1

    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity_level
    return params


def single_policy_simulation(params, controller_file):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    controller = load(controller_file)  # Load fresh controller
    data = load_in_controller(controller, params)
    return data.calc_EV_prop(), data.calc_total_policy_distortion()


def single_policy_with_seeds(params, controller_files):
    """
    Run policy scenarios using pre-saved controllers for consistency.
    """
    num_cores = multiprocessing.cpu_count()

    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(single_policy_simulation)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )

    EV_uptake_arr, total_cost_arr = zip(*res)
    
    return np.asarray(EV_uptake_arr), np.asarray(total_cost_arr)


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval for a dataset.
    """
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    margin = sem * norm.ppf((1 + confidence) / 2)
    return mean - margin, mean + margin


def objective_function(intensity_level, params, controller_files, policy_name, target_ev_uptake):
    """
    Objective function for optimization.
    """
    params = update_policy_intensity(params, policy_name, intensity_level)

    EV_uptake_arr, total_cost_arr = single_policy_with_seeds(params, controller_files)

    mean_error = np.mean(target_ev_uptake - EV_uptake_arr)
    mean_EV_uptake = np.mean(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)
    conf_int = compute_confidence_interval(EV_uptake_arr)

    print(f"Intensity: {intensity_level}, Mean EV uptake: {mean_EV_uptake}, Error: {mean_error}, CI: {conf_int}")

    return mean_error, mean_EV_uptake, mean_total_cost, conf_int


def manual_optimization(bounds, params, controller_files, policy_name, intensity_init, target_ev_uptake,
                        step_size=0.01, max_iter=100, adaptive_factor=0.5, min_step_size=1e-4, max_step_size=1.0):
    """
    Adaptive step-size optimization to adjust policy intensity.
    """
    intensity = intensity_init
    prev_error = None
    data = []

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}, Intensity: {intensity}, Step Size: {step_size}")

        error, ev_uptake, total_cost, conf_int = objective_function(
            intensity, params, controller_files, policy_name, target_ev_uptake
        )

        data.append([error, ev_uptake, total_cost, conf_int, intensity, step_size])

        if conf_int[0] <= target_ev_uptake <= conf_int[1]:
            print("Converged successfully.")
            break

        if prev_error is not None:
            if abs(error) < abs(prev_error):
                step_size = min(step_size * (1 + adaptive_factor), max_step_size)
            else:
                step_size = max(step_size * adaptive_factor, min_step_size)

        next_intensity = intensity + step_size * np.sign(error)

        if next_intensity > bounds[1]:
            intensity = bounds[1]
            break
        elif next_intensity < bounds[0]:
            intensity = bounds[0]
            break
        else:
            intensity = next_intensity

        prev_error = error

    return intensity, error, ev_uptake, total_cost, data


def optimize_policy_intensity(params, controller_files, policy_name, intensity_init, target_ev_uptake,
                              bounds, step_size=0.1, adaptive_factor=0.5, step_size_bounds=(0.1, 100), max_iterations=30):
    """
    Optimizes the intensity of a policy to reach the target EV uptake.
    """
    min_step_size, max_step_size = step_size_bounds

    optimized_intensity, error, mean_ev_uptake, mean_total_cost, data = manual_optimization(
        bounds, params, controller_files, policy_name, intensity_init, target_ev_uptake,
        step_size=step_size, max_iter=max_iterations, adaptive_factor=adaptive_factor,
        min_step_size=min_step_size, max_step_size=max_step_size
    )

    print(f"Optimized {policy_name}: Intensity = {optimized_intensity}, EV uptake = {mean_ev_uptake}, Cost = {mean_total_cost}")
    return optimized_intensity, mean_ev_uptake, mean_total_cost, data

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

    print("done controlere!")
    return controller_files  # Return list of file paths

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
         BOUNDS_LOAD="package/analysis/policy_bounds.json",
         policy_list=None, 
         target_ev_uptake=0.5
         ):
    """
    Main function for optimizing policy intensities.
    """
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        policy_params_dict = json.load(f)

    bounds_dict = policy_params_dict["bounds_dict"]
    step_size_dict = policy_params_dict["step_size_dict"]
    init_values = policy_params_dict["init_val_dict"]

    base_params_list = params_list_with_seed(base_params)
    file_name = produce_name_datetime("endogenous_policy_intensity")
    
    createFolder(file_name)
    controller_files = parallel_multi_run(base_params_list, file_name)

    save_object(base_params, file_name + "/Data", "base_params")

    policy_outcomes, runs_data = {}, {}

    for policy_name in policy_list:
        initial_step_size = init_values[policy_name] * 0.1

        intensity_level, mean_ev_uptake, mean_total_cost, policy_data = optimize_policy_intensity(
            base_params, controller_files, policy_name,
            intensity_init=init_values[policy_name], target_ev_uptake=target_ev_uptake,
            bounds=bounds_dict[policy_name], step_size=initial_step_size,
            adaptive_factor=0.5, step_size_bounds=step_size_dict[policy_name]
        )

        runs_data[policy_name] = policy_data
        policy_outcomes[policy_name] = [mean_ev_uptake, mean_total_cost, intensity_level]

    save_object(policy_outcomes, file_name + "/Data", "policy_outcomes")
    save_object(runs_data, file_name + "/Data", "runs_data")

    shutil.rmtree(Path(file_name) / "Calibration_runs", ignore_errors=True)


if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD = "package/constants/base_params_endogenous_policy_single_gen.json",
        BOUNDS_LOAD = "package/analysis/policy_bounds_vary_single_policy_gen.json", 
        policy_list=["Carbon_price"], 
        target_ev_uptake=0.6
        )
