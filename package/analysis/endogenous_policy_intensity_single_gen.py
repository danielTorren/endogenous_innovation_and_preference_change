from copy import deepcopy
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
from skopt import gp_minimize
from skopt.space import Real


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
    return data.calc_EV_prop(), data.calc_total_policy_distortion(), data.calc_net_policy_distortion(), data.social_network.emissions_cumulative, data.social_network.emissions_cumulative_driving, data.social_network.emissions_cumulative_production, data.social_network.utility_cumulative, data.utility_cum_2030, data.firm_manager.profit_cumulative


def single_policy_with_seeds(params, controller_files):
    """
    Run policy scenarios using pre-saved controllers for consistency.
    """
    num_cores = multiprocessing.cpu_count()

    res = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(single_policy_simulation)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )

    EV_uptake_arr, total_cost_arr, net_cost_arr,emissions_cumulative_arr, emissions_cumulative_driving_arr, emissions_cumulative_production_arr, utility_cumulative_arr, utility_cumulative_30_arr, profit_cumulative_arr = zip(*res)
    
    return np.asarray(EV_uptake_arr), np.asarray(total_cost_arr), np.asarray(net_cost_arr), np.asarray(emissions_cumulative_arr), np.asarray(emissions_cumulative_driving_arr), np.asarray(emissions_cumulative_production_arr), np.asarray(utility_cumulative_arr), np.asarray(utility_cumulative_30_arr), np.asarray(profit_cumulative_arr)


########################################################

def objective_function(intensity_level, base_params, controller_files, policy_name, target_ev_uptake):
    """
    Objective function for optimization. Returns the absolute error between
    the mean EV uptake and the target.
    """
    sim_params = deepcopy(base_params)  # Deep copy to reset every time
    sim_params = update_policy_intensity(sim_params, policy_name, intensity_level)

    EV_uptake_arr, _, _, _, _, _, _, _ = single_policy_with_seeds(sim_params, controller_files)
    mean_ev_uptake = np.mean(EV_uptake_arr)
    error = abs(target_ev_uptake - mean_ev_uptake)

    return error



def optimize_policy_intensity_BO(base_params, controller_files, policy_name, target_ev_uptake, bounds, n_calls, noise):
    """
    Optimizes the intensity of a policy to reach the target EV uptake using Bayesian Optimization.
    """

    search_space = [Real(bounds[0], bounds[1])]
    print("search_space", search_space)

    result = gp_minimize(
        lambda x: objective_function(x[0], base_params, controller_files, policy_name, target_ev_uptake),
        search_space,
        n_calls=n_calls,
        noise=noise,
        acq_func="EI"
    )

    print("DONE BO")

    # Get best intensity level
    best_intensity = result.x[0]
    print(f"Policy {policy_name}, best intensity: {best_intensity}")

    # Run simulation with optimized intensity to get final results
    params_final = deepcopy(base_params)
    params_final = update_policy_intensity(params_final, policy_name, best_intensity)

    EV_uptake_arr, total_cost_arr, net_cost_arr, emissions_cumulative_arr, emissions_cumulative_driving_arr, \
    emissions_cumulative_production_arr, utility_cumulative_arr,utility_cumulative_30_arr, profit_cumulative_arr = single_policy_with_seeds(params_final, controller_files)

    mean_ev_uptake = np.mean(EV_uptake_arr)
    sd_ev_uptake = np.std(EV_uptake_arr)
    mean_total_cost = np.mean(total_cost_arr)
    mean_net_cost = np.mean(net_cost_arr)
    mean_emissions_cumulative = np.mean(emissions_cumulative_arr)
    mean_emissions_cumulative_driving = np.mean(emissions_cumulative_driving_arr)
    mean_emissions_cumulative_production = np.mean(emissions_cumulative_production_arr)
    mean_utility_cumulative = np.mean(utility_cumulative_arr)
    mean_utility_cumulative_30 = np.mean(utility_cumulative_30_arr)
    mean_profit_cumulative = np.mean(profit_cumulative_arr)

    print(f"Optimized {policy_name}: Intensity = {best_intensity}, EV uptake = {mean_ev_uptake}, STD EV uptake = {sd_ev_uptake}, "
          f"Cost = {mean_total_cost}, Net cost = {mean_net_cost}, E = {mean_emissions_cumulative}, "
          f"E_D = {mean_emissions_cumulative_driving}, E_P = {mean_emissions_cumulative_production}, "
          f"U = {mean_utility_cumulative}, Profit = {mean_profit_cumulative}")

    return (
            best_intensity,
            mean_ev_uptake, 
            sd_ev_uptake, 
            mean_total_cost, 
            mean_net_cost, 
            mean_emissions_cumulative, 
            mean_emissions_cumulative_driving, 
            mean_emissions_cumulative_production, 
            mean_utility_cumulative, 
            mean_utility_cumulative_30, 
            mean_profit_cumulative,
            EV_uptake_arr,
            net_cost_arr,
            emissions_cumulative_driving_arr,
            emissions_cumulative_production_arr,
            utility_cumulative_arr,
            profit_cumulative_arr
    )

####################################################

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

    print("Calibration complete!")
    return controller_files  # Return list of file paths


def set_up_calibration_runs(base_params, root):

    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    base_params_list = params_list_with_seed(base_params)
    file_name = produce_name_datetime(root)

    createFolder(file_name)

    ########################################################################################################################
    #RUN CALIBRATION RUNS
    controller_files = parallel_multi_run(base_params_list, file_name)

    #UPDATE BASE PARAMS CORRECT NOW
    base_params["duration_future"] = future_time_steps

    save_object(base_params, file_name + "/Data", "base_params")

    print("Finished Calibration Runs")

    return controller_files, base_params, file_name

def simulate_future_policies(file_name, controller_files, policy_list, policy_params_dict, n_calls, base_params, target_ev_uptake, noise):

    ########################################################################################################################
    #RUN POLICY OUTCOMES

    bounds_dict = policy_params_dict["bounds_dict"]

    print("TOTAL RUNS BO: ", len(policy_list)*n_calls*base_params["seed_repetitions"])
    policy_outcomes = {}

    for policy_name in policy_list:
        (
            best_intensity,
            mean_ev_uptake, 
            sd_ev_uptake, 
            mean_total_cost, 
            mean_net_cost, 
            mean_emissions_cumulative, 
            mean_emissions_cumulative_driving, 
            mean_emissions_cumulative_production, 
            mean_utility_cumulative, 
            mean_utility_cumulative_30, 
            mean_profit_cumulative,
            ev_uptake,
            net_cost,
            emissions_cumulative_driving,
            emissions_cumulative_production,
            utility_cumulative,
            profit_cumulative 
        ) = optimize_policy_intensity_BO(
            base_params, controller_files, policy_name, target_ev_uptake=target_ev_uptake,
            bounds=bounds_dict[policy_name], n_calls=n_calls, noise = noise
        )

        policy_outcomes[policy_name] = {
            "optimized_intensity": best_intensity,
            "mean_EV_uptake": mean_ev_uptake,
            "sd_ev_uptake": sd_ev_uptake,
            "mean_total_cost": mean_total_cost,
            "mean_net_cost": mean_net_cost, 
            "mean_emissions_cumulative": mean_emissions_cumulative, 
            "mean_emissions_cumulative_driving": mean_emissions_cumulative_driving, 
            "mean_emissions_cumulative_production": mean_emissions_cumulative_production, 
            "mean_utility_cumulative": mean_utility_cumulative, 
            "mean_utility_cumulative_30": mean_utility_cumulative_30, 
            "mean_profit_cumulative": mean_profit_cumulative,
            "ev_uptake": ev_uptake,
            "net_cost": net_cost,
            "emissions_cumulative_driving": emissions_cumulative_driving,
            "emissions_cumulative_production": emissions_cumulative_production,
            "utility_cumulative": utility_cumulative,
            "profit_cumulative": profit_cumulative,
            "confidence_interval": 1.96 * sd_ev_uptake / np.sqrt(64)
        }

    save_object(policy_outcomes, file_name + "/Data", "policy_outcomes")

    return policy_outcomes 

def main(BASE_PARAMS_LOAD="package/constants/base_params.json",
         BOUNDS_LOAD="package/analysis/policy_bounds.json",
         policy_list=None, 
         target_ev_uptake=0.5,
         n_calls=40, 
         noise = 0.01
         ):
    """
    Main function for optimizing policy intensities.
    """
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        policy_params_dict = json.load(f)

    controller_files, base_params, file_name = set_up_calibration_runs(base_params, "endog_single")

    simulate_future_policies(file_name, controller_files, policy_list, policy_params_dict, n_calls, base_params, target_ev_uptake, noise)

    shutil.rmtree(Path(file_name) / "Calibration_runs", ignore_errors=True)

    conditions = {
        "policy_list": policy_list,
        "target_ev_uptake": target_ev_uptake,
        "n_calls": n_calls,
        "noise": noise
    }

    save_object(conditions, file_name + "/Data", "conditions")

if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_single_gen.json",
        BOUNDS_LOAD="package/analysis/policy_bounds_endog_single_gen.json",
        policy_list=[
            "Carbon_price",
            "Adoption_subsidy",
            "Production_subsidy"
        ],
        target_ev_uptake=0.95,
        n_calls=5,
        noise=0.05
    )
