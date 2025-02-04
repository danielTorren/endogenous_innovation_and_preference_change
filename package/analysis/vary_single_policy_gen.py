from copy import deepcopy
import json
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from package.resources.run import load_in_controller, generate_data
from package.resources.utility import createFolder, save_object, produce_name_datetime


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


def grid_search_policy_with_seeds(grid_scenarios, controller_list):
    """
    Perform parallel execution of all policy scenarios and seeds.
    """
    num_cores = multiprocessing.cpu_count()

    def run_scenario(scenario_params, controller):
        EV_uptake, total_cost, cum_em = single_policy_simulation(scenario_params, deepcopy(controller))
        return EV_uptake, total_cost, cum_em

    results = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(run_scenario)(grid_scenarios[i], deepcopy(controller_list[i % len(controller_list)]))
        for i in range(len(grid_scenarios))
    )

    return np.asarray(results)

####################################################
def parallel_multi_run(
        params_dict: list[dict]
):
    num_cores = multiprocessing.cpu_count()
    #res = [generate_data(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_data)(i) for i in params_dict)

    return res

####################################################

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    policy_list=None,
    repetitions=100,
    bounds_LOAD="package/analysis/policy_bounds.json"
):
    if policy_list is None:
        policy_list = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Production_subsidy",
            "Research_subsidy",
        ]
    else:
        raise ValueError("Policy list not specified = []")

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(bounds_LOAD) as f:
        bounds = json.load(f)

    future_time_steps = base_params["duration_future"]
    base_params["duration_future"] = 0

    file_name = produce_name_datetime("vary_single_policy_gen")
    print("File Name:", file_name)

    base_params_list = params_list_with_seed(base_params)

    grid_scenarios = generate_single_policy_scenarios_with_seeds(base_params, policy_list, repetitions, bounds)
    
    print("base_params_list runs", len(base_params_list))
    print("grid_scenarios runs ", len(grid_scenarios))

    controller_list = parallel_multi_run(base_params_list)
    
    createFolder(file_name)

    print("FINISHED RUNS")

    save_object(controller_list, file_name + "/Data", "controller_list")
    save_object(base_params, file_name + "/Data", "base_params")

    base_params["duration_future"] = future_time_steps

    results = grid_search_policy_with_seeds(grid_scenarios, controller_list)

    print("DONE ALL POLICY RUNS")
    save_object(results, file_name + "/Data", "results")
    save_object(policy_list, file_name + "/Data", "policy_list")
    save_object(bounds, file_name + "/Data", "bounds")

    data_array = results.reshape( len(policy_list), repetitions, base_params["seed_repetitions"], 3)

    save_object(data_array, file_name + "/Data", "data_array")


if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_vary_single_policy_gen.json",
        repetitions=20,
        policy_list = [
            "Carbon_price",
            "Discriminatory_corporate_tax",
            "Electricity_subsidy",
            "Adoption_subsidy",
            "Adoption_subsidy_used",
            "Production_subsidy",
            "Research_subsidy",
        ],
        bounds_LOAD="package/analysis/policy_bounds_vary_single_policy_gen.json",
    )
