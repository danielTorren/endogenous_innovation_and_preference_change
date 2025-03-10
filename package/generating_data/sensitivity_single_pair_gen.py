from copy import deepcopy
import json
from package.resources.run import ev_prop_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime,
    params_list_with_seed
)
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from package.resources.run import generate_data


def produce_param_list(params: dict, property_dict_1, property_dict_2) -> list[dict]:
    params_list = []


    num_param_combos = int(len(property_dict_1["property_list"])*len(property_dict_2["property_list"]))
    num_reps = params["seed_repetitions"]
    print("num_param_combos", num_param_combos)
    print(" num_reps ",  num_reps )
    for i in property_dict_1["property_list"]:
        for j in  property_dict_2["property_list"]:
            params_updated = deepcopy(params)
            params_updated[property_dict_1["subdict"]][property_dict_1["property_varied"]] = i
            params_updated[property_dict_2["subdict"]][property_dict_2["property_varied"]] = j
            params_list_pairs = params_list_with_seed(params_updated)
            params_list.extend(params_list_pairs)

    return params_list, num_param_combos, num_reps

def single_simulation(params):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    data = generate_data(params)  # Run calibration
    mean_price, min_price, max_price = data.social_network.calc_price_mean_max_min()
    return data.calc_EV_prop(), data.calc_total_policy_distortion(), data.calc_net_policy_distortion(), data.social_network.emissions_cumulative, data.social_network.emissions_cumulative_driving, data.social_network.emissions_cumulative_production, data.social_network.utility_cumulative, data.firm_manager.profit_cumulative, mean_price, max_price, min_price, data.firm_manager.last_step_calc_profit_margin(), data.social_network.calc_mean_car_age()


def runs_with_seeds(params_list, num_param_combos, num_reps):
    """
    Run policy scenarios using pre-saved controllers for consistency.
    """
    num_cores = multiprocessing.cpu_count()

    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(single_simulation)(params_loop) for params_loop in params_list
        )

    EV_uptake_arr, total_cost_arr, net_cost_arr,emissions_cumulative_arr, emissions_cumulative_driving_arr, emissions_cumulative_production_arr, utility_cumulative_arr, profit_cumulative_arr, price_mean, price_max, price_min, mean_mark_up, mean_car_age = zip(*res)
    
    (params_list)
    results = {
        "ev_uptake": np.asarray(EV_uptake_arr).reshape(num_param_combos, num_reps),
        "total_cost": np.asarray(total_cost_arr).reshape(num_param_combos, num_reps),
        "net_cost": np.asarray(net_cost_arr).reshape(num_param_combos, num_reps),
        "emissions_cumulative": np.asarray(emissions_cumulative_arr).reshape(num_param_combos, num_reps),
        "emissions_cumulative_driving": np.asarray(emissions_cumulative_driving_arr).reshape(num_param_combos, num_reps),
        "emissions_cumulative_production": np.asarray(emissions_cumulative_production_arr).reshape(num_param_combos, num_reps),
        "utility_cumulative": np.asarray(utility_cumulative_arr).reshape(num_param_combos, num_reps),
        "profit_cumulative": np.asarray(profit_cumulative_arr).reshape(num_param_combos, num_reps),
        "price_mean": np.asarray(price_mean).reshape(num_param_combos, num_reps), 
        "price_max": np.asarray(price_max).reshape(num_param_combos, num_reps), 
        "price_min": np.asarray(price_min).reshape(num_param_combos, num_reps), 
        "mean_mark_up": np.asarray(mean_mark_up).reshape(num_param_combos, num_reps), 
        "mean_car_age": np.asarray(mean_car_age).reshape(num_param_combos, num_reps)
    }
    return results


def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD_1 = "package/constants/vary_single.json",
        VARY_LOAD_2 = "package/constants/vary_single.json"
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    with open(VARY_LOAD_1) as f:
        vary_1 = json.load(f)
    with open(VARY_LOAD_2) as f:
        vary_2 = json.load(f)

    # Check if 'property_list' exists in vary_1, if not, create it
    if "property_list" not in vary_1:
        vary_1["property_list"] = np.linspace(vary_1["min"], vary_1["max"], vary_1["reps"])

    # Check if 'property_list' exists in vary_2, if not, create it
    if "property_list" not in vary_2:
        vary_2["property_list"] = np.linspace(vary_2["min"], vary_2["max"], vary_2["reps"])

    root = "sensitivity_2D"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list, num_param_combos, num_reps = produce_param_list(base_params, vary_1, vary_2)
    
    print("TOTAL SUB RUNS: ", len(params_list))
    
    results = runs_with_seeds(params_list, num_param_combos, num_reps) 
    createFolder(fileName)
    
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(vary_1 , fileName + "/Data", "vary_1")
    save_object(vary_2 , fileName + "/Data", "vary_2")
    save_object(results , fileName + "/Data", "results")

    return fileName

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/base_params_sensitivity_2D.json",
        VARY_LOAD_1 ="package/constants/vary_single_beta_segments_sensitivity.json", 
        VARY_LOAD_2 ="package/constants/vary_single_gamma_segments_sensitivity.json",
        )