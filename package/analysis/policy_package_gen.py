import json
import numpy as np
from copy import deepcopy
from itertools import product
from skopt import gp_minimize
from skopt.space import Real
from package.resources.utility import save_object
from package.analysis.endogenous_policy_intensity_single_gen import (
    set_up_calibration_runs,
    single_policy_with_seeds
)

def update_policy_intensity(params, policy_name, intensity_level):
    params["parameters_policies"]["States"][policy_name] = 1
    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity_level
    return params

def simulate_policy_scenario(sim_params, controller_files):
    EV_uptake_arr, *_ = single_policy_with_seeds(sim_params, controller_files)
    return np.mean(EV_uptake_arr)


def optimize_endogenous_carbon_price(base_params, controller_files, elec_val, prod_val, bounds_dict, n_calls=15):
    search_space = [Real(*bounds_dict["Carbon_price"], name="Carbon_price")]

    def objective(carbon_price):
        params = deepcopy(base_params)
        params = update_policy_intensity(params, "Electricity_subsidy", elec_val)
        params = update_policy_intensity(params, "Production_subsidy", prod_val)
        params = update_policy_intensity(params, "Carbon_price", carbon_price[0])

        ev_uptake = simulate_policy_scenario(params, controller_files)

        return abs(ev_uptake - 0.95)

    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=42, acq_func="EI")
    best_carbon = result.x[0]

    # Confirm final output
    final_params = deepcopy(base_params)
    final_params = update_policy_intensity(final_params, "Electricity_subsidy", elec_val)
    final_params = update_policy_intensity(final_params, "Production_subsidy", prod_val)
    final_params = update_policy_intensity(final_params, "Carbon_price", best_carbon)

    EV_uptake_arr, total_cost_arr, net_cost_arr, emissions_cumulative_arr, emissions_cumulative_driving_arr, \
    emissions_cumulative_production_arr, utility_cumulative_arr,utility_cumulative_30_arr, profit_cumulative_arr = single_policy_with_seeds(final_params, controller_files)

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

    print(f"Intensities= {best_carbon , elec_val, prod_val}, EV uptake = {mean_ev_uptake}, STD EV uptake = {sd_ev_uptake}, "
          f"Cost = {mean_total_cost}, Net cost = {mean_net_cost}, E = {mean_emissions_cumulative}, "
          f"E_D = {mean_emissions_cumulative_driving}, E_P = {mean_emissions_cumulative_production}, "
          f"U = {mean_utility_cumulative}, Profit = {mean_profit_cumulative}")

    return (
            [best_carbon , elec_val, prod_val],
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

def grid_search_with_endogenous_carbon(base_params, controller_files, bounds_dict, steps=6, n_calls=10):
    elec_range = np.linspace(*bounds_dict["Electricity_subsidy"], steps)
    prod_range = np.linspace(*bounds_dict["Production_subsidy"], steps)

    policy_outcomes = {}
    for elec_val, prod_val in product(elec_range, prod_range):

        (
            best_intensities,
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
        ) = optimize_endogenous_carbon_price(
            base_params, controller_files, elec_val, prod_val, bounds_dict, n_calls=n_calls
        )

        policy_outcomes[(elec_val, prod_val)] = {
            "optimized_intensity": best_intensities,
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
            "confidence_interval": 1.96 * sd_ev_uptake / np.sqrt(base_params["seed_repetitions"])
        }

    return policy_outcomes

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_policy_package_gen.json",
    BOUNDS_LOAD="package/analysis/policy_bounds_policy_package_gen.json",
    n_calls=10,
    steps=6
):
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        bounds_dict = json.load(f)
    print("Total runs:", base_params["seed_repetitions"]*n_calls*steps*steps)
    controller_files, base_params, file_name = set_up_calibration_runs(base_params, "2d_grid_endogenous_third")

    policy_outcomes = grid_search_with_endogenous_carbon(base_params, controller_files, bounds_dict, steps,n_calls)

    save_object(policy_outcomes, file_name + "/Data", "policy_outcomes")

if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_policy_package_gen.json",
        BOUNDS_LOAD="package/analysis/policy_bounds_policy_package_gen.json",
        policy_names=["Carbon_price", "Electricity_subsidy", "Production_subsidy"],
        n_calls=10,
        steps = 4
    )
