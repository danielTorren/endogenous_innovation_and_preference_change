from package.resources.utility import load_object, save_object
import numpy as np
from package.analysis.endogenous_policy_intensity_single_gen import update_policy_intensity, set_up_calibration_runs
from package.resources.utility import (
    save_object, 
    load_object,
)
import shutil  # Cleanup
from pathlib import Path  # Path handling
from copy import deepcopy
from package.resources.run import load_in_controller
from joblib import Parallel, delayed, load
import multiprocessing


def single_policy_simulation(params, controller_file):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    controller = load(controller_file)  # Load fresh controller
    data = load_in_controller(controller, params)
    return (
        data.social_network.history_driving_emissions,#Emmissions flow
        data.social_network.history_production_emissions,#Emmissions flow
        data.social_network.history_total_emissions,#Emmissions flow
        data.social_network.history_prop_EV, 
        data.social_network.history_car_age, 
        data.social_network.history_lower_percentile_price_ICE_EV,
        data.social_network.history_upper_percentile_price_ICE_EV,
        data.social_network.history_mean_price_ICE_EV,
        data.social_network.history_median_price_ICE_EV, 
        data.social_network.history_total_utility,
        data.firm_manager.history_market_concentration,
        data.firm_manager.history_total_profit, 
        data.social_network.history_quality_ICE, 
        data.social_network.history_quality_EV, 
        data.social_network.history_efficiency_ICE, 
        data.social_network.history_efficiency_EV, 
        data.social_network.history_production_cost_ICE, 
        data.social_network.history_production_cost_EV, 
        data.firm_manager.history_mean_profit_margins_ICE,
        data.firm_manager.history_mean_profit_margins_EV,
        data.social_network.history_mean_car_age,
        data.firm_manager.history_past_new_bought_vehicles_prop_ev,
        data.history_policy_net_cost,
        data.social_network.history_total_utility_bottom,
        data.social_network.history_ev_adoption_rate_bottom
    )

def single_policy_with_seeds(params, controller_files):
    """
    Run policy scenarios using pre-saved controllers for consistency.
    """
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(single_policy_simulation)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )
    
    (
        history_driving_emissions_arr,#Emmissions flow
        history_production_emissions_arr,
        history_total_emissions_arr,#Emmissions flow
        history_prop_EV_arr, 
        history_car_age_arr, 
        history_lower_percentile_price_ICE_EV_arr,
        history_upper_percentile_price_ICE_EV_arr,
        history_mean_price_ICE_EV_arr,
        history_median_price_ICE_EV_arr, 
        history_total_utility_arr, 
        history_market_concentration_arr,
        history_total_profit_arr, 
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        history_mean_car_age,
        history_past_new_bought_vehicles_prop_ev,
        history_policy_net_cost,
        history_total_utility_bottom,
        history_ev_adoption_rate_bottom
    ) = zip(*res)

        # Return results as arrays where applicable
    return (
        np.asarray(history_driving_emissions_arr),#Emmissions flow
        np.asarray(history_production_emissions_arr),
        np.asarray(history_total_emissions_arr),#Emmissions flow
        np.asarray(history_prop_EV_arr), 
        np.asarray(history_car_age_arr), 
        np.asarray(history_lower_percentile_price_ICE_EV_arr),
        np.asarray(history_upper_percentile_price_ICE_EV_arr),
        np.asarray(history_mean_price_ICE_EV_arr),
        np.asarray(history_median_price_ICE_EV_arr), 
        np.asarray(history_total_utility_arr), 
        np.asarray(history_market_concentration_arr),
        np.asarray(history_total_profit_arr),
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        np.asarray(history_mean_car_age),
        np.asarray(history_past_new_bought_vehicles_prop_ev),
        np.asarray(history_policy_net_cost),
        np.asarray(history_total_utility_bottom),
        np.asarray(history_ev_adoption_rate_bottom)
    )

def calc_low_intensities(pairwise_outcomes_complied, min_val, max_val):
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    # Init policy ranges
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    # Build a merged dict with symmetric keys
    merged_pairs = {}
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        key = tuple(sorted((policy1, policy2)))
        if key not in merged_pairs:
            merged_pairs[key] = []

        for entry in data:
            entry_copy = entry.copy()
            entry_copy["original_order"] = (policy1, policy2)
            merged_pairs[key].append(entry_copy)

            # Update ranges
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])

    best_entries = {}

    for (policyA, policyB), entries in merged_pairs.items():
        min_max_intensity = float('inf')
        best_entry_raw = None

        for entry in entries:
            if not (min_val <= entry["mean_ev_uptake"] <= max_val):
                continue

            p1, p2 = entry["original_order"]
            val1 = entry["policy1_value"]
            val2 = entry["policy2_value"]

            norm1 = (val1 - policy_ranges[p1]["min"]) / (policy_ranges[p1]["max"] - policy_ranges[p1]["min"]) if policy_ranges[p1]["max"] > policy_ranges[p1]["min"] else 0
            norm2 = (val2 - policy_ranges[p2]["min"]) / (policy_ranges[p2]["max"] - policy_ranges[p2]["min"]) if policy_ranges[p2]["max"] > policy_ranges[p2]["min"] else 0

            max_intensity = max(norm1, norm2)

            if max_intensity < min_max_intensity:
                min_max_intensity = max_intensity
                best_entry_raw = entry

        if best_entry_raw:
            # Reorder policy values to match (policyA, policyB)
            p1, p2 = best_entry_raw["original_order"]
            val1 = best_entry_raw["policy1_value"]
            val2 = best_entry_raw["policy2_value"]

            if (policyA, policyB) == (p1, p2):
                best_entry = {
                    "policy1_value": val1,
                    "policy2_value": val2,
                    "mean_ev_uptake": best_entry_raw["mean_ev_uptake"],
                    "original_order": (policyA, policyB)
                }
            else:
                # Flip values
                best_entry = {
                    "policy1_value": val2,
                    "policy2_value": val1,
                    "mean_ev_uptake": best_entry_raw["mean_ev_uptake"],
                    "original_order": (policyA, policyB)
                }

            best_entries[(policyA, policyB)] = best_entry

            print(f"{policyA}-{policyB} -> ({best_entry['policy1_value']:.3f}, {best_entry['policy2_value']:.3f})  EV uptake: {best_entry['mean_ev_uptake']:.3f}")

    return best_entries


def main(fileName_load,
        min_ev_uptake = 0.945,
        max_ev_uptake = 0.955
        ):

    #pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")

    pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")

    top_policies = calc_low_intensities(pairwise_outcomes_complied,  min_ev_uptake, max_ev_uptake)
    #del top_policies[("Adoption_subsidy","Carbon_price")]
    #top_policies = {("Adoption_subsidy","Carbon_price"): top_policies[("Adoption_subsidy","Carbon_price")]}

    ##########################################################################################

    base_params_calibration = load_object(fileName_load + "/Data", "base_params")

    base_params_calibration["duration_future"] = 312#564#2050#144#SET UP FUTURE

    if "duration_calibration" not in base_params_calibration:
        base_params_calibration["duration_calibration"] = base_params_calibration["duration_no_carbon_price"]

    base_params_calibration["parameters_policies"]["States"] = {
        "Carbon_price": 0,
        "Targeted_research_subsidy": 0,
        "Electricity_subsidy": 0,
        "Adoption_subsidy": 0,
        "Adoption_subsidy_used": 0,
        "Production_subsidy": 0,
        "Research_subsidy": 0
    }

    controller_files, base_params, root_folder  = set_up_calibration_runs(base_params_calibration, "pair_low_intensity_policies")
    print("DONE calibration")
    #
    #NOW SAVE
    save_object(top_policies, root_folder + "/Data", "top_policies")
    save_object(min_ev_uptake, root_folder + "/Data", "min_ev_uptake")
    save_object(max_ev_uptake, root_folder + "/Data", "max_ev_uptake")

    ###########################################################################################

    base_params["parameters_scenarios"]["Grid_emissions_intensity"] = 1
    base_params["save_timeseries_data_state"] = 1

    #RESET TO B SURE
    #RUN BAU
    (
        history_driving_emissions_arr,#Emmissions flow
        history_production_emissions_arr,
        history_total_emissions_arr,#Emmissions flow
        history_prop_EV_arr, 
        history_car_age_arr, 
        history_lower_percentile_price_ICE_EV_arr,
        history_upper_percentile_price_ICE_EV_arr,
        history_mean_price_ICE_EV_arr,
        history_median_price_ICE_EV_arr, 
        history_total_utility_arr, 
        history_market_concentration_arr,
        history_total_profit_arr, 
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        history_mean_car_age,
        history_past_new_bought_vehicles_prop_ev,
        history_policy_net_cost,
        history_total_utility_bottom,
        history_ev_adoption_rate_bottom

    ) = single_policy_with_seeds(base_params, controller_files)

    outputs_BAU = {
            "history_driving_emissions": history_driving_emissions_arr,
            "history_production_emissions": history_production_emissions_arr,
            "history_total_emissions": history_total_emissions_arr,
            "history_prop_EV": history_prop_EV_arr,
            "history_total_utility": history_total_utility_arr,
            "history_market_concentration": history_market_concentration_arr,
            "history_total_profit": history_total_profit_arr,
            "history_mean_profit_margins_ICE": history_mean_profit_margins_ICE,
            "history_mean_profit_margins_EV": history_mean_profit_margins_EV,
            "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
            "history_policy_net_cost": history_policy_net_cost,
            "history_mean_car_age": history_mean_car_age,
            "history_lower_percentile_price_ICE_EV_arr": history_lower_percentile_price_ICE_EV_arr,
            "history_upper_percentile_price_ICE_EV_arr": history_upper_percentile_price_ICE_EV_arr,
            "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
            "history_median_price_ICE_EV_arr": history_median_price_ICE_EV_arr,
            "history_past_new_bought_vehicles_prop_ev": history_past_new_bought_vehicles_prop_ev,
            "history_total_utility_bottom": history_total_utility_bottom,
            "history_ev_adoption_rate_bottom": history_ev_adoption_rate_bottom
    }

    save_object(outputs_BAU, root_folder + "/Data", "outputs_BAU")
    print("DONE BAU")

    ##############################################################################################################################

    print("TOTAL RUNS", len(top_policies)*base_params["seed_repetitions"])
    outputs = {}

    for (policy1, policy2), welfare_data in top_policies.items():

        policy1_value = welfare_data["policy1_value"]
        policy2_value = welfare_data["policy2_value"]

        print(f"Running time series for {policy1},{policy1_value} & {policy2},{policy2_value}")

        params_policy = deepcopy(base_params)
        params_policy = update_policy_intensity(params_policy, policy1, policy1_value)
        params_policy = update_policy_intensity(params_policy, policy2, policy2_value)



        (
            history_driving_emissions_arr,#Emmissions flow
            history_production_emissions_arr,
            history_total_emissions_arr,#Emmissions flow
            history_prop_EV_arr, 
            history_car_age_arr, 
            history_lower_percentile_price_ICE_EV_arr,
            history_upper_percentile_price_ICE_EV_arr,
            history_mean_price_ICE_EV_arr,
            history_median_price_ICE_EV_arr, 
            history_total_utility_arr, 
            history_market_concentration_arr,
            history_total_profit_arr, 
            history_quality_ICE, 
            history_quality_EV, 
            history_efficiency_ICE, 
            history_efficiency_EV, 
            history_production_cost_ICE, 
            history_production_cost_EV, 
            history_mean_profit_margins_ICE,
            history_mean_profit_margins_EV,
            history_mean_car_age,
            history_past_new_bought_vehicles_prop_ev,
            history_policy_net_cost,
            history_total_utility_bottom,
            history_ev_adoption_rate_bottom
        ) = single_policy_with_seeds(params_policy, controller_files)

        outputs[(policy1, policy2)] = {
            "history_driving_emissions": history_driving_emissions_arr,
            "history_production_emissions": history_production_emissions_arr,
            "history_total_emissions": history_total_emissions_arr,
            "history_prop_EV": history_prop_EV_arr,
            "history_total_utility": history_total_utility_arr,
            "history_market_concentration": history_market_concentration_arr,
            "history_total_profit": history_total_profit_arr,
            "history_mean_profit_margins_ICE": history_mean_profit_margins_ICE,
            "history_mean_profit_margins_EV": history_mean_profit_margins_EV,
            "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
            "history_policy_net_cost": history_policy_net_cost,
            "history_mean_car_age": history_mean_car_age,
            "history_lower_percentile_price_ICE_EV_arr": history_lower_percentile_price_ICE_EV_arr,
            "history_upper_percentile_price_ICE_EV_arr": history_upper_percentile_price_ICE_EV_arr,
            "history_mean_price_ICE_EV_arr": history_mean_price_ICE_EV_arr,
            "history_median_price_ICE_EV_arr": history_median_price_ICE_EV_arr,
            "history_past_new_bought_vehicles_prop_ev": history_past_new_bought_vehicles_prop_ev,
            "history_total_utility_bottom": history_total_utility_bottom,
            "history_ev_adoption_rate_bottom": history_ev_adoption_rate_bottom
        }

    save_object(outputs, root_folder + "/Data", "outputs")
    save_object(base_params, root_folder + "/Data", "base_params")
    print(f"All top 10 policies processed and saved in '{root_folder}'")

    #######################################################################################################
    #DELETE CALIBRATION RUNS
    shutil.rmtree(Path(root_folder) / "Calibration_runs", ignore_errors=True)

if __name__ == "__main__":
    main(
        fileName_load="results/endog_pair_00_11_36__02_04_2025",
        min_ev_uptake = 0.945,
        max_ev_uptake = 0.955
    )