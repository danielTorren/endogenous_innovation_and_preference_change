from package.resources.utility import load_object, save_object
import numpy as np
from package.resources.run import load_in_controller
from package.analysis.endogenous_policy_intensity_single_gen import update_policy_intensity, set_up_calibration_runs
from package.resources.utility import (
    save_object, 
    load_object,
)
from joblib import Parallel, delayed, load
import multiprocessing
import shutil  # Cleanup
from pathlib import Path  # Path handling
from copy import deepcopy
from package.analysis.top_ten_gen import single_policy_with_seeds

def calc_low_intensities(pairwise_outcomes_complied, min_val, max_val):
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    # Compute policy ranges
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        for entry in data:
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])

    best_entries = {}

    # Find the entry with the lowest max intensity per pair
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        min_max_intensity = float('inf')
        best_entry = None

        for entry in filtered_data:
            # Normalize intensity
            policy1_norm = (entry["policy1_value"] - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"])
            policy2_norm = (entry["policy2_value"] - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"])

            # Determine the max of the two normalized values
            max_intensity = max(policy1_norm, policy2_norm)

            if max_intensity < min_max_intensity:
                min_max_intensity = max_intensity
                best_entry = entry

        if best_entry:
            best_entries[(policy1, policy2)] = best_entry
            print((policy1, policy2), best_entries[(policy1, policy2)]["policy1_value"],best_entries[(policy1, policy2)]["policy2_value"])
    #print(best_entries)

    return best_entries


def main(fileName_load):

    pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")

    min_ev_uptake = 0.945
    max_ev_uptake = 1#0.955

    #base_params = load_object(f"{fileName}/Data", "base_params")
    
    top_policies = calc_low_intensities(pairwise_outcomes_complied,  min_ev_uptake, max_ev_uptake)
    print()
    ##########################################################################################

    base_params = load_object(fileName_load + "/Data", "base_params")
    if "duration_calibration" not in base_params:
        base_params["duration_calibration"] = base_params["duration_no_carbon_price"]

    base_params["parameters_policies"]["States"] = {
        "Carbon_price": 0,
        "Targeted_research_subsidy": 0,
        "Electricity_subsidy": 0,
        "Adoption_subsidy": 0,
        "Adoption_subsidy_used": 0,
        "Production_subsidy": 0,
        "Research_subsidy": 0
    }

    controller_files, base_params, root_folder  = set_up_calibration_runs(base_params, "low_intensity_policies")
    print("DONE calibration")
    #
    #NOW SAVE
    save_object(top_policies, root_folder + "/Data", "top_policies")
    save_object(min_ev_uptake, root_folder + "/Data", "min_ev_uptake")
    save_object(max_ev_uptake, root_folder + "/Data", "max_ev_uptake")

    ###########################################################################################

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
    history_policy_net_cost
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
            "history_policy_net_cost": history_policy_net_cost
    }
    save_object(outputs_BAU, root_folder + "/Data", "outputs_BAU")
    print("DONE BAU")
    ##############################################################################################################################

    print("TOTAL RUNS", len(top_policies)*base_params["seed_repetitions"])
    outputs = {}

    for (policy1, policy2), welfare_data in top_policies.items():

        policy1_value = welfare_data["policy1_value"]
        policy2_value = welfare_data["policy2_value"]

        print(f"Running time series for {policy1} & {policy2}")


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
        history_policy_net_cost
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
            "history_policy_net_cost": history_policy_net_cost
        }

    save_object(outputs, root_folder + "/Data", "outputs")
    save_object(base_params, root_folder + "/Data", "base_params")
    print(f"All top 10 policies processed and saved in '{root_folder}'")

    #######################################################################################################
    #DELETE CALIBRATION RUNS
    shutil.rmtree(Path(root_folder) / "Calibration_runs", ignore_errors=True)


if __name__ == "__main__":
    main(
        fileName_load="results/endog_pair_15_49_56__21_03_2025",
    )