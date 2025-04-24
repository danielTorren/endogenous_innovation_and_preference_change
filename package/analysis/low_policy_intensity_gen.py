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
    # Collect all policies and initialize min/max ranges
    policy_ranges = {}


    for (p1, p2), entries in pairwise_outcomes_complied.items():
        for entry in entries:
            for policy, key in zip((p1, p2), ("policy1_value", "policy2_value")):
                policy_ranges.setdefault(policy, {"min": float('inf'), "max": float('-inf')})
                policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], entry[key])
                policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], entry[key])

    # Merge symmetric pairs
    merged = {}
    for (p1, p2), entries in pairwise_outcomes_complied.items():
        key = tuple(sorted((p1, p2)))
        merged.setdefault(key, []).extend([entry | {"original_order": (p1, p2)} for entry in entries])

    #print(list(merged.keys()), len(list(merged.keys())))

    best_entries = {}
    for (pa, pb), entries in merged.items():
        best = None
        min_intensity = float('inf')
        for entry in entries:
            if (min_val <= entry["mean_ev_uptake"] <= max_val):
                p1, p2 = entry["original_order"]
                v1 = entry["policy1_value"]
                v2 = entry["policy2_value"]

                norm1 = (v1 - policy_ranges[p1]["min"]) / (policy_ranges[p1]["max"] - policy_ranges[p1]["min"] or 1)
                norm2 = (v2 - policy_ranges[p2]["min"]) / (policy_ranges[p2]["max"] - policy_ranges[p2]["min"] or 1)

                if (m := max(norm1, norm2)) < min_intensity:
                    min_intensity = m
                    best = (v1, v2, entry["mean_ev_uptake"], entry["original_order"])

            if best:
                v1, v2, uptake, (orig_p1, orig_p2) = best
                # Flip if needed to match key order
                best_entries[(pa, pb)] = {
                    "policy1_value": v1 if (pa, pb) == (orig_p1, orig_p2) else v2,
                    "policy2_value": v2 if (pa, pb) == (orig_p1, orig_p2) else v1,
                    "mean_ev_uptake": uptake,
                    "original_order": (pa, pb)
                }

    return best_entries


def main(fileNames,
        min_ev_uptake = 0.945,
        max_ev_uptake = 0.955
        ):

    pairwise_outcomes_complied = {}
    #pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")
    if len(fileNames) == 1:
        fileName = fileNames[0]
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    fileName_load = fileNames[0]
    #pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")

    top_policies = calc_low_intensities(pairwise_outcomes_complied,  min_ev_uptake, max_ev_uptake)
    
    print("top_policies", list(top_policies.keys()), len( list(top_policies.keys())))
    quit()
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

    #base_params["parameters_scenarios"]["Grid_emissions_intensity"] = 1
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

    ######################################################################################################
    #SINGLE POLICIES

    #RUN THE CARBON TAX
    base_params_carbon_tax = deepcopy(base_params)
    base_params_carbon_tax["parameters_policies"]["States"]["Carbon_price"] = 1
    base_params_carbon_tax["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price"] = 0.983
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

    ) = single_policy_with_seeds(base_params_carbon_tax, controller_files)

    outputs_carbon_tax= {
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

    save_object(outputs_carbon_tax, root_folder + "/Data", "outputs_carbon_tax")
    print("DONE SINGLE CARBON TAX")


    #RUN THE ADOPTION SUBSIDY
    base_params_adoption_subsidy = deepcopy(base_params)
    base_params_adoption_subsidy["parameters_policies"]["States"]["Adoption_subsidy"] = 1
    base_params_adoption_subsidy["parameters_policies"]["Values"]["Adoption_subsidy"] = 36638.50
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

    ) = single_policy_with_seeds(base_params_adoption_subsidy, controller_files)

    outputs_adoption_subsidy= {
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

    save_object(outputs_adoption_subsidy, root_folder + "/Data", "outputs_adoption_subsidy")
    print("DONE SINGLE Adoption Subsidy")

    #######################################################################################################
    #DELETE CALIBRATION RUNS
    shutil.rmtree(Path(root_folder) / "Calibration_runs", ignore_errors=True)

if __name__ == "__main__":
    main(
        fileNames=["results/endog_pair_13_13_45__09_04_2025"],
        min_ev_uptake = 0.94,
        max_ev_uptake = 0.96#0.96
    )