from package.resources.run import load_in_controller
from package.analysis.endogenous_policy_intensity_single_gen import update_policy_intensity, set_up_calibration_runs
from package.resources.utility import (
    save_object, 
    load_object,
)
from joblib import Parallel, delayed, load
import multiprocessing
import numpy as np
import shutil  # Cleanup
from pathlib import Path  # Path handling
from copy import deepcopy

def calc_top_policies_welfare(pairwise_outcomes_complied, min_val, max_val):
    #GET BEST POLICY FROM EACH COMBINATION
    policy_welfare = {}

    # Collect data and compute intensity ranges
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            welfare = entry["mean_utility_cumulative"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"]
            policy_key = (policy1, policy2)
            if policy_key not in policy_welfare or welfare > policy_welfare[policy_key]["welfare"]:
                policy_welfare[policy_key] = {
                    "welfare": welfare,
                    "policy1_value": entry["policy1_value"],
                    "policy2_value": entry["policy2_value"]
                }

    # Return top 10 policy combinations by welfare
    top_10 = dict(sorted(policy_welfare.items(), key=lambda item: item[1]["welfare"], reverse=True)[:10])

    return top_10

def calc_top_policies_pairs(pairwise_outcomes_complied, min_val, max_val):
    #GET BEST POLICY FROM EACH COMBINATION
    policy_welfare = {}

    # Collect data and compute intensity ranges
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            welfare = entry["mean_utility_cumulative"] #+ entry["mean_profit_cumulative"] - entry["mean_net_cost"]
            policy_key = (policy1, policy2)

            if (policy_key not in policy_welfare or welfare > policy_welfare[policy_key]["welfare"]) and (entry["policy1_value"] > 0 and entry["policy2_value"] > 0):#GET THE BEST OF THAT POLICY OMPETATION
                policy_welfare[policy_key] = {
                    "welfare": welfare,
                    "policy1_value": entry["policy1_value"],
                    "policy2_value": entry["policy2_value"]
                }

    return policy_welfare

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
        data.history_policy_net_cost
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
        history_total_emissions,#Emmissions flow
        history_prop_EV, 
        history_car_age, 
        history_lower_percentile_price_ICE_EV,
        history_upper_percentile_price_ICE_EV,
        history_mean_price_ICE_EV,
        history_median_price_ICE_EV, 
        history_total_utility, 
        history_market_concentration,
        history_total_profit, 
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        history_policy_net_cost
    ) = zip(*res)

        # Return results as arrays where applicable
    return (
        np.asarray(history_driving_emissions_arr),#Emmissions flow
        np.asarray(history_production_emissions_arr),
        np.asarray(history_total_emissions),#Emmissions flow
        np.asarray(history_prop_EV), 
        np.asarray(history_car_age), 
        np.asarray(history_lower_percentile_price_ICE_EV),
        np.asarray(history_upper_percentile_price_ICE_EV),
        np.asarray(history_mean_price_ICE_EV),
        np.asarray(history_median_price_ICE_EV), 
        np.asarray(history_total_utility), 
        np.asarray(history_market_concentration),
        np.asarray(history_total_profit),
        history_quality_ICE, 
        history_quality_EV, 
        history_efficiency_ICE, 
        history_efficiency_EV, 
        history_production_cost_ICE, 
        history_production_cost_EV, 
        history_mean_profit_margins_ICE,
        history_mean_profit_margins_EV,
        np.asarray(history_policy_net_cost)
    )


def main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],
        min_val = 0.945,
        max_val = 1
        ):
    
    fileName = fileNames[0]

    ##########################################################################################

    pairwise_outcomes_complied = {}
    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)
    #print(list(pairwise_outcomes_complied.keys()))
    top_policies = calc_top_policies_pairs(pairwise_outcomes_complied, min_val, max_val)

    print("Top policies done")
    ##########################################################################################

    base_params = load_object(fileName + "/Data", "base_params")
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

    controller_files, base_params, root_folder  = set_up_calibration_runs(base_params, "top_ten")
    print("DONE calibration")
    #
    #NOW SAVE
    save_object(top_policies, root_folder + "/Data", "top_policies")

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
        fileNames=["results/all_policies_16_41_47__24_03_2025"],
        min_val = 0.945,
        max_val = 0.955
    )