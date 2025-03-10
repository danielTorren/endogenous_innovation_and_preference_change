import json
from package.resources.run import parallel_run_multi_seed
from package.analysis.endogenous_policy_intensity_single_gen import update_policy_intensity
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed,
    load_object
)
import numpy as np

def produce_param_list_for_policy_pair(base_params, policy1_name, policy2_name, policy1_value, policy2_value):
    params = base_params.copy()

    params = update_policy_intensity(params, policy1_name, policy1_value)
    params = update_policy_intensity(params, policy2_name, policy2_value)

    params_list = params_list_with_seed(params)

    return params_list

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
            welfare = entry["mean_utility_cumulative"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"]
            policy_key = (policy1, policy2)
            if welfare > policy_welfare[policy_key]["welfare"]:#GET THE BEST OF THAT POLICY OMPETATION
                policy_welfare[policy_key] = {
                    "welfare": welfare,
                    "policy1_value": entry["policy1_value"],
                    "policy2_value": entry["policy2_value"]
                }

    # Return top 10 policy combinations by welfare
    #top_10 = dict(sorted(policy_welfare.items(), key=lambda item: item[1]["welfare"], reverse=True)[:10])
    print("policy_welfare", policy_welfare)
    return policy_welfare

def main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025",
        min_val = 0.945,
        max_val = 0.955
        ):
    
    fileName = fileNames[0]
    base_params = load_object(fileName + "/Data", "base_params")
    base_params["parameters_policies"]["States"] = {
        "Carbon_price": 0,
        "Targeted_research_subsidy": 0,
        "Electricity_subsidy": 0,
        "Adoption_subsidy": 0,
        "Adoption_subsidy_used": 0,
        "Production_subsidy": 0,
        "Research_subsidy": 0
    }
    #RESET TO B SURE

    pairwise_outcomes_complied = {}
    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    base_params["save_timeseries_data_state"] = 1

    top_policies = calc_top_policies_pairs(pairwise_outcomes_complied, min_val, max_val)

    root_folder = produce_name_datetime("top10_policy_runs")
    createFolder(root_folder)

    print("TOTAL RUNS", len(top_policies)*base_params["seed_repetitions"])
    outputs = {}

    for (policy1, policy2), welfare_data in top_policies.items():

        policy1_value = welfare_data["policy1_value"]
        policy2_value = welfare_data["policy2_value"]

        print(f"Running time series for {policy1} & {policy2}")

        params_list = produce_param_list_for_policy_pair(base_params, policy1, policy2, policy1_value, policy2_value)

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
        history_mean_profit_margins_EV
        ) = parallel_run_multi_seed(
            params_list
        )

        outputs[(policy1, policy2)] = {
            "history_driving_emissions": history_driving_emissions_arr,
            "history_production_emissions": history_production_emissions_arr,
            "history_total_emissions": history_total_emissions_arr,
            "history_prop_EV": history_prop_EV_arr,
            "history_total_utility": history_total_utility_arr,
            "history_market_concentration": history_market_concentration_arr,
            "history_total_profit": history_total_profit_arr,
            "history_mean_profit_margins_ICE": history_mean_profit_margins_ICE,
            "history_mean_profit_margins_EV": history_mean_profit_margins_EV
        }

    save_object(outputs, root_folder + "/Data", "outputs")
    save_object(base_params, root_folder + "/Data", "base_params")

    print(f"All top 10 policies processed and saved in '{root_folder}'")


if __name__ == "__main__":
    main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025",
        min_val = 0.945,
        max_val = 0.955
    )