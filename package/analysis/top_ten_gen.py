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

def produce_param_list_for_policy_pair(base_params, policy1_name, policy2_name, policy1_value, policy2_value):
    params = base_params.copy()

    params = update_policy_intensity(params, policy1_name, policy1_value)
    params = update_policy_intensity(params, policy2_name, policy2_value)

    params_list = params_list_with_seed(params)

    return params_list

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
        TOP_TEN_LOAD="results/endogenous_policy_intensity_19_30_46__06_03_2025"
        ):
    
    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    top_policies = load_object(f"{TOP_TEN_LOAD}/Data", "top_10")

    root_folder = produce_name_datetime("top10_policy_runs")
    createFolder(root_folder)

    print("TOTAL RUNS", 10*base_params["seed_repetitions"])
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
        #history_car_age_arr, 
        #history_lower_percentile_price_ICE_EV_arr,
        #history_upper_percentile_price_ICE_EV_arr,
        #history_mean_price_ICE_EV_arr,
        #history_median_price_ICE_EV_arr, 
        history_total_utility_arr, 
        history_market_concentration_arr,
        history_total_profit_arr, 
        #history_quality_ICE, 
        #history_quality_EV, 
        #history_efficiency_ICE, 
        #history_efficiency_EV, 
        #history_production_cost_ICE, 
        #history_production_cost_EV, 
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

    save_object(outputs, TOP_TEN_LOAD + "/Data", "outputs")
    save_object(base_params, TOP_TEN_LOAD + "/Data", "base_params")

    print(f"All top 10 policies processed and saved in '{root_folder}'")


if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
        TOP_TEN_LOAD="results/endogenous_policy_intensity_19_30_46__06_03_2025"
    )