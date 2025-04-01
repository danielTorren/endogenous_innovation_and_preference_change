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

# Define all possible measures and their indices
MEASURES = {
    "EV_Uptake": 0,
    "Policy_Distortion": 1,
    "Net_Policy_Cost": 2,
    "Cumulative_Emissions": 3,
    "Driving_Emissions": 4,
    "Production_Emissions": 5,
    "Cumulative_Utility": 6,
    "Cumulative_Profit": 7
}

scale = 350
epsilon = 0

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

def calc_low_intensities_from_array(data_array, policy_pairs, policy_info_dict, min_val, max_val):
    best_entries = {}

    for pair_idx, (policy1, policy2) in enumerate(policy_pairs):
        pair_data = data_array[pair_idx]  # Shape: (len1, len2, seeds, metrics)
        ev_uptake = np.mean(pair_data[:, :, :, MEASURES["EV_Uptake"]], axis=2)

        # Get intensity values from bounds
        p1_min, p1_max = policy_info_dict['bounds_dict'][policy1]
        p2_min, p2_max = policy_info_dict['bounds_dict'][policy2]

        p1_vals = np.linspace(p1_min, p1_max, pair_data.shape[0])
        p2_vals = np.linspace(p2_min, p2_max, pair_data.shape[1])

        min_max_norm = float("inf")
        best_entry = None

        for i in range(pair_data.shape[0]):
            for j in range(pair_data.shape[1]):
                mean_ev = ev_uptake[i, j]
                if  (min_val <= mean_ev <= max_val):
                    # Normalize both intensities
                    p1_norm = (p1_vals[i] - p1_min) / (p1_max - p1_min)
                    p2_norm = (p2_vals[j] - p2_min) / (p2_max - p2_min)
                    max_intensity = max(p1_norm, p2_norm)

                    if max_intensity < min_max_norm:
                        min_max_norm = max_intensity
                        best_entry = {
                            "policy1": policy1,
                            "policy2": policy2,
                            "policy1_value": p1_vals[i],
                            "policy2_value": p2_vals[j],
                            "mean_ev_uptake": mean_ev,
                            "policy1_value_relative": p1_norm,
                            "policy2_value_relative": p2_norm,
                        }

        if best_entry:
            best_entries[(policy1, policy2)] = best_entry
            print(f"{policy1}-{policy2} -> ({best_entry['policy1_value']:.3f}, {best_entry['policy2_value']:.3f})  EV uptake: {best_entry['mean_ev_uptake']:.3f}")
            print(best_entry['policy1_value_relative'], best_entry['policy2_value_relative'])
    return best_entries



def main(fileName_load,
        min_ev_uptake = 0.945,
        max_ev_uptake = 0.955
        ):

    #pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")

    data_array = load_object(fileName_load + "/Data", "data_array")
    policy_pairs = load_object(fileName_load + "/Data", "policy_pairs")
    policy_info_dict = load_object(fileName_load + "/Data", "policy_info_dict")

    top_policies_full = calc_low_intensities_from_array(
        data_array,
        policy_pairs,
        policy_info_dict,
        min_ev_uptake,
        max_ev_uptake
    )
    keys = [('Carbon_price', 'Electricity_subsidy'), ('Carbon_price', 'Adoption_subsidy_used'), ('Electricity_subsidy', 'Production_subsidy')]
    top_policies = {key: top_policies_full[key] for key in keys}

    print(list(top_policies.keys()))

    ##########################################################################################

    base_params_calibration = load_object(fileName_load + "/Data", "base_params")
    base_params_calibration["duration_future"] = 564#2050#144#SET UP FUTURE

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

    controller_files, base_params, root_folder  = set_up_calibration_runs(base_params_calibration, "2D_low_intensity_policies")
    print("DONE calibration")
    #
    #NOW SAVE
    save_object(top_policies, root_folder + "/Data", "top_policies")
    save_object(min_ev_uptake, root_folder + "/Data", "min_ev_uptake")
    save_object(max_ev_uptake, root_folder + "/Data", "max_ev_uptake")

    ###########################################################################################

    base_params["save_timeseries_data_state"] = 1
    print(base_params)

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
        fileName_load="results/vary_two_policies_gen_23_53_57__31_03_2025",
        min_ev_uptake = 0.945,
        max_ev_uptake = 1
    )