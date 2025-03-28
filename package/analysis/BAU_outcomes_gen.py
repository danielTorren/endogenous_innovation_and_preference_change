import json
import numpy as np
from package.analysis.endogenous_policy_intensity_single_gen import  set_up_calibration_runs, single_policy_with_seeds
from package.resources.utility import (
    save_object, 
)
import shutil  # Cleanup
from pathlib import Path  # Path handling

def main(
    BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json"
):
    """
    Main function for running pairwise policy optimization.
    """
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    controller_files, base_params, file_name = set_up_calibration_runs(base_params, "BAU_runs")

    ###################################################################################################################
    EV_uptake_arr, total_cost_arr, net_cost_arr, emissions_cumulative_arr, emissions_cumulative_driving_arr, emissions_cumulative_production_arr, utility_cumulative_arr, profit_cumulative_arr = single_policy_with_seeds(base_params, controller_files)

    mean_ev_uptake = np.mean(EV_uptake_arr)
    sd_ev_uptake = np.std(EV_uptake_arr)
    confidence_interval = 1.96 * np.std(EV_uptake_arr) / np.sqrt(len(EV_uptake_arr))
    mean_total_cost = np.mean(total_cost_arr)
    mean_net_cost = np.mean(net_cost_arr)
    mean_emissions_cumulative = np.mean(emissions_cumulative_arr)
    mean_emissions_cumulative_driving = np.mean(emissions_cumulative_driving_arr)
    mean_emissions_cumulative_production = np.mean(emissions_cumulative_production_arr)
    mean_utility_cumulative = np.mean(utility_cumulative_arr)
    mean_profit_cumulative = np.mean(profit_cumulative_arr)

    outcomes = {
        "mean_ev_uptake": mean_ev_uptake,
        "sd_ev_uptake": sd_ev_uptake,
        "confidence_interval": confidence_interval,
        "mean_total_cost": mean_total_cost,
        "mean_net_cost": mean_net_cost, 
        "mean_emissions_cumulative": mean_emissions_cumulative, 
        "mean_emissions_cumulative_driving": mean_emissions_cumulative_driving, 
        "mean_emissions_cumulative_production": mean_emissions_cumulative_production, 
        "mean_utility_cumulative": mean_utility_cumulative, 
        "mean_profit_cumulative": mean_profit_cumulative,
        "ev_uptake": EV_uptake_arr,
        "net_cost": net_cost_arr,
        "emissions_cumulative": emissions_cumulative_arr,
        "emissions_cumulative_driving": emissions_cumulative_driving_arr,
        "emissions_cumulative_production": emissions_cumulative_production_arr,
        "utility_cumulative": utility_cumulative_arr,
        "profit_cumulative": profit_cumulative_arr
    }

    print("outcomes", outcomes)

    save_object(outcomes, file_name + "/Data", "outcomes")#

    shutil.rmtree(Path(file_name) / "Calibration_runs", ignore_errors=True)

    return "Done"

if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
    )