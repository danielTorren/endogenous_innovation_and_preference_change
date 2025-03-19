"""Runs a single simulation to produce data which is saved

Created: 22/12/2023
"""
# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
from package.plotting_data.single_experiment_plot import main as plotting_main
import pyperclip
import numpy as np
def main(
    base_params
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    controller = generate_data(base_params, print_simu= 1)  # run the simulation 
    print("E, total",controller.social_network.emissions_cumulative)
    print("uptake end calibration", controller.social_network.history_ev_adoption_rate[controller.duration_burn_in + controller.duration_calibration - 1])
    print("uptake end",controller.calc_EV_prop())
    print("distortion",controller.calc_total_policy_distortion(), controller.calc_net_policy_distortion())
    print("mean price", np.asarray(controller.social_network.history_mean_price[-1]))
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
    "seed_repetitions": 64,
    "duration_burn_in_research": 0,
    "duration_burn_in": 240,
    "duration_calibration": 276,
    "duration_future": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "seed": 8,
    "seed_inputs": 20,
    "ev_research_start_time":60,
    "ev_production_start_time": 60,
    "EV_rebate_state": 1,
    "parameters_rebate_calibration":{
        "start_time": 108,
        "rebate": 10000,
        "used_rebate": 1000
    },
    "parameters_scenarios":{
            "Gas_price": 1,
            "Electricity_price": 1,
            "Grid_emissions_intensity": 0.5
    },
    "parameters_policies":{
        "States":{
            "Carbon_price": 0,
            "Electricity_subsidy": 0,
            "Adoption_subsidy": 0,
            "Adoption_subsidy_used": 0,
            "Production_subsidy": 0,
            "Research_subsidy": 0
        },
        "Values":{
            "Carbon_price":{
                "Carbon_price_init": 0,
                "Carbon_price": 0.1,
                "Carbon_price_state": "flat"
            },
            "Electricity_subsidy":0.5,
            "Adoption_subsidy":10000,
            "Adoption_subsidy_used":20000,
            "Production_subsidy":58150,
            "Research_subsidy":1000000
        }
    },
    "parameters_second_hand":{
        "age_limit_second_hand": 12,
        "max_num_cars_prop": 0.3,
        "burn_in_second_hand_market": 12,
        "scrap_price": 669.8
    },
    "parameters_ICE":{
        "prop_explore": 0.1,
        "N": 15,
        "K": 3,
        "A": 3,
        "rho":[1,0,0],
        "production_emissions":10000,
        "delta": 0.0022,
        "delta_P": 0.0116,
        "transportType": 2,
        "mean_Price": 39290,
        "min_Efficiency": 0.79,
        "max_Efficiency": 3.09,
        "min_Quality": 0,
        "max_Quality": 1,
        "fuel_tank": 469.4,
        "min_Cost": 0,
        "max_Cost": 58150
    }, 
    "parameters_EV":{
        "prop_explore": 0.1,
        "N": 15,
        "K": 3,
        "A": 4,
        "rho":[1,0,0,0.5],
        "delta_P":0.0087,
        "production_emissions":14000,
        "transportType": 3,
        "min_Efficiency": 2.73,
        "max_Efficiency": 9.73,
        "min_Battery_size": 0,
        "max_Battery_size": 130,
        "min_Cost": 0,
        "max_Cost": 58150
    },
    "parameters_firm_manager": {
        "J": 16,
        "init_car_age_mean": 0,
        "init_car_age_std": 0,
        "time_steps_tracking_market_data":12,
        "gamma_threshold_percentile": 50,
        "num_beta_segments": 5,
        "num_gamma_segments": 2
    },
    "parameters_firm":{
        "lambda": 1e-3,
        "memory_cap": 30,
        "prob_innovate": 0.083,
        "prob_change_production": 0.083,
        "init_price_multiplier": 1.2,
        "min_profit": 0,
        "max_cars_prod": 10
    },
    "parameters_social_network":{
        "num_individuals":5000,
        "chi_max": 0.9,
        "a_chi": 1.2, 
        "b_chi": 2.2873,
        "SW_network_density": 0.01,
        "SW_prob_rewire": 0.1,
        "WTP_E_mean": 46646.65434,
        "WTP_E_sd": 39160.31118,
        "nu": 1174,
        "gamma_epsilon": 1e-5,
        "nu_epsilon": 1e-5,
        "prob_switch_car":0.083,
        "proportion_zero_target": 0.005,
        "income_mu":11.225,
        "income_sigma":0.927
    },
    "parameters_vehicle_user":{
        "kappa": 2.25e-4,
        "U_segments_init": 0,
        "W_calibration":1e10,
        "min_W": 1e-10,
        "r": 0.00407412378,
        "mu": 0.2,
        "alpha": 0.5,
        "zeta":0.29697
    }
}
    
    fileName = main(base_params=base_params)
    print("SIMULATION FINISHED")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1
    print("fileName",fileName)
    if RUN_PLOT:
        plotting_main(fileName = fileName)
