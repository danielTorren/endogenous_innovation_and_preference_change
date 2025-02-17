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

def main(
    base_params
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    controller = generate_data(base_params, print_simu= 1)  # run the simulation 
    print("E, total",controller.social_network.emissions_cumulative)
    print("uptake end calibration", controller.social_network.history_ev_adoption_rate[420])
    print("uptake end",controller.calc_EV_prop())
    print("distortion",controller.calc_total_policy_distortion())
    print("mean price", controller.social_network.history_mean_price[-1])
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params ={
    "seed_repetitions": 1,
    "duration_burn_in": 144,
    "duration_no_carbon_price": 276,
    "duration_future": 156,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "seeds":{
        "init_tech_seed": 96,
        "landscape_seed_ICE": 27,
        "social_network_seed": 66,
        "network_structure_seed": 8,
        "init_vals_environmental_seed": 66,
        "init_vals_innovative_seed":99,
        "init_vals_price_seed": 8,
        "innovation_seed": 75,
        "landscape_seed_EV": 11, 
        "choice_seed": 9,
        "remove_seed": 48,
        "init_vals_poisson_seed": 95,
        "init_vals_range_seed": 77
    },
    "ev_research_start_time":96,
    "ev_production_start_time": 120,
    "EV_rebate_state": 1,
    "parameters_rebate_calibration":{
        "start_time": 120,
        "rebate": 10,
        "used_rebate": 1
    },
    "parameters_scenarios":{
        "States":{
            "Gas_price": "Current",
            "Electricity_price": "Current",
            "Grid_emissions_intensity": "Weaker"
        },
        "Values":{
            "Gas_price":{
                "Low": 0.5, 
                "Current": 1,
                "High": 1.5
            },
            "Electricity_price":{
                "Low": 0.5, 
                "Current": 1,
                "High": 1.5
            },
            "Grid_emissions_intensity":{
                "Weaker": 0.5, 
                "Decarbonised": 0
            }
        }
    },
    "parameters_policies":{
        "States":{
            "Carbon_price": 1,
            "Discriminatory_corporate_tax": 0,
            "Electricity_subsidy": 0,
            "Adoption_subsidy": 0,
            "Adoption_subsidy_used": 0,
            "Production_subsidy": 0,
            "Research_subsidy": 0
        },
        "Values":{
            "Carbon_price":{
                "Carbon_price_init": 0,
                "Carbon_price": 0.2,
                "Carbon_price_state": "flat"
            },
            "Discriminatory_corporate_tax":0.95,
            "Electricity_subsidy":0.99,
            "Adoption_subsidy":20,
            "Adoption_subsidy_used":20,
            "Production_subsidy":20,
            "Research_subsidy":200
        }
    },
    "parameters_second_hand":{
        "age_limit_second_hand": 12,
        "max_num_cars_prop": 0.3,
        "burn_in_second_hand_market": 12,
        "scrap_price": 0.6698
    },
    "parameters_ICE":{
        "prop_explore": 0.1,
        "N": 15,
        "K": 3,
        "A": 3,
        "rho":[0,0],
        "production_emissions":10,
        "delta": 0.002,
        "delta_P": 0.0116,
        "transportType": 2,
        "mean_Price": 40,
        "min_Price": 20,
        "max_Price": 100,
        "min_Efficiency": 0.79,
        "max_Efficiency": 3.09,
        "min_Cost": 12.91,
        "max_Cost": 43.72,
        "min_Quality": 0,
        "max_Quality": 1,
        "fuel_tank": 674
    }, 
    "parameters_EV":{
        "prop_explore": 0.1,
        "N": 15,
        "K": 3,
        "A": 4,
        "rho":[0,0,0],
        "delta": 0.004,
        "delta_P":0.0087,
        "production_emissions":14,
        "transportType": 3,
        "min_Efficiency": 2.73,
        "max_Efficiency": 9.73,
        "min_Battery_size": 10,
        "max_Battery_size": 150
    },
    "parameters_firm_manager": {
        "J": 20,
        "init_car_age_max": 240,
        "time_steps_tracking_market_data":12,
        "gamma_threshold_percentile": 50,
        "num_beta_segments": 4
    },
    "parameters_firm":{
        "lambda": 1e-4,
        "memory_cap": 30,
        "prob_innovate": 0.083,
        "prob_change_production": 0.083,
        "init_price_multiplier": 1.1,
        "min profit": 0.1
    },
    "parameters_social_network":{
        "num_individuals": 3000,
        "chi_max": 0.9,
        "a_chi": 0.01,
        "b_chi": 5,
        "SW_network_density": 0.05,
        "SW_prob_rewire": 0.1,
        "WTP_E_mean": 46646.65434,
        "WTP_E_sd": 39160.31118,
        "WTP_R_mean": 0.086,
        "WTP_R_sd": 0.051,
        "gamma_epsilon": 1e-5,
        "nu_epsilon": 1e-5,
        "prob_switch_car":0.083,
        "proportion_zero_target": 0.01
    },
    "parameters_vehicle_user":{
        "kappa":1,
        "U_segments_init": 0,
        "W_calibration":1e20,
        "min_W": 1e-5,
        "r":  0.005,#0.0016515813,
        "mu": 1,
        "alpha": 0.5,
        "zeta":0.3
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
