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
    print("uptake",  controller.calc_EV_prop())
    print("distortion",controller.calc_total_policy_distortion())
    
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
    "seed_repetitions": 12,
    "duration_burn_in": 60,
    "duration_no_carbon_price": 276,
    "duration_future": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "seeds":{
        "init_tech_seed": 99,
        "landscape_seed_ICE": 22,
        "social_network_seed": 66,
        "network_structure_seed": 8,
        "init_vals_environmental_seed": 66,
        "init_vals_innovative_seed":99,
        "init_vals_price_seed": 8,
        "innovation_seed": 77,
        "landscape_seed_EV": 14, 
        "choice_seed": 9,
        "remove_seed": 48
    },
    "ev_research_start_time": 96,
    "ev_production_start_time": 96,
    "EV_rebate_state": 1,
    "parameters_rebate_calibration":{
        "start_time": 120,
        "rebate": 10000,
        "used_rebate": 1000
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
            "Carbon_price": "Zero",
            "Discriminatory_corporate_tax": "Zero",
            "Electricity_subsidy": "Zero",
            "Adoption_subsidy": "Zero",
            "Adoption_subsidy_used": "Zero",
            "Production_subsidy": "Zero",
            "Research_subsidy": "Zero"
        },
        "Values":{
            "Carbon_price":{
                "Zero":{
                    "Carbon_price_init": 0,
                    "Carbon_price": 0,
                    "Carbon_price_state": "flat"
                },
                "Low":{
                    "Carbon_price_init": 0,
                    "Carbon_price": 0.1,
                    "Carbon_price_state": "flat"
                },
                "High":{
                    "Carbon_price_init": 0,
                    "Carbon_price": 2,
                    "Carbon_price_state": "flat"
                }
            },
            "Discriminatory_corporate_tax":{
                "Zero": 0,
                "Low":0.05,
                "High":0.99
            },
            "Electricity_subsidy":{                
                "Zero":0,
                "Low": 0.01,
                "High": 1
            },
            "Adoption_subsidy":{
                "Zero": 0,
                "Low": 2500,
                "High":20000
            },
            "Adoption_subsidy_used":{
                "Zero":0,
                "Low":500,
                "High":20000
            },
            "Production_subsidy":{
                "Zero":0,
                "Low":2500,
                "High":20000
            },
            "Research_subsidy":{
                "Zero":0,
                "Low":2500,
                "High":20000
            }
        }
    },
    "parameters_second_hand":{
        "age_limit_second_hand": 12,
        "max_num_cars_prop": 0.3,
        "burn_in_second_hand_market": 12,
        "scrap_price": 500
    },
    "parameters_ICE":{
        "N": 15,
        "K": 3,
        "A": 3,
        "rho":[0,0],
        "production_emissions":6000,
        "delta": 0.002,
        "transportType": 2,
        "min_Price": 20000,
        "max_Price": 120000,
        "min_Efficiency": 0.5,
        "max_Efficiency": 1.5,
        "min_Cost": 5000,
        "max_Cost": 30000
    },
    "parameters_EV":{
        "N": 15,
        "K": 3,
        "A": 3,
        "rho":[0,0],
        "production_emissions":9000,
        "transportType": 3,
        "min_Efficiency": 4,
        "max_Efficiency": 7
    },
    "parameters_firm_manager": {
        "J": 10,
        "init_car_age_max": 240,
        "time_steps_tracking_market_data":12,
        "gamma_threshold_percentile": 50,
        "num_beta_segments": 4,
        "minimum_segment_utility": 1e4
    },
    "parameters_firm":{
        "memory_cap": 30,
        "prob_innovate": 0.083,
        "prob_change_production": 0.083,
        "lambda_pow": 5,
        "num_cars_production": 16,
        "init_U": 1e4,
        "init_price_multiplier": 1.1
    },
    "parameters_social_network":{
        "num_individuals": 5000,
        "d_max": 2500,
        "chi_max": 0.9,
        "SW_network_density": 0.05,
        "SW_prob_rewire": 0.1,
        "WTP_mean": 210,
        "WTP_sd": 175,
        "gamma_epsilon": 1e-5,
        "car_lifetime_months": 192,
        "a_innovativeness": 0.9,
        "b_innovativeness": 1,
        "prob_switch_car":0.083
    },
    "parameters_vehicle_user":{
        "kappa":10,
        "r": 0.00417,
        "mu": 0.5,
        "alpha": 3.5
    }
}
    
    fileName = main(base_params=base_params)
    print("SIMULATION FINISHED")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1

    if RUN_PLOT:
        plotting_main(fileName = fileName)
