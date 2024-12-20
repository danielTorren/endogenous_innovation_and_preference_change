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
    #print(controller.social_network.emissions_flow_history )
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
    "seed_repetitions": 10,
    "cars_init_state": 1,
    "duration_no_carbon_price": 276,
    "duration_future":0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "choice_seed": 9,
    "age_limit_second_hand": 3,
    "ev_research_start_time": 120,
    "EV_rebate_state": 1,
    "parameters_rebate":{
        "start_time": 120,
        "end_time":276,
        "rebate": 10000,
        "used_rebate": 4000
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
            "Carbon_price": "High",
            "Adoption_subsidy": "Zero"
        },
        "Values":{
            "Carbon_price":{
                "Zero":{
                    "carbon_price_init": 0,
                    "carbon_price": 0,
                    "carbon_price_state": "linear"
                },
                "Low":{
                    "carbon_price_init": 0,
                    "carbon_price": 0.03,
                    "carbon_price_state": "linear"
                },
                "High":{
                    "carbon_price_init": 0,
                    "carbon_price": 0.1,
                    "carbon_price_state": "linear"
                }
            },
            "Adoption_subsidys":{
                "Zero":{
                    "rebate": 0,
                    "used_rebate": 0
                },
                "Low":{
                    "rebate": 2500,
                    "used_rebate": 500
                },
                "High":{
                    "rebate": 7500,
                    "used_rebate": 1500
                }
            }
        }
    },
    "parameters_ICE":{
        "landscape_seed": 18, 
        "N": 15,
        "K": 2,
        "A": 3,
        "rho":[0,0],
        "production_emissions":6000,
        "delta": 4e-5,
        "transportType": 2,
        "min_Quality": 0,
        "max_Quality": 16,
        "min_Efficiency": 0.5,
        "max_Efficiency": 1.5,
        "min_Cost": 1000,
        "max_Cost": 30000
    },
    "parameters_EV":{
        "landscape_seed": 14,
        "N": 15,
        "K": 2,
        "A": 3,
        "rho":[0,0],
        "production_emissions":9000,
        "transportType": 3,
        "min_Efficiency": 4,
        "max_Efficiency": 7
    },
    "parameters_firm_manager": {
        "init_tech_seed": 99,
        "J": 10,
        "init_car_age_max": 240
    },
    "parameters_firm":{
        "memory_cap": 30,
        "prob_innovate": 0.4,
        "lambda_pow": 2,
        "innovation_seed": 77,
        "num_cars_production": 5
    },
    "parameters_social_network":{
        "num_individuals": 500,
        "network_structure_seed": 8,
        "init_vals_environmental_seed": 66,
        "init_vals_innovative_seed":99, 
        "init_vals_price_seed": 8, 
        "social_network_seed": 66,
        "d_min_seed": 45,
        "d_i_min": 100,
        "SW_K": 20,
        "SW_prob_rewire": 0.1,
        "prob_rewire": 0.1,
        "a_environment": 0.9,
        "b_environment": 1,
        "WTP_mean": 210,
        "WTP_sd": 175,
        "car_lifetime_months": 192,
        "a_innovativeness": 2.29,
        "b_innovativeness": 2.15,
        "selection_bias": 5
    },
    "parameters_vehicle_user":{
        "kappa": 3,
        "alpha": 0.5,
        "r": 0.05,
        "mu": 0.3,
        "second_hand_car_max_consider": 200,
        "new_car_max_consider": 200
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
