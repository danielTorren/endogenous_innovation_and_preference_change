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
    print("uptake: calibration, end",  controller.social_network.history_ev_adoption_rate[controller.duration_burn_in + controller.duration_no_carbon_price],controller.calc_EV_prop())
    print("distortion",controller.calc_total_policy_distortion())
    print("mean price", controller.social_network.history_mean_price[-1])
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
    "seed_repetitions": 12,
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
        "remove_seed": 48
    },
    "ev_research_start_time":60,
    "ev_production_start_time": 96,
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
                    "Carbon_price_state": "linear"
                },
                "Low":{
                    "Carbon_price_init": 0,
                    "Carbon_price": 0.1,
                    "Carbon_price_state": "linear"
                },
                "High":{
                    "Carbon_price_init": 0,
                    "Carbon_price": 2,#2000
                    "Carbon_price_state": "linear"
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
                "Low": 2.5,
                "High":20
            },
            "Adoption_subsidy_used":{
                "Zero":0,
                "Low":0.5,
                "High":20
            },
            "Production_subsidy":{
                "Zero":0,
                "Low":2.5,
                "High":20
            },
            "Research_subsidy":{
                "Zero":0,
                "Low":2.5,
                "High":20
            }
        }
    },
    "parameters_second_hand":{
        "age_limit_second_hand": 12,
        "max_num_cars_prop": 0.1,
        "burn_in_second_hand_market": 12,
        "scrap_price": 1
    },
    "parameters_ICE":{
        "N": 15,
        "K": 3,
        "A": 3,
        "rho":[0,0],
        "production_emissions":6,
        "delta": 0.001,#0.010411,
        "transportType": 2,
        "mean_Price": 40,
        "min_Price": 20,
        "max_Price": 100,
        "min_Efficiency": 0.5,
        "max_Efficiency": 1.5,
        "min_Cost": 5,
        "max_Cost": 50
    },
    "parameters_EV":{
        "N": 15,
        "K": 3,
        "A": 3,
        "rho":[0,0],
        "production_emissions":9,
        "transportType": 3,
        "min_Efficiency": 4,
        "max_Efficiency": 7
    },
    "parameters_firm_manager": {
        "J": 20,
        "init_car_age_max": 240,
        "time_steps_tracking_market_data":12,
        "gamma_threshold_percentile": 50,
        "num_beta_segments": 4,
    },
    "parameters_firm":{
        "memory_cap": 30,
        "prob_innovate": 0.083,
        "prob_change_production": 0.083,
        "lambda_pow": 5,
        "init_price_multiplier": 1.1,
        "min profit": 0.1#need to make 100 dollars
    },
    "parameters_social_network":{
        "num_individuals": 10000,
        "chi_max": 0.9,
        "SW_network_density": 0.01,
        "SW_prob_rewire": 0.1,
        "WTP_mean": 0.210,
        "WTP_sd": 0.175,
        "gamma_epsilon": 1e-5,
        "car_lifetime_months": 240,
        "a_innovativeness": 1,
        "b_innovativeness": 1,
        "prob_switch_car":0.083,
    },
    "parameters_vehicle_user":{
        "kappa":1,
        "U_segments_init": 0,#needs to be high to avoid issues with initial step and arg of lambert function
        "W_calibration":1e20,#does matter too much the quality is fairly inert to the whole thing
        "min_W": 0,#SET TO ZERO IF IT WORKS WELL
        "r": 0.0016,
        "mu": 1,
        "nu":1,
        "alpha": 0.5
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
