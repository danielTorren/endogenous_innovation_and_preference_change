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

    controller = generate_data(base_params, print_simu= 0)  # run the simulation 
    print("E, total",controller.social_network.emissions_cumulative)
    print("uptake",  controller.calc_EV_prop())
    print("distortion",controller.calc_total_policy_distortion())
    
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
    "seed_repetitions": 10,
    "duration_burn_in": 60,
    "duration_no_carbon_price": 276,
    "duration_future": 156,#156,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "seeds":{
        "choice_seed": 9,
        "remove_seed": 48,
        "landscape_seed_ICE": 26,
        "landscape_seed_EV": 14, 
        "init_tech_seed": 99,
        "innovation_seed": 77,
        "network_structure_seed": 8,
        "init_vals_environmental_seed": 66,
        "init_vals_innovative_seed":99, 
        "init_vals_price_seed": 8, 
        "social_network_seed": 66
    },
    "ev_research_start_time": 60,
    "ev_production_start_time": 60,
    "EV_rebate_state": 1,
    "parameters_rebate_calibration":{
        "start_time": 120,
        "rebate": 10000,
        "used_rebate": 1000,
        "rebate_count_cap": 70000,
        "pop": 39370000,
        "rebate_low": 2500,
        "used_rebate_low": 1000
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
                    "Carbon_price": 10,
                    "Carbon_price_state": "linear"
                }
            },
            "Discriminatory_corporate_tax":{
                "Zero":{
                    "corporate_tax": 0
                },
                "Low":{
                    "corporate_tax": 0.05
                },
                "High":{
                    "corporate_tax": 1
                }
            },
            "Electricity_subsidy":{                
                "Zero":{
                    "electricity_price_subsidy": 0
                },
                "Low":{
                    "electricity_price_subsidy": 0.1
                },
                "High":{
                    "electricity_price_subsidy": 1
                }
            },
            "Adoption_subsidy":{
                "Zero":{
                    "rebate": 0
                },
                "Low":{
                    "rebate": 2500
                },
                "High":{
                    "rebate": 50000
                }
            },
            "Adoption_subsidy_used":{
                "Zero":{
                    "rebate": 0
                },
                "Low":{
                    "rebate": 500
                },
                "High":{
                    "rebate": 1000
                }
            },
            "Production_subsidy":{
                "Zero":{
                    "rebate": 0
                },
                "Low":{
                    "rebate": 2500
                },
                "High":{
                    "rebate": 5000
                }
            },
            "Research_subsidy":{
                "Zero":{
                    "rebate": 0
                },
                "Low":{
                    "rebate": 2500
                },
                "High":{
                    "rebate": 5000
                }
            }
        }
    },
    "parameters_second_hand":{
        "age_limit_second_hand": 120,#36,
        "burn_in_second_hand_market": 12,
        "scrap_price": 1000,
        "max_num_cars_prop": 0.3
    },
    "parameters_ICE":{
        "N": 15,
        "K": 4,
        "A": 3,
        "rho":[0,0],
        "production_emissions":6000,
        "delta": 0.001,
        "transportType": 2,
        "min_Price": 20000,
        "max_Price": 80000,
        "min_Efficiency": 0.3,
        "max_Efficiency": 1.444,
        "min_Cost": 5000,
        "max_Cost": 30000
    },
    "parameters_EV":{
        "N": 15,
        "K": 4,
        "A": 3,
        "rho":[0,0],
        "production_emissions":9000,
        "transportType": 3,
        "min_Efficiency": 4,
        "max_Efficiency": 7
    },
    "parameters_firm_manager": {
        "J": 30,
        "init_car_age_max": 240,
        "time_steps_tracking_market_data":12,
        "beta_threshold_percentile": 50,
        "gamma_threshold_percentile": 50,
        "num_beta_segments": 5
    },
    "parameters_firm":{
        "memory_cap": 30,
        "prob_innovate": 0.083,
        "prob_change_production": 0.083,
        "lambda_pow": 5,
        "num_cars_production": 16,
        "init_U": 10e5,
        "init_price_multiplier": 3
    },
    "parameters_social_network":{
        "num_individuals": 3000,
        "d_max": 4000,
        "d_min": 500,
        "SW_network_density": 0.05,
        "SW_prob_rewire": 0.1,
        "WTP_mean": 0,#210,
        "WTP_sd": 0.0,#175,
        "gamma_epsilon": 0,#1e-8,
        "car_lifetime_months": 192,
        "a_innovativeness": 0.7,
        "b_innovativeness": 1,
        "chi_max": 0.9,
        "prob_switch_car": 0.083
    },
    "parameters_vehicle_user":{
        "kappa": 1,#30,
        "r": 0.00417,
        "mu": 0.2,
        "nu": 1#1,#10e-4
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
