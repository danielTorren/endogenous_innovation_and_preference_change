# imports
from package.resources.run import generate_data
import cProfile

def main(base_params): 
    Data = generate_data(base_params)  # run the simulation

if __name__ == '__main__':

    ###################################################################
    base_params = {
    "seed_repetitions": 10,
    "duration_burn_in": 60,#24,
    "duration_no_carbon_price": 276,
    "duration_future":0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "choice_seed": 9,
    "ev_research_start_time": 60,#200
    "ev_production_start_time": 108,#2008
    "EV_rebate_state": 1,
    "parameters_rebate_calibration":{
        "start_time": 120,
        "end_time":276,
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
            "Carbon_price": "High",
            "Adoption_subsidy": "High"
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
                    "carbon_price": 0.1,
                    "carbon_price_state": "linear"
                },
                "High":{
                    "carbon_price_init": 0,
                    "carbon_price": 0.5,
                    "carbon_price_state": "linear"
                }
            },
            "Adoption_subsidy":{
                "Zero":{
                    "rebate": 0,
                    "used_rebate": 0
                },
                "Low":{
                    "rebate": 2500,
                    "used_rebate": 500
                },
                "High":{
                    "rebate": 10000,
                    "used_rebate": 1000
                }
            }
        }
    },
    "parameters_second_hand":{
        "remove_seed": 48,
        "age_limit_second_hand": 3,
        "max_num_cars": 5000,
        "burn_in_second_hand_market": 12,#12,#first year no second hand cars

    },
    "parameters_ICE":{
        "landscape_seed": 18, 
        "N": 15,
        "K": 2,
        "A": 3,
        "rho":[0,0],
        "production_emissions":6000,
        "delta": 0.002,
        "transportType": 2,
        "min_Quality": 0,
        "max_Quality": 15,
        "min_Efficiency": 0.5,
        "max_Efficiency": 1.5,
        "min_Cost": 5000,
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
        "min_Quality": 0,
        "max_Quality": 7,
        "min_Efficiency": 4,
        "max_Efficiency": 7
    },
    "parameters_firm_manager": {
        "init_tech_seed": 99,
        "J": 30,
        "init_car_age_max": 240,
        "time_steps_tracking_market_data":12
    },
    "parameters_firm":{
        "memory_cap": 30,
        "prob_innovate": 0.083,
        "prob_change_production": 0.083,
        "lambda_pow": 5,
        "innovation_seed": 77,
        "num_cars_production": 8,
        "init_U_sum": 10e7,
        "init_price_multiplier": 1,
        "price_adjust_monthly": 0.01
    },
    "parameters_social_network":{
        "num_individuals": 10000,
        "network_structure_seed": 8,
        "init_vals_environmental_seed": 66,
        "init_vals_innovative_seed":99, 
        "init_vals_price_seed": 8, 
        "social_network_seed": 66,
        "d_min_seed": 45,
        "d_i_min": 700,
        "d_i_min_sd": 700,
        "SW_network_density": 0.05,
        "SW_prob_rewire": 0.1,
        "WTP_mean": 210,
        "WTP_sd": 175,
        "car_lifetime_months": 192,
        "a_innovativeness": 0.7287,
        "b_innovativeness": 0.9186,
        "selection_bias": 5,
        "prob_switch_car": 0.083,#0.021#CAN SWITCH CARS ONCE EVERY 4 years
    },
    "parameters_vehicle_user":{
        "kappa": 9.9481,#17.7372,#30,
        "alpha": 0.5,
        "r": 0.00417,#5%/12
        "mu": 0.1
    }
}
    # Create a profiler object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Run your model code here
    main(base_params)

    # Stop profiling
    pr.disable()

    # Save profiling results to a file
    pr.dump_stats('profile_results.prof')

    # Visualize with snakeviz
    # Run in terminal: snakeviz profile_results.prof