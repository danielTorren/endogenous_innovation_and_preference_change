# imports
from package.resources.run import generate_data
import cProfile

def main(base_params): 
    Data = generate_data(base_params)  # run the simulation

if __name__ == '__main__':

    ###################################################################
    base_params = {
        "cars_init_state": 1,
        "duration_no_carbon_price": 276,#that is 23 years,#01/2000-12/2022
        "duration_future":156,#156,#01/2023 - 12/2035  (TOTAL of 432 steps )
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "choice_seed": 9,
        "age_limit_second_hand": 3,
        "ev_research_start_time": 60,#2005
        "EV_nu_diff_state": 1,
        "EV_rebate_state": 1,
        "parameters_rebate":{#THIS IS A CALIBRATION PARAMETER, REFLECTIVE OF CALIFORNIA
            "start_time": 120,#2010
            "end_time":276,
            "rebate": 7500,
            "used_rebate": 1500
        },
        "parameters_scenarios":{
            "States":{
                "Gas_price": "Current", #"Current","High",#
                "Electricity_price": "Current", #"Current","High",#
                "Grid_emissions_intensity": "Decarbonised",#"Weaker", # "Decarbonised"
                "EV_Substitutability": "Parity", #"Improved", #"Parity"
            },
            "Values":{
                "Gas_price":{
                    "Low": 0.5, 
                    "Current": 1,
                    "High": 1.5,
                },
                "Electricity_price":{
                    "Low": 0.5, 
                    "Current": 1,
                    "High": 1.5,
                },
                "Grid_emissions_intensity":{
                    "Weaker": 0.5, 
                    "Decarbonised": 0
                },
                "EV_Substitutability":{
                    "Improved": 1.2, 
                    "Parity": 1
                }
            }
        },
        "parameters_policies":{
            "States":{
                "Carbon_price": "High",#"Zero","Low","High"
                "Adoption_subsidy": "High", #"High"
            },
            "Values":{
                "Carbon_price":{
                    "Zero":{
                        "carbon_price_init": 0,
                        "carbon_price": 0,#0.1,#$/kgCO2 ie $100/tonneC02 would bee 100/1000 = 0.1
                        "carbon_price_state": "linear"
                    },
                    "Low":{
                        "carbon_price_init": 0,
                        "carbon_price": 0.03,#0.1,#$/kgCO2 ie $100/tonneC02 would bee 100/1000 = 0.1
                        "carbon_price_state": "linear"
                    },
                    "High":{
                        "carbon_price_init": 0,
                        "carbon_price": 0.1,#0.1,#$/kgCO2 ie $100/tonneC02 would bee 100/1000 = 0.1
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
        "parameters_EV":{
            "landscape_seed": 14,
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "nu_i_t_multiplier":0.0355,#1,#1/17.5 mph in kph, https://www.blairramirezlaw.com/worst-days-for-commuting-in-los-angeles
            "production_emissions":9000,#kgC02
            "delta": 8*10e-5,#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
            "transportType": 3,
            "min_max_Quality": [900, 2700],#parametersised based on eta which is used to parameterise max min distances
            "min_max_Efficiency": [4,7],
            "min_max_Cost": [1000,30000],
        },
        "parameters_ICE":{
            "landscape_seed": 18, 
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "nu_i_t":0.0355,#1,#1/17.5 mph in kph, https://www.blairramirezlaw.com/worst-days-for-commuting-in-los-angeles
            "production_emissions":6000,#kgC02,
            "delta": 8*10e-5,#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
            "transportType": 2,
            "min_max_Quality": [900, 2700],#[50,200],#[450,700],#[50,200],
            "min_max_Efficiency":[0.5,1.5], #historial min and max for period are (0.953754,1.252405)
            "min_max_Cost": [1000,30000],
        },
        "parameters_firm_manager": {
            "init_tech_seed": 99,
            "J": 10,
            "init_car_age_max": 240
        },
        "parameters_firm":{
            "memory_cap": 30,
            "prob_innovate": 0.2,#0.08333,
            "lambda_pow": 2,
            "init_price": 1,
            "init_base_U": 10e5,#JUST NEEDS TO BE BIG ENOGUHT THAT THE INIT UTILITY IS NOT NEGATIVE
            "innovation_seed": 77,
            "num_cars_production": 5
        },
        "parameters_social_network":{
            "num_individuals": 500,#500,#200,
            "network_structure_seed": 8,
            "SW_K": 30,
            "SW_prob_rewire": 0.1,
            "init_vals_environmental_seed": 66,
            "init_vals_innovative_seed":99, 
            "init_vals_price_seed": 8, 
            "d_min_seed": 45,
            "social_network_seed": 66,
            "d_i_min": 100,#in km
            "gamma_multiplier": 1,#THE INTENITION OF THIS IS TOO MATCH THE SCALE OF THE EMISSIONS AND COST
            "beta_multiplier": 1,
            "a_environment": 4,#large easy ev adoption 
            "b_environment": 2,#2,
            "a_innovativeness": 0.9,#0.9,#1.2,#0.6,#1,#TRY TO MATCH 18% of people innovators from LAVE-Trans#low easy ev adoption 
            "b_innovativeness": 1,#1,#2,#1,#2,
            "a_price": 4,#most people price sensitive
            "b_price": 1,#1,
            "selection_bias": 5
        },
        "parameters_vehicle_user":{
            "kappa": 3,
            "alpha": 0.8,
            "r": 0.02,
            "eta": 1,#10e3,#THE INTENITION OF THIS IS TOO MATCH THE SCALE OF THE EMISSIONS AND COST
            "mu": 0.3,
            "second_hand_car_max_consider": 200,
            "new_car_max_consider": 200
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