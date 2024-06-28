"""Runs a single simulation to produce data which is saved

Created: 22/12/2023
"""
# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    save_data_csv_firms,
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
    print("Emmissions = ",controller.social_network.total_carbon_emissions_cumulative)
    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")


    return fileName

if __name__ == "__main__":

    base_params = {
    #"seed_reps":1,
    "duration_no_OD_no_stock_no_policy": 0,
    "duration_OD_no_stock_no_policy": 0,
    "duration_OD_stock_no_policy":240,
    "duration_OD_stock_policy": 360,
    "duration_future":360,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "parameters_carbon_policy":{
        "carbon_price_init": 0,
        "carbon_price": 1,
        "carbon_price_state": "linear"
    },
    "parameters_future_carbon_policy":{
        "carbon_price_init": 1,
        "carbon_price": 0,
        "carbon_price_state": "linear"
    },
    "parameters_future_information_provision_policy":{

    },
    "parameters_firm_manager": {
        "static_tech_state": 0,
        "init_tech_heterogenous_state": 1,
        "J": 30,
        "N": 15,
        "K": 2,
        "alpha":1,
        "rho":[0,0.5],
        "init_tech_seed": 10,
        "landscape_seed": 6,
        "A": 3,
        "markup": 0.1,
        "memory_cap": 30,
        "segment_number": 3,
        "rank_number": 10
    },
    "parameters_social_network":{
        "fixed_preferences_state":0,  
        "redistribution_state": 1,      
        "save_timeseries_data_state": 1,
        "preference_drift_state": 0,
        "heterogenous_init_preferences": 0,
        "cumulative_emissions_preference_state":1,
        "heterogenous_reaction_cumulative_emissions_state":1,
        "heterogenous_gamma_state": 1,
        "init_public_transport_state": 0,
        "consumption_imitation_state":1,
        "network_structure_seed": 8, 
        "init_vals_seed": 8, 
        "preference_drift_seed": 4, 
        "num_individuals": 500, 
        "network_density": 0.05, 
        "prob_rewire": 0.1,
        "homophily": 0,
        "a_preferences": 1, 
        "b_preferences": 10,
        "clipping_epsilon": 1e-5, 
        "preference_drift_std": 0.001, 
        "confirmation_bias": 20,
        "emissions_max": 10000,
        "upsilon": 0.02,
        "upsilon_E": 0.0,
        "upsilon_E_std": 0.1,
        "omega": 2,
        "delta": 0.01,
        "kappa": 2,
        "gamma": 0.5,
        "lambda_poisson_gamma": 0.5,
        "public_transport_attributes": [0.3,0.7,0.1],
        "price_constant": 0.5
        }
    }

    
    fileName = main(base_params=base_params)
    print("SIMULATION FINISHED")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1
    social_plots = 1
    firm_plots = 1

    if RUN_PLOT:
        plotting_main(fileName = fileName, social_plots = social_plots, firm_plots = firm_plots)
