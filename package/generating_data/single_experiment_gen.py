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

    controller = generate_data(base_params)  # run the simulation 

    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")


    return fileName

if __name__ == "__main__":

    base_params = {
        "burn_in_no_OD": 0,
        "burn_in_duration_no_policy": 0,
        "no_climate_impact": 0,
        "policy_duration": 1000,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "parameters_carbon_policy":{
            "carbon_price": 0,
            "carbon_price_state": "flat",
        },
        "parameters_firm_manager": {
            "J": 30,
            "N": 8,
            "K": 4,
            "alpha":1,
            "rho":[0,0.5],#enviromnet not correlated to cost but quality is correlated to cost
            "init_tech_seed": 10,
            "landscape_seed": 6,
            "init_tech_heterogenous_state": 1,
            "A": 3,
            "markup": 0.25,
        },
        "parameters_firm": {
            "static_tech_state": 0,
            "memory_cap": 30,
            "segment_number": 3,
            "theta": 1,
            "rank_number": 4
        },
        "parameters_social_network":{  
            "fixed_preferences_state": 0,
            "redistribution_state": 1,      
            "save_timeseries_data_state": 1,
            "imperfect_learning_state": 1,
            "network_structure_seed": 8, 
            "init_vals_seed": 8, 
            "imperfect_learning_seed": 4, 
            "num_individuals": 500, 
            "network_density": 0.1, 
            "prob_rewire": 0.1,
            "homophily": 0,
            "substitutability": 2,
            "a_preferences": 1, 
            "b_preferences": 4,
            "clipping_epsilon": 1e-3, 
            "preference_drift_std": 0.02, 
            "confirmation_bias": 5,
            "emissions_max": 1e5,
            "upsilon": 0.05,
            "upsilon_E": 0.05 
        },
        "parameters_individual":{
            "individual_upsilon": 0.05,
            "omega": 2,
            "delta": 0.1,
            "kappa": 1,
            "gamma": 0.1
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
