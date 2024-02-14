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
#from package.plotting_data import single_experiment_plot
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
    #save_object(controller, fileName + "/Data", "controller")
    save_object(controller.social_network, fileName + "/Data", "social_network")
    save_object(controller.firm_manager, fileName + "/Data", "firm_manager")
    save_object(base_params, fileName + "/Data", "base_params")
    #SAVE AS CSV SO THAT MIQUEL CAN USE THEM
    save_csv = 0
    if save_csv:
        save_data_csv_list_firm_manager = ["history_decimal_value_current_tech","history_list_neighouring_technologies_strings","history_filtered_list_strings", "history_random_technology_string"]
        save_data_csv_firms(controller.firm_manager.firms_list, save_data_csv_list_firm_manager, fileName + "/Data")

    return fileName

if __name__ == '__main__':

    base_params = {
        "burn_in_duration": 0,
        "policy_duration": 3000,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        'carbon_price': 0,
        "parameters_firm_manager": {
            "J": 30,
            "N": 15,#15,
            "K": 1,#6,#6,
            "alpha":1,
            "rho":-1,#0.75,
            "landscape_seed": 3,
            "init_tech_heterogenous_state": 0,
            "init_carbon_premium_heterogenous_state": 0,
            "expected_carbon_premium": 0.05,
            "expected_carbon_premium_init_sigma": 0.01,
            "nk_multiplier": 1
        },
        "parameters_firm": {
            "static_carbon_premium_heterogenous_state": 1,
            "research_cost": 1,
            "markup_adjustment": 1,
            "firm_phi": 0.005,
            "markup_init": 0.25,
            "firm_budget": 50,
            "static_tech_state": 0,
            "memory_cap": 30,
            "jump_scale": 2,
            "segment_number": 3,
            "theta": 1,#COME BACK TO THIS AT SOMEPOINT
            "num_individuals_surveyed": 30,#Add just this so its a PROPORTION OF POPULATION
            "survey_cost": 1
        },
        "parameters_social_network":{  
            "fixed_preferences_state": 1,#"fixed_preferences",#"dynamic_multi_sector_weights",#"fixed_preferences",#"dynamic_multi_sector_weights",
            "heterogenous_emissions_intensity_penalty_state": 0,
            "heterogenous_substitutability_state": 0,
            "heterogenous_expenditure_state":0,
            "redistribution_state": 0,      
            'save_timeseries_data_state': 1,
            'imperfect_learning_state': 0,
            'quantity_state':"replicator", # "optimal"
            "social_influence_state": "relative_EI",#common_knowledge,lowest_EI,relative_EI,relative_price_EI
            'network_structure_seed': 8, 
            'init_vals_seed': 5, 
            'imperfect_learning_seed': 4, 
            'num_individuals': 100, 
            'network_density': 0.1, 
            'individual_phi': 0.005,
            'prob_rewire': 0.1,
            'homophily': 0.9,
            'substitutability': 1.5, 
            "std_substitutability":0.5,
            'a_preferences': 2, 
            'b_preferences': 2, 
            #"shift_preferences": -1,
            'clipping_epsilon': 1e-3, 
            'clipping_epsilon_init_preference': 1e-3,
            'std_low_carbon_preference': 0.01, 
            'std_learning_error': 0.02, 
            "std_emissions_intensity_penalty": 3,
            "mean_emissions_intensity_penalty": 1,
            'confirmation_bias': 10, 
            'total_expenditure': 1,
            "expenditure_inequality_const":1,
            "preference_mul": 1,
            "chi_ms": 1
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
        plotting_main(fileName = fileName, social_plots = social_plots,firm_plots = firm_plots)
