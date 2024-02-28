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
        "burn_in_no_OD": 30,
        "burn_in_duration_no_policy": 470,
        "policy_duration": 500,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        'carbon_price': 1,
        "parameters_firm_manager": {
            "J": 30,
            "N": 15,#15,
            "K": 6,#6,#6,
            "alpha":1,
            "rho":0,#0.75,
            "landscape_seed": 3,
            "init_tech_heterogenous_state": 1,
            "init_carbon_premium_heterogenous_state": 0,
            "expected_carbon_premium": 0.05,
            "expected_carbon_premium_init_sigma": 0.01,
            "nk_multiplier": 1,
            "c_max": 10,
            "c_min": 1,
            "ei_max": 10,
            "ei_min": 1,
        },
        "parameters_firm": {
            "static_carbon_premium_heterogenous_state": 1,
            "endogenous_mark_up_state":0,
            "research_cost": 0.1,
            "markup_adjustment": 1,
            "firm_phi": 0.01,
            "markup_init": 0.25,
            "firm_budget": 500,
            "static_tech_state": 1,
            "memory_cap": 30,
            "jump_scale": 2,
            "segment_number": 3,
            "theta": 1,#COME BACK TO THIS AT SOMEPOINT
            "num_individuals_surveyed": 30,#Add just this so its a PROPORTION OF POPULATION
            "survey_cost": 0.1,
            "survey_bool": 1,
            "survey_stoch_prob":0.1,
            "sunk_captital_cost": 0.01
        },
        "parameters_social_network":{  
            "fixed_preferences_state": 0,
            "heterogenous_emissions_intensity_penalty_state": 0,
            "heterogenous_substitutability_state": 0,
            "heterogenous_expenditure_state":0,
            "redistribution_state": 1,      
            'save_timeseries_data_state': 1,
            'imperfect_learning_state': 0,
            'quantity_state':"replicator_utility",#"replicator", # "optimal","alt_optimal","replicator_utility" 
            "social_influence_state": "threshold_average",#common_knowledge,lowest_EI,relative_EI,relative_price_EI, threshold_EI, threshold_price, threshold_average
            'network_structure_seed': 8, 
            'init_vals_seed': 5, 
            'imperfect_learning_seed': 4, 
            'num_individuals': 100, 
            'network_density': 0.1, 
            'individual_phi': 0.05,
            'prob_rewire': 0.1,
            'homophily': 0,
            'substitutability': 20, #0.95 parameter
            "std_substitutability":0.5,
            'a_preferences': 1, 
            'b_preferences': 4, #TRY TO APPROXIMATE THE SKEWNORM STUFF MIQUEL HAS
            "preference_mul": 1,
            'clipping_epsilon': 1e-3, 
            'clipping_epsilon_init_preference': 1e-3,
            'std_low_carbon_preference': 0.01, 
            'std_learning_error': 0.02, 
            "std_emissions_intensity_penalty": 0.1,
            "emissions_intensity_penalty": 1,
            'confirmation_bias': 5, 
            'total_expenditure': 1,
            "expenditure_inequality_const":1,
            "chi_ms": 1,
            "omega": 2#THE LOWER, THE LESS STRINGENT IT IS SO THE GREENER IMITATION
        }
    }
    
    fileName = main(base_params=base_params)
    print("SIMULATION FINISHED")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1
    social_plots = 1
    firm_plots = 0

    if RUN_PLOT:
        plotting_main(fileName = fileName, social_plots = social_plots, firm_plots = firm_plots)
