"""Runs a single simulation to produce data which is saved

Created: 22/12/2023
"""
# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    save_data_csv_firm_manager,
    produce_name_datetime
)
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
    save_data_csv_list_firm_manager = ["history_time_firm_manager","history_emissions_intensities_vec","history_prices_vec", "history_market_share_vec"]
    save_data_csv_firm_manager(controller.firm_manager,save_data_csv_list_firm_manager, fileName + "/Data")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':

    base_params = {
        "burn_in_duration": 0,
        "policy_duration": 5,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 10,
        "parameters_firm_manager": {
            "J": 3,
            "N": 5,
            "K": 0,
            "alpha":1,
            "rho": 1,
        },
        "parameters_firm": {
                "research_cost": 1,
                "expected_carbon_premium": 0,
                "markup_adjustment": 0.1,
                "firm_phi": 0.01,
                "markup_init": 0,
                "firm_budget": 100
        },
        "parameters_social_network":{        
            'save_timeseries_data_state': 1,
            'heterogenous_intrasector_preferences_state': 1.0,
            'compression_factor_state': 10, 
            'ratio_preference_or_consumption_state': 0.0, 
            "nu_change_state": "dynamic_culturally_determined_weights",
            'network_structure_seed': 8, 
            'init_vals_seed': 14, 
            'imperfect_learning_seed': 4, 
            'num_individuals': 10, 
            'individual_phi': 0.005, 
            'network_density': 0.1, 
            'prob_rewire': 0.1, 
            'homophily': 0.0, 
            'substitutability': 1.5, 
            'a_identity': 2, 
            'b_identity': 2, 
            'clipping_epsilon': 1e-3, 
            'clipping_epsilon_init_preference': 1e-3,
            'std_low_carbon_preference': 0.01, 
            'std_learning_error': 0.02, 
            'confirmation_bias': 5, 
            'expenditure': 1,
            'carbon_price': 0, 
        }
        }
    
    fileName = main(base_params=base_params)

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 0

    #if RUN_PLOT:
    #    single_experiment_plot.main(fileName = fileName)
