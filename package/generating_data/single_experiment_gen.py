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
    #print(
    #    "emisisons driving final", 
    #    controller.social_network.total_driving_emissions,
    #    controller.social_network.total_utility,
    #    controller.social_network.total_distance_travelled
    #    )

    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
        "duration_no_carbon_price":360,
        "duration_small_carbon_price":0,
        "duration_large_carbon_price":0,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "choice_seed": 9,
        "age_limit_second_hand": 5,
        "parameters_carbon_policy":{
            "carbon_price_init": 0,
            "carbon_price": 0.3,
            "carbon_price_state": "linear"
        },
        "parameters_future_carbon_policy":{
            "carbon_price_init": 1,
            "carbon_price": 0,
            "carbon_price_state": "linear"
        },
        "parameters_ICE":{
            "landscape_seed": 14, 
            "N": 8,
            "K": 1,
            "A": 3,
            "rho":[0,0.5],
            "fuel_cost_c_z": 1,
            "e_z_t":0.01,
            "nu_z_i_t":0.5,
            "emissions":0.1,
            "delta_z":0.5,
            "transportType": 2,
            "min_max_Quality": [50,100],
            "min_max_Efficiency": [1,5],
            "min_max_Cost": [100,200],
        },
        "parameters_EV":{
            "landscape_seed": 13,
            "N": 8,
            "K": 1,
            "A": 3,
            "rho":[0,0.5],
            "fuel_cost_c_z": 1,#0.01,
            "e_z_t": 0.01,#0.001,
            "nu_z_i_t":0.5,
            "emissions":0.1,
            "delta_z":0.5,
            "transportType": 3,
            "min_max_Quality": [50,100],
            "min_max_Efficiency": [1,20],
            "min_max_Cost": [100,200],
        },
        "parameters_urban_public_transport":{
            "attributes": [75,2.5,10],
            "price": 0.1,
            "id": -1, 
            "firm" : -1, 
            "transportType" : 0,
            "fuel_cost_c_z": 0,
            "e_z_t":0.001,
            "nu_z_i_t":0.8,
            "emissions":0,
            "delta_z":0
        },
        "parameters_rural_public_transport":{
            "attributes": [75,2.5,10],
            "price": 0.1,
            "id" : -2, 
            "firm" : -2,
            "transportType" : 1,
            "fuel_cost_c_z": 0,
            "e_z_t":0.001,
            "nu_z_i_t":1,
            "emissions":0,
            "delta_z":0
        },
        "parameters_firm_manager": {
            "init_tech_seed": 99,
            "J": 10
        },
        "parameters_firm":{
            "memory_cap": 30,
            "prob_innovate": 0.3,
            "lambda_pow": 2,
            "init_price": 0.8,
            "innovation_seed": 77
        },
        "parameters_social_network":{
            "num_individuals": 200,
            "save_timeseries_data_state": 1,
            "network_structure_seed": 8,
            "init_vals_environmental_seed": 66,
            "init_vals_innovative_seed":99, 
            "init_vals_price_seed": 8, 
            "d_min_seed": 45,
            "network_density": 0.05, 
            "prob_rewire": 0.1,
            "a_environment": 4,#easy ev adoption 
            "b_environment": 1,
            "a_innovativeness": 1,#easy ev adoption 
            "b_innovativeness": 4,
            "a_price": 4,#most people price sensitive
            "b_price": 1,

        },
        "parameters_vehicle_user":{
            "kappa": 2,
            "alpha": 0.5,
            "d_i_min": 0,
            "r": 1,
            "eta": 1,
            "mu": 1
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
