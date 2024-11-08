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
    print(
        "emisisons driving final", 
        controller.social_network.total_driving_emissions,
        controller.social_network.total_utility,
        controller.social_network.total_distance_travelled
        )

    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
        "duration_no_carbon_price":20,
        "duration_small_carbon_price":0,
        "duration_large_carbon_price":0,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "choice_seed": 3,
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
            "landscape_seed": 10, 
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "e_z_t":0.01,
            "nu_z_i_t":0.5,
            "emissions":0.1,
            "delta_z":0.05,
            "transportType": 2,
            "min_max_Quality": [50,200],
            "min_max_Efficiency": [1,5],
            "min_max_Cost": [10,50],
        },
        "parameters_EV":{
            "landscape_seed": 20,
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "e_z_t":0.005,
            "nu_z_i_t":0.5,
            "emissions":0.2,
            "delta_z":0.05,
            "transportType": 3,
            "min_max_Quality": [50,200],
            "min_max_Efficiency": [1,5],
            "min_max_Cost": [10,50],
        },
        "parameters_urban_public_transport":{
            "attributes": [10,1,10],
            "price": 0.1,
            "id": -1, 
            "firm" : -1, 
            "transportType" : 0,
            "e_z_t":0.001,
            "nu_z_i_t":0.8,
            "emissions":0.01,
            "delta_z":0
        },
        "parameters_rural_public_transport":{
            "attributes": [10,1,10],
            "price": 0.1,
            "id" : -2, 
            "firm" : -2,
            "transportType" : 1,
            "e_z_t":0.001,
            "nu_z_i_t":1,
            "emissions":0.01,
            "delta_z":0
        },
        "parameters_firm_manager": {
            "init_tech_seed": 20,
            "J": 10
        },
        "parameters_firm":{
            "memory_cap": 30,
            "prob_innovate": 0.3,
            "lambda_pow": 2,
            "init_price": 0.8,
            "innovation_seed": 66
        },
        "parameters_social_network":{
            "num_individuals": 40,
            "save_timeseries_data_state": 1,
            "emissions_flow_social_influence_state": 0,
            "network_structure_seed": 8,
            "init_vals_environmental_seed": 66,
            "init_vals_innovative_seed":99, 
            "init_vals_price_seed": 8, 
            "d_min_seed": 45,
            "network_density": 0.05, 
            "prob_rewire": 0.1,
            "homophily": 0,
            "a_environment": 1, 
            "b_environment": 1,
            "a_innovativeness": 0.5,
            "b_innovativeness": 4,
            "a_price": 1,
            "b_price": 1,
            "clipping_epsilon": 1e-5, 
            "confirmation_bias": 20
        },
        "parameters_vehicle_user":{
            "nu": 1,
            "vehicles_available": 1,
            "EV_bool": 0,
            "kappa": 10,
            "alpha": 0.8,
            "d_i_min": 1,
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
