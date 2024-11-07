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

    createFolder(fileName)
    save_object(controller, fileName + "/Data", "controller")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == "__main__":

    base_params = {
        "duration_no_carbon_price":30,
        "duration_small_carbon_price":0,
        "duration_large_carbon_price":0,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
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
            "alpha":0.3,
            "e_z_t":0.5,
            "nu_z_i_t":0.5,
            "eta":0.5,
            "emissions":0.1,
            "delta_z":0.9,
            "transportType": 2
        },
        "parameters_EV":{
            "landscape_seed": 20,
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "alpha":0.3,
            "e_z_t":0.5,
            "nu_z_i_t":0.5,
            "eta":0.5,
            "emissions":0.1,
            "delta_z":0.8,
            "transportType": 3
        },
        "parameters_urban_public_transport":{
            "attributes": [0.5,0.5,0.5],
            "price": 0.1,
            "id": -1, 
            "firm" : -1, 
            "transportType" : 0,
            "e_z_t":0.5,
            "nu_z_i_t":0.5,
            "eta":0.5,
            "emissions":0.1,
            "delta_z":0
        },
        "parameters_rural_public_transport":{
            "attributes": [0.5,0.5,0.5],
            "price": 0.3,
            "id" : -2, 
            "firm" : -2,
            "transportType" : 1,
            "e_z_t":0.5,
            "nu_z_i_t":0.5,
            "eta":0.5,
            "emissions":0.1,
            "delta_z":0
        },
        "parameters_firm_manager": {
            "init_tech_seed": 20,
            "J": 30
        },
        "parameters_firm":{
            "alpha":0.3,
            "r": 1,
            "eta":1,
            "memory_cap": 30,
            "prob_innovate": 0.1,
            "lambda_pow": 2,
            "init_price": 0.8
        },
        "parameters_social_network":{
            "num_individuals": 1000,
            "save_timeseries_data_state": 1,
            "emissions_flow_social_influence_state": 0,
            "network_structure_seed": 8,
            "init_vals_environmental_seed": 66,
            "init_vals_innovative_seed":99, 
            "init_vals_price_seed": 8, 
            "network_density": 0.05, 
            "prob_rewire": 0.1,
            "homophily": 0,
            "a_environment": 1, 
            "b_environment": 1,
            "a_innovativeness": 1,
            "b_innovativeness": 1,
            "a_price": 1,
            "b_price": 1,
            "clipping_epsilon": 1e-5, 
            "confirmation_bias": 20
        },
        "parameters_vehicle_user":{
            "nu": 1,
            "vehicles_available": 1,
            "EV_bool": 0,
            "kappa": 2,
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
