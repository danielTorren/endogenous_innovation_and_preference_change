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
        "duration_no_carbon_price":120,
        "duration_small_carbon_price":120,
        "duration_large_carbon_price":360,
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "choice_seed": 9,
        "age_limit_second_hand": 6,
        "ev_research_start_time": 60,
        "parameters_carbon_policy":{
            "carbon_price_init": 0,
            "carbon_price": 0,
            "carbon_price_state": "linear"
        },
        "parameters_future_carbon_policy":{
            "carbon_price_init": 0,
            "carbon_price": 100,
            "carbon_price_state": "linear"
        },
        "parameters_EV":{
            "landscape_seed": 14,
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "fuel_cost_c_z": 1,#0.01,
            "e_z_t": 1,#0.75,#0.001,
            "nu_z_i_t":1,
            "production_emissions":1,
            "delta_z":0.0005,#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
            "transportType": 3,
            "min_max_Quality": [50,200],
            "min_max_Efficiency": [1,10],
            "min_max_Cost": [10,80],
        },
        "parameters_HEV":{
            "landscape_seed": 33, 
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "fuel_cost_c_z": 1,
            "e_z_t":1,#0.169,
            "nu_z_i_t":1,
            "production_emissions":1,
            "delta_z":0.0005,#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
            "transportType": 4,
            "min_max_Quality": [50,200],
            "min_max_Efficiency": [1,10],
            "min_max_Cost": [10,80],
        },
        "parameters_ICE":{
            "landscape_seed": 18, 
            "N": 15,
            "K": 2,
            "A": 3,
            "rho":[0,0.5],
            "fuel_cost_c_z": 1,
            "e_z_t":1,#0.241,
            "nu_z_i_t":1,
            "production_emissions":1,
            "delta_z":0.0005,#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
            "transportType": 2,
            "min_max_Quality": [50,200],
            "min_max_Efficiency": [1,10],
            "min_max_Cost": [10,80],
        },
        "parameters_urban_public_transport":{
            "attributes": [30,1,1],
            "price": 1,
            "id": -1, 
            "firm" : -1, 
            "transportType" : 0,
            "fuel_cost_c_z": 0,
            "e_z_t":0.001,
            "nu_z_i_t":1,
            "production_emissions":10e4,
            "delta_z":0
        },
        "parameters_rural_public_transport":{
            "attributes": [30,1,1],
            "price": 1,
            "id" : -2, 
            "firm" : -2,
            "transportType" : 1,
            "fuel_cost_c_z": 0,
            "e_z_t":0.001,
            "nu_z_i_t":4,
            "production_emissions":10e4,
            "delta_z":0
        },
        "parameters_firm_manager": {
            "init_tech_seed": 99,
            "J": 10
        },
        "parameters_firm":{
            "memory_cap": 30,
            "prob_innovate": 0.08333,
            "lambda_pow": 2,
            "init_price": 1,
            "init_base_U": 10,#JUST NEEDS TO BE BIG ENOGUHT THAT THE INIT UTILITY IS NOT NEGATIVE
            "innovation_seed": 77,
            "num_cars_production": 2
        },
        "parameters_social_network":{
            "num_individuals": 300,#200,
            "prop_urban": 0.8,
            "network_structure_seed": 8,
            "init_vals_environmental_seed": 66,
            "init_vals_innovative_seed":99, 
            "init_vals_price_seed": 8, 
            "d_min_seed": 45,
            "d_i_min": 0,#10e1,
            "SBM_block_num": 2,
            "SBM_network_density_input_intra_block": 0.2,
            "SBM_network_density_input_inter_block": 0.005,
            "prob_rewire": 0.1,
            "a_environment": 2,#large easy ev adoption 
            "b_environment": 2,#2,
            "a_innovativeness": 0.6,#1,#TRY TO MATCH 18% of people innovators from LAVE-Trans#low easy ev adoption 
            "b_innovativeness": 1,#2,
            "a_price": 4,#most people price sensitive
            "b_price": 1#1,
        },
        "parameters_vehicle_user":{
            "kappa": 3,
            "alpha": 0.8,
            "r": 1,
            "eta": 2,
            "mu": 0.9,
            "second_hand_car_max_consider": 200,
            "new_car_max_consider": 200
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
