"""Run simulation 

Created: 22/12/2023
"""

# imports
import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
import multiprocessing
from package.model.controller import Controller
from package.resources.utility import load_object
from copy import deepcopy

# modules
####################################################################################################################################################
#LOAD IN THE DATA TIME SERIES
#THE IDEA HERE IS THAT I LOAD IN ALL THE TIME SERIES DATA FOR CALIFORNIA FROM 2000 TO 2024 WHere i have it

#####################################################################################################################################################
####SINGLE SHOT RUN
def generate_data(parameters: dict,print_simu = 0):
    """
    Generate the data from a single simulation run

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

    Returns
    -------
    social_network: Network
        Social network that has evolved from initial conditions
    """

    #load_in_output_data()
    #future_electricity_emissions_intensity_data = future_calibration_data()
    
    calibration_data_input = load_object("package/calibration_data", "calibration_data_input")
    parameters["calibration_data"] =  calibration_data_input

    if print_simu:
        start_time = time.time()
    parameters["time_steps_max"] = parameters["duration_burn_in"] + parameters["duration_no_carbon_price"] + parameters["duration_future"]

    #print("tim step max", parameters["time_steps_max"],parameters["burn_in_duration"], parameters["carbon_price_duration"])
    controller = Controller(parameters)
    #controller = CombinedController(parameters)
    

    #### RUN TIME STEPS
    """FIX THIS!!!"""
    #while controller.t_controller < parameters["time_steps_max"]:
    while controller.t_controller < parameters["time_steps_max"]-1:
        controller.next_step()
        #print("step: ", round((controller.t_controller/parameters["time_steps_max"]),3)*100)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
        
    return controller

def load_in_controller(controller_load, base_params_future):

    #Need to change stuff so that it now runs till the future
    base_params_future["time_steps_max"] = controller_load.duration_burn_in + controller_load.duration_no_carbon_price + base_params_future["duration_future"]

    controller = controller_load
    controller.setup_continued_run_future(base_params_future)

    #### RUN TIME STEPS
    while controller.t_controller < base_params_future["time_steps_max"]-1:
        controller.next_step()

    return controller


#########################################################################################
#multi-run
def generate_emissions(params):
    data = generate_data(params)
    return data.social_network.emissions_flow_history 

def emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_emissions)(i) for i in params_dict)

    return np.asarray(emissions_list)
########################################################################################
def generate_ev_prop(params):
    data = generate_data(params)
    return data.social_network.history_prop_EV

def ev_prop_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    ev_prop_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_ev_prop)(i) for i in params_dict)

    return np.asarray(ev_prop_list)

########################################################################################
def generate_distance(params):
    data = generate_data(params)
    return data.social_network.history_distance_individual

def distance_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    distance_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_distance)(i) for i in params_dict)

    return np.asarray(distance_list)

def generate_distance_ev_prop(params):
    data = generate_data(params)
    return data.social_network.history_distance_individual, data.social_network.history_prop_EV

def distance_ev_prop_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_distance_ev_prop)(i) for i in params_dict)
    distance_list, ev_prop_list = zip(
        *res
    )
    return np.asarray(distance_list), np.asarray(ev_prop_list)

#########################################################################################
def generate_multi(params):
    data = generate_data(params)
    return data.social_network.history_distance_individual, data.social_network.history_prop_EV, data.social_network.history_car_age, data.social_network.history_mean_price, data.social_network.history_driving_emissions
#data_flat_age, data_flat_price , data_flat_emissions 

def distance_ev_prop_age_price_emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_multi)(i) for i in params_dict)
    distance_list, ev_prop_list, age_list, price_list, emissions_list = zip(
        *res
    )
    return np.asarray(distance_list), np.asarray(ev_prop_list), np.asarray(age_list) , np.asarray(price_list) , np.asarray(emissions_list) 

#########################################################################################

def policy_generate_multi(params, controller_load):
    print("controller loaded!")
    data = load_in_controller(controller_load, params)
    print("E: ",data.social_network.total_production_emissions, data.social_network.total_driving_emissions)
    return data.social_network.history_distance_individual, data.social_network.history_prop_EV, data.social_network.history_car_age, data.social_network.history_mean_price, data.social_network.history_driving_emissions, data.social_network.history_quality_ICE, data.social_network.history_quality_EV, data.social_network.history_efficiency_ICE, data.social_network.history_efficiency_EV, data.social_network.history_production_cost_ICE, data.social_network.history_production_cost_EV, data.social_network.history_distance_individual_ICE, data.social_network.history_distance_individual_EV

def policy_parallel_run(
        params_dict,
        controller
) -> npt.NDArray:
    res = [policy_generate_multi(params_dict[i],deepcopy(controller)) for i in range(len(params_dict))]
    
    num_cores = multiprocessing.cpu_count()
    #res = Parallel(n_jobs=num_cores, verbose=10)(delayed(policy_generate_multi)(i, params_dict[i], deepcopy(controller) ) for i in range(len(params_dict)))
    distance_list, ev_prop_list, age_list, price_list, emissions_list, quality_ICE_list, quality_EV_list, efficiency_ICE_list, efficiency_EV_list, production_cost_ICE_list, production_cost_EV_list, distance_individual_ICE_list, distance_individual_EV_list = zip(
        *res
    )
    return np.asarray(distance_list), np.asarray(ev_prop_list), np.asarray(age_list) , np.asarray(price_list) , np.asarray(emissions_list) , quality_ICE_list, quality_EV_list, efficiency_ICE_list, efficiency_EV_list, production_cost_ICE_list, production_cost_EV_list, distance_individual_ICE_list, distance_individual_EV_list

#########################################################################################

#########################################################################################

######################################################################################

def parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    data_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_data)(i) for i in params_dict)

    return np.asarray(data_list)

######################################################################################

def parallel_run_multi_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_data)(i) for i in params_dict)

    return res