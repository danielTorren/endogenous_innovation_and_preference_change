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
    parameters["time_steps_max"] = parameters["duration_no_carbon_price"] + parameters["duration_future"]

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
    #print("E: ",controller.social_network.total_production_emissions, controller.social_network.total_driving_emissions)
    #quit()
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
#multi-run
def generate_preferences(params):
    data = generate_data(params)
    return data.social_network.low_carbon_preference_arr

def preferences_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    preferences_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_preferences)(i) for i in params_dict)

    return np.asarray(preferences_list)
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