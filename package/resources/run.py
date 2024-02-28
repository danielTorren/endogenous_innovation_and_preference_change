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



# modules
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

    if print_simu:
        start_time = time.time()
    parameters["time_steps_max"] = parameters["burn_in_no_OD"] + parameters["burn_in_duration_no_policy"] + parameters["policy_duration"]

    #print("tim step max", parameters["time_steps_max"],parameters["burn_in_duration"], parameters["carbon_price_duration"])
    controller = Controller(parameters)

    #### RUN TIME STEPS
    while controller.t_controller < parameters["time_steps_max"]:
        controller.next_step()
        #print("step: ", round((controller.t_controller/parameters["time_steps_max"]),3)*100)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    #print("E: ",controller.social_network.total_carbon_emissions_cumulative)
    #quit()
    return controller

#########################################################################################
#multi-run
def generate_emissions_intensities(params):
    data = generate_data(params)
    return data.social_network.total_carbon_emissions_cumulative, data.firm_manager.weighted_emissions_intensity

def emissions_intensities_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_emissions_intensities)(i) for i in params_dict)

    emissions_list, emissions_intensities_list = zip(
        *res
    )
    return np.asarray(emissions_list), np.asarray(emissions_intensities_list)