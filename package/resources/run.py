"""Run simulation 

Created: 22/12/2023
"""

# imports
import time
#import numpy as np
#import numpy.typing as npt
#from joblib import Parallel, delayed
#import multiprocessing
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

    parameters["time_steps_max"] = parameters["burn_in_duration"] + parameters["policy_duration"]

    #print("tim step max", parameters["time_steps_max"],parameters["burn_in_duration"], parameters["carbon_price_duration"])
    controller = Controller(parameters)

    #### RUN TIME STEPS
    while controller.t < parameters["time_steps_max"]:
        controller.next_step()
        #print("step", social_network.t)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return controller