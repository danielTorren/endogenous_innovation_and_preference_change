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
from package.resources.utility import load_object, get_num_workers
from copy import deepcopy

# modules
####################################################################################################################################################
#LOAD IN THE DATA TIME SERIES
#THE IDEA HERE IS THAT I LOAD IN ALL THE TIME SERIES DATA FOR CALIFORNIA FROM 2001 TO 2022 WHere i have it

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
    parameters["time_steps_max"] = parameters["duration_burn_in"] + parameters["duration_calibration"] + parameters["duration_future"]

    controller = Controller(parameters)

    #print(controller.t_controller)
    #print("outside")
    #### RUN TIME STEPS
    """FIX THIS!!!"""
    while controller.t_controller < parameters["time_steps_max"]:
        #print(controller.t_controller,"before next step")
        controller.next_step()
        #print(controller.t_controller,"inside")

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
        
    return controller

def load_in_controller(controller_load, base_params_future):

    #Need to change stuff so that it now runs till the future
    base_params_future["time_steps_max"] = controller_load.duration_burn_in + controller_load.duration_calibration + base_params_future["duration_future"]
    #print(base_params_future["save_timeseries_data_state"])
    #print(base_params_future["time_steps_max"],  controller_load.duration_burn_in,  controller_load.duration_calibration , base_params_future["duration_future"])

    controller_load.setup_continued_run_future(base_params_future)
    #### RUN TIME STEPS
    # Matches generate_data()'s own loop convention (< time_steps_max, not
    # time_steps_max - 1) -- the "-1" here previously simulated the future
    # period one month short of the requested duration_future every time
    # (calibration itself, via generate_data, has always ended exactly at
    # t_controller == duration_burn_in + duration_calibration, so this loop
    # needs to add exactly duration_future MORE steps, not duration_future - 1).
    while controller_load.t_controller < base_params_future["time_steps_max"]:
        controller_load.next_step()

    return controller_load


#########################################################################################
#multi-run
def generate_emissions(params):
    data = generate_data(params)
    return data.social_network.emissions_flow_history 

def emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_emissions)(i) for i in params_dict)

    return np.asarray(emissions_list)
########################################################################################
def generate_ev_prop(params):
    data = generate_data(params)
    return data.social_network.history_prop_EV, data.calc_price_range_ice()

def ev_prop_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_ev_prop(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_ev_prop)(i) for i in params_dict)
    ev_prop_list, price_range_ice_list = zip(
        *res
    )

    return np.asarray(ev_prop_list), np.asarray(price_range_ice_list)

########################################################################################
def generate_distance(params):
    data = generate_data(params)
    return data.social_network.history_distance_individual

def distance_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    distance_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_distance)(i) for i in params_dict)

    return np.asarray(distance_list)

def generate_distance_ev_prop(params):
    data = generate_data(params)
    return data.social_network.history_distance_individual, data.social_network.history_prop_EV

def distance_ev_prop_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_distance_ev_prop)(i) for i in params_dict)
    distance_list, ev_prop_list = zip(
        *res
    )
    return np.asarray(distance_list), np.asarray(ev_prop_list)

#########################################################################################
def generate_multi(params):
    data = generate_data(params)
    return data.social_network.history_prop_EV, data.social_network.history_mean_price_ICE_EV, data.firm_manager.history_mean_profit_margins_ICE
#data_flat_age, data_flat_price , data_flat_emissions 

def ev_prop_price_emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_multi)(i) for i in params_dict)
    ev_prop_list, price_list, margins_list = zip(
        *res
    )
    return np.asarray(ev_prop_list), np.asarray(price_list), np.asarray(margins_list)
#########################################################################################

def ev_prop_emissions(params):
    data = generate_data(params)
    return data.social_network.history_prop_EV, data.social_network.history_total_emissions
#data_flat_age, data_flat_price , data_flat_emissions 

def ev_prop_emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(ev_prop_emissions)(i) for i in params_dict)
    ev_prop_list, emissions_list = zip(
        *res
    )
    return np.asarray(ev_prop_list), np.asarray(emissions_list)
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
    
    num_cores = get_num_workers()
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
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    data_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_data)(i) for i in params_dict)

    return np.asarray(data_list)

######################################################################################

def parallel_run_multi_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = get_num_workers()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_data)(i) for i in params_dict)

    return res

##########################################################################################################################
def generate_sensitivity_output_flat(params: dict):
    """
    Generates sensitivity output from input parameters.
    Assumes `generate_data` is a defined function that returns
    an object with the required attributes.
    """
    data = generate_data(params)
    return (
        data.social_network.emissions_cumulative,
        data.calc_EV_prop(),
        data.firm_manager.total_profit,
        data.firm_manager.calc_last_step_HHI(),
        data.social_network.utility_cumulative,
        data.social_network.calc_mean_car_age()
    )

def parallel_run_sa(params_dict: list[dict]):
    """
    Runs the sensitivity analysis in parallel using the given parameter dictionary.
    """
    num_cores = get_num_workers()
    #res = [generate_sensitivity_output_flat(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_sensitivity_output_flat)(params) for params in params_dict
    )
    
    # Unpack the results
    (
        emissions_list,
        ev_prop_list,
        profit_list,
        HHI_list,
        utility_list,
        age_list
    ) = zip(*res)
    
    # Return results as arrays where applicable
    return (
        np.asarray(emissions_list),
        np.asarray(ev_prop_list),
        np.asarray(profit_list),
        np.asarray(HHI_list),
        np.asarray(utility_list),
        np.asarray(age_list)
    )

##################################################################################

def generate_sensitivity_output_ev(params: dict):
    """
    Generates sensitivity output from input parameters.
    Assumes `generate_data` is a defined function that returns
    an object with the required attributes.
    """
    data = generate_data(params)
    return data.calc_EV_prop()

def parallel_run_sa_ev(params_dict: list[dict]):
    """
    Runs the sensitivity analysis in parallel using the given parameter dictionary.
    """
    num_cores = get_num_workers()
    #ev_prop_list= [generate_sensitivity_output_ev(i) for i in params_dict]
    ev_prop_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_sensitivity_output_ev)(params) for params in params_dict
    )
    
    # Return results as arrays where applicable
    return np.asarray(ev_prop_list)



#########################################################################################

# Keys stacked into a (n_seeds, n_steps, ...) array across seeds. Everything else
# stays a per-seed list, because the per-seed entries are ragged.
MULTI_SEED_ARRAY_KEYS = (
    "history_driving_emissions",
    "history_production_emissions",
    "history_total_emissions",
    "history_prop_EV",
    "history_lower_percentile_price_ICE_EV",
    "history_upper_percentile_price_ICE_EV",
    "history_mean_price_ICE_EV",
    "history_median_price_ICE_EV",
    "history_total_utility",
    "history_market_concentration",
    "history_total_profit",
    "history_mean_car_age",
    "history_past_new_bought_vehicles_prop_ev",
    # Calibration-target diagnostics — see Social_Network.record_calibration_targets.
    "history_mean_car_age_fleet",
    "history_new_car_price_quantiles",
    "history_used_car_price_quantiles",
    "history_used_stock_quality_spread",
    "history_purchase_counts",
)


def generate_multi_seed(params: dict):
    """
    Run one seed and return every output the multi-seed plots need, keyed by name.

    `cars_on_sale` is a SNAPSHOT of the final time step (the car objects still on
    sale when the run ends), not a time series.
    """
    data = generate_data(params)
    social_network = data.social_network
    firm_manager = data.firm_manager
    return {
        "history_driving_emissions": social_network.history_driving_emissions,#Emmissions flow
        "history_production_emissions": social_network.history_production_emissions,#Emmissions flow
        "history_total_emissions": social_network.history_total_emissions,#Emmissions flow
        "history_prop_EV": social_network.history_prop_EV,
        "history_car_age": social_network.history_car_age,
        "history_lower_percentile_price_ICE_EV": social_network.history_lower_percentile_price_ICE_EV,
        "history_upper_percentile_price_ICE_EV": social_network.history_upper_percentile_price_ICE_EV,
        "history_mean_price_ICE_EV": social_network.history_mean_price_ICE_EV,
        "history_median_price_ICE_EV": social_network.history_median_price_ICE_EV,
        "history_total_utility": social_network.history_total_utility,
        "history_market_concentration": firm_manager.history_market_concentration,
        "history_total_profit": firm_manager.history_total_profit,
        "history_quality_ICE": social_network.history_quality_ICE,
        "history_quality_EV": social_network.history_quality_EV,
        "history_efficiency_ICE": social_network.history_efficiency_ICE,
        "history_efficiency_EV": social_network.history_efficiency_EV,
        "history_production_cost_ICE": social_network.history_production_cost_ICE,
        "history_production_cost_EV": social_network.history_production_cost_EV,
        "history_mean_profit_margins_ICE": firm_manager.history_mean_profit_margins_ICE,
        "history_mean_profit_margins_EV": firm_manager.history_mean_profit_margins_EV,
        "history_mean_car_age": social_network.history_mean_car_age,
        "history_past_new_bought_vehicles_prop_ev": firm_manager.history_past_new_bought_vehicles_prop_ev,
        "cars_on_sale": firm_manager.cars_on_sale_all_firms,
        # Calibration-target diagnostics — see Social_Network.record_calibration_targets.
        # history_mean_car_age_fleet is the WHOLE-FLEET mean age, which is what the
        # 10-12 year target refers to; history_mean_car_age above averages only the
        # vehicles chosen this step.
        "history_mean_car_age_fleet": social_network.history_mean_car_age_fleet,
        "history_new_car_price_quantiles": social_network.history_new_car_price_quantiles,
        "history_used_car_price_quantiles": social_network.history_used_car_price_quantiles,
        "history_used_stock_quality_spread": social_network.history_used_stock_quality_spread,
        "history_purchase_counts": social_network.history_purchase_counts,
    }

def parallel_run_multi_seed(params_list):
    """Run every seed and collect the outputs into one dict of seed-major data."""
    num_cores = get_num_workers()
    #res = [generate_multi_seed(i) for i in params_list]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_multi_seed)(params) for params in params_list
    )

    outputs = {key: [run[key] for run in res] for key in res[0]}
    for key in MULTI_SEED_ARRAY_KEYS:
        outputs[key] = np.asarray(outputs[key])

    return outputs