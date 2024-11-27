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
#from package.model_collated.controller import Controller
#from package.model.combinedController import CombinedController
import pandas as pd


# modules
####################################################################################################################################################
#LOAD IN THE DATA TIME SERIES
#THE IDEA HERE IS THAT I LOAD IN ALL THE TIME SERIES DATA FOR CALIFORNIA FROM 2000 TO 2024 WHere i have it
def load_in_calibration_data():

    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE)
    gasoline_gco2_per_gallon = 8887 #grams CO2/ gallon

    #CPI
    CPI_california_df = pd.read_excel("package/calibration_data/CPI_california.xlsx") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    CPI_california_df["Date"] = pd.to_datetime(CPI_california_df["Date"])
    CPI_california_df["Weighted Average"] = CPI_california_df["Weighted Average"].interpolate(method='linear')
    # Handle the first NaN explicitly by assigning the second value
    if pd.isna(CPI_california_df.loc[0, "Weighted Average"]):
        CPI_california_df.loc[0, "Weighted Average"] = CPI_california_df.loc[1, "Weighted Average"]
    # Set "Date" as the index
    CPI_california_df.set_index('Date', inplace=True)
    # Get the value for "Weighted Average" on 2020-01-01
    reference_value = CPI_california_df.loc["2020-01-01", "Weighted Average"]
    # Normalize by the 2020 reference value
    CPI_california_df["2020 relative Weighted Average"] = CPI_california_df["Weighted Average"] / reference_value

    #Gasoline Price
    gas_price_california_df = pd.read_excel("package/calibration_data/gas_price_california.xlsx") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    gas_price_california_df["Date"] = pd.to_datetime(gas_price_california_df["Date"])
    gas_price_california_df.set_index('Date', inplace=True)
    gas_price_california_df["Real Dollars per Gallon"] = gas_price_california_df["Dollars per Gallon"]/CPI_california_df["2020 relative Weighted Average"]
    gas_price_california_df["Real Dollars per Kilowatt-Hour"] = gas_price_california_df["Real Dollars per Gallon"]/gasoline_Kilowatt_Hour_per_gallon

    #Electricity Price
    electricity_price_df = pd.read_excel("package/calibration_data/electricity_price.xlsx") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    electricity_price_df["Date"] = pd.to_datetime( electricity_price_df["Date"])
    electricity_price_df.set_index('Date', inplace=True)
    electricity_price_df["Dollars per Kilowatt-Hour (City Average)"] = electricity_price_df[["Dollars per Kilowatt-Hour (San Francisco)", "Dollars per Kilowatt-Hour (Los Angeles)"]].mean(axis=1)
    electricity_price_df["Real Dollars per Kilowatt-Hour (City Average)"] = electricity_price_df["Dollars per Kilowatt-Hour (City Average)"]/CPI_california_df["2020 relative Weighted Average"]

    #Electricty
    electricity_emissions_intensity_df = pd.read_csv("package/calibration_data/emissions_intensity_emberChartData.csv") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    #gco2_per_kwh
    electricity_emissions_intensity_df["Date"] = pd.to_datetime( electricity_emissions_intensity_df["Date"])
    electricity_emissions_intensity_df.set_index('Date', inplace=True)
    electricity_emissions_intensity_df["KgCO2 per Kilowatt-Hour"] = electricity_emissions_intensity_df["emissions_intensity_gco2_per_kwh"]/1000
    #NOW - NEED TO GET ALL THE PRICES INTO 2020 DOLLARS BY DIVIDING BY THE CPI 

    #Emissions Gasoline - WE 
    gasoline_Kgco2_per_Kilowatt_Hour =  (gasoline_gco2_per_gallon/gasoline_Kilowatt_Hour_per_gallon)/1000

    # Align Data
    aligned_data = CPI_california_df.join([
        gas_price_california_df["Real Dollars per Kilowatt-Hour"],
        electricity_price_df["Real Dollars per Kilowatt-Hour (City Average)"],
        electricity_emissions_intensity_df["KgCO2 per Kilowatt-Hour"]
    ], how='inner')  # Only keep rows with data in all columns

    return aligned_data, gasoline_Kgco2_per_Kilowatt_Hour

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

    calibration_data, gasoline_Kgco2_per_Kilowatt_Hour = load_in_calibration_data()
    parameters[]["time_steps_max"]


    if print_simu:
        start_time = time.time()
    parameters["time_steps_max"] = parameters["duration_no_carbon_price"] + parameters["duration_small_carbon_price"] + parameters["duration_large_carbon_price"]

    #print("tim step max", parameters["time_steps_max"],parameters["burn_in_duration"], parameters["carbon_price_duration"])
    controller = Controller(parameters)
    #controller = CombinedController(parameters)
    

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
    return data.social_network.total_carbon_emissions_cumulative

def emissions_intensities_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_intensities(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_emissions_intensities)(i) for i in params_dict)

    return np.asarray(emissions_list)
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