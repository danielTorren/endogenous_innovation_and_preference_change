"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports

import matplotlib.animation as animation 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from package.model.firmManager import Firm_Manager
from package.resources.utility import ( 
    load_object,
)
import pandas as pd
import matplotlib.lines as mlines
##########################################################################################
#SOCIAL NETWORK PLOTS

def plot_time_series(data, title, xlabel, ylabel):
    """
    Plot a generic time series data.
    Parameters:
        data (list): Time series data to be plotted.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data, marker='o')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    

def plot_emissions(social_network):
    """
    Plot driving and production emissions over time.
    Parameters:
        social_network (Social_Network): Instance of Social_Network with simulation data.
    """
    driving_emissions = social_network.history_driving_emissions
    production_emissions = social_network.history_production_emissions

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(driving_emissions, label='Driving Emissions', marker='o')
    ax.plot(production_emissions, label='Production Emissions', marker='s')
    ax.set_title("Emissions Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Emissions")
    ax.legend()
    ax.grid()

def plot_total_utility(social_network):
    """
    Plot total utility over time.
    Parameters:
        social_network (Social_Network): Instance of Social_Network with simulation data.
    """
    total_utility = social_network.history_total_utility  # Access the recorded total utility over time

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(total_utility, marker='o')
    ax.set_title("Total Utility Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Utility")
    ax.grid()

def plot_total_distance(social_network):
    """
    Plot total distance traveled over time.
    Parameters:
        social_network (Social_Network): Instance of Social_Network with simulation data.
    """
    total_distance = social_network.history_total_distance_driven  # Access the recorded total distance over time

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(total_distance, marker='o')
    ax.set_title("Total Distance Traveled Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Distance Traveled")
    ax.grid()

def plot_ev_adoption_rate(social_network):
    """
    Plot the rate of EV adoption over time.
    Parameters:
        social_network (Social_Network): Instance of Social_Network with simulation data.
    """
    # Calculate the EV adoption rate for each time step by averaging `ev_adoption_vec` values
    ev_adoption_rate =social_network.history_ev_adoption_rate

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ev_adoption_rate, marker='o')
    ax.set_title("EV Adoption Rate Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("EV Adoption Rate")
    ax.grid()

def plot_cumulative_emissions(social_network):
    """
    Plot cumulative driving and production emissions over time.
    Parameters:
        social_network (Social_Network): Instance of Social_Network with simulation data.
    """
    cumulative_driving_emissions = np.cumsum(social_network.history_driving_emissions)
    cumulative_production_emissions = np.cumsum(social_network.history_production_emissions)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_driving_emissions, label="Cumulative Driving Emissions", marker='o')
    ax.plot(cumulative_production_emissions, label="Cumulative Production Emissions", marker='s')
    ax.set_title("Cumulative Emissions Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Emissions")
    ax.legend()
    ax.grid()
    
#############################################################################################################################
#FIRM MANAGER PLOTS

def plot_total_profit(firm_manager):
    """
    Plot total profit over time.
    Parameters:
        firm_manager (Firm_Manager): Instance of Firm_Manager with simulation data.
    """
    total_profit = firm_manager.history_total_profit

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(total_profit, marker='o')
    ax.set_title("Total Profit Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Profit")
    ax.grid()
    

def plot_market_concentration(firm_manager):
    """
    Plot market concentration (HHI) over time.
    Parameters:
        firm_manager (Firm_Manager): Instance of Firm_Manager with simulation data.
    """
    market_concentration = firm_manager.history_market_concentration

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(market_concentration, marker='s')
    ax.set_title("Market Concentration (HHI) Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Market Concentration (HHI)")
    ax.grid()

###################################################################################################################
#vehicle_user_plots

def plot_all_users_emissions(vehicle_users):
    """
    Plot driving and production emissions over time for each VehicleUser on the same plot.
    Parameters:
        vehicle_users (list): List of VehicleUser instances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for user in vehicle_users:
        ax.plot(user.history_emissions_driving, alpha=0.6)
        ax.plot(user.history_emissions_production, alpha=0.6)

    ax.set_title("Driving and Production Emissions Over Time (All Users)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Emissions")
    ax.grid()
    

def plot_all_users_utility(vehicle_users):
    """
    Plot utility over time for each VehicleUser on the same plot.
    Parameters:
        vehicle_users (list): List of VehicleUser instances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for user in vehicle_users:
        ax.plot(user.history_utility, alpha=0.6)

    ax.set_title("Utility Over Time (All Users)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Utility")
    ax.grid()
    

def plot_all_users_distance(vehicle_users):
    """
    Plot distance traveled over time for each VehicleUser on the same plot.
    Parameters:
        vehicle_users (list): List of VehicleUser instances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for user in vehicle_users:
        ax.plot(user.history_distance_driven, alpha=0.6)

    ax.set_title("Distance Traveled Over Time (All Users)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Distance Traveled")
    ax.grid()

#############################################################################################
# firm plots

def plot_all_firms_profit(firms_list):
    """
    Plot profit over time for each firm on the same plot.
    Parameters:
        firms_list (list): List of Firm instances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for firm in firms_list:
        ax.plot(firm.history_profit, alpha=0.6)
    
    ax.set_title("Firm Profit Over Time (All Firms)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Profit")
    ax.grid()
    

def plot_all_firms_vehicles_chosen(firms_list):
    """
    Plot the number of vehicles chosen by users for each firm over time on the same plot.
    Parameters:
        firms_list (list): List of Firm instances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for firm in firms_list:
        ax.plot(firm.history_firm_cars_users, alpha=0.6)
    
    ax.set_title("Number of Vehicles Chosen by Users Over Time (All Firms)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Vehicles Chosen by Users")
    ax.grid()
    
    
def main(
    fileName = "results/single_experiment_15_05_51__26_02_2024",
    dpi_save = 600,
    social_plots = 1,
    vehicle_user_plots = 1,
    firm_manager_plots = 1,
    firm_plots = 1,
    ) -> None: 



    base_params = load_object(fileName + "/Data", "base_params")
    data_controller= load_object(fileName + "/Data", "controller")
    social_network = data_controller.social_network
    firm_manager = data_controller.firm_manager

    if social_plots:
        ###SOCIAL NETWORK PLOTS
        # Assuming `social_network` is an instance of Social_Network with completed simulation data

        # Plot driving and production emissions over time
        plot_emissions(social_network)

        # Plot total utility over time
        plot_total_utility(social_network)

        # Plot total distance traveled over time
        plot_total_distance(social_network)

        # Plot EV adoption rate over time
        plot_ev_adoption_rate(social_network)

        # Plot cumulative emissions over time
        plot_cumulative_emissions(social_network)

    if vehicle_user_plots:
        # Assuming `social_network` is an instance of Social_Network with VehicleUser instances

        # Plot driving and production emissions for all users over time
        plot_all_users_emissions(social_network.vehicleUsers_list)

        # Plot utility over time for all users
        plot_all_users_utility(social_network.vehicleUsers_list)

        # Plot distance traveled over time for all users
        plot_all_users_distance(social_network.vehicleUsers_list)


    if firm_manager_plots:
        ##FIRM PLOTS
        # Assuming `firm_manager` is an instance of Firm_Manager after running the simulation

        # Plot total profit over time
        plot_total_profit(firm_manager)

        # Plot market concentration (HHI) over time
        plot_market_concentration(firm_manager)

    if firm_plots:
        # Assuming `firm_manager` is an instance of Firm_Manager with a list of Firm instances in firm_manager.firms_list
        # Plot profit over time for all firms
        plot_all_firms_profit(firm_manager.firms_list)

        # Plot the number of vehicles chosen by users for each firm over time
        plot_all_firms_vehicles_chosen(firm_manager.firms_list)

    plt.show()


if __name__ == "__main__":
    plots = main(
        fileName = "results/single_experiment_12_44_02__17_07_2024",
    )


