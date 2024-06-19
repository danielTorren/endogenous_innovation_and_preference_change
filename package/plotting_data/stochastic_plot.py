"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports

import matplotlib.animation as animation 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from package.model.firm_manager import Firm_Manager
from package.resources.utility import ( 
    load_object,
)
import pandas as pd
from package.resources.plot import (
    plot_low_carbon_preferences_timeseries,
    plot_total_flow_carbon_emissions_timeseries,
    plot_network_timeseries
)
import matplotlib.lines as mlines

def burn_in(ax,data):
    if data.duration_no_OD_no_stock_no_policy > 0:
        ax.axvline(x = data.duration_no_OD_no_stock_no_policy,ls="--",  color="black")#OD
    
    if data.duration_OD_no_stock_no_policy > 0:
        ax.axvline(x = (data.duration_no_OD_no_stock_no_policy+data.duration_OD_no_stock_no_policy),ls="--",  color="black")#OD


def plot_firm_count_and_market_concentration(fileName, data_social_network_list):
    # Initialize a list to store the data for all networks
    all_data = []
    all_market_concentration = []

    # Define a list of colors for plotting
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Ensure there are enough colors
    #if len(data_social_network_list) > len(colors):
    #    colors = colors * ((len(data_social_network_list) // len(colors)) + 1)

    for idx, data_social_network in enumerate(data_social_network_list):
        # Initialize a dictionary to store the data for the current network
        data = {}

        # Extract data
        for time_point, snapshot in enumerate(data_social_network.history_firm_count):
            for firm, cars in snapshot.items():
                if firm not in data:
                    data[firm] = []
                total_cars_sold = sum(cars.values())
                data[firm].append((time_point, total_cars_sold))

        # Convert to a DataFrame
        df = pd.DataFrame()
        for firm, sales in data.items():
            times, car_sales = zip(*sales)
            df[firm] = pd.Series(car_sales, index=times)

        # Fill missing values with 0 (if any firm didn't sell cars at some time points)
        df = df.fillna(0)

        # Calculate market concentration (Herfindahl-Hirschman Index)
        market_concentration = []
        for time_point in df.index:
            total_cars = df.loc[time_point].sum()
            if total_cars > 0:
                market_shares = df.loc[time_point] / total_cars
                concentration = (market_shares ** 2).sum()
            else:
                concentration = np.nan
            market_concentration.append(concentration)

        all_data.append(df)
        all_market_concentration.append((df.index, market_concentration))

    # Plot the number of cars sold by each firm over time
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for idx, (df, market_concentration) in enumerate(zip(all_data, all_market_concentration)):
        #color = colors[idx]
        ax1.plot(df.index, market_concentration[1], linestyle='-', linewidth=2)

    ax1.set_ylabel('Market Concentration (HHI)')
    #ax1.legend(loc='upper right')

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/list_firm_count_and_market_concentration"
    fig.savefig(f + ".png", dpi=600, format="png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def plot_mean_market_concentration(fileName, data_social_network_list):
    # Initialize a list to store the market concentration for all networks
    all_market_concentration = []

    for data_social_network in data_social_network_list:
        # Initialize a dictionary to store the data for the current network
        data = {}

        # Extract data
        for time_point, snapshot in enumerate(data_social_network.history_firm_count):
            for firm, cars in snapshot.items():
                if firm not in data:
                    data[firm] = []
                total_cars_sold = sum(cars.values())
                data[firm].append((time_point, total_cars_sold))

        # Convert to a DataFrame
        df = pd.DataFrame()
        for firm, sales in data.items():
            times, car_sales = zip(*sales)
            df[firm] = pd.Series(car_sales, index=times)

        # Fill missing values with 0 (if any firm didn't sell cars at some time points)
        df = df.fillna(0)

        # Calculate market concentration (Herfindahl-Hirschman Index)
        market_concentration = []
        for time_point in df.index:
            total_cars = df.loc[time_point].sum()
            if total_cars > 0:
                market_shares = df.loc[time_point] / total_cars
                concentration = (market_shares ** 2).sum()
            else:
                concentration = np.nan
            market_concentration.append(concentration)

        all_market_concentration.append(market_concentration)

    # Convert the list of market concentration lists to a DataFrame for easier manipulation
    concentration_df = pd.DataFrame(all_market_concentration).transpose()

    # Calculate mean and standard error of the mean (SEM) for each time point
    mean_concentration = concentration_df.mean(axis=1)
    sem_concentration = concentration_df.apply(sem, axis=1, nan_policy='omit')

    # Calculate the 95% confidence interval
    ci_upper = mean_concentration + 1.96 * sem_concentration
    ci_lower = mean_concentration - 1.96 * sem_concentration

    # Plot the mean and 95% confidence interval
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(mean_concentration, color='blue', linestyle='-', linewidth=2, label='Mean Market Concentration')
    ax1.fill_between(mean_concentration.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')

    ax1.set_ylabel('Market Concentration (HHI)')
    ax1.legend(loc='upper right')

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/mean_and_ci_firm_count_and_market_concentration"
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName = "results/single_experiment_15_05_51__26_02_2024",
    ) -> None: 

    data_controller_list = load_object(fileName + "/Data", "data_list")
    data_social_network_list = [data_controller.social_network for data_controller in data_controller_list]

    plot_firm_count_and_market_concentration(fileName, data_social_network_list)
    plot_mean_market_concentration(fileName, data_social_network_list)
    plt.show()

if __name__ == "__main__":
    plots = main(
        fileName = "results/stochastic_runs_16_52_02__19_06_2024",
    )


