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
from package.resources.plot import (
    plot_low_carbon_preferences_timeseries,
    plot_total_flow_carbon_emissions_timeseries,
    plot_network_timeseries
)
import matplotlib.lines as mlines
from scipy.stats import sem

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

def list_weighted_owned_average_plots_no_public(fileName, data_social_network_list):
    # Initialize data structures to hold weighted averages over time for all networks
    all_weighted_averages = {
        'environmental_score': [],
        'cost': [],
        'quality': []
    }

    for data_social_network in data_social_network_list:
        time_series_cars = data_social_network.history_car_owned_vec

        # Initialize data structures to hold weighted averages for the current network
        weighted_averages = {
            'environmental_score': [],
            'cost': [],
            'quality': []
        }

        for snapshot in time_series_cars:
            # Filter out None values
            valid_cars = [car for car in snapshot if car is not None]
            
            if valid_cars:
                attributes_matrix = np.asarray([car.attributes_fitness for car in valid_cars])
                weighted_averages['environmental_score'].append(np.mean(attributes_matrix[:, 1]))
                weighted_averages['cost'].append(np.mean(attributes_matrix[:, 0]))
                weighted_averages['quality'].append(np.mean(attributes_matrix[:, 2]))
            else:
                # If no valid cars in snapshot, append NaN to maintain the time series length
                weighted_averages['environmental_score'].append(np.nan)
                weighted_averages['cost'].append(np.nan)
                weighted_averages['quality'].append(np.nan)

        all_weighted_averages['environmental_score'].append(weighted_averages['environmental_score'])
        all_weighted_averages['cost'].append(weighted_averages['cost'])
        all_weighted_averages['quality'].append(weighted_averages['quality'])

    # Convert lists to DataFrames for easier manipulation
    df_environmental_score = pd.DataFrame(all_weighted_averages['environmental_score']).transpose()
    df_cost = pd.DataFrame(all_weighted_averages['cost']).transpose()
    df_quality = pd.DataFrame(all_weighted_averages['quality']).transpose()

    # Function to calculate mean and 95% confidence interval
    def calculate_mean_and_ci(df):
        mean_vals = df.mean(axis=1)
        sem_vals = df.apply(sem, axis=1, nan_policy='omit')
        ci_upper = mean_vals + 1.96 * sem_vals
        ci_lower = mean_vals - 1.96 * sem_vals
        return mean_vals, ci_upper, ci_lower

    # Calculate mean and CI for each metric
    mean_env, ci_upper_env, ci_lower_env = calculate_mean_and_ci(df_environmental_score)
    mean_cost, ci_upper_cost, ci_lower_cost = calculate_mean_and_ci(df_cost)
    mean_quality, ci_upper_quality, ci_lower_quality = calculate_mean_and_ci(df_quality)

    # Plot the data using subfigures
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    subfig1 = axes[0]
    subfig2 = axes[1]
    subfig3 = axes[2]

    burn_in(subfig1, data_social_network_list[0])
    burn_in(subfig2, data_social_network_list[0])
    burn_in(subfig3, data_social_network_list[0])

    # Subfigure for environmental score
    subfig1.plot(mean_env, color='blue', linestyle='-', linewidth=2, label='Mean Environmental Score')
    subfig1.fill_between(mean_env.index, ci_lower_env, ci_upper_env, color='blue', alpha=0.2, label='95% CI')
    subfig1.set_xlabel('Time')
    subfig1.set_ylabel('Weighted Average Environmental Score- Owned')
    #subfig1.set_title('Emissions')
    subfig1.grid(True)
    subfig1.legend()

    # Subfigure for cost
    subfig2.plot(mean_cost, color='green', linestyle='-', linewidth=2, label='Mean Cost')
    subfig2.fill_between(mean_cost.index, ci_lower_cost, ci_upper_cost, color='green', alpha=0.2, label='95% CI')
    subfig2.set_xlabel('Time')
    subfig2.set_ylabel('Weighted Average Cost - Owned')
    #subfig2.set_title('Cost')
    subfig2.grid(True)
    subfig2.legend()

    # Subfigure for quality
    subfig3.plot(mean_quality, color='red', linestyle='-', linewidth=2, label='Mean Quality')
    subfig3.fill_between(mean_quality.index, ci_lower_quality, ci_upper_quality, color='red', alpha=0.2, label='95% CI')
    subfig3.set_xlabel('Time')
    subfig3.set_ylabel('Weighted Average Quality - Owned')
    #subfig3.set_title('Quality')
    subfig3.grid(True)
    subfig3.legend()

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/owned_weighted_vals_NO_PUBLIC"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_em_flow(fileName, data_social_network_list):
    # Initialize a list to store the emission flow for all networks
    all_emission_flows = []

    for data_social_network in data_social_network_list:
        time_series = data_social_network.history_flow_carbon_emissions
        all_emission_flows.append(time_series)

    # Convert the list of emission flow lists to a DataFrame for easier manipulation
    emission_df = pd.DataFrame(all_emission_flows).transpose()

    # Calculate mean and standard error of the mean (SEM) for each time point
    mean_emission = emission_df.mean(axis=1)
    sem_emission = emission_df.apply(sem, axis=1, nan_policy='omit')

    # Calculate the 95% confidence interval
    ci_upper = mean_emission + 1.96 * sem_emission
    ci_lower = mean_emission - 1.96 * sem_emission

    # Plot the mean and 95% confidence interval
    fig, ax = plt.subplots(figsize=(10, 6))

    burn_in(ax, data_social_network_list[0])

    ax.plot(mean_emission.index, mean_emission, color='blue', linestyle='-', linewidth=2, label='Mean Emission Flow')
    ax.fill_between(mean_emission.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')

    # Labeling the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Emission flow, $E_F$')
    ax.legend()
    ax.grid(True)

    # Show the plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/em_flow"
    fig.savefig(f + ".png", dpi=600, format="png")

def list_scatter_trace_plots_offered(fileName, data_firm_manager_list, x_param, y_param):
    # Initialize lists to hold values over time for all firm managers
    all_time_points = []
    all_x_values = []
    all_y_values = []
    all_firm_ids = []

    param_map = {'cost': 0, 'environmental_score': 1, 'quality': 2}
    x_param_index = param_map[x_param]
    y_param_index = param_map[y_param]

    if x_param == "cost":
        x_param_title = "Cost"
    elif x_param == "environmental_score":
        x_param_title = "Environmental Score"
    elif x_param == "quality": 
        x_param_title = "Quality"

    if y_param == "cost":
        y_param_title = "Cost"
    elif y_param == "environmental_score":
        y_param_title = "Environmental Score"
    elif y_param == "quality": 
        y_param_title = "Quality"

    for data_firm_manager in data_firm_manager_list:
        time_series_cars = data_firm_manager.history_cars_on_sale_all_firms

        for t, snapshot in enumerate(time_series_cars):
            for car in snapshot:
                attributes = car.attributes_fitness
                all_time_points.append(t)
                all_x_values.append(attributes[x_param_index])
                all_y_values.append(attributes[y_param_index])
                all_firm_ids.append(car.firm_id)

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'time': all_time_points,
        'x': all_x_values,
        'y': all_y_values,
        'firm_id': all_firm_ids
    })

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(df['x'], df['y'], c=df['time'], cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)

    ax.set_xlabel(x_param_title)
    ax.set_ylabel(y_param_title)
    ax.set_title(f'{x_param_title} vs {y_param_title} Over Time - Firms')
    ax.grid(True)

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time')

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + f"/scatter_{x_param}_vs_{y_param}_offered"
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_mean_preference(fileName, data_social_network_list):

    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_identity_series_list = []
    for sn in data_social_network_list:
        # Calculate mean of the identity time series for each individual
        mean_identity_series = np.mean(sn.history_preference_list, axis=1)
        mean_identity_series_list.append(mean_identity_series)
    
    mean_identity_series_arr = np.asarray(mean_identity_series_list)
    
    # Calculate mean and standard error of the mean (SEM) for each time point
    mean_emission = mean_identity_series_arr.mean(axis=0)
    sem_emission = sem(mean_identity_series_arr, axis=0, nan_policy='omit')

    # Calculate the 95% confidence interval
    ci_upper = mean_emission + 1.96 * sem_emission
    ci_lower = mean_emission - 1.96 * sem_emission

    time_points = data_social_network_list[0].history_time_social_network
    
    ax.plot(time_points, mean_emission, color='blue', linestyle='-', linewidth=2, label='Mean Environmental Preference')
    ax.fill_between(time_points, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Environmental preference")
    #ax.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/timeseries_preference_MEAN"
    fig.savefig(f + ".png", dpi=600, format="png")
    plt.show()

def main(
    fileName = "results/single_experiment_15_05_51__26_02_2024",
    ) -> None: 

    data_controller_list = load_object(fileName + "/Data", "data_list")
    data_social_network_list = [data_controller.social_network for data_controller in data_controller_list]
    data_firm_manager_list = [data_controller.firm_manager for data_controller in data_controller_list]
    
    #plot_firm_count_and_market_concentration(fileName, data_social_network_list)
    plot_mean_market_concentration(fileName, data_social_network_list)
    list_weighted_owned_average_plots_no_public(fileName, data_social_network_list)
    plot_em_flow(fileName, data_social_network_list)
    #list_scatter_trace_plots_offered(fileName, data_firm_manager_list, "cost", "environmental_score")
    plot_mean_preference(fileName, data_social_network_list)
    plt.show()

if __name__ == "__main__":
    plots = main(
        fileName = "results/stochastic_runs_15_11_15__16_07_2024",
    )


