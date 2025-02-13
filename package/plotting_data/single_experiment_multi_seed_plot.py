from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.plotting_data.single_experiment_plot import save_and_show, format_plot

def add_vertical_lines(ax, base_params, color='black', linestyle='--'):
    """
    Adds dashed vertical lines to the plot at specified steps, adjusted to exclude the burn-in period.

    Parameters:
    ax : matplotlib.axes.Axes
        The Axes object to add the lines to.
    base_params : dict
        Dictionary containing relevant parameters for placing vertical lines.
    color : str, optional
        Color of the dashed lines. Default is 'black'.
    linestyle : str, optional
        Style of the dashed lines. Default is '--'.
    """
    burn_in = base_params["duration_burn_in"]
    no_carbon_price = base_params["duration_no_carbon_price"]
    #ev_research_start_time = base_params["ev_research_start_time"]
    ev_production_start_time = base_params["ev_production_start_time"]
    #second_hand_burn_in = base_params["parameters_second_hand"]["burn_in_second_hand_market"]

    # Adding the vertical lines after the burn-in period
    #ax.axvline(burn_in + second_hand_burn_in, color=color, linestyle='-.', label="Second-hand market start")
    ax.axvline(burn_in + ev_production_start_time, color="red", linestyle=':', label="EV sale start")

    if base_params["EV_rebate_state"]:
        rebate_start_time = burn_in + base_params["parameters_rebate_calibration"]["start_time"]
        ax.axvline(rebate_start_time, color="red", linestyle='-.', label="EV adoption subsidy start")
    
    if base_params["duration_future"] > 0:
        policy_start_time = burn_in + no_carbon_price
        ax.axvline(policy_start_time, color="red", linestyle='--', label="Policy start")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

def plot_data_across_seeds(base_params, fileName, data, title, x_label, y_label, save_name, dpi=600): 
    """
    Plot data across multiple seeds with mean, confidence intervals, and individual traces, 
    starting from the end of the burn-in period.

    Args:
        base_params: Dictionary containing simulation parameters (e.g., duration_burn_in).
        fileName: Name of the file for saving the plot.
        data: 2D array where rows are the time series for each seed (shape: [num_seeds, num_time_steps]).
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        save_name: Name of the file to save the plot.
        dpi: Dots per inch for saving the plot.
    """
    # Determine the start of the data after the burn-in period
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = data[:, burn_in_step:]
    time_steps = np.arange(burn_in_step, data.shape[1])

    # Calculate mean and 95% confidence interval
    mean_values = np.mean(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)  # 95% CI

    # Create subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

    # Plot individual traces (faded lines)
    for seed_data in data_after_burn_in:
        ax1.plot(time_steps, seed_data, color='gray', alpha=0.3, linewidth=0.8)

    # Plot mean and 95% CI
    ax1.plot(time_steps, mean_values, label='Mean', color='blue')
    ax1.fill_between(
        time_steps, 
        mean_values - ci_range, 
        mean_values + ci_range, 
        color='blue', 
        alpha=0.3, 
        label='95% Confidence Interval'
    )

    # Add labels, vertical lines, and legend
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    add_vertical_lines(ax1, base_params)
    ax1.legend()

    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, save_name, dpi)


def plot_calibrated_index_emissions(real_data, base_params, fileName, data, title, x_label, y_label, save_name, dpi=600): 

    # Adjust the real data's time steps
    init_index = base_params["duration_burn_in"]
    time_steps_real = np.arange(init_index, init_index + len(real_data) * 12, 12)

    # Determine the start of the data after the burn-in period
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = data[:, burn_in_step:]
    time_steps = np.arange(burn_in_step, data.shape[1])

    # Calculate mean and 95% confidence interval
    mean_values = np.mean(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)  # 95% CI

    # Create subplots
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # Plot mean and 95% CI
    ax1.plot(time_steps, mean_values, label='Mean', color='blue')
    ax1.fill_between(
        time_steps, 
        mean_values - ci_range, 
        mean_values + ci_range, 
        color='blue', 
        alpha=0.3, 
        label='95% Confidence Interval'
    )

    # Add labels, vertical lines, and legend
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    add_vertical_lines(ax1, base_params)
    ax1.legend()

    ax2.plot(time_steps, mean_values/max(mean_values), label='Simulated Emissions index Mean', color='blue')
    ax2.plot(time_steps_real, real_data,label='California Emissions index', color='Orange', linestyle="dashed")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Normalised Emissions index")

    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, save_name, dpi)

def plot_ev_uptake_dual(real_data, base_params, fileName, data, title, x_label, y_label, save_name, dpi=600):
    """
    Plot data across multiple seeds with mean, confidence intervals, and individual traces,
    starting from the end of the burn-in period.

    Args:
        real_data: Real-world data to compare against.
        base_params: Dictionary containing simulation parameters (e.g., duration_burn_in).
        fileName: Name of the file for saving the plot.
        data: 2D array where rows are the time series for each seed (shape: [num_seeds, num_time_steps]).
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        save_name: Name of the file to save the plot.
        dpi: Dots per inch for saving the plot.
    """
    # Determine the start of the data after the burn-in period
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = data[:, burn_in_step:]
    time_steps = np.arange(burn_in_step, data.shape[1])

    # Adjust the real data's time steps
    init_index = base_params["duration_burn_in"] + 120
    time_steps_real = np.arange(init_index, init_index + len(real_data) * 12, 12)

    # Calculate mean and 95% confidence interval for data after burn-in
    mean_values = np.mean(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)  # 95% CI

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # First subplot: Mean and 95% CI with real data
    ax1.plot(time_steps_real, real_data, label="California Data", color='orange', linestyle="--")
    ax1.plot(time_steps, mean_values, label='Mean', color='blue')
    ax1.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='blue',
        alpha=0.3,
        label='95% Confidence Interval'
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    add_vertical_lines(ax1, base_params)
    ax1.legend()

    # Second subplot: Individual traces with real data
    ax2.plot(time_steps_real, real_data, label="California Data", color='orange', linestyle="--")
    for seed_data in data_after_burn_in:
        ax2.plot(time_steps, seed_data, alpha=0.7)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    add_vertical_lines(ax2, base_params)

    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, save_name, dpi)


def plot_vehicle_attribute_time_series_by_type_split(
    base_params, 
    history_quality_ICE, history_quality_EV, 
    history_efficiency_ICE, history_efficiency_EV, 
    history_production_cost_ICE, history_production_cost_EV, 
    fileName, dpi=600
):
    """
    Plots time series of Quality, Efficiency (separate for ICE and EV),
    and Production Cost for both ICE and EV with means and confidence intervals across multiple seeds,
    starting from the end of the burn-in period.

    Args:
        base_params: Parameters of the simulation for adding additional context (like vertical lines).
        history_quality_ICE, history_quality_EV: 3D arrays [num_seeds, num_time_steps, num_values_per_time_step].
        history_efficiency_ICE, history_efficiency_EV: 3D arrays [num_seeds, num_time_steps, num_values_per_time_step].
        history_production_cost_ICE, history_production_cost_EV: 3D arrays [num_seeds, num_time_steps, num_values_per_time_step].
        fileName: Directory or file name to save the plots.
        dpi: Resolution for saving the plots.
    """
    # Combine data into attributes for easier iteration
    attributes = {
        "Quality": (history_quality_ICE, history_quality_EV),
        "Efficiency": (history_efficiency_ICE, history_efficiency_EV),
        "Production Cost": (history_production_cost_ICE, history_production_cost_EV),
    }

    # Get burn-in step
    burn_in_step = base_params["duration_burn_in"]

    # Create figure and gridspec
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])  # 3x2 layout

    # Collect handles and labels for a single legend
    handles, labels = [], []

    for idx, (attribute_name, (ice_data, ev_data)) in enumerate(attributes.items()):
        # Axes for ICE and EV
        ax_ice = fig.add_subplot(gs[idx, 0])
        ax_ev = fig.add_subplot(gs[idx, 1])
        if idx == 0: 
            ax_ice.set_title("ICE")
            ax_ev.set_title("EV")
        
        # Plot ICE
        plot_attribute_multiple_seeds(ax_ice, f"{attribute_name} (ICE)", ice_data, base_params)
        # Plot EV
        plot_attribute_multiple_seeds(ax_ev, f"{attribute_name} (EV)", ev_data, base_params)

        # Collect legend handles and labels
        if idx == 0:  # Only collect from the first subplot to avoid duplicates
            handles, labels = ax_ice.get_legend_handles_labels()

        # Set x-labels for the last row only
        if idx == len(attributes) - 1:
            ax_ice.set_xlabel("Time Step (Month)")
            ax_ev.set_xlabel("Time Step (Month)")

    # Add a single legend for the entire figure at the bottom
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), prop={'size': 8})

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Increase bottom margin

    # Save and show the plot
    save_and_show(fig, fileName, "vehicle_attribute_time_series_multiple_seeds", dpi)


def plot_attribute_multiple_seeds(ax, title, data, base_params):
    """
    Helper function to plot an attribute (e.g., Quality, Efficiency, Production Cost) across multiple seeds,
    starting from the end of the burn-in period.

    Args:
        ax: Matplotlib axis to plot on.
        title: Title for the plot.
        data: 3D array [num_seeds, num_time_steps, num_values_per_time_step].
        base_params: Simulation parameters for additional context (e.g., vertical lines).
    """
    # Add vertical lines for base parameters
    add_vertical_lines(ax, base_params)

    # Calculate seed means and confidence intervals
    seed_means = []
    for seed_idx, data_seed in enumerate(data):
        # Compute mean for each time step, ignoring NaN
        time_step_means = [np.nanmean(values) if len(values) > 0 else np.nan for values in data_seed]
        seed_means.append(time_step_means)

    seed_means = np.array(seed_means)  # Convert to numpy array
    overall_mean = np.nanmean(seed_means, axis=0)  # Mean across seeds for each time step
    ci = 1.96 * sem(seed_means, axis=0, nan_policy='omit')  # 95% confidence interval, ignoring NaN

    # Plot each seed's mean
    for seed_idx, seed_mean in enumerate(seed_means):
        ax.plot(seed_mean, alpha=0.5)  # Add label only once

    # Plot overall mean and confidence interval
    ax.plot(overall_mean, color="black", label="Overall Mean", linewidth=2)
    ax.fill_between(
        range(len(overall_mean)), 
        overall_mean - ci, 
        overall_mean + ci, 
        color="gray", alpha=0.3, label="95% Confidence Interval"
    )

    # Set title and labels
    ax.set_ylabel(title)



def plot_distance_individuals_mean_median_type_multiple_seeds(
    base_params, 
    history_distance_individual_EV, 
    history_distance_individual_ICE, 
    fileName, dpi=600
):
    """
    Plots mean and median individual distances for EV and ICE with 95% confidence intervals across multiple seeds,
    starting from the end of the burn-in period.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_distance_individual_EV: 2D array [num_seeds, num_time_steps].
        history_distance_individual_ICE: 2D array [num_seeds, num_time_steps].
        fileName: Directory or file name to save the plot.
        dpi: Resolution for saving the plot.
    """
    # Determine the burn-in step
    burn_in_step = base_params["duration_burn_in"]

    # Slice data to exclude the burn-in period
    history_distance_individual_EV = np.nanmean(np.asarray(history_distance_individual_EV), axis=2)[:, burn_in_step:]
    history_distance_individual_ICE = np.nanmean(np.asarray(history_distance_individual_ICE), axis=2)[:, burn_in_step:]

    # Compute mean and median across seeds for EV
    mean_ev = np.nanmean(history_distance_individual_EV, axis=0)
    median_ev = np.nanmedian(history_distance_individual_EV, axis=0)
    std_error_ev = sem(history_distance_individual_EV, axis=0)
    ci_ev = t.ppf(0.975, df=history_distance_individual_EV.shape[0] - 1) * std_error_ev

    # Compute mean and median across seeds for ICE
    mean_ice = np.nanmean(history_distance_individual_ICE, axis=0)
    median_ice = np.nanmedian(history_distance_individual_ICE, axis=0)
    std_error_ice = sem(history_distance_individual_ICE, axis=0)
    ci_ice = t.ppf(0.975, df=history_distance_individual_ICE.shape[0] - 1) * std_error_ice

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot EV data (green)
    ax.plot(mean_ev, color='green', linestyle='-', linewidth=2, label='EV Mean Distance')
    ax.plot(median_ev, color='green', linestyle='--', linewidth=2, label='EV Median Distance')

    # Plot ICE data (blue)
    ax.plot(mean_ice, color='blue', linestyle='-', linewidth=2, label='ICE Mean Distance')
    ax.plot(median_ice, color='blue', linestyle='--', linewidth=2, label='ICE Median Distance')

    # Add confidence intervals for EV and ICE
    ax.fill_between(
        np.arange(len(mean_ev)),  # Time steps (x-axis)
        mean_ev - ci_ev,
        mean_ev + ci_ev,
        color='green',
        alpha=0.2,
        label='EV 95% Confidence Interval'
    )
    ax.fill_between(
        np.arange(len(mean_ice)),  # Time steps (x-axis)
        mean_ice - ci_ice,
        mean_ice + ci_ice,
        color='blue',
        alpha=0.2,
        label='ICE 95% Confidence Interval'
    )

    # Add vertical lines for base parameters (e.g., milestones or events)
    add_vertical_lines(ax, base_params)

    # Format the plot
    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Distance, km")
    ax.legend()

    # Save and show the plot
    save_and_show(fig, fileName, "user_distance_mean_median_type_multiple_seeds", dpi)

def plot_history_car_age_multiple_seeds(
    base_params, history_car_age, fileName, dpi
):
    """
    Plots the mean and 95% confidence interval for car ages across multiple seeds,
    starting from the end of the burn-in period.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_car_age: 2D array [num_seeds, num_time_steps], where each row represents a seed's time series of car ages.
        fileName: Directory or file name to save the plot.
        dpi: Resolution for the saved plot.
    """
    # Determine the burn-in step
    burn_in_step = base_params["duration_burn_in"]

    # Slice the data to exclude the burn-in period
    history_car_age = np.mean(history_car_age, axis=2)[:, burn_in_step:]

    # Calculate statistics across seeds
    means = np.mean(history_car_age, axis=0)
    std_errors = sem(history_car_age, axis=0)
    ci = t.ppf(0.975, df=history_car_age.shape[0] - 1) * std_errors
    lower_bounds = means - ci
    upper_bounds = means + ci

    # Time steps after the burn-in period
    time_steps = np.arange(burn_in_step, burn_in_step + history_car_age.shape[1])

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, means, label="Mean Age, Months", color="blue")
    ax.fill_between(
        time_steps, lower_bounds, upper_bounds, color="blue", alpha=0.2, label="95% Confidence Interval"
    )

    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Age, months")

    # Add vertical lines for base parameters (e.g., milestones or events)
    add_vertical_lines(ax, base_params)

    # Add legend
    ax.legend()

    # Save and show the plot
    save_and_show(fig, fileName, "car_age_multiple_seeds", dpi)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

def plot_history_mean_price_multiple_seeds(
    base_params, 
    history_mean_price_ICE_EV, 
    history_median_price_ICE_EV, 
    fileName, 
    dpi=600
):
    """
    Plots the mean and 95% confidence interval for prices (new and second-hand cars) in the first subplot,
    starting from the end of the burn-in period.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_mean_price: 3D array [num_seeds, num_time_steps, 2] for mean prices.
        history_median_price: 3D array [num_seeds, num_time_steps, 2] for median prices.
        fileName: Directory or file name to save the plot.
        dpi: Resolution for the saved plot.
    """
    # Extract burn-in step
    burn_in_step = base_params["duration_burn_in"]

    # Extract new and second-hand prices, excluding burn-in period
    mean_new_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 0,0]  # Mean prices for new cars
    mean_second_hand_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 1,0]  # Mean prices for second-hand cars

    mean_new_EV = history_mean_price_ICE_EV[:, burn_in_step:, 0,1]  # Mean prices for new cars
    mean_second_hand_EV = history_mean_price_ICE_EV[:, burn_in_step:, 1, 1]  # Mean prices for second-hand cars

    # Time steps after burn-in
    time_steps = np.arange(burn_in_step, burn_in_step + mean_new_ICE.shape[1])

    # Compute mean and 95% CI across seeds
    #ICE
    overall_mean_new_ICE = np.mean(mean_new_ICE, axis=0)
    ci_new_ICE = t.ppf(0.975, df=mean_new_ICE.shape[0] - 1) * sem(mean_new_ICE, axis=0)

    overall_mean_second_hand_ICE = np.mean(mean_second_hand_ICE, axis=0)
    ci_second_hand_ICE = t.ppf(0.975, df=mean_second_hand_ICE.shape[0] - 1) * sem(mean_second_hand_ICE, axis=0)

    #EV
    overall_mean_new_EV = np.mean(mean_new_EV, axis=0)
    ci_new_EV = t.ppf(0.975, df=mean_new_EV.shape[0] - 1) * sem(mean_new_EV, axis=0)

    overall_mean_second_hand_EV = np.mean(mean_second_hand_EV, axis=0)
    ci_second_hand_EV = t.ppf(0.975, df=mean_second_hand_EV.shape[0] - 1) * sem(mean_second_hand_EV, axis=0)

    # Create the figure
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Plot individual traces (faded lines) for new and second-hand car prices
    for seed_new, seed_second_hand in zip(mean_new_ICE, mean_second_hand_ICE):
        ax1.plot(time_steps, seed_new, color='blue', alpha=0.3, linewidth=0.8)
        ax1.plot(time_steps, seed_second_hand, color='blue', alpha=0.3, linewidth=0.8, linestyle = "dashed")

    for seed_new, seed_second_hand in zip(mean_new_EV, mean_second_hand_EV):
        ax1.plot(time_steps, seed_new, color='green', alpha=0.3, linewidth=0.8)
        ax1.plot(time_steps, seed_second_hand, color='green', alpha=0.3, linewidth=0.8,  linestyle = "dashed")

    #ICE
    # Plot Mean and 95% CI for New Car Prices
    ax1.plot(time_steps, overall_mean_new_ICE, label="New Car Mean Price ICE", color="blue")
    ax1.fill_between(
        time_steps,
        overall_mean_new_ICE - ci_new_ICE,
        overall_mean_new_ICE + ci_new_ICE,
        color="blue",
        alpha=0.2,
        label="New Car 95% Confidence Interval ICE"
    )

    # Plot Mean and 95% CI for Second-hand Car Prices
    ax1.plot(time_steps, overall_mean_second_hand_ICE, label="Second-hand Car Mean Price ICE", color="blue", linestyle = "dashed")
    ax1.fill_between(
        time_steps,
        overall_mean_second_hand_ICE - ci_second_hand_ICE,
        overall_mean_second_hand_ICE + ci_second_hand_ICE,
        color="blue",
        alpha=0.2,
        label="Second-hand Car 95% Confidence Interval ICE"
    )

    #EV
    # Plot Mean and 95% CI for New Car Prices
    ax1.plot(time_steps, overall_mean_new_EV, label="New Car Mean Price EV", color="green")
    ax1.fill_between(
        time_steps,
        overall_mean_new_EV - ci_new_EV,
        overall_mean_new_EV + ci_new_EV,
        color="green",
        alpha=0.2,
        label="New Car 95% Confidence Interval EV"
    )

    # Plot Mean and 95% CI for Second-hand Car Prices
    ax1.plot(time_steps, overall_mean_second_hand_EV, label="Second-hand Car Mean Price EV", color="green", linestyle = "dashed")
    ax1.fill_between(
        time_steps,
        overall_mean_second_hand_EV - ci_second_hand_EV,
        overall_mean_second_hand_EV + ci_second_hand_EV,
        color="green",
        alpha=0.2,
        label="Second-hand Car 95% Confidence Interval EV"
    )

    # Format the plot
    ax1.set_xlabel("Time Step, months")
    ax1.set_ylabel("Price, $")
    add_vertical_lines(ax1, base_params)
    ax1.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    save_and_show(fig, fileName, "history_mean_price_with_traces_by_type", dpi)


# Sample main function
def main(fileName, dpi=600):

    base_params = load_object(fileName + "/Data", "base_params")
    print(base_params)
    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")

    history_total_emissions_arr = load_object(fileName + "/Data", "history_total_emissions_arr")
    history_prop_EV_arr= load_object(fileName + "/Data", "history_prop_EV_arr")
    #history_car_age_arr= load_object( fileName + "/Data", "history_car_age_arr")
    history_mean_price_ICE_EV_arr= load_object( fileName + "/Data", "history_mean_price_ICE_EV_arr")
    history_median_price_ICE_EV_arr= load_object( fileName + "/Data", "history_median_price_ICE_EV_arr")
    history_total_utility_arr= load_object(fileName + "/Data", "history_total_utility_arr")
    history_market_concentration_arr= load_object( fileName + "/Data", "history_market_concentration_arr")
    history_total_profit_arr= load_object( fileName + "/Data", "history_total_profit_arr")
    #history_quality_ICE= load_object( fileName + "/Data", "history_quality_ICE")
    #history_quality_EV= load_object( fileName + "/Data", "history_quality_EV")
    #history_efficiency_ICE= load_object( fileName + "/Data", "history_efficiency_ICE")
    #history_efficiency_EV= load_object( fileName + "/Data", "history_efficiency_EV")
    #history_production_cost_ICE= load_object( fileName + "/Data", "history_production_cost_ICE")
    #history_production_cost_EV= load_object( fileName + "/Data", "history_production_cost_EV")

    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]
    CO2_index_2010_22 = calibration_data_output["CO2_index"]

    #plot_ev_stock_multi_seed(base_params, EV_stock_prop_2010_22, data_array_ev_prop, fileName, dpi)
    #plot_ev_stock_multi_seed_pair(base_params, EV_stock_prop_2010_22, history_prop_EV_arr, fileName, dpi)

    # Plot each dataset
    #"""
    plot_calibrated_index_emissions(CO2_index_2010_22,base_params, fileName,history_total_emissions_arr, 
                    "Total Emissions Over Time", 
                    "Time Step, months", 
                    "Total Emissions, kgCO2", 
                    "history_total_emissions_and_index")

    plot_data_across_seeds(base_params, fileName,history_total_emissions_arr, 
                        "Total Emissions Over Time", 
                        "Time Step, months", 
                        "Total Emissions, kgCO2", 
                        "history_total_emissions")

    plot_ev_uptake_dual(EV_stock_prop_2010_22, base_params, fileName,history_prop_EV_arr, 
                        "Proportion of EVs Over Time", 
                        "Time Step, months", 
                        "Proportion of EVs", 
                        "history_prop_EV")


    plot_data_across_seeds(base_params, fileName,history_total_utility_arr, 
                        "Total Utility Over Time", 
                        "Time Step, months", 
                        "Total Utility", 
                        "history_total_utility")

    plot_data_across_seeds(base_params, fileName,history_market_concentration_arr, 
                        "Market Concentration Over Time", 
                        "Time Step, months", 
                        "Market Concentration", 
                        "history_market_concentration")

    plot_data_across_seeds(base_params, fileName,history_total_profit_arr, 
                        "Total Profit Over Time", 
                        "Time Step, months", 
                        "Total Profit, $", 
                        "history_total_profit")

    """
    plot_vehicle_attribute_time_series_by_type_split(
        base_params, 
        history_quality_ICE, history_quality_EV, 
        history_efficiency_ICE, history_efficiency_EV, 
        history_production_cost_ICE, history_production_cost_EV, 
        fileName
    )
    """

    plot_history_mean_price_multiple_seeds(
    base_params, 
    history_mean_price_ICE_EV_arr, 
    history_median_price_ICE_EV_arr, 
    fileName
    )

    
    plt.show()

if __name__ == "__main__":
    main("results/multi_seed_single_12_04_19__13_02_2025")