import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.plotting_data.single_experiment_plot import save_and_show, add_vertical_lines, format_plot

def plot_data_across_seeds( base_params,fileName, data, title, x_label, y_label, save_name, dpi=600):
    """
    Plot data across multiple seeds with mean, confidence intervals, and individual traces.

    Args:
        data: 2D array where rows are the time series for each seed (shape: [num_seeds, num_time_steps]).
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        save_name: Name of the file to save the plot.
        dpi: Dots per inch for saving the plot.
    """
    # Define x-axis as the range of the columns
    time_steps = np.arange(data.shape[1])

    # Calculate mean and 95% confidence interval
    mean_values = np.mean(data, axis=0)
    ci_range = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)  # 95% CI

    # Create subplots
    fig, ax1  = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

    # First subplot: Mean and 95% CI
    ax1.plot(time_steps, mean_values, label='Mean', color='blue')
    ax1.fill_between(
        time_steps, 
        mean_values - ci_range, 
        mean_values + ci_range, 
        color='blue', 
        alpha=0.3, 
        label='95% Confidence Interval'
    )
    #ax1.set_title(f"{title} (Mean and 95% CI)")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    add_vertical_lines(ax1, base_params)
    ax1.legend()
    #x1.grid()


    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, save_name, dpi)
    
def plot_ev_uptake_dual(real_data, base_params, fileName, data, title, x_label, y_label, save_name, dpi=600):
    """
    Plot data across multiple seeds with mean, confidence intervals, and individual traces.

    Args:
        data: 2D array where rows are the time series for each seed (shape: [num_seeds, num_time_steps]).
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        save_name: Name of the file to save the plot.
        dpi: Dots per inch for saving the plot.
    """
    # Define x-axis as the range of the columns
    init_index = base_params["duration_burn_in"] + 120
    time_steps_real = np.arange(init_index, init_index + len(real_data)*12, 12)

    time_steps = np.arange(data.shape[1])

    # Calculate mean and 95% confidence interval
    mean_values = np.mean(data, axis=0)
    ci_range = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)  # 95% CI

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)


    ax1.plot(time_steps_real, real_data, label="California Data", color='orange', linestyle= "--")
    ax2.plot(time_steps_real, real_data, label="California Data", color='orange', linestyle= "--")

    # First subplot: Mean and 95% CI
    ax1.plot(time_steps, mean_values, label='Mean', color='blue')
    ax1.fill_between(
        time_steps, 
        mean_values - ci_range, 
        mean_values + ci_range, 
        color='blue', 
        alpha=0.3, 
        label='95% Confidence Interval'
    )
    #ax1.set_title(f"{title} (Mean and 95% CI)")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    add_vertical_lines(ax1, base_params)
    ax1.legend()
    #ax1.grid()

    

    # Second subplot: Individual traces
    for seed_idx, seed_data in enumerate(data):
        ax2.plot(time_steps, seed_data, alpha=0.7)
    
    #ax2.set_title(f"{title} (Individual Seed Traces)")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    #ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    add_vertical_lines(ax2, base_params)
    #ax2.grid()

    
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
    and Production Cost for both ICE and EV with means and confidence intervals across multiple seeds.

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
    Helper function to plot an attribute (e.g., Quality, Efficiency, Production Cost) across multiple seeds.

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
    #ax.set_title(title)
    
    ax.set_ylabel(title)  # Use the first word of the title as the y-axis label
    #ax.grid()


def plot_distance_individuals_mean_median_type_multiple_seeds(
    base_params, 
    history_distance_individual_EV, 
    history_distance_individual_ICE, 
    fileName, dpi=600
):
    """
    Plots mean and median individual distances for EV and ICE with 95% confidence intervals across multiple seeds.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_distance_individual_EV: 2D array [num_seeds, num_time_steps].
        history_distance_individual_ICE: 2D array [num_seeds, num_time_steps].
        fileName: Directory or file name to save the plot.
        dpi: Resolution for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    history_distance_individual_EV = np.nanmean(np.asarray(history_distance_individual_EV),axis = 2)
    history_distance_individual_ICE = np.nanmean(np.asarray(history_distance_individual_ICE),axis = 2)

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

    # Format plot
    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Distance, km")

    #format_plot(ax, "User Distance Over Time", "Time Step, months", "Individual Distance")
    ax.legend()

    # Save and show the plot
    save_and_show(fig, fileName, "user_distance_mean_median_type_multiple_seeds", dpi)

def plot_history_car_age_multiple_seeds(
    base_params, history_car_age, fileName, dpi
):
    """
    Plots the mean, median, and 95% confidence interval for car ages across multiple seeds.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_car_age: 2D array [num_seeds, num_time_steps], where each row represents a seed's time series of car ages.
        fileName: Directory or file name to save the plot.
        dpi: Resolution for the saved plot.
    """
    # Calculate statistics across seeds

    history_car_age = np.mean(history_car_age, axis=2)
    means = np.mean(history_car_age, axis=0)
    #medians = np.median(history_car_age, axis=0)
    std_errors = sem(history_car_age, axis=0)
    ci = t.ppf(0.975, df=history_car_age.shape[0] - 1) * std_errors
    lower_bounds = means - ci
    upper_bounds = means + ci

    # Time steps
    time_steps = np.arange(history_car_age.shape[1])

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.plot(time_steps, medians, label="Median Age", color="red")
    ax.plot(time_steps, means, label="Mean Age, Months", color="blue")
    ax.fill_between(
        time_steps, lower_bounds, upper_bounds, color="blue", alpha=0.2, label="95% Confidence Interval"
    )
    #ax.set_title("Mean Age and 95% Confidence Interval Over Time")
    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Age, months")
    #ax.grid(True)

    # Add vertical lines for base parameters (e.g., milestones or events)
    add_vertical_lines(ax, base_params)

    # Add legend
    ax.legend()

    # Save and show the plot
    save_and_show(fig, fileName, "car_age_multiple_seeds", dpi)

def plot_history_mean_price_multiple_seeds(
    base_params, 
    history_mean_price, 
    history_median_price, 
    fileName, 
    dpi=600
):
    """
    Plots the mean, median, and 95% confidence interval for prices (new and second-hand cars) in the first subplot,
    and the traces of individual seeds in the second subplot.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_mean_price: 3D array [num_seeds, num_time_steps, 2] for mean prices.
        history_median_price: 3D array [num_seeds, num_time_steps, 2] for median prices.
        fileName: Directory or file name to save the plot.
        dpi: Resolution for the saved plot.
    """
    # Extract new and second-hand prices
    mean_new = history_mean_price[:, :, 0]  # Mean prices for new cars
    mean_second_hand = history_mean_price[:, :, 1]  # Mean prices for second-hand cars

    median_new = history_median_price[:, :, 0]  # Median prices for new cars
    median_second_hand = history_median_price[:, :, 1]  # Median prices for second-hand cars

    # Time steps
    time_steps = np.arange(mean_new.shape[1])

    # Compute mean and 95% CI across seeds
    overall_mean_new = np.mean(mean_new, axis=0)
    ci_new = t.ppf(0.975, df=mean_new.shape[0] - 1) * sem(mean_new, axis=0)

    overall_mean_second_hand = np.mean(mean_second_hand, axis=0)
    ci_second_hand = t.ppf(0.975, df=mean_second_hand.shape[0] - 1) * sem(mean_second_hand, axis=0)

    # Create the figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

    # First subplot: Mean and 95% CI
    # New car prices
    ax1.plot(time_steps, overall_mean_new, label="New Car Mean Price", color="blue")
    ax1.fill_between(
        time_steps,
        overall_mean_new - ci_new,
        overall_mean_new + ci_new,
        color="blue",
        alpha=0.2,
        label="New Car 95% Confidence Interval"
    )

    # Second-hand car prices
    ax1.plot(time_steps, overall_mean_second_hand, label="Second-hand Car Mean Price", color="green")
    ax1.fill_between(
        time_steps,
        overall_mean_second_hand - ci_second_hand,
        overall_mean_second_hand + ci_second_hand,
        color="green",
        alpha=0.2,
        label="Second-hand Car 95% Confidence Interval"
    )

    # Format the first subplot
    #ax1.set_title("Mean Prices and 95% Confidence Interval (New and Second-hand Cars)")
    ax1.set_ylabel("Price, $")
    #ax1.grid(True)
    add_vertical_lines(ax1, base_params)
    ax1.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    save_and_show(fig, fileName, "history_mean_price_with_traces", dpi)

def plot_history_mean_price_multiple_seeds_dual(
    base_params, 
    history_mean_price, 
    history_median_price, 
    fileName, 
    dpi=600
):
    """
    Plots the mean, median, and 95% confidence interval for prices (new and second-hand cars) in the first subplot,
    and the traces of individual seeds in the second subplot.

    Args:
        base_params: Parameters of the simulation for adding vertical lines.
        history_mean_price: 3D array [num_seeds, num_time_steps, 2] for mean prices.
        history_median_price: 3D array [num_seeds, num_time_steps, 2] for median prices.
        fileName: Directory or file name to save the plot.
        dpi: Resolution for the saved plot.
    """
    # Extract new and second-hand prices
    mean_new = history_mean_price[:, :, 0]  # Mean prices for new cars
    mean_second_hand = history_mean_price[:, :, 1]  # Mean prices for second-hand cars

    median_new = history_median_price[:, :, 0]  # Median prices for new cars
    median_second_hand = history_median_price[:, :, 1]  # Median prices for second-hand cars

    # Time steps
    time_steps = np.arange(mean_new.shape[1])

    # Compute mean and 95% CI across seeds
    overall_mean_new = np.mean(mean_new, axis=0)
    ci_new = t.ppf(0.975, df=mean_new.shape[0] - 1) * sem(mean_new, axis=0)

    overall_mean_second_hand = np.mean(mean_second_hand, axis=0)
    ci_second_hand = t.ppf(0.975, df=mean_second_hand.shape[0] - 1) * sem(mean_second_hand, axis=0)

    # Create the figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    ax1 = axes[0]
    ax2 = axes[1]
    # First subplot: Mean and 95% CI
    # New car prices
    ax1.plot(time_steps, overall_mean_new, label="New Car Mean Price", color="blue")
    ax1.fill_between(
        time_steps,
        overall_mean_new - ci_new,
        overall_mean_new + ci_new,
        color="blue",
        alpha=0.2,
        label="New Car 95% Confidence Interval"
    )

    # Second-hand car prices
    ax1.plot(time_steps, overall_mean_second_hand, label="Second-hand Car Mean Price", color="green")
    ax1.fill_between(
        time_steps,
        overall_mean_second_hand - ci_second_hand,
        overall_mean_second_hand + ci_second_hand,
        color="green",
        alpha=0.2,
        label="Second-hand Car 95% Confidence Interval"
    )

    # Format the first subplot
    #ax1.set_title("Mean Prices and 95% Confidence Interval (New and Second-hand Cars)")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    add_vertical_lines(ax1, base_params)
    ax1.legend()

    # Second subplot: Traces of individual seeds
    # New car prices
    for seed_idx in range(mean_new.shape[0]):
        ax2.plot(time_steps, mean_new[seed_idx], label=f"Seed {seed_idx + 1} Mean (New)", alpha=0.5, color="blue")
        ax2.plot(time_steps, mean_second_hand[seed_idx], label=f"Seed {seed_idx + 1} Mean (Second-hand)", alpha=0.5, color="green")

    # Format the second subplot
    #ax2.set_title("Traces of Mean Prices Over Time (New and Second-hand Cars)")
    ax2.set_xlabel("Time Step, months")
    ax2.set_ylabel("Price")
    ax2.grid(True)

    # Add legend to second subplot (limited to 10 labels for clarity)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:10], labels[:10], fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust layout and save the plot
    plt.tight_layout()
    save_and_show(fig, fileName, "history_mean_price_with_traces", dpi)

# Sample main function
def main(fileName, dpi=600):

    base_params = load_object(fileName + "/Data", "base_params")

    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")

    history_total_emissions_arr = load_object(fileName + "/Data", "history_total_emissions_arr")
    history_prop_EV_arr= load_object(fileName + "/Data", "history_prop_EV_arr")
    history_car_age_arr= load_object( fileName + "/Data", "history_car_age_arr")
    history_mean_price_arr= load_object( fileName + "/Data", "history_mean_price_arr")
    history_median_price_arr= load_object( fileName + "/Data", "history_median_price_arr")
    history_total_utility_arr= load_object(fileName + "/Data", "history_total_utility_arr")
    history_market_concentration_arr= load_object( fileName + "/Data", "history_market_concentration_arr")
    history_total_profit_arr= load_object( fileName + "/Data", "history_total_profit_arr")
    history_quality_ICE= load_object( fileName + "/Data", "history_quality_ICE")
    history_quality_EV= load_object( fileName + "/Data", "history_quality_EV")
    history_efficiency_ICE= load_object( fileName + "/Data", "history_efficiency_ICE")
    history_efficiency_EV= load_object( fileName + "/Data", "history_efficiency_EV")
    history_production_cost_ICE= load_object( fileName + "/Data", "history_production_cost_ICE")
    history_production_cost_EV= load_object( fileName + "/Data", "history_production_cost_EV")
    history_distance_individual_ICE = load_object( fileName + "/Data", "history_distance_individual_ICE")
    history_distance_individual_EV = load_object( fileName + "/Data", "history_distance_individual_EV")

    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    #plot_ev_stock_multi_seed(base_params, EV_stock_prop_2010_22, data_array_ev_prop, fileName, dpi)
    #plot_ev_stock_multi_seed_pair(base_params, EV_stock_prop_2010_22, history_prop_EV_arr, fileName, dpi)


    # Plot each dataset

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

    plot_vehicle_attribute_time_series_by_type_split(
        base_params, 
        history_quality_ICE, history_quality_EV, 
        history_efficiency_ICE, history_efficiency_EV, 
        history_production_cost_ICE, history_production_cost_EV, 
        fileName
    )

    plot_distance_individuals_mean_median_type_multiple_seeds(
        base_params, 
        history_distance_individual_EV, 
        history_distance_individual_ICE, 
        fileName
    )

    plot_history_car_age_multiple_seeds(
        base_params, history_car_age_arr, fileName, dpi
    )

    plot_history_mean_price_multiple_seeds(
    base_params, 
    history_mean_price_arr, 
    history_median_price_arr, 
    fileName
    )

    
    plt.show()

if __name__ == "__main__":
    main("results/multi_seed_single_12_50_54__21_01_2025")