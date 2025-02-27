import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data

# Ensure directory existence
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Set up plot format helper
def format_plot(ax, title, xlabel, ylabel, legend=True, grid=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    if grid:
        ax.grid()

# Helper function to save and show plots
def save_and_show(fig, fileName, plot_name, dpi=600):
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/{plot_name}.png", dpi=dpi, format="png")
import matplotlib.pyplot as plt

def add_vertical_lines(ax, base_params, color='black', linestyle='--'):
    """
    Adds dashed vertical lines to the plot at specified steps, ensuring data is plotted only from the end of the burn-in period onwards.

    Parameters:
    ax : matplotlib.axes.Axes
        The Axes object to add the lines to.
    base_params : dict
        Dictionary containing 'duration_burn_in' and 'duration_no_carbon_price'.
    color : str, optional
        Color of the dashed lines. Default is 'black'.
    linestyle : str, optional
        Style of the dashed lines. Default is '--'.
    """
    burn_in = base_params["duration_burn_in"]
    no_carbon_price = base_params["duration_no_carbon_price"]
    ev_research_start_time = base_params["ev_research_start_time"]
    ev_production_start_time = base_params["ev_production_start_time"]
    #second_hand_burn_in = base_params["parameters_second_hand"]["burn_in_second_hand_market"]

    # Ensure the x-axis limits start at the end of the burn-in period
    #ax.set_xlim(left=burn_in)
    
    # Adding the dashed lines
    #ax.axvline(second_hand_burn_in, color=color, linestyle='-.', label="Second hand market start")
    #ax.axvline(burn_in, color=color, linestyle='--', label="Burn-in period end")
    ax.axvline(ev_research_start_time, color=color, linestyle=':', label="EV research start")
    ax.axvline(ev_production_start_time, color="red", linestyle=':', label="EV sale start")
    
    if base_params["EV_rebate_state"]:
        ax.axvline(base_params["parameters_rebate_calibration"]["start_time"], 
                   color="red", linestyle='-.', label="EV adoption subsidy start")
    if base_params["duration_future"] > 0:
        ax.axvline(no_carbon_price, color="red", linestyle='--', label="Policy start")

def plot_total_utility(base_params, social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_total_utility[base_params["duration_burn_in"]:], marker='o')
    format_plot(ax, "Total Utility Over Time", "Time Step", "Total Utility", legend=False)
    save_and_show(fig, fileName, "total_utility", dpi)

def plot_carbon_price(base_params,controller, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    "FIX THIS!!"
    ax.plot(time_series, controller.carbon_price_time_series[:len(time_series)], marker='o')
    format_plot(ax, "Carbon price Over Time", "Time Step", "Carbon price", legend=False)
    save_and_show(fig, fileName, "carbon_price", dpi)

def plot_total_profit(base_params, firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, np.asarray(firm_manager.history_total_profit[base_params["duration_burn_in"]:])/base_params["computing_coefficient"], marker='o')
    format_plot(ax, "Total Profit Over Time", "Time Step", "Total Profit, $", legend=False)
    save_and_show(fig, fileName, "total_profit", dpi)

def plot_ev_consider_adoption_rate(base_params,social_network, time_series, fileName, EV_stock_prop_2010_22, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    EV_stock_prop_2010_22

    time_yearly = np.arange(12 + 108, 12 + 264, 12)
    ax.plot(time_yearly,EV_stock_prop_2010_22, label = "California data", linestyle= "dashed", color = "orange")
    ax.plot(time_series, social_network.history_consider_ev_rate[base_params["duration_burn_in"]:], label = "Consider", color = "blue")
    ax.plot(time_series, social_network.history_ev_adoption_rate[base_params["duration_burn_in"]:], label = "Adopt", color = "green")
    add_vertical_lines(ax, base_params)
    ax.legend()
    format_plot(ax, "EV Adoption Rate Over Time", "Time Step", "EV Adoption Rate", legend=False)
    save_and_show(fig, fileName, "plot_ev_consider_adoption_rate", dpi)



def plot_prop_EV_on_sale(base_params,firm_manager, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked area (continuous stacked bar equivalent)
    ax.plot(firm_manager.history_prop_EV[base_params["duration_burn_in"]:])

    add_vertical_lines(ax, base_params)
    # Set plot labels and limits
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Proportion of Cars on Sale EV")
    ax.set_ylim(0, 1)  # Proportion range
    ax.legend(loc="upper left")

    # Save and show the plot
    save_and_show(fig, fileName, "plot_prop_EV_on_sale", dpi)

def plot_history_prop_EV_research(base_params,firm_manager, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))


    # Plot stacked area (continuous stacked bar equivalent)
    ax.plot(firm_manager.history_prop_EV_research[base_params["duration_burn_in"]:], color = "Green")
    #ax.plot(firm_manager.history_prop_ICE_research[base_params["duration_burn_in"]:], color = "Blue", label = "ICE")

    add_vertical_lines(ax, base_params)
    # Set plot labels and limits
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Yearly average all firms: Proportion of Reserach EV")
    ax.set_ylim(0, 1)  # Proportion range

    # Save and show the plot
    save_and_show(fig, fileName, "plot_history_prop_EV_research", dpi)



def plot_transport_users_stacked(base_params,social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate total users at each time step
    total_users = social_network.num_individuals

    # Calculate proportions
    ice_prop = np.array(social_network.history_ICE_users) / total_users
    ev_prop = np.array(social_network.history_EV_users) / total_users

    # Plot stacked area (continuous stacked bar equivalent)
    ax.stackplot(time_series, ice_prop[base_params["duration_burn_in"]:], ev_prop[base_params["duration_burn_in"]:],
                 labels=['ICE', 'EV' ],
                 alpha=0.8)
    add_vertical_lines(ax, base_params)
    # Set plot labels and limits
    ax.set_title("Transport Users Over Time (Proportion)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Proportion of Transport Users")
    ax.set_ylim(0, 1)  # Proportion range
    ax.legend(loc="upper left")

    # Save and show the plot
    save_and_show(fig, fileName, "plot_transport_users_stacked", dpi)

def plot_vehicle_attribute_time_series_by_type_split(base_params, social_network, time_series, fileName, dpi=600):
    """
    Plots time series of Quality, Efficiency (separate for ICE and EV), 
    and Production Cost for both ICE and EV with means and confidence intervals.

    Args:
        social_network (object): Contains the history of vehicle attributes for ICE and EV.
        time_series (list): Time steps to plot on the x-axis.
        file_name (str): Directory or file name to save the plots.
        dpi (int): Resolution for saving the plots.
    """
    # Attributes for ICE and EV
    attributes = {
        "Quality": ("history_quality_ICE", "history_quality_EV"),
        "Production Cost": ("history_production_cost_ICE", "history_production_cost_EV"),
    }
    efficiency_attributes = ("history_efficiency_ICE", "history_efficiency_EV")

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])  # Create a 2x2 grid layout

    # Plot Quality (big plot, top-left)
    ax_quality = fig.add_subplot(gs[0, 0])
    plot_attribute(ax_quality, "Quality", attributes["Quality"], base_params, social_network, time_series)

    # Plot Production Cost (big plot, top-right)
    ax_cost = fig.add_subplot(gs[0, 1])
    plot_attribute(ax_cost, "Production Cost", attributes["Production Cost"], base_params, social_network, time_series)

    # Plot Efficiency ICE (bottom-left)
    ax_efficiency_ice = fig.add_subplot(gs[1, 0])
    plot_single_efficiency(
        ax_efficiency_ice, "Efficiency (ICE)", efficiency_attributes[0], base_params, social_network, time_series, color="blue"
    )

    # Plot Efficiency EV (bottom-right)
    ax_efficiency_ev = fig.add_subplot(gs[1, 1])
    plot_single_efficiency(
        ax_efficiency_ev, "Efficiency (EV)", efficiency_attributes[1], base_params, social_network, time_series, color="green"
    )

    fig.suptitle("Vehicle Attributes (ICE and EV) Over Time - In use")
    plt.tight_layout()

    # Save and show the plot
    save_and_show(fig, fileName, "vehicle_attribute_time_series_ICE_EV", dpi)

def plot_attribute(ax, attribute_name, attr_names, base_params, social_network, time_series):
    """
    Helper function to plot a single attribute (Quality or Production Cost).
    """
    
    
    # Extract histories for ICE and EV
    ice_attr, ev_attr = attr_names
    ice_history = getattr(social_network, ice_attr, [])
    ev_history = getattr(social_network, ev_attr, [])

    # Calculate means and confidence intervals
    ice_means = [np.mean(values) if values else np.nan for values in ice_history]
    ice_max = [np.max(values) if values else np.nan for values in ice_history]
    ice_min = [np.min(values) if values else np.nan for values in ice_history]
    ice_confidence_intervals = [1.96 * sem(values) if values else 0 for values in ice_history]

    ev_means = [np.mean(values) if values else np.nan for values in ev_history]
    ev_max = [np.max(values) if values else np.nan for values in ev_history]
    ev_min = [np.min(values) if values else np.nan for values in ev_history]
    ev_confidence_intervals = [1.96 * sem(values) if values else 0 for values in ev_history]

    #print(len(ice_means), len(ice_means ), len(ev_means))
    #quit()
    # Plot ICE data
    ax.plot(time_series, ice_means[base_params["duration_burn_in"]:], label=f"ICE {attribute_name}", color="blue")
    ax.plot(time_series, ice_max[base_params["duration_burn_in"]:], color="blue", linestyle = "--")
    ax.plot(time_series, ice_min[base_params["duration_burn_in"]:], color="blue", linestyle = "--")
    ax.fill_between(
        time_series,
        (np.array(ice_means) - np.array(ice_confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(ice_means) + np.array(ice_confidence_intervals))[base_params["duration_burn_in"]:],
        color="blue", alpha=0.2
    )

    # Plot EV data
    ax.plot(time_series, ev_means[base_params["duration_burn_in"]:], label=f"EV {attribute_name}", color="green")
    ax.plot(time_series, ev_max[base_params["duration_burn_in"]:], color="green", linestyle = "--")
    ax.plot(time_series, ev_min[base_params["duration_burn_in"]:], color="green", linestyle = "--")
    ax.fill_between(
        time_series,
        (np.array(ev_means) - np.array(ev_confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(ev_means) + np.array(ev_confidence_intervals))[base_params["duration_burn_in"]:],
        color="green", alpha=0.2
    )

    # Set title and labels
    ax.set_title(f"{attribute_name} Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(attribute_name)
    ax.grid()
    ax.legend()

    add_vertical_lines(ax, base_params)

def plot_single_efficiency(ax, title, attr_name, base_params, social_network, time_series, color):
    """
    Helper function to plot Efficiency for a single vehicle type (ICE or EV).
    """
    

    history = getattr(social_network, attr_name, [])
    means = [np.mean(values) if values else np.nan for values in history]
    confidence_intervals = [1.96 * sem(values) if values else 0 for values in history]
    maxs = [np.max(values) if values else np.nan for values in history]
    mins = [np.min(values) if values else np.nan for values in history]
    # Plot data
    ax.plot(time_series, means[base_params["duration_burn_in"]:], label=title, color=color)
    ax.plot(time_series, maxs[base_params["duration_burn_in"]:], color=color, linestyle = "--")
    ax.plot(time_series, mins[base_params["duration_burn_in"]:], color=color, linestyle = "--")
    ax.fill_between(
        time_series,
        (np.array(means) - np.array(confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(means) + np.array(confidence_intervals))[base_params["duration_burn_in"]:],
        color=color, alpha=0.2
    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Efficiency")
    ax.grid()

    add_vertical_lines(ax, base_params)

def plot_prod_vehicle_attribute_time_series_by_type_split(base_params, firm_manager, time_series, fileName, dpi=600):
    """
    Plots time series of Quality, Efficiency (separate for ICE and EV), 
    and Production Cost for both ICE and EV with means and confidence intervals.

    Args:
        social_network (object): Contains the history of vehicle attributes for ICE and EV.
        time_series (list): Time steps to plot on the x-axis.
        file_name (str): Directory or file name to save the plots.
        dpi (int): Resolution for saving the plots.
    """
    # Attributes for ICE and EV
    attributes = {
        "Quality": ("history_quality_ICE", "history_quality_EV"),
        "Production Cost": ("history_production_cost_ICE", "history_production_cost_EV"),
    }
    efficiency_attributes = ("history_efficiency_ICE", "history_efficiency_EV")

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])  # Create a 2x2 grid layout

    # Plot Quality (big plot, top-left)
    ax_quality = fig.add_subplot(gs[0, 0])
    plot_attribute(ax_quality, "Quality", attributes["Quality"], base_params, firm_manager, time_series)

    # Plot Production Cost (big plot, top-right)
    ax_cost = fig.add_subplot(gs[0, 1])
    plot_attribute(ax_cost, "Production Cost", attributes["Production Cost"], base_params, firm_manager, time_series)

    # Plot Efficiency ICE (bottom-left)
    ax_efficiency_ice = fig.add_subplot(gs[1, 0])
    plot_single_efficiency(
        ax_efficiency_ice, "Efficiency (ICE)", efficiency_attributes[0], base_params, firm_manager, time_series, color="blue"
    )

    # Plot Efficiency EV (bottom-right)
    ax_efficiency_ev = fig.add_subplot(gs[1, 1])
    plot_single_efficiency(
        ax_efficiency_ev, "Efficiency (EV)", efficiency_attributes[1], base_params, firm_manager, time_series, color="green"
    )

    fig.suptitle("Prod Vehicle Attributes (ICE and EV) Over Time")
    plt.tight_layout()

    # Save and show the plot
    save_and_show(fig, fileName, "vehicle_prod_attribute_time_series_ICE_EV", dpi)

def plot_preferences(social_network, fileName, dpi=600):
    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(12, 6))
    axes[0].hist(social_network.beta_vec, bins=30, alpha=0.5, label=r'$\beta_i$ (Price sentivity)')
    axes[1].hist(social_network.gamma_vec, bins=30, alpha=0.5, label=r'$\gamma_i$ (Environmental concern)')
    axes[2].hist(social_network.chi_vec, bins=30, alpha=0.5, label=r'$\chi$ (EV Threshold)')
    
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    axes[0].set_ylabel("Frequency")
    axes[0].set_xlabel("Value")
    axes[1].set_xlabel("Value")
    axes[2].set_xlabel("Value")

    save_and_show(fig, fileName, "preferences", dpi)

def plot_history_research_type(firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.asarray([firm.history_research_type for firm in firm_manager.firms_list])
    for firm_data in data:
        ax.scatter(time_series,firm_data)
    format_plot(ax, "EV Research Proportion Over Time", "Time Step", "Proportion Research EV", legend=False)
    save_and_show(fig, fileName, "history_research_type", dpi)

def plot_segment_count_grid(base_params, firm_manager, time_series, fileName):
    """
    Plots the count of individuals in each segment over time in a grid layout.

    Parameters:
        firm_manager: Object containing historical segment data.
        time_series: List or array of time steps.
        fileName: Path to save the plot.
    """
    import itertools

    # Retrieve the segment codes
    num_beta_segments = firm_manager.num_beta_segments
    all_segment_codes = list(itertools.product(range(num_beta_segments), range(2), range(2)))

    # Initialize a dictionary to store aggregated segment counts over time
    segment_counts = {code: [] for code in all_segment_codes}

    # Organize segment data
    for t, segment_data in enumerate(firm_manager.history_segment_count):
        for segment_idx, count in enumerate(segment_data):
            segment_code = all_segment_codes[segment_idx]
            if segment_code not in segment_counts:
                segment_counts[segment_code] = []
            # Ensure the list for this segment has enough time steps
            while len(segment_counts[segment_code]) <= t:
                segment_counts[segment_code].append(0)
            segment_counts[segment_code][t] += count

    # Ensure all segment counts have the same length as the time series
    max_time_steps = len(time_series)
    for segment in segment_counts:
        while len(segment_counts[segment]) < max_time_steps:
            segment_counts[segment].append(0)

    # Determine the number of rows and columns for the grid
    num_segments = len(segment_counts)
    cols = 4
    rows = (num_segments + cols - 1) // cols  # Ceiling division for rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each segment
    for i, (segment, counts) in enumerate(segment_counts.items()):
        ax = axes[i]
        ax.plot(time_series, counts[base_params["duration_burn_in"]:], label=f"Segment {segment}")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid()
        ax.set_title(f"Segment {segment}")

    # Remove unused axes
    for j in range(len(segment_counts), len(axes)):
        fig.delaxes(axes[j])

    # Add labels and adjust layout
    fig.supxlabel("Time Step")
    fig.supylabel("Segment Count")
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    save_and_show(fig, fileName, "segment_count_grid", 600)

def plot_history_car_age( base_params,social_network,time_series, fileName, dpi):
    """
    Plots the mean and 95% confidence interval for a time series of ages.
    
    Args:
    - time_series: A list of time steps.
    - ages_list: A list of lists, where each inner list contains ages at a given time step.
    - fileName: The name of the file where the plot will be saved.
    - dpi: Resolution for the saved plot.
    """
    # Calculate mean and confidence interval
    means = []
    lower_bounds = []
    upper_bounds = []
    medians = []

    ages_list = social_network.history_car_age
    for ages in ages_list:
        ages = np.array(ages)
        valid_ages = ages[~np.isnan(ages)]  # Exclude NaNs from calculations
        mean = np.mean(valid_ages)
        median  = np.median(valid_ages)
        confidence = t.ppf(0.975, len(valid_ages)-1) * sem(valid_ages)
        
        means.append(mean)
        medians.append(median)
        lower_bounds.append(mean - confidence)
        upper_bounds.append(mean + confidence)

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, medians[base_params["duration_burn_in"]:], label="Median Age", color='red')
    ax.plot(time_series, means[base_params["duration_burn_in"]:], label="Mean Age", color='blue')
    ax.fill_between(time_series, lower_bounds[base_params["duration_burn_in"]:], upper_bounds[base_params["duration_burn_in"]:], color='blue', alpha=0.2, label="95% Confidence Interval")
    ax.set_title("Mean Age and 95% Confidence Interval Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Age")

    ax.grid(True)
    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "car age owned", dpi)   


def plot_calibration_data(base_params, controller, time_series, fileName, dpi=600):
    fig, axes = plt.subplots(ncols = 3,figsize=(10, 6))
    #print(social_network.history_second_hand_bought)
    #quit()

    axes[0].plot((controller.gas_price_california_vec[base_params["duration_burn_in"]:])/base_params["computing_coefficient"])
    axes[1].plot((controller.electricity_price_vec[base_params["duration_burn_in"]:])/base_params["computing_coefficient"])
    axes[2].plot(controller.electricity_emissions_intensity_vec[base_params["duration_burn_in"]:])

    axes[0].set_ylabel("Gas price california in 2020 Dollars")
    axes[1].set_ylabel("Electricity price california in 2020 Dollars")
    axes[2].set_ylabel("Urban Electricity emissions intensity in kgCO2/kWhr")
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_calibration_data", dpi)   

def plot_ev_stock(base_params, real_data, social_network, fileName, dpi=600):
    data_truncated = convert_data(social_network.history_prop_EV, base_params)

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(data_truncated, label = "Simulated data")
    ax.plot(real_data, label = "California data")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("EV stock %")
    ax.legend(loc="best")
    save_and_show(fig, fileName, "plot_ev_stock", dpi)

def plot_history_count_buy_stacked(base_params, social_network, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Convert to numpy array
    data = np.asarray(social_network.history_count_buy)  # shape: (timesteps, categories)

    # Compute proportions: divide each row by its sum
    row_sums = data.sum(axis=1, keepdims=True)
    data_proportions = data / row_sums  # shape remains (timesteps, categories)

    # Prepare x-axis and labels
    x = np.arange(data.shape[0])
    labels = ["current", "new", "second hand"]

    # Create a stacked area chart
    ax.stackplot(x, (data_proportions.T), labels=labels)

    # Labeling
    ax.set_xlabel("Time")
    ax.set_ylabel("Proportion of buys")
    ax.set_ylim([0, 1])
    ax.set_xlim(left = base_params["duration_burn_in"])

    add_vertical_lines(ax, base_params)
    ax.legend(loc='lower right')
    save_and_show(fig, fileName, "count_buy_stacked", dpi)

def plot_history_mean_price_by_type(base_params, social_network, fileName, dpi=600):

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    
    data = np.asarray(social_network.history_mean_price_ICE_EV)/base_params["computing_coefficient"]
    data_ICE_first = data[:,0,0]
    data_EV_first = data[:,0,1]
    data_ICE_second = data[:,1,0]
    data_EV_second = data[:,1,1]

    ax.plot(data_ICE_first[base_params["duration_burn_in"]:], label = "ICE new", color = "blue", linestyle= "solid")
    ax.plot(data_EV_first[base_params["duration_burn_in"]:], label = "EV new", color = "green", linestyle= "solid")
    ax.plot(data_ICE_second[base_params["duration_burn_in"]:], label = "ICE second hand", color = "blue", linestyle= "dashed")
    ax.plot(data_EV_second[base_params["duration_burn_in"]:], label = "EV second hand", color = "green", linestyle= "dashed")

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Price, $")

    add_vertical_lines(ax, base_params)
    ax.legend()
    save_and_show(fig, fileName, "history_mean_price_by_type", dpi)

def plot_history_median_price_by_type(base_params, social_network, fileName, dpi=600):

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    
    data = np.asarray(social_network.history_median_price_ICE_EV)/base_params["computing_coefficient"]
    data_ICE_first = data[:,0,0]
    data_EV_first = data[:,0,1]
    data_ICE_second = data[:,1,0]
    data_EV_second = data[:,1,1]

    ax.plot(data_ICE_first[base_params["duration_burn_in"]:], label = "ICE new", color = "blue", linestyle= "solid")
    ax.plot(data_EV_first[base_params["duration_burn_in"]:], label = "EV new", color = "green", linestyle= "solid")
    ax.plot(data_ICE_second[base_params["duration_burn_in"]:], label = "ICE second hand", color = "blue", linestyle= "dashed")
    ax.plot(data_EV_second[base_params["duration_burn_in"]:], label = "EV second hand", color = "green", linestyle= "dashed")

    ax.set_xlabel("Time")
    ax.set_ylabel("Median Price, $")

    add_vertical_lines(ax, base_params)
    ax.legend()
    save_and_show(fig, fileName, "history_median_price_by_type", dpi)

def plot_kg_co2_per_year_per_vehicle_by_type(base_params, social_network, time_series, fileName, dpi = 600):
    
    data_time_series_ICE = np.asarray(social_network.history_driving_emissions_ICE)/np.asarray(social_network.history_ICE_users)
    history_ICE_users = np.asarray(social_network.history_ICE_users)

    # Replace 0 with np.nan in history_ICE_users
    history_ICE_users = np.where(history_ICE_users == 0, np.nan, history_ICE_users)

    # Perform the division
    data_time_series_EV = np.asarray(social_network.history_driving_emissions_EV) / history_ICE_users

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(time_series, data_time_series_ICE[base_params["duration_burn_in"]:], label = "ICE")
    ax.plot(time_series, data_time_series_EV[base_params["duration_burn_in"]:], label = "EV")
    
    #https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator-calculations-and-references

    #GASOLINE = 4.29 metric tons CO2e/vehicle /year
    # =  0.3575 metric tons CO2e/vehicle /month
    # = 357.5 kg CO2e/vehicle/month 

    #EVs
    # = 1.13 metric tons CO2e/vehicle/year
    # = 0.09416666666 metric tons CO2e/vehicle /month
    # = 94.1666666667 kg CO2e/vehicle /month

    ax.axhline(y = 357.5, label = "ICE real", color = "red", linestyle = "--")
    ax.axhline(y = 94.1666666667, label = "EV real", color = "black", linestyle = "--")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Kg CO2 per vehicle")

    #add_vertical_lines(ax, base_params)
    
    ax.legend(loc="center left")
    save_and_show(fig, fileName, "plot_kg_co2_per_year_per_vehicle_by_type", dpi)

def plot_fuel_costs_verus_carbon_price_kWhr(base_params,controller, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))

    carbon_tax_paid = np.asarray(controller.carbon_price_time_series)*controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
    total_ice = controller.history_gas_price + carbon_tax_paid[:-2]

    ax.plot(controller.history_electricity_price[base_params["duration_burn_in"]:], label = "Fuel cost per kWhr, EV")
    ax.plot(controller.history_gas_price[base_params["duration_burn_in"]:], label = "Fuel cost per kWhr, ICE")
    ax.plot(carbon_tax_paid[base_params["duration_burn_in"]:] , label = "carbon tax cost per kWhr, ICE")
    ax.plot(total_ice[base_params["duration_burn_in"]:], label = "TOTAL cost per kWhr, ICE")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Dollars per kWhr")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_fuel_costs_verus_carbon_price_kWhr", dpi)

def plot_fuel_costs_verus_carbon_price_km(base_params,controller, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))

    carbon_tax_paid = np.asarray(controller.carbon_price_time_series)*controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
    #total_ice = controller.history_gas_price + carbon_tax_paid[:-2]

    Eff_omega_a_t_ICE_median = np.asarray([np.median(values) if values else np.nan for values in controller.social_network.history_efficiency_ICE])
    Eff_omega_a_t_EV_median = np.asarray([np.median(values) if values else np.nan for values in controller.social_network.history_efficiency_EV])
    #Eff_omega_a_t_ICE_median = np.median(controller.social_network.history_efficiency_ICE, axis = 0)
    #Eff_omega_a_t_EV_median = np.median(controller.social_network.history_efficiency_EV, axis = 0)

    ax.plot((controller.history_electricity_price/Eff_omega_a_t_EV_median)[base_params["duration_burn_in"]:], label = "Fuel cost per km, EV")
    ax.plot((controller.history_gas_price/Eff_omega_a_t_ICE_median)[base_params["duration_burn_in"]:], label = "Fuel cost per km, ICE")
    ax.plot((carbon_tax_paid[:-2]/Eff_omega_a_t_ICE_median)[base_params["duration_burn_in"]:] , label = "carbon tax cost per km, ICE")
    #ax.plot(total_ice/Eff_omega_a_t_ICE_median[:-1], label = "TOTAL cost per km, ICE")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Dollars per km")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_fuel_costs_verus_carbon_price_km", dpi)

def plot_fuel_emissions_verus_carbon_price_km(base_params,controller, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))

    #carbon_tax_paid = np.asarray(controller.carbon_price_time_series)*controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]

    Eff_omega_a_t_ICE_median = np.asarray([np.median(values) if values else np.nan for values in controller.social_network.history_efficiency_ICE])
    Eff_omega_a_t_EV_median = np.asarray([np.median(values) if values else np.nan for values in controller.social_network.history_efficiency_EV])
    #Eff_omega_a_t_ICE_median = np.median(controller.social_network.history_efficiency_ICE, axis = 0)
    #Eff_omega_a_t_EV_median = np.median(controller.social_network.history_efficiency_EV, axis = 0)

    ax.plot((controller.electricity_emissions_intensity_vec[:-1]/Eff_omega_a_t_EV_median)[base_params["duration_burn_in"]:], label = "Emissions per km, EV")
    ax.plot((controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]/Eff_omega_a_t_ICE_median)[base_params["duration_burn_in"]:], label = "Emissions per km, ICE")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Emissions per km")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_fuel_Emissions_verus_carbon_price_km", dpi)

def emissions_decomposed(base_params, social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    

    # Right plot: Stacked area plot for ICE and EV emissions
    driving_ICE = np.cumsum(social_network.history_driving_emissions_ICE[base_params["duration_burn_in"]:])
    driving_EV = np.cumsum(social_network.history_driving_emissions_EV[base_params["duration_burn_in"]:])
    production_ICE = np.cumsum(social_network.history_production_emissions_ICE[base_params["duration_burn_in"]:])
    production_EV = np.cumsum(social_network.history_production_emissions_EV[base_params["duration_burn_in"]:])
    
    ax.stackplot(
        time_series, 
        driving_ICE, 
        driving_EV, 
        production_ICE, 
        production_EV,
        labels=['Driving Emissions ICE', 'Driving Emissions EV', 'Production Emissions ICE', 'Production Emissions EV']
    )
    ax.plot(time_series, np.cumsum(social_network.history_total_emissions[base_params["duration_burn_in"]:]), 
                 label='Total Emissions', color='black', linewidth=1.5)
    ax.set_title("Cumulative Emissions by Source")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Emissions")
    ax.legend()

    # Format and save
    plt.tight_layout()
    save_and_show(fig, fileName, "emissions_decomposed", dpi)

def plot_profit_margins_by_type(base_params, firm_manager,time_series,  fileName, dpi=600):

    time_points_new_ICE = []
    time_points_new_EV = []
    profit_margins_ICE = []
    profit_margins_EV = []
    
    for i, pm_list in enumerate(firm_manager.history_profit_margins_ICE[base_params["duration_burn_in"]:]):
        time_points_new_ICE.extend([time_series[i]] * len(pm_list))  
        profit_margins_ICE.extend(pm_list)

    for i, pm_list in enumerate(firm_manager.history_profit_margins_EV[base_params["duration_burn_in"]:]):
        time_points_new_EV.extend([time_series[i]] * len(pm_list))  
        profit_margins_EV.extend(pm_list)
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    print()
    ax.scatter(time_points_new_ICE, profit_margins_ICE, marker='o', alpha=0.7, color = "blue", label = "ICE")
    ax.scatter(time_points_new_EV, profit_margins_EV, marker='o', alpha=0.7, color = "green", label = "EV")

    ax.set_xlabel("Time")
    ax.set_ylabel("Profit margin (P-C)/C")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "plot_profit_margins_by_type", dpi)

def plot_history_W(base_params, firm_manager,time_series,  fileName, dpi=600):


    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray(firm_manager.history_W).T
    for i, data_segment in enumerate(data):
        ax.plot(time_series, data_segment[base_params["duration_burn_in"]:], label = i)

    ax.set_xlabel("Time")
    ax.set_ylabel("W")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "plot_history_W", dpi)

def plot_market_concentration_yearly(base_params,firm_manager, time_series, fileName, dpi=600):
    # Ensure the data is in numpy arrays for easier manipulation
    time_steps = np.array(time_series)
    concentration = np.array(firm_manager.history_market_concentration[base_params["duration_burn_in"]:])

    # Calculate the number of years
    num_months = len(time_steps)
    num_years = num_months // 12  # Integer division to get the number of full years

    # Reshape the concentration data into years (12 months per year)
    yearly_concentration = concentration[:num_years * 12].reshape(num_years, 12)

    # Calculate yearly averages
    yearly_avg_concentration = np.mean(yearly_concentration, axis=1)

    # Create a list of years for the x-axis
    years = np.arange(1, num_years + 1)  # Year labels (1, 2, 3, ..., num_years)

    # Plot the yearly averages
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, yearly_avg_concentration, marker='o')
    format_plot(ax, "Yearly Average Market Concentration", "Year", "Market Concentration", legend=False)
    save_and_show(fig, fileName, "market_concentration_yearly", dpi)

def plot_battery(base_params, firm_manager,social_network,time_series,  fileName, dpi=600):

    used_history = social_network.history_battery_EV
    prod_history = firm_manager.history_battery_EV
    # Calculate means and confidence intervals
    prod_means = [np.mean(values) if values else np.nan for values in prod_history]
    prod_max = [np.max(values) if values else np.nan for values in prod_history]
    prod_min = [np.min(values) if values else np.nan for values in prod_history]
    prod_confidence_intervals = [1.96 * sem(values) if values else 0 for values in prod_history]

    used_means = [np.mean(values) if values else np.nan for values in used_history]
    used_max = [np.max(values) if values else np.nan for values in used_history]
    used_min = [np.min(values) if values else np.nan for values in used_history]
    used_confidence_intervals = [1.96 * sem(values) if values else 0 for values in used_history]

    eff_used_history = social_network.history_efficiency_EV
    eff_prod_history = firm_manager.history_efficiency_EV

    # Calculate range_prod and range_used
    range_prod = [np.array(battery) * np.array(eff) if battery and eff else np.nan 
                for battery, eff in zip(prod_history, eff_prod_history)]

    range_used = [np.array(battery) * np.array(eff) if battery and eff else np.nan 
                for battery, eff in zip(used_history, eff_used_history)]

    # Calculate means, max, min, and confidence intervals for range_prod
    range_prod_means = [np.mean(values) if not np.isnan(values).all() else np.nan for values in range_prod]
    range_prod_max = [np.max(values) if not np.isnan(values).all() else np.nan for values in range_prod]
    range_prod_min = [np.min(values) if not np.isnan(values).all() else np.nan for values in range_prod]
    range_prod_confidence_intervals = [1.96 * sem(values) if values.size > 0 and not np.isnan(values).all() else 0 for values in range_prod]

    # Calculate means, max, min, and confidence intervals for range_used
    range_used_means = [np.mean(values) if not np.isnan(values).all() else np.nan for values in range_used]
    range_used_max = [np.max(values) if not np.isnan(values).all() else np.nan for values in range_used]
    range_used_min = [np.min(values) if not np.isnan(values).all() else np.nan for values in range_used]
    range_used_confidence_intervals = [1.96 * sem(values) if values.size > 0 and not np.isnan(values).all() else 0 for values in range_used]
    #print(len(ice_means), len(ice_means ), len(ev_means))
    #quit()
    # Plot ICE data
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10, 6))

    ax1 = axes[0][0]
    ax2 = axes[0][1]
    ax3 = axes[1][0]
    ax4 = axes[1][1]
    #prod
    ax1.plot(time_series, prod_means[base_params["duration_burn_in"]:], label=f"Produced", color="blue")
    ax1.plot(time_series, prod_max[base_params["duration_burn_in"]:], color="blue", linestyle = "--")
    ax1.plot(time_series, prod_min[base_params["duration_burn_in"]:], color="blue", linestyle = "--")
    ax1.fill_between(
        time_series,
        (np.array(prod_means) - np.array(prod_confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(prod_means) + np.array(prod_confidence_intervals))[base_params["duration_burn_in"]:],
        color="blue", alpha=0.2
    )
    ax1.set_ylabel("Battery Size (kWhr)")
    ax1.set_xlabel("Time Step")
    ax1.set_title("Production")

    #used
    ax2.plot(time_series, used_means[base_params["duration_burn_in"]:], label=f"Used", color="orange")
    ax2.plot(time_series, used_max[base_params["duration_burn_in"]:], color="orange", linestyle = "--")
    ax2.plot(time_series, used_min[base_params["duration_burn_in"]:], color="orange", linestyle = "--")
    ax2.fill_between(
        time_series,
        (np.array(used_means) - np.array(used_confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(used_means) + np.array(used_confidence_intervals))[base_params["duration_burn_in"]:],
        color="orange", alpha=0.2
    )
    #ax1.set_ylabel("Battery Size (kWhr)")
    ax2.set_xlabel("Time Step")
    ax2.set_title("In use")
    add_vertical_lines(ax1, base_params)
    add_vertical_lines(ax2, base_params)

    #prod
    ax3.plot(time_series, range_prod_means[base_params["duration_burn_in"]:], label=f"Produced", color="blue")
    ax3.plot(time_series, range_prod_max[base_params["duration_burn_in"]:], color="blue", linestyle = "--")
    ax3.plot(time_series, range_prod_min[base_params["duration_burn_in"]:], color="blue", linestyle = "--")
    ax3.fill_between(
        time_series,
        (np.array(range_prod_means) - np.array(range_prod_confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(range_prod_means) + np.array(range_prod_confidence_intervals))[base_params["duration_burn_in"]:],
        color="blue", alpha=0.2
    )
    ax3.set_ylabel("EV Range (km)")
    ax3.set_xlabel("Time Step")
    #ax3.set_title("Production")

    #used
    ax4.plot(time_series, range_used_means[base_params["duration_burn_in"]:], label=f"Used", color="orange")
    ax4.plot(time_series, range_used_max[base_params["duration_burn_in"]:], color="orange", linestyle = "--")
    ax4.plot(time_series, range_used_min[base_params["duration_burn_in"]:], color="orange", linestyle = "--")
    ax4.fill_between(
        time_series,
        (np.array(range_used_means) - np.array(range_used_confidence_intervals))[base_params["duration_burn_in"]:],
        (np.array(range_used_means) + np.array(range_used_confidence_intervals))[base_params["duration_burn_in"]:],
        color="orange", alpha=0.2
    )
    #ax1.set_ylabel("Battery Size (kWhr)")
    ax4.set_xlabel("Time Step")
    #ax4.set_title("In use")
    add_vertical_lines(ax3, base_params)
    add_vertical_lines(ax4, base_params)

    save_and_show(fig, fileName, "battery_evolution", dpi)

def plot_price_history(base_params,firm_manager, time_series, fileName, dpi=600):
    """
    Plots the price history of cars on sale over time.

    Args:
    - firm_manager: An object with `history_cars_on_sale_price` attribute, 
      a list of lists representing car prices at each time step.
    - time_series: A list of time steps corresponding to the data in `history_cars_on_sale_price`.
    - fileName: The name of the file where the plot will be saved.
    - dpi: Dots per inch (resolution) for the saved plot.
    """
    # Flatten the data
    time_points = []
    prices = []
    
    for i, price_list in enumerate(firm_manager.history_cars_on_sale_price[base_params["duration_burn_in"]:]):
        time_points.extend([time_series[i]] * len(price_list))  # Repeat the time step for each price
        prices.extend(price_list)  # Add all prices for the current time step
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time_points, np.asarray(prices)/base_params["computing_coefficient"], marker='o', alpha=0.7)
    ax.set_title("Price History of Cars on Sale")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price, $")
    ax.grid(True)
    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "price_cars_sale", dpi)  

def plot_car_sale_prop(base_params, social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    #print(social_network.history_second_hand_bought)
    #quit()
    ax.plot(time_series, social_network.history_second_hand_bought[base_params["duration_burn_in"]:],label = "second hand",  marker='o')
    ax.plot(time_series, social_network.history_new_car_bought[base_params["duration_burn_in"]:],label = "new cars",  marker='o')
    add_vertical_lines(ax, base_params)
    ax.legend()
    format_plot(ax, "New versus Second hand cars", "Time Step", "# Cars bought", legend=False)
    save_and_show(fig, fileName, "num_cars_bought_type", dpi)      

def plot_distance_individuals_mean_median_type(base_params, social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data for EV and ICE
    data_ev = np.asarray(social_network.history_distance_individual_EV).T
    data_ice = np.asarray(social_network.history_distance_individual_ICE).T

    # Compute statistics for EV
    mean_ev = np.nanmean(data_ev, axis=0)
    median_ev = np.nanmedian(data_ev, axis=0)
    std_error_ev = sem(data_ev, axis=0, nan_policy='omit')
    ci_ev = t.ppf(0.975, df=data_ev.shape[0] - 1) * std_error_ev

    # Compute statistics for ICE
    mean_ice = np.nanmean(data_ice, axis=0)
    median_ice = np.nanmedian(data_ice, axis=0)
    std_error_ice = sem(data_ice, axis=0, nan_policy='omit')
    ci_ice = t.ppf(0.975, df=data_ice.shape[0] - 1) * std_error_ice

    # Plot EV data (green)
    ax.plot(time_series, mean_ev[base_params["duration_burn_in"]:], color='green', linestyle='-', linewidth=2, label='EV Mean Distance')
    ax.plot(time_series, median_ev[base_params["duration_burn_in"]:], color='green', linestyle='--', linewidth=2, label='EV Median Distance')

    # Plot ICE data (blue)
    ax.plot(time_series, mean_ice[base_params["duration_burn_in"]:], color='blue', linestyle='-', linewidth=2, label='ICE Mean Distance')
    ax.plot(time_series, median_ice[base_params["duration_burn_in"]:], color='blue', linestyle='--', linewidth=2, label='ICE Median Distance')

    # Add confidence intervals for EV and ICE
    ax.fill_between(
        time_series,
        (mean_ev - ci_ev)[base_params["duration_burn_in"]:],
        (mean_ev + ci_ev)[base_params["duration_burn_in"]:],
        color='green',
        alpha=0.2,
        label='EV 95% Confidence Interval'
    )
    ax.fill_between(
        time_series,
        (mean_ice - ci_ice)[base_params["duration_burn_in"]:],
        (mean_ice + ci_ice)[base_params["duration_burn_in"]:],
        color='blue',
        alpha=0.2,
        label='ICE 95% Confidence Interval'
    )

    # Add vertical lines for base parameters (e.g., milestones or events)
    add_vertical_lines(ax, base_params)

    # Format plot
    format_plot(ax, "User Distance Over Time", "Time Step", "Individual Distance")
    ax.legend()
    
    # Save and show the plot
    save_and_show(fig, fileName, "user_distance_mean_median_type", dpi)
# Sample main function
def main(fileName, dpi=400):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_controller = load_object(fileName + "/Data", "controller")
    except FileNotFoundError:
        print("Data files not found.")
        return

    social_network = data_controller.social_network
    firm_manager = data_controller.firm_manager
    time_series = np.arange(0, len(data_controller.time_series) - base_params["duration_burn_in"])
    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    plot_preferences(social_network, fileName, dpi)
    #plot_ev_stock(base_params, EV_stock_prop_2010_22, social_network, fileName, dpi)
    plot_ev_consider_adoption_rate(base_params, social_network, time_series, fileName, EV_stock_prop_2010_22, dpi)
    plot_history_prop_EV_research(base_params,firm_manager, fileName)
    plot_market_concentration_yearly(base_params,firm_manager, time_series, fileName, dpi)
    plot_kg_co2_per_year_per_vehicle_by_type(base_params, social_network, time_series, fileName, dpi)
    plot_battery(base_params, firm_manager,social_network,time_series,  fileName, dpi)
    plot_vehicle_attribute_time_series_by_type_split(base_params, social_network, time_series, fileName, dpi)
    plot_prod_vehicle_attribute_time_series_by_type_split(base_params, firm_manager, time_series, fileName, dpi)
    emissions_decomposed(base_params,social_network, time_series, fileName, dpi)
    plot_transport_users_stacked(base_params, social_network, time_series, fileName, dpi)
    #plot_profit_margins_by_type(base_params, firm_manager, time_series,  fileName)
    plot_distance_individuals_mean_median_type(base_params, social_network, time_series, fileName)
    plot_history_count_buy_stacked(base_params, social_network, fileName, dpi)
    plot_total_utility(base_params,social_network, time_series, fileName, dpi)

    
    plot_total_profit(base_params,firm_manager, time_series, fileName, dpi)
    plot_history_car_age(base_params, social_network, time_series,fileName, dpi)
    plot_segment_count_grid(base_params,firm_manager, time_series, fileName)
    plot_car_sale_prop(base_params,social_network, time_series, fileName, dpi)
    plot_history_median_price_by_type(base_params, social_network, fileName, dpi)
    plot_history_mean_price_by_type(base_params, social_network, fileName, dpi)
    plot_history_W(base_params, firm_manager,time_series,  fileName)
    plot_price_history(base_params, firm_manager, time_series, fileName, dpi)
    plot_calibration_data(base_params, data_controller, time_series, fileName, dpi)
    plot_prop_EV_on_sale(base_params,firm_manager, fileName)

    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_09_54_13__27_02_2025")