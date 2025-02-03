from cProfile import label
import os
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import scipy.stats as stats
from package.resources.utility import load_object
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from matplotlib.cm import get_cmap

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

def add_vertical_lines(ax, base_params, color='black', linestyle='--'):
    """
    Adds dashed vertical lines to the plot at specified steps.

    Parameters:
    ax : matplotlib.axes.Axes
        The Axes object to add the lines to.
    base_params : dict
        Dictionary containing 'duration_burn_in' and 'duration_no_carbon_price'.
    color : str, optional
        Color of the dashed lines. Default is 'red'.
    linestyle : str, optional
        Style of the dashed lines. Default is '--'.
    """
    burn_in = base_params["duration_burn_in"]
    no_carbon_price = base_params["duration_no_carbon_price"]
    ev_research_start_time = base_params["ev_research_start_time"]
    ev_production_start_time = base_params["ev_production_start_time"]
    second_hand_burn_in = base_params["parameters_second_hand"]["burn_in_second_hand_market"]
    # Adding the dashed lines
    ax.axvline(second_hand_burn_in, color=color, linestyle='-.', label="Second hand market start")
    ax.axvline(burn_in, color=color, linestyle='--', label="Burn-in period end")
    ax.axvline( burn_in  + ev_research_start_time, color=color, linestyle=':', label="EV research start")
    ax.axvline( burn_in  + ev_production_start_time, color="red", linestyle=':', label="EV sale start")
    
    if base_params["EV_rebate_state"]:
        ax.axvline( burn_in  + base_params["parameters_rebate_calibration"]["start_time"], color="red", linestyle='-.', label="EV adoption subsidy start")
    if base_params["duration_future"] > 0:
        ax.axvline(burn_in + no_carbon_price, color="red", linestyle='--', label="Policy start")
    
# Plot functions with `time_series` where applicable
def plot_emissions(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_driving_emissions, label='Driving Emissions')
    ax.plot(time_series, social_network.history_production_emissions, label='Production Emissions')
    ax.plot(time_series, social_network.history_total_emissions, label='Total Emissions')
    format_plot(ax, "Emissions Over Time", "Time Step", "production_emissions")
    save_and_show(fig, fileName, "production_emissions", dpi)

def plot_total_utility(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_total_utility, marker='o')
    format_plot(ax, "Total Utility Over Time", "Time Step", "Total Utility", legend=False)
    save_and_show(fig, fileName, "total_utility", dpi)

def plot_carbon_price(controller, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    "FIX THIS!!"
    ax.plot(time_series, controller.carbon_price_time_series[:len(time_series)], marker='o')
    format_plot(ax, "Carbon price Over Time", "Time Step", "Carbon price", legend=False)
    save_and_show(fig, fileName, "carbon_price", dpi)

def plot_total_profit(firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, firm_manager.history_total_profit, marker='o')
    format_plot(ax, "Total Profit Over Time", "Time Step", "Total Profit", legend=False)
    save_and_show(fig, fileName, "total_profit", dpi)

def plot_market_concentration(firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, firm_manager.history_market_concentration, marker='o')
    format_plot(ax, "Market concentration Over Time", "Time Step", "Market concentration", legend=False)
    save_and_show(fig, fileName, "market_concentration", dpi)

def plot_total_distance(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_total_distance_driven, marker='o')
    format_plot(ax, "Total Distance Traveled Over Time", "Time Step", "Total Distance Traveled", legend=False)
    save_and_show(fig, fileName, "total_distance", dpi)

def plot_ev_adoption_rate(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_ev_adoption_rate, marker='o')
    format_plot(ax, "EV Adoption Rate Over Time", "Time Step", "EV Adoption Rate", legend=False)
    save_and_show(fig, fileName, "ev_adoption_rate", dpi)

def plot_ev_consider_rate(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_consider_ev_rate, marker='o')
    format_plot(ax, "EV Consideration Rate Over Time", "Time Step", "EV Consider Rate", legend=False)
    save_and_show(fig, fileName, "ev_consider_rate", dpi)

def plot_ev_consider_adoption_rate( base_params,social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_consider_ev_rate, marker='o', label = "Consider")
    ax.plot(time_series, social_network.history_ev_adoption_rate, marker='o', label = "Adopt")
    add_vertical_lines(ax, base_params)
    ax.legend()
    format_plot(ax, "EV Adoption Rate Over Time", "Time Step", "EV Adoption Rate", legend=False)
    save_and_show(fig, fileName, "plot_ev_consider_adoption_rate", dpi)


def plot_tranport_users(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_ICE_users, label='ICE', marker='o')
    ax.plot(time_series, social_network.history_EV_users, label='EV', marker='o')
    format_plot(ax, "Transport Users Over Time", "Time Step", "# Transport Users")
    save_and_show(fig, fileName, "transport_users", dpi)

def plot_transport_users_stacked(base_params,social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate total users at each time step
    total_users = social_network.num_individuals

    # Calculate proportions
    ice_prop = np.array(social_network.history_ICE_users) / total_users
    ev_prop = np.array(social_network.history_EV_users) / total_users

    # Plot stacked area (continuous stacked bar equivalent)
    ax.stackplot(time_series, ice_prop, ev_prop,
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



def plot_transport_new_cars_stacked(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate total users at each time step
    total_users = np.array(social_network.history_new_ICE_cars_bought) + np.array(social_network.history_new_EV_cars_bought)

    # Calculate proportions
    ice_prop = np.array(social_network.history_new_ICE_cars_bought) / total_users
    ev_prop = np.array( social_network.history_new_EV_cars_bought) / total_users

    # Plot stacked area (continuous stacked bar equivalent)
    ax.stackplot(time_series, ice_prop, ev_prop,
                 labels=['ICE', 'EV' ],
                 alpha=0.8)

    # Set plot labels and limits
    ax.set_title("New cars Over Time (Proportion)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Proportion of New Cars")
    ax.set_ylim(0, 1)  # Proportion range
    ax.legend(loc="lower right")

    # Save and show the plot
    save_and_show(fig, fileName, "plot_transport_new_cars_stacked", dpi)

def plot_vehicle_attribute_time_series(social_network, time_series, fileName, dpi=600):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    attributes = {
        "Quality (Quality_a_t)": social_network.history_quality,
        "Efficiency (Eff_omega_a_t)": social_network.history_efficiency,
        "Production Cost (ProdCost_t)": social_network.history_production_cost,
    }
    
    for i, (attribute_name, attribute_history) in enumerate(attributes.items()):
        mean_values = [np.mean(values) for values in attribute_history]
        confidence_intervals = [1.96 * sem(values) for values in attribute_history]
        
        axs[i].plot(time_series, mean_values, label="Mean " + attribute_name, color="blue")
        axs[i].fill_between(
            time_series,
            np.array(mean_values) - np.array(confidence_intervals),
            np.array(mean_values) + np.array(confidence_intervals),
            color="blue", alpha=0.2, label="95% Confidence Interval"
        )
        
        axs[i].set_title(f"{attribute_name} Over Time")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(attribute_name)
    axs[-1].legend()
    
    fig.suptitle("Vehicle Attributes Over Time")
    save_and_show(fig, fileName, "vehicle_attribute_time_series", dpi)

def plot_vehicle_attribute_time_series_by_type(base_params, social_network, time_series, fileName, dpi=600):
    """
    Plots time series of Quality, Efficiency, and Production Cost for both ICE and EV vehicles
    with means and confidence intervals.

    Args:
        social_network (object): Contains the history of vehicle attributes for ICE and EV.
        time_series (list): Time steps to plot on the x-axis.
        file_name (str): Directory or file name to save the plots.
        dpi (int): Resolution for saving the plots.
    """
    # Attributes for ICE and EV
    attributes = {
        "Quality": ("history_quality_ICE", "history_quality_EV"),
        "Efficiency": ("history_efficiency_ICE", "history_efficiency_EV"),
        "Production Cost": ("history_production_cost_ICE", "history_production_cost_EV"),
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (attribute_name, (ice_attr, ev_attr)) in enumerate(attributes.items()):
        ax = axs[i]
        add_vertical_lines(ax, base_params)
        # Extract histories for ICE and EV
        ice_history = getattr(social_network, ice_attr, [])
        ev_history = getattr(social_network, ev_attr, [])

        
        # Calculate means and confidence intervals
        ice_means = [np.mean(values) if values else np.nan for values in ice_history]
        ice_confidence_intervals = [1.96 * sem(values) if values else 0 for values in ice_history]
        
        ev_means = [np.mean(values) if values else np.nan for values in ev_history]
        ev_confidence_intervals = [1.96 * sem(values) if values else 0 for values in ev_history]
    

        # Plot ICE data
        ax.plot(time_series, ice_means, label=f"ICE {attribute_name}", color="blue")
        ax.fill_between(
            time_series,
            np.array(ice_means) - np.array(ice_confidence_intervals),
            np.array(ice_means) + np.array(ice_confidence_intervals),
            color="blue", alpha=0.2
        )
        
        # Plot EV data
        ax.plot(time_series, ev_means, label=f"EV {attribute_name}", color="green")
        ax.fill_between(
            time_series,
            np.array(ev_means) - np.array(ev_confidence_intervals),
            np.array(ev_means) + np.array(ev_confidence_intervals),
            color="green", alpha=0.2
        )

        # Set title and labels
        ax.set_title(f"{attribute_name} Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(attribute_name)
        
        ax.grid()
    axs[-1].legend()
    fig.suptitle("Vehicle Attributes (ICE and EV) Over Time")
    plt.tight_layout()

    # Save and show the plot
    save_and_show(fig, fileName, "vehicle_attribute_time_series_ICE_EV", dpi)

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

    fig.suptitle("Vehicle Attributes (ICE and EV) Over Time")
    plt.tight_layout()

    # Save and show the plot
    save_and_show(fig, fileName, "vehicle_attribute_time_series_ICE_EV", dpi)


def plot_attribute(ax, attribute_name, attr_names, base_params, social_network, time_series):
    """
    Helper function to plot a single attribute (Quality or Production Cost).
    """
    add_vertical_lines(ax, base_params)
    
    # Extract histories for ICE and EV
    ice_attr, ev_attr = attr_names
    ice_history = getattr(social_network, ice_attr, [])
    ev_history = getattr(social_network, ev_attr, [])

    # Calculate means and confidence intervals
    ice_means = [np.mean(values) if values else np.nan for values in ice_history]
    ice_confidence_intervals = [1.96 * sem(values) if values else 0 for values in ice_history]

    ev_means = [np.mean(values) if values else np.nan for values in ev_history]
    ev_confidence_intervals = [1.96 * sem(values) if values else 0 for values in ev_history]

    # Plot ICE data
    ax.plot(time_series, ice_means, label=f"ICE {attribute_name}", color="blue")
    ax.fill_between(
        time_series,
        np.array(ice_means) - np.array(ice_confidence_intervals),
        np.array(ice_means) + np.array(ice_confidence_intervals),
        color="blue", alpha=0.2
    )

    # Plot EV data
    ax.plot(time_series, ev_means, label=f"EV {attribute_name}", color="green")
    ax.fill_between(
        time_series,
        np.array(ev_means) - np.array(ev_confidence_intervals),
        np.array(ev_means) + np.array(ev_confidence_intervals),
        color="green", alpha=0.2
    )

    # Set title and labels
    ax.set_title(f"{attribute_name} Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(attribute_name)
    ax.grid()
    ax.legend()


def plot_single_efficiency(ax, title, attr_name, base_params, social_network, time_series, color):
    """
    Helper function to plot Efficiency for a single vehicle type (ICE or EV).
    """
    add_vertical_lines(ax, base_params)

    history = getattr(social_network, attr_name, [])
    means = [np.mean(values) if values else np.nan for values in history]
    confidence_intervals = [1.96 * sem(values) if values else 0 for values in history]

    # Plot data
    ax.plot(time_series, means, label=title, color=color)
    ax.fill_between(
        time_series,
        np.array(means) - np.array(confidence_intervals),
        np.array(means) + np.array(confidence_intervals),
        color=color, alpha=0.2
    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Efficiency")
    ax.grid()



def plot_research_time_series_multiple_firms(firms, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(12, 8))
    for firm in firms:
        x_vals = [attr[0] for attr in firm.history_attributes_researched]
        y_vals = [attr[1] for attr in firm.history_attributes_researched]
        color_vals = [attr[2] for attr in firm.history_attributes_researched]
        
        ax.scatter(x_vals, y_vals, c=color_vals, cmap='viridis', edgecolor='black', s=50)
    
    ax.set_xlabel("Quality (First Attribute)")
    ax.set_ylabel("Efficiency (Second Attribute)")
    ax.set_title("Research Attributes Over Time for Multiple Firms")
    plt.colorbar(cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=min(color_vals), vmax=max(color_vals))), label="Production Cost")
    save_and_show(fig, fileName, "research_time_series_multiple_firms", dpi)

def plot_scatter_research_time_series_multiple_firms(firms, fileName, dpi=600) -> None:
    """
    Plots scatter plots for research attributes (quality vs. efficiency and quality vs. cost) 
    over time for multiple firms. Color indicates the time step.

    Parameters:
        firms: list
            List of firm objects containing historical research attributes.
        fileName: str
            Directory or file name to save the plots.
        dpi: int
            DPI for the saved plots.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Efficiency vs. Quality
    ax1 = axes[0]
    for firm in firms:
        x_vals = [attr[0] for attr in firm.history_attributes_researched]  # Quality
        y_vals = [attr[1] for attr in firm.history_attributes_researched]  # Efficiency
        time_steps = range(len(firm.history_attributes_researched))  # Time steps
        
        sc = ax1.scatter(x_vals, y_vals, c=time_steps, cmap='viridis', edgecolor='black', s=50)

    ax1.set_xlabel("Quality (First Attribute)")
    ax1.set_ylabel("Efficiency (Second Attribute)")
    ax1.grid()

    # Subplot 2: Cost vs. Quality
    ax2 = axes[1]
    for firm in firms:
        x_vals = [attr[0] for attr in firm.history_attributes_researched]  # Quality
        y_vals = [attr[2] for attr in firm.history_attributes_researched]  # Production Cost
        time_steps = range(len(firm.history_attributes_researched))  # Time steps

        sc = ax2.scatter(x_vals, y_vals, c=time_steps, cmap='viridis', edgecolor='black', s=50)

    ax2.set_xlabel("Quality (First Attribute)")
    ax2.set_ylabel("Production Cost (Third Attribute)")
    ax2.grid()

    fig.suptitle("Evolution of firm research attributes")
    # Add a colorbar for time steps
    cbar = fig.colorbar(
        ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=0, vmax=max(len(firm.history_attributes_researched) for firm in firms))),
        ax=ax2, orientation='vertical', label="Time Step"
    )

    # Save and show the plots
    plt.tight_layout()
    save_and_show(fig, fileName, "research_scatter_time_series_multiple_firms", dpi)


def plot_second_hand_market_len(market, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, market.history_num_second_hand, marker='o')
    format_plot(ax, "Second-Hand Market Over Time", "Time Step", "# Second-Hand Cars", legend=False)
    save_and_show(fig, fileName, "second_hand_market_len", dpi)

def plot_preferences(social_network, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(social_network.beta_vec, bins=30, alpha=0.5, label='Beta Vec (Price)')
    ax.hist(social_network.gamma_vec, bins=30, alpha=0.5, label='Gamma Vec (Environmental)')
    ax.hist(social_network.chi_vec, bins=30, alpha=0.5, label='Chi Vec (EV Threshold)')
    format_plot(ax, "Histogram of Beta, Gamma, and Chi Vectors", "Value", "Frequency")
    save_and_show(fig, fileName, "preferences", dpi)

def plot_preferences_scatter(social_network, fileName, dpi=600):
    """
    Plots a scatter plot of Beta vs. Gamma, 
    with Chi dictating the color of the dots.

    :param social_network: An object containing the following attributes:
                           - beta_vec: array-like (Beta values)
                           - gamma_vec: array-like (Gamma values)
                           - chi_vec: array-like (Chi values)
    :param fileName:      String used for naming the output file
    :param dpi:           Integer specifying the resolution (DPI) 
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a scatter plot, coloring points based on Chi
    sc = ax.scatter(
        social_network.beta_vec, 
        social_network.gamma_vec, 
        c=social_network.chi_vec, 
        cmap='viridis', 
        alpha=0.7
    )
    
    # Add a colorbar to show the range of Chi values
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Chi (EV Threshold)")
    
    # Label axes and set a title
    ax.set_xlabel("Beta (Price)")
    ax.set_ylabel("Gamma (Environmental)")
    ax.set_title("Scatter Plot of Beta vs. Gamma (Color by Chi)")
    
    # If you already have a formatting function for plots, call it here:
    # format_plot(ax, "Scatter Plot of Beta, Gamma, Chi Vectors", "Beta", "Gamma")

    # Save (and/or show) the figure
    # Replace save_and_show with your own function that saves and/or shows the figure
    # For example:
    save_and_show(fig, fileName, "preferences_scatter", dpi)


def plot_sale_EV_prop(firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, firm_manager.history_cars_on_sale_EV_prop, label="EV")
    ax.plot(time_series, firm_manager.history_cars_on_sale_ICE_prop, label="ICE")
    format_plot(ax, "Cars on Sale Over Time", "Time Step", "# Cars on Sale")
    save_and_show(fig, fileName, "sale_EV_prop", dpi)

def plot_history_research_type(firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.asarray([firm.history_research_type for firm in firm_manager.firms_list])
    for firm_data in data:
        ax.scatter(time_series,firm_data)
    format_plot(ax, "EV Research Proportion Over Time", "Time Step", "Proportion Research EV", legend=False)
    save_and_show(fig, fileName, "history_research_type", dpi)

def plot_history_num_cars_on_sale(firm_manager, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.asarray([firm.history_num_cars_on_sale for firm in firm_manager.firms_list])
    for firm_data in data:
        ax.plot(time_series,firm_data, marker='o')
    format_plot(ax, "# cars on_sale", "Time Step", "num cars on sale", legend=False)
    save_and_show(fig, fileName, "num_cars_on_sale", dpi)

def plot_history_attributes_cars_on_sale_all_firms(social_network, time_series, fileName, dpi=600):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    data_EV = social_network.history_attributes_EV_cars_on_sale_all_firms
    data_ICE = social_network.history_attributes_ICE_cars_on_sale_all_firms

    max_len_EV = max(len(step) for step in data_EV)
    max_len_ICE = max(len(step) for step in data_ICE)

    # Pad the data to ensure consistent length
    data_EV_padded = [step + [[np.nan] * 3] * (max_len_EV - len(step)) for step in data_EV]
    data_ICE_padded = [step + [[np.nan] * 3] * (max_len_ICE - len(step)) for step in data_ICE]

    # Convert to arrays
    data_EV_array = np.asarray(data_EV_padded)
    data_ICE_array = np.asarray(data_ICE_padded)

    # Ensure time series length matches the data
    time_series = np.asarray(time_series)

    for i, attribute_name in enumerate(["Quality", "Efficiency", "Production Cost"]):
        # Extract attribute data for EV and ICE
        ev_values = data_EV_array[:, :, i]
        ice_values = data_ICE_array[:, :, i]

        # Ensure consistent shapes for plotting
        for t in range(len(time_series)):
            axs[i].scatter(
                [time_series[t]] * ev_values.shape[1], ev_values[t], color="green", alpha=0.2, s=10, label="EV Data" if t == 0 else ""
            )
            axs[i].scatter(
                [time_series[t]] * ice_values.shape[1], ice_values[t], color="blue", alpha=0.2, s=10, label="ICE Data" if t == 0 else ""
            )
        axs[i].set_title(f"{attribute_name} Over Time")
        axs[i].set_xlabel("Time Steps")
        axs[i].legend()

    fig.suptitle("Attributes of Cars on Sale Over Time")
    save_and_show(fig, fileName, "history_attributes_cars_on_sale_all_firms", dpi)

def history_car_cum_distances(social_network, time_series, fileName, dpi=600):
    cols = 3
    fig, axs = plt.subplots(1, cols, figsize=(15, 6))

    social_network.history_cars_cum_distances_driven
    social_network.history_cars_cum_driven_emissions
    social_network.history_cars_cum_emissions


    max_len_distance = max(len(step) for step in social_network.history_cars_cum_distances_driven)
    max_len_driven_emissions = max(len(step) for step in social_network.history_cars_cum_driven_emissions)
    max_len_emissions = max(len(step) for step in social_network.history_cars_cum_emissions)

    # Pad the data to ensure consistent length
    data_distance_padded = [step + [np.nan] * (max_len_distance - len(step)) for step in social_network.history_cars_cum_distances_driven]
    data_driven_emissions_padded = [step + [np.nan] * (max_len_driven_emissions - len(step)) for step in social_network.history_cars_cum_driven_emissions]
    data_emissions_padded = [step + [np.nan] * (max_len_emissions - len(step)) for step in social_network.history_cars_cum_emissions]
    
    data_distance_array = np.asarray(data_distance_padded)
    data_driven_emissions_array = np.asarray(data_driven_emissions_padded)
    data_emissions_array = np.asarray(data_emissions_padded)

    data_list = [data_distance_array, data_driven_emissions_array, data_emissions_array]
    y_list = ["Distance", "Driven Emissions", "Total Emissions"]
    # Ensure time series length matches the data
    time_series = np.asarray(time_series)

    for i in range(cols):
        for t,time in enumerate(time_series):  # Loop through variables (columns)
            axs[i].scatter([time] * data_list[i].shape[1],  data_list[i][t], alpha=0.2)
        axs[i].set_xlabel("Time Steps")
        axs[i].set_ylabel(y_list[i])

    fig.suptitle("Cars use properties Over Time")
    save_and_show(fig, fileName, "history_car_use", dpi)
    

def plot_history_attributes_cars_on_sale_all_firms_alt(social_network, time_series, fileName, dpi=600):
    """
    Plots history attributes of cars on sale for all firms, separating EV and ICE cars over two rows.
    
    Args:
    - social_network: An object containing history attributes of EV and ICE cars.
    - time_series: A list of time steps corresponding to the data.
    - fileName: The name of the file to save the plot.
    - dpi: Resolution for saving the plot.
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 8), sharex=True)

    # Extract data
    data_EV = social_network.history_attributes_EV_cars_on_sale_all_firms
    data_ICE = social_network.history_attributes_ICE_cars_on_sale_all_firms

    # Calculate the maximum lengths for padding
    max_len_EV = max(len(step) for step in data_EV)
    max_len_ICE = max(len(step) for step in data_ICE)

    # Pad the data to ensure consistent length
    data_EV_padded = [step + [[np.nan] * 3] * (max_len_EV - len(step)) for step in data_EV]
    data_ICE_padded = [step + [[np.nan] * 3] * (max_len_ICE - len(step)) for step in data_ICE]

    # Convert to arrays
    data_EV_array = np.asarray(data_EV_padded)
    data_ICE_array = np.asarray(data_ICE_padded)

    # Ensure time series length matches the data
    time_series = np.asarray(time_series)

    for i, attribute_name in enumerate(["Quality", "Efficiency", "Production Cost"]):
        # EV Data Plot (Top Row)
        ev_values = data_EV_array[:, :, i]
        for t in range(len(time_series)):
            axs[0, i].scatter(
                [time_series[t]] * ev_values.shape[1], ev_values[t], color="green", alpha=0.2, s=10, label="EV Data" if t == 0 else ""
            )
        axs[0, i].set_title(f"EV - {attribute_name} Over Time")
        axs[0, i].set_xlabel("Time Steps")
        axs[0, i].legend()

        # ICE Data Plot (Bottom Row)
        ice_values = data_ICE_array[:, :, i]
        for t in range(len(time_series)):
            axs[1, i].scatter(
                [time_series[t]] * ice_values.shape[1], ice_values[t], color="blue", alpha=0.2, s=10, label="ICE Data" if t == 0 else ""
            )
        axs[1, i].set_title(f"ICE - {attribute_name} Over Time")
        axs[1, i].set_xlabel("Time Steps")
        axs[1, i].legend()

    # Set common titles and layout
    fig.suptitle("Attributes of Cars on Sale Over Time (EV and ICE)", fontsize=16)
    fig.tight_layout(pad=3.0)

    # Save and display the plot
    save_and_show(fig, fileName, "atributeshistory.png", 600) 

def plot_segment_count_grid(firm_manager, time_series, fileName):
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
        ax.plot(time_series, counts, label=f"Segment {segment}")
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

def plot_car_sale_prop(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    #print(social_network.history_second_hand_bought)
    #quit()
    ax.plot(time_series, social_network.history_second_hand_bought,label = "second hand",  marker='o')
    ax.plot(time_series, social_network.history_new_car_bought,label = "new cars",  marker='o')
    ax.legend()
    format_plot(ax, "New versus Second hand cars", "Time Step", "# Cars bought", legend=False)
    save_and_show(fig, fileName, "num_cars_bought_type", dpi)      

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
    
    for i, price_list in enumerate(firm_manager.history_cars_on_sale_price):
        time_points.extend([time_series[i]] * len(price_list))  # Repeat the time step for each price
        prices.extend(price_list)  # Add all prices for the current time step
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time_points, prices, marker='o', alpha=0.7)
    ax.set_title("Price History of Cars on Sale")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "price_cars_sale", dpi)   

def plot_price_history_new_second_hand(base_params,social_network, time_series, fileName, dpi=600):
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

    time_points_new = []
    prices_new = []

    time_points_second_hand= []
    prices_second_hand = []
    
    for i, price_list in enumerate(social_network.history_car_prices_sold_new):
        time_points_new .extend([time_series[i]] * len(price_list))  # Repeat the time step for each price
        prices_new.extend(price_list)  # Add all prices for the current time step
    
    for i, price_list in enumerate(social_network.history_car_prices_sold_second_hand):
        time_points_second_hand.extend([time_series[i]] * len(price_list))  # Repeat the time step for each price
        prices_second_hand.extend(price_list)  # Add all prices for the current time step
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time_points_new, prices_new, marker='o', alpha=0.7, label = "New")
    ax.scatter(time_points_second_hand, prices_second_hand, marker='o', alpha=0.7, label = "Second hand")
    ax.set_title("Price History of Cars Sold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "price_cars_sale_by type", dpi)   


    
def plot_history_car_age_scatter(social_network, time_series, fileName, dpi):
    """
    Plots individual car ages as scatter points over time.
    
    Args:
    - social_network: An object containing `history_car_age` (a list of lists of car ages at each time step).
    - time_series: A list of time steps.
    - fileName: The name of the file where the plot will be saved.
    - dpi: Resolution for the saved plot.
    """
    # Extract car ages history
    ages_list = social_network.history_car_age

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    for t, ages in zip(time_series, ages_list):
        ages = np.array(ages)
        valid_ages = ages[~np.isnan(ages)]  # Exclude NaNs from the scatter plot
        ax.scatter([t] * len(valid_ages), valid_ages, alpha=0.6, label="Ages" if t == time_series[0] else None)

    ax.set_title("Car Ages Over Time (Scatter Plot)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Car Age")
    ax.grid(True)
    ax.legend(["Individual Ages"], loc="upper right")

    # Save and show the plot
    save_and_show(fig, fileName, "Car Age Over Time (Scatter)", dpi)

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
    ax.plot(time_series, medians, label="Median Age", color='red')
    ax.plot(time_series, means, label="Mean Age", color='blue')
    ax.fill_between(time_series, lower_bounds, upper_bounds, color='blue', alpha=0.2, label="95% Confidence Interval")
    ax.set_title("Mean Age and 95% Confidence Interval Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Age")

    ax.grid(True)
    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "car age owned", dpi)   

def plot_transport_composition_segments(firm_manager, time_series, fileName):
    """
    Plots the transport composition (ICE and EV proportions) for each segment over time.
    
    Parameters:
        firm_manager: Firm_Manager
            The firm manager object containing the historical market data.
        time_series: list or array
            The time steps to plot on the x-axis.
        fileName: str
            Directory or file name to save the plot.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 12), sharex=True, sharey=True)

    # Limit to the first 8 segments (assuming binary codes from "000" to "111")
    segment_codes = [format(i, '03b') for i in range(8)]

    for i, segment_code in enumerate(segment_codes):
        row, col = divmod(i, 2)
        ax = axes[row, col]

        # Extract ICE and EV counts over time for the current segment
        ice_counts = np.array([data[segment_code]["ICE"] for data in firm_manager.history_market_data])
        ev_counts = np.array([data[segment_code]["EV"] for data in firm_manager.history_market_data])
        total_counts = ice_counts + ev_counts

        # Avoid division by zero
        ice_prop = np.divide(ice_counts, total_counts, where=total_counts > 0)
        ev_prop = np.divide(ev_counts, total_counts, where=total_counts > 0)

        # Plot stacked proportions
        ax.stackplot(time_series, ice_prop, ev_prop, labels=['ICE', 'EV'], alpha=0.8)
        ax.set_title(f"Segment {segment_code}")
        ax.grid()
    

    fig.supxlabel("Time Step")
    fig.supylabel("Proportion of Transport Type")

    # Add a single legend outside the grid
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Firms", loc="upper center", bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    save_and_show(fig, fileName, "transport_composition_segments.png", 600)

def plot_u_sum_segments(firm_manager, time_series, fileName):
    """
    Plots the market concentration (HHI) per segment across time steps.
    
    Parameters:
        firm_manager: Firm_Manager
            The firm manager object containing historical market data.
        time_series: list or array
            The time steps to plot on the x-axis.
        fileName: str
            Directory or file name to save the plot.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 12), sharex=True, sharey=True)

    segment_codes = [format(i, '03b') for i in range(8)]  # Binary segment codes

    for i, segment_code in enumerate(segment_codes):
        row, col = divmod(i, 2)
        ax = axes[row, col]

        # Extract market concentration (U_sum or HHI) for the segment across all time steps
        hhi_data = [data[segment_code]["U_sum"] for data in firm_manager.history_market_data]

        ax.plot(time_series, hhi_data, label=f"Segment {segment_code}", color="purple")
        ax.set_title(f"Segment {segment_code}")
        ax.grid()

    fig.supxlabel("Time Step")
    fig.supylabel("U sum")
    plt.tight_layout()
    save_and_show(fig, fileName, "u_sum_segments.png", 600)



def plot_total_utility_vs_total_profit(social_network, firm_manager, time_steps, file_name, dpi=600):
    """
    Plots a scatter plot of Total Utility vs Total Profit with time as a color bar.

    Args:
        social_network (object): Contains the history of total utility.
        firm_manager (object): Contains the history of total profits.
        time_steps (list): Time steps corresponding to the data points.
        file_name (str): Directory or file name to save the plot.
        dpi (int): Resolution for saving the plot.
    """
    # Extract data
    total_utility = social_network.history_total_utility
    total_profit = firm_manager.history_total_profit

    # Normalize the time values for the color map
    norm = Normalize(vmin=min(time_steps), vmax=max(time_steps))
    cmap = cm.viridis

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        total_profit,
        total_utility,
        c=time_steps,
        cmap=cmap,
        edgecolor='k',
        alpha=0.7
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Step", rotation=270, labelpad=15)

    # Add labels and title
    ax.set_title("Total Utility vs Total Profit Over Time")
    ax.set_xlabel("Total Profit")
    ax.set_ylabel("Total Utility")
    ax.grid(True)

    # Save and show the plot
    save_path = f"{file_name}/total_utility_vs_total_profit.png"
    fig.savefig(save_path, dpi=dpi, format="png")

def plot_calibration_data(controller, time_series, fileName, dpi=600):
    fig, axes = plt.subplots(ncols = 3,figsize=(10, 6))
    #print(social_network.history_second_hand_bought)
    #quit()

    axes[0].plot(controller.gas_price_california_vec)
    axes[1].plot(controller.electricity_price_vec)
    axes[2].plot(controller.electricity_emissions_intensity_vec)

    axes[0].set_ylabel("Gas price california in 2020 Dollars")
    axes[1].set_ylabel("Electricity price california in 2020 Dollars")
    axes[2].set_ylabel("Urban Electricity emissions intensity in kgCO2/kWhr")
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_calibration_data", dpi)   

#PLOT INDIVIDUALS LEVEL DATA:

def plot_emissions_individuals(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray(social_network.history_driving_emissions_individual).T
    for i, time in enumerate(time_series):
        ax.plot(time_series, data[i])

    format_plot(ax, "User emissions Over Time", "Time Step", "driving_emissions")
    save_and_show(fig, fileName, "production_emissions", dpi)

def plot_distance_individuals(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray(social_network.history_distance_individual).T
    for i, time in enumerate(time_series):
        if i == 100:
            break
        ax.plot(time_series, data[i])

    format_plot(ax, "User distance Over Time", "Time Step", "indivudal_distance")
    save_and_show(fig, fileName, "user_distance", dpi)

def plot_distance_individuals_mean_median(base_params, social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data and compute statistics
    data = np.asarray(social_network.history_distance_individual).T
    mean_distance = np.mean(data, axis=0)
    standard_error = sem(data, axis=0)
    confidence_interval = t.ppf(0.975, df=data.shape[0] - 1) * standard_error
    
    median_distance = np.median(data, axis=0)
    # Plot mean and confidence interval
    ax.plot(time_series, median_distance, color='red', label='Mediann Distance', linewidth=2)

    ax.plot(time_series, mean_distance, color='blue', label='Mean Distance', linewidth=2)
    ax.fill_between(
        time_series, 
        mean_distance - confidence_interval, 
        mean_distance + confidence_interval, 
        color='blue', 
        alpha=0.2, 
        label='95% Confidence Interval'
    )
    add_vertical_lines(ax, base_params)

    # Format and save plot
    format_plot(ax, "User Distance Over Time", "Time Step", "Individual Distance")
    ax.legend()
    save_and_show(fig, fileName, "user_distance_mean_median", dpi)

def plot_distance_individuals_mean_median_type(base_params, social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data and compute statistics
    data = np.asarray(social_network.history_distance_individual).T
    mean_distance = np.mean(data, axis=0)
    standard_error = sem(data, axis=0)
    confidence_interval = t.ppf(0.975, df=data.shape[0] - 1) * standard_error
    
    median_distance = np.median(data, axis=0)
    # Plot mean and confidence interval
    ax.plot(time_series, median_distance, color='red', label='Mediann Distance', linewidth=2)

    ax.plot(time_series, mean_distance, color='blue', label='Mean Distance', linewidth=2)
    ax.fill_between(
        time_series, 
        mean_distance - confidence_interval, 
        mean_distance + confidence_interval, 
        color='blue', 
        alpha=0.2, 
        label='95% Confidence Interval'
    )
    add_vertical_lines(ax, base_params)

    # Format and save plot
    format_plot(ax, "User Distance Over Time", "Time Step", "Individual Distance")
    ax.legend()
    save_and_show(fig, fileName, "user_distance_mean_median", dpi)

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
    ax.plot(time_series, mean_ev, color='green', linestyle='-', linewidth=2, label='EV Mean Distance')
    ax.plot(time_series, median_ev, color='green', linestyle='--', linewidth=2, label='EV Median Distance')

    # Plot ICE data (blue)
    ax.plot(time_series, mean_ice, color='blue', linestyle='-', linewidth=2, label='ICE Mean Distance')
    ax.plot(time_series, median_ice, color='blue', linestyle='--', linewidth=2, label='ICE Median Distance')

    # Add confidence intervals for EV and ICE
    ax.fill_between(
        time_series,
        mean_ev - ci_ev,
        mean_ev + ci_ev,
        color='green',
        alpha=0.2,
        label='EV 95% Confidence Interval'
    )
    ax.fill_between(
        time_series,
        mean_ice - ci_ice,
        mean_ice + ci_ice,
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


def plot_utility_individuals(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray(social_network.history_utility_individual).T
    for i, time in enumerate(time_series):
        ax.plot(time_series, data[i])

    format_plot(ax, "User utility Over Time", "Time Step", "indivudal_utility")
    save_and_show(fig, fileName, "user_utility", dpi)

def plot_transport_type_individuals(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray(social_network.history_transport_type_individual).T
    for i, time in enumerate(time_series):
        ax.plot(time_series, data[i])

    format_plot(ax, "User transport type Over Time", "Time Step", "indivudal_transport_type")
    save_and_show(fig, fileName, "user_transport_type", dpi)

def plot_density_by_value(fileName, social_network, time_series, dpi_save=600):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    
    emissions_data = np.asarray(social_network.history_driving_emissions_individual).T
    distance_data = np.asarray(social_network.history_distance_individual).T 
    utility_data = np.asarray(social_network.history_utility_individual).T 

    # Custom colormap (White to Blue)
    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    cmap_name = 'white_to_blue'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)


    
    # Helper function to plot density
    def plot_density(ax, data, title, xlabel, ylabel):
        histograms = []
        
        # Fixed number of bins
        n_bins = 30
        bins = np.linspace(np.nanmin(data), np.nanmax(data), n_bins)  # Adjust range (0, 100) as needed for your data

        for t in range(data.shape[1]):  # Iterate over time steps
            hist, _ = np.histogram(data[:, t], bins=bins, density=True)
            histograms.append(hist)
        
        histograms = np.array(histograms).T
        T, B = np.meshgrid(time_series, bins[:-1])  # Create meshgrid for plotting
        pcm = ax.pcolormesh(T, B, histograms, cmap=custom_cmap, shading='auto')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return pcm
    
    # Plot emissions density
    pcm = plot_density(axs[0], emissions_data, "Emissions Density Over Time", "Time Step", "Emission Value")
    fig.colorbar(pcm, ax=axs[0], label='Density')
    
    # Plot distance density
    pcm = plot_density(axs[1], distance_data, "Distance Density Over Time", "Time Step", "Distance Value")
    fig.colorbar(pcm, ax=axs[1], label='Density')
    
    # Plot utility density
    pcm = plot_density(axs[2], utility_data, "Utility Density Over Time", "Time Step", "Utility Value")
    fig.colorbar(pcm, ax=axs[2], label='Density')
    
    plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/value_density_plot_grid"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

#################################################################################################################################################
#Plots by segments of society

# Compute proportions for rich and poor
def compute_proportions(indices, data):
    num_users = len(indices)
    transport_data = data[indices]  # Select data for the given group
    
    # Calculate proportions for each transport type at each time step
    ice_prop = np.sum(transport_data == 2, axis=0) / num_users
    ev_prop = np.sum(transport_data == 3, axis=0) / num_users
    
    return ice_prop, ev_prop
    
def plot_transport_users_stacked_rich_poor(social_network, time_series, fileName, x_percentile=50, dpi=600):
    # Calculate the threshold for rich and poor
    beta_threshold = np.percentile(social_network.beta_vec, x_percentile)
    
    # Separate indices for rich and poor
    rich_indices = np.where(social_network.beta_vec <= beta_threshold)[0]
    poor_indices = np.where(social_network.beta_vec > beta_threshold)[0]
    
    # Extract transport type data
    data = np.asarray(social_network.history_transport_type_individual).T

    # Get proportions for rich and poor
    rich_ice, rich_ev = compute_proportions(rich_indices, data)
    poor_ice, poor_ev = compute_proportions(poor_indices, data)
    
    # Create subplots for rich and poor
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

    # Plot stacked areas for rich individuals
    axs[0].stackplot(time_series, rich_ice, rich_ev,
                     labels=['ICE', 'EV'],
                     alpha=0.8)
    axs[0].set_title("Transport Users Over Time (Rich Proportion)")
    axs[0].set_ylabel("Proportion")
    axs[0].legend(loc="upper left")

    # Plot stacked areas for poor individuals
    axs[1].stackplot(time_series, poor_ice, poor_ev,
                     labels=['ICE', 'EV'],
                     alpha=0.8)
    axs[1].set_title("Transport Users Over Time (Poor Proportion)")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Proportion")
    axs[1].legend(loc="upper left")
    
    # Set y-axis limits
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)

    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, "transport_users_stacked_rich_poor", dpi)

def plot_transport_users_stacked_two_by_four(base_params,social_network, time_series, fileName, percentiles, dpi=600):
    """
    Plots transport user proportions in a 2x4 layout: each column represents a vector (Beta, Gamma, Chi, Origin),
    with rich in the upper row and poor in the lower row.

    Args:
        social_network: The social network object containing transport data and vectors.
        time_series: Array of time steps.
        fileName: Output directory for saving plots.
        percentiles: Dictionary with vector names as keys and percentiles as values.
        dpi: Resolution of the saved plot.
    """
    # Vectors to process
    vectors = {
        'Beta': social_network.beta_vec,
        'Gamma': social_network.gamma_vec,
        'Chi': social_network.chi_vec
    }
    
    # Extract transport type data
    data = np.asarray(social_network.history_transport_type_individual).T

    # Create a grid of subplots (2 rows x 4 columns)
    fig, axs = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True)
    
    # Iterate through general vectors and plot
    for i, (label, vec) in enumerate(vectors.items()):
        # Get the percentile threshold for this vector
        threshold = np.percentile(vec, percentiles[label])
        rich_indices = np.where(vec <= threshold)[0]
        poor_indices = np.where(vec > threshold)[0]
        
        # Compute proportions
        rich_ice, rich_ev = compute_proportions(rich_indices, data)
        poor_ice, poor_ev = compute_proportions(poor_indices, data)
        
        # Plot stacked areas for rich in the upper row
        axs[0, i].stackplot(time_series, rich_ice, rich_ev,
                            labels=['ICE', 'EV'],
                            alpha=0.8)
        axs[0, i].set_title(f"{label} -Values Below {percentiles[label]}% - {round(threshold,2)}")
        #axs[0, i].set_ylabel("Proportion")
        #axs[0, i].legend(loc="upper left")
        
        # Plot stacked areas for poor in the lower row
        axs[1, i].stackplot(time_series, poor_ice, poor_ev,
                            labels=['ICE', 'EV'],
                            alpha=0.8)
        axs[1, i].set_title(f"{label} -Values Above {percentiles[label]}% - {round(threshold,2)}")
        #axs[1, i].set_ylabel("Proportion")
        #axs[1, i].legend(loc="upper left")
        add_vertical_lines(axs[0, i], base_params)
        add_vertical_lines(axs[1, i], base_params)
    
    fig.supylabel("Proportion")
    fig.supxlabel("Time Step")
    
    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, "transport_users_stacked_two_by_four", dpi)

def plot_mean_emissions_one_row(base_params, social_network, time_series, fileName, percentiles, dpi=600):
    """
    Plots mean emissions in a single row layout: each column represents a vector (Beta, Gamma, Chi, Origin),
    with rich and poor plotted together and distinguished by a legend.

    Args:
        social_network: The social network object containing emissions data and vectors.
        time_series: Array of time steps.
        fileName: Output directory for saving plots.
        percentiles: Dictionary with vector names as keys and percentiles as values.
        dpi: Resolution of the saved plot.
    """
    # Vectors to process
    vectors = {
        'Beta': social_network.beta_vec,
        'Gamma': social_network.gamma_vec,
        'Chi': social_network.chi_vec
    }
    
    # Extract emissions data
    emissions_data = np.asarray(social_network.history_driving_emissions_individual).T

    # Compute mean and 95% CI for a general case
    def compute_mean_and_ci(indices):
        selected_data = emissions_data[indices]
        mean = np.mean(selected_data, axis=0)
        se = stats.sem(selected_data, axis=0)  # Standard error
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(indices) - 1)  # 95% CI
        return mean, ci

    # Create a single row of subplots (1 row x 4 columns)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
    
    # Iterate through general vectors and plot
    for i, (label, vec) in enumerate(vectors.items()):
        # Get the percentile threshold for this vector
        threshold = np.percentile(vec, percentiles[label])
        rich_indices = np.where(vec <= threshold)[0]
        poor_indices = np.where(vec > threshold)[0]
        
        # Compute mean and CI
        rich_mean, rich_ci = compute_mean_and_ci(rich_indices)
        poor_mean, poor_ci = compute_mean_and_ci(poor_indices)
        
        # Plot mean and CI for rich
        axs[i].plot(time_series, rich_mean, label=f"{label} -Values Below {percentiles[label]}% - {round(threshold,2)}", color="blue")
        axs[i].fill_between(time_series, rich_mean - rich_ci, rich_mean + rich_ci, color="blue", alpha=0.2)
        
        # Plot mean and CI for poor
        axs[i].plot(time_series, poor_mean, label=f"{label} -Values Above {percentiles[label]}% - {round(threshold,2)}", color="red")
        axs[i].fill_between(time_series, poor_mean - poor_ci, poor_mean + poor_ci, color="red", alpha=0.2)
        
        axs[i].set_title(label)
        axs[i].legend(loc="upper left")
        add_vertical_lines(axs[i], base_params)
    
    
    fig.supylabel("Driving Emissions")
    fig.supxlabel("Time Step")
    
    # Adjust layout and save
    #plt.tight_layout()
    save_and_show(fig, fileName, "mean_driving_emissions_one_row", dpi)

def plot_mean_distance_one_row(base_params, social_network, time_series, fileName, percentiles, dpi=600):
    """
    Plots mean distance in a single row layout: each column represents a vector (Beta, Gamma, Chi, Origin),
    with rich and poor plotted together and distinguished by a legend.

    Args:
        social_network: The social network object containing distance data and vectors.
        time_series: Array of time steps.
        fileName: Output directory for saving plots.
        percentiles: Dictionary with vector names as keys and percentiles as values.
        dpi: Resolution of the saved plot.
    """
    # Vectors to process
    vectors = {
        'Beta': social_network.beta_vec,
        'Gamma': social_network.gamma_vec,
        'Chi': social_network.chi_vec
    }
    
    # Extract distance data
    distance_data = np.asarray(social_network.history_distance_individual).T

    # Compute mean and 95% CI
    def compute_mean_and_ci(indices):
        selected_data = distance_data[indices]
        mean = np.mean(selected_data, axis=0)
        se = stats.sem(selected_data, axis=0)  # Standard error
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(indices) - 1)  # 95% CI
        return mean, ci

    # Create a single row of subplots (1 row x 4 columns)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
    
    # Iterate through general vectors and plot
    for i, (label, vec) in enumerate(vectors.items()):
        # Get the percentile threshold for this vector
        threshold = np.percentile(vec, percentiles[label])
        rich_indices = np.where(vec <= threshold)[0]
        poor_indices = np.where(vec > threshold)[0]
        
        # Compute mean and CI
        rich_mean, rich_ci = compute_mean_and_ci(rich_indices)
        poor_mean, poor_ci = compute_mean_and_ci(poor_indices)
        
        # Plot mean and CI for rich
        axs[i].plot(time_series, rich_mean, label=f"{label} -Values Below {percentiles[label]}% - {round(threshold,2)}", color="blue")
        axs[i].fill_between(time_series, rich_mean - rich_ci, rich_mean + rich_ci, color="blue", alpha=0.2)
        
        # Plot mean and CI for poor
        axs[i].plot(time_series, poor_mean, label=f"{label} -Values Above {percentiles[label]}% - {round(threshold,2)}", color="red")
        axs[i].fill_between(time_series, poor_mean - poor_ci, poor_mean + poor_ci, color="red", alpha=0.2)
        
        axs[i].set_title(label)
        axs[i].legend(loc="upper left")
        add_vertical_lines(axs[i], base_params)
    
    fig.supylabel("Mean Distance")
    fig.supxlabel("Time Step")
    
    # Adjust layout and save
    #plt.tight_layout()
    save_and_show(fig, fileName, "mean_distance_one_row", dpi)

def plot_mean_utility_one_row(base_params, social_network, time_series, fileName, percentiles, dpi=600):
    """
    Plots mean utility in a single row layout: each column represents a vector (Beta, Gamma, Chi, Origin),
    with rich and poor plotted together and distinguished by a legend.

    Args:
        social_network: The social network object containing utility data and vectors.
        time_series: Array of time steps.
        fileName: Output directory for saving plots.
        percentiles: Dictionary with vector names as keys and percentiles as values.
        dpi: Resolution of the saved plot.
    """
    # Vectors to process
    vectors = {
        'Beta': social_network.beta_vec,
        'Gamma': social_network.gamma_vec,
        'Chi': social_network.chi_vec
    }
    
    # Extract utility data
    utility_data = np.asarray(social_network.history_utility_individual).T

    # Compute mean and 95% CI
    def compute_mean_and_ci(indices):
        selected_data = utility_data[indices]
        mean = np.mean(selected_data, axis=0)
        se = stats.sem(selected_data, axis=0)  # Standard error
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(indices) - 1)  # 95% CI
        return mean, ci

    # Create a single row of subplots (1 row x 4 columns)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
    
    # Iterate through general vectors and plot
    for i, (label, vec) in enumerate(vectors.items()):
        # Get the percentile threshold for this vector
        threshold = np.percentile(vec, percentiles[label])
        rich_indices = np.where(vec <= threshold)[0]
        poor_indices = np.where(vec > threshold)[0]
        
        # Compute mean and CI
        rich_mean, rich_ci = compute_mean_and_ci(rich_indices)
        poor_mean, poor_ci = compute_mean_and_ci(poor_indices)
        
        # Plot mean and CI for rich
        axs[i].plot(time_series, rich_mean, label=f"{label} -Values Below {percentiles[label]}% - {round(threshold,2)}", color="blue")
        axs[i].fill_between(time_series, rich_mean - rich_ci, rich_mean + rich_ci, color="blue", alpha=0.2)
        
        # Plot mean and CI for poor
        axs[i].plot(time_series, poor_mean, label=f"{label} -Values Above {percentiles[label]}% - {round(threshold,2)}", color="red")
        axs[i].fill_between(time_series, poor_mean - poor_ci, poor_mean + poor_ci, color="red", alpha=0.2)
        
        axs[i].set_title(label)
        axs[i].legend(loc="upper left")
        add_vertical_lines(axs[i], base_params)
    
    fig.supylabel("Mean Utility")
    fig.supxlabel("Time Step")
    
    # Adjust layout and save
    #plt.tight_layout()
    save_and_show(fig, fileName, "mean_utility_one_row", dpi)

def plot_conditional_transport_users_4x4(
    base_params, social_network, time_series, fileName, percentiles, dpi=600
):
    """
    Plots transport user proportions for combinations of conditions in a 4x4 grid with condition-based titles.

    Args:
        social_network: The social network object containing transport data and vectors.
        time_series: Array of time steps.
        fileName: Output directory for saving plots.
        percentiles: Dictionary with vector names as keys and percentile thresholds as values.
        dpi: Resolution of the saved plot.
    """
    # Vectors to process
    vectors = {
        'Beta': social_network.beta_vec,
        'Gamma': social_network.gamma_vec,
        'Chi': social_network.chi_vec
    }
    
    # Extract transport type data
    data = np.asarray(social_network.history_transport_type_individual).T

    # Define combinations for the 4 variables
    condition_combinations = [
        (beta_cond, gamma_cond, chi_cond)
        for beta_cond in ['low', 'high']
        for gamma_cond in ['low', 'high']
        for chi_cond in ['low', 'high']
    ]

    # Create a grid of subplots (4x4 layout)
    fig, axs = plt.subplots(2, 4, figsize=(12, 12), sharex=True, sharey=True)
    for i,test in  enumerate(condition_combinations):
        print(i,test)
   
    for i, (beta_cond, gamma_cond, chi_cond) in enumerate(condition_combinations):
        print(i)

        # Compute indices for the current condition combination
        beta_threshold = np.percentile(vectors['Beta'], percentiles['Beta'])
        gamma_threshold = np.percentile(vectors['Gamma'], percentiles['Gamma'])
        chi_threshold = np.percentile(vectors['Chi'], percentiles['Chi'])
        
        beta_indices = (
            np.where(vectors['Beta'] <= beta_threshold)[0]
            if beta_cond == 'low'
            else np.where(vectors['Beta'] > beta_threshold)[0]
        )
        gamma_indices = (
            np.where(vectors['Gamma'] <= gamma_threshold)[0]
            if gamma_cond == 'low'
            else np.where(vectors['Gamma'] > gamma_threshold)[0]
        )
        chi_indices = (
            np.where(vectors['Chi'] <= chi_threshold)[0]
            if chi_cond == 'low'
            else np.where(vectors['Chi'] > chi_threshold)[0]
        )
        
        # Combine all indices (intersection of conditions)
        combined_indices = np.intersect1d(
            np.intersect1d(beta_indices, gamma_indices),
            chi_indices
        )

        #print("SEGEMENT COUNT", len(combined_indices))

        row, col = divmod(i, 2)
        ax = axs[row, col]

        # Compute proportions for the combined indices
        if len(combined_indices) > 0:
            ice_prop, ev_prop = compute_proportions(combined_indices, data)

            # Plot the data
            ax.stackplot(time_series, ice_prop, ev_prop,
                         labels=['ICE', 'EV'],
                         alpha=0.8)
        
        # Generate the condition-based title
        title = (
            f"Pop: { len(combined_indices)}, $\\beta$: {'B' if beta_cond == 'low' else 'T'}{percentiles['Beta']}%, "
            f"$\\gamma$: {'B' if gamma_cond == 'low' else 'T'}{percentiles['Gamma']}%, "
            f"$\\chi$: {'B' if chi_cond == 'low' else 'T'}{percentiles['Chi']}%, "
        )
        ax.set_title(title, fontsize=8)
        ax.grid()
        add_vertical_lines(ax, base_params)

    # Add shared labels
    fig.supylabel("Proportion")
    
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize="9", frameon=False)
    #fig.subplots_adjust(top=0.1)  # Adjust bottom margin for legend
    fig.supxlabel("Time Step")
    # Adjust layout and save

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for the legend
    save_and_show(fig, fileName, "conditional_transport_users_4x4_conditions", dpi)

def plot_transport_users_beta_gamma_chi(
    firm_manager,
    base_params,
    social_network,
    time_series,
    fileName,
    percentiles,
    dpi=600
):
    """
    Create two separate plots:
    1. Beta bins: 5 subplots across one row (Beta Bin 0-4).
    2. Gamma and Chi: 4 subplots in two rows (Rich and Poor for Gamma and Chi).
    """

    # 1) Prepare data (ICE vs. EV) for each time step, shape = (num_individuals, num_time_steps).
    data = np.asarray(social_network.history_transport_type_individual).T

    def compute_proportions(indices, data_array):
        """
        Returns two arrays: proportion_ICE, proportion_EV
        Each array is length = len(time_series)
        """
        sub_data = data_array[indices, :]  # shape = (#selected, time_steps)

        ice_counts = np.sum(sub_data == 2, axis=0)
        ev_counts = np.sum(sub_data == 3, axis=0)
        total = len(indices)

        if total == 0:
            return np.zeros(sub_data.shape[1]), np.zeros(sub_data.shape[1])

        prop_ice = ice_counts / total
        prop_ev = ev_counts / total
        return prop_ice, prop_ev

    # Beta bins
    beta_bin_idx = firm_manager.beta_segment_idx

    # Gamma and Chi thresholds
    gamma_threshold = np.percentile(social_network.gamma_vec, percentiles["Gamma"])
    chi_threshold = np.percentile(social_network.chi_vec, percentiles["Chi"])

    gamma_rich_indices = np.where(social_network.gamma_vec <= gamma_threshold)[0]
    gamma_poor_indices = np.where(social_network.gamma_vec > gamma_threshold)[0]
    chi_rich_indices = np.where(social_network.chi_vec <= chi_threshold)[0]
    chi_poor_indices = np.where(social_network.chi_vec > chi_threshold)[0]

    # --------------------------------------------------------------------------
    # Plot 1: Beta bins (5 subplots in one row)
    # --------------------------------------------------------------------------
    fig1, axs1 = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)

    for bin_val in range(5):
        bin_indices = np.where(beta_bin_idx == bin_val)[0]
        bin_ice, bin_ev = compute_proportions(bin_indices, data)

        axs1[bin_val].stackplot(time_series, bin_ice, bin_ev, labels=['ICE', 'EV'], alpha=0.8)
        axs1[bin_val].set_title(f"Beta Bin {bin_val}")
        add_vertical_lines(axs1[bin_val], base_params)

    fig1.supylabel("Proportion")
    fig1.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig1, fileName, "transport_users_beta_bins", dpi)

    # --------------------------------------------------------------------------
    # Plot 2: Gamma and Chi (4 subplots in two rows)
    # --------------------------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    # Gamma Rich
    gamma_rich_ice, gamma_rich_ev = compute_proportions(gamma_rich_indices, data)
    axs2[0, 0].stackplot(time_series, gamma_rich_ice, gamma_rich_ev, labels=['ICE', 'EV'], alpha=0.8)
    axs2[0, 0].set_title(f"Gamma <= {round(gamma_threshold,2)}")
    add_vertical_lines(axs2[0, 0], base_params)

    # Gamma Poor
    gamma_poor_ice, gamma_poor_ev = compute_proportions(gamma_poor_indices, data)
    axs2[1, 0].stackplot(time_series, gamma_poor_ice, gamma_poor_ev, labels=['ICE', 'EV'], alpha=0.8)
    axs2[1, 0].set_title(f"Gamma > {round(gamma_threshold,2)}")
    add_vertical_lines(axs2[1, 0], base_params)

    # Chi Rich
    chi_rich_ice, chi_rich_ev = compute_proportions(chi_rich_indices, data)
    axs2[0, 1].stackplot(time_series, chi_rich_ice, chi_rich_ev, labels=['ICE', 'EV'], alpha=0.8)
    axs2[0, 1].set_title(f"Chi <= {round(chi_threshold,2)}")
    add_vertical_lines(axs2[0, 1], base_params)

    # Chi Poor
    chi_poor_ice, chi_poor_ev = compute_proportions(chi_poor_indices, data)
    axs2[1, 1].stackplot(time_series, chi_poor_ice, chi_poor_ev, labels=['ICE', 'EV'], alpha=0.8)
    axs2[1, 1].set_title(f"Chi > {round(chi_threshold,2)}")
    add_vertical_lines(axs2[1, 1], base_params)

    fig2.supylabel("Proportion")
    fig2.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig2, fileName, "transport_users_gamma_chi", dpi)


def plot_mean_emissions_beta_gamma_chi_split(
    firm_manager,
    base_params,
    social_network,
    time_series,
    fileName,
    percentiles,
    dpi=600
):
    emissions_data = np.asarray(social_network.history_driving_emissions_individual).T

    def compute_mean_and_ci(indices):
        selected_data = emissions_data[indices]
        mean = np.mean(selected_data, axis=0)
        se = stats.sem(selected_data, axis=0)
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(indices) - 1)
        return mean, ci

    beta_bin_idx = firm_manager.beta_segment_idx
    gamma_threshold = np.percentile(social_network.gamma_vec, percentiles["Gamma"])
    chi_threshold = np.percentile(social_network.chi_vec, percentiles["Chi"])

    gamma_rich_indices = np.where(social_network.gamma_vec <= gamma_threshold)[0]
    gamma_poor_indices = np.where(social_network.gamma_vec > gamma_threshold)[0]
    chi_rich_indices = np.where(social_network.chi_vec <= chi_threshold)[0]
    chi_poor_indices = np.where(social_network.chi_vec > chi_threshold)[0]

    # Beta bins figure
    fig_beta, axs_beta = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
    for bin_val in range(5):
        bin_indices = np.where(beta_bin_idx == bin_val)[0]
        mean, ci = compute_mean_and_ci(bin_indices)
        axs_beta[bin_val].plot(time_series, mean, label=f"Beta Bin {bin_val}", color="blue")
        axs_beta[bin_val].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
        axs_beta[bin_val].set_title(f"Beta Bin {bin_val}")
        add_vertical_lines(axs_beta[bin_val], base_params)

    fig_beta.supylabel("Mean Driving Emissions")
    fig_beta.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig_beta, fileName, "mean_emissions_beta_bins", dpi)

    # Gamma and Chi figure
    fig_gc, axs_gc = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    # Gamma
    mean, ci = compute_mean_and_ci(gamma_rich_indices)
    axs_gc[0, 0].plot(time_series, mean, label="Gamma Rich", color="blue")
    axs_gc[0, 0].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
    axs_gc[0, 0].set_title("Gamma Rich")
    add_vertical_lines(axs_gc[0, 0], base_params)

    mean, ci = compute_mean_and_ci(gamma_poor_indices)
    axs_gc[1, 0].plot(time_series, mean, label="Gamma Poor", color="red")
    axs_gc[1, 0].fill_between(time_series, mean - ci, mean + ci, color="red", alpha=0.2)
    axs_gc[1, 0].set_title("Gamma Poor")
    add_vertical_lines(axs_gc[1, 0], base_params)

    # Chi
    mean, ci = compute_mean_and_ci(chi_rich_indices)
    axs_gc[0, 1].plot(time_series, mean, label="Chi Rich", color="blue")
    axs_gc[0, 1].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
    axs_gc[0, 1].set_title("Chi Rich")
    add_vertical_lines(axs_gc[0, 1], base_params)

    mean, ci = compute_mean_and_ci(chi_poor_indices)
    axs_gc[1, 1].plot(time_series, mean, label="Chi Poor", color="red")
    axs_gc[1, 1].fill_between(time_series, mean - ci, mean + ci, color="red", alpha=0.2)
    axs_gc[1, 1].set_title("Chi Poor")
    add_vertical_lines(axs_gc[1, 1], base_params)

    fig_gc.supylabel("Mean Driving Emissions")
    fig_gc.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig_gc, fileName, "mean_emissions_gamma_chi", dpi)

def plot_mean_distance_beta_gamma_chi_split(
    firm_manager,
    base_params,
    social_network,
    time_series,
    fileName,
    percentiles,
    dpi=600
):
    """
    Plots mean distance for Beta bins in one figure and Gamma/Chi in another figure.
    """
    # Extract distance data
    distance_data = np.asarray(social_network.history_distance_individual).T

    def compute_mean_and_ci(indices):
        selected_data = distance_data[indices]
        mean = np.mean(selected_data, axis=0)
        se = stats.sem(selected_data, axis=0)
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(indices) - 1)
        return mean, ci

    beta_bin_idx = firm_manager.beta_segment_idx
    gamma_threshold = np.percentile(social_network.gamma_vec, percentiles["Gamma"])
    chi_threshold = np.percentile(social_network.chi_vec, percentiles["Chi"])

    gamma_rich_indices = np.where(social_network.gamma_vec <= gamma_threshold)[0]
    gamma_poor_indices = np.where(social_network.gamma_vec > gamma_threshold)[0]
    chi_rich_indices = np.where(social_network.chi_vec <= chi_threshold)[0]
    chi_poor_indices = np.where(social_network.chi_vec > chi_threshold)[0]

    # Beta bins figure
    fig_beta, axs_beta = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
    for bin_val in range(5):
        bin_indices = np.where(beta_bin_idx == bin_val)[0]
        mean, ci = compute_mean_and_ci(bin_indices)
        axs_beta[bin_val].plot(time_series, mean, label=f"Beta Bin {bin_val}", color="blue")
        axs_beta[bin_val].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
        axs_beta[bin_val].set_title(f"Beta Bin {bin_val}")
        add_vertical_lines(axs_beta[bin_val], base_params)

    fig_beta.supylabel("Mean Distance")
    fig_beta.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig_beta, fileName, "mean_distance_beta_bins", dpi)

    # Gamma and Chi figure
    fig_gc, axs_gc = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    # Gamma
    mean, ci = compute_mean_and_ci(gamma_rich_indices)
    axs_gc[0, 0].plot(time_series, mean, label="Gamma Rich", color="blue")
    axs_gc[0, 0].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
    axs_gc[0, 0].set_title("Gamma Rich")
    add_vertical_lines(axs_gc[0, 0], base_params)

    mean, ci = compute_mean_and_ci(gamma_poor_indices)
    axs_gc[1, 0].plot(time_series, mean, label="Gamma Poor", color="red")
    axs_gc[1, 0].fill_between(time_series, mean - ci, mean + ci, color="red", alpha=0.2)
    axs_gc[1, 0].set_title("Gamma Poor")
    add_vertical_lines(axs_gc[1, 0], base_params)

    # Chi
    mean, ci = compute_mean_and_ci(chi_rich_indices)
    axs_gc[0, 1].plot(time_series, mean, label="Chi Rich", color="blue")
    axs_gc[0, 1].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
    axs_gc[0, 1].set_title("Chi Rich")
    add_vertical_lines(axs_gc[0, 1], base_params)

    mean, ci = compute_mean_and_ci(chi_poor_indices)
    axs_gc[1, 1].plot(time_series, mean, label="Chi Poor", color="red")
    axs_gc[1, 1].fill_between(time_series, mean - ci, mean + ci, color="red", alpha=0.2)
    axs_gc[1, 1].set_title("Chi Poor")
    add_vertical_lines(axs_gc[1, 1], base_params)

    fig_gc.supylabel("Mean Distance")
    fig_gc.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig_gc, fileName, "mean_distance_gamma_chi", dpi)

def plot_mean_utility_beta_gamma_chi_split(
    firm_manager,
    base_params,
    social_network,
    time_series,
    fileName,
    percentiles,
    dpi=600
):
    """
    Plots mean utility for Beta bins in one figure and Gamma/Chi in another figure.
    """
    # Extract utility data
    utility_data = np.asarray(social_network.history_utility_individual).T

    def compute_mean_and_ci(indices):
        selected_data = utility_data[indices]
        mean = np.mean(selected_data, axis=0)
        se = stats.sem(selected_data, axis=0)
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(indices) - 1)
        return mean, ci

    beta_bin_idx = firm_manager.beta_segment_idx
    gamma_threshold = np.percentile(social_network.gamma_vec, percentiles["Gamma"])
    chi_threshold = np.percentile(social_network.chi_vec, percentiles["Chi"])

    gamma_rich_indices = np.where(social_network.gamma_vec <= gamma_threshold)[0]
    gamma_poor_indices = np.where(social_network.gamma_vec > gamma_threshold)[0]
    chi_rich_indices = np.where(social_network.chi_vec <= chi_threshold)[0]
    chi_poor_indices = np.where(social_network.chi_vec > chi_threshold)[0]

    # Beta bins figure
    fig_beta, axs_beta = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
    for bin_val in range(5):
        bin_indices = np.where(beta_bin_idx == bin_val)[0]
        mean, ci = compute_mean_and_ci(bin_indices)
        axs_beta[bin_val].plot(time_series, mean, label=f"Beta Bin {bin_val}", color="blue")
        axs_beta[bin_val].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
        axs_beta[bin_val].set_title(f"Beta Bin {bin_val}")
        add_vertical_lines(axs_beta[bin_val], base_params)

    fig_beta.supylabel("Mean Utility")
    fig_beta.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig_beta, fileName, "mean_utility_beta_bins", dpi)

    # Gamma and Chi figure
    fig_gc, axs_gc = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    # Gamma
    mean, ci = compute_mean_and_ci(gamma_rich_indices)
    axs_gc[0, 0].plot(time_series, mean, label="Gamma Rich", color="blue")
    axs_gc[0, 0].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
    axs_gc[0, 0].set_title("Gamma Rich")
    add_vertical_lines(axs_gc[0, 0], base_params)

    mean, ci = compute_mean_and_ci(gamma_poor_indices)
    axs_gc[1, 0].plot(time_series, mean, label="Gamma Poor", color="red")
    axs_gc[1, 0].fill_between(time_series, mean - ci, mean + ci, color="red", alpha=0.2)
    axs_gc[1, 0].set_title("Gamma Poor")
    add_vertical_lines(axs_gc[1, 0], base_params)

    # Chi
    mean, ci = compute_mean_and_ci(chi_rich_indices)
    axs_gc[0, 1].plot(time_series, mean, label="Chi Rich", color="blue")
    axs_gc[0, 1].fill_between(time_series, mean - ci, mean + ci, color="blue", alpha=0.2)
    axs_gc[0, 1].set_title("Chi Rich")
    add_vertical_lines(axs_gc[0, 1], base_params)

    mean, ci = compute_mean_and_ci(chi_poor_indices)
    axs_gc[1, 1].plot(time_series, mean, label="Chi Poor", color="red")
    axs_gc[1, 1].fill_between(time_series, mean - ci, mean + ci, color="red", alpha=0.2)
    axs_gc[1, 1].set_title("Chi Poor")
    add_vertical_lines(axs_gc[1, 1], base_params)

    fig_gc.supylabel("Mean Utility")
    fig_gc.supxlabel("Time Step")
    plt.tight_layout()
    save_and_show(fig_gc, fileName, "mean_utility_gamma_chi", dpi)


#################################################################################################################################################

#controller plots

def plot_time_series_controller(base_params, reference_data, time_series, reference_label,reference_save, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot reference data (assumes reference_data is a single time-series array)
    ax.plot(time_series, reference_data, color="red", linewidth=2, label=reference_label)

    # Formatting
    ax.set_xlabel("Time Step")
    ax.set_ylabel(reference_label)

    ax.grid(True)
    add_vertical_lines(ax, base_params)
    ax.legend(loc="best")
    # Save and show
    plt.tight_layout()
    save_and_show(fig, fileName, reference_save, dpi)


def plot_social_network(social_network, fileName):
    adjacency_matrix, network = social_network.adjacency_matrix, social_network.network
    beta_vec = social_network.beta_vec

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Choose a layout for the nodes
    pos = nx.spring_layout(network, seed=42)  # You can pick any layout you like

    node_labels = list(network.nodes())
    values = [beta_vec[node] for node in node_labels]

    # Draw the nodes
    node_collection = nx.draw_networkx_nodes(
        network,
        pos,
        ax=ax,
        node_size=300,
        node_color=values,
        cmap=plt.cm.viridis,
        edgecolors='black'
    )

    # Draw the edges
    nx.draw_networkx_edges(network, pos, ax=ax, width=1.0, alpha=0.7)

    ax.axis('off')

    # Create a ScalarMappable to generate a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                               norm=plt.Normalize(vmin=min(values), vmax=max(values)))
    sm.set_array([])  # Needed for older versions of matplotlib

    # Add a colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Price sensitivity, $\\beta$")

    save_and_show(fig, fileName, "network", dpi=300)
#############################################################################################


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


def plot_history_count_buy(base_params, social_network, fileName, dpi=600):

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    
    data = np.asarray(social_network.history_count_buy)
    data_trans = data.T
    labels =  ["current", "new", "second hand"]
    for i in range(data_trans.shape[0]):
        ax.plot(data_trans[i], label = labels[i])
    ax.set_xlabel("Time")
    ax.set_ylabel("Count buys")

    add_vertical_lines(ax, base_params)
    ax.legend()
    save_and_show(fig, fileName, "count_buy", dpi)

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
    ax.stackplot(x, data_proportions.T, labels=labels)

    # Labeling
    ax.set_xlabel("Time")
    ax.set_ylabel("Proportion of buys")
    ax.set_ylim([0, 1])
    add_vertical_lines(ax, base_params)
    ax.legend(loc='lower right')
    save_and_show(fig, fileName, "count_buy_stacked", dpi)

def plot_history_mean_price(base_params, social_network, fileName, dpi=600):

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    
    data = np.asarray(social_network.history_mean_price)
    data_trans = data.T
    labels =  ["new", "second hand"]
    for i in range(data_trans.shape[0]):
        ax.plot(data_trans[i], label = labels[i])
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Price")

    add_vertical_lines(ax, base_params)
    ax.legend()
    save_and_show(fig, fileName, "history_mean_price", dpi)

def plot_history_median_price(base_params, social_network, fileName, dpi=600):

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    
    data = np.asarray(social_network.history_median_price)
    data_trans = data.T
    labels =  ["new", "second hand"]
    for i in range(data_trans.shape[0]):
        ax.plot(data_trans[i], label = labels[i])
    ax.set_xlabel("Time")
    ax.set_ylabel("Median Price")

    add_vertical_lines(ax, base_params)
    ax.legend()
    save_and_show(fig, fileName, "history_median_price", dpi)

def plot_history_quality_users_raw_adjusted(social_network, fileName, dpi=600):
    # Convert the history list to a NumPy array for easier manipulation
    data = np.array(social_network.history_quality_users_raw_adjusted, dtype=object)

    # Extract Quality (x-axis) and Adjusted Quality (y-axis)
    quality = [item[0] for step in data for item in step]
    adjusted_quality = [item[1] for step in data for item in step]

    # Generate an array of time steps corresponding to the data points
    time_steps = np.repeat(np.arange(len(data)), [len(step) for step in data])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(quality, adjusted_quality, c=time_steps, cmap='viridis', alpha=0.8)

    # Add labels and a color bar
    ax.set_xlabel("Quality (Quality_a_t)")
    ax.set_ylabel("Adjusted Quality (Quality_a_t * (1 - delta)**L_a_t)")
    ax.set_title("Quality vs Adjusted Quality Over Time")
    cbar = plt.colorbar(scatter, ax=ax, label="Time Step")

    # Save and show the plot
    fig.tight_layout()
    save_and_show(fig, fileName, "plot_history_quality_users_raw_adjusted", dpi)


def plot_history_second_hand_merchant_price_paid(base_params,social_network, time_series, fileName, dpi=600):

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

    time_points_new = []
    prices_new = []
    
    for i, price_list in enumerate(social_network.history_second_hand_merchant_price_paid):
        time_points_new .extend([time_series[i]] * len(price_list))  # Repeat the time step for each price
        prices_new.extend(price_list)  # Add all prices for the current time step
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time_points_new, prices_new, marker='o', alpha=0.7)
    ax.set_title("Price paid out by second hand merchant")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "price_paid_out_by_second_hand_merchant", dpi)   
    
def plot_history_profit_second_hand(second_hand_merchant, fileName, dpi=600):

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(np.asarray(second_hand_merchant.history_spent), label = "Spent")
    ax.plot(np.asarray(second_hand_merchant.history_income), label = "Income")
    ax.plot(np.asarray(second_hand_merchant.history_assets), label = "Assets")
    ax.plot(np.asarray(second_hand_merchant.history_profit), label = "Profit")
    ax.plot(np.asarray(second_hand_merchant.history_scrap_loss), label = "Scrap loss")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Dollars")
    save_and_show(fig, fileName, "history_profit_second_hand", dpi)


def plot_history_age_second_hand_car_removed(base_params, second_hand_merchant, time_series, fileName, dpi=600):
    """
    Plots the age history of cars removed over time, with per-time-step mean and median lines.

    Args:
    - base_params: Configuration or parameters for the plot (e.g., milestone markers).
    - second_hand_merchant: Object containing history of car ages removed.
    - time_series: List of time steps corresponding to the data.
    - fileName: The name of the file where the plot will be saved.
    - dpi: Resolution for the saved plot.
    """
    # Flatten the data for scatter plot
    time_points_new = []
    ages_new = []

    # Store means and medians for each time step
    mean_per_timestep = []
    median_per_timestep = []

    for i, age_list in enumerate(second_hand_merchant.history_age_second_hand_car_removed):
        # Skip empty lists or lists with only NaNs
        if len(age_list) == 0 or np.all(np.isnan(age_list)):
            mean_per_timestep.append(np.nan)
            median_per_timestep.append(np.nan)
            continue

        time_points_new.extend([time_series[i]] * len(age_list))  # Repeat the time step for each age
        ages_new.extend(age_list)  # Add all ages for the current time step

        # Compute mean and median for the current time step
        mean_per_timestep.append(np.nanmean(age_list))
        median_per_timestep.append(np.nanmedian(age_list))

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time_points_new, ages_new, marker='o', alpha=0.7, label='Age of Cars Removed')

    # Plot per-time-step mean and median
    ax.plot(
        time_series, mean_per_timestep, color='blue', linestyle='-', linewidth=2, label='Mean (Per Time Step)'
    )
    ax.plot(
        time_series, median_per_timestep, color='red', linestyle='--', linewidth=2, label='Median (Per Time Step)'
    )

    # Add vertical lines for base parameters
    add_vertical_lines(ax, base_params)

    # Formatting the plot
    ax.set_title("Age of Cars Scrapped Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Age")
    ax.grid(True)
    ax.legend()

    # Save and show the plot
    save_and_show(fig, fileName, "age_scrapped_second_hand_merchant", dpi)

def plot_segment_production_time_series(base_params, firms, fileName, dpi=600):
    """
    Plots the evolution of the number of cars produced in each segment by each firm over time.

    Parameters:
        firms: list
            List of firm objects containing historical segment production counts.
        fileName: str
            Directory or file name to save the plots.
        dpi: int
            DPI for the saved plots.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color palette for firms
    cmap = get_cmap("tab10")
    
    # Iterate through each firm and plot its data
    for firm_idx, firm in enumerate(firms):
        segment_counts_by_time = firm.history_segment_production_counts
        segment_time_series = {}

        # Accumulate counts for each segment across time
        for t, segment_counts in enumerate(segment_counts_by_time):
            for segment, count in segment_counts.items():
                if segment not in segment_time_series:
                    segment_time_series[segment] = [0] * len(segment_counts_by_time)
                segment_time_series[segment][t] = count

        # Plot the time series for each segment
        for segment, counts in segment_time_series.items():
            ax.plot(
                range(len(counts)),
                counts,
                label=f"Firm {firm_idx + 1} - Segment {segment}",
                color=cmap(firm_idx),
                linestyle='-'
            )

    # Add labels, legend, and grid
    ax.set_title("Evolution of Cars Produced in Each Segment by Each Firm")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Cars Produced")
    ax.grid()

    add_vertical_lines(ax, base_params)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Save and show the plot
    save_and_show(fig, fileName, "plot_segment_production_time_series", dpi)

def plot_aggregated_segment_production_time_series(base_params, firms, fileName, dpi=600):
    """
    Plots the aggregated number of cars produced in each segment over time across all firms.

    Parameters:
        base_params: Configuration or parameters for the plot (e.g., milestone markers).
        firms: list
            List of firm objects containing historical segment production counts.
        fileName: str
            Directory or file name to save the plots.
        dpi: int
            DPI for the saved plots.
    """
    # Initialize a dictionary to store aggregated segment counts over time
    aggregated_segment_counts = {}

    # Iterate through each firm and aggregate segment data
    for firm in firms:
        for t, segment_counts in enumerate(firm.history_segment_production_counts):
            for segment, count in segment_counts.items():
                if segment not in aggregated_segment_counts:
                    aggregated_segment_counts[segment] = []
                # Ensure the list for this segment has enough time steps
                while len(aggregated_segment_counts[segment]) <= t:
                    aggregated_segment_counts[segment].append(0)
                aggregated_segment_counts[segment][t] += count

    # Determine the maximum number of time steps
    max_time_steps = max(len(v) for v in aggregated_segment_counts.values())

    # Ensure all lists in aggregated_segment_counts are the same length
    for segment in aggregated_segment_counts:
        while len(aggregated_segment_counts[segment]) < max_time_steps:
            aggregated_segment_counts[segment].append(0)

    # Plot the aggregated data
    fig, ax = plt.subplots(figsize=(12, 8))

    for segment, counts in aggregated_segment_counts.items():
        ax.plot(
            range(len(counts)),
            counts,
            label=f"Segment {segment}",
            linestyle='-'
        )

    # Add labels, legend, and grid
    ax.set_title("Aggregated Cars Produced in Each Segment Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Cars Produced")
    ax.grid()

    add_vertical_lines(ax, base_params)

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

    # Save and show the plot
    save_and_show(fig, fileName, "aggregated_segment_production_time_series", dpi)

def plot_kg_co2_per_year_per_vehicle_by_type(base_params, social_network, time_series, fileName, dpi = 600):
    
    data_time_series_ICE = np.asarray(social_network.history_driving_emissions_ICE)/np.asarray(social_network.history_ICE_users)
    history_ICE_users = np.asarray(social_network.history_ICE_users)

    # Replace 0 with np.nan in history_ICE_users
    history_ICE_users = np.where(history_ICE_users == 0, np.nan, history_ICE_users)

    # Perform the division
    data_time_series_EV = np.asarray(social_network.history_driving_emissions_EV) / history_ICE_users

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(time_series, data_time_series_ICE, label = "ICE")
    ax.plot(time_series, data_time_series_EV, label = "EV")
    
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

def plot_emissions_intensity(base_params, real_data , social_network, time_series, fileName, dpi = 600):
    
    data_time_series = np.asarray(social_network.history_driving_emissions_ICE)/np.asarray(social_network.history_total_distance_driven_ICE)#kg_C02_per_km
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(time_series, data_time_series, label = "Simulated data")
    time_series_yearly = range(base_params["duration_burn_in"],base_params["duration_burn_in"] + 23*12, 12)
    ax.plot(time_series_yearly, real_data, label = "California data")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Kg CO2 per km, ICE cars")

    add_vertical_lines(ax, base_params)
    
    ax.legend(loc="center left")
    save_and_show(fig, fileName, "plot_emissions_per_km_ICE", dpi)
    
def plot_num_bought_by_type(base_params, social_network, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(social_network.history_new_ICE_cars_bought, label = "ICE")
    ax.plot(social_network.history_new_EV_cars_bought, label = "EV")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Count new cars bought")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_num_bought_by_type", dpi)

def plot_fuel_costs_verus_carbon_price_kWhr(base_params,controller, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))

    carbon_tax_paid = np.asarray(controller.carbon_price_time_series)*controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
    total_ice = controller.history_gas_price + carbon_tax_paid[:-2]

    ax.plot(controller.history_electricity_price, label = "Fuel cost per kWhr, EV")
    ax.plot(controller.history_gas_price, label = "Fuel cost per kWhr, ICE")
    ax.plot(carbon_tax_paid , label = "carbon tax cost per kWhr, ICE")
    ax.plot(total_ice, label = "TOTAL cost per kWhr, ICE")
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

    ax.plot(controller.history_electricity_price/Eff_omega_a_t_EV_median, label = "Fuel cost per km, EV")
    ax.plot(controller.history_gas_price/Eff_omega_a_t_ICE_median, label = "Fuel cost per km, ICE")
    ax.plot(carbon_tax_paid[:-2]/Eff_omega_a_t_ICE_median , label = "carbon tax cost per km, ICE")
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

    ax.plot(controller.electricity_emissions_intensity_vec[:-1]/Eff_omega_a_t_EV_median, label = "Emissions per km, EV")
    ax.plot(controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]/Eff_omega_a_t_ICE_median, label = "Emissions per km, ICE")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("Emissions per km")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_fuel_Emissions_verus_carbon_price_km", dpi)

def plot_zero_util_count(base_params, social_network, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(social_network.history_zero_util_count)
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("pop zero util count")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_zero_util_count", dpi)

def plot_history_zero_profit_options_prod_sum(base_params, firm_manager, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(firm_manager.history_zero_profit_options_prod_sum)
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("prop zero profit options prod sum")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "plot_history_zero_profit_options_prod_sum", dpi)


def plot_history_zero_profit_options_research_sum(base_params, firm_manager, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(firm_manager.history_zero_profit_options_research_sum)
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("prop zero profit options research sum")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "history_zero_profit_options_research_sum", dpi)



def plot_history_history_drive_min_num(base_params, social_network, fileName, dpi = 600):

    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(social_network.history_drive_min_num)
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("prop drive min num")

    add_vertical_lines(ax, base_params)
    
    ax.legend()
    save_and_show(fig, fileName, "history_drive_min_num", dpi)

def emissions_decomposed(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    

    # Right plot: Stacked area plot for ICE and EV emissions
    driving_ICE = np.cumsum(social_network.history_driving_emissions_ICE)
    driving_EV = np.cumsum(social_network.history_driving_emissions_EV)
    production_ICE = np.cumsum(social_network.history_production_emissions_ICE)
    production_EV = np.cumsum(social_network.history_production_emissions_EV)
    
    ax.stackplot(
        time_series, 
        driving_ICE, 
        driving_EV, 
        production_ICE, 
        production_EV,
        labels=['Driving Emissions ICE', 'Driving Emissions EV', 'Production Emissions ICE', 'Production Emissions EV']
    )
    ax.plot(time_series, np.cumsum(social_network.history_total_emissions), 
                 label='Total Emissions', color='black', linewidth=1.5)
    ax.set_title("Cumulative Emissions by Source")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Emissions")
    ax.legend()

    # Format and save
    plt.tight_layout()
    save_and_show(fig, fileName, "emissions_decomposed", dpi)

def plot_history_second_hand_merchant_offer_price(base_params,social_network, time_series, fileName, dpi=600):
    
    # Flatten the data

    time_points_new = []
    prices_new = []
    
    for i, price_list in enumerate(social_network.history_second_hand_merchant_offer_price):
        time_points_new .extend([time_series[i]] * len(price_list))  # Repeat the time step for each price
        prices_new.extend(price_list)  # Add all prices for the current time step
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time_points_new, prices_new, marker='o', alpha=0.7)
    #ax.set_title("Second hand merchant offer price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Second hand merchant offer price")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "plot_history_second_hand_merchant_offer_price", dpi)   


def plot_profit_margins_by_type(base_params, firm_manager,time_series,  fileName, dpi=600):

    time_points_new_ICE = []
    time_points_new_EV = []
    profit_margins_ICE = []
    profit_margins_EV = []
    
    for i, pm_list in enumerate(firm_manager.history_profit_margins_ICE):
        time_points_new_ICE.extend([time_series[i]] * len(pm_list))  
        profit_margins_ICE.extend(pm_list)

    for i, pm_list in enumerate(firm_manager.history_profit_margins_EV):
        time_points_new_EV.extend([time_series[i]] * len(pm_list))  
        profit_margins_EV.extend(pm_list)
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    print()
    ax.scatter(time_points_new_ICE, profit_margins_ICE, marker='o', alpha=0.7, color = "blue", label = "ICE")
    ax.scatter(time_points_new_EV, profit_margins_EV, marker='o', alpha=0.7, color = "green", label = "EV")

    ax.set_xlabel("Time")
    ax.set_ylabel("Profit margin (P-C)")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "plot_profit_margins_by_type", dpi)

def plot_history_W(base_params, firm_manager,time_series,  fileName, dpi=600):


    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    print()
    
    data = np.asarray(firm_manager.history_W).T
    for i, data_segment in enumerate(data):
        ax.plot(time_series, data_segment, label = i)

    ax.set_xlabel("Time")
    ax.set_ylabel("W")
    ax.grid(True)

    add_vertical_lines(ax, base_params)
    ax.legend()
    # Save and show the plot
    save_and_show(fig, fileName, "plot_history_W", dpi)


# Sample main function
def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_controller = load_object(fileName + "/Data", "controller")
    except FileNotFoundError:
        print("Data files not found.")
        return

    social_network = data_controller.social_network
    firm_manager = data_controller.firm_manager
    second_hand_merchant = data_controller.second_hand_merchant
    time_series = data_controller.time_series
    

    #plot_kg_co2_per_year_per_vehicle_by_type(base_params, social_network, time_series, fileName, dpi)
    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]
    #plot_ev_stock(base_params, EV_stock_prop_2010_22, social_network, fileName, dpi=600)

    emissions_per_km_2000_22 = calibration_data_output["kg_CO2_per_km"]
    #plot_emissions_intensity(base_params, emissions_per_km_2000_22 , social_network, time_series, fileName, dpi)


    #emissions_decomposed(social_network, time_series, fileName, dpi)

    plot_history_W(base_params, firm_manager,time_series,  fileName)

    plot_profit_margins_by_type(base_params, firm_manager,time_series,  fileName)
    #plt.show()

    plot_transport_users_stacked(base_params, social_network, time_series, fileName, dpi)

    plot_distance_individuals_mean_median_type(base_params, social_network, time_series, fileName)

    #plot_history_profit_second_hand(second_hand_merchant, fileName, dpi)

    #plot_history_second_hand_merchant_price_paid(base_params,social_network, time_series, fileName, dpi)
    #plot_history_second_hand_merchant_offer_price(base_params,social_network, time_series, fileName, dpi)
    #plot_history_quality_users_raw_adjusted(social_network, fileName, dpi)
    plot_price_history_new_second_hand(base_params,social_network, time_series, fileName, dpi)

    plot_history_median_price(base_params, social_network, fileName, dpi)
    plot_history_mean_price(base_params, social_network, fileName, dpi)

    plot_history_count_buy_stacked(base_params, social_network, fileName, dpi)

    plot_total_utility(social_network, time_series, fileName, dpi)

    plot_ev_consider_adoption_rate( base_params, social_network, time_series, fileName, dpi)

    #plot_scatter_research_time_series_multiple_firms(firm_manager.firms_list, fileName)
    #plot_second_hand_market_len(second_hand_merchant, time_series, fileName, dpi)
    #plot_preferences_scatter(social_network, fileName, dpi)
    plot_preferences(social_network, fileName, dpi)
    plot_sale_EV_prop(firm_manager, time_series, fileName, dpi)
    plot_history_research_type(firm_manager, time_series, fileName, dpi)
    plot_car_sale_prop(social_network, time_series, fileName, dpi)

    plot_total_utility_vs_total_profit(social_network, firm_manager, time_series, fileName)
    plot_total_profit(firm_manager, time_series, fileName, dpi)
    plot_market_concentration(firm_manager, time_series, fileName, dpi)
    
    #plot_history_history_drive_min_num(base_params, social_network, fileName, dpi)
    #plot_zero_util_count(base_params, social_network, fileName, dpi)
    #plot_history_zero_profit_options_prod_sum(base_params, firm_manager, fileName, dpi)
    #plot_history_zero_profit_options_research_sum(base_params, firm_manager, fileName, dpi)

    plot_history_num_cars_on_sale(firm_manager, time_series, fileName)

    plot_num_bought_by_type(base_params, social_network, fileName, dpi)

    # All plot function calls

    plot_fuel_costs_verus_carbon_price_km(base_params,data_controller, fileName, dpi)

    #plot_fuel_emissions_verus_carbon_price_km(base_params,data_controller, fileName, dpi)

    plot_fuel_costs_verus_carbon_price_kWhr(base_params,data_controller, fileName, dpi)

    
    #plot_calibration_data(data_controller, time_series, fileName)

    #plot_aggregated_segment_production_time_series(base_params,firm_manager.firms_list, fileName, dpi)
    #plot_segment_production_time_series(base_params,firm_manager.firms_list, fileName, dpi)
 
    plot_history_age_second_hand_car_removed(base_params,second_hand_merchant, time_series, fileName, dpi)

    plot_history_car_age(base_params, social_network, time_series,fileName, dpi)
    #plot_history_car_age_scatter(social_network, time_series,fileName, dpi)
    #plot_total_distance(social_network, time_series, fileName, dpi)
    #plot_price_history(base_params, firm_manager, time_series, fileName, dpi)
    
    #SEGEMENT PLOTS
    #plot_segment_count_grid(firm_manager, time_series, fileName)

    #THIS TAKES FOREVER AND IS NOT VERY INSIGHTFUL
    #history_car_cum_distances(social_network, time_series, fileName, dpi=600)

    #CALIBRATION PLOTS

    #plot_emissions_individuals(social_network, time_series, fileName)
    #plot_distance_individuals(social_network, time_series, fileName)
    #plot_utility_individuals(social_network, time_series, fileName)
    #plot_transport_type_individuals(social_network, time_series, fileName)
    #plot_density_by_value(fileName, social_network, time_series)

    #plot_transport_users_stacked_rich_poor(social_network, time_series, fileName, x_percentile=90)
    #plot_emissions(social_network, time_series, fileName, dpi)
    #plot_vehicle_attribute_time_series_by_type(base_params, social_network, time_series, fileName, dpi)
    
    plot_vehicle_attribute_time_series_by_type_split(base_params, social_network, time_series, fileName, dpi)
    #"""
    
    plot_transport_new_cars_stacked(social_network, time_series, fileName, dpi)
    #"""
    #percentiles = {'Beta': base_params["parameters_firm_manager"]["beta_threshold_percentile"], 'Gamma': base_params["parameters_firm_manager"]["gamma_threshold_percentile"], 'Chi': 50}
    #percentiles = {'Gamma': base_params["parameters_firm_manager"]["gamma_threshold_percentile"], 'Chi': 50}
    #plot_transport_users_stacked_two_by_four(base_params,social_network, time_series, fileName, percentiles)
    #plot_mean_emissions_one_row(base_params,social_network, time_series, fileName, percentiles)
    #plot_mean_distance_one_row(base_params,social_network, time_series, fileName, percentiles)
    #plot_mean_utility_one_row(base_params,social_network, time_series, fileName, percentiles)

    #plot_transport_users_beta_gamma_chi(firm_manager,base_params,social_network, time_series, fileName, percentiles)
    #plot_mean_emissions_beta_gamma_chi_split(firm_manager,base_params,social_network, time_series, fileName, percentiles)
    #plot_mean_distance_beta_gamma_chi_split(firm_manager,base_params,social_network, time_series, fileName, percentiles)
    #plot_mean_utility_beta_gamma_chi_split(firm_manager,base_params,social_network, time_series, fileName, percentiles)

    #plot_conditional_transport_users_4x4(base_params,social_network, time_series, fileName, percentiles)
    
    #PLOT ACTUAL VALUES USED
    #plot_time_series_controller(data_controller.history_gas_price, time_series,"Gas price","gas_price", fileName)
    #plot_time_series_controller(data_controller.history_electricity_price, time_series,"Electricity price","electricity_price", fileName)
    #plot_time_series_controller(data_controller.history_electricity_emissions_intensity, time_series,"Electricity emissions intensity","electricity_emissions_intensity", fileName)
    #plot_time_series_controller(data_controller.history_rebate, time_series,"EV rebate","rebate", fileName)
    #plot_time_series_controller(data_controller.history_used_rebate, time_series,"Used EV rebate","used_rebate", fileName)

    #plot_social_network(social_network, fileName)
    
    plot_carbon_price(data_controller, time_series, fileName)

    #CHECKING OUTPUTS


    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_10_29_00__03_02_2025")
