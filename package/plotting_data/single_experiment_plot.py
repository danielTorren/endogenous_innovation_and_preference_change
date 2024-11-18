import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


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

# Plot functions with `time_series` where applicable

def plot_emissions(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_driving_emissions, label='Driving Emissions', marker='o')
    ax.plot(time_series, social_network.history_production_emissions, label='Production Emissions', marker='s')
    format_plot(ax, "Emissions Over Time", "Time Step", "Emissions")
    save_and_show(fig, fileName, "emissions", dpi)

def plot_total_utility(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_total_utility, marker='o')
    format_plot(ax, "Total Utility Over Time", "Time Step", "Total Utility", legend=False)
    save_and_show(fig, fileName, "total_utility", dpi)

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

def plot_tranport_users(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, social_network.history_rural_public_transport_users, label='Rural Public Transport', marker='o')
    ax.plot(time_series, social_network.history_urban_public_transport_users, label='Urban Public Transport', marker='o')
    ax.plot(time_series, social_network.history_ICE_users, label='ICE', marker='o')
    ax.plot(time_series, social_network.history_EV_users, label='EV', marker='o')
    format_plot(ax, "Transport Users Over Time", "Time Step", "# Transport Users")
    save_and_show(fig, fileName, "transport_users", dpi)

def plot_transport_users_stacked(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate total users at each time step
    total_users = social_network.num_individuals

    # Calculate proportions
    rural_prop = np.array(social_network.history_rural_public_transport_users) / total_users
    urban_prop = np.array(social_network.history_urban_public_transport_users) / total_users
    ice_prop = np.array(social_network.history_ICE_users) / total_users
    ev_prop = np.array(social_network.history_EV_users) / total_users

    # Plot stacked area (continuous stacked bar equivalent)
    ax.stackplot(time_series, rural_prop, urban_prop, ice_prop, ev_prop,
                 labels=['Rural Public Transport', 'Urban Public Transport', 'ICE', 'EV'],
                 alpha=0.8)

    # Set plot labels and limits
    ax.set_title("Transport Users Over Time (Proportion)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Proportion of Transport Users")
    ax.set_ylim(0, 1)  # Proportion range
    ax.legend(loc="upper left")

    # Save and show the plot
    save_and_show(fig, fileName, "plot_transport_users_stacked", dpi)


def plot_vehicle_attribute_time_series(social_network, time_series, fileName, dpi=600):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    attributes = {
        "Quality (Quality_a_t)": social_network.history_quality,
        "Efficiency (Eff_omega_a_t)": social_network.history_efficiency,
        "Production Cost (ProdCost_z_t)": social_network.history_production_cost,
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

    fig.suptitle("Evolution of firm reserach attributes")
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
        ax.plot(time_series,firm_data, marker='o')
    format_plot(ax, "EV Research Proportion Over Time", "Time Step", "Proportion Research EV", legend=False)
    save_and_show(fig, fileName, "history_research_type", dpi)

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

def plot_segment_count_grid(firm_manager,time_series, fileName):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)


    data_trans = np.asarray(firm_manager.history_segment_count).T
    num_segments = data_trans.shape[0]

    for i, data in enumerate(data_trans):
        if i >= 16:  # Limit to the first 16 segments if there are more
            break
        row, col = divmod(i, 4)
        ax = axes[row, col]
        segment_code = format(i, '04b')
        
        ax.plot(time_series, data, label=f"Segment {segment_code}")

        ax.legend(loc='upper right')
        ax.grid()

    # Adjust layout
    fig.supxlabel("Time Step")
    fig.supylabel("# Segment")
    plt.tight_layout()
    # Save the figure with a new name
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    save_and_show(fig, fileName, "segment_count_grid.png", 600)    


def plot_car_sale_prop(social_network, time_series, fileName, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    #print(social_network.history_second_hand_bought)
    #quit()
    ax.plot(time_series, social_network.history_second_hand_bought,label = "second hand",  marker='o')
    ax.plot(time_series, social_network.history_new_car_bought,label = "new cars",  marker='o')
    ax.legend()
    format_plot(ax, "New versus Second hand cars", "Time Step", "# Cars bought", legend=False)
    save_and_show(fig, fileName, "num_cars_bought_type", dpi)      

import matplotlib.pyplot as plt

def plot_price_history(firm_manager, time_series, fileName, dpi=600):
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

    # Save and show the plot
    save_and_show(fig, fileName, "price_cars_sale", dpi)   
  
def plot_history_car_age(social_network,time_series, fileName, dpi):
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
    ages_list = social_network.history_car_age
    for ages in ages_list:
        ages = np.array(ages)
        valid_ages = ages[~np.isnan(ages)]  # Exclude NaNs from calculations
        mean = np.mean(valid_ages)
        confidence = t.ppf(0.975, len(valid_ages)-1) * sem(valid_ages)
        
        means.append(mean)
        lower_bounds.append(mean - confidence)
        upper_bounds.append(mean + confidence)

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, means, label="Mean Age", color='blue')
    ax.fill_between(time_series, lower_bounds, upper_bounds, color='blue', alpha=0.2, label="95% Confidence Interval")
    ax.set_title("Mean Age and 95% Confidence Interval Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Age")
    ax.legend()
    ax.grid(True)

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
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)

    # Limit to the first 16 segments (assuming binary codes from "0000" to "1111")
    segment_codes = [format(i, '04b') for i in range(16)]

    for i, segment_code in enumerate(segment_codes):
        row, col = divmod(i, 4)
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
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)

    segment_codes = [format(i, '04b') for i in range(16)]  # Binary segment codes

    for i, segment_code in enumerate(segment_codes):
        row, col = divmod(i, 4)
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




    # All plot function calls
    plot_emissions(social_network, time_series, fileName, dpi)
    plot_total_utility(social_network, time_series, fileName, dpi)
    plot_total_distance(social_network, time_series, fileName, dpi)
    plot_ev_adoption_rate(social_network, time_series, fileName, dpi)
    plot_ev_consider_rate(social_network, time_series, fileName, dpi)
    plot_tranport_users(social_network, time_series, fileName, dpi)
    plot_transport_users_stacked(social_network, time_series, fileName, dpi)
    plot_vehicle_attribute_time_series(social_network, time_series, fileName, dpi)
    #plot_research_time_series_multiple_firms([firm_manager.firms_list[0]], fileName, dpi)
    plot_scatter_research_time_series_multiple_firms(firm_manager.firms_list, fileName)
    plot_second_hand_market_len(second_hand_merchant, time_series, fileName, dpi)

    plot_preferences(social_network, fileName, dpi)
    plot_sale_EV_prop(firm_manager, time_series, fileName, dpi)
    plot_history_research_type(firm_manager, time_series, fileName, dpi)
    plot_car_sale_prop(social_network, time_series, fileName, dpi)
    plot_history_attributes_cars_on_sale_all_firms_alt(social_network, time_series, fileName, dpi)
    plot_price_history(firm_manager, time_series, fileName, dpi)
    plot_history_car_age(social_network, time_series,fileName, dpi)

    #SEGEMENT PLOTS
    plot_segment_count_grid(firm_manager, time_series, fileName)
    
    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_11_07_56__18_11_2024")
