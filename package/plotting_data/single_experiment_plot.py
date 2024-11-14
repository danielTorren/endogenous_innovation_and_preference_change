import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from package.resources.utility import load_object
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    total_users = (np.array(social_network.history_rural_public_transport_users) +
                   np.array(social_network.history_urban_public_transport_users) +
                   np.array(social_network.history_ICE_users) +
                   np.array(social_network.history_EV_users))

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
    ax.plot(time_series, np.mean(data, axis=0), marker='o')
    format_plot(ax, "EV Research Proportion Over Time", "Time Step", "Proportion Research EV", legend=False)
    save_and_show(fig, fileName, "history_research_type", dpi)

def plot_history_attributes_cars_on_sale_all_firms(social_network, time_series, fileName, dpi=600):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    data_EV = social_network.history_attributes_EV_cars_on_sale_all_firms
    data_ICE = social_network.history_attributes_ICE_cars_on_sale_all_firms

    max_len_EV = max(len(step) for step in data_EV)
    max_len_ICE = max(len(step) for step in data_ICE)
    data_EV_padded = [step + [[np.nan] * 3] * (max_len_EV - len(step)) for step in data_EV]
    data_ICE_padded = [step + [[np.nan] * 3] * (max_len_ICE - len(step)) for step in data_ICE]
    data_EV_array, data_ICE_array = np.asarray(data_EV_padded), np.asarray(data_ICE_padded)

    for i, attribute_name in enumerate(["Attribute 1", "Attribute 2", "Attribute 3"]):
        axs[i].scatter(time_series, data_EV_array[:, :, i].flatten(), color="green", alpha=0.2, s=10, label="EV Data")
        axs[i].scatter(time_series, data_ICE_array[:, :, i].flatten(), color="blue", alpha=0.2, s=10, label="ICE Data")
        axs[i].set_title(f"{attribute_name} Over Time")
        axs[i].set_xlabel("Time Steps")

    fig.suptitle("Attributes of Cars on Sale Over Time")
    save_and_show(fig, fileName, "history_attributes_cars_on_sale_all_firms", dpi)

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
    fig.savefig(f"{save_path}/segment_count_grid.png", dpi=600, format="png")

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
    plot_second_hand_market_len(second_hand_merchant, time_series, fileName, dpi)
    plot_segment_count_grid(firm_manager, time_series, fileName)
    plot_preferences(social_network, fileName, dpi)
    plot_sale_EV_prop(firm_manager, time_series, fileName, dpi)
    plot_history_research_type(firm_manager, time_series, fileName, dpi)
    #plot_history_attributes_cars_on_sale_all_firms(social_network, time_series, fileName, dpi)

    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_20_31_10__14_11_2024")
