import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from package.resources.utility import load_object
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def ensure_directory_exists(path):
    """Ensure that a directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

# Modify each plotting function to include file saving
def plot_emissions(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_driving_emissions, label='Driving Emissions', marker='o')
    ax.plot(social_network.history_production_emissions, label='Production Emissions', marker='s')
    ax.set_title("Emissions Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Emissions")
    ax.legend()
    ax.grid()
    
    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/emissions.png", dpi=600, format="png")

def plot_total_utility(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_total_utility, marker='o')
    ax.set_title("Total Utility Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Utility")
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/total_utility.png", dpi=600, format="png")

def plot_total_distance(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_total_distance_driven, marker='o')
    ax.set_title("Total Distance Traveled Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Distance Traveled")
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/total_distance.png", dpi=600, format="png")

def plot_ev_adoption_rate(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_ev_adoption_rate, marker='o')
    ax.set_title("EV Adoption Rate Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("EV Adoption Rate")
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/ev_adoption_rate.png", dpi=600, format="png")

def plot_ev_consider_rate(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_consider_ev_rate, marker='o')
    ax.set_title("EV Consideration Rate Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("EV Consider Rate")
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/ev_consider_rate.png", dpi=600, format="png")

def plot_second_hand_users(social_network, fileName):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_second_hand_users, marker='o')
    ax.set_title("Second hand users Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("# second  hand car user")
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/history_second_hand_user.png", dpi=600, format="png")


def plot_tranport_users(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(social_network.history_rural_public_transport_users, label='Rural public transport', marker='o')
    ax.plot(social_network.history_urban_public_transport_users, label='Urban public transport', marker='o')
    ax.plot(social_network.history_ICE_users, label='ICE', marker='o')
    ax.plot(social_network.history_EV_users, label='EV', marker='o')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("# Transport Users")
    ax.legend()
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/transport_users.png", dpi=600, format="png")

def plot_cumulative_emissions(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.cumsum(social_network.history_driving_emissions), label="Cumulative Driving Emissions", marker='o')
    ax.plot(np.cumsum(social_network.history_production_emissions), label="Cumulative Production Emissions", marker='s')
    ax.set_title("Cumulative Emissions Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Emissions")
    ax.legend()
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/cumulative_emissions.png", dpi=600, format="png")

def plot_vehicle_attribute_time_series(social_network, fileName):
    fig, axs = plt.subplots(1, 3, figsize=(10, 15))
    attributes = {
        "Quality (Q_a_t)": social_network.history_quality,
        "Efficiency (omega_a_t)": social_network.history_efficiency,
        "Production Cost (c_z_t)": social_network.history_production_cost,
    }
    
    for i, (attribute_name, attribute_history) in enumerate(attributes.items()):
        mean_values = [np.mean(values) for values in attribute_history]
        confidence_intervals = [1.96 * sem(values) for values in attribute_history]
        time_steps = range(len(attribute_history))
        
        axs[i].plot(time_steps, mean_values, label="Mean " + attribute_name, color="blue")
        axs[i].fill_between(
            time_steps,
            np.array(mean_values) - np.array(confidence_intervals),
            np.array(mean_values) + np.array(confidence_intervals),
            color="blue",
            alpha=0.2,
            label="95% Confidence Interval"
        )
        
        axs[i].set_title(f"{attribute_name} Over Time")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(attribute_name)
        axs[i].legend()
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/vehicle_attribute_time_series.png", dpi=600, format="png")

def plot_research_time_series_multiple_firms(firms, fileName=None):
    """
    Plot the time series of multiple firms' research with x as the first item in attributes fitness,
    y as the second, and color representing the third for each time step.
    
    Parameters:
    - firms (list): List of Firm instances with research history data.
    - fileName (str): Optional file path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get all third attribute values to set color scale
    all_color_vals = [attr[2] for firm in firms for attr in firm.history_attributes_researched]
    norm = mcolors.Normalize(vmin=min(all_color_vals), vmax=max(all_color_vals))
    cmap = cm.viridis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the color bar

    # Loop through each firm to plot its research path over time
    for firm in firms:
        # Extract time series data for x, y, and color
        x_vals = [attr[0] for attr in firm.history_attributes_researched]
        y_vals = [attr[1] for attr in firm.history_attributes_researched]
        color_vals = [attr[2] for attr in firm.history_attributes_researched]
        
        # Plot each firm's trajectory with changing colors and a gradient for time
        for j in range(len(x_vals) - 1):
            time_factor = (j + 1) / len(x_vals)  # Scale from 0 to 1 over time steps
            ax.plot(
                [x_vals[j], x_vals[j+1]], [y_vals[j], y_vals[j+1]],
                color=cmap(norm(color_vals[j])),
                linewidth=1 + time_factor * 2,  # Increase line width over time
                alpha=0.5 + 0.5 * time_factor  # Increase alpha (opacity) over time
            )

        # Add markers for time steps and label some of them to show progression
        scatter = ax.scatter(x_vals, y_vals, c=color_vals, cmap=cmap, norm=norm, edgecolor='black', s=50, zorder=3)
        
        # Label points at specific intervals to indicate time passage
        for k in range(0, len(x_vals), max(1, len(x_vals) // 5)):  # Label every 20% of the time series
            ax.text(
                x_vals[k], y_vals[k], f"T{k}",
                fontsize=8, color='black', ha='right', va='bottom'
            )

    # Add labels, title, and color bar for third attribute
    ax.set_xlabel("Quality (First Attribute)")
    ax.set_ylabel("Efficiency (Second Attribute)")
    ax.set_title("Research Attributes Over Time for Multiple Firms")
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Production Cost (Third Attribute)")

    # Optional saving of the plot
    if fileName:
        save_path = fileName + "/Plots"
        plt.savefig(f"{save_path}/multi_firm_research_time_series.png", dpi=600, format="png")

    plt.show()




# Sample main to save all plots
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

    #plot_emissions(social_network, fileName)
    #plot_total_utility(social_network, fileName)
    #plot_total_distance(social_network, fileName)
    #plot_ev_adoption_rate(social_network, fileName)
    #plot_ev_consider_rate(social_network, fileName)
    #plot_tranport_users(social_network, fileName)
    #plot_cumulative_emissions(social_network, fileName)
    plot_vehicle_attribute_time_series(social_network, fileName)
    plot_research_time_series_multiple_firms([firm_manager.firms_list[0]], fileName)
    #plot_second_hand_users(social_network, fileName)
    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_18_31_07__08_11_2024")
