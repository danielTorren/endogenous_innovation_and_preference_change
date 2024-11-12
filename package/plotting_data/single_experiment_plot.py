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
    fig.suptitle("Chosen vehicle Attribute")
    attributes = {
        "Quality (Quality_a_t)": social_network.history_quality,
        "Efficiency (Eff_omega_a_t)": social_network.history_efficiency,
        "Production Cost (ProdCost_z_t)": social_network.history_production_cost,
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
    
    #plt.tight_layout()
    
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

def plot_segment_count(firm_manager, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))

    data_trans = np.asarray(firm_manager.history_segment_count).T
    for i, data in enumerate(data_trans):
        segment_code = format(i, '04b')
        ax.plot(data, label=segment_code)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("# Segment")
    
    # Position the legend just outside the plot area to the right
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/segment_count.png", dpi=600, format="png")

def plot_segment_count_grid(firm_manager, fileName):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)


    data_trans = np.asarray(firm_manager.history_segment_count).T
    num_segments = data_trans.shape[0]

    for i, data in enumerate(data_trans):
        if i >= 16:  # Limit to the first 16 segments if there are more
            break
        row, col = divmod(i, 4)
        ax = axes[row, col]
        segment_code = format(i, '04b')
        
        ax.plot(data, label=f"Segment {segment_code}")

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

def plot_second_hand_market_len(market, fileName):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(market.history_num_second_hand)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("# Second hand cars")
    ax.legend()
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/second_hand_car_len.png", dpi=600, format="png")

def plot_preferences(social_network, fileName):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(social_network.beta_vec, bins=30, alpha=0.5, label='Beta Vec (Price)')   # Adjust bins as needed
    ax.hist(social_network.gamma_vec, bins=30, alpha=0.5, label='Gamma Vec (Environmental)')
    ax.hist(social_network.chi_vec, bins=30, alpha=0.5, label='Chi Vec (EV threshold)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Beta, Gamma, and Chi Vectors')
    ax.legend()
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/preferences.png", dpi=600, format="png")

def plot_sale_EV_prop(firm_manager, fileName):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(firm_manager.history_cars_oon_sale_EV_prop, label = "EV")
    ax.plot(firm_manager.history_cars_oon_sale_ICE_prop, label = "ICE")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("# cars on sale")
    ax.legend()
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/ev_cars_sale.png", dpi=600, format="png")


def plot_history_research_type(firm_manager, fileName):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray([firm.history_research_type for firm in firm_manager.firms_list])
    data_timeseries = np.mean(data, axis = 0)
    ax.plot(data_timeseries)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("prop research EV")
    ax.legend()
    ax.grid()

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/ev_research.png", dpi=600, format="png")



def plot_history_attributes_cars_on_sale_all_firms(social_network, fileName):
    cols = 3
    fig, axs = plt.subplots(1, cols, figsize=(12, 8))

    # Convert data to numpy arrays with NaNs for missing entries
    data_EV = social_network.history_attributes_EV_cars_on_sale_all_firms
    data_ICE = social_network.history_attributes_ICE_cars_on_sale_all_firms
    
    # Determine the maximum number of vehicles across all time steps
    max_len_EV = max(len(time_step) for time_step in data_EV)
    max_len_ICE = max(len(time_step) for time_step in data_ICE)

    # Pad data with NaNs to have consistent dimensions (time steps, vehicles, attributes)
    padded_data_EV = [
        np.array(time_step + [[np.nan, np.nan, np.nan]] * (max_len_EV - len(time_step))) 
        for time_step in data_EV 
    ]
    padded_data_ICE = [
        np.array(time_step + [[np.nan, np.nan, np.nan]] * (max_len_ICE - len(time_step)))
        for time_step in data_ICE 
    ]

    # Stack into 3D arrays (time steps, vehicles, attributes)
    data_EV_array = np.asarray(padded_data_EV)
    data_ICE_array = np.asarray(padded_data_ICE)

    # Transpose to shape (attributes, time steps, vehicles)
    data_EV_trans = np.transpose(data_EV_array, (2, 0, 1))
    data_ICE_trans = np.transpose(data_ICE_array, (2, 0, 1))

    fig.suptitle("Offered Vehicle Attributes Over Time")

    for i in range(cols):
        time_steps = np.arange(data_EV_trans.shape[1])

        # Scatter plot for individual EV and ICE data points at each time step
        for t in time_steps:
            # Plot all EV data points at time step `t` for attribute `i`
            axs[i].scatter(
                [t] * max_len_EV, data_EV_trans[i][t], color="green", alpha=0.2, s=10, label="EV Data" if t == 0 else ""
            )
            # Plot all ICE data points at time step `t` for attribute `i`
            axs[i].scatter(
                [t] * max_len_ICE, data_ICE_trans[i][t], color="blue", alpha=0.2, s=10, label="ICE Data" if t == 0 else ""
            )

        # Label each subplot
        axs[i].set_title(f"Attribute {i+1}")
        axs[i].set_xlabel("Time Steps")

    axs[-1].legend(loc="upper right")
    axs[0].set_ylabel("Attribute Value")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    save_path = os.path.join(fileName, "Plots")
    ensure_directory_exists(save_path)
    fig.savefig(f"{save_path}/vehicle_attribute_time_series_ON_OFFER.png", dpi=600, format="png")

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
    second_hand_merchant = data_controller.second_hand_merchant

    
    plot_emissions(social_network, fileName)
    plot_total_utility(social_network, fileName)
    plot_total_distance(social_network, fileName)
    plot_ev_adoption_rate(social_network, fileName)
    plot_ev_consider_rate(social_network, fileName)
    plot_tranport_users(social_network, fileName)
    #plot_cumulative_emissions(social_network, fileName)
    plot_vehicle_attribute_time_series(social_network, fileName)
    plot_research_time_series_multiple_firms([firm_manager.firms_list[0]], fileName)
    plot_second_hand_users(social_network, fileName)
    plot_second_hand_market_len(second_hand_merchant, fileName)
    #plot_segment_count(firm_manager, fileName)
    plot_segment_count_grid(firm_manager, fileName)
    plot_preferences(social_network, fileName)
    plot_sale_EV_prop(firm_manager, fileName)
    plot_history_research_type(firm_manager,fileName)
    plot_history_attributes_cars_on_sale_all_firms(social_network, fileName)
    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_16_56_27__11_11_2024")
