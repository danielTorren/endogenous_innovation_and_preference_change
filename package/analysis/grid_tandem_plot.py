import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_heatmap(file_name, data, vary_double, label, dpi=300):
    """
    Plots a heatmap of EV uptake across carbon price and emissions intensity decarbonization.

    Parameters:
    - file_name: The base file name where the results are saved.
    - results: Results array from the grid search.
    - vary_double: Dictionary containing the range and repetitions for carbon price and emissions intensity.
    - dpi: Dots per inch for the saved figure.
    """
    # Extract EV uptake (first metric in results)

    # Calculate mean EV uptake across seeds for each grid point
    data_mean = data.mean(axis=2)

    # Define grid values for carbon price and emissions intensity
    carbon_prices = np.linspace(vary_double["carbon_price"][0], vary_double["carbon_price"][1], vary_double["reps"])
    emissions_intensities = np.linspace(vary_double["decrb"][0], vary_double["decrb"][1], vary_double["reps"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    heatmap = ax.imshow(
        data_mean,
        extent=[carbon_prices.min(), carbon_prices.max(), emissions_intensities.min(), emissions_intensities.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax, label=f'{label}')

    # Customize the plot
    ax.set_xlabel('Carbon Price ($/kgCO2)', fontsize=12)
    ax.set_ylabel('Emissions Intensity Decarbonization', fontsize=12)
    #ax.set_title(f'{label} Heatmap', fontsize=14)
    ax.grid(alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/{label}_heatmap.png", dpi=dpi)


def plot_ev_uptake_lines(file_name, results, vary_double, dpi=300):
    """
    Plots EV uptake as lines, colored according to emissions intensity decarbonization.

    Parameters:
    - file_name: The base file name where the results are saved.
    - results: Results array from the grid search.
    - vary_double: Dictionary containing the range and repetitions for carbon price and emissions intensity.
    - dpi: Dots per inch for the saved figure.
    """
    # Extract EV uptake (first metric in results)
    ev_uptake = results[:, :, :, 0]  # Shape: (grid_resolution, grid_resolution, seed_repetitions)

    # Calculate mean EV uptake across seeds for each grid point
    ev_uptake_mean = ev_uptake.mean(axis=2)

    # Define grid values for carbon price and emissions intensity
    carbon_prices = np.linspace(vary_double["carbon_price"][0], vary_double["carbon_price"][1], vary_double["reps"])
    emissions_intensities = np.linspace(vary_double["decrb"][0], vary_double["decrb"][1], vary_double["reps"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a colormap for emissions intensity decarbonization
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=emissions_intensities.min(), vmax=emissions_intensities.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Plot EV uptake lines, colored by emissions intensity
    for i, ei in enumerate(emissions_intensities):
        ax.plot(
            carbon_prices,
            ev_uptake_mean[i, :],
            color=cmap(norm(ei)),
            label=f"EI = {ei:.2f}",
            linewidth=2,
            alpha=0.7
        )

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, label='Emissions Intensity Decarbonization')

    # Customize the plot
    ax.set_xlabel('Carbon Price ($/kgCO2)', fontsize=12)
    ax.set_ylabel('EV Uptake', fontsize=12)
    ax.set_title('EV Uptake vs Carbon Price (Colored by Emissions Intensity)', fontsize=14)
    ax.grid(alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/ev_uptake_lines.png", dpi=dpi)
    plt.show()


def main(file_name):
    """
    Main function to load data and plot the heatmap.
    """
    # Load data
    results = load_object(file_name + "/Data", "results_grid_search")
    vary_double = load_object(file_name + "/Data", "vary_double")
    #data.calc_EV_prop(), data.calc_total_policy_distortion(), data.calc_net_policy_distortion(), data.social_network.emissions_cumulative, data.social_network.emissions_cumulative_driving, data.social_network.emissions_cumulative_production, data.social_network.utility_cumulative, data.firm_manager.profit_cumulative
    
    # Plot EV uptake lines
    plot_ev_uptake_lines(file_name, results, vary_double, dpi=300)
    plt.show()
    # Plot heatmap
    plot_heatmap(file_name, results[:, :, :, 0], vary_double, "EV uptake",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 1], vary_double, "Cost",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 2], vary_double, "Net Costs",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 2], vary_double, "Total Emissions",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 4], vary_double, "Driving Emissions",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 5], vary_double, "Production Emissions",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 6], vary_double, "Utility",dpi=300)
    plot_heatmap(file_name, results[:, :, :, 6], vary_double, "Profit",dpi=300)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    main(file_name="results/grid_search_carbon_price_emissions_intensity_12_55_08__06_03_2025")  # Replace with the actual file name