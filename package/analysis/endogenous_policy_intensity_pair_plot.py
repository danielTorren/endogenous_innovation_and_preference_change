from package.resources.utility import (
    load_object
)
import matplotlib.pyplot as plt

from package.plotting_data.single_experiment_plot import save_and_show
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import createFolder, save_object
from matplotlib.colors import Normalize

def plot_policy_pair_intensities(pairwise_outcomes, policy1_name, policy2_name, file_name, dpi=600):
    """
    Plots the sweep of Policy 1 intensity vs optimized Policy 2 intensity.
    
    Parameters:
    - pairwise_outcomes: Dict of results from policy_pair_sweep.
    - policy1_name: Name of Policy 1 (x-axis).
    - policy2_name: Name of Policy 2 (y-axis, optimized).
    - file_name: Base file name for saving plots.
    - dpi: Resolution for saved plots.
    """
    # Extract the data for this policy pair
    data = pairwise_outcomes[(policy1_name, policy2_name)]

    p1_values = np.array([entry["policy1_value"] for entry in data])
    p2_values = np.array([entry["policy2_value"] for entry in data])
    mean_costs = np.array([entry["mean_total_cost"] for entry in data])
    mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])


    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the policy pair intensity relationship
    scatter = ax.scatter(p1_values, p2_values, c=mean_costs, s=100, edgecolor='k', cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Mean Total Cost')

    # Optionally, add text labels for each point (show EV uptake)
    for (x, y, uptake) in zip(p1_values, p2_values, mean_uptake):
        ax.annotate(f'{uptake:.2f}', (x, y), fontsize=8, ha='right')

    ax.set_xlabel(f'{policy1_name} Intensity')
    ax.set_ylabel(f'Optimized {policy2_name} Intensity')
    ax.set_title(f'{policy2_name} Optimized Intensity vs {policy1_name} Intensity')
    
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    plt.savefig(f'{file_name}/Plots/{policy1_name}_{policy2_name}_intensity_plot.png', dpi=dpi)
    plt.show()


def plot_all_policy_pairs(pairwise_outcomes, file_name,measure, dpi=600):
    """
    Plots all policy pair sweeps as subfigures in one figure.
    
    Parameters:
    - pairwise_outcomes: Dict of results from policy_pair_sweep.
    - file_name: Base file name for saving plots.
    - dpi: Resolution for saved plots.
    """
    num_pairs = len(pairwise_outcomes)
    num_cols =   num_pairs # You can adjust this for different layouts
    num_rows = (num_pairs + num_cols - 1) // num_cols  # Rows needed to fit all pairs

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,5), constrained_layout = True)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D (works for 1-row cases)

    for ax, ((policy1, policy2), data) in zip(axes.flat, pairwise_outcomes.items()):
        p1_values = np.array([entry["policy1_value"] for entry in data])
        p2_values = np.array([entry["policy2_value"] for entry in data])
        mean_costs = np.array([entry[measure] for entry in data])
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])

        scatter = ax.scatter(p1_values, p2_values, c=mean_costs, s=40, edgecolor='k', cmap='viridis')
        ax.set_title(f'{policy1} vs {policy2}', fontsize=4)
        ax.set_xlabel(f'{policy1} Intensity', fontsize=4)
        ax.set_ylabel(f'{policy2} Intensity', fontsize=4)
        
        ax.grid(True)

        # Optionally annotate with EV uptake
        for (x, y, uptake) in zip(p1_values, p2_values, mean_uptake):
            ax.annotate(f'{uptake:.2f}', (x, y), fontsize=7, ha='right')

    #axes[0][0].set_ylabel(f'Optimized Carbon Price Intensity', fontsize=8)

    # Remove empty subplots (if num_pairs doesn't fit perfectly into grid)
    for idx in range(num_pairs, num_rows * num_cols):
        fig.delaxes(axes.flat[idx])

    # Add colorbar across all subplots
    cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', shrink=0.9, aspect=25)
    cbar.set_label(measure)

    #plt.suptitle('Policy Pair Intensity Sweeps', fontsize=14)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined figure
    plt.savefig(f'{file_name}/Plots/policy_pair_intensity_sweep_{measure}.png', dpi=dpi)



def plot_all_policy_combinations_on_one_plot(pairwise_outcomes_complied, file_name, dpi=600):
    """
    Plots all policy combinations on a single scatter plot with individual policy normalization.

    Parameters:
    - pairwise_outcomes_complied: Dictionary containing all policy combinations and their outcomes.
    - file_name: Base file name for saving plots.
    - dpi: Resolution for saved plots.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a colormap for policy combinations
    cmap = plt.get_cmap('viridis')

    # Store min and max values for each policy
    policy_min_max = {}

    # Loop through each policy pair to collect min and max values
    for (policy1_name, policy2_name), data in pairwise_outcomes_complied.items():
        p1_values = np.array([entry["policy1_value"] for entry in data])
        p2_values = np.array([entry["policy2_value"] for entry in data])

        # Store min and max values for each policy
        if policy1_name not in policy_min_max:
            policy_min_max[policy1_name] = (p1_values.min(), p1_values.max())
        if policy2_name not in policy_min_max:
            policy_min_max[policy2_name] = (p2_values.min(), p2_values.max())

    # Loop through each policy pair and plot on the same axes
    for (policy1_name, policy2_name), data in pairwise_outcomes_complied.items():
        # Extract data
        p1_values = np.array([entry["policy1_value"] for entry in data])
        p2_values = np.array([entry["policy2_value"] for entry in data])
        mean_costs = np.array([entry["mean_total_cost"] for entry in data])
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])

        # Filter data based on EV uptake
        mask = (mean_uptake >= 0.945) & (mean_uptake <= 0.955)
        p1_values_filtered = p1_values[mask]
        p2_values_filtered = p2_values[mask]
        mean_costs_filtered = mean_costs[mask]
        mean_uptake_filtered = mean_uptake[mask]

        # Normalize each policy individually
        p1_min, p1_max = policy_min_max[policy1_name]
        p2_min, p2_max = policy_min_max[policy2_name]

        p1_normalized = (p1_values_filtered - p1_min) / (p1_max - p1_min)
        p2_normalized = (p2_values_filtered - p2_min) / (p2_max - p2_min)

        # Calculate policy combination intensity (optional, for coloring)
        policy_intensity = np.sqrt(p1_normalized**2 + p2_normalized**2)

        # Plot the policy pair intensity relationship
        scatter = ax.scatter(
            p1_normalized,
            p2_normalized,
            c=policy_intensity,
            s=100,
            edgecolor='k',
            cmap=cmap,
            label=f'{policy1_name} vs {policy2_name}'
        )

        # Annotate each point with normalized policy intensity
        for (x, y) in zip(p1_normalized, p2_normalized):
            ax.annotate(f'({x:.2f}, {y:.2f})', (x, y), fontsize=8, ha='right')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Policy Combination Intensity')

    # Add legend with min and max values for each policy
    legend_text = []
    for policy, (min_val, max_val) in policy_min_max.items():
        legend_text.append(f'{policy} ({min_val:.2f}, {max_val:.2f})')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Policy Ranges:\n" + "\n".join(legend_text))

    # Set labels and title
    ax.set_xlabel('Normalized Policy 1 Intensity')
    ax.set_ylabel('Normalized Policy 2 Intensity')
    ax.set_title('All Policy Combinations (EV Uptake: 0.945 - 0.955)')

    # Add grid
    ax.grid(True)

    # Save to file
    plt.tight_layout()
    plt.savefig(f'{file_name}/Plots/all_policy_combinations_normalized_plot.png', dpi=dpi)


def main(fileNames):
    # Load observed data
    pairwise_outcomes_complied = {}

    fileName = fileNames[0]

    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(fileName + "/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(fileName + "/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)
            save_object(pairwise_outcomes_complied, fileName + "/Data", "pairwise_outcomes_complied")

    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "mean_total_cost")
    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "mean_emissions_cumulative")
    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "mean_utility_cumulative")
    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "mean_profit_cumulative")
    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "sd_ev_uptake")
    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "mean_emissions_cumulative_driving")
    plot_all_policy_pairs(pairwise_outcomes_complied, fileName, "mean_emissions_cumulative_production")

    plt.show()

if __name__ == "__main__":
    main(
        fileNames=[
            "results/endogenous_policy_intensity_16_35_14__05_03_2025"
            ]

    )
