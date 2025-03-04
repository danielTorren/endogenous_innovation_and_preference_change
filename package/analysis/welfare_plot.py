from package.resources.utility import (
    load_object
)
import matplotlib.pyplot as plt

from package.plotting_data.single_experiment_plot import save_and_show
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import createFolder, save_object

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

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,5), sharey=True)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D (works for 1-row cases)

    for ax, ((policy1, policy2), data) in zip(axes.flat, pairwise_outcomes.items()):
        p1_values = np.array([entry["policy1_value"] for entry in data])
        p2_values = np.array([entry["policy2_value"] for entry in data])
        mean_costs = np.array([entry[measure] for entry in data])
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])

        scatter = ax.scatter(p1_values, p2_values, c=mean_costs, s=80, edgecolor='k', cmap='viridis')
        #ax.set_title(f'{policy1} vs {policy2}', fontsize=10)
        ax.set_xlabel(f'{policy1} Intensity', fontsize=4)
        
        ax.grid(True)

        # Optionally annotate with EV uptake
        for (x, y, uptake) in zip(p1_values, p2_values, mean_uptake):
            ax.annotate(f'{uptake:.2f}', (x, y), fontsize=7, ha='right')

    axes[0][0].set_ylabel(f'Optimized Carbon Price Intensity', fontsize=8)

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


def plot_all_policy_pairs_welfare(pairwise_outcomes, file_name, dpi=600):
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

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,5), sharey=True)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D (works for 1-row cases)

    for ax, ((policy1, policy2), data) in zip(axes.flat, pairwise_outcomes.items()):
        p1_values = np.array([entry["policy1_value"] for entry in data])
        p2_values = np.array([entry["policy2_value"] for entry in data])


        mean_profit = np.array([entry["mean_profit_cumulative"] for entry in data])
        mean_utility = np.array([entry["mean_utility_cumulative"] for entry in data])
        mean_cost = np.array([entry["mean_total_cost"] for entry in data])

        mean_welfare = mean_utility + mean_profit - mean_cost

        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])

        scatter = ax.scatter(p1_values, p2_values, c=mean_welfare, s=80, edgecolor='k', cmap='viridis')
        #ax.set_title(f'{policy1} vs {policy2}', fontsize=10)
        ax.set_xlabel(f'{policy1} Intensity', fontsize=4)
        
        ax.grid(True)

        # Optionally annotate with EV uptake
        for (x, y, uptake) in zip(p1_values, p2_values, mean_uptake):
            ax.annotate(f'{uptake:.2f}', (x, y), fontsize=7, ha='right')

    axes[0][0].set_ylabel(f'Optimized Carbon Price Intensity', fontsize=8)

    # Remove empty subplots (if num_pairs doesn't fit perfectly into grid)
    for idx in range(num_pairs, num_rows * num_cols):
        fig.delaxes(axes.flat[idx])

    # Add colorbar across all subplots
    cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', shrink=0.9, aspect=25)
    cbar.set_label("welfare")

    #plt.suptitle('Policy Pair Intensity Sweeps', fontsize=14)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined figure
    plt.savefig(f'{file_name}/Plots/policy_pair_intensity_sweep_welfare.png', dpi=dpi)


def main(fileNames):
    # Load observed data
    pairwise_outcomes_complied = {}
    for fileName in fileNames:
        base_params = load_object(fileName + "/Data", "base_params")
        pairwise_outcomes = load_object(fileName + "/Data", "pairwise_outcomes")
        pairwise_outcomes_complied.update(pairwise_outcomes)
        save_object(pairwise_outcomes_complied, fileName + "/Data", "pairwise_outcomes_complied")
    
    plot_all_policy_pairs_welfare(pairwise_outcomes_complied, fileName) 

    plt.show()

if __name__ == "__main__":
    main(
        fileNames=[
            "results/endogenous_policy_intensity_22_02_22__28_02_2025",
            "results/endogenous_policy_intensity_17_23_17__28_02_2025",
            "results/endogenous_policy_intensity_17_21_20__28_02_2025",
            ]

    )
