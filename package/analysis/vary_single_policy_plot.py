import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_policy_intensity_effects_raw_lines(data_array, policy_list, file_name, policy_info_dict, dpi=600):
    """
    Plots the effects of different policy intensities on EV uptake, policy distortion, and cumulative emissions.
    
    Parameters:
    - data_array: The array containing the simulation results.
    - policy_list: List of policies to plot.
    - file_name: Base file name for saving plots.
    - policy_info_dict: Dictionary containing policy bounds and other information.
    - dpi: Dots per inch for the saved figures.
    """
    num_policies = len(policy_list)
    measures = ["EV Uptake", "Policy Distortion", "Cumulative Emissions", "Net Policy Cost"]
    num_measures = len(measures)
    
    # Create a figure with rows for measures and columns for policies
    fig, axes = plt.subplots(num_measures, num_policies, figsize=(15, 6))
    
    # Ensure axes is a 2D array
    if num_measures == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_policies == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i, policy in enumerate(policy_list):
        # Extract data for the current policy
        policy_data = data_array[i]
        
        # Get the intensity bounds for the policy
        min_val, max_val = policy_info_dict['bounds_dict'][policy]
        intensities = np.linspace(min_val, max_val, policy_data.shape[0])  # Generate intensity values
        
        # Flip the cost values (since it's cost)
        policy_data[:, :, 3] = -policy_data[:, :, 3]

        # Plot all measures for the current policy
        for j, measure in enumerate(measures):
            ax = axes[j, i]
            # Plot all seeds at once
            ax.plot(intensities, policy_data[:, :, j], alpha=1, linewidth=1)
            
            # Set titles and labels
            if j == 0:
                ax.set_title(f'{policy}', fontsize=10)
            ax.set_ylabel(measure, fontsize=9)

    # Set common x-axis label
    fig.supxlabel('Policy Intensity', fontsize=10)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{file_name}/Plots/policy_intensity_effects_raw_lines.png', dpi=dpi)

def plot_policy_intensity_effects_raw(data_array, policy_list, file_name, policy_info_dict, dpi=600):
    """
    Plots the effects of different policy intensities on EV uptake, policy distortion, and cumulative emissions.
    
    Parameters:
    - data_array: The array containing the simulation results.
    - policy_list: List of policies to plot.
    - file_name: Base file name for saving plots.
    - policy_info_dict: Dictionary containing policy bounds and other information.
    - dpi: Dots per inch for the saved figures.
    """
    num_policies = len(policy_list)
    measures = ["EV Uptake", "Policy Distortion", "Cumulative Emissions", "Net Policy Cost"]
    num_measures = len(measures)
    
    # Create a figure with rows for measures and columns for policies
    fig, axes = plt.subplots(num_measures, num_policies, figsize=(15, 6), )
    
    # Ensure axes is a 2D array
    if num_measures == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_policies == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i, policy in enumerate(policy_list):
        # Extract data for the current policy
        policy_data = data_array[i]
        
        # Get the intensity bounds for the policy
        min_val, max_val = policy_info_dict['bounds_dict'][policy]
        intensities = np.linspace(min_val, max_val, policy_data.shape[0])  # Generate intensity values
        

        policy_data[:, :, 3] = -policy_data[:, :, 3] #ITS COST SO FLIP IT


        # Define plotting function
        def plot_without_median(ax, intensities, label):
            for intensity_idx, intensity in enumerate(intensities):
                ax.scatter([intensity] * policy_data.shape[1], policy_data[intensity_idx, :, measures.index(label)], 
                           color='grey', alpha=0.1, s=5)
            


        # Plot EV Uptake
        plot_without_median(axes[0, i], intensities, "EV Uptake")

        # Plot Policy Distortion
        plot_without_median(axes[1, i], intensities, "Policy Distortion")

        # Plot Cumulative Emissions
        plot_without_median(axes[2, i], intensities,  "Cumulative Emissions")

        plot_without_median(axes[3, i], intensities, "Net Policy Cost")

        # Set titles and labels
        axes[0, i].set_title(f'{policy}', fontsize=10)
        axes[0, i].set_ylabel('EV Uptake', fontsize=9)
        axes[1, i].set_ylabel('Policy Distortion', fontsize=9)
        axes[2, i].set_ylabel('Cumulative Emissions', fontsize=9)
        axes[3, i].set_ylabel('Net Policy Cost', fontsize=9)

    # Plot EV Uptake
    #handles, labels_ = axes[0, 0].get_legend_handles_labels()

    # Create a single legend at the bottom of the figure
    #fig.legend(handles, ['Mean', 'Median'], loc='lower left', fontsize=10, ncol=2, frameon=False)

    # Set common x-axis label
    fig.supxlabel('Policy Intensity', fontsize=10)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{file_name}/Plots/policy_intensity_effects_raw.png', dpi=dpi)

import numpy as np
import matplotlib.pyplot as plt

def plot_policy_intensity_effects_means(data_array, policy_list, file_name, policy_info_dict, measures_dict, selected_measures, dpi=600):
    """
    Plots the effects of different policy intensities on specified measures (e.g., EV uptake, policy distortion, etc.).

    Parameters:
    - data_array: The array containing the simulation results.
    - policy_list: List of policies to plot.
    - file_name: Base file name for saving plots.
    - policy_info_dict: Dictionary containing policy bounds and other information.
    - measures_dict: Dictionary mapping measure names to their index in data_array.
    - selected_measures: List of measures to plot (subset of keys from measures_dict).
    - dpi: Dots per inch for the saved figures.
    """
    num_policies = len(policy_list)
    num_measures = len(selected_measures)

    fig, axes = plt.subplots(num_measures, num_policies, figsize=(15, 6 * num_measures), sharey="row")

    # Ensure axes is a 2D array for consistency
    if num_measures == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_policies == 1:
        axes = np.expand_dims(axes, axis=1)

    def plot_with_median(ax, intensities, mean_values, median_values, std_values, label):
        ax.plot(intensities, mean_values, label=f'Mean {label}')
        ax.fill_between(intensities, mean_values - std_values, mean_values + std_values, alpha=0.2, label="std")
        ax.plot(intensities, median_values, linestyle='dashed', label=f'Median {label}', color='red')
        ax.grid(alpha=0.3)

    for i, policy in enumerate(policy_list):
        policy_data = data_array[i]
        min_val, max_val = policy_info_dict['bounds_dict'][policy]
        intensities = np.linspace(min_val, max_val, policy_data.shape[0])

        for j, measure in enumerate(selected_measures):
            measure_idx = measures_dict[measure]
            ax = axes[j, i]

            mean_values = np.mean(policy_data[:, :, measure_idx], axis=1)
            median_values = np.median(policy_data[:, :, measure_idx], axis=1)
            std_values = np.std(policy_data[:, :, measure_idx], axis=1)

            plot_with_median(ax, intensities, mean_values, median_values, std_values, measure)

            if i == 0:
                ax.set_ylabel(measure, fontsize=9)
            if j == 0:
                ax.set_title(f'{policy}', fontsize=10)
            if j == num_measures - 1:
                ax.set_xlabel('Policy Intensity', fontsize=9)

    # Generate the string of indices for the selected measures
    measure_indices_str = "".join(str(measures_dict[measure]) for measure in selected_measures)

    # Use the string in the filename
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/policy_intensity_effects_means_{measure_indices_str}.png", dpi=dpi)


def main(file_name):
    # Load the data array, policy list, and policy info dictionary
    data_array = load_object(file_name + "/Data", "data_array")
    policy_list = load_object(file_name + "/Data", "policy_list")
    policy_info_dict = load_object(file_name + "/Data", "policy_info_dict")
    base_params = load_object(file_name + "/Data", "base_params")

    measures_dict = {
        "EV Uptake": 0,
        "Policy Distortion": 1,
        "Net Policy Cost": 2,
        "Cumulative Emissions": 3,
        "Driving Emissions": 4,
        "Production Emissions": 5,
        "Cumulative Utility": 6,
        "Cumulative Profit": 7
    }#

    selected_measures = [
        "EV Uptake",
        #"Policy Distortion",
        "Net Policy Cost",
        #"Cumulative Emissions",
        #"Driving Emissions",
        #"Production Emissions",
        "Cumulative Utility",
        "Cumulative Profit"
    ]
    plot_policy_intensity_effects_means(data_array, policy_list, file_name, policy_info_dict, measures_dict,selected_measures=selected_measures, dpi=300)
    selected_measures = [
        #"EV Uptake",
        #"Policy Distortion",
        #"Net Policy Cost",
        "Cumulative Emissions",
        "Driving Emissions",
        "Production Emissions",
        #"Cumulative Utility",
        #"Cumulative Profit"
    ]
    plot_policy_intensity_effects_means(data_array, policy_list, file_name, policy_info_dict, measures_dict,selected_measures=selected_measures, dpi=300)


    # Plot the effects of policy intensities
    #plot_policy_intensity_effects_raw_lines(data_array, policy_list, file_name, policy_info_dict, dpi=300)
    
    #plot_policy_intensity_effects_raw(data_array, policy_list, file_name, policy_info_dict,dpi=300)

    plt.show()
if __name__ == "__main__":
    main(file_name="results/vary_single_policy_gen_09_53_27__06_03_2025")