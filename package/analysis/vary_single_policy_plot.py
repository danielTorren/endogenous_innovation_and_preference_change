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

def plot_policy_intensity_effects_means(data_array, policy_list, file_name, policy_info_dict, dpi=600):
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
        
        # Calculate mean, median, and standard deviation
        mean_ev_uptake = np.mean(policy_data[:, :, 0], axis=1)
        median_ev_uptake = np.median(policy_data[:, :, 0], axis=1)
        std_ev_uptake = np.std(policy_data[:, :, 0], axis=1)

        mean_policy_distortion = np.mean(policy_data[:, :, 1], axis=1)
        median_policy_distortion = np.median(policy_data[:, :, 1], axis=1)
        std_policy_distortion = np.std(policy_data[:, :, 1], axis=1)

        mean_cum_em = np.mean(policy_data[:, :, 2], axis=1)
        median_cum_em = np.median(policy_data[:, :, 2], axis=1)
        std_cum_em = np.std(policy_data[:, :, 2], axis=1)

        policy_data[:, :, 3] = -policy_data[:, :, 3] #ITS COST SO FLIP IT
        mean_net_policy_distortion = np.mean(policy_data[:, :, 3], axis=1)
        median_net_policy_distortion = np.median(policy_data[:, :, 3], axis=1)
        std_net_policy_distortion = np.std(policy_data[:, :, 3], axis=1)

        # Define plotting function
        def plot_with_median(ax, intensities, mean_values, median_values, std_values, label):
            
            ax.plot(intensities, mean_values, label=f'Mean {label}')
            ax.fill_between(intensities, mean_values - std_values, mean_values + std_values, alpha=0.2, label="std")
            ax.plot(intensities, median_values, linestyle='dashed', label=f'Median {label}', color='red')


        # Plot EV Uptake
        plot_with_median(axes[0, i], intensities, mean_ev_uptake, median_ev_uptake, std_ev_uptake, "EV Uptake")

        # Plot Policy Distortion
        plot_with_median(axes[1, i], intensities, mean_policy_distortion, median_policy_distortion, std_policy_distortion, "Policy Distortion")

        # Plot Cumulative Emissions
        plot_with_median(axes[2, i], intensities, mean_cum_em, median_cum_em, std_cum_em, "Cumulative Emissions")

        plot_with_median(axes[3, i], intensities, mean_net_policy_distortion, median_net_policy_distortion, std_net_policy_distortion, "Net Policy Cost")

        # Set titles and labels
        axes[0, i].set_title(f'{policy}', fontsize=10)
        axes[0, i].set_ylabel('EV Uptake', fontsize=9)
        axes[1, i].set_ylabel('Policy Distortion', fontsize=9)
        axes[2, i].set_ylabel('Cumulative Emissions', fontsize=9)
        axes[3, i].set_ylabel('Net Policy Cost', fontsize=9)

    # Plot EV Uptake
    handles, labels_ = axes[0, 0].get_legend_handles_labels()

    # Create a single legend at the bottom of the figure
    fig.legend(handles, ['Mean','Std',  'Median'], loc='lower left', fontsize=10, ncol=2, frameon=False)

    # Set common x-axis label
    fig.supxlabel('Policy Intensity', fontsize=10)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{file_name}/Plots/policy_intensity_effects_means.png', dpi=dpi)

def main(file_name):
    # Load the data array, policy list, and policy info dictionary
    data_array = load_object(file_name + "/Data", "data_array")
    policy_list = load_object(file_name + "/Data", "policy_list")
    policy_info_dict = load_object(file_name + "/Data", "policy_info_dict")
    base_params = load_object(file_name + "/Data", "base_params")
    print(base_params)
    
    # Plot the effects of policy intensities
    plot_policy_intensity_effects_raw_lines(data_array, policy_list, file_name, policy_info_dict, dpi=300)
    plot_policy_intensity_effects_means(data_array, policy_list, file_name, policy_info_dict, dpi=300)
    plot_policy_intensity_effects_raw(data_array, policy_list, file_name, policy_info_dict,dpi=300)

    plt.show()
if __name__ == "__main__":
    main(file_name="results/vary_single_policy_gen_16_19_15__05_03_2025")