import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

def plot_policy_intensity_effects(data_array, policy_list, file_name,policy_info_dict, dpi=600):
    """
    Plots the effects of different policy intensities on EV uptake, policy distortion, and cumulative emissions.
    
    Parameters:
    - data_array: The array containing the simulation results.
    - policy_list: List of policies to plot.
    - file_name: Base file name for saving plots.
    - dpi: Dots per inch for the saved figures.
    """
    num_policies = len(policy_list)
    measures = ["EV Uptake", "Policy Distortion", "Cumulative Emissions"]
    num_measures = len(measures)
    
    # Create a figure with rows for measures and columns for policies
    fig, axes = plt.subplots(num_measures, num_policies, figsize=(20,10))
    
    # If there's only one measure or policy, axes will not be a 2D array
    if num_measures == 1:
        axes = axes[np.newaxis, :]
    if num_policies == 1:
        axes = axes[:, np.newaxis]
    
    for i, policy in enumerate(policy_list):
        # Extract data for the current policy
        policy_data = data_array[i]
        
        min = policy_info_dict['bounds_dict'][policy][0]
        max = policy_info_dict['bounds_dict'][policy][1]

        intensities = np.linspace(min, max, policy_data.shape[0])  # Assuming intensities are normalized from 0 to 1
        
        # Calculate mean and standard deviation across seeds
        mean_ev_uptake = np.mean(policy_data[:, :, 0], axis=1)
        std_ev_uptake = policy_data[:, :, 0].std(axis=1)
        mean_policy_distortion = np.mean(policy_data[:, :, 1], axis=1)
        std_policy_distortion = policy_data[:, :, 1].std(axis=1)
        mean_cum_em = np.mean(policy_data[:, :, 2], axis=1)
        std_cum_em = policy_data[:, :, 2].std(axis=1)
        
        # Plot EV Uptake
        axes[0, i].plot(intensities, mean_ev_uptake, label='Mean EV Uptake')
        axes[0, i].fill_between(intensities, mean_ev_uptake - std_ev_uptake, mean_ev_uptake + std_ev_uptake, alpha=0.2)
        
        # Plot Policy Distortion
        axes[1, i].plot(intensities, mean_policy_distortion, label='Mean Policy Distortion')
        axes[1, i].fill_between(intensities, mean_policy_distortion - std_policy_distortion, mean_policy_distortion + std_policy_distortion, alpha=0.2)

        # Plot Cumulative Emissions
        axes[2, i].plot(intensities, mean_cum_em, label='Mean Cumulative Emissions')
        axes[2, i].fill_between(intensities, mean_cum_em - std_cum_em, mean_cum_em + std_cum_em, alpha=0.2)

        axes[0, i].set_title(f'{policy}', fontsize = 8)
        
        axes[0, i].set_ylabel('EV Uptake', fontsize = 8)
        axes[1, i].set_ylabel('Policy Distortion', fontsize =8)
        axes[2, i].set_ylabel('Cumulative Emissions', fontsize = 8)

    fig.supxlabel('Policy Intensity')
    
    plt.tight_layout()
    plt.savefig(f'{file_name}/Plots/policy_intensity_effects.png', dpi=dpi)
    plt.show()

def main(file_name):
    # Load the data array and policy list
    data_array = load_object(file_name + "/Data", "data_array")
    policy_list = load_object(file_name + "/Data", "policy_list")
    policy_info_dict = load_object(file_name + "/Data", "policy_info_dict")

    # Plot the effects of policy intensities
    plot_policy_intensity_effects(data_array, policy_list, file_name, policy_info_dict)

if __name__ == "__main__":
    main(file_name="results/vary_single_policy_gen_15_54_03__13_02_2025")