import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

def plot_policy_intensity_effects(data_array, policy_list, file_name, dpi=600):
    """
    Plots the effects of different policy intensities on EV uptake, policy distortion, and cumulative emissions.
    
    Parameters:
    - data_array: The array containing the simulation results.
    - policy_list: List of policies to plot.
    - file_name: Base file name for saving plots.
    - dpi: Dots per inch for the saved figures.
    """
    num_policies = len(policy_list)
    fig, axes = plt.subplots(num_policies, 3, figsize=(15, 5 * num_policies))
    
    for i, policy in enumerate(policy_list):
        # Extract data for the current policy
        policy_data = data_array[i]
        intensities = np.linspace(0, 1, policy_data.shape[0])  # Assuming intensities are normalized from 0 to 1
        
        print(policy_data.shape)
        #print(policy_data[:, 0])

        # Calculate mean and standard deviation across seeds
        mean_ev_uptake = np.mean(policy_data[:,:, 0], axis=1)
        print(mean_ev_uptake , mean_ev_uptake.shape)
        quit()
        std_ev_uptake = policy_data.std(axis=1)[:, 0]
        mean_policy_distortion = policy_data.mean(axis=1)[:, 1]
        std_policy_distortion = policy_data.std(axis=1)[:, 1]
        mean_cum_em = policy_data.mean(axis=1)[:, 2]
        std_cum_em = policy_data.std(axis=1)[:, 2]
        
        # Plot EV Uptake
        axes[i, 0].plot(intensities, mean_ev_uptake, label='Mean EV Uptake')
        axes[i, 0].fill_between(intensities, mean_ev_uptake - std_ev_uptake, mean_ev_uptake + std_ev_uptake, alpha=0.2)
        axes[i, 0].set_title(f'{policy} - EV Uptake')
        axes[i, 0].set_xlabel('Policy Intensity')
        axes[i, 0].set_ylabel('EV Uptake')
        axes[i, 0].legend()
        
        # Plot Policy Distortion
        axes[i, 1].plot(intensities, mean_policy_distortion, label='Mean Policy Distortion')
        axes[i, 1].fill_between(intensities, mean_policy_distortion - std_policy_distortion, mean_policy_distortion + std_policy_distortion, alpha=0.2)
        axes[i, 1].set_title(f'{policy} - Policy Distortion')
        axes[i, 1].set_xlabel('Policy Intensity')
        axes[i, 1].set_ylabel('Policy Distortion')
        axes[i, 1].legend()
        
        # Plot Cumulative Emissions
        axes[i, 2].plot(intensities, mean_cum_em, label='Mean Cumulative Emissions')
        axes[i, 2].fill_between(intensities, mean_cum_em - std_cum_em, mean_cum_em + std_cum_em, alpha=0.2)
        axes[i, 2].set_title(f'{policy} - Cumulative Emissions')
        axes[i, 2].set_xlabel('Policy Intensity')
        axes[i, 2].set_ylabel('Cumulative Emissions')
        axes[i, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{file_name}_policy_intensity_effects.png', dpi=dpi)
    plt.show()

def main(file_name):
    # Load the data array and policy list
    data_array = load_object(file_name + "/Data", "data_array")
    policy_list = load_object(file_name + "/Data", "policy_list")

    # Plot the effects of policy intensities
    plot_policy_intensity_effects(data_array, policy_list, file_name)

if __name__ == "__main__":
    main(file_name="results/vary_single_policy_gen_13_14_52__13_02_2025")