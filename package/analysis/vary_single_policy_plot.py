import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

def plot_policy_intensity_effects_means_95(title_dict, data_array, policy_list, file_name, policy_info_dict, measures_dict, selected_measures, dpi=300):
    """
    Plots the effects of different policy intensities on specified measures with 95% confidence intervals.

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

    fig, axes = plt.subplots(num_measures, num_policies, figsize=(10, 15), sharey="row", sharex="col")

    # Ensure axes is a 2D array for consistency
    if num_measures == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_policies == 1:
        axes = np.expand_dims(axes, axis=1)

    def plot_with_median(ax, intensities, mean_values, median_values, ci_values, label):
        ax.plot(intensities, mean_values, label=f'Mean')
        ax.fill_between(intensities, mean_values - ci_values, mean_values + ci_values, alpha=0.2, label="95% Confidence Interval")
        #ax.plot(intensities, median_values, linestyle='dashed', label=f'Median {label}', color='red')
        ax.grid(alpha=0.3)

    for i, policy in enumerate(policy_list):
        policy_data = data_array[i]
        min_val, max_val = policy_info_dict['bounds_dict'][policy]
        intensities = np.linspace(min_val, max_val, policy_data.shape[0])

        for j, measure in enumerate(selected_measures):
            measure_idx = measures_dict[measure]
            ax = axes[j, i]
            if measure == "EV Uptake":
                data_case = policy_data[:, :, measure_idx]
            else:
                data_case = policy_data[:, :, measure_idx]*1e-9

            mean_values = np.mean(data_case, axis=1)
            median_values = np.median(data_case, axis=1)
            std_values = np.std(data_case, axis=1)
            n = policy_data.shape[1]
            ci_values = 1.96 * std_values / np.sqrt(n)

            if measure == "Cumulative Utility":
                mean_values *= 12 
                median_values *= 12 
                ci_values *= 12 

            if measure == "EV Uptake":
                ax.axhline(0.95, linestyle='--', label = r"$95\%$ EV Adoption", c = "black")

            plot_with_median(ax, intensities, mean_values, median_values, ci_values, measure)

            if i == 0:
                ax.set_ylabel(title_dict[measure], fontsize=8)
            if j == 0:
                ax.set_title(f"{policy.replace('_', ' ')}", fontsize=10)

    fig.supxlabel('Policy Intensity', fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Create figure-wide legend below the subplots
    fig.legend(handles, labels, loc='lower left', ncol=3, fontsize=8, bbox_to_anchor=(0.01, 0.005))
    plt.tight_layout(rect=[0.01, 0.08, 0.98, 1])  # Leaves space at the bottom

    # Generate the string of indices for the selected measures
    measure_indices_str = "".join(str(measures_dict[measure]) for measure in selected_measures)

    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/policy_intensity_effects_means_{measure_indices_str}.png", dpi=dpi)
    plt.savefig(f"{file_name}/Plots/policy_intensity_effects_means_{measure_indices_str}.eps")

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
    }

    title_dict = {
        "EV Uptake": "EV Adoption Proportion",
        "Net Policy Cost":  "Net Cost, bn $",
        "Cumulative Emissions": "Emissions, MTC02",
        "Driving Emissions": "Emissions (Driving), MTC02",
        "Production Emissions": "Emissions (Production), MTC02",
        "Cumulative Utility": "Utility, bn $",
        "Cumulative Profit": "Profit, bn $"
    }

    selected_measures = [
        "EV Uptake",
        "Net Policy Cost",
        "Cumulative Emissions",
        "Driving Emissions",
        "Production Emissions",
        "Cumulative Utility",
        "Cumulative Profit"
    ]

    plot_policy_intensity_effects_means_95(title_dict,data_array, policy_list, file_name, policy_info_dict, measures_dict,selected_measures=selected_measures, dpi=300)
    
    plt.show()
if __name__ == "__main__":
    main(file_name="results/vary_single_policy_gen_21_35_31__01_04_2025")#vary_single_policy_gen_16_43_02__06_03_2025