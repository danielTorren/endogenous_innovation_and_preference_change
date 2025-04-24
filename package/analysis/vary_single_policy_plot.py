import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object


policy_titles = {
    "Carbon_price": "Carbon Price",
    "Electricity_subsidy": "Electricity Subsidy",
    "Adoption_subsidy": "New Car Rebate",
    "Adoption_subsidy_used": "Used Car Rebate",
    "Production_subsidy": "Production Subsidy"
}


def plot_policy_intensity_effects_means_95(height, title_dict, data_array, policy_list, file_name, policy_info_dict, measures_dict, selected_measures, dpi=300):
    """
    Plots the effects of different policy intensities on specified measures with 95% confidence intervals.
    """
    num_policies = len(policy_list)
    num_measures = len(selected_measures)

    fig, axes = plt.subplots(num_measures, num_policies, figsize=(12, height), sharey="row", sharex="col")

    # Ensure axes is a 2D array
    if num_measures == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_policies == 1:
        axes = np.expand_dims(axes, axis=1)

    def plot_with_median(ax, intensities, mean_values, median_values, ci_values, label):
        ax.plot(intensities, mean_values, label='Mean')
        ax.fill_between(intensities, mean_values - ci_values, mean_values + ci_values, alpha=0.2, label="95% Confidence Interval")
        ax.grid(alpha=0.3)

    for i, policy in enumerate(policy_list):
        policy_data = data_array[i]
        min_val, max_val = policy_info_dict['bounds_dict'][policy]
        intensities = np.linspace(min_val, max_val, policy_data.shape[0])

        # Define consistent tick positions across all measures for this policy
        num_ticks = 3
        xtick_positions = np.linspace(min_val, max_val, num_ticks)

        for j, measure in enumerate(selected_measures):
            measure_idx = measures_dict[measure]
            ax = axes[j, i]

            # Select and optionally scale data
            if measure == "EV Uptake":
                data_case = policy_data[:, :, measure_idx]
            else:
                data_case = policy_data[:, :, measure_idx] * 1e-9

            mean_values = np.mean(data_case, axis=1)
            median_values = np.median(data_case, axis=1)
            std_values = np.std(data_case, axis=1)
            n = policy_data.shape[1]
            ci_values = 1.96 * std_values / np.sqrt(n)

            # Optional scaling for Utility
            if measure == "Cumulative Utility":
                mean_values *= 12
                median_values *= 12
                ci_values *= 12

            if measure == "EV Uptake":
                ax.axhline(0.95, linestyle='--', label=r"$95\%$ EV Adoption", c="black")

            plot_with_median(ax, intensities, mean_values, median_values, ci_values, measure)

            # Axis labels and titles
            if i == 0:
                ax.set_ylabel(title_dict[measure], fontsize=9)
            if j == 0:
                ax.set_title(policy_titles[policy], fontsize=15)

            # Set consistent ticks for all columns
            ax.set_xticks(xtick_positions)

            if policy == "Carbon_price":
                # Convert to $/tonne CO2 (from $/kg)
                ax.set_xticklabels([f"{int(t * 1000)}" for t in xtick_positions])

    fig.supxlabel('Policy Intensity', fontsize=15)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', ncol=3, bbox_to_anchor=(0.01, 0.005), fontsize=9)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 1])  # Space for legend

    measure_indices_str = "".join(str(measures_dict[measure]) for measure in selected_measures)
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
        "Net Policy Cost":  "Cum. Net Cost, bn $",
        "Cumulative Emissions": "Cum. Emissions, MTC02",
        "Driving Emissions": "Cum. Emissions (Driving), MTC02",
        "Production Emissions": "Cum. Emissions (Production), MTC02",
        "Cumulative Utility": "Cum. Utility, bn $",
        "Cumulative Profit": "Cum. Profit, bn $"
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

    plot_policy_intensity_effects_means_95(15, title_dict,data_array, policy_list, file_name, policy_info_dict, measures_dict,selected_measures=selected_measures, dpi=300)
    
    selected_measures = [
        "EV Uptake",
        "Net Policy Cost",
        #"Cumulative Emissions",
        #"Driving Emissions",
        #"Production Emissions",
        "Cumulative Utility",
        "Cumulative Profit"
    ]
    plot_policy_intensity_effects_means_95(8.5, title_dict,data_array, policy_list, file_name, policy_info_dict, measures_dict,selected_measures=selected_measures, dpi=300)
    
    selected_measures = [
        #"EV Uptake",
        #"Net Policy Cost",
        "Cumulative Emissions",
        "Driving Emissions",
        "Production Emissions",
        #"Cumulative Utility",
        #"Cumulative Profit"
    ]
    plot_policy_intensity_effects_means_95(6.4,title_dict,data_array, policy_list, file_name, policy_info_dict, measures_dict,selected_measures=selected_measures, dpi=300)
    


    plt.show()
if __name__ == "__main__":
    main(file_name="results/vary_single_policy_gen_19_11_57__08_04_2025")#vary_single_policy_gen_18_57_38__03_04_2025")#vary_single_policy_gen_18_32_15__03_04_2025")#vary_single_policy_gen_16_43_02__06_03_2025