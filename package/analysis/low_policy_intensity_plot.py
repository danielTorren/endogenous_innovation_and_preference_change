import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from sympy import N
from package.resources.utility import load_object
from matplotlib.lines import Line2D  # Add this at the top of your file if not already imported

policy_titles = {
    "Carbon_price": "Carbon Price",
    "Electricity_subsidy": "Electricity Subsidy",
    "Adoption_subsidy": "New Car Rebate",
    "Adoption_subsidy_used": "Used Car Rebate",
    "Production_subsidy": "Production Subsidy"
}

# Make sure to reuse the original add_vertical_lines function
def add_vertical_lines(ax, base_params, color='black', linestyle='--', annotation_height_prop=[0.2, 0.2, 0.2]):
    """
    Adds dashed vertical lines to the plot at specified steps with vertical annotations.
    """
    # Determine the middle of the plot if no custom height is provided
    y_min, y_max = ax.get_ylim()

    annotation_height_0 = y_min + annotation_height_prop[0]*(y_max - y_min)
    # Add vertical line with annotation
    ev_sale_start_time = 144 - 1
    ax.axvline(ev_sale_start_time, color="black", linestyle=':')
    ax.annotate("Policy end", xy=(ev_sale_start_time, annotation_height_0),
                rotation=90, verticalalignment='center', horizontalalignment='right',
                fontsize=8, color='black')
    
########################################################################################################################

def flip_policy_pair(data_dict, key1, key2):
    old_key = (key1, key2)
    new_key = (key2, key1)
    if old_key in data_dict:
        value = data_dict.pop(old_key)
        # If it's a best_entries-like structure with 'policy1_value' and 'policy2_value', flip them
        if isinstance(value, dict) and "policy1_value" in value and "policy2_value" in value:
            value = {
                "policy1_value": value["policy2_value"],
                "policy2_value": value["policy1_value"],
                "mean_ev_uptake": value["mean_ev_uptake"],
                "original_order": new_key  # update the original order
            }
        data_dict[new_key] = value

def plot_combined_policy_figures_with_utilty_flow_cost_both(
    base_params,
    fileName,
    outputs,
    outputs_BAU,
    top_policies,
    output_carbon_tax,
    output_adoption_subsidy,
    dpi=300
):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import sem, t
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    time_steps = np.arange(base_params["duration_future"] - 1)
    start = base_params["duration_burn_in"] + base_params["duration_calibration"]

    policy_titles = {
        "Carbon_price": "Carbon Price",
        "Electricity_subsidy": "Electricity Subsidy",
        "Adoption_subsidy": "New Car Rebate",
        "Adoption_subsidy_used": "Used Car Rebate",
        "Production_subsidy": "Production Subsidy"
    }

    # Add single policy outputs into outputs and top_policies
    outputs[("Carbon_price",)] = output_carbon_tax
    outputs[("Adoption_subsidy",)] = output_adoption_subsidy

    top_policies[("Carbon_price",)] = {
        "policy1_value": 0.983,
        "policy2_value": None,
        "mean_ev_uptake": np.mean(output_carbon_tax["history_prop_EV"][:, -1]),
        "original_order": ("Carbon_price",)
    }
    top_policies[("Adoption_subsidy",)] = {
        "policy1_value": 36638.50,
        "policy2_value": None,
        "mean_ev_uptake": np.mean(output_adoption_subsidy["history_prop_EV"][:, -1]),
        "original_order": ("Adoption_subsidy",)
    }

    all_policies = sorted(set(p for k in outputs.keys() for p in k))
    color_map = plt.get_cmap('Set1', 10)
    policy_colors = {p: color_map(i) for i, p in enumerate(all_policies)}
    marker_list = ['o', 's', '^', 'D', 'v']
    policy_markers = {p: marker_list[i % len(marker_list)] for i, p in enumerate(all_policies)}

    def label_from_key(key):
        if len(key) == 1:
            p = key[0]
            return f"{policy_titles.get(p, p)} ({round(top_policies[key]['policy1_value'], 2)})"
        else:
            p1, p2 = key
            return f"{policy_titles.get(p1, p1)} ({round(top_policies[key]['policy1_value'], 2)}), " \
                   f"{policy_titles.get(p2, p2)} ({round(top_policies[key]['policy2_value'], 2)})"

    def plot_line_with_ci(ax, data, color_line, color_marker, marker, linestyle='-', label=None):
        mean = np.nanmean(data, axis=0)
        ci = sem(data, axis=0, nan_policy='omit') * t.ppf(0.975, df=data.shape[0] - 1)
        ax.plot(time_steps, mean, color=color_line, marker='o', markevery=32,
                markerfacecolor=color_marker, markeredgecolor=color_marker, markersize=4,
                linestyle=linestyle, label=label)
        ax.fill_between(time_steps, mean - ci, mean + ci, color=color_line, alpha=0.2)

    def add_vertical_lines(ax, base_params, color='black', linestyle='--', annotation_height_prop=[0.2, 0.2, 0.2]):
        y_min, y_max = ax.get_ylim()
        annotation_height_0 = y_min + annotation_height_prop[0] * (y_max - y_min)
        ev_sale_start_time = 144 - 1
        ax.axvline(ev_sale_start_time, color=color, linestyle=':')
        ax.annotate("Policy end", xy=(ev_sale_start_time, annotation_height_0),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color=color)

    # --- EV Share: Adoption and Sales
    ax1 = axs[0, 0]
    plot_line_with_ci(ax1, outputs_BAU["history_prop_EV"][:, start:], 'black', 'black', 'o', '-', 'BAU - EV Adoption')
    plot_line_with_ci(ax1, outputs_BAU["history_past_new_bought_vehicles_prop_ev"], 'black', 'black', 'o', '--', 'BAU - EV Sales')

    for key, output in outputs.items():
        if len(key) == 1:
            color = color_marker = policy_colors[key[0]]
            marker = policy_markers[key[0]]
        else:
            color = policy_colors[key[0]]
            color_marker = policy_colors[key[1]]
            marker = policy_markers[key[1]]

        label = label_from_key(key)
        plot_line_with_ci(ax1, output["history_prop_EV"][:, start:], color, color_marker, marker, '-', None)
        plot_line_with_ci(ax1, output["history_past_new_bought_vehicles_prop_ev"], color, color_marker, marker, '--', label)

    ax1.set_ylabel("EV Share", fontsize=16)
    add_vertical_lines(ax1, base_params, annotation_height_prop=[0.6, 0.2, 0.2])
    # --- Legend
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='EV Adoption'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='EV Sales')
    ]
    ax1.legend(handles=custom_lines, loc='lower right', fontsize='small', ncols = 2)

    # --- EV Price: New and Used
    ax2 = axs[0, 1]
    for i, label_txt, linestyle in [(0, 'New', '-'), (1, 'Used', '--')]:
        plot_line_with_ci(ax2, outputs_BAU["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                          'black', 'black', 'o', linestyle, f"BAU - {label_txt}")

    for key, output in outputs.items():
        if len(key) == 1:
            color = color_marker = policy_colors[key[0]]
        else:
            color = policy_colors[key[0]]
            color_marker = policy_colors[key[1]]

        for i, linestyle in [(0, '-'), (1, '--')]:
            plot_line_with_ci(ax2, output["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                              color, color_marker, marker, linestyle)

    ax2.set_ylabel("EV Sale Price, $", fontsize=16)
    custom_legend = [
        Line2D([0], [0], color="black", linestyle='-', label='New'),
        Line2D([0], [0], color="black", linestyle='--', label='Used'),
    ]
    ax2.legend(handles=custom_legend, loc='lower right', fontsize="small", ncols = 2)
    add_vertical_lines(ax2, base_params, annotation_height_prop=[0.9, 0.2, 0.2])

    # --- Flow and Cumulative Emissions
    for idx, (ax, label, transform) in enumerate(zip(
        [axs[1, 0], axs[1, 1]],
        ["Flow Emissions, MTCO2", "Cumulative Emissions, MTCO2"],
        [lambda x: x * 1e-9, lambda x: np.cumsum(x, axis=1) * 1e-9])):
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_emissions"]),
                          'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            if len(key) == 1:
                color = color_marker = policy_colors[key[0]]
                marker = policy_markers[key[0]]
            else:
                color = policy_colors[key[0]]
                color_marker = policy_colors[key[1]]
                marker = policy_markers[key[1]]
            label_txt = label_from_key(key) if idx == 0 else None
            plot_line_with_ci(ax, transform(output["history_total_emissions"]),
                              color, color_marker, marker, '-', label_txt)
        ax.set_ylabel(label, fontsize=16)
        add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2])

    # --- Flow and Cumulative Utility
    for idx, (ax, key_name, transform, label) in enumerate(zip(
        [axs[2, 0], axs[2, 1]],
        ["history_total_utility", "history_total_utility"],
        [lambda x: x * 1e-9, lambda x: np.cumsum(x, axis=1) * 1e-9],
        ["Flow Utility, bn $", "Cumulative Utility, bn $"])):
        plot_line_with_ci(ax, transform(outputs_BAU[key_name]), 'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            if len(key) == 1:
                color = color_marker = policy_colors[key[0]]
                marker = policy_markers[key[0]]
            else:
                color = policy_colors[key[0]]
                color_marker = policy_colors[key[1]]
                marker = policy_markers[key[1]]
            label_txt = label_from_key(key) if idx == 0 else None
            plot_line_with_ci(ax, transform(output[key_name]),
                              color, color_marker, marker, '-', label_txt)
        ax.set_ylabel(label, fontsize=16)
        add_vertical_lines(ax, base_params, annotation_height_prop=[0.2, 0.2, 0.2])

    # --- Car Age and Net Cost
    for idx, (ax, key_name, label, scale) in enumerate(zip(
        [axs[3, 0], axs[3, 1]],
        ["history_mean_car_age", "history_policy_net_cost"],
        ["Car Age, months", "Cumulative Net Cost, bn $"],
        [1, 1e-9])):
        plot_line_with_ci(ax, outputs_BAU[key_name] * scale, 'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            if len(key) == 1:
                color = color_marker = policy_colors[key[0]]
                marker = policy_markers[key[0]]
            else:
                color = policy_colors[key[0]]
                color_marker = policy_colors[key[1]]
                marker = policy_markers[key[1]]
            label_txt = label_from_key(key) if idx == 1 else None
            plot_line_with_ci(ax, output[key_name] * scale, color, color_marker, marker, '-', label_txt)
        ax.set_ylabel(label, fontsize=16)
        add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2])

    # Legend and formatting
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.00), fontsize=10)

    # X-axis labels
    start_year = 2024
    tick_years = np.arange(start_year, start_year + (time_steps[-1] // 12) + 5, 5)
    tick_positions = (tick_years - start_year) * 12
    for ax in axs[3]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(year) for year in tick_years])
        ax.set_xlabel("Year", fontsize=16)

    fig.tight_layout(rect=[0.01, 0.051, 0.98, 1])
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(f"{fileName}/Plots/combined_policy_dashboard_with_utility_flow_cost_both.png", dpi=dpi)


def main(fileName):
    base_params = load_object(fileName + "/Data", "base_params")
    outputs_BAU = load_object(fileName + "/Data", "outputs_BAU")
    outputs = load_object(fileName + "/Data", "outputs")
    top_policies = load_object(fileName + "/Data", "top_policies")

    # Flip ordering for consistency if needed
    flip_policy_pair(top_policies, 'Adoption_subsidy_used', 'Carbon_price')
    flip_policy_pair(outputs, 'Adoption_subsidy_used', 'Carbon_price')
    flip_policy_pair(top_policies, 'Adoption_subsidy', 'Carbon_price')
    flip_policy_pair(outputs, 'Adoption_subsidy', 'Carbon_price')

    # Load single policy results
    outputs_carbon_tax = load_object(fileName + "/Data", "outputs_carbon_tax")
    outputs_adoption_subsidy = load_object(fileName + "/Data", "outputs_adoption_subsidy")

    # Add to outputs using tuple keys
    outputs[("Carbon_price",)] = outputs_carbon_tax
    outputs[("Adoption_subsidy",)] = outputs_adoption_subsidy

    # Plot
    plot_combined_policy_figures_with_utilty_flow_cost_both(
        base_params,
        fileName,
        outputs,
        outputs_BAU,
        top_policies,
        outputs_carbon_tax,
        outputs_adoption_subsidy,
        dpi=300
    )

    plt.show()

    

if __name__ == "__main__":
    main(fileName = "results/pair_low_intensity_policies_19_31_05__24_04_2025")
