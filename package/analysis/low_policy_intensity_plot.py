import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from sympy import N
from package.resources.utility import load_object
from matplotlib.lines import Line2D  # Add this at the top of your file if not already imported
from matplotlib.colors import ListedColormap

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
    dpi=300,
    single_policy_intensities=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import sem, t
    from matplotlib.lines import Line2D

    # duration_future, not duration_future - 1: load_in_controller now
    # actually simulates the full requested duration (see its fixed
    # off-by-one loop in package.resources.run).
    time_steps = np.arange(base_params["duration_future"])
    start = base_params["duration_burn_in"] + base_params["duration_calibration"]

    policy_titles_local = {
        "Carbon_price": "Carbon Price",
        "Electricity_subsidy": "Electricity Subsidy",
        "Adoption_subsidy": "New Car Rebate",
        "Adoption_subsidy_used": "Used Car Rebate",
        "Production_subsidy": "Production Subsidy"
    }

    # Add single policy outputs into outputs and top_policies
    outputs[("Carbon_price",)] = output_carbon_tax
    outputs[("Adoption_subsidy",)] = output_adoption_subsidy

    # Intensities come from the gen run's single_policy_intensities, which it
    # took from the pair-gen's endogenous single-policy solution. The fallbacks
    # are the values that used to be hardcoded here, for older results folders
    # saved before that file existed.
    intensities = single_policy_intensities or {}
    top_policies[("Carbon_price",)] = {
        "policy1_value": intensities.get("Carbon_price", 0.910),
        "policy2_value": None,
        "mean_ev_uptake": np.mean(output_carbon_tax["history_prop_EV"][:, -1]),
        "original_order": ("Carbon_price",)
    }
    top_policies[("Adoption_subsidy",)] = {
        "policy1_value": intensities.get("Adoption_subsidy", 36875.57),
        "policy2_value": None,
        "mean_ev_uptake": np.mean(output_adoption_subsidy["history_prop_EV"][:, -1]),
        "original_order": ("Adoption_subsidy",)
    }

    all_policies = sorted(set(p for k in outputs.keys() for p in k))

    okabe_ito_colors = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7', '#000000']
    color_map = ListedColormap(okabe_ito_colors)
    policy_colors = {p: color_map(i) for i, p in enumerate(all_policies)}
    marker_list = ['o', 's', '^', 'D', 'v']
    policy_markers = {p: marker_list[i % len(marker_list)] for i, p in enumerate(all_policies)}

    def label_from_key(key):
        if len(key) == 1:
            p = key[0]
            return f"{policy_titles_local.get(p, p)} ({round(top_policies[key]['policy1_value'], 2)})"
        else:
            p1, p2 = key
            return (f"{policy_titles_local.get(p1, p1)} ({round(top_policies[key]['policy1_value'], 2)}), "
                    f"{policy_titles_local.get(p2, p2)} ({round(top_policies[key]['policy2_value'], 2)})")

    def plot_line_with_ci(ax, data, color_line, color_marker, marker, linestyle='-', label=None):
        mean = np.nanmean(data, axis=0)
        ci = sem(data, axis=0, nan_policy='omit') * t.ppf(0.975, df=data.shape[0] - 1)
        ax.plot(time_steps, mean, color=color_line, marker='o', markevery=32,
                markerfacecolor=color_marker, markeredgecolor=color_marker, markersize=4,
                linestyle=linestyle, label=label)
        ax.fill_between(time_steps, mean - ci, mean + ci, color=color_line, alpha=0.2)

    def _add_vline(ax, annotation_height_prop=[0.2, 0.2, 0.2]):
        y_min, y_max = ax.get_ylim()
        annotation_height_0 = y_min + annotation_height_prop[0] * (y_max - y_min)
        ax.axvline(143, color='black', linestyle=':')
        ax.annotate("Policy end", xy=(143, annotation_height_0),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    def _get_style(key):
        if len(key) == 1:
            return policy_colors[key[0]], policy_colors[key[0]], policy_markers[key[0]]
        return policy_colors[key[0]], policy_colors[key[1]], policy_markers[key[1]]

    # --- Panel functions (closures over outputs, outputs_BAU, etc.) ---

    def panel_ev_share(ax, add_policy_labels=True):
        plot_line_with_ci(ax, outputs_BAU["history_prop_EV"][:, start:], 'black', 'black', 'o', '-', 'BAU - EV Adoption')
        plot_line_with_ci(ax, outputs_BAU["history_past_new_bought_vehicles_prop_ev"], 'black', 'black', 'o', '--', 'BAU - EV Sales')
        for key, output in outputs.items():
            color, color_marker, marker = _get_style(key)
            lbl = label_from_key(key) if add_policy_labels else None
            plot_line_with_ci(ax, output["history_prop_EV"][:, start:], color, color_marker, marker, '-', None)
            plot_line_with_ci(ax, output["history_past_new_bought_vehicles_prop_ev"], color, color_marker, marker, '--', lbl)
        ax.set_ylabel("EV Share", fontsize=16)
        _add_vline(ax, annotation_height_prop=[0.6, 0.2, 0.2])
        ax.legend(handles=[
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='EV Adoption'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='EV Sales'),
        ], loc='lower right', fontsize='small', ncols=2)

    def panel_ev_price(ax, add_policy_labels=False):
        for i, label_txt, linestyle in [(0, 'New', '-'), (1, 'Used', '--')]:
            plot_line_with_ci(ax, outputs_BAU["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                              'black', 'black', 'o', linestyle, f"BAU - {label_txt}")
        for key, output in outputs.items():
            color, color_marker, marker = _get_style(key)
            for i, linestyle in [(0, '-'), (1, '--')]:
                lbl = label_from_key(key) if (add_policy_labels and i == 0) else None
                plot_line_with_ci(ax, output["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                                  color, color_marker, marker, linestyle, lbl)
        ax.set_ylabel("EV Sale Price, $", fontsize=16)
        ax.legend(handles=[
            Line2D([0], [0], color="black", linestyle='-', label='New'),
            Line2D([0], [0], color="black", linestyle='--', label='Used'),
        ], loc='lower right', fontsize="small", ncols=2)
        _add_vline(ax, annotation_height_prop=[0.9, 0.2, 0.2])

    def panel_emissions(ax, cumulative=False, add_labels=True):
        transform = (lambda x: np.cumsum(x, axis=1) * 1e-9) if cumulative else (lambda x: x * 1e-9)
        ylabel = "Cumulative Emissions, MTCO2" if cumulative else "Flow Emissions, MTCO2"
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_emissions"]), 'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            color, color_marker, marker = _get_style(key)
            lbl = label_from_key(key) if add_labels else None
            plot_line_with_ci(ax, transform(output["history_total_emissions"]), color, color_marker, marker, '-', lbl)
        ax.set_ylabel(ylabel, fontsize=16)
        _add_vline(ax, annotation_height_prop=[0.5, 0.2, 0.2])

    def panel_utility(ax, cumulative=False, add_labels=False):
        transform = (lambda x: np.cumsum(x, axis=1) * 1e-9) if cumulative else (lambda x: x * 1e-9)
        ylabel = "Cumulative Utility, bn $" if cumulative else "Flow Utility, bn $"
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_utility"]), 'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            color, color_marker, marker = _get_style(key)
            lbl = label_from_key(key) if add_labels else None
            plot_line_with_ci(ax, transform(output["history_total_utility"]), color, color_marker, marker, '-', lbl)
        ax.set_ylabel(ylabel, fontsize=16)
        _add_vline(ax, annotation_height_prop=[0.2, 0.2, 0.2])

    def panel_car_age(ax, add_labels=False):
        plot_line_with_ci(ax, outputs_BAU["history_mean_car_age"], 'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            color, color_marker, marker = _get_style(key)
            lbl = label_from_key(key) if add_labels else None
            plot_line_with_ci(ax, output["history_mean_car_age"], color, color_marker, marker, '-', lbl)
        ax.set_ylabel("Car Age, months", fontsize=16)
        _add_vline(ax, annotation_height_prop=[0.5, 0.2, 0.2])

    def panel_net_cost(ax, add_labels=True):
        plot_line_with_ci(ax, outputs_BAU["history_policy_net_cost"] * 1e-9, 'black', 'black', 'o', '-', 'BAU')
        for key, output in outputs.items():
            color, color_marker, marker = _get_style(key)
            lbl = label_from_key(key) if add_labels else None
            plot_line_with_ci(ax, output["history_policy_net_cost"] * 1e-9, color, color_marker, marker, '-', lbl)
        ax.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
        _add_vline(ax, annotation_height_prop=[0.5, 0.2, 0.2])

    # --- X-axis helpers ---
    start_year = 2024
    tick_years = np.arange(start_year, start_year + (time_steps[-1] // 12) + 5, 5)
    tick_positions = (tick_years - start_year) * 12

    def set_xaxis(ax):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(year) for year in tick_years])
        ax.set_xlabel("Year", fontsize=16)

    # --- Combined figure ---
    fig, axs = plt.subplots(4, 2, figsize=(15, 16), sharex=True)

    panel_ev_share(axs[0, 0], add_policy_labels=True)
    panel_ev_price(axs[0, 1], add_policy_labels=False)
    panel_emissions(axs[1, 0], cumulative=False, add_labels=True)
    panel_emissions(axs[1, 1], cumulative=True, add_labels=False)
    panel_utility(axs[2, 0], cumulative=False, add_labels=False)
    panel_utility(axs[2, 1], cumulative=True, add_labels=False)
    panel_car_age(axs[3, 0], add_labels=False)
    panel_net_cost(axs[3, 1], add_labels=False)

    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.00), fontsize=10)

    for ax in axs[3]:
        set_xaxis(ax)

    fig.tight_layout(rect=[0.01, 0.051, 0.98, 1])
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(f"{fileName}/Plots/combined_policy_dashboard_with_utility_flow_cost_both.png", dpi=dpi)

    # Print cumulative emissions at final time step relative to BAU
    bau_cumulative_final = np.cumsum(outputs_BAU["history_total_emissions"], axis=1)[:, -1] * 1e-9
    bau_mean_final = np.nanmean(bau_cumulative_final)

    for key, output in outputs.items():
        policy_cumulative_final = np.cumsum(output["history_total_emissions"], axis=1)[:, -1] * 1e-9
        policy_mean_final = np.nanmean(policy_cumulative_final)
        delta = policy_mean_final - bau_mean_final
        direction = "reduction" if delta < 0 else "increase"
        print(f"{label_from_key(key)}: cumulative emissions {direction} of {abs(delta):.4f} MTCO2 relative to BAU")

    bau_cumulative_final = np.cumsum(outputs_BAU["history_total_emissions"], axis=1)[:, -1] * 1e-9
    bau_mean_final = np.nanmean(bau_cumulative_final)

    for key, output in outputs.items():
        policy_cumulative_final = np.cumsum(output["history_total_emissions"], axis=1)[:, -1] * 1e-9
        policy_mean_final = np.nanmean(policy_cumulative_final)
        pct_change = (policy_mean_final - bau_mean_final) / bau_mean_final * 100
        direction = "reduction" if pct_change < 0 else "increase"
        print(f"{label_from_key(key)}: cumulative emissions {direction} of {abs(pct_change):.2f}% relative to BAU")

    # --- Individual figures ---
    individual_panels = [
        ("ev_share",              lambda ax: panel_ev_share(ax, add_policy_labels=True)),
        ("ev_price",              lambda ax: panel_ev_price(ax, add_policy_labels=True)),
        ("flow_emissions",        lambda ax: panel_emissions(ax, cumulative=False, add_labels=True)),
        ("cumulative_emissions",  lambda ax: panel_emissions(ax, cumulative=True,  add_labels=True)),
        ("flow_utility",          lambda ax: panel_utility(ax,   cumulative=False, add_labels=True)),
        ("cumulative_utility",    lambda ax: panel_utility(ax,   cumulative=True,  add_labels=True)),
        ("car_age",               lambda ax: panel_car_age(ax,   add_labels=True)),
        ("net_cost",              lambda ax: panel_net_cost(ax,  add_labels=True)),
    ]

    for name, plot_fn in individual_panels:
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        plot_fn(ax_ind)
        set_xaxis(ax_ind)
        handles_ind, labels_ind = ax_ind.get_legend_handles_labels()
        if handles_ind:
            fig_ind.legend(handles_ind, labels_ind, loc='lower center', ncol=2,
                           bbox_to_anchor=(0.5, 0.00), fontsize=8)
            fig_ind.tight_layout(rect=[0.01, 0.18, 0.98, 1])
        else:
            fig_ind.tight_layout()
        fig_ind.savefig(f"{fileName}/Plots/individual_{name}.png", dpi=dpi)
        plt.close(fig_ind)


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

    # The intensities those two were run at, written by low_policy_intensity_gen.
    # Absent in folders generated before it started saving them.
    try:
        single_policy_intensities = load_object(fileName + "/Data", "single_policy_intensities")
    except FileNotFoundError:
        single_policy_intensities = None
        print("[!] No single_policy_intensities in this results folder -- labelling the "
              "single-policy lines with the old hardcoded intensities.")

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
        dpi=200,
        single_policy_intensities=single_policy_intensities
    )

    plt.show()



if __name__ == "__main__":
    main(fileName = "results/pair_low_intensity_policies_15_53_33__31_03_2026")
