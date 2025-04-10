import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path
from package.resources.utility import load_object, createFolder
from package.resources.utility import load_object

# Peace-sign style 1/3-circle marker
def third_circle_marker(start_angle, offset=0):
    angles = np.linspace(np.radians(start_angle + offset), np.radians(start_angle + 120 + offset), 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([verts, [0, 0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

def full_circle_marker():
    angles = np.linspace(0, 2 * np.pi, 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([verts, [verts[0]]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    return Path(verts, codes)

# Scale size based on range
def scale_marker_size(value, policy, policy_ranges, scale_marker):
    min_val = policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    if max_val - min_val == 0:
        return scale_marker
    norm = (value - min_val) / (max_val - min_val)
    return np.maximum(0, norm * scale_marker)

def plot_emissions_tradeoffs_from_triples(
        base_params,
        policy_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.94,
        max_ev_uptake=0.96,
        dpi=300
    ):
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 12), sharex=True)

    policy_bounds = {
        "Carbon_price": [0, 0.5],
        "Electricity_subsidy": [0.1, 0.5],
        "Production_subsidy": [4000, 13000]
    }

    policy_names = ["Carbon_price", "Electricity_subsidy", "Production_subsidy"]
    color_map = plt.get_cmap("Set1", len(policy_names))
    policy_colors = {policy: color_map(i) for i, policy in enumerate(policy_names)}
    policy_ranges = {policy: {"min": float("inf"), "max": float("-inf")} for policy in policy_names}
    scale_marker = 350

    for (elec_val, prod_val), entry in policy_outcomes.items():
        ev = entry["mean_EV_uptake"]
        if not (min_ev_uptake <= ev <= max_ev_uptake):
            continue

        carbon_val = entry["optimized_intensity"][0]

        # Update min/max
        for name, val in zip(policy_names, [carbon_val, elec_val, prod_val]):
            policy_ranges[name]["min"] = min(policy_ranges[name]["min"], val)
            policy_ranges[name]["max"] = max(policy_ranges[name]["max"], val)

        # Emissions, Cost, Utility
        e = entry["mean_emissions_cumulative"] * 1e-9
        u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
        c = entry["mean_net_cost"] * 1e-9

        # Errors
        emissions_array = np.array(entry["emissions_cumulative_driving"]) + np.array(entry["emissions_cumulative_production"])
        emissions_array *= 1e-9
        utility_array = np.array(entry["utility_cumulative"]) / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
        cost_array = np.array(entry["net_cost"]) * 1e-9
        n_seeds = len(emissions_array)
        e_err = 1.96 * np.std(emissions_array) / np.sqrt(n_seeds)
        u_err = 1.96 * np.std(utility_array) / np.sqrt(n_seeds)
        c_err = 1.96 * np.std(cost_array) / np.sqrt(n_seeds)

        # Marker sizes
        sizes = {name: scale_marker_size(val, name, policy_ranges, scale_marker)
                 for name, val in zip(policy_names, [carbon_val, elec_val, prod_val])}

        # === Top: Net Cost vs Emissions ===
        ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
        ax_top.scatter(e, c, s=scale_marker, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)

        ax_top.scatter(e, c, s=sizes["Carbon_price"], marker=third_circle_marker(0), color=policy_colors["Carbon_price"], edgecolor="black", zorder=2)
        ax_top.scatter(e, c, s=sizes["Electricity_subsidy"], marker=third_circle_marker(120), color=policy_colors["Electricity_subsidy"], edgecolor="black", zorder=2)
        ax_top.scatter(e, c, s=sizes["Production_subsidy"], marker=third_circle_marker(240), color=policy_colors["Production_subsidy"], edgecolor="black", zorder=2)

        # === Bottom: Utility vs Emissions ===
        ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
        ax_bottom.scatter(e, u, s=scale_marker, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)

        ax_bottom.scatter(e, u, s=sizes["Carbon_price"], marker=third_circle_marker(0), color=policy_colors["Carbon_price"], edgecolor="black", zorder=2)
        ax_bottom.scatter(e, u, s=sizes["Electricity_subsidy"], marker=third_circle_marker(120), color=policy_colors["Electricity_subsidy"], edgecolor="black", zorder=2)
        ax_bottom.scatter(e, u, s=sizes["Production_subsidy"], marker=third_circle_marker(240), color=policy_colors["Production_subsidy"], edgecolor="black", zorder=2)

    # BAU
    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
    bau_cost = outcomes_BAU["mean_net_cost"] * 1e-9
    ax_top.scatter(bau_em, bau_cost, s=scale_marker, color='black', edgecolor='black', label="BAU")
    ax_bottom.scatter(bau_em, bau_ut, s=scale_marker, color='black', edgecolor='black')

    # Axis labels
    ax_top.set_ylabel("Net Cost, bn $")
    ax_bottom.set_ylabel("Utility, bn $")
    ax_bottom.set_xlabel("Emissions, MTCO2")

    # Legend
    legend_elements = [
        Patch(facecolor=policy_colors[p], edgecolor='black',
              label=f"{p.replace('_', ' ')} ({policy_ranges[p]['min']:.2f}–{policy_ranges[p]['max']:.2f})")
        for p in policy_names
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label='BAU'))
    confidence = plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval')
    low_proxy = plt.Line2D([0], [0], marker=third_circle_marker(0),
                           color='gray', markerfacecolor='gray', markeredgecolor='black',
                           linestyle='None', label='Low Intensity', markersize=8)
    high_proxy = plt.Line2D([0], [0], marker=third_circle_marker(0),
                            color='gray', markerfacecolor='gray', markeredgecolor='black',
                            linestyle='None', label='High Intensity', markersize=12)
    legend_elements += [confidence, low_proxy, high_proxy]
    ax_bottom.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Save
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/emissions_tradeoffs/emissions_tradeoff_triples_peace.png", dpi=dpi)

def  main(fileName, fileName_BAU):

    base_params =  load_object(f"{fileName}/Data", "base_params")
    policy_outcomes =  load_object(f"{fileName}/Data", "policy_outcomes")
    outcomes_BAU = load_object(f"{fileName_BAU}/Data", "outcomes")

    plot_emissions_tradeoffs_from_triples(
        base_params,
        policy_outcomes,
        outcomes_BAU,
        file_name=fileName,
        min_ev_uptake=0.94,
        max_ev_uptake=0.96
    )
    plt.show()

if __name__ == "__main__":
    main(
        fileName="results/2d_grid_endogenous_third_00_47_36__10_04_2025",
        fileName_BAU="results/BAU_runs_12_22_12__10_04_2025"
    )
