import copy
from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from package.resources.utility import (
    createFolder, save_object, produce_name_datetime
)

# Marker helper functions
def half_circle_marker(start, end, offset=-45):
    angles = np.linspace(np.radians(start + offset), np.radians(end + offset), 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([verts, [0, 0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

# Intensity-to-marker-size scaling
def scale_marker_size(value, policy, policy_ranges, scale_marker):
    min_val = policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    if max_val - min_val == 0:
        return scale_marker  # Fixed size if no variation
    norm = (value - min_val) / (max_val - min_val)
    return np.maximum(0, norm * scale_marker)

def full_circle_marker():
    angles = np.linspace(0, 2 * np.pi, 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([verts, [verts[0]]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    return Path(verts, codes)



def plot_emissions_tradeoffs_from_outcomes(
        base_params, 
        pairwise_outcomes_complied, 
        single_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.9, 
        max_ev_uptake=1.0, 
        dpi=300
        ):
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # --- Setup
    color_map = plt.get_cmap('Set1', 10)
    all_policies = sorted({p for pair in pairwise_outcomes_complied for p in pair})
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}
    policy_ranges = {policy: {"min": 0, "max": 0} for policy in all_policies}

    plotted_points = []
    scale_marker = 350
    scale_small = 1
    ########################################################################################

    # --- Create zoom-in inset axes for the top plot
    axins = inset_axes(ax_top, width="70%", height="45%", 
                    bbox_to_anchor=(0.01, 0.52, 1, 1),
                    bbox_transform=ax_top.transAxes,
                    loc='lower left'
                    )
    
    e_min = 0.125
    e_max = 0.15#0.145
    c_min = -0.08
    c_max = 0.3

    axins.set_xlim( e_min, e_max)
    axins.set_ylim(c_min, c_max)

    axins.set_xticks([])
    axins.set_yticks([])

    ##########################################################################################

    # --- Create zoom-in inset axes for the top plot
    axins2 = inset_axes(ax_bottom, width="30%", height="30%", 
                    bbox_to_anchor=(0.15, 0.01, 1, 1),
                    bbox_transform=ax_bottom.transAxes,
                    loc='lower left'
                    )
    
    e_min2 = 0.131
    e_max2 = 0.135
    u_min = 72
    u_max = 78

    axins2.set_xlim(e_min2, e_max2)
    axins2.set_ylim(u_min, u_max)

    axins2.set_xticks([])
    axins2.set_yticks([])

    ##########################################################################################
    
    #axins.tick_params(axis='both', which='major', labelsize=8)

    # --- Gather data from pairs
    for (policy1, policy2), results in pairwise_outcomes_complied.items():
        for entry in results:
            ev = entry["mean_ev_uptake"]
            if (min_ev_uptake <= ev <= max_ev_uptake):

                e = entry["mean_emissions_cumulative"] * 1e-9
                u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                c = entry["mean_net_cost"] * 1e-9

                p1_val = entry["policy1_value"]
                p2_val = entry["policy2_value"]

                policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], p1_val)
                policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], p2_val)

                plotted_points.append((e, u, c, policy1, policy2, p1_val, p2_val))

    # --- BAU
    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
    bau_cost = outcomes_BAU["mean_net_cost"] * 1e-9

    ax_top.scatter(bau_em, bau_cost, s=scale_marker, color='black', edgecolor='black', label="BAU")
    ax_bottom.scatter(bau_em, bau_ut, s=scale_marker, color='black', edgecolor='black')
    # Also plot BAU on the inset axes
    if e_min <= bau_em <= e_max and c_min <= bau_cost <= c_max :
        axins.scatter(bau_em, bau_cost, s=scale_marker*scale_small, color='black', edgecolor='black')
    if e_min2 <= bau_em <= e_max2 and u_min <= bau_ut <= u_max :
        axins2.scatter(bau_em, bau_ut, s=scale_marker*scale_small, color='black', edgecolor='black')

    for (policy1, policy2), results in pairwise_outcomes_complied.items():
        for entry in results:
            print( (policy1, policy2),entry["mean_ev_uptake"])
            if (min_ev_uptake <= entry["mean_ev_uptake"] <= max_ev_uptake):
                e = entry["mean_emissions_cumulative"] * 1e-9
                u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                c = entry["mean_net_cost"] * 1e-9

                p1_val = entry["policy1_value"]
                p2_val = entry["policy2_value"]

                entry["emissions_cumulative"] = entry["emissions_cumulative_driving"] + entry["emissions_cumulative_production"]
                emissions_array = np.array(entry["emissions_cumulative"]) * 1e-9
                utility_array = np.array(entry["utility_cumulative"]) / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                cost_array = np.array(entry["net_cost"]) * 1e-9

                n_seeds = len(emissions_array)
                e_err = 1.96 * np.std(emissions_array) / np.sqrt(n_seeds)
                u_err = 1.96 * np.std(utility_array) / np.sqrt(n_seeds)
                c_err = 1.96 * np.std(cost_array) / np.sqrt(n_seeds)

                policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], p1_val)
                policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], p2_val)

                color1 = policy_colors[policy1]
                color2 = policy_colors[policy2]
                size1 = scale_marker_size(p1_val, policy1, policy_ranges, scale_marker)
                size2 = scale_marker_size(p2_val, policy2, policy_ranges, scale_marker)

                # --- Top Panel: Net Cost vs Emissions
                ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                ax_top.scatter(e, c, s=scale_marker, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle = "--", alpha = 0.5)
                ax_top.scatter(e, c, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                ax_top.scatter(e, c, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)
                
                # --- Plot in inset axes if within zoom range
                if e_min <= e <= e_max and c_min <= c <= c_max :
                    axins.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                    axins.scatter(e, c, s=scale_marker*scale_small, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)
                    axins.scatter(e, c, s=size1*scale_small, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                    axins.scatter(e, c, s=size2*scale_small, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)

                # --- Plot in inset axes if within zoom range
                if e_min2 <= e <= e_max2 and u_min <= u <= u_max :
                    axins2.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                    axins2.scatter(e, u, s=scale_marker*scale_small, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)
                    axins2.scatter(e, u, s=size1*scale_small, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                    axins2.scatter(e, u, s=size2*scale_small, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)

                # --- Bottom Panel: Utility vs Emissions
                ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                ax_bottom.scatter(e, u, s=scale_marker, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle = "--", alpha = 0.5)
                ax_bottom.scatter(e, u, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                ax_bottom.scatter(e, u, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)

    # --- Plot Single Policy Outcomes
    for policy, entry in single_outcomes.items():
        print(policy,entry["mean_EV_uptake"])
        ev = entry["mean_EV_uptake"]
        if (min_ev_uptake <= ev <= max_ev_uptake):
            e = entry["mean_emissions_cumulative"] * 1e-9
            u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
            c = entry["mean_net_cost"] * 1e-9

            entry["emissions_cumulative"] = entry["emissions_cumulative_driving"] + entry["emissions_cumulative_production"]
            emissions_array = np.array(entry["emissions_cumulative"]) * 1e-9
            utility_array = np.array(entry["utility_cumulative"]) / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
            cost_array = np.array(entry["net_cost"]) * 1e-9

            n_seeds = len(emissions_array)
            e_err = 1.96 * np.std(emissions_array) / np.sqrt(n_seeds)
            u_err = 1.96 * np.std(utility_array) / np.sqrt(n_seeds)
            c_err = 1.96 * np.std(cost_array) / np.sqrt(n_seeds)

            color = policy_colors.get(policy, 'gray')
            size = scale_marker

            # --- Top Panel: Net Cost vs Emissions
            ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
            ax_top.scatter(e, c, s=size, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)
            
            # --- Plot in inset axes if within zoom range
            if e_min <= e <= e_max and c_min <= c <= c_max :
                axins.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                axins.scatter(e, c, s=size*scale_small, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)
            
            # --- Plot in inset axes if within zoom range
            if e_min2 <= e <= e_max2 and u_min <= u <= u_max :
                axins2.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                axins2.scatter(e, u, s=size*scale_small, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)

            # --- Bottom Panel: Utility vs Emissions
            ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
            ax_bottom.scatter(e, u, s=size, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)

    # --- Configure zoom area


    #axins.set_title('Zoom', fontsize=10)
    
    # Draw box in main plot showing zoom area
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax_top, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    mark_inset(ax_bottom, axins2, loc1=2, loc2=3, fc="none", ec="0.5")


    # --- Labels
    ax_top.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
    ax_bottom.set_ylabel("Cumulative Utility, bn $", fontsize=16)
    ax_bottom.set_xlabel("Cumulative Emissions, MTCO2", fontsize=16)

    policy_titles = {
        "Carbon_price": "Carbon Price",
        "Electricity_subsidy": "Electricity Subsidy",
        "Adoption_subsidy": "New Car Rebate",
        "Adoption_subsidy_used": "Used Car Rebate",
        "Production_subsidy": "Production Subsidy"
    }
    # --- Legend
    legend_elements = [Patch(facecolor=policy_colors[policy], edgecolor='black',
                 label=f"{policy_titles[policy]} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
                       for policy in all_policies]

    # Custom intensity markers
    legend_elements += [Patch(facecolor='black', edgecolor='black', label='BAU')]

    low_proxy = plt.Line2D([0], [0], marker=half_circle_marker(0, 180),
                           color='gray', markerfacecolor='gray', markeredgecolor='black',
                           linestyle='None', label='Low Intensity', markersize=8)

    high_proxy = plt.Line2D([0], [0], marker=half_circle_marker(0, 180),
                            color='gray', markerfacecolor='gray', markeredgecolor='black',
                            linestyle='None', label='High Intensity', markersize=12)

    confidence = plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval')
    legend_elements += [confidence, low_proxy, high_proxy]

    ax_bottom.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # --- Save
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/emissions_tradeoffs/emissions_tradeoff.png", dpi=dpi)


def main(fileNames, fileName_BAU, fileNames_single_policies):
    #EDNOGENOSU SINGLE POLICY
    single_policy_outcomes = load_object(f"{fileNames_single_policies}/Data", "policy_outcomes")

    #PAIRS OF POLICY 
    fileName = fileNames[0]
    base_params = load_object(f"{fileName}/Data", "base_params")
    
    file_name = produce_name_datetime("all_policies")
    createFolder(file_name)

    pairwise_outcomes_complied = {}
    
    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    outcomes_BAU = load_object(f"{fileName_BAU}/Data", "outcomes")

    #pairwise_outcomes_complied = {k: v for k, v in pairwise_outcomes_complied.items() if set(k) == {"Electricity_subsidy", "Adoption_subsidy"}}


    min_ev_uptake = 0.94
    max_ev_uptake = 0.96

    plot_emissions_tradeoffs_from_outcomes(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                            file_name,
                                            min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300)

    save_object(pairwise_outcomes_complied, file_name + "/Data", "pairwise_outcomes")
    save_object(base_params, file_name + "/Data", "base_params")

    plt.show()


if __name__ == "__main__":
    main(
        fileNames=["results/endog_pair_13_13_45__09_04_2025"],
        fileName_BAU="results/BAU_runs_12_22_12__10_04_2025",
        fileNames_single_policies = "results/endog_single_10_43_00__09_04_2025"#"results/endogenous_policy_intensity_18_43_26__06_03_2025"
    )