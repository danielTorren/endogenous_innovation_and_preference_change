from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection

# Marker helper functions
def half_circle_marker(start, end, offset=-45):
    angles = np.linspace(np.radians(start + offset), np.radians(end + offset), 100)
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

# Intensity-to-marker-size scaling
def scale_marker_size(value, policy, policy_ranges):
    min_val = policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    if max_val - min_val == 0:
        return 100  # Fixed size if no variation
    norm = (value - min_val) / (max_val - min_val)
    scale = 350
    epsilon = 0.2
    return np.maximum(scale * epsilon, norm * scale)

def plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name,
                                        min_val, max_val, outcomes_BAU,
                                        single_policy_outcomes, measure, y_label, dpi=600):

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}
    policy_points = {policy: [] for policy in all_policies}

    # --- Single policy outcomes
    plotted_points_single = []
    for policy, entry in single_policy_outcomes.items():
        if min_val <= entry["mean_EV_uptake"] <= max_val:
            emissions = entry["mean_emissions_cumulative"] * 1e-9
            welfare = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                       if measure == "mean_utility_cumulative" else entry[measure] * 1e-9)
            intensity = entry["optimized_intensity"]
            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], intensity)
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], intensity)
            policy_points[policy].append((emissions, welfare, intensity))
            plotted_points_single.append((emissions, welfare, policy, intensity))

    # --- Policy pair outcomes
    plotted_points = []
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            emissions = entry["mean_emissions_cumulative"] * 1e-9
            welfare = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                       if measure == "mean_utility_cumulative" else entry[measure] * 1e-9)
            intensity1 = entry["policy1_value"]
            intensity2 = entry["policy2_value"]
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], intensity1)
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], intensity1)
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], intensity2)
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], intensity2)

            plotted_points.append((emissions, welfare, policy1, policy2, intensity1, intensity2))

        # --- Colored Line Rendering
        if len(filtered_data) < 2:
            continue

        color1 = np.array(policy_colors[policy1])
        color2 = np.array(policy_colors[policy2])
        segments = []
        colors = []

        # Prepare and sort points
        emissions_list, welfare_list, blend_ratios = [], [], []
        for entry in filtered_data:
            e = entry["mean_emissions_cumulative"] * 1e-9
            w = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                 if measure == "mean_utility_cumulative" else entry[measure] * 1e-9)
            i1 = entry["policy1_value"]
            i2 = entry["policy2_value"]

            n1 = (i1 - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"]) if policy_ranges[policy1]["max"] > policy_ranges[policy1]["min"] else 0.5
            n2 = (i2 - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"]) if policy_ranges[policy2]["max"] > policy_ranges[policy2]["min"] else 0.5
            r = n2 / (n1 + n2) if (n1 + n2) > 0 else 0.5

            emissions_list.append(e)
            welfare_list.append(w)
            blend_ratios.append(r)

        emissions_arr = np.array(emissions_list)
        welfare_arr = np.array(welfare_list)
        blend_arr = np.array(blend_ratios)
        sort_idx = np.argsort(emissions_arr)

        for i in range(len(sort_idx) - 1):
            idx1, idx2 = sort_idx[i], sort_idx[i + 1]
            p1 = [emissions_arr[idx1], welfare_arr[idx1]]
            p2 = [emissions_arr[idx2], welfare_arr[idx2]]
            blend = (blend_arr[idx1] + blend_arr[idx2]) / 2
            blended_color = blend * color2 + (1 - blend) * color1
            segments.append([p1, p2])
            colors.append(blended_color)

        lc = LineCollection(segments, colors=colors, linewidths=3, zorder=3)
        ax.add_collection(lc)

    # --- BAU point
    bau_welfare = (outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                   if measure == "mean_utility_cumulative" else outcomes_BAU[measure] * 1e-9)
    bau_emissions = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    ax.scatter(bau_emissions, bau_welfare, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")


    # --- Plot half-circle markers
    for x, y, policy1, policy2, val1, val2 in plotted_points:
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        size1 = scale_marker_size(val1, policy1, policy_ranges)
        size2 = scale_marker_size(val2, policy2, policy_ranges)

        if val1 == 0:
            pass
            #ax.scatter(x, y, s=size2, marker=full_circle_marker(), color=color2, edgecolor="black")
        elif val2 == 0:
            pass
           #ax.scatter(x, y, s=size1, marker=full_circle_marker(), color=color1, edgecolor="black")
        else:
            if measure == "mean_utility_cumulative":
                if x < bau_emissions and y > bau_welfare:
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")
            elif measure == "mean_profit_cumulative":
                if x < bau_emissions:
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")
            elif measure == "mean_net_cost":
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")

    # --- Single policy points
    for x, y, policy, intensity in plotted_points_single:
        size = scale_marker_size(intensity, policy, policy_ranges)
        ax.scatter(x, y, s=size, marker='o', color=policy_colors[policy], edgecolor="black")

    # --- Labels, legends, save
    ax.set_xlabel("Cumulative Emissions, MTC02")
    ax.set_ylabel(f"{y_label}, bn $")
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))
    ax.legend(handles=legend_elements, loc="best", title="Policy Intensity Range")

    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions_{measure}.png'
    plt.savefig(save_path, dpi=dpi)
    plt.savefig(f'{file_name}/Plots/welfare_vs_cumulative_emissions_{measure}_VECTOR.eps', format='eps', dpi=dpi)

def main(fileNames, fileName_BAU, fileNames_single_policies):
    #EDNOGENOSU SINGLE POLICY

    single_policy_outcomes = load_object(f"{fileNames_single_policies}/Data", "policy_outcomes")

    #PAIRS OF POLICY 
    fileName = fileNames[0]
    base_params = load_object(f"{fileName}/Data", "base_params")
    
    pairwise_outcomes_complied = {}
    
    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    #BAU
    outcomes_BAU = load_object(f"{fileName_BAU}/Data", "outcomes")

    min_ev_uptake = 0.945
    max_ev_uptake = 0.965

    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_utility_cumulative","Utility", dpi=300)
    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_profit_cumulative","Profit",  dpi=300)
    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_net_cost", "Net Cost", dpi=300)
    
    plt.show()

if __name__ == "__main__":
    main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],#["results/endog_pair_19_10_07__11_03_2025"],#
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025",
        fileNames_single_policies = "results/endogenous_policy_intensity_18_17_27__06_03_2025"#"results/endogenous_policy_intensity_18_43_26__06_03_2025"
    )
