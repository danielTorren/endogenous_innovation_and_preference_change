import copy
from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
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
    epsilon = 0
    return np.maximum(scale * epsilon, norm * scale)



def plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name,
                                        min_val, max_val, outcomes_BAU,
                                        single_policy_outcomes, measure, y_label, dpi=300):

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
        if policy in all_policies:
            if min_val <= entry["mean_EV_uptake"] <= max_val:
                emissions = entry["mean_emissions_cumulative"] * 1e-9
                welfare = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                        if measure == "mean_utility_cumulative" else entry[measure] * 1e-9)
                intensity = entry["optimized_intensity"]
                policy_ranges[policy]["min"] = 0
                policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], intensity)
                policy_points[policy].append((emissions, welfare, intensity))
                plotted_points_single.append((emissions, welfare, policy, intensity))

    # --- Policy pair outcomes

    # Collect all pairwise points for global frontier filtering
    all_emissions, all_welfare, all_blends, all_colors1, all_colors2 = [], [], [], [], []


    plotted_points = []
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        print("pairwise_outcomes_complied",(policy1, policy2) )
                # Filter again for uptake bounds
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            emissions = entry["mean_emissions_cumulative"] * 1e-9
            welfare = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                       if measure == "mean_utility_cumulative" else entry[measure] * 1e-9)
            intensity1 = entry["policy1_value"]
            intensity2 = entry["policy2_value"]
            policy_ranges[policy1]["min"] = 0
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], intensity1)
            policy_ranges[policy2]["min"] = 0
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], intensity2)

            plotted_points.append((emissions, welfare, policy1, policy2, intensity1, intensity2))


        policy_ranges[policy1]["min"] = 0#FORCE MIN TO 0
        policy_ranges[policy2]["min"] = 0

        for entry in filtered_data:
            e = entry["mean_emissions_cumulative"] * 1e-9
            w = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                 if measure == "mean_utility_cumulative" else entry[measure] * 1e-9)
            p1 = entry["policy1_value"]
            p2 = entry["policy2_value"]

            color1 = np.array(policy_colors[policy1])
            color2 = np.array(policy_colors[policy2])

            n1 = (p1 - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"]) if policy_ranges[policy1]["max"] > policy_ranges[policy1]["min"] else 0.5
            n2 = (p2 - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"]) if policy_ranges[policy2]["max"] > policy_ranges[policy2]["min"] else 0.5
            r = n2 / (n1 + n2) if (n1 + n2) > 0 else 0.5

            all_emissions.append(e)
            all_welfare.append(w)
            all_blends.append(r)
            all_colors1.append(color1)
            all_colors2.append(color2)

    all_emissions = np.array(all_emissions)
    all_welfare = np.array(all_welfare)
    all_blends = np.array(all_blends)
    all_colors1 = np.array(all_colors1)
    all_colors2 = np.array(all_colors2)

    """
    # --- Pareto frontier detection
    frontier_idx = []
    if measure in ["mean_utility_cumulative", "mean_profit_cumulative"]:
        sorted_idx = np.argsort(all_emissions)
        max_y = -np.inf
        for i in sorted_idx:
            if all_welfare[i] > max_y:
                frontier_idx.append(i)
                max_y = all_welfare[i]
    elif measure == "mean_net_cost":
        sorted_idx = np.argsort(all_emissions)
        min_y = np.inf
        for i in sorted_idx:
            if all_welfare[i] < min_y:
                frontier_idx.append(i)
                min_y = all_welfare[i]

    if len(frontier_idx) >= 2:
        frontier_idx = np.array(frontier_idx)
        e_f = all_emissions[frontier_idx]
        w_f = all_welfare[frontier_idx]
        b_f = all_blends[frontier_idx]
        c1_f = all_colors1[frontier_idx]
        c2_f = all_colors2[frontier_idx]

        # Build gradient segments along frontier
        sort_idx = np.argsort(e_f)
        segments = []
        colors = []

        num_subdiv = 20  # controls smoothness of gradient

        for i in range(len(sort_idx) - 1):
            idx1, idx2 = sort_idx[i], sort_idx[i + 1]
            x0, y0 = e_f[idx1], w_f[idx1]
            x1, y1 = e_f[idx2], w_f[idx2]

            c1 = b_f[idx1] * c2_f[idx1] + (1 - b_f[idx1]) * c1_f[idx1]
            c2 = b_f[idx2] * c2_f[idx2] + (1 - b_f[idx2]) * c1_f[idx2]

            for j in range(num_subdiv):
                t0 = j / num_subdiv
                t1 = (j + 1) / num_subdiv

                # Line interpolation
                p_start = [x0 * (1 - t0) + x1 * t0, y0 * (1 - t0) + y1 * t0]
                p_end = [x0 * (1 - t1) + x1 * t1, y0 * (1 - t1) + y1 * t1]

                # Color interpolation
                color = c1 * (1 - t0) + c2 * t0

                segments.append([p_start, p_end])
                colors.append(color)

        lc = LineCollection(segments, colors=colors, linewidths=3, zorder=3)
        ax.add_collection(lc)
    """

    # --- BAU point
    bau_welfare = (outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                   if measure == "mean_utility_cumulative" else outcomes_BAU[measure] * 1e-9)
    bau_emissions = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    ax.scatter(bau_emissions, bau_welfare, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")

    size_single = 350
    # --- Single policy points
    for x, y, policy, intensity in plotted_points_single:
        
        ax.scatter(x, y, s=size_single, marker='o', color=policy_colors[policy], edgecolor="black")


    # --- Plot half-circle markers
    for x, y, policy1, policy2, val1, val2 in plotted_points:
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        size1 = scale_marker_size(val1, policy1, policy_ranges)
        size2 = scale_marker_size(val2, policy2, policy_ranges)


        # Add dotted black outline with low alpha
        ax.scatter(x, y, s=size_single, marker=full_circle_marker(), facecolor='none',
                    edgecolor='black', linewidth=1.0, alpha=1, linestyle='dashed')


        if val1 == 0:
            pass
            #ax.scatter(x, y, s=size2, marker=full_circle_marker(), color=color2, edgecolor="black")
        elif val2 == 0:
            pass
           #ax.scatter(x, y, s=size1, marker=full_circle_marker(), color=color1, edgecolor="black")
        else:
            if measure == "mean_utility_cumulative":
                #if x < bau_emissions and y > bau_welfare:
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")
            elif measure == "mean_profit_cumulative":
                #if x < bau_emissions:
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")
            elif measure == "mean_net_cost":
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")


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


def plot_ev_uptake(pairwise_outcomes_complied, file_name,
                   outcomes_BAU, single_policy_outcomes, dpi=600):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}
    policy_points = {policy: [] for policy in all_policies}

    plotted_points_single = []
    for policy, entry in single_policy_outcomes.items():
        if policy in all_policies:
            welfare = entry["mean_EV_uptake"]
            intensity = entry["optimized_intensity"]
            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], intensity)
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], intensity)
            policy_points[policy].append((welfare, intensity))
            plotted_points_single.append((policy, welfare, intensity))

    plotted_points = []
    for (policy1, policy2), data in sorted(pairwise_outcomes_complied.items(), key=lambda x: tuple(sorted(x[0]))):
        for entry in data:
            welfare = entry["mean_ev_uptake"]
            intensity1 = entry["policy1_value"]
            intensity2 = entry["policy2_value"]
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], intensity1)
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], intensity1)
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], intensity2)
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], intensity2)
            plotted_points.append((welfare, policy1, policy2, intensity1, intensity2))

    # --- Plot everything with index as X axis
    x_index = 0

    # BAU
    bau_welfare = outcomes_BAU["mean_ev_uptake"]
    ax.scatter(x_index, bau_welfare, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")
    x_index += 1

    # Single policies
    for policy, welfare, intensity in sorted(plotted_points_single, key=lambda x: x[0]):
        size = scale_marker_size(intensity, policy, policy_ranges)
        ax.scatter(x_index, welfare, s=size, marker='o', color=policy_colors[policy], edgecolor="black")
        x_index += 1

    # Policy pairs
    for welfare, policy1, policy2, val1, val2 in plotted_points:
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        size1 = scale_marker_size(val1, policy1, policy_ranges)
        size2 = scale_marker_size(val2, policy2, policy_ranges)

        if val1 == 0 or val2 == 0:
            pass  # keep your marker logic untouched
        else:
            ax.scatter(x_index, welfare, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
            ax.scatter(x_index, welfare, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")
        x_index += 1

    # Axes and legend
    ax.set_xlabel("Policy index (ordered)")
    ax.set_ylabel("EV uptake")

    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))
    ax.legend(handles=legend_elements, loc="best", title="Policy Intensity Range")

    save_path = f'{file_name}/Plots/ev_uptake.png'
    plt.savefig(save_path, dpi=dpi)


def plot_ev_uptake_std(pairwise_outcomes_complied, file_name,
                       outcomes_BAU, single_policy_outcomes, dpi=600):


    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    # --- Collect single policy points
    plotted_points_single = []
    for policy, entry in single_policy_outcomes.items():
        if policy in all_policies:
            std = entry["sd_ev_uptake"]
            intensity = entry["optimized_intensity"]
            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], intensity)
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], intensity)
            plotted_points_single.append((policy, std, intensity))

    # --- Collect policy pair points
    plotted_points = []
    for (policy1, policy2), data in sorted(pairwise_outcomes_complied.items(), key=lambda x: tuple(sorted(x[0]))):
        for entry in data:
            std = entry["sd_ev_uptake"]
            intensity1 = entry["policy1_value"]
            intensity2 = entry["policy2_value"]
            policy_ranges[policy1]["min"] = 0
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], intensity1)
            policy_ranges[policy2]["min"] = 0
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], intensity2)
            plotted_points.append((std, policy1, policy2, intensity1, intensity2))

    # --- Plot using indexed x-axis
    x_index = 0

    # BAU
    bau_std = outcomes_BAU["sd_ev_uptake"]
    ax.scatter(x_index, bau_std, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")
    x_index += 1

    # Single policies
    for policy, std, intensity in sorted(plotted_points_single, key=lambda x: x[0]):
        size = scale_marker_size(intensity, policy, policy_ranges)
        ax.scatter(x_index, std, s=size, marker='o', color=policy_colors[policy], edgecolor="black")
        x_index += 1

    # Policy pairs
    for std, policy1, policy2, val1, val2 in plotted_points:
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        size1 = scale_marker_size(val1, policy1, policy_ranges)
        size2 = scale_marker_size(val2, policy2, policy_ranges)

        if val1 == 0 or val2 == 0:
            pass  # same as before
        else:
            ax.scatter(x_index, std, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
            ax.scatter(x_index, std, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")
        x_index += 1

    # Axes and legend
    ax.set_xlabel("Policy index (ordered)")
    ax.set_ylabel("Standard Deviation of EV Uptake")

    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))
    ax.legend(handles=legend_elements, loc="best", title="Policy Intensity Range")

    save_path = f'{file_name}/Plots/ev_uptake_std.png'
    plt.savefig(save_path, dpi=dpi)


def plot_ev_uptake_vs_std(pairwise_outcomes_complied, file_name,
                          outcomes_BAU, single_policy_outcomes, dpi=600):

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    plotted_points_single = []
    for policy, entry in single_policy_outcomes.items():
        if policy in all_policies:
            mean_val = entry["mean_EV_uptake"]
            std_val = entry["sd_ev_uptake"]
            intensity = entry["optimized_intensity"]
            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], intensity)
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], intensity)
            plotted_points_single.append((policy, std_val, mean_val, intensity))

    plotted_points = []
    for (policy1, policy2), data in sorted(pairwise_outcomes_complied.items(), key=lambda x: tuple(sorted(x[0]))):
        for entry in data:
            mean_val = entry["mean_ev_uptake"]
            std_val = entry["sd_ev_uptake"]
            intensity1 = entry["policy1_value"]
            intensity2 = entry["policy2_value"]
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], intensity1)
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], intensity1)
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], intensity2)
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], intensity2)
            plotted_points.append((std_val, mean_val, policy1, policy2, intensity1, intensity2))

    # --- BAU
    bau_mean = outcomes_BAU["mean_ev_uptake"]
    bau_std = outcomes_BAU["sd_ev_uptake"]
    ax.scatter(bau_std, bau_mean, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")

    # --- Single policies
    for policy, std_val, mean_val, intensity in sorted(plotted_points_single, key=lambda x: x[0]):
        size = scale_marker_size(intensity, policy, policy_ranges)
        ax.scatter(std_val, mean_val, s=size, marker='o', color=policy_colors[policy], edgecolor="black")

    # --- Policy pairs with half-circle markers
    for std_val, mean_val, policy1, policy2, val1, val2 in plotted_points:
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        size1 = scale_marker_size(val1, policy1, policy_ranges)
        size2 = scale_marker_size(val2, policy2, policy_ranges)

        if val1 == 0 or val2 == 0:
            pass
        else:
            ax.scatter(std_val, mean_val, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
            ax.scatter(std_val, mean_val, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")

    # --- Labels and legend
    ax.set_xlabel("Standard Deviation of EV Uptake")
    ax.set_ylabel("Mean EV Uptake")

    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))
    ax.legend(handles=legend_elements, loc="best", title="Policy Intensity Range")

    save_path = f'{file_name}/Plots/ev_uptake_vs_std.png'
    plt.savefig(save_path, dpi=dpi)


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
    """
    min_ev_uptake = 0
    max_ev_uptake = 1

    plot_ev_uptake(pairwise_outcomes_complied, file_name,
                                         outcomes_BAU,
                                        single_policy_outcomes, dpi=300)
    plot_ev_uptake_std(pairwise_outcomes_complied, file_name,
                                            outcomes_BAU,
                                            single_policy_outcomes, dpi=300)
    
    plot_ev_uptake_vs_std(pairwise_outcomes_complied, file_name,
                          outcomes_BAU, single_policy_outcomes, dpi=600)
    
    """
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_utility_cumulative","Utility", dpi=300)
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_profit_cumulative","Profit",  dpi=300)
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_net_cost", "Net Cost", dpi=300)
    #plt.show()

    min_ev_uptake = 0.945
    max_ev_uptake = 0.955

    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_utility_cumulative","Utility", dpi=300)
    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_profit_cumulative","Profit",  dpi=300)
    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_net_cost", "Net Cost", dpi=300)
    

    save_object(pairwise_outcomes_complied, file_name + "/Data", "pairwise_outcomes")
    save_object(base_params, file_name + "/Data", "base_params")

    plt.show()

if __name__ == "__main__":
    main(
        fileNames=["results/endog_pair_10_19_13__01_04_2025"],
        fileName_BAU="results/BAU_runs_13_06_24__01_04_2025",
        fileNames_single_policies = "results/endog_single_17_38_02__31_03_2025"#"results/endogenous_policy_intensity_18_43_26__06_03_2025"
    )
