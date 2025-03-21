from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Patch

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

def plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name, min_val, max_val, outcomes_BAU, single_policy_outcomes, measure, y_label, dpi=600): 
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}
    policy_points = {policy: [] for policy in all_policies}
    policy_pair_points = {}

    # Process single policy outcomes (for range calibration only)
    for policy, entry in single_policy_outcomes.items():
        mean_uptake = entry["mean_EV_uptake"]
        if (mean_uptake >= min_val) and (mean_uptake <= max_val):
            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], entry["optimized_intensity"])
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], entry["optimized_intensity"])
            policy_points[policy].append((entry["mean_emissions_cumulative"] * 1e-9, entry[measure] * 1e-9, entry["optimized_intensity"]))

    # Process pairwise outcomes
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        filtered_data = [entry for i, entry in enumerate(data) if (mean_uptake[i] >= min_val) and (mean_uptake[i] <= max_val)]
        pair_key = (policy1, policy2)
        policy_pair_points[pair_key] = []

        for entry in filtered_data:
            welfare = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9 if measure == "mean_utility_cumulative" else entry[measure] * 1e-9
            emissions = entry["mean_emissions_cumulative"] * 1e-9
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])
            policy_pair_points[pair_key].append((emissions, welfare, entry["policy1_value"], entry["policy2_value"]))

    plotted_points = []
    policy_welfare = {}

    # Collect data and compute intensity ranges
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            if measure == "mean_utility_cumulative":
                welfare = (entry["mean_utility_cumulative"]/base_params["parameters_social_network"]["prob_switch_car"])*1e-9
            else:
                 welfare = entry[measure]*1e-9
            
            emissions = (entry["mean_emissions_cumulative"])*1e-9

            policy_key = (policy1, policy2)
            if policy_key not in policy_welfare or welfare > policy_welfare[policy_key]["welfare"]:
                policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
                policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
                policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
                policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])

                plotted_points.append((emissions, welfare, policy1, policy2, entry["policy1_value"], entry["policy2_value"]))

                policy_welfare[policy_key] = {
                    "welfare": welfare,
                    "policy1_value": entry["policy1_value"],
                    "policy2_value": entry["policy2_value"]
                }

    # Draw smooth, brush-like regions
    for (policy1, policy2), points_data in policy_pair_points.items():

        pair_emissions, pair_welfare, pair_intensities = [], [], []

        for x, y, intensity1, intensity2 in points_data:
            pair_emissions.append(x)
            pair_welfare.append(y)
            norm1 = (intensity1 - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"])
            norm2 = (intensity2 - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"])
            ratio = norm2 / (norm1 + norm2)
            pair_intensities.append(ratio)

        for i, (policy, points) in enumerate([(policy1, policy_points[policy1]), (policy2, policy_points[policy2])]):
            if points:
                x, y, _ = points[0]
                pair_emissions.append(x)
                pair_welfare.append(y)
                pair_intensities.append(1.0 if i == 1 else 0.0)

        # Convert to numpy arrays
        x = np.array(pair_emissions)
        y = np.array(pair_welfare)
        z = np.array(pair_intensities)

        # Normalize x and y for stable interpolation
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x - x.min()
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y - y.min()

        points = np.column_stack((x_norm, y_norm))

        # Create normalized grid
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
        # Rescale grid_x and grid_y to original data space
        grid_x_real = grid_x * (x.max() - x.min()) + x.min()
        grid_y_real = grid_y * (y.max() - y.min()) + y.min()


        # Interpolate intensity values onto grid
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear', fill_value=np.nan)

        # Smooth result
        grid_z = gaussian_filter(grid_z, sigma=4)

        # Create color map
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        custom_cmap = LinearSegmentedColormap.from_list(f'{policy1}_{policy2}', [color1, color2])

        # Plot with original data extent
        ax.imshow(grid_z.T, extent=(grid_x_real.min(), grid_x_real.max(), grid_y_real.min(), grid_y_real.max()),
                origin='lower', cmap=custom_cmap, alpha=0.45,
                interpolation='bilinear', aspect='auto')

        
        # Create mask where interpolated values are valid
        valid_mask = ~np.isnan(grid_z)

        # Create a contour outline at the edge of the valid region
        # Use a dummy mask value (e.g., 0.5) to plot a line around it
        ax.contour(grid_x_real, grid_y_real, valid_mask.astype(float), levels=[0.5],
                colors='black', linewidths=1.5)
        
    # Plot points (half circles sized by intensity)
    for x, y, policy1, policy2, policy1_value, policy2_value in plotted_points:
        if policy1_value > 0 and policy2_value > 0:#BOTH need to be off the single policy
            color1 = policy_colors[policy1]
            color2 = policy_colors[policy2]

            size1 = scale_marker_size(policy1_value, policy1, policy_ranges)
            size2 = scale_marker_size(policy2_value, policy2, policy_ranges)

            if policy1_value == 0:
                ax.scatter(x, y, s=size2, marker=full_circle_marker(), color=color2, edgecolor="black")
            elif policy2_value == 0:
                ax.scatter(x, y, s=size1, marker=full_circle_marker(), color=color1, edgecolor="black")
            else:
                ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
                ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")



    # Plot BAU point
    bau_welfare = outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9 if measure == "mean_utility_cumulative" else outcomes_BAU[measure] * 1e-9
    bau_emissions = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    ax.scatter(bau_emissions, bau_welfare, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")

    plotted_points_single = []
    # Process single policy outcomes
    for policy, entry in single_policy_outcomes.items():
        mean_uptake = entry["mean_EV_uptake"]
        if (mean_uptake >= min_val) and (mean_uptake <= max_val):
            intensity = entry["optimized_intensity"]
            welfare = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9 if measure == "mean_utility_cumulative" else entry[measure] * 1e-9
            emissions = entry["mean_emissions_cumulative"] * 1e-9
            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], intensity)
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], intensity)
            policy_points[policy].append((emissions, welfare, intensity))
            plotted_points_single.append((emissions, welfare, policy, intensity))

    # Plot single policy points
    for x, y, policy, intensity in plotted_points_single:
        min_val = policy_ranges[policy]["min"]
        max_val = policy_ranges[policy]["max"]
        if max_val - min_val == 0:
            size = 100
        else:
            norm = (intensity - min_val) / (max_val - min_val)
            size = max(70, norm * 300)
        ax.scatter(x, y, s=size, marker='o', color=policy_colors[policy], edgecolor="black")

    # Axis labels
    ax.set_xlabel("Cumulative Emissions, MTC02")
    ax.set_ylabel(f"{y_label}, bn $")

    # Policy legend only (no marker size legends needed anymore)
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    ax.legend(handles=legend_elements, loc="best", title="Policy Intensity Range")

    # Save plot
    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions_{measure}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_profit_cumulative","Profit",  dpi=300)
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_net_cost", "Net Cost", dpi=300)
    plt.show()

if __name__ == "__main__":
    main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],#["results/endog_pair_19_10_07__11_03_2025"],#
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025",
        fileNames_single_policies = "results/endogenous_policy_intensity_18_17_27__06_03_2025"#"results/endogenous_policy_intensity_18_43_26__06_03_2025"
    )
