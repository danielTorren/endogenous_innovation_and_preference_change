from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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

from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap

def plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, file_name, min_val, max_val, outcomes_BAU, single_policy_outcomes, measure, y_label, dpi=600):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    # Dictionary to store policy points for shading later
    policy_points = {policy: [] for policy in all_policies}
    policy_pair_points = {}
    
    # Process single policy outcomes first
    for policy, entry in single_policy_outcomes.items():
        mean_uptake = entry["mean_EV_uptake"]
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        if mask:
            if measure == "mean_utility_cumulative":
                welfare = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"]) * 1e-9
            else:
                welfare = entry[measure] * 1e-9
            
            emissions = (entry["mean_emissions_cumulative"]) * 1e-9

            policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], entry["optimized_intensity"])
            policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], entry["optimized_intensity"])
            
            # Store point for this single policy
            policy_points[policy].append((emissions, welfare, entry["optimized_intensity"]))

    # Process pairwise policy outcomes
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]
        
        # Store all points for this policy pair
        pair_key = (policy1, policy2)
        policy_pair_points[pair_key] = []
        
        for entry in filtered_data:
            if measure == "mean_utility_cumulative":
                welfare = (entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"]) * 1e-9
            else:
                welfare = entry[measure] * 1e-9
            
            emissions = (entry["mean_emissions_cumulative"]) * 1e-9
            
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])
            
            # Store point for this policy pair
            policy_pair_points[pair_key].append((emissions, welfare, entry["policy1_value"], entry["policy2_value"]))

    # BAU point
    if measure == "mean_utility_cumulative":
        bau_welfare = (outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"]) * 1e-9
    else:
        bau_welfare = outcomes_BAU[measure] * 1e-9
    bau_emissions = (outcomes_BAU["mean_emissions_cumulative"]) * 1e-9
    
    # Create intensity-based gradient shaded regions for policy pairs
    for (policy1, policy2), points_data in policy_pair_points.items():
        if len(points_data) < 3:  # Need at least 3 points for triangulation
            continue
            
        # Extract points and intensity values
        pair_points = []
        pair_intensities = []
        
        # Add policy pair points
        for x, y, intensity1, intensity2 in points_data:
            pair_points.append((x, y))
            
            # Calculate relative intensity - how much of policy1 vs policy2
            # Normalize both intensities to their ranges
            if policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"] > 0:
                norm_intensity1 = (intensity1 - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"])
            else:
                norm_intensity1 = 0.5
                
            if policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"] > 0:
                norm_intensity2 = (intensity2 - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"])
            else:
                norm_intensity2 = 0.5
            
            # Value between 0 and 1 - closer to 0 means more policy1, closer to 1 means more policy2
            if norm_intensity1 + norm_intensity2 > 0:
                intensity_ratio = norm_intensity2 / (norm_intensity1 + norm_intensity2)
            else:
                intensity_ratio = 0.5
                
            pair_intensities.append(intensity_ratio)
        
        # Add single policy points if available
        for i, (policy, points) in enumerate([(policy1, policy_points[policy1]), (policy2, policy_points[policy2])]):
            if points:
                # Add the single policy point
                x, y, intensity = points[0]
                pair_points.append((x, y))
                # Set intensity ratio to favor this policy completely
                pair_intensities.append(1.0 if i == 1 else 0.0)  # 0.0 for policy1, 1.0 for policy2
            else:
                # Find point with minimum intensity for this policy within the pair
                if policy == policy1:
                    min_value_idx = np.argmin([p[2] for p in points_data])  # policy1_value
                    x, y = points_data[min_value_idx][0], points_data[min_value_idx][1]
                    pair_points.append((x, y))
                    pair_intensities.append(0.0)  # Favor policy1
                else:
                    min_value_idx = np.argmin([p[3] for p in points_data])  # policy2_value
                    x, y = points_data[min_value_idx][0], points_data[min_value_idx][1]
                    pair_points.append((x, y))
                    pair_intensities.append(1.0)  # Favor policy2
        
        # Convert to numpy arrays
        pair_points = np.array(pair_points)
        pair_intensities = np.array(pair_intensities)
        
        # Use Delaunay triangulation to create a mesh
        if len(pair_points) >= 3:
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(pair_points)
                
                # Create a custom colormap between the two policy colors
                color1 = policy_colors[policy1]
                color2 = policy_colors[policy2]
                custom_cmap = LinearSegmentedColormap.from_list(f'{policy1}_{policy2}', [color1, color2])
                
                # For each triangle in the mesh, calculate average intensity and color
                for simplex in tri.simplices:
                    vertices = pair_points[simplex]
                    # Get average intensity for this triangle
                    avg_intensity = np.mean(pair_intensities[simplex])
                    # Get color from custom colormap
                    color = custom_cmap(avg_intensity)
                    # Fill triangle with appropriate color
                    triangle = plt.Polygon(vertices, color=color, alpha=0.7, edgecolor=None)
                    ax.add_patch(triangle)
            except Exception as e:
                print(f"Error creating triangulation for {policy1}-{policy2}: {e}")
                # Fallback: simple alpha blending if triangulation fails
                hull_points = np.array(pair_points)
                try:
                    hull = ConvexHull(hull_points)
                    vertices = hull_points[hull.vertices]
                    # Use average intensity for the entire region
                    avg_intensity = np.mean(pair_intensities)
                    blend_color = tuple(c1 * (1-avg_intensity) + c2 * avg_intensity 
                                     for c1, c2 in zip(color1, color2))
                    ax.fill(vertices[:, 0], vertices[:, 1], color=blend_color, alpha=0.5)
                except:
                    print(f"Both triangulation and convex hull failed for {policy1}-{policy2}")
    
    # Plot BAU point
    ax.scatter(bau_emissions, bau_welfare, color='black', marker='o', s=400, 
              edgecolor='black', label="Business as Usual (BAU)")
    
    # Plot single policy points
    for policy, points in policy_points.items():
        for x, y, intensity in points:
            size = scale_marker_size(intensity, policy, policy_ranges)
            ax.scatter(x, y, s=size, marker=full_circle_marker(), 
                      color=policy_colors[policy], edgecolor="black")
    
    # Plot policy pair points
    for (policy1, policy2), points_data in policy_pair_points.items():
        for x, y, intensity1, intensity2 in points_data:
            # Use half circles for the paired policies
            color1 = policy_colors[policy1]
            color2 = policy_colors[policy2]
            size1 = scale_marker_size(intensity1, policy1, policy_ranges)
            size2 = scale_marker_size(intensity2, policy2, policy_ranges)
            
            # Use an average size for the marker
            avg_size = (size1 + size2) / 2
            
            # Plot left half-circle (policy1)
            ax.scatter(x, y, s=avg_size, marker=half_circle_marker(180, 359), 
                      color=color1, edgecolor="black")
            # Plot right half-circle (policy2)
            ax.scatter(x, y, s=avg_size, marker=half_circle_marker(0, 179), 
                      color=color2, edgecolor="black")
    
    # Axis labels and title
    ax.set_xlabel("Cumulative Emissions, MTC02")
    ax.set_ylabel(f"{y_label}, bn $")

    # Policy color legend with intensity ranges
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))

    # Adding intensity indicators to legend using Line2D
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', markersize=5, linestyle='None', label="Low Policy Intensity"))
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', markersize=10, linestyle='None', label="High Policy Intensity"))

    ax.legend(handles=legend_elements, loc="best", title="Policies & Intensity Ranges")

    # Save plot
    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions_{measure}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=dpi)
    plt.savefig(f'{file_name}/Plots/welfare_vs_cumulative_emissions_{measure}_VECTOR.eps', format='eps', dpi=dpi)





def plot_welfare_vs_emissions(base_params, pairwise_outcomes_complied, file_name, min_val, max_val, outcomes_BAU, single_policy_outcomes, dpi=600):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True)

    color_map = plt.get_cmap('Set1', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    plotted_points = []
    plotted_points_single = []
    policy_welfare = {}

    # Collect data and compute intensity ranges
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            welfare = (entry["mean_utility_cumulative"]/base_params["parameters_social_network"]["prob_switch_car"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"])*1e-9
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
    #SINGLE POLICY
    # Collect data and compute intensity ranges

    for policy, entry in single_policy_outcomes.items():
        mean_uptake = entry["mean_EV_uptake"]
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        if mask:
            welfare = (entry["mean_utility_cumulative"]/base_params["parameters_social_network"]["prob_switch_car"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"])*1e-9
            emissions = (entry["mean_emissions_cumulative"])*1e-9

            policy_key = policy
            if policy_key not in policy_welfare or welfare > policy_welfare[policy_key]["welfare"]:
                policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], entry["optimized_intensity"])
                policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], entry["optimized_intensity"])

                plotted_points_single.append((emissions, welfare, policy, entry["optimized_intensity"]))

                policy_welfare[policy_key] = {
                    "welfare": welfare,
                    "policy_value": entry["optimized_intensity"],
                }

    if not plotted_points:
        print("No data available for the given uptake range.")
        return {}


    #SINGLE POLICIES
    for x, y, policy, policy_value in plotted_points_single:
        color1 = policy_colors[policy]
        size1 = scale_marker_size(policy_value, policy, policy_ranges)
        ax.scatter(x, y, s=size1, marker=full_circle_marker(), color=color1, edgecolor="black")

    # BAU point
    bau_welfare = (outcomes_BAU["mean_utility_cumulative"]/base_params["parameters_social_network"]["prob_switch_car"] + outcomes_BAU["mean_profit_cumulative"] - outcomes_BAU["mean_net_cost"])*1e-9
    bau_emissions = (outcomes_BAU["mean_emissions_cumulative"])*1e-9
    ax.scatter(bau_emissions, bau_welfare, color='black', marker='o', s=400, edgecolor='black', label="Business as Usual (BAU)")

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


    # Axis labels and title
    ax.set_xlabel("Cumulative Emissions, MTC02")
    ax.set_ylabel("Welfare, bn $")
    #ax.set_title("Welfare vs Cumulative Emissions (Marker Size = Policy Intensity)")

    # Policy color legend with intensity ranges
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))
    
    # Adding intensity indicators to legend


    # Adding intensity indicators to legend using Line2D
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', markersize=5, linestyle='None', label="Low Policy Intensity"))
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', markersize=10, linestyle='None', label="High Policy Intensity"))

    ax.legend(handles=legend_elements, loc="lower right", title="Policies & Intensity Ranges")

    # Save plot
    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.show()

    # Return top 10 policy combinations by welfare
    #top_10 = dict(sorted(policy_welfare.items(), key=lambda item: item[1]["welfare"], reverse=True)[:10])
    #return top_10


def extract_policy_data(pairwise_outcomes_complied, single_policy_outcomes,  min_ev_uptake,max_ev_uptake ):
    """
    Extracts policy data and prepares it for LaTeX table formatting.
    
    Args:
        pairwise_outcomes_complied (dict): Dictionary containing outcomes for policy pairs.
        single_policy_outcomes (dict): Dictionary containing outcomes for single policies.
        
    Returns:
        list: A list of tuples containing policy1, policy2, and mean sd_ev_uptake.
    """
    policy_data = {}
    policy_list = set()
    
    # Extract data for policy pairs and aggregate SD EV Uptake values
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        key = (policy1, policy2)
        if key not in policy_data:
            policy_data[key] = []
        
        for entry in data:
            if (entry["policy1_value"] > 0 and entry["policy2_value"] > 0) and ( min_ev_uptake  < entry["mean_ev_uptake"] <  max_ev_uptake ):
                policy_data[key].append(entry["sd_ev_uptake"])
                policy_list.update([policy1, policy2])
    
    # Compute the mean SD EV Uptake for each policy combination
    policy_data_mean = [(p1, p2, np.mean(values)) for (p1, p2), values in policy_data.items()]
    
    # Extract data for single policies and add to the diagonal
    for policy, entry in single_policy_outcomes.items():
        policy_data_mean.append((policy, policy, entry["sd_ev_uptake"]))
        policy_list.add(policy)
    
    return policy_data_mean, sorted(policy_list)

def plot_policy_heatmap(file_name, policy_data,min_ev_uptake,max_ev_uptake, dpi):
    """
    Plots a heatmap using numpy arrays where:
    - X-axis represents Policy 2
    - Y-axis represents Policy 1
    - Color intensity represents the mean SD EV Uptake
    
    Args:
        policy_data (list of tuples): List containing (Policy 1, Policy 2, SD EV Uptake).
    """
    
    # Extract unique policies
    policy_data, policies = extract_policy_data(*policy_data, min_ev_uptake,max_ev_uptake)

    num_policies = len(policies)
    
    # Create a lookup dictionary for indices
    policy_index = {policy: i for i, policy in enumerate(policies)}
    
    # Initialize the heatmap matrix with NaNs
    heatmap_matrix = np.full((num_policies, num_policies), np.nan)
    
    # Fill the matrix with mean SD EV Uptake values
    for policy1, policy2, sd_ev_uptake in policy_data:
        print(policy1, policy2, sd_ev_uptake)
        i, j = policy_index[policy1], policy_index[policy2]
        heatmap_matrix[i, j] = sd_ev_uptake

    # Strip underscores from policy names
    policies_cleaned = [p.replace("_", " ") for p in policies]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(heatmap_matrix, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label='STD EV Uptake')
    
    # Format axes with proper tick labels
    ax.set_xticks(range(num_policies))
    ax.set_xticklabels(policies_cleaned, rotation=45, ha='right')
    ax.set_yticks(range(num_policies))
    ax.set_yticklabels(policies_cleaned)
    
    ax.set_xlabel("Policy 2")
    ax.set_ylabel("Policy 1")
    
    plt.tight_layout()
    
    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.show()





def main(fileNames, fileName_BAU, fileNames_single_policies):
    #EDNOGENOSU SINGLE POLICY

    single_policy_outcomes = load_object(f"{fileNames_single_policies}/Data", "policy_outcomes")
    #print("single_policy_outcomes", list(single_policy_outcomes.keys()))

    
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

    #print(policy_table)
    plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_utility_cumulative","Utility", dpi=300)
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_profit_cumulative","Profit",  dpi=300)
    #plot_welfare_component_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes,"mean_net_cost", "Net Cost", dpi=300)

    #policy_data = extract_policy_data(pairwise_outcomes_complied, single_policy_outcomes, min_ev_uptake,max_ev_uptake)
    #plot_policy_heatmap(fileName, policy_data=(pairwise_outcomes_complied, single_policy_outcomes), min_ev_uptake,max_ev_uptake, dpi=300)

    #quit()

    #plot_welfare_vs_emissions(base_params, pairwise_outcomes_complied, fileName,  min_ev_uptake , max_ev_uptake,  outcomes_BAU, single_policy_outcomes, dpi=300)
    plt.show()

if __name__ == "__main__":
    main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],#["results/endog_pair_19_10_07__11_03_2025"],#
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025",
        fileNames_single_policies = "results/endogenous_policy_intensity_18_17_27__06_03_2025"#"results/endogenous_policy_intensity_18_43_26__06_03_2025"
    )
