from package.resources.utility import load_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch

def plot_policy_outcomes(pairwise_outcomes_complied, file_name, min_val, max_val, x_measure, y_measure, dpi=600):
    fig, ax = plt.subplots(figsize=(15, 8))

    color_map = plt.get_cmap('Set3', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}

    # Extract all x and y values to determine the data range
    x_values = []
    y_values = []
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}  # Store min/max for each policy

    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]
        for entry in filtered_data:
            x_values.append(entry[x_measure])
            y_values.append(entry[y_measure])

            # Update policy value ranges
            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])

    # Calculate tight axes limits based on scatter plot positions
    if x_values and y_values:  # Ensure there is data
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # Add padding to the axes limits
        padding_x = (x_max - x_min) * 0.05  # 10% padding
        padding_y = (y_max - y_min) * 0.05  # 10% padding

        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
    else:
        # Fallback if no data is available
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Define custom half-circle markers
    def half_circle_marker(angle_start, angle_end):
        radius = 1.0
        angles = np.linspace(np.radians(angle_start), np.radians(angle_end), 100)
        verts = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.vstack([verts, [(0, 0)]])  # Close the shape
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
        return Path(verts, codes)

    # Plot the data using scatter with custom markers
    marker_size = 100  # Size of the markers in points^2
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            x = entry[x_measure]
            y = entry[y_measure]

            color1 = policy_colors[policy1]
            color2 = policy_colors[policy2]

            # Plot the first half-circle (policy1)
            ax.scatter(x, y, s=marker_size, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", linewidth=0.5)
            # Plot the second half-circle (policy2)
            ax.scatter(x, y, s=marker_size, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", linewidth=0.5)

            # Normalize policy intensity values
            norm_policy1 = (entry["policy1_value"] - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"])
            norm_policy2 = (entry["policy2_value"] - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"])

            # Calculate text position relative to the marker size and data range
            text_offset_y = (y_max - y_min) * 0.016  # Adjust based on data range

            # Add annotations for normalized policy intensity
            ax.text(x, y - text_offset_y, f"{norm_policy1:.3f}", fontsize=8, color='black', ha='center', va='top')  # Below the marker
            ax.text(x, y + text_offset_y, f"{norm_policy2:.3f}", fontsize=8, color='black', ha='center', va='bottom')  # Above the marker

    ax.set_xlabel(x_measure.replace("_", " ").capitalize())
    ax.set_ylabel(y_measure.replace("_", " ").capitalize())
    ax.grid(True)

    # Add policy value ranges to the legend
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black', label=f"{policy}\n({policy_ranges[policy]['min']:.3f}-{policy_ranges[policy]['max']:.3f})")
        for policy in all_policies
    ]
    ax.legend(handles=legend_elements, loc='best', title="Policies and Ranges: (Relative Intensities)")

    plt.tight_layout()
    #plt.show()

    save_path = f'{file_name}/Plots/{x_measure}_vs_{y_measure}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()
# -----------------------------------------------------------

def plot_all_measure_combinations(pairwise_outcomes_complied, file_name, min_val, max_val, dpi=600):
    first_entry = next(iter(pairwise_outcomes_complied.values()))[0]
    all_measures = list(first_entry.keys())

    # Remove measures that are not outcomes (like policy values)
    exclude_measures = {
        "policy1_value", "policy2_value", "policy1_name", "policy2_name", "mean_ev_uptake", "sd_ev_uptake", "mean_total_cost",
        }
    outcome_measures = [m for m in all_measures if m not in exclude_measures]
    print("outcome_measures", outcome_measures)
    for i, x_measure in enumerate(outcome_measures):
        for j, y_measure in enumerate(outcome_measures):
            if i < j:
                plot_policy_outcomes(pairwise_outcomes_complied, file_name, min_val, max_val,
                                     x_measure, y_measure, dpi)

def plot_welfare_vs_emissions(pairwise_outcomes_complied, file_name, min_val, max_val, dpi=600):
    """
    Plots welfare vs cumulative emissions from the pairwise outcomes data with split ring markers,
    annotations for policy intensity, and a policy legend. 
    If one policy intensity is zero, the marker is fully colored for the active policy.
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    color_map = plt.get_cmap('Set3', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    x_values = []
    y_values = []
    plotted_points = []

    # Collect all data points and compute min/max for normalization
    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])

        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            welfare = entry["mean_utility_cumulative"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"]
            emissions = entry["mean_emissions_cumulative"]

            x_values.append(emissions)
            y_values.append(welfare)

            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])

            plotted_points.append((emissions, welfare, policy1, policy2, entry["policy1_value"], entry["policy2_value"]))

    if not x_values or not y_values:
        print("No data available for the given uptake range.")
        return

    def half_circle_marker(angle_start, angle_end):
        angles = np.linspace(np.radians(angle_start), np.radians(angle_end), 100)
        verts = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.vstack([verts, [(0, 0)]])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
        return Path(verts, codes)

    def full_circle_marker():
        angles = np.linspace(0, 2 * np.pi, 100)
        verts = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.vstack([verts, [verts[0]]])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        return Path(verts, codes)

    marker_size = 100
    for (x, y, policy1, policy2, policy1_value, policy2_value) in plotted_points:
        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]

        # Normalize policy intensities
        norm_policy1 = (policy1_value - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"])
        norm_policy2 = (policy2_value - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"])

        # Plot marker
        if policy1_value == 0:
            # Full color for policy2 if policy1 is zero
            ax.scatter(x, y, s=marker_size, marker=full_circle_marker(), color=color2, edgecolor="black", linewidth=0.5)
        elif policy2_value == 0:
            # Full color for policy1 if policy2 is zero
            ax.scatter(x, y, s=marker_size, marker=full_circle_marker(), color=color1, edgecolor="black", linewidth=0.5)
        else:
            # Split ring if both policies have non-zero intensity
            ax.scatter(x, y, s=marker_size, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", linewidth=0.5)
            ax.scatter(x, y, s=marker_size, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", linewidth=0.5)

        # Annotations for normalized intensities (always shown)
        text_offset_y = (max(y_values) - min(y_values)) * 0.016
        ax.text(x, y - text_offset_y, f"{norm_policy1:.3f}", fontsize=8, color='black', ha='center', va='top')
        ax.text(x, y + text_offset_y, f"{norm_policy2:.3f}", fontsize=8, color='black', ha='center', va='bottom')

    ax.set_xlabel("Cumulative Emissions")
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare vs Cumulative Emissions")

    ax.grid(True)

    # Legend
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy}\n({policy_ranges[policy]['min']:.3f}-{policy_ranges[policy]['max']:.3f})")
        for policy in sorted(all_policies)
    ]
    ax.legend(handles=legend_elements, loc='best', title="Policies and Intensity Ranges")

    plt.tight_layout()

    # Save plot
    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)

    print(f"Saved welfare plot to {save_path}")


def main(
        fileNames,
        fileName_BAU = "results/BAU"
        ):
    
    outcomes_BAU = load_object(fileName_BAU + "/Data", "outcomes")

    # Load observed data
    pairwise_outcomes_complied = {}

    fileName = fileNames[0]

    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(fileName + "/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(fileName + "/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)
            #save_object(pairwise_outcomes_complied, fileName + "/Data", "pairwise_outcomes_complied")

    plot_welfare_vs_emissions(pairwise_outcomes_complied, fileName, 0.945, 0.955, dpi=600)
    plt.show()
    plot_all_measure_combinations(pairwise_outcomes_complied, fileName, 0.945, 0.955, dpi=600)

    plt.show()

if __name__ == "__main__":
    main(
        fileNames=[
            "results/endogenous_policy_intensity_19_30_46__06_03_2025"
        ], 
        fileName_BAU = "results/BAU"
    )