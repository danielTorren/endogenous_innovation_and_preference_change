from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch

def plot_policy_outcomes(pairwise_outcomes_complied, file_name, min_val, max_val, x_measure, y_measure, outcomes_BAU, dpi=600):
    fig, ax = plt.subplots(figsize=(15, 8))

    color_map = plt.get_cmap('Set3', 10)
    all_policies = set()
    for (policy1, policy2) in pairwise_outcomes_complied.keys():
        all_policies.update([policy1, policy2])

    policy_colors = {policy: color_map(i) for i, policy in enumerate(sorted(all_policies))}
    policy_ranges = {policy: {"min": float('inf'), "max": float('-inf')} for policy in all_policies}

    x_values = []
    y_values = []
    plotted_points = []

    for (policy1, policy2), data in pairwise_outcomes_complied.items():
        mean_uptake = np.array([entry["mean_ev_uptake"] for entry in data])
        mask = (mean_uptake >= min_val) & (mean_uptake <= max_val)
        filtered_data = [entry for i, entry in enumerate(data) if mask[i]]

        for entry in filtered_data:
            x = entry[x_measure]
            y = entry[y_measure]

            x_values.append(x)
            y_values.append(y)

            policy_ranges[policy1]["min"] = min(policy_ranges[policy1]["min"], entry["policy1_value"])
            policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], entry["policy1_value"])
            policy_ranges[policy2]["min"] = min(policy_ranges[policy2]["min"], entry["policy2_value"])
            policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], entry["policy2_value"])

            plotted_points.append((x, y, policy1, policy2, entry["policy1_value"], entry["policy2_value"]))

    # Include BAU point into axis limit calculation
    bau_x = outcomes_BAU[x_measure]
    bau_y = outcomes_BAU[y_measure]

    if x_values and y_values:
        x_min = min(min(x_values), bau_x)
        x_max = max(max(x_values), bau_x)
        y_min = min(min(y_values), bau_y)
        y_max = max(max(y_values), bau_y)

        ax.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
        ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

    # Define custom half-circle markers
    def half_circle_marker(start, end):
        angles = np.linspace(np.radians(start), np.radians(end), 100)
        verts = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.vstack([verts, [0, 0]])
        return Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY])

    # Plot split-ring markers and annotations
    for x, y, p1, p2, v1, v2 in plotted_points:
        color1 = policy_colors[p1]
        color2 = policy_colors[p2]

        norm1 = (v1 - policy_ranges[p1]["min"]) / (policy_ranges[p1]["max"] - policy_ranges[p1]["min"])
        norm2 = (v2 - policy_ranges[p2]["min"]) / (policy_ranges[p2]["max"] - policy_ranges[p2]["min"])

        if v1 == 0:
            ax.scatter(x, y, s=100, color=color2, edgecolor="black", marker='o')
        elif v2 == 0:
            ax.scatter(x, y, s=100, color=color1, edgecolor="black", marker='o')
        else:
            ax.scatter(x, y, s=100, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
            ax.scatter(x, y, s=100, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")

        offset = (y_max - y_min) * 0.016
        ax.text(x, y - offset, f"{norm1:.3f}", fontsize=8, ha='center', va='top', color='black')
        ax.text(x, y + offset, f"{norm2:.3f}", fontsize=8, ha='center', va='bottom', color='black')

    # Plot BAU as either line (if net cost involved) or marker
    #if "net_cost" in x_measure:
    #    ax.axvline(bau_x, color='black', linestyle='--', linewidth=1.5, label="Business as Usual (BAU)")
    #elif "net_cost" in y_measure:
    #    ax.axhline(bau_y, color='black', linestyle='--', linewidth=1.5, label="Business as Usual (BAU)")
    #else:
    ax.scatter(bau_x, bau_y, color='black', edgecolor='black', s=100, label="Business as Usual (BAU)")

    # Axis labels and grid
    ax.set_xlabel(x_measure.replace("_", " ").capitalize())
    ax.set_ylabel(y_measure.replace("_", " ").capitalize())
    ax.grid(True)

    # Build policy legend
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy}\n({policy_ranges[policy]['min']:.3f}-{policy_ranges[policy]['max']:.3f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))

    ax.legend(handles=legend_elements, loc='best', title="Policies and Intensity Ranges")

    plt.tight_layout()

    os.makedirs(f'{file_name}/Plots', exist_ok=True)
    save_path = f'{file_name}/Plots/{x_measure}_vs_{y_measure}.png'
    plt.savefig(save_path, dpi=dpi)
    #plt.close()

    #print(f"Saved plot to {save_path}")

def plot_all_measure_combinations(pairwise_outcomes_complied, file_name, min_val, max_val, outcomes_BAU, dpi=600):
    first_entry = next(iter(pairwise_outcomes_complied.values()))[0]
    exclude = {"policy1_value", "policy2_value", "policy1_name", "policy2_name", "mean_ev_uptake", "sd_ev_uptake", "mean_total_cost"}
    measures = [m for m in first_entry.keys() if m not in exclude]

    for i, x_measure in enumerate(measures):
        for j, y_measure in enumerate(measures):
            if i < j:
                plot_policy_outcomes(pairwise_outcomes_complied, file_name, min_val, max_val, x_measure, y_measure, outcomes_BAU, dpi)

def plot_welfare_vs_emissions(pairwise_outcomes_complied, file_name, min_val, max_val, outcomes_BAU, dpi=600):
    """
    Plots welfare vs cumulative emissions with split ring markers, annotations, and policy legend.
    - If BAU is provided, it is shown as either a black marker (normal case) or dashed line if net cost is involved.
    - Welfare is: utility + profit - net policy cost

    Returns:
        dict: Top 10 policy combinations by welfare, including welfare and policy intensities.
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
    policy_welfare = {}

    # Gather and preprocess data
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

            # Track the **best welfare entry** for each policy pair
            policy_key = (policy1, policy2)
            if policy_key not in policy_welfare or welfare > policy_welfare[policy_key]["welfare"]:
                policy_welfare[policy_key] = {
                    "welfare": welfare,
                    "policy1_value": entry["policy1_value"],
                    "policy2_value": entry["policy2_value"]
                }

    if not x_values or not y_values:
        print("No data available for the given uptake range.")
        return {}

    # Half-circle marker generator
    def half_circle_marker(start, end):
        angles = np.linspace(np.radians(start), np.radians(end), 100)
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

    # Plot split-ring markers with annotations
    for x, y, policy1, policy2, policy1_value, policy2_value in plotted_points:
        color1, color2 = policy_colors[policy1], policy_colors[policy2]

        norm1 = (policy1_value - policy_ranges[policy1]["min"]) / (policy_ranges[policy1]["max"] - policy_ranges[policy1]["min"])
        norm2 = (policy2_value - policy_ranges[policy2]["min"]) / (policy_ranges[policy2]["max"] - policy_ranges[policy2]["min"])

        if policy1_value == 0:
            ax.scatter(x, y, s=100, marker=full_circle_marker(), color=color2, edgecolor="black")
        elif policy2_value == 0:
            ax.scatter(x, y, s=100, marker=full_circle_marker(), color=color1, edgecolor="black")
        else:
            ax.scatter(x, y, s=100, marker=half_circle_marker(0, 180), color=color1, edgecolor="black")
            ax.scatter(x, y, s=100, marker=half_circle_marker(180, 360), color=color2, edgecolor="black")

        text_offset_y = (max(y_values) - min(y_values)) * 0.016
        ax.text(x, y - text_offset_y, f"{norm1:.3f}", fontsize=8, ha='center', va='top')
        ax.text(x, y + text_offset_y, f"{norm2:.3f}", fontsize=8, ha='center', va='bottom')

    # Add BAU point
    bau_welfare = outcomes_BAU["mean_utility_cumulative"] + outcomes_BAU["mean_profit_cumulative"] - outcomes_BAU["mean_net_cost"]
    bau_emissions = outcomes_BAU["mean_emissions_cumulative"]

    ax.scatter(bau_emissions, bau_welfare, color='black', marker='o', s=100, edgecolor='black', label="Business as Usual (BAU)")

    ax.set_xlabel("Cumulative Emissions")
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare vs Cumulative Emissions")

    ax.grid(True)

    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
              label=f"{policy}\n({policy_ranges[policy]['min']:.3f}-{policy_ranges[policy]['max']:.3f})")
        for policy in sorted(all_policies)
    ]
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))

    ax.legend(handles=legend_elements, loc='best', title="Policies and Intensity Ranges")

    plt.tight_layout()

    save_path = f'{file_name}/Plots/welfare_vs_cumulative_emissions.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    #plt.close()

    # Extract and return top 10 policy combinations by welfare (including intensities)
    top_10 = dict(sorted(policy_welfare.items(), key=lambda item: item[1]["welfare"], reverse=True)[:10])

    return top_10

def main(fileNames, fileName_BAU):
    fileName = fileNames[0]
    pairwise_outcomes_complied = {}

    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    outcomes_BAU = load_object(f"{fileName_BAU}/Data", "outcomes")
    top_10 = plot_welfare_vs_emissions(pairwise_outcomes_complied, fileName, 0.94, 0.96, outcomes_BAU, dpi=300)
    plt.show()
    save_object(top_10, f"{fileName}/Data", "top_10")
    
    plot_all_measure_combinations(pairwise_outcomes_complied, fileName, 0.94, 0.96, outcomes_BAU, dpi=300)

if __name__ == "__main__":
    main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025"
    )
