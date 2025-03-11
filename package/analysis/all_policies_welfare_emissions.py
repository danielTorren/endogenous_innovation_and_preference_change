from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D  # Import this at the top
# Marker helper functions
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

# Intensity-to-marker-size scaling
def scale_marker_size(value, policy,policy_ranges):
    min_val = policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    if max_val - min_val == 0:
        return 100  # Fixed size if no variation
    norm = (value - min_val) / (max_val - min_val)
    #return 150 + norm * 350  # From 100 to 400 (tweakable)
    return norm * 350  # From 100 to 400 (tweakable)


def plot_welfare_vs_emissions(pairwise_outcomes_complied, file_name, min_val, max_val, outcomes_BAU, single_policy_outcomes, dpi=600):
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
            welfare = (entry["mean_utility_cumulative"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"])*1e-9
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
            welfare = (entry["mean_utility_cumulative"] + entry["mean_profit_cumulative"] - entry["mean_net_cost"])*1e-9
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
    bau_welfare = (outcomes_BAU["mean_utility_cumulative"] + outcomes_BAU["mean_profit_cumulative"] - outcomes_BAU["mean_net_cost"])*1e-9
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

def main(fileNames, fileName_BAU, fileNames_single_policies):
    #EDNOGENOSU SINGLE POLICY

    single_policy_outcomes = load_object(f"{fileNames_single_policies}/Data", "policy_outcomes")
    print("single_policy_outcomes", single_policy_outcomes)
    
    #PAIRS OF POLICY 
    fileName = fileNames[0]
    pairwise_outcomes_complied = {}

    if len(fileNames) == 1:
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    #BAU
    outcomes_BAU = load_object(f"{fileName_BAU}/Data", "outcomes")

    plot_welfare_vs_emissions(pairwise_outcomes_complied, fileName, 0.945, 0.955, outcomes_BAU, single_policy_outcomes, dpi=300)


if __name__ == "__main__":
    main(
        fileNames=["results/endogenous_policy_intensity_19_30_46__06_03_2025"],
        fileName_BAU="results/BAU_runs_13_30_12__07_03_2025",
        fileNames_single_policies = "results/endogenous_policy_intensity_18_43_26__06_03_2025"
    )
