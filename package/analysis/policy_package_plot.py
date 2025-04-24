import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path
from package.resources.utility import load_object

# Peace-sign style 1/3-circle marker with better segment definition
def third_circle_marker(start_angle, offset=0):
    # Create a third of a circle (120 degrees) with proper path definition
    angles = np.linspace(np.radians(start_angle + offset), np.radians(start_angle + 120 + offset), 30)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Connect back to center
    verts = np.vstack([[0, 0], verts, [0, 0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

def full_circle_marker():
    angles = np.linspace(0, 2 * np.pi, 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

# Scale marker size with improved normalization
def scale_marker_size(value, policy, policy_ranges, scale_marker, min_size_ratio=0.2):
    min_val = policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    
    if max_val - min_val == 0:
        return scale_marker
    
    # Normalize value between 0 and 1
    norm = (value - min_val) / (max_val - min_val)
    
    # Apply minimum size to ensure visibility (at least min_size_ratio of max size)
    return scale_marker * (min_size_ratio + (1 - min_size_ratio) * norm)

def plot_emissions_tradeoffs_from_triples(
        base_params,
        policy_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.94,
        max_ev_uptake=0.96,
        dpi=300
    ):
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 14), sharex=True)

    # Define policy bounds - these could be calculated from data

    policy_names = ["Carbon_price", "Electricity_subsidy", "Production_subsidy"]
    # Explicit policy-to-color mapping using Set1 colormap indices
    set1 = plt.get_cmap("Set1")
    policy_colors = {
        "Carbon_price": set1(2),           # Red-ish
        "Electricity_subsidy": set1(3),    # Blue-ish
        "Production_subsidy": set1(4),     # Green-ish
    }

    
    policy_ranges = {
        "Carbon_price": {"min": 0, "max": 0.894},
        "Electricity_subsidy": {"min": 0, "max": 1},
        "Production_subsidy": {"min": 0, "max": 36700}
    }
        
    # Base marker size - adjust for visual clarity
    base_marker_size = 350
    
    # Second pass to plot data points
    for (elec_val, prod_val), entry in policy_outcomes.items():
        ev = entry["mean_EV_uptake"]
        if not (min_ev_uptake <= ev <= max_ev_uptake):
            continue
            
        carbon_val = entry["optimized_intensity"][0]
        
        # Calculate emissions, utility, cost
        e = entry["mean_emissions_cumulative"] * 1e-9
        u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
        c = entry["mean_net_cost"] * 1e-9
        
        # Calculate confidence intervals
        emissions_array = np.array(entry["emissions_cumulative_driving"]) + np.array(entry["emissions_cumulative_production"])
        emissions_array *= 1e-9
        utility_array = np.array(entry["utility_cumulative"]) / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
        cost_array = np.array(entry["net_cost"]) * 1e-9
        n_seeds = len(emissions_array)
        e_err = 1.96 * np.std(emissions_array) / np.sqrt(n_seeds)
        u_err = 1.96 * np.std(utility_array) / np.sqrt(n_seeds)
        c_err = 1.96 * np.std(cost_array) / np.sqrt(n_seeds)
        
        # Calculate marker sizes based on policy intensity
        sizes = {}
        for name, val in zip(policy_names, [carbon_val, elec_val, prod_val]):
            sizes[name] = scale_marker_size(val, name, policy_ranges, base_marker_size, min_size_ratio=0)
        
        # === Top: Net Cost vs Emissions ===
        # Plot error bars
        ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.3, zorder=1)
        
        # Plot outline circle
        ax_top.scatter(e, c, s=base_marker_size, marker=full_circle_marker(), 
                     facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=2, linestyle = "--")
        
        # Plot policy segments
        ax_top.scatter(e, c, s=sizes["Carbon_price"], marker=third_circle_marker(0), 
                     color=policy_colors["Carbon_price"], edgecolor='black', linewidth=0.5, zorder=3)
        ax_top.scatter(e, c, s=sizes["Electricity_subsidy"], marker=third_circle_marker(120), 
                     color=policy_colors["Electricity_subsidy"], edgecolor='black', linewidth=0.5, zorder=3)
        ax_top.scatter(e, c, s=sizes["Production_subsidy"], marker=third_circle_marker(240), 
                     color=policy_colors["Production_subsidy"], edgecolor='black', linewidth=0.5, zorder=3)
        
        # === Bottom: Utility vs Emissions ===
        # Plot error bars
        ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.3, zorder=1)
        
        # Plot outline circle
        ax_bottom.scatter(e, u, s=base_marker_size, marker=full_circle_marker(), 
                        facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=2)
        
        # Plot policy segments
        ax_bottom.scatter(e, u, s=sizes["Carbon_price"], marker=third_circle_marker(0), 
                        color=policy_colors["Carbon_price"], edgecolor='black', linewidth=0.5, zorder=3)
        ax_bottom.scatter(e, u, s=sizes["Electricity_subsidy"], marker=third_circle_marker(120), 
                        color=policy_colors["Electricity_subsidy"], edgecolor='black', linewidth=0.5, zorder=3)
        ax_bottom.scatter(e, u, s=sizes["Production_subsidy"], marker=third_circle_marker(240), 
                        color=policy_colors["Production_subsidy"], edgecolor='black', linewidth=0.5, zorder=3)
    
    # BAU point (Business As Usual)
    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
    bau_cost = outcomes_BAU["mean_net_cost"] * 1e-9
    
    # Plot BAU points
    bau_size = base_marker_size * 0.7
    ax_top.scatter(bau_em, bau_cost, s=bau_size, color='black', marker='o', label="BAU", zorder=4)
    ax_bottom.scatter(bau_em, bau_ut, s=bau_size, color='black', marker='o', zorder=4)
    
    # Add grid lines
    ax_top.grid(True, linestyle='--', alpha=0.7)
    ax_bottom.grid(True, linestyle='--', alpha=0.7)
    
    # Axis labels with larger font
    ax_top.set_ylabel("Net Cost (billion $)", fontsize=12)
    ax_bottom.set_ylabel("Utility (billion $)", fontsize=12)
    ax_bottom.set_xlabel("Emissions (MT CO₂)", fontsize=12)
    
    # Add titles
    ax_top.set_title("Policy Tradeoffs: Net Cost vs. Emissions", fontsize=14)
    ax_bottom.set_title("Policy Tradeoffs: Utility vs. Emissions", fontsize=14)
    
    # Create legend
    legend_elements = []
    
    # Policy color indicators
    for p in policy_names:
        legend_elements.append(
            Patch(facecolor=policy_colors[p], edgecolor='black',
                  label=f"{p.replace('_', ' ')} ({policy_ranges[p]['min']:.2f}–{policy_ranges[p]['max']:.2f})")
        )
        
        #
    # BAU indicator
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label='BAU'))
    
    # Error bar indicator
    confidence = plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', 
                          label='95% Confidence Interval')
    
    # Size indicators for low and high policy intensity
    small_marker = plt.Line2D([0], [0], marker='o', color='white', markerfacecolor='gray', 
                             markeredgecolor='black', markersize=8, linestyle='None', 
                             label='Low Policy Intensity')
    
    large_marker = plt.Line2D([0], [0], marker='o', color='white', markerfacecolor='gray', 
                             markeredgecolor='black', markersize=14, linestyle='None', 
                             label='High Policy Intensity')
    
    legend_elements += [confidence, small_marker, large_marker]
    
    # Add legend to bottom plot
    ax_bottom.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    plt.savefig(f"{file_name}/Plots/emissions_tradeoffs/emissions_tradeoff_triples.png", dpi=dpi)
    print(f"Figure saved to {file_name}/Plots/emissions_tradeoffs/emissions_tradeoff_triples.png")

def main(fileName, fileName_BAU):
    base_params = load_object(f"{fileName}/Data", "base_params")
    policy_outcomes = load_object(f"{fileName}/Data", "policy_outcomes")
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
        fileName="results/2d_grid_endogenous_third_13_03_38__10_04_2025",
        fileName_BAU="results/BAU_runs_12_22_12__10_04_2025"
    )