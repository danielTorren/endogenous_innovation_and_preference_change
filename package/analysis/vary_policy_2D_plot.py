import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from package.resources.utility import load_object, createFolder
from matplotlib.patches import Patch
from matplotlib.collections import PathCollection
from matplotlib.path import Path

# Define all possible measures and their indices
MEASURES = {
    "EV_Uptake": 0,
    "Policy_Distortion": 1,
    "Net_Policy_Cost": 2,
    "Cumulative_Emissions": 3,
    "Driving_Emissions": 4,
    "Production_Emissions": 5,
    "Cumulative_Utility": 6,
    "Cumulative_Profit": 7
}

scale = 350
epsilon = 0

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

def scale_marker_size(value, policy, policy_ranges):
    min_val = 0#policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    if max_val - min_val == 0:
        return 100  # Fixed size if no variation
    norm = (value - min_val) / (max_val - min_val)
    return np.maximum(scale * epsilon, norm * scale)

def plot_emissions_tradeoffs(outcomes_BAU, data_array, policy_pairs, file_name, policy_info_dict,
                           min_ev_uptake=0.9, max_ev_uptake=1.0, dpi=300):
    """
    Creates a 2-panel figure showing:
    - Top: Net Policy Cost vs Emissions
    - Bottom: Utility vs Emissions
    Maintains all original visual encoding (marker sizes, opacity, legend)
    """
    # Create figure with two subplots sharing x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 9), 
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Get measure indices
    emissions_idx = MEASURES["Cumulative_Emissions"]
    utility_idx = MEASURES["Cumulative_Utility"]
    cost_idx = MEASURES["Net_Policy_Cost"]
    ev_idx = MEASURES["EV_Uptake"]
    
    # Color setup
    color_map = plt.get_cmap('Set1', 10)
    all_policies = sorted(list(set([p for pair in policy_pairs for p in pair])))
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}
    
    # Calculate policy ranges
    policy_ranges = {}
    for policy in all_policies:
        p_min, p_max = policy_info_dict['bounds_dict'][policy]
        policy_ranges[policy] = {"min": p_min, "max": p_max}
    
    # Process each policy pair
    for pair_idx, (policy1, policy2) in enumerate(policy_pairs):
        # Get data for this pair
        pair_data = data_array[pair_idx]
        
        # Calculate means and 95% CIs
        emissions = np.mean(pair_data[:, :, :, emissions_idx], axis=2) * 1e-9  # MTCO2
        utility = np.mean(pair_data[:, :, :, utility_idx], axis=2) * 12 * 1e-9  # $bn/year
        cost = np.mean(pair_data[:, :, :, cost_idx], axis=2) * 1e-9  # $bn
        ev = np.mean(pair_data[:, :, :, ev_idx], axis=2)
        
        # Calculate errors (95% CI)
        n_seeds = pair_data.shape[2]
        emissions_err = 1.96 * np.std(pair_data[:, :, :, emissions_idx], axis=2) / np.sqrt(n_seeds) * 1e-9
        utility_err = 1.96 * np.std(pair_data[:, :, :, utility_idx], axis=2) / np.sqrt(n_seeds) * 12 * 1e-9
        cost_err = 1.96 * np.std(pair_data[:, :, :, cost_idx], axis=2) / np.sqrt(n_seeds) * 1e-9
        
        # Get policy intensities
        p1_intensities = np.linspace(policy_ranges[policy1]["min"],
                                    policy_ranges[policy1]["max"],
                                    data_array.shape[1])
        p2_intensities = np.linspace(policy_ranges[policy2]["min"],
                                    policy_ranges[policy2]["max"],
                                    data_array.shape[2])
        
        # Plot each combination
        for i in range(len(p1_intensities)):
            for j in range(len(p2_intensities)):
                if not (min_ev_uptake <= ev[i,j] <= max_ev_uptake):
                    continue
                
                # Calculate visual properties
                opacity = 1
                size1 = scale_marker_size(p1_intensities[i], policy1, policy_ranges)
                size2 = scale_marker_size(p2_intensities[j], policy2, policy_ranges)
                
                # --- Top Panel: Cost vs Emissions ---
                # Error bars
                ax_top.plot([emissions[i,j] - emissions_err[i,j], emissions[i,j] + emissions_err[i,j]], 
                           [cost[i,j], cost[i,j]], 
                           color='gray', alpha=opacity*0.5, zorder=1)
                ax_top.plot([emissions[i,j], emissions[i,j]], 
                           [cost[i,j] - cost_err[i,j], cost[i,j] + cost_err[i,j]], 
                           color='gray', alpha=opacity*0.5, zorder=1)
                
                # Dotted outline
                ax_top.scatter(emissions[i,j], cost[i,j], 
                              s=350, marker=full_circle_marker(), 
                              facecolor='none', edgecolor='black', 
                              linewidth=1.0, alpha=1, linestyle='dashed', zorder=2)
                
                # Half-circle markers
                ax_top.scatter(emissions[i,j], cost[i,j],
                              s=size1, marker=half_circle_marker(0, 180),
                              color=policy_colors[policy1], edgecolor="black",
                              alpha=opacity, zorder=2)
                ax_top.scatter(emissions[i,j], cost[i,j],
                              s=size2, marker=half_circle_marker(180, 360),
                              color=policy_colors[policy2], edgecolor="black",
                              alpha=opacity, zorder=2)
                
                # --- Bottom Panel: Utility vs Emissions ---
                # Error bars
                ax_bottom.plot([emissions[i,j] - emissions_err[i,j], emissions[i,j] + emissions_err[i,j]], 
                              [utility[i,j], utility[i,j]], 
                              color='gray', alpha=opacity*0.5, zorder=1)
                ax_bottom.plot([emissions[i,j], emissions[i,j]], 
                              [utility[i,j] - utility_err[i,j], utility[i,j] + utility_err[i,j]], 
                              color='gray', alpha=opacity*0.5, zorder=1)
                
                # Dotted outline
                ax_bottom.scatter(emissions[i,j], utility[i,j], 
                                s=350, marker=full_circle_marker(), 
                                facecolor='none', edgecolor='black', 
                                linewidth=1.0, alpha=1, linestyle='dashed')
                
                # Half-circle markers
                ax_bottom.scatter(emissions[i,j], utility[i,j],
                                s=size1, marker=half_circle_marker(0, 180),
                                color=policy_colors[policy1], edgecolor="black",
                                alpha=opacity, zorder=2)
                ax_bottom.scatter(emissions[i,j], utility[i,j],
                                s=size2, marker=half_circle_marker(180, 360),
                                color=policy_colors[policy2], edgecolor="black",
                                alpha=opacity, zorder=2)
    
    # --- BAU point with proper error bars ---
    bau_net_cost = (outcomes_BAU["mean_net_cost"])*1e-9
    bau_utility = outcomes_BAU["mean_utility_cumulative"]*12*1e-9
    bau_emission = outcomes_BAU["mean_emissions_cumulative"] * 1e-9

    # Calculate standard errors (make sure these keys exist in your outcomes_BAU dictionary)

    outcomes_BAU["emissions_cumulative"] = outcomes_BAU["emissions_cumulative_driving"] + outcomes_BAU["emissions_cumulative_production"]

    bau_emission_err = 1.96 * np.std(outcomes_BAU["emissions_cumulative"]) / np.sqrt(n_seeds) * 1e-9
    bau_utility_err = 1.96 * np.std(outcomes_BAU["utility_cumulative"]) / np.sqrt(n_seeds) * 12 * 1e-9
    bau_net_cost_err = 1.96 * np.std(outcomes_BAU["net_cost"]) / np.sqrt(n_seeds) * 1e-9

    # Plot BAU points
    bau_marker_size = 350  # Same as other markers
    ax_top.scatter(bau_emission, bau_net_cost, color='black', marker='o', s=bau_marker_size, 
                edgecolor='black', zorder=3, label='BAU')
    ax_bottom.scatter(bau_emission, bau_utility, color='black', marker='o', s=bau_marker_size, 
                    edgecolor='black', zorder=3, label='BAU')

    # Plot error bars (matching your existing style)
    # Top panel (Cost vs Emissions)
    ax_top.plot([bau_emission - bau_emission_err, bau_emission + bau_emission_err],
                [bau_net_cost, bau_net_cost],
                color='gray', alpha=0.5, zorder=2, linewidth=1.5)
    ax_top.plot([bau_emission, bau_emission],
                [bau_net_cost - bau_net_cost_err, bau_net_cost + bau_net_cost_err],
                color='gray', alpha=0.5, zorder=2, linewidth=1.5)

    # Bottom panel (Utility vs Emissions)
    ax_bottom.plot([bau_emission - bau_emission_err, bau_emission + bau_emission_err],
                [bau_utility, bau_utility],
                color='gray', alpha=0.5, zorder=2, linewidth=1.5)
    ax_bottom.plot([bau_emission, bau_emission],
                [bau_utility - bau_utility_err, bau_utility + bau_utility_err],
                color='gray', alpha=0.5, zorder=2, linewidth=1.5)
    
    # --- Formatting ---
    # Top panel
    ax_top.set_ylabel('Net Cost, bn $', fontsize=12)
    ax_top.grid(alpha=0.3)
    
    # Bottom panel
    ax_bottom.set_xlabel('Emissions, MTCO2', fontsize=12)
    ax_bottom.set_ylabel('Utility, bn $', fontsize=12)
    ax_bottom.grid(alpha=0.3)

    # --- Create custom legend elements with intensity indicators ---
    legend_elements = []
    
    # Add policy color patches
    for policy in all_policies:
        legend_elements.append(
            Patch(facecolor=policy_colors[policy], edgecolor='black',
                 label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
        )
    
    # Add intensity indicator half-circles
    scale = 350  # Base size matching your plot
    low_size = scale * 0.3  # Small size for low intensity
    high_size = scale * 1.0  # Large size for high intensity
    
    # Create proxies with Line2D using your Path as marker
    low_intensity_proxy = plt.Line2D([0], [0],
        marker=half_circle_marker(0, 180),
        markersize=np.sqrt(low_size / np.pi),  # Convert area (scatter size) to marker radius
        color='gray', markerfacecolor='gray', markeredgecolor='black', linestyle='None', label='Low intensity'
    )

    high_intensity_proxy = plt.Line2D([0], [0],
        marker=half_circle_marker(0, 180),
        markersize=np.sqrt(high_size / np.pi),
        color='gray', markerfacecolor='gray', markeredgecolor='black', linestyle='None', label='High intensity'
    )
        
    legend_elements.extend([
        low_intensity_proxy,
        high_intensity_proxy,
        plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval'),
        Patch(facecolor='black', edgecolor='black',
             label='Business as Usual')
    ])
    
    # Place legend in bottom right
    leg = ax_bottom.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        framealpha=1,
        handletextpad=1.5,
        borderpad=1
    )
    
    # Adjust legend layout
    for handle in leg.legendHandles:
        if isinstance(handle, PathCollection):  # The half-circle markers
            handle.set_sizes([30])  # Standardize legend marker size
    
    plt.tight_layout()
    
    # Save
    createFolder(f"{file_name}/Plots/emissions_tradeoffs")
    plt.savefig(
        f"{file_name}/Plots/emissions_tradeoffs/cost_utility_vs_emissions_{min_ev_uptake}_{max_ev_uptake}.png",
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.show()

def calc_low_intensities_from_array(data_array, policy_pairs, policy_info_dict, min_val, max_val):
    best_entries = {}

    for pair_idx, (policy1, policy2) in enumerate(policy_pairs):
        pair_data = data_array[pair_idx]  # Shape: (len1, len2, seeds, metrics)
        ev_uptake = np.mean(pair_data[:, :, :, MEASURES["EV_Uptake"]], axis=2)

        # Get intensity values from bounds
        p1_min, p1_max = policy_info_dict['bounds_dict'][policy1]
        p2_min, p2_max = policy_info_dict['bounds_dict'][policy2]

        p1_vals = np.linspace(p1_min, p1_max, pair_data.shape[0])
        p2_vals = np.linspace(p2_min, p2_max, pair_data.shape[1])

        min_max_norm = float("inf")
        best_entry = None

        for i in range(pair_data.shape[0]):
            for j in range(pair_data.shape[1]):
                mean_ev = ev_uptake[i, j]
                if  (min_val <= mean_ev <= max_val):
                    # Normalize both intensities
                    p1_norm = (p1_vals[i] - p1_min) / (p1_max - p1_min)
                    p2_norm = (p2_vals[j] - p2_min) / (p2_max - p2_min)
                    max_intensity = max(p1_norm, p2_norm)

                    if max_intensity < min_max_norm:
                        min_max_norm = max_intensity
                        best_entry = {
                            "policy1": policy1,
                            "policy2": policy2,
                            "policy1_value": p1_vals[i],
                            "policy2_value": p2_vals[j],
                            "mean_ev_uptake": mean_ev,
                            "policy1_value_relative": p1_norm,
                            "policy2_value_relative": p2_norm,
                        }

        if best_entry:
            best_entries[(policy1, policy2)] = best_entry
            print(f"{policy1}-{policy2} -> ({best_entry['policy1_value']:.3f}, {best_entry['policy2_value']:.3f})  EV uptake: {best_entry['mean_ev_uptake']:.3f}")
            print(best_entry['policy1_value_relative'], best_entry['policy2_value_relative'])
    return best_entries

def plot_emissions_tradeoffs_low(outcomes_BAU, data_array, policy_pairs, file_name, policy_info_dict,
                             min_ev_uptake=0.9, max_ev_uptake=1.0, dpi=300):
    """
    Creates a 2-panel figure showing:
    - Top: Net Policy Cost vs Emissions
    - Bottom: Utility vs Emissions
    Only plots the lowest-intensity policy combinations per pair within the EV uptake range.
    """

    # Create figure
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 9), 
                                            sharex=True,
                                            gridspec_kw={'height_ratios': [1, 1.2]})

    # Get indices
    emissions_idx = MEASURES["Cumulative_Emissions"]
    utility_idx = MEASURES["Cumulative_Utility"]
    cost_idx = MEASURES["Net_Policy_Cost"]
    ev_idx = MEASURES["EV_Uptake"]

    # Color setup
    color_map = plt.get_cmap('Set1', 10)
    all_policies = sorted(list(set([p for pair in policy_pairs for p in pair])))
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}

    # Policy bounds
    policy_ranges = {
        policy: {"min": policy_info_dict['bounds_dict'][policy][0],
                 "max": policy_info_dict['bounds_dict'][policy][1]}
        for policy in all_policies
    }

    # --- Get lowest-intensity satisfying entries
    low_intensity_points = calc_low_intensities_from_array(
        data_array, policy_pairs, policy_info_dict,
        min_ev_uptake, max_ev_uptake
    )

    # Plot only the selected optimal points
    for pair_idx, (policy1, policy2) in enumerate(policy_pairs):
        pair_data = data_array[pair_idx]
        emissions = np.mean(pair_data[:, :, :, emissions_idx], axis=2) * 1e-9
        utility = np.mean(pair_data[:, :, :, utility_idx], axis=2) * 12 * 1e-9
        cost = np.mean(pair_data[:, :, :, cost_idx], axis=2) * 1e-9
        ev = np.mean(pair_data[:, :, :, ev_idx], axis=2)

        n_seeds = pair_data.shape[2]
        emissions_err = 1.96 * np.std(pair_data[:, :, :, emissions_idx], axis=2) / np.sqrt(n_seeds) * 1e-9
        utility_err = 1.96 * np.std(pair_data[:, :, :, utility_idx], axis=2) / np.sqrt(n_seeds) * 12 * 1e-9
        cost_err = 1.96 * np.std(pair_data[:, :, :, cost_idx], axis=2) / np.sqrt(n_seeds) * 1e-9

        p1_vals = np.linspace(policy_ranges[policy1]["min"],
                              policy_ranges[policy1]["max"],
                              data_array.shape[1])
        p2_vals = np.linspace(policy_ranges[policy2]["min"],
                              policy_ranges[policy2]["max"],
                              data_array.shape[2])

        key = (policy1, policy2)
        if key not in low_intensity_points:
            continue

        # Get best (i, j) for this pair
        best = low_intensity_points[key]
        i = (np.abs(p1_vals - best['policy1_value'])).argmin()
        j = (np.abs(p2_vals - best['policy2_value'])).argmin()

        # Marker sizes
        size1 = scale_marker_size(p1_vals[i], policy1, policy_ranges)
        size2 = scale_marker_size(p2_vals[j], policy2, policy_ranges)

        # Top Panel
        ax_top.plot([emissions[i, j] - emissions_err[i, j], emissions[i, j] + emissions_err[i, j]],
                    [cost[i, j], cost[i, j]],
                    color='gray', alpha=0.5, zorder=1)
        ax_top.plot([emissions[i, j], emissions[i, j]],
                    [cost[i, j] - cost_err[i, j], cost[i, j] + cost_err[i, j]],
                    color='gray', alpha=0.5, zorder=1)

        ax_top.scatter(emissions[i, j], cost[i, j],
                       s=size1, marker=half_circle_marker(0, 180),
                       color=policy_colors[policy1], edgecolor="black", zorder=2)
        ax_top.scatter(emissions[i, j], cost[i, j],
                       s=size2, marker=half_circle_marker(180, 360),
                       color=policy_colors[policy2], edgecolor="black", zorder=2)

        # Bottom Panel
        ax_bottom.plot([emissions[i, j] - emissions_err[i, j], emissions[i, j] + emissions_err[i, j]],
                       [utility[i, j], utility[i, j]],
                       color='gray', alpha=0.5, zorder=1)
        ax_bottom.plot([emissions[i, j], emissions[i, j]],
                       [utility[i, j] - utility_err[i, j], utility[i, j] + utility_err[i, j]],
                       color='gray', alpha=0.5, zorder=1)

        ax_bottom.scatter(emissions[i, j], utility[i, j],
                          s=size1, marker=half_circle_marker(0, 180),
                          color=policy_colors[policy1], edgecolor="black", zorder=2)
        ax_bottom.scatter(emissions[i, j], utility[i, j],
                          s=size2, marker=half_circle_marker(180, 360),
                          color=policy_colors[policy2], edgecolor="black", zorder=2)

    # --- BAU point
    bau_net_cost = outcomes_BAU["mean_net_cost"] * 1e-9
    bau_utility = outcomes_BAU["mean_utility_cumulative"] * 12 * 1e-9
    bau_emission = outcomes_BAU["mean_emissions_cumulative"] * 1e-9

    bau_emission_err = 1.96 * np.std(outcomes_BAU["emissions_cumulative"]) / np.sqrt(n_seeds) * 1e-9
    bau_utility_err = 1.96 * np.std(outcomes_BAU["utility_cumulative"]) / np.sqrt(n_seeds) * 12 * 1e-9
    bau_net_cost_err = 1.96 * np.std(outcomes_BAU["net_cost"]) / np.sqrt(n_seeds) * 1e-9

    ax_top.scatter(bau_emission, bau_net_cost, color='black', marker='o', s=350,
                   edgecolor='black', zorder=3, label='BAU')
    ax_bottom.scatter(bau_emission, bau_utility, color='black', marker='o', s=350,
                      edgecolor='black', zorder=3)

    ax_top.plot([bau_emission - bau_emission_err, bau_emission + bau_emission_err],
                [bau_net_cost, bau_net_cost], color='gray', alpha=0.5, linewidth=1.5)
    ax_top.plot([bau_emission, bau_emission],
                [bau_net_cost - bau_net_cost_err, bau_net_cost + bau_net_cost_err],
                color='gray', alpha=0.5, linewidth=1.5)

    ax_bottom.plot([bau_emission - bau_emission_err, bau_emission + bau_emission_err],
                   [bau_utility, bau_utility], color='gray', alpha=0.5, linewidth=1.5)
    ax_bottom.plot([bau_emission, bau_emission],
                   [bau_utility - bau_utility_err, bau_utility + bau_utility_err],
                   color='gray', alpha=0.5, linewidth=1.5)

    # Labels
    ax_top.set_ylabel('Net Policy Cost ($bn)', fontsize=12)
    ax_top.grid(alpha=0.3)
    ax_bottom.set_xlabel('Cumulative Emissions (MTCO2)', fontsize=12)
    ax_bottom.set_ylabel('Cumulative Utility ($bn)', fontsize=12)
    ax_bottom.grid(alpha=0.3)

    # Legend
    legend_elements = []
    for policy in all_policies:
        legend_elements.append(Patch(facecolor=policy_colors[policy], edgecolor='black',
                                     label=f"{policy.replace('_', ' ')}"))

    # Half-circle marker sizes
    scale = 350
    low_size = scale * 0.3
    high_size = scale * 1.0
    low_intensity_proxy = plt.scatter([], [], s=low_size, marker=half_circle_marker(0, 180),
                                      facecolor='gray', edgecolor='black')
    high_intensity_proxy = plt.scatter([], [], s=high_size, marker=half_circle_marker(0, 180),
                                       facecolor='gray', edgecolor='black')

    legend_elements.extend([
        low_intensity_proxy,
        plt.Line2D([0], [0], marker='', color='w', label='Low intensity'),
        high_intensity_proxy,
        plt.Line2D([0], [0], marker='', color='w', label='High intensity'),
        Patch(facecolor='gray', edgecolor='gray', alpha=0.5, label='95% CI'),
        Patch(facecolor='black', edgecolor='black', label='Business as Usual')
    ])

    leg = ax_bottom.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        framealpha=1,
        handletextpad=1.5,
        borderpad=1
    )

    for handle in leg.legendHandles:
        if isinstance(handle, PathCollection):
            handle.set_sizes([30])

    plt.tight_layout()

    createFolder(f"{file_name}/Plots/emissions_tradeoffs")
    plt.savefig(
        f"{file_name}/Plots/emissions_tradeoffs/cost_utility_vs_emissions_single_lowest_{min_ev_uptake}_{max_ev_uptake}.png",
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.show()


def main(file_name, fileName_BAU):
    # Load data
    data_array = load_object(file_name + "/Data", "data_array")
    policy_pairs = load_object(file_name + "/Data", "policy_pairs")
    policy_info_dict = load_object(file_name + "/Data", "policy_info_dict")
    outcomes_BAU = load_object(f"{fileName_BAU}/Data", "outcomes")
   
    plot_emissions_tradeoffs(
        outcomes_BAU = outcomes_BAU,
        data_array=data_array,
        policy_pairs=policy_pairs,
        file_name=file_name,
        policy_info_dict=policy_info_dict,
        min_ev_uptake=0.945,  # Your preferred EV uptake range
        max_ev_uptake=0.955,
        dpi=300
    )
    plt.show()

if __name__ == "__main__":
    main(
        file_name="results/vary_two_policies_gen_00_12_09__02_04_2025",
        fileName_BAU="results/BAU_runs_11_23_45__02_04_2025",
        )