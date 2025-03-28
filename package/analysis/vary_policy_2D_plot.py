import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from package.resources.utility import load_object, createFolder
from matplotlib.patches import Patch
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
    min_val = policy_ranges[policy]["min"]
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
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 10), 
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
                              s=scale, marker=full_circle_marker(), 
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
    
    # --- BAU point

    bau_net_cost = (outcomes_BAU["mean_net_cost"])*1e-9
    bau_utility= outcomes_BAU["mean_utility_cumulative"]*12*1e-9
    bau_emission = outcomes_BAU["mean_emissions_cumulative"] * 1e-9

    bau_emission_err = 1.96 * np.std(outcomes_BAU["ev_uptake"]) / np.sqrt(n_seeds) * 1e-9
    bau_utility_err = 1.96 * np.std(outcomes_BAU["utility_cumulative"]) / np.sqrt(n_seeds) * 12 * 1e-9
    bau_net_cost_err = 1.96 * np.std(outcomes_BAU["net_cost"]) / np.sqrt(n_seeds) * 1e-9
    
    print()
    ax_top.scatter(bau_emission, bau_net_cost, color='black', marker='o', s=scale, edgecolor='black')
    ax_bottom.scatter(bau_emission, bau_utility, color='black', marker='o', s=scale, edgecolor='black')
    # Error bars
    ax_bottom.plot(bau_emission_err, bau_net_cost_err, color='gray', alpha=opacity*0.5, zorder=1)
    ax_bottom.plot(bau_emission_err, bau_utility_err, color='gray', alpha=opacity*0.5, zorder=1)
    
    # --- Formatting ---
    # Top panel
    ax_top.set_ylabel('Net Policy Cost ($bn)', fontsize=12)
    ax_top.grid(alpha=0.3)
    
    # Bottom panel
    ax_bottom.set_xlabel('Cumulative Emissions (MTCO2)', fontsize=12)
    ax_bottom.set_ylabel('Cumulative Utility ($bn)', fontsize=12)
    ax_bottom.grid(alpha=0.3)

    
    # Create comprehensive legend
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
             label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f}-{policy_ranges[policy]['max']:.2f})")
        for policy in all_policies
    ]
    legend_elements.extend([
        Patch(facecolor='gray', edgecolor='gray', alpha=0.5,
             label='95% Confidence Interval'),
    ])
    legend_elements.append(Patch(facecolor='black', edgecolor='black', label="Business as Usual (BAU)"))
    
    # Place legend in bottom right of lower plot
    ax_bottom.legend(
        handles=legend_elements, 
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),  # Anchors to bottom right corner
        ncol=1,  # Single column for vertical layout
        fontsize=10,
        framealpha=1  # Make legend background opaque
    )

    plt.tight_layout()
    
    # Save
    createFolder(f"{file_name}/Plots/emissions_tradeoffs")
    plt.savefig(
        f"{file_name}/Plots/emissions_tradeoffs/cost_utility_vs_emissions.png",
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
        min_ev_uptake=0.94,  # Your preferred EV uptake range
        max_ev_uptake=0.96,
        dpi=300
    )
    plt.show()

if __name__ == "__main__":
    main(
        file_name="results/vary_two_policies_gen_19_45_01__27_03_2025",
        fileName_BAU="results/BAU_runs_11_18_33__23_03_2025",
        )