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
    scale = 350
    epsilon = 0
    return np.maximum(scale * epsilon, norm * scale)

# Opacity calculation function
def calculate_opacity_old(ev_uptake, min_ev_uptake, max_ev_uptake):
    return np.clip((ev_uptake - min_ev_uptake) / (max_ev_uptake - min_ev_uptake), 0, 1.0)

def calculate_opacity_other(ev_uptake, min_ev_uptake=0.9, max_ev_uptake=0.95):
    """
    Calculate opacity that reaches 1 at max_ev_uptake (default 0.95)
    and stays at 1 for all higher values.
    """
    if ev_uptake >= max_ev_uptake:
        return 1.0
    return np.clip((ev_uptake - min_ev_uptake) / (max_ev_uptake - min_ev_uptake), 0, 1)


def calculate_opacity(ev_uptake, peak_ev=0.95, width=0.05):
    """
    Opacity peaks at `peak_ev` (default 0.95) and decreases linearly 
    as EV uptake moves away from this value.
    
    Args:
        ev_uptake (float): Current EV uptake value (0-1).
        peak_ev (float): EV uptake value where opacity is maximum (default 0.95).
        width (float): Controls how quickly opacity falls off (default ±0.05).
    
    Returns:
        float: Opacity between 0 and 1.
    """
    distance = np.abs(ev_uptake - peak_ev)
    opacity = 1 - (distance / width)
    return np.clip(opacity, 0, 1)

# Example usage:
# generate_all_measure_plots(data_array, policy_pairs, "results/output", policy_info_dict)

def plot_policy_pair_effects(data_array, policy_pairs, file_name, policy_info_dict, dpi=300):
    """
    Creates one figure per policy pair with subplots for each measure and a shared colorbar.
    
    Parameters:
    - data_array: Array of shape (num_policy_pairs, p1_steps, p2_steps, seeds, outputs)
    - policy_pairs: List of (policy1, policy2) tuples
    - file_name: Base directory for saving plots
    - policy_info_dict: Dictionary containing policy bounds
    - dpi: Resolution for saved figures
    """
    createFolder(f"{file_name}/Plots/2D_policy_grids")
    
    num_measures = len(SELECTED_MEASURES)
    
    for pair_idx, (policy1, policy2) in enumerate(policy_pairs):
        # Get bounds for both policies
        p1_min, p1_max = policy_info_dict['bounds_dict'][policy1]
        p2_min, p2_max = policy_info_dict['bounds_dict'][policy2]
        
        # Create intensity arrays
        p1_intensities = np.linspace(p1_min, p1_max, data_array.shape[1])
        p2_intensities = np.linspace(p2_min, p2_max, data_array.shape[2])
        
        # Get data for this policy pair
        pair_data = data_array[pair_idx]  # shape (p1_steps, p2_steps, seeds, outputs)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, num_measures, figsize=(4*num_measures, 4))
        if num_measures == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Create normalization for colorbar
        norm = Normalize(vmin=p2_min, vmax=p2_max)
        cmap = plt.cm.viridis
        
        # Plot each measure
        for ax_idx, measure_name in enumerate(SELECTED_MEASURES):
            measure_idx = MEASURES_DICT[measure_name]
            
            # Calculate mean across seeds
            mean_values = np.mean(pair_data[:, :, :, measure_idx], axis=2)
            
            # Special scaling for utility
            if measure_name == "Cumulative Utility":
                mean_values *= 12
            
            # Plot lines for each p2 intensity
            for j in range(len(p2_intensities)):
                color = cmap(norm(p2_intensities[j]))
                axes[ax_idx].plot(p1_intensities, mean_values[:, j], color=color, alpha=0.7)
            
            # Set labels
            axes[ax_idx].set_xlabel(f'{policy1} Intensity')
            axes[ax_idx].set_ylabel(measure_name)
            axes[ax_idx].grid(alpha=0.3)
        
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[-1], location='right', pad=0.02)
        cbar.set_label(f'{policy2} Intensity')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            f"{file_name}/Plots/2D_policy_grids/policy_pair_{pair_idx}_{policy1}_{policy2}.png",
            dpi=dpi
        )

def plot_policy_pair_effects_stacked(data_array, policy_pairs, file_name, policy_info_dict, 
                                   group_size=5, dpi=300):
    """
    Creates plots with:
    - Rows: Policy combinations
    - Columns: Measures
    - One colorbar per row showing Policy 2 intensity
    - Better spacing and label handling
    """
    createFolder(f"{file_name}/Plots/2D_policy_stacked_row_cbars")
    
    num_measures = len(SELECTED_MEASURES)
    num_pairs = len(policy_pairs)
    
    # Split into groups
    groups = [policy_pairs[:group_size], policy_pairs[-group_size:]] if num_pairs > group_size else [policy_pairs]
    
    for group_idx, group_pairs in enumerate(groups):
        # Adjust figure dimensions and spacing
        fig_width = 2.5 * num_measures  # Increased width for better spacing
        fig_height = 2.5 * len(group_pairs)  # Increased height
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Calculate spacing parameters
        left_margin = 0.08
        right_margin = 0.08
        bottom_margin = 0.08
        top_margin = 0.92
        hspace = 0.4  # Vertical space between rows
        wspace = 0.3  # Horizontal space between columns
        
        # Calculate available space after margins
        plot_width = 1 - left_margin - right_margin - 0.05  # 0.05 for colorbar
        plot_height = top_margin - bottom_margin
        
        # Create main axes for plots
        plot_axes = []
        for row_idx, (policy1, policy2) in enumerate(group_pairs):
            row_axes = []
            row_bottom = bottom_margin + (len(group_pairs) - row_idx - 1) * (plot_height/len(group_pairs))
            
            for col_idx in range(num_measures):
                # Position calculations with more spacing
                left = left_margin + col_idx * (plot_width/num_measures)
                bottom = row_bottom + hspace * 0.2  # Extra padding
                width = (plot_width/num_measures) * 0.8  # Reduced width for spacing
                height = (plot_height/len(group_pairs)) * 0.7  # Reduced height for spacing
                
                ax = fig.add_axes([left, bottom, width, height])
                row_axes.append(ax)
            plot_axes.append(row_axes)
        
        # Create colorbar axes for each row
        cbar_axes = []
        for row_idx in range(len(group_pairs)):
            row_bottom = bottom_margin + (len(group_pairs) - row_idx - 1) * (plot_height/len(group_pairs))
            
            left = left_margin + plot_width + 0.01  # Position after plots
            bottom = row_bottom + hspace * 0.2  # Match plot position
            width = 0.02
            height = (plot_height/len(group_pairs)) * 0.7  # Match plot height
            
            cax = fig.add_axes([left, bottom, width, height])
            cbar_axes.append(cax)
        
        # Plot data
        for row_idx, (policy1, policy2) in enumerate(group_pairs):
            pair_idx = policy_pairs.index((policy1, policy2))
            
            # Get bounds and intensities
            p1_min, p1_max = policy_info_dict['bounds_dict'][policy1]
            p2_min, p2_max = policy_info_dict['bounds_dict'][policy2]
            p1_intensities = np.linspace(p1_min, p1_max, data_array.shape[1])
            p2_intensities = np.linspace(p2_min, p2_max, data_array.shape[2])
            
            # Get data for this policy pair
            pair_data = data_array[pair_idx]
            
            # Plot each measure
            for col_idx, measure_name in enumerate(SELECTED_MEASURES):
                ax = plot_axes[row_idx][col_idx]
                measure_idx = MEASURES_DICT[measure_name]
                
                # Calculate mean across seeds
                mean_values = np.mean(pair_data[:, :, :, measure_idx], axis=2)
                
                # Special scaling for utility
                if measure_name == "Cumulative Utility":
                    mean_values *= 12

                ax.grid(alpha=0.3)

                # Plot lines for each p2 intensity
                for j in range(len(p2_intensities)):
                    color = plt.cm.viridis((p2_intensities[j] - p2_min)/(p2_max - p2_min))
                    ax.plot(p1_intensities, mean_values[:, j], color=color, alpha=0.7, linewidth=1)
                
                # Set labels with better formatting
                ax.set_xlabel(f'{policy1.replace("_", " ")}', fontsize=9, labelpad=5)
                ax.set_ylabel(measure_name, fontsize=9, labelpad=5)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Add measure title at top
                #if row_idx == 0:
                #    ax.set_title(measure_name, fontsize=10, pad=10)
        
        # Add colorbars with better formatting
        for row_idx, (policy1, policy2) in enumerate(group_pairs):
            p2_min, p2_max = policy_info_dict['bounds_dict'][policy2]
            norm = Normalize(vmin=p2_min, vmax=p2_max)
            
            sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cbar_axes[row_idx])
            cb.set_label(f'{policy2.replace("_", " ")}', fontsize=9, labelpad=10)
            cb.ax.tick_params(labelsize=8)
        
        # Adjust layout further
        plt.subplots_adjust(
            left=left_margin,
            right=right_margin + plot_width + 0.05,
            bottom=bottom_margin,
            top=top_margin,
            wspace=wspace,
            hspace=hspace
        )
        
        plt.savefig(
            f"{file_name}/Plots/2D_policy_stacked_row_cbars/policy_group_{group_idx}.png",
            dpi=dpi,
            bbox_inches='tight'
        )


def plot_measure_comparison(data_array, policy_pairs, file_name, policy_info_dict,
                                      x_measure_idx, x_measure_name,
                                      y_measure_idx, y_measure_name,
                                      min_ev_uptake=0, max_ev_uptake=1,
                                      dpi=300, annotate_ev=True,  opacity_state = True):
    """
    Enhanced version with 2D error bars showing 95% confidence intervals
    """
    # Create output directory
    plot_name = f"{y_measure_name}_vs_{x_measure_name}_with_errors"
    #createFolder(f"{file_name}/Plots/{plot_name}")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 10))  # Larger figure for error bars
    ax.grid(True, alpha=0.3)
    
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
        # Get intensities
        p1_intensities = np.linspace(policy_ranges[policy1]["min"], 
                                    policy_ranges[policy1]["max"], 
                                    data_array.shape[1])
        p2_intensities = np.linspace(policy_ranges[policy2]["min"], 
                                    policy_ranges[policy2]["max"], 
                                    data_array.shape[2])
        
        # Get data for this pair
        pair_data = data_array[pair_idx]  # p1_steps × p2_steps × seeds × outputs
        
        # Calculate means and confidence intervals
        x_data = pair_data[:, :, :, x_measure_idx]
        y_data = pair_data[:, :, :, y_measure_idx]
        ev_data = pair_data[:, :, :, MEASURES["EV_Uptake"]]
        
        # Calculate means
        x_means = np.mean(x_data, axis=2)
        y_means = np.mean(y_data, axis=2)
        ev_means = np.mean(ev_data, axis=2)
        
        # Calculate 95% confidence intervals (1.96 * standard error)
        n_seeds = x_data.shape[2]  # Number of seeds
        x_err = 1.96 * np.std(x_data, axis=2) / np.sqrt(n_seeds)
        y_err = 1.96 * np.std(y_data, axis=2) / np.sqrt(n_seeds)
        
        # Apply scaling if needed

        if x_measure_name == "Cumulative_Utility":
            x_means = (x_means * 1e-9)* 12
            x_err = (x_err * 1e-9)* 12
        else:
            x_means = x_means * 1e-9
            x_err = x_err * 1e-9

        if y_measure_name == "Cumulative_Utility":
            y_means = (y_means * 1e-9)* 12
            y_err = (y_err * 1e-9 )* 12
        else:
            y_means = y_means * 1e-9
            y_err = y_err * 1e-9
        
        # Plot each combination with error bars
        for i in range(len(p1_intensities)):
            for j in range(len(p2_intensities)):
                x = x_means[i, j]
                y = y_means[i, j]
                x_error = x_err[i, j]
                y_error = y_err[i, j]
                ev = ev_means[i, j]
                if  opacity_state:
                    opacity = calculate_opacity(ev, min_ev_uptake, max_ev_uptake)
                else:
                    opacity = 1

                
                # Skip if outside EV uptake range
                if not (min_ev_uptake <= ev <= max_ev_uptake):
                    continue
                
                # Plot error bars first (behind markers)
                if p1_intensities[i] > 0 and p2_intensities[j] > 0:
                    # Horizontal error
                    ax.plot([x - x_error, x + x_error], [y, y], 
                           color='gray', alpha=opacity*0.5, zorder=1)
                    # Vertical error
                    ax.plot([x, x], [y - y_error, y + y_error], 
                           color='gray', alpha=opacity*0.5, zorder=1)
                
                # Add dotted black outline with low alpha
                size_single = 350
                ax.scatter(x, y, s=size_single, marker=full_circle_marker(), facecolor='none',
                            edgecolor='black', linewidth=1.0, alpha=1, linestyle='dashed')
        
                # Plot half-circle markers
                if p1_intensities[i] > 0 and p2_intensities[j] > 0:
                    size1 = scale_marker_size(p1_intensities[i], policy1, policy_ranges)
                    size2 = scale_marker_size(p2_intensities[j], policy2, policy_ranges)
                    
                    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180),
                              color=policy_colors[policy1], edgecolor="black", 
                              alpha=opacity, zorder=2)
                    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360),
                              color=policy_colors[policy2], edgecolor="black", 
                              alpha=opacity, zorder=2)
                    
                    # Add EV uptake annotation if requested
                    if annotate_ev:
                        ax.annotate(f"{ev:.3f}", (x, y), textcoords="offset points",
                                   xytext=(10,5), ha='center', fontsize=8, zorder=3)

    # Labels and legend
    ax.set_xlabel(f"{x_measure_name} with 95% CI")
    ax.set_ylabel(f"{y_measure_name} with 95% CI")
    
    # Create custom legend elements for error bars
    error_patch = Patch(facecolor='gray', edgecolor='gray', alpha=0.5,
                       label='95% Confidence Interval')
    
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
             label=f"{policy.replace('_', ' ')} ({policy_ranges[policy]['min']:.2f}-{policy_ranges[policy]['max']:.2f})")
        for policy in all_policies
    ]
    legend_elements.extend([
        error_patch,
        Patch(facecolor='gray', edgecolor='black', alpha=0.5,
             label=f'Opacity ∝ EV uptake ({min_ev_uptake}-{max_ev_uptake})')
    ])
    
    ax.legend(handles=legend_elements, loc="best", title="Policy Intensity Ranges")
    
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/{plot_name}.png", dpi=dpi)

def generate_all_measure_plots(data_array, policy_pairs, file_name, policy_info_dict,
                             min_ev_uptake=0, max_ev_uptake=1, dpi=300, opacity_state = True):
    """
    Helper function to generate all combinations of measure plots
    
    Parameters:
    - data_array: 5D array of results
    - policy_pairs: List of policy pairs
    - file_name: Output directory
    - policy_info_dict: Policy information dictionary
    - min/max_ev_uptake: EV uptake range for opacity
    - dpi: Output resolution
    """
    # Create list of measures to compare
    measure_combinations = [
        (MEASURES["Cumulative_Emissions"], "Cumulative_Emissions (MTC02)",
         MEASURES["Cumulative_Utility"], "Cumulative_Utility (bn $)"),
        
        (MEASURES["Cumulative_Emissions"], "Cumulative_Emissions (MTC02)",
         MEASURES["Cumulative_Profit"], "Cumulative_Profit (bn $)"),
        
        (MEASURES["Cumulative_Emissions"], "Cumulative_Emissions (MTC02)",
         MEASURES["Net_Policy_Cost"], "Net_Policy_Cost (bn $)"),
        
        (MEASURES["Cumulative_Utility"], "Cumulative_Utility (bn $)",
         MEASURES["Cumulative_Profit"], "Cumulative_Profit (bn $)"),
        
        (MEASURES["Cumulative_Utility"], "Cumulative_Utility (bn $)",
         MEASURES["Net_Policy_Cost"], "Net_Policy_Cost (bn $)"),
        # Add more combinations as needed
    ]
    
    # Generate all plots
    for x_idx, x_name, y_idx, y_name in measure_combinations:
        plot_measure_comparison(
            data_array=data_array,
            policy_pairs=policy_pairs,
            file_name=file_name,
            policy_info_dict=policy_info_dict,
            x_measure_idx=x_idx,
            x_measure_name=x_name,
            y_measure_idx=y_idx,
            y_measure_name=y_name,
            min_ev_uptake=min_ev_uptake,
            max_ev_uptake=max_ev_uptake,
            dpi=dpi,
            annotate_ev=False,
            opacity_state = opacity_state
        )
        print("DONE: ", x_name, y_name)

def plot_3d_policy_comparison(data_array, policy_pairs, file_name, policy_info_dict, 
                            min_ev_uptake=0.9, max_ev_uptake=1.0, dpi=300):
    """
    Creates a 3D scatter plot comparing utility, emissions, and net policy cost.
    
    Parameters:
    - data_array: Array of shape (num_policy_pairs, p1_steps, p2_steps, seeds, outputs)
    - policy_pairs: List of (policy1, policy2) tuples
    - file_name: Base directory for saving plots
    - policy_info_dict: Dictionary containing policy bounds
    - min/max_ev_uptake: EV uptake range for filtering points
    - dpi: Resolution for saved figures
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get measure indices
    utility_idx = MEASURES["Cumulative_Utility"]
    cost_idx = MEASURES["Net_Policy_Cost"]
    emissions_idx = MEASURES["Cumulative_Emissions"]
    ev_idx = MEASURES["EV_Uptake"]
    
    # Color setup
    color_map = plt.get_cmap('Set1', 10)
    all_policies = sorted(list(set([p for pair in policy_pairs for p in pair])))
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}
    
    # Process each policy pair
    for pair_idx, (policy1, policy2) in enumerate(policy_pairs):
        # Get data for this pair
        pair_data = data_array[pair_idx]  # p1_steps × p2_steps × seeds × outputs
        
        # Calculate means across seeds
        utility = np.mean(pair_data[:, :, :, utility_idx], axis=2) * 12 * 1e-9  # Convert to $bn/year
        cost = np.mean(pair_data[:, :, :, cost_idx], axis=2) * 1e-9  # Convert to $bn
        emissions = np.mean(pair_data[:, :, :, emissions_idx], axis=2) * 1e-6  # Convert to MTCO2
        ev = np.mean(pair_data[:, :, :, ev_idx], axis=2)
        
        # Get policy intensities
        p1_intensities = np.linspace(policy_info_dict['bounds_dict'][policy1][0],policy_info_dict['bounds_dict'][policy1][1],data_array.shape[1])
        p2_intensities = np.linspace(policy_info_dict['bounds_dict'][policy2][0], policy_info_dict['bounds_dict'][policy2][1], data_array.shape[2])
        
        # Plot each combination
        for i in range(len(p1_intensities)):
            for j in range(len(p2_intensities)):
                if not (min_ev_uptake <= ev[i,j] <= max_ev_uptake):
                    continue
                
                # Calculate marker properties
                opacity = calculate_opacity(ev[i,j], min_ev_uptake, max_ev_uptake)
                size = 100 + 500 * (ev[i,j] - min_ev_uptake)/(max_ev_uptake - min_ev_uptake)
                
                # Plot point with composite color
                ax.scatter(
                    cost[i,j], emissions[i,j], utility[i,j],
                    s=size,
                    c=policy_colors[policy1],
                    marker='o',
                    alpha=opacity,
                    edgecolors=[policy_colors[policy2]],
                    linewidths=2
                )
    
    # Labels and title
    ax.set_xlabel('Net Policy Cost ($bn)', labelpad=10)
    ax.set_ylabel('Cumulative Emissions (MTCO2)', labelpad=10)
    ax.set_zlabel('Cumulative Utility ($bn/year)', labelpad=10)
    ax.set_title('Policy Trade-offs: Cost vs Emissions vs Utility', pad=20)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=policy_colors[policy], edgecolor='black',
             label=f"{policy.replace('_', ' ')}")
        for policy in all_policies
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust view
    ax.view_init(elev=20, azim=-45)
    
    # Save and show
    createFolder(f"{file_name}/Plots/3D_policy_comparison")
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/3D_policy_comparison/cost_emissions_utility.png", dpi=dpi)
    plt.show()


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
                              s=350, marker=full_circle_marker(), 
                              facecolor='none', edgecolor='black', 
                              linewidth=1.0, alpha=1, linestyle='dashed')
                
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
    bau_net_cost = outcomes_BAU["mean_net_cost"]* 1e-9
    bau_utility= outcomes_BAU["mean_utility_cumulative"] *12* 1e-9

    bau_emissions = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    ax_bottom.scatter(bau_emissions, bau_net_cost, color='black', marker='o', s=350, edgecolor='black', label="Business as Usual (BAU)")
    ax_top.scatter(bau_emissions, bau_utility, color='black', marker='o', s=350, edgecolor='black')

    # --- Formatting ---
    # Top panel
    ax_top.set_ylabel('Net Policy Cost ($bn)', fontsize=12)
    ax_top.grid(alpha=0.3)
    #ax_top.set_title('Policy Cost vs Emissions', pad=15, fontsize=14)
    
    # Bottom panel
    ax_bottom.set_xlabel('Cumulative Emissions (MTCO2)', fontsize=12)
    ax_bottom.set_ylabel('Cumulative Utility ($bn/year)', fontsize=12)
    ax_bottom.grid(alpha=0.3)
    #ax_bottom.set_title('Utility vs Emissions', pad=15, fontsize=14)
    
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
    
    """
    # Create plots
    plot_policy_pair_effects(
        data_array=data_array,
        policy_pairs=policy_pairs,
        file_name=file_name,
        policy_info_dict=policy_info_dict,
        dpi=300
    )
    
    # Create stacked plots
    plot_policy_pair_effects_stacked(
        data_array=data_array,
        policy_pairs=policy_pairs,
        file_name=file_name,
        policy_info_dict=policy_info_dict,
        group_size=5,  # Number of policy pairs per figure
        dpi=300
    )
    plot_3d_policy_comparison(
        data_array=data_array,
        policy_pairs=policy_pairs,
        file_name=file_name,
        policy_info_dict=policy_info_dict,
        min_ev_uptake=0.948,
        max_ev_uptake=1.0,
        dpi=300
    )

    #generate_all_measure_plots(data_array, policy_pairs, file_name, policy_info_dict,
    #                            min_ev_uptake=0.9, max_ev_uptake=1, dpi=300)
    
    generate_all_measure_plots(data_array, policy_pairs, file_name, policy_info_dict,
                                min_ev_uptake=0.948, max_ev_uptake=1, dpi=300, opacity_state = False)
        
    """

    plot_emissions_tradeoffs(
        outcomes_BAU,
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