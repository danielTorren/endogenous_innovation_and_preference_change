import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from package.resources.utility import load_object, createFolder
from matplotlib.patches import Patch
from matplotlib.path import Path


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path
from package.resources.utility import createFolder

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

def calculate_opacity(ev_uptake, min_ev_uptake=0.9, max_ev_uptake=0.95):
    """
    Calculate opacity that reaches 1 at max_ev_uptake (default 0.95)
    and stays at 1 for all higher values.
    """
    if ev_uptake >= max_ev_uptake:
        return 1.0
    return np.clip((ev_uptake - min_ev_uptake) / (max_ev_uptake - min_ev_uptake), 0, 1)


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

def main(file_name):
    # Load data
    data_array = load_object(file_name + "/Data", "data_array")
    policy_pairs = load_object(file_name + "/Data", "policy_pairs")
    policy_info_dict = load_object(file_name + "/Data", "policy_info_dict")
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
    """

    #generate_all_measure_plots(data_array, policy_pairs, file_name, policy_info_dict,
    #                            min_ev_uptake=0.9, max_ev_uptake=1, dpi=300)
    
    generate_all_measure_plots(data_array, policy_pairs, file_name, policy_info_dict,
                                min_ev_uptake=0.948, max_ev_uptake=1, dpi=300, opacity_state = False)
    
    plt.show()

if __name__ == "__main__":
    main(file_name="results/vary_two_policies_gen_19_45_01__27_03_2025")