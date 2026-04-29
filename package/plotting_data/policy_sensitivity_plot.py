import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from package.resources.utility import load_object
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def plot_policy_efficiency_heatmap(data_array, phys_list, pol_list, phys_name, pol_name, fileName):
    """
    2. HEATMAP
    Shows the final state (average of last 10 steps) of EV adoption 
    across the parameter space.
    """
    # Average over seeds and the last 10 time steps for stability
    final_impact = np.mean(data_array[:, :, :, -10:], axis=(2, 3))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(final_impact, cmap="YlGnBu", aspect='auto')

    ax.set_xticks(np.arange(len(pol_list)))
    ax.set_yticks(np.arange(len(phys_list)))
    ax.set_xticklabels(pol_list)
    ax.set_yticklabels(phys_list)

    plt.colorbar(im, label="Final EV Proportion")
    ax.set_xlabel(pol_name)
    ax.set_ylabel(phys_name)
    ax.set_title("Policy Surface Analysis")

    # Annotate values
    for i in range(len(phys_list)):
        for j in range(len(pol_list)):
            ax.text(j, i, f"{final_impact[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(f"{fileName}/policy_surface_heatmap.png", dpi=300)


def plot_seed_distribution_box(data_array, phys_list, pol_list, phys_name, pol_name, fileName):
    """
    3. BOXPLOT
    Visualizes the variance/uncertainty introduced by seeds at the 
    final timestep for each combination.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = []
    labels = []
    plot_data = []

    for i, phys_val in enumerate(phys_list):
        for j, pol_val in enumerate(pol_list):
            # Final timestep values for all seeds
            final_values = data_array[i, j, :, -1]
            plot_data.append(final_values)
            labels.append(f"Ph:{phys_val}\nPo:{pol_val}")
    
    ax.boxplot(plot_data)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("Final EV Proportion")
    ax.set_title("Variance across Seeds per Configuration")
    
    plt.tight_layout()
    plt.savefig(f"{fileName}/seed_variance_boxplot.png", dpi=300)

def animate_ev_uptake_calendar(data_array, phys_list, pol_list, phys_name, pol_name, fileName):
    """
    Animates EV adoption starting from Jan 2010.
    Assumes data_array starts at Jan 2001.
    """
    # 1. Pre-process: Average over seeds
    time_series_data = np.mean(data_array, axis=2)
    total_steps = time_series_data.shape[-1]
    
    # 2. Setup Time Logic
    start_date = datetime(2001, 1, 1)
    target_start_date = datetime(2020, 1, 1)
    
    # Calculate how many months to skip to get to 2010
    skip_months = 180 + (target_start_date.year - start_date.year) * 12 + (target_start_date.month - start_date.month)

    if skip_months >= total_steps:
        raise ValueError(f"Data array only has {total_steps} steps. It doesn't reach 2010.")

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize with 2010 data
    initial_data = time_series_data[:, :, skip_months]
    im = ax.imshow(initial_data, cmap="YlGnBu", aspect='auto', vmin=0, vmax=1)

    # UI Formatting
    ax.set_xticks(np.arange(len(pol_list)))
    ax.set_yticks(np.arange(len(phys_list)))
    ax.set_xticklabels(pol_list)
    ax.set_yticklabels(phys_list)
    print("phys_name : ", phys_name )
    if phys_name == "beta_multiplier":
        ax.set_ylabel("Beta median multiplier")
    elif phys_name == "a_chi":
        ax.set_ylabel(r"\chi a parameter")
    else:
        ax.set_ylabel(phys_name)
    
    if pol_name == "carbon_price":
        ax.set_xlabel("Carbon price")
    else:
        ax.set_xlabel(pol_name)

    plt.colorbar(im, label="EV Proportion")
    
    title = ax.set_title("")

    def update(frame):
        # Calculate current index relative to start of data
        current_idx = skip_months + frame
        
        # Stop if we exceed array bounds
        if current_idx >= total_steps:
            return [im, title]
            
        # Update heatmap
        im.set_array(time_series_data[:, :, current_idx])
        
        # Update Date Title
        current_date = start_date + relativedelta(months=(current_idx -180))
        date_str = current_date.strftime("%B %Y")
        title.set_text(f"{date_str}")
        
        return [im, title]

    # Number of frames is total steps minus the 108 months we skipped
    num_frames = total_steps - skip_months
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=150, blit=True)

    # Save
    save_path = f"{fileName}/ev_adoption_2020_onwards.gif"
    ani.save(save_path, writer='pillow', fps=8)
    
    plt.close()
    print(f"Animation saved: {save_path}")

def plot_cross_comparison_grid(data_array, data_array_bau, phys_list, pol_list, phys_name, pol_name, fileName):
    num_phys = len(phys_list)
    num_pol = len(pol_list)
    num_seeds = data_array.shape[2]
    time_steps = np.arange(data_array.shape[3])

    fig, axes = plt.subplots(1, num_phys, figsize=(6 * num_phys, 5), sharey=True)
    if num_phys == 1:
        axes = [axes]

    # +1 color for BAU
    colors = plt.cm.turbo(np.linspace(0, 1, num_pol))

    for i, phys_val in enumerate(phys_list):
        ax = axes[i]

        # --- Plot BAU first in black dashed ---
        bau_data = data_array_bau[i, :, :]  # (Seeds, Time)
        bau_mean = np.mean(bau_data, axis=0)
        bau_ci = stats.sem(bau_data, axis=0) * stats.t.ppf(0.975, num_seeds - 1)
        ax.plot(time_steps, bau_mean, label="BAU", color='black', linestyle='--', linewidth=1.5)
        ax.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.15)

        # --- Plot policy lines ---
        for j, pol_val in enumerate(pol_list):
            data = data_array[i, j, :, :]
            mean_vals = np.mean(data, axis=0)
            ci = stats.sem(data, axis=0) * stats.t.ppf(0.975, num_seeds - 1)
            line, = ax.plot(time_steps, mean_vals, label=f"{pol_name}: {pol_val}", color=colors[j])
            ax.fill_between(time_steps, mean_vals - ci, mean_vals + ci, color=line.get_color(), alpha=0.2)

        ax.set_title(f"{phys_name}: {phys_val}")
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.set_ylabel("EV Proportion")

    fig.supxlabel("Time Step")
    plt.legend(title=pol_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{fileName}/cross_grid_comparison.png", dpi=300)

def plot_policy_efficiency_heatmap_BAU(data_array, data_array_bau, phys_list, pol_list, phys_name, pol_name, fileName):
    """
    2. HEATMAP with BAU as first column
    Shows the final state (average of last 10 steps) of EV adoption 
    across the parameter space, with BAU as the baseline column.
    """
    # Calculate final impact for policy runs (average over seeds and last 10 steps)
    policy_impact = np.mean(data_array[:, :, :, -10:], axis=(2, 3))  # Shape: (len(phys_list), len(pol_list))
    
    # Calculate final impact for BAU runs (average over seeds and last 10 steps)
    # data_array_bau shape: (len(phys_list), seed_reps, time_steps)
    bau_impact = np.mean(data_array_bau[:, :, -10:], axis=(1, 2))  # Shape: (len(phys_list),)
    
    # Combine: BAU as first column, then all policy columns
    # Result shape: (len(phys_list), len(pol_list) + 1)
    combined_impact = np.column_stack([bau_impact, policy_impact])
    
    # Create combined lists for labels
    combined_pol_list = ["BAU"] + pol_list
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(combined_impact, cmap="YlGnBu", aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(combined_pol_list)))
    ax.set_yticks(np.arange(len(phys_list)))
    ax.set_xticklabels(combined_pol_list, rotation=45, ha='right')
    ax.set_yticklabels(phys_list)
    
    # Colorbar
    cbar = plt.colorbar(im, label="EV Uptake Proportion 2035")
    
    # Labels and title
    if phys_name == "beta_multiplier":
        ax.set_ylabel("$Beta$ median multiplier")
    elif phys_name == "a_chi":
        ax.set_ylabel(r"$a_{\chi}$ parameter")
    else:
        ax.set_ylabel(phys_name)
    
    ax.set_xlabel(f"{pol_name}")
    
    # Annotate values
    for i in range(len(phys_list)):
        for j in range(len(combined_pol_list)):
            text_color = "white" if combined_impact[i, j] > 0.5 else "black"
            ax.text(j, i, f"{combined_impact[i, j]:.3f}", ha="center", va="center", 
                   color=text_color, fontsize=9)
    
    # Add a separator line between BAU and policy columns
    ax.axvline(x=0.5, color='red', linestyle='-', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{fileName}/policy_surface_heatmap_with_BAU.png", dpi=300)

def load_and_plot_combined(main_results_folder, bau_results_folder=None, output_folder=None):
    """
    Load and plot results from either combined or separate runs.
    
    Parameters:
    -----------
    main_results_folder : str
        Path to main results folder (contains data_cross_ev and metadata)
    bau_results_folder : str or None
        Path to BAU results folder (contains data_cross_bau). If None, assumes BAU data is in main folder.
    output_folder : str or None
        Where to save plots. If None, saves to main_results_folder.
    """
    print(f"Loading main data from: {main_results_folder}")
    
    # Load main data
    data_array_ev = load_object(f"{main_results_folder}/Data", "data_cross_ev")
    metadata = load_object(f"{main_results_folder}/Data", "vary_metadata")
    
    phys_list = metadata["phys"]["property_list"]
    pol_list = metadata["policy"]["property_list"]
    phys_name = metadata["phys"]["property_varied"]
    pol_name = metadata["policy"]["property_varied"]
    
    # Handle BAU data loading
    if bau_results_folder is not None:
        print(f"Loading BAU data from: {bau_results_folder}")
        data_array_bau = load_object(f"{bau_results_folder}/Data", "data_cross_bau")
    else:
        # Try to load from main folder (legacy mode)
        try:
            print("Attempting to load BAU data from main folder (legacy mode)...")
            data_array_bau = load_object(f"{main_results_folder}/Data", "data_cross_bau")
            print("BAU data found in main folder")
        except FileNotFoundError:
            print("Error: No BAU data found. Please provide bau_results_folder or run combined simulation.")
            raise
    
    # Set output folder
    if output_folder is None:
        output_folder = main_results_folder
    
    # Verify compatibility
    data_array_bau_shape = data_array_bau.shape
    if data_array_bau_shape[0] != len(phys_list):
        raise ValueError(f"BAU data has {data_array_bau_shape[0]} physical values, but main data has {len(phys_list)}. They must match.")
    
    print(f"\nData shapes:")
    print(f"  Main data: {data_array_ev.shape}")
    print(f"  BAU data: {data_array_bau.shape}")
    print(f"  Physical values: {phys_list}")
    print(f"  Policy values: {pol_list}")
    print(f"  Physical parameter: {phys_name}")
    print(f"  Policy parameter: {pol_name}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("  - Generating Policy Heatmap with BAU...")
    plot_policy_efficiency_heatmap_BAU(data_array_ev, data_array_bau, phys_list, pol_list, 
                                      phys_name, pol_name, output_folder)
    
    print("  - Generating Cross-Variation Grid...")
    plot_cross_comparison_grid(data_array_ev, data_array_bau, phys_list, pol_list, 
                              phys_name, pol_name, output_folder)
    
    print("  - Generating Policy Heatmap (standard)...")
    plot_policy_efficiency_heatmap(data_array_ev, phys_list, pol_list, 
                                  phys_name, pol_name, output_folder)
    
    print("  - Generating Seed Variance Boxplot...")
    plot_seed_distribution_box(data_array_ev, phys_list, pol_list, 
                              phys_name, pol_name, output_folder)
    
    print("  - Generating Heatmap Animation...")
    animate_ev_uptake_calendar(data_array_ev, phys_list, pol_list, 
                              phys_name, pol_name, output_folder)
    
    print(f"\nAll plots saved to: {output_folder}")
    plt.show()

def main_load_and_plot(results_folder):
    """
    Legacy function for backward compatibility.
    Assumes both main and BAU data are in the same folder.
    """
    load_and_plot_combined(results_folder, bau_results_folder=None)

if __name__ == "__main__":
    # Example usage 1: Legacy mode (BAU data in same folder as main results)
    # path = "results/cross_a_chi_vs_Adoption_subsidy_00_26_28__29_04_2026" 
    # main_load_and_plot(path)
    
    # Example usage 2: Separate mode (main results and BAU results from different runs)
    main_path = "results/cross_beta_multiplier_vs_Carbon_price_00_28_21__02_04_2026"
    bau_path = "results/cross_beta_multiplier_vs_Carbon_price_BAU_16_32_02__29_04_2026"  # Or wherever your BAU results are saved

    #main_path = "results/cross_a_chi_vs_Carbon_price_14_11_13__02_04_2026"
    #bau_path = "results/cross_a_chi_vs_Carbon_price_BAU_15_37_46__29_04_2026"  # Or wherever your BAU results are saved

    load_and_plot_combined(main_path, bau_path)
    
    # You can also specify a custom output folder
    # load_and_plot_combined(main_path, bau_path, output_folder="my_combined_plots")