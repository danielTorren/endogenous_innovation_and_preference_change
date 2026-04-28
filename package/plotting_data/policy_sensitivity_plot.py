import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from package.resources.utility import load_object
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from dateutil.relativedelta import relativedelta


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


def main_load_and_plot(results_folder):
    print(f"Loading data from: {results_folder}")

    data_array_ev = load_object(f"{results_folder}/Data", "data_cross_ev")
    data_array_bau = load_object(f"{results_folder}/Data", "data_cross_bau")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    phys_list = metadata["phys"]["property_list"]
    pol_list = metadata["policy"]["property_list"]
    phys_name = metadata["phys"]["property_varied"]
    pol_name = metadata["policy"]["property_varied"]

    print("Generating Cross-Variation Grid...")
    plot_cross_comparison_grid(data_array_ev, data_array_bau, phys_list, pol_list, phys_name, pol_name, results_folder)

    print("Generating Policy Heatmap...")
    plot_policy_efficiency_heatmap(data_array_ev, phys_list, pol_list, phys_name, pol_name, results_folder)

    print("Generating Seed Variance Boxplot...")
    plot_seed_distribution_box(data_array_ev, phys_list, pol_list, phys_name, pol_name, results_folder)

    print("Generating Heatmap Animation...")
    animate_ev_uptake_calendar(data_array_ev, phys_list, pol_list, phys_name, pol_name, results_folder)

    print("All plots saved.")
    plt.show()
    
if __name__ == "__main__":
    # Replace with your actual folder path
    path = "results/cross_nu_vs_Carbon_price_13_49_58__03_04_2026" 
    main_load_and_plot(path)
    #cross_a_chi_vs_Carbon_price_14_11_13__02_04_2026
    #ross_beta_multiplier_vs_Carbon_price_00_28_21__02_04_2026
    #cross_nu_vs_Carbon_price_13_49_58__03_04_2026