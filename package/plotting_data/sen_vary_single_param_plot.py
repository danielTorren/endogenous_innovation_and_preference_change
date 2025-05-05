from turtle import pos
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from package.resources.utility import load_object

def load_ev_data(folder):
    base_params = load_object(os.path.join(folder, "Data"), "base_params")
    ev_prop = load_object(os.path.join(folder, "Data"), "data_array_ev_prop")
    vary_single = load_object(os.path.join(folder, "Data"), "vary_single")
    property_list = vary_single["property_list"]
    property_name = vary_single["property_varied"]
    return base_params, ev_prop, property_list, property_name

def plot_multi_ev_prop_grid(folders, real_data, base_params, dpi=300):
    num_vars = len(folders)
    n_rows = 3
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=True)

    axes_flat = axes.flatten()

    for idx, (ax, folder) in enumerate(zip(axes_flat, folders)):

        # Plot real-world data (assume starts after burn-in + offset)
        burn_in_step = base_params["duration_burn_in"]
        init_index = burn_in_step + 120
        time_steps_real = np.arange(init_index, init_index + len(real_data) * 12, 12)
        ax.plot(time_steps_real, real_data, label="California Data 2010-23", color='orange', linestyle="dotted")

        # Load simulation data
        base_params_folder, ev_prop, property_list, property_name = load_ev_data(folder)

        num_deltas = ev_prop.shape[0]
        num_seeds = ev_prop.shape[1]
        burn_in = base_params_folder["duration_burn_in"]
        time_series = np.arange(ev_prop.shape[2])

        colors = plt.cm.viridis(np.linspace(0, 1, num_deltas))  # Color map for different deltas

        for i in range(num_deltas):
            data_after_burn_in = ev_prop[i, :, burn_in:]

            # Mean and confidence interval
            mean_data = np.mean(data_after_burn_in, axis=0)
            sem_data = stats.sem(data_after_burn_in, axis=0)
            ci_range = sem_data * stats.t.ppf(0.975, num_seeds - 1)  # 95% CI

            # Plot mean line
            ax.plot(time_series[burn_in:], mean_data, label=f'{property_name} = {property_list[i]:.1e}', color=colors[i])

            # Plot confidence interval
            ax.fill_between(time_series[burn_in:], mean_data - ci_range, mean_data + ci_range,
                            color=colors[i], alpha=0.3)

        # Add panel letter (a, b, c, etc.)
        ax.text(
            0.1, 0.3,  # (x,y) position in axes fraction coordinates
            f"{chr(97 + idx)}",  # 'a)', 'b)', 'c)', ...
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="center"
        )

        ax.grid()
        ax.legend(fontsize=8, loc =  "upper left")

    # Set labels only on appropriate plots
    for i in range(n_cols):
        axes[2, i].set_xlabel("Time Step")
    for i in range(n_rows):
        axes[i, 0].set_ylabel("EV Uptake Proportion")

    plt.tight_layout()
    for folder in enumerate(zip(axes_flat, folders)):
        plt.savefig("results/ev_prop_grid_combined.png", dpi=dpi)
    plt.show()

def main():
    real_data = load_object("package/calibration_data", "calibration_data_output")["EV Prop"]
    base_params = load_object("results/sen_vary_a_chi_14_01_26__28_04_2025/Data", "base_params")

    folders = [
        "results/sen_vary_a_chi_14_01_26__28_04_2025",
        "results/sen_vary_kappa_14_07_19__28_04_2025",
        "results/sen_vary_K_14_13_37__28_04_2025",
        "results/sen_vary_lambda_14_19_50__28_04_2025",
        "results/sen_vary_delta_14_26_21__28_04_2025",
        "results/sen_vary_mu_14_32_36__28_04_2025"
    ]

    plot_multi_ev_prop_grid(folders, real_data, base_params)

if __name__ == "__main__":
    main()
