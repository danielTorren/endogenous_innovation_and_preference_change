import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object
import os

def load_ev_data(folder):
    base_params = load_object(os.path.join(folder, "Data"), "base_params")
    ev_prop = load_object(os.path.join(folder, "Data"), "data_array_ev_prop")
    vary_single = load_object(os.path.join(folder, "Data"), "vary_single")
    property_list = vary_single["property_list"]
    property_name = vary_single["property_varied"]
    return base_params, ev_prop, property_list, property_name

def plot_multi_ev_prop_grid(folders, real_data, dpi=300):
    num_vars = len(folders)
    fig, axes = plt.subplots(1, num_vars, figsize=(5*num_vars, 5), sharey=True)

    if num_vars == 1:
        axes = [axes]  # Make iterable

    for ax, folder in zip(axes, folders):
        base_params, ev_prop, property_list, property_name = load_ev_data(folder)

        num_deltas = ev_prop.shape[0]
        num_seeds = ev_prop.shape[1]
        burn_in = base_params["duration_burn_in"]
        time_series = np.arange(ev_prop.shape[2])

        for i in range(num_deltas):
            mean_trace = np.mean(ev_prop[i, :, burn_in:], axis=0)
            ax.plot(time_series[burn_in:], mean_trace, label=f'{property_list[i]:.2e}')

        ax.set_title(f'Varying {property_name}')
        ax.set_xlabel("Time Step")
        ax.grid()
        ax.legend()

    axes[0].set_ylabel("EV Uptake Proportion")

    plt.tight_layout()
    plt.savefig("results/multi_param_run/ev_prop_grid.png", dpi=dpi)
    plt.show()

def main():
    real_data = load_object("package/calibration_data", "calibration_data_output")["EV Prop"]

    # List your output folders here
    folders = [
        "results/multi_param_run/delta_16_40_01__28_04_2025",
        "results/multi_param_run/price_16_41_10__28_04_2025",
        "results/multi_param_run/a_chi_16_42_22__28_04_2025",
        "results/multi_param_run/kappa_16_43_33__28_04_2025"
    ]

    plot_multi_ev_prop_grid(folders, real_data)

if __name__ == "__main__":
    main()
