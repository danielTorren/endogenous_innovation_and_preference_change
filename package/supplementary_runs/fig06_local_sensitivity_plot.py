"""
New combining logic for Figure 6 of the supplementary material: a 10-panel
(a-j) grid of EV uptake proportion over time, each panel showing the local
variation of one parameter, with a dotted "California Data 2010-23" reference
line in every panel.

Each panel's mean+95% CI curves reuse the exact logic of
package/plotting_data/vary_single_param_plot.py::plot_ev_prop_combined, which
already draws one such panel for a single parameter -- this module only adds
the code to lay ten of them out on one figure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from package.resources.utility import load_object


def _plot_panel(ax, base_params, data_array, property_list, name_property, real_data):
    num_values = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]
    time_series = np.arange(time_steps)
    colors = plt.cm.viridis(np.linspace(0, 1, num_values))

    burn_in_step = base_params["duration_burn_in"]
    init_index = burn_in_step + 120
    time_steps_real = np.arange(init_index, init_index + len(real_data) * 12, 12)
    ax.plot(time_steps_real, real_data, label="California Data 2010-23",
            color="orange", linestyle="dotted")

    for i, (value, color) in enumerate(zip(property_list, colors)):
        data_after_burn_in = data_array[i, :, burn_in_step:]
        mean_data = np.mean(data_after_burn_in, axis=0)
        sem_data = stats.sem(data_after_burn_in, axis=0)
        ci_range = sem_data * stats.t.ppf(0.975, num_seeds - 1)

        label = f"{name_property} = {value:.1e}" if isinstance(value, float) else f"{name_property} = {value}"
        ax.plot(time_series[burn_in_step:], mean_data, color=color, label=label, linewidth=1.8)
        ax.fill_between(time_series[burn_in_step:], mean_data - ci_range, mean_data + ci_range,
                         color=color, alpha=0.25)

    ax.grid(alpha=0.4)
    ax.legend(fontsize=7, loc="upper left")


def plot_fig6_combined(panel_folders, output_folder=None, dpi=300):
    """
    panel_folders: ordered list of (letter, results_folder) pairs, one per
    parameter -- e.g. [("a", "results/single_param_vary_..."), ("b", ...), ...].
    Laid out left-to-right, top-to-bottom in a 2x5 grid (a-e top row, f-j
    bottom row), matching the paper's Figure 6.
    """
    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    real_data = calibration_data_output["EV Prop"]

    fig, axes = plt.subplots(2, 5, figsize=(25, 9), sharex=True)
    axes_flat = axes.flatten()

    for ax, (letter, folder) in zip(axes_flat, panel_folders):
        base_params = load_object(f"{folder}/Data", "base_params")
        data_array_ev_prop = load_object(f"{folder}/Data", "data_array_ev_prop")
        vary_single = load_object(f"{folder}/Data", "vary_single")

        _plot_panel(ax, base_params, data_array_ev_prop, vary_single["property_list"],
                    vary_single["property_varied"], real_data)
        ax.set_title(f"{letter}) {vary_single['property_varied']}", loc="left", fontweight="bold")

    fig.supxlabel("Time Step")
    fig.supylabel("EV Uptake Proportion")
    plt.tight_layout()

    if output_folder is None:
        output_folder = panel_folders[0][1]
    save_path = os.path.join(output_folder, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/fig6_local_sensitivity_combined.png", dpi=dpi)
    print(f"Saved to {save_path}/fig6_local_sensitivity_combined.png")
    return fig


if __name__ == "__main__":
    # Fill in with the ten folder names printed by
    # fig06_local_sensitivity_gen.py's run, in panel order a-j.
    panel_folders = [
        ("a", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("b", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("c", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("d", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("e", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("f", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("g", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("h", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("i", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
        ("j", "results/single_param_vary_XX_XX_XX__XX_XX_XXXX"),
    ]
    plot_fig6_combined(panel_folders, output_folder="results/fig6_local_sensitivity_combined")
