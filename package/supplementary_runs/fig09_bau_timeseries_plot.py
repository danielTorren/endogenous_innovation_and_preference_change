"""
New plotting logic for Figure 9 of the supplementary material: BAU (no
policy) EV uptake proportion (top row) and monthly emissions flow (bottom
row) over the extended policy period (2024-2050), for three grid-
decarbonisation columns (90% / 50% / 25% reduction in electricity emissions
intensity) crossed with three electricity-price lines (50% decrease / no
change / 50% increase).

Reuses the same data source and date-axis convention as
package/plotting_data/inputs_and_emissions_plot.py::plot_elasticity_comparison
(package/generating_data/inputs_and_emissions_gen.py::run_physical_duo, with
start_step=456 marking the Jan-2024 start of the extended policy period) --
this module only adds the raw time-series layout the paper's Figure 9 uses,
as opposed to that file's elasticity transform (Figure 10).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
from package.resources.utility import load_object

DECARB_LABELS = {
    0.1: "Electricity Emissions Intensity:\n90% Reduction",
    0.5: "Electricity Emissions Intensity:\n50% Reduction",
    0.75: "Electricity Emissions Intensity:\n25% Reduction",
    1.0: "Electricity Emissions Intensity:\nNo Change",
}
PRICE_LABELS = {
    0.5: "Electricity Price: 50% Reduction",
    1.0: "Electricity Price: 0% Change",
    1.5: "Electricity Price: 150% Increase",
}
PRICE_COLORS = {0.5: "indigo", 1.0: "teal", 1.5: "yellowgreen"}


def plot_fig9_bau_timeseries(results_folder, decarb_columns=(0.1, 0.5, 0.75), dpi=300):
    data_ev = load_object(f"{results_folder}/Data", "data_phys_duo_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_duo_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    grid_intensities = metadata[0]["property_list"]
    elec_prices = metadata[1]["property_list"]
    print(f"Grid intensities in data: {grid_intensities}")
    print(f"Electricity prices in data: {elec_prices}")

    start_step = 456
    time_length = data_ev.shape[3]
    time_indices = np.arange(start_step, time_length)
    dates = [datetime(2024, 1, 1) + timedelta(days=30.44 * (i - start_step)) for i in time_indices]

    col_indices = [grid_intensities.index(v) for v in decarb_columns]

    fig, axes = plt.subplots(2, len(decarb_columns), figsize=(6 * len(decarb_columns), 8), sharex=True, sharey="row")

    for col, grid_idx in enumerate(col_indices):
        grid_val = grid_intensities[grid_idx]
        ax_ev = axes[0, col]
        ax_em = axes[1, col]

        for price in elec_prices:
            price_idx = elec_prices.index(price)
            ev_data = data_ev[grid_idx, price_idx, :, start_step:]
            em_data = data_em[grid_idx, price_idx, :, start_step:]

            mean_ev = np.mean(ev_data, axis=0)
            ci_ev = stats.sem(ev_data, axis=0) * 1.96
            mean_em = np.mean(em_data, axis=0)
            ci_em = stats.sem(em_data, axis=0) * 1.96

            color = PRICE_COLORS.get(price, None)
            label = PRICE_LABELS.get(price, f"Elec price: {price}x")

            ax_ev.plot(dates, mean_ev, color=color, label=label, lw=2)
            ax_ev.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=color, alpha=0.2)

            ax_em.plot(dates, mean_em, color=color, label=label, lw=2)
            ax_em.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=color, alpha=0.2)

        ax_ev.set_title(DECARB_LABELS.get(grid_val, f"Grid intensity: {grid_val}x"), fontweight="bold")
        ax_ev.grid(True, linestyle="--", alpha=0.6)
        ax_em.grid(True, linestyle="--", alpha=0.6)

    axes[0, 0].set_ylabel("EV Uptake Proportion")
    axes[1, 0].set_ylabel(r"Monthly Emissions, kgCO$_2$")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = os.path.join(results_folder, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/fig9_bau_timeseries.png", dpi=dpi, bbox_inches="tight")
    print(f"Saved to {save_path}/fig9_bau_timeseries.png")
    return fig


if __name__ == "__main__":
    plot_fig9_bau_timeseries("results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_XX_XX_XX__XX_XX_XXXX")
