"""
Reproduces Figure 10 of the supplementary material: elasticity of EV uptake
and CO2 emissions to a 1% increment in electricity price or grid
decarbonisation, at two shock magnitudes each (50%/150% for price,
50%/90% reduction for grid intensity).

Thin wrapper around the existing
package/plotting_data/inputs_and_emissions_plot.py::plot_elasticity_comparison,
which already produces exactly this 2x2 layout. The only reason not to call
it directly is that the shared data generated for Figure 9
(fig09_10_bau_gen.py) also includes a third grid-intensity value (0.75, 25%
reduction) that Figure 9's columns need but Figure 10 does not show --
grid_intensities_to_plot restricts the elasticity lines to the two the paper
plots (0.1, 0.5), while still using the full dataset to compute the baseline.
"""
from package.plotting_data.inputs_and_emissions_plot import plot_elasticity_comparison


def plot_fig10(results_folder):
    return plot_elasticity_comparison(
        results_folder,
        grid_intensities_to_plot=[0.1, 0.5],
        elec_prices_to_plot=[0.5, 1.5],
    )


if __name__ == "__main__":
    plot_fig10("results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_XX_XX_XX__XX_XX_XXXX")
