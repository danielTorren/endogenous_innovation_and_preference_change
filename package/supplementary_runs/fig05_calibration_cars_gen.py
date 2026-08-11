"""
Reproduces Figure 5 of the supplementary material: simulated ICE/EV price vs.
driving range at the end of the calibration period (64 seeds), against
real-world reference vehicles.

This is a thin wrapper around existing infrastructure -- no new simulation or
plotting logic. Generation is package/generating_data/calibration_gen.py
(saves cars_on_sale for every seed under base_params_calibration.json's
seed_repetitions=64) and plotting is
package/plotting_data/calibration_plot.py::main, which the generator already
calls automatically and which produces, among other panels,
Plots/multi_seed_2d_scatter.png -- exactly Figure 5.
"""
from package.generating_data.calibration_gen import main as generate_calibration
from package.plotting_data.calibration_plot import main as plot_calibration


def run_fig5(base_params_load="package/constants/base_params_calibration.json"):
    file_name = generate_calibration(BASE_PARAMS_LOAD=base_params_load)
    plot_calibration(fileName=file_name)
    print(f"Figure 5 saved to {file_name}/Plots/multi_seed_2d_scatter.png")
    return file_name


if __name__ == "__main__":
    run_fig5()
