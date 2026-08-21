"""
Reproduces Figures 7 and 8 of the supplementary material: Sobol global
sensitivity analysis (first-order and total-order indices respectively) of
six model outputs (cumulative emissions, EV adoption proportion, firm
profit, market concentration HHI, utility, mean car age) with respect to the
same ten parameters Figure 6 varies locally: K_ICE, K_EV, delta, lambda,
a_chi, b_chi, kappa, mu, r, alpha.

Both generation and plotting already existed unchanged elsewhere in the
repo -- this is a thin wrapper that pins the settings sized for this
reproduction run:
  - package/generating_data/sensitivity_analysis_calibration_gen.py::main
    (SALib.sample.saltelli over package/constants/variable_parameters_dict_SA.json,
    against package/constants/base_params_SA.json -- same burn-in/calibration
    duration and seed_repetitions=64 as every other figure here)
  - package/plotting_data/sensitivity_analysis_calibration_plot.py::main
    (SALib.analyze.sobol -> the first/total-order error-bar panels)

N_samples=256 with calc_second_order=False: runs = N_samples x (D+2) x
seed_repetitions = 256 x 12 x 64 = 196,608 runs, ~8.5 h wall-clock on 128
cores at ~20 s/run. calc_second_order=False is sized down from the existing
script's own default (True, which would need 2D+2=22 rather than D+2=12
runs per sample) since the paper's Figures 7/8 only show first- and
total-order indices, not the second-order interaction terms.
"""
from package.generating_data.sensitivity_analysis_calibration_gen import main as generate_sobol
from package.plotting_data.sensitivity_analysis_calibration_plot import main as plot_sobol

N_SAMPLES = 256
BASE_PARAMS_LOAD = "package/constants/base_params_SA.json"
VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json"
CALC_SECOND_ORDER = False


def run_fig7_8(
    n_samples=N_SAMPLES,
    base_params_load=BASE_PARAMS_LOAD,
    variable_params_load=VARIABLE_PARAMS_LOAD,
    calc_second_order=CALC_SECOND_ORDER,
):
    file_name = generate_sobol(
        N_samples=n_samples,
        BASE_PARAMS_LOAD=base_params_load,
        VARIABLE_PARAMS_LOAD=variable_params_load,
        calc_second_order=calc_second_order,
    )
    plot_sobol(fileName=file_name)
    print(f"Figures 7/8 saved under {file_name}/Prints/: "
          f"10_{n_samples}_First_multi_output_sensitivity_plot_2row.png is "
          f"Figure 7, 10_{n_samples}_Total_multi_output_sensitivity_plot_2row.png "
          f"is Figure 8")
    return file_name


if __name__ == "__main__":
    run_fig7_8()
