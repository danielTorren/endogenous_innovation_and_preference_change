"""
Reproduces Figure 6 of the supplementary material: a 10-panel (a-j) grid of
EV uptake proportion under local variation of ten key parameters -- alpha,
r (discount rate), mu (used car markup), kappa, b_chi, a_chi,
lambda (research intensity), delta (fuel efficiency depreciation), K_EV and
K_ICE (landscape complexity).

Generation reuses package/generating_data/vary_single_param_gen.py once per
parameter -- this script's only job is to run it 10 times against
base_params_calibration.json (duration_future=0, matching the paper's
Time Step axis which stops at the end of calibration, ~456) and then combine
the ten runs with fig06_local_sensitivity_plot.py.
"""
from package.generating_data.vary_single_param_gen import main as generate_vary_single
from package.supplementary_runs.fig06_local_sensitivity_plot import plot_fig6_combined

BASE_PARAMS_LOAD = "package/constants/base_params_calibration.json"

# (panel letter, vary-single config) -- matches Figure 6's panel order a-j
PANELS = [
    ("a", "package/constants/vary_single_alpha.json"),
    ("b", "package/constants/vary_single_r.json"),
    ("c", "package/constants/vary_single_mu.json"),
    ("d", "package/constants/vary_single_kappa.json"),
    ("e", "package/constants/vary_single_b_innov.json"),
    ("f", "package/constants/vary_single_a_innov.json"),
    ("g", "package/constants/vary_single_lambda.json"),
    ("h", "package/constants/vary_single_delta.json"),
    ("i", "package/constants/vary_single_landscape_K_EV.json"),
    ("j", "package/constants/vary_single_landscape_K_ICE.json"),
]


def run_fig6(base_params_load=BASE_PARAMS_LOAD, panels=PANELS):
    panel_folders = []
    for letter, vary_load in panels:
        print(f"\n=== Panel {letter}: {vary_load} ===")
        folder = generate_vary_single(BASE_PARAMS_LOAD=base_params_load, VARY_LOAD=vary_load)
        panel_folders.append((letter, folder))

    print("\nPanel folders:")
    for letter, folder in panel_folders:
        print(f"  {letter}: {folder}")

    output_folder = "results/" + panel_folders[0][1].split("/")[-1] + "_fig6_combined"
    plot_fig6_combined(panel_folders, output_folder=output_folder)
    return panel_folders, output_folder


if __name__ == "__main__":
    run_fig6()
