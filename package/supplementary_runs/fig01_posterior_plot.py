"""
Reproduces Figure 1 of the supplementary material: posterior density (1D
marginals + 2D joint density) for the two SBI-calibrated innovativeness
parameters a_chi, b_chi.

Generation is NOT duplicated here -- it already exists at
package/calibration/sbi_single_seed_gen.py, submitted via
package/calibration/submit_sbi_single_seed.slurm (NPE over
[a_chi, b_chi, delta, kappa] against base_params_calibration.json). Re-run that
if you need a fresh posterior against the current calibration parameters;
otherwise point RESULTS_FOLDER below at any existing results/sbi_single_seed_*
folder that already has samples.pkl/var_dict.pkl.

samples.pkl columns are in the order the gen script's parameters_list was
built in -- currently [a_chi, b_chi, delta, kappa] -- so slicing the first two
columns gives exactly the two parameters Figure 1 needs.
"""
from sbi.analysis import pairplot
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show


def plot_fig1_posterior(results_folder, n_params=2, dpi=300):
    samples = load_object(f"{results_folder}/Data", "samples")
    var_dict = load_object(f"{results_folder}/Data", "var_dict")

    param_names = [p["name"] for p in var_dict][:n_params]
    param_bounds = [p["bounds"] for p in var_dict][:n_params]
    samples_subset = samples[:, :n_params]

    print(f"Plotting posterior for {param_names} from {len(samples_subset)} samples")

    fig, ax = pairplot(
        samples_subset,
        limits=param_bounds,
        figsize=(10, 10),
        points_colors="r",
        labels=param_names,
    )

    save_and_show(fig, results_folder, "fig1_posterior_a_chi_b_chi", dpi=dpi)
    print(f"Saved to {results_folder}/Plots/fig1_posterior_a_chi_b_chi.png")
    return fig


if __name__ == "__main__":
    # Most recent run against the CURRENT base_params_calibration.json
    # (a_chi=1.351, b_chi=3.117) as of when this script was written -- swap
    # in a newer results/sbi_single_seed_* folder if you re-ran generation.
    plot_fig1_posterior(
        results_folder="results/sbi_single_seed_15_38_23__11_08_2026",
    )
