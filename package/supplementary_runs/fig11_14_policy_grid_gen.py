"""
Reproduces Figures 11-14 of the supplementary material: 2035 EV uptake
proportion heatmaps over a physical parameter (rows) x policy intensity
(columns) grid, with a BAU column, for the two most sensitive parameters
identified in the local/global sensitivity analysis (beta median multiplier,
a_chi) crossed with the two policies that can reach the 2035 target (carbon
price, new car rebate):
  Figure 11: beta_multiplier x carbon_tax
  Figure 12: beta_multiplier x new_car_rebate
  Figure 13: a_chi        x carbon_tax
  Figure 14: a_chi        x new_car_rebate

Generation and plotting reuse
package/generating_data/policy_sensitivity_gen.py::run_cross_variation and
package/plotting_data/policy_sensitivity_plot.py::load_and_plot_combined
unchanged -- both already produce this exact BAU-column heatmap. This module
just runs the four (physical, policy) pairs against the JSON configs that
already match the paper's parameter ranges byte-for-byte:
  vary_policy_beta_multiplier.json: [0.25 .. 2] step varies, 8 values
  vary_policy_a_chi.json:           [0.5 .. 3], 8 values
  vary_policy_carbon_tax.json:      [0.05 .. 0.8], 6 values
  vary_policy_new_car_rebate.json:  [10000 .. 35000], 6 values

Each figure is an independent ~3,500-run job (8 or 6 phys values x 6 or 8 pol
values x 64 seeds, plus a BAU sweep) -- run the four via
submit_fig11_beta_carbon_gen.slurm .. submit_fig14_achi_rebate_gen.slurm as
four separate SLURM jobs so they run in parallel on the cluster rather than
one another's queue.
"""
import sys
from package.generating_data.policy_sensitivity_gen import (
    run_bau_only,
    run_cross_variation,
)
from package.plotting_data.policy_sensitivity_plot import load_and_plot_combined

BASE_PARAMS_LOAD = "package/constants/base_params_vary_policy_joint.json"

FIGURES = {
    "11": {
        "var_physical": "package/constants/vary_policy_beta_multiplier.json",
        "var_policy": "package/constants/vary_policy_carbon_tax.json",
    },
    "12": {
        "var_physical": "package/constants/vary_policy_beta_multiplier.json",
        "var_policy": "package/constants/vary_policy_new_car_rebate.json",
    },
    "13": {
        "var_physical": "package/constants/vary_policy_a_chi.json",
        "var_policy": "package/constants/vary_policy_carbon_tax.json",
    },
    "14": {
        "var_physical": "package/constants/vary_policy_a_chi.json",
        "var_policy": "package/constants/vary_policy_new_car_rebate.json",
    },
}


def run_policy_grid_figure(fig_number, base_params_load=BASE_PARAMS_LOAD):
    cfg = FIGURES[str(fig_number)]
    folder_name = run_cross_variation(
        BASE_PARAMS_LOAD=base_params_load,
        VAR_PHYSICAL_LOAD=cfg["var_physical"],
        VAR_POLICY_LOAD=cfg["var_policy"],
    )
    load_and_plot_combined(folder_name)
    return folder_name


def run_bau_for(fig_number, base_params_load=BASE_PARAMS_LOAD):
    """
    Run ONLY the all-policies-off BAU sweep for a figure's physical parameter,
    into its own results/cross_<phys>_vs_<pol>_BAU_<timestamp> folder.

    BAU depends only on the physical parameter, not the policy one, so the same
    sweep is valid for both figures sharing a physical parameter: one call with
    fig_number 13 (or 14) serves Figures 13 AND 14, and one with 11 (or 12)
    serves Figures 11 AND 12. That is why build_figures.RUNS keys this as
    "grid_bau_achi"/"grid_bau_beta" rather than per-figure -- and why running
    the full run_cross_variation for both members of a pair repeats these 512
    runs needlessly.

    Use this to finish a run whose policy phase completed and was saved but
    whose BAU phase died (e.g. the OOM kill on the 21/08/2026 jobs): the
    cross_* folder already on disk keeps its data_cross_ev, and
    load_and_plot_combined takes the BAU array from this separate folder via
    its bau_results_folder argument.
    """
    cfg = FIGURES[str(fig_number)]
    folder_name = run_bau_only(
        BASE_PARAMS_LOAD=base_params_load,
        VAR_PHYSICAL_LOAD=cfg["var_physical"],
        VAR_POLICY_LOAD=cfg["var_policy"],
    )
    print(f"\nBAU-only folder: {folder_name}")
    print("Point build_figures.RUNS['grid_bau_achi'] (Figures 13/14) or")
    print("['grid_bau_beta'] (Figures 11/12) at it, whichever matches.")
    return folder_name


def plot_only(fig_number, policy_folder, bau_folder=None):
    """
    Re-plot a figure from folders already on disk -- no simulation. bau_folder
    is where data_cross_bau lives; None means policy_folder holds it itself.
    """
    return load_and_plot_combined(policy_folder, bau_results_folder=bau_folder)


if __name__ == "__main__":
    # Full run (policy grid + its own BAU sweep + plot):
    #   python -m package.supplementary_runs.fig11_14_policy_grid_gen 13
    # BAU sweep only, for a figure whose policy phase already succeeded:
    #   python -m package.supplementary_runs.fig11_14_policy_grid_gen 13 bau
    # Plot only, from folders on disk:
    #   python -m package.supplementary_runs.fig11_14_policy_grid_gen 13 plot \
    #       results/cross_a_chi_vs_Carbon_price_<ts> results/cross_..._BAU_<ts>
    fig_number = sys.argv[1] if len(sys.argv) > 1 else "11"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"

    if mode == "bau":
        run_bau_for(fig_number)
    elif mode == "plot":
        if len(sys.argv) < 4:
            raise SystemExit(
                "plot mode needs the policy folder: "
                "... <fig> plot <policy_folder> [bau_folder]"
            )
        plot_only(fig_number, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == "full":
        run_policy_grid_figure(fig_number)
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected full, bau or plot")
