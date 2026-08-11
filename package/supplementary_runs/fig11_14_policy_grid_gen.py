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
from package.generating_data.policy_sensitivity_gen import run_cross_variation
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


if __name__ == "__main__":
    # e.g. `python -m package.supplementary_runs.fig11_14_policy_grid_gen 11`
    fig_number = sys.argv[1] if len(sys.argv) > 1 else "11"
    run_policy_grid_figure(fig_number)
