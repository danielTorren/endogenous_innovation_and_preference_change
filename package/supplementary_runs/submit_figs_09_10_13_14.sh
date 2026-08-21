#!/bin/bash
# Submit only the jobs needed for supplementary Figures 9, 10, 13 and 14 --
# the four figures whose \includegraphics in docs/paper/supplementary.tex still
# point at the old pics/ PNGs rather than supplementary_figs/Supp_Figure_N.png.
#
#   Figures 9 + 10  <- one job (submit_fig09_10_bau_gen.slurm): both are plotted
#                      off the same decarb x elec-price generation run, so there
#                      is nothing to gain from splitting them.
#   Figure 13       <- submit_fig13_achi_carbon_gen.slurm
#   Figure 14       <- submit_fig14_achi_rebate_gen.slurm
#
# The three jobs are independent (no shared state, no dependencies), so they
# queue and can run concurrently. See README.md for the run-count math behind
# each job's --time budget.
#
# Usage (from anywhere):
#   bash package/supplementary_runs/submit_figs_09_10_13_14.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p slurm_logs   # the #SBATCH --output paths are relative to the submit dir

JOBS=(
    submit_fig09_10_bau_gen.slurm      # Figures 9 and 10
    submit_fig13_achi_carbon_gen.slurm # Figure 13
    submit_fig14_achi_rebate_gen.slurm # Figure 14
)

for job in "${JOBS[@]}"; do
    echo "Submitting $job"
    sbatch "$SCRIPT_DIR/$job"
done

cat <<'NOTE'

Once the jobs finish, each log's first lines name its fresh results/ folder:
  Fig 9/10 -> results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_<ts>
              Plots/fig9_bau_timeseries.png        (Figure 9)
              elasticity_comparison.png            (Figure 10, folder top level)
  Fig 13   -> results/cross_a_chi_vs_Carbon_price_<ts>
              policy_surface_heatmap_with_BAU.png  (Figure 13, folder top level)
  Fig 14   -> results/cross_a_chi_vs_Adoption_subsidy_<ts>
              policy_surface_heatmap_with_BAU.png  (Figure 14, folder top level)
NOTE
