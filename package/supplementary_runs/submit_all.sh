#!/bin/bash
# Submit every supplementary-material figure job at once. All jobs are
# independent (no shared state, no dependencies between them), so they queue
# and can run concurrently on the cluster -- see README.md for the run-count
# math behind each job's --time budget.
#
# Figure 1 is NOT included here: it does no new simulation, just re-plots an
# already-existing SBI posterior (see fig01_posterior_plot.py), so there's
# nothing to gain from bundling it into an overnight batch -- run
# submit_fig01_posterior_plot.slurm on its own (it finishes in seconds).
#
# Usage (from anywhere):
#   bash package/supplementary_runs/submit_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

JOBS=(
    submit_fig05_calibration_cars_gen.slurm
    submit_fig06_local_sensitivity_gen.slurm
    submit_fig07_08_sobol_gen.slurm
    submit_fig09_10_bau_gen.slurm
    submit_fig11_beta_carbon_gen.slurm
    submit_fig12_beta_rebate_gen.slurm
    submit_fig13_achi_carbon_gen.slurm
    submit_fig14_achi_rebate_gen.slurm
)

for job in "${JOBS[@]}"; do
    echo "Submitting $job"
    sbatch "$SCRIPT_DIR/$job"
done
