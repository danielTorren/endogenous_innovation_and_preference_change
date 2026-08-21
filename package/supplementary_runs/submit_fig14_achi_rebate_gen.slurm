#!/bin/bash
#SBATCH --job-name=fig14_achi_rebate_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_vary_policy_joint.json
#SBATCH --mem=128G                # 2G per worker at 64 workers. MEASURED: one full-length run
                                   # (600 steps, 3000 individuals) peaks at ~1.0 GiB RSS, so the
                                   # previous 64G gave 64 workers exactly 1.0 GiB each -- no headroom
                                   # for the parent, the per-worker interpreter, or allocator
                                   # fragmentation. That is what OOM-killed the 21/08/2026 jobs.
#SBATCH --time=24:00:00           # 8 a_chi values x 6 rebate levels x 64 seeds = 3072 policy runs, plus
                                   # a BAU sweep of 8 x 64 = 512 runs -> 3584 runs total, duration_future=144
                                   # (600 steps/run) -> 56 sequential 64-wide batches. Pad generously on
                                   # the first run, then trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Figure 14: a_chi (rows) x new_car_rebate (columns), BAU as first column.
# Run this as an INDEPENDENT job alongside submit_fig11_beta_carbon_gen.slurm,
# submit_fig12_beta_rebate_gen.slurm and submit_fig13_achi_carbon_gen.slurm --
# they share no state and are sized to run in parallel on the cluster.

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

set -euo pipefail

module purge
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

uv sync

# Prints one results/cross_a_chi_vs_Adoption_subsidy_<timestamp> folder near
# the top of this job's log. load_and_plot_combined saves everything at that
# folder's TOP LEVEL (not under Plots/): policy_surface_heatmap_with_BAU.png
# is Figure 14 itself; policy_surface_heatmap.png, seed_variance_boxplot.png,
# cross_grid_comparison.png and ev_adoption_2020_onwards.gif are the extra
# diagnostic views load_and_plot_combined always produces alongside it.
uv run python -m package.supplementary_runs.fig11_14_policy_grid_gen 14

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig11_14_policy_grid_gen 14
