#!/bin/bash
#SBATCH --job-name=fig13_14_achi_bau_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_vary_policy_joint.json
#SBATCH --mem=128G                # 2G per worker. MEASURED: one full-length run (600 steps, 3000
                                   # individuals) peaks at ~1.0 GiB RSS, so 64G gave 64 workers
                                   # exactly 1.0 GiB each with no headroom -- that is what
                                   # OOM-killed the BAU phase of the 21/08/2026 jobs.
#SBATCH --time=02:00:00           # 8 a_chi values x 64 seeds = 512 runs, all policies off,
                                   # duration_future=144 (600 steps/run) -> 8 sequential 64-wide
                                   # batches, ~3 min at the ~20 s/run measured elsewhere. Padded.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# RECOVERY JOB for supplementary Figures 13 and 14.
#
# The 21/08/2026 runs of submit_fig13_achi_carbon_gen.slurm and
# submit_fig14_achi_rebate_gen.slurm each finished their 3,072-run policy phase
# and SAVED it (results/cross_a_chi_vs_Carbon_price_10_00_29__21_08_2026 and
# results/cross_a_chi_vs_Adoption_subsidy_10_00_29__21_08_2026 both hold
# Data/data_cross_ev), then were OOM-killed at the start of the BAU phase. Only
# the BAU sweep is missing, so this job runs just that -- ~3 min instead of
# re-running 37.5 min of policy grid per figure.
#
# One sweep serves BOTH figures: BAU has every policy switched off, so it varies
# only with a_chi, not with which policy the figure crosses it against. That is
# why build_figures.RUNS keys it once as "grid_bau_achi".
#
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

# Prints "BAU-only folder: results/cross_a_chi_vs_Carbon_price_BAU_<timestamp>"
# near the END of this job's log. Paste that into
# RUNS["grid_bau_achi"] in package/paper_figures/build_figures.py, then build
# Figures 13 and 14 with no further simulation:
#   python -m package.paper_figures.build_figures --only S13,S14
uv run python -m package.supplementary_runs.fig11_14_policy_grid_gen 13 bau

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig11_14_policy_grid_gen 13 bau
