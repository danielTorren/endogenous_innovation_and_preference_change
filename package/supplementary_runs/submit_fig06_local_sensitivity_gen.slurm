#!/bin/bash
#SBATCH --job-name=fig06_local_sensitivity_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_calibration.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=12:00:00           # 10 parameters x 4 values x 64 seeds = 2560 runs, duration_future=0
                                   # (456 steps/run) -> 40 sequential 64-wide batches. Pad generously
                                   # on the first run, then trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

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

# Runs all 10 local-sensitivity parameter sweeps (package/generating_data/
# vary_single_param_gen.py, one call per parameter) and combines them into
# one 10-panel figure. Prints one results/single_param_vary_<timestamp>
# folder per parameter, plus a final combined
# results/single_param_vary_<...>_fig6_combined/Plots/fig6_local_sensitivity_combined.png.
uv run python -m package.supplementary_runs.fig06_local_sensitivity_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig06_local_sensitivity_gen
