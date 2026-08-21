#!/bin/bash
#SBATCH --job-name=fig05_calibration_cars_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_calibration.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=02:00:00           # 64 seeds run concurrently (1 per core), duration_future=0 so each
                                   # run is only duration_burn_in+duration_calibration=456 steps --
                                   # one single joblib batch, not a sequence of batches. Pad generously
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

# Runs package/generating_data/calibration_gen.py against
# base_params_calibration.json (64 seeds), then
# package/plotting_data/calibration_plot.py's full plotting suite, into one
# fresh results/calibration_gen_<timestamp> folder printed near the top of
# this job's log. Figure 5 itself is Plots/multi_seed_2d_scatter.png there
# (Plots/multi_seed_2d_contour.png is a KDE-contour variant of the same data).
uv run python -m package.supplementary_runs.fig05_calibration_cars_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig05_calibration_cars_gen
