#!/bin/bash
#SBATCH --job-name=vary_single_policy_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_vary_single_policy_gen.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=3-00:00:00         # Phase 1: 64 calibration runs (duration_future forced to 0)
                                   # Phase 2: 5 policies x 100 intensities x 64 seeds = 32,000 runs
                                   # at duration_future=144 -> 500 sequential 64-wide batches.
                                   # This is the second-largest of the four analysis jobs; pad generously
                                   # on the first run, then trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

set -euo pipefail

module purge
# This repo requires Python >=3.13 (see .python-version / pyproject.toml).
# Confirmed available via `module avail Python` on jupyter-torrenperaire.
module load Python/3.13.5-GCCcore-14.3.0

# sbatch runs a non-interactive shell, which does not source ~/.bashrc -- so
# `uv` (installed via the astral.sh installer to ~/.local/bin) would
# otherwise be "command not found" here even though it works fine in an
# interactive terminal. Add it explicitly rather than relying on rc-file
# sourcing.
export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout is redirected straight to a log file above (not a terminal), so
# Python fully block-buffers it by default -- every print() in the gen script
# would otherwise sit in that buffer and only appear in slurm_logs/*.log once
# it fills or the process exits, well after the work it's reporting happened.
export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend (documented no-op on Agg instead
# of raising when $DISPLAY is unset on a compute node).
export MPLBACKEND=Agg

# Resolve the repo root. sbatch stages a COPY of this script in a spool
# directory before running it, so ${BASH_SOURCE[0]}-based path resolution
# (dirname/../..) resolves relative to that spool copy, not this script's
# real location in the repo. $SLURM_SUBMIT_DIR is set by Slurm to wherever
# `sbatch` was actually invoked from, which is reliable regardless of script
# staging; fall back to the old method for direct (non-sbatch) invocation.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

uv sync

# Submit from the repo root:
#   sbatch package/analysis/submit_vary_single_policy_gen.slurm
#
# Inputs (both hardcoded in the script's __main__ block, edit there to change):
#   package/constants/base_params_vary_single_policy_gen.json
#   package/analysis/policy_bounds_vary_single_policy_gen.json
# Output: one fresh results/vary_single_policy_gen_<timestamp> folder, printed
# as "File Name:" near the top of this job's log. The Calibration_runs/
# subfolder of pickled controllers is deleted at the end of a successful run --
# if the job dies mid-way it stays behind and can be large, so clean it up.
uv run python -m package.analysis.vary_single_policy_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.analysis.vary_single_policy_gen
