#!/bin/bash
#SBATCH --job-name=policy_test_future_sight
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in surrogate/constants/base_params_surrogate.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run
#SBATCH --time=08:00:00           # Phase 1 calibration + (1 BAU + 10 policies) x 2 expectation modes
                                   # = 22 Phase-2 scenarios x 64 seeds, duration_future=312; pad generously first time
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Requires package/policy_test_future_sight/input_data/pareto.npz to already
# exist — run package.surrogate.run first (its own submit_surrogate.slurm),
# then copy its results/surrogate_optimisation/Data/pareto.npz to that path.

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
# Python fully block-buffers it by default -- every print() in gen.py would
# otherwise sit in that buffer and only appear in slurm_logs/*.log once it
# fills or the process exits, well after the work it's reporting happened.
export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend -- plot.py's plt.show() call is
# already wrapped in try/except, but this is the real fix (documented no-op
# on Agg instead of raising when $DISPLAY is unset on a compute node).
export MPLBACKEND=Agg

# Resolve the repo root. sbatch stages a COPY of this script in a spool
# directory before running it, so ${BASH_SOURCE[0]}-based path resolution
# (dirname/../..) resolves relative to that spool copy, not this script's
# real location in the repo -- that previously caused a hard-to-diagnose
# "No pyproject.toml found" failure a couple of seconds into the job.
# $SLURM_SUBMIT_DIR is set by Slurm to wherever `sbatch` was actually
# invoked from, which is reliable regardless of script staging; fall back to
# the old method for direct/manual (non-sbatch) invocation.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

uv sync

# One command does calibration (Phase 1, always fresh -- see
# get_or_create_calibration) + all 22 scenarios (Phase 2) + plotting.
#
# To instead reuse an already-completed calibration (e.g. from
# package.surrogate's own run), pass the folder printed near the top of that
# job's log:
#   sbatch --export=ALL,EXISTING_CALIB_FOLDER=results/surrogate_calibration_XX_XX_XX__XX_XX_XXXX \
#       package/policy_test_future_sight/submit_policy_test_future_sight.slurm
if [ -n "${EXISTING_CALIB_FOLDER:-}" ]; then
    uv run python -m package.policy_test_future_sight.gen "--existing_calib_folder=${EXISTING_CALIB_FOLDER}"
else
    uv run python -m package.policy_test_future_sight.gen
fi

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.policy_test_future_sight.gen
