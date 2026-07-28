#!/bin/bash
#SBATCH --job-name=variable_carbon_price
#SBATCH --output=variable_carbon_price_%j.log
#SBATCH --error=variable_carbon_price_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_variable_carbon_price.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run
#SBATCH --time=24:00:00           # Phase 1 calibration + (1 BAU + 10 ramps + 5 flats) x 2 expectation modes
                                   # = 32 Phase-2 scenarios x 64 seeds; pad generously first time
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

# Force a non-interactive matplotlib backend -- plot.py's plt.show() calls
# are already wrapped in try/except, but this is the real fix (documented
# no-op on Agg instead of raising when $DISPLAY is unset on a compute node).
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

# One command does calibration (Phase 1, skipped automatically on a rerun
# against the same results/variable_carbon_price/Data/calib_folder.pkl) +
# every scenario (Phase 2) + plotting.
#
# To instead reuse a calibration already produced by package.car_ban (the two
# base_params_*.json configs are byte-for-byte identical, so either's
# calibration is valid for both), pass the folder printed near the top of
# that job's log, e.g.:
#   sbatch --export=ALL,EXISTING_CALIB_FOLDER=results/surrogate_calibration_XX_XX_XX__XX_XX_XXXX \
#       package/variable_carbon_price/submit_variable_carbon_price.slurm
if [ -n "${EXISTING_CALIB_FOLDER:-}" ]; then
    uv run python -m package.variable_carbon_price.gen "--existing_calib_folder=${EXISTING_CALIB_FOLDER}"
else
    uv run python -m package.variable_carbon_price.gen
fi

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.variable_carbon_price.gen
