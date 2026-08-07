#!/bin/bash
#SBATCH --job-name=endog_policy_single_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_endogenous_policy_single_gen.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=12:00:00           # Phase 1: 64 calibration runs (duration_future forced to 0)
                                   # Phase 2: BAU + 2 policies x (n_calls=10 BO evaluations + 1 final)
                                   # = 23 sequential 64-wide batches at duration_future=144.
                                   # The BO loop is strictly sequential -- each gp_minimize call must
                                   # finish its 64-seed batch before proposing the next intensity -- so
                                   # wall-clock scales with n_calls, not with total run count.
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
#   sbatch package/analysis/submit_endogenous_policy_intensity_single_gen.slurm
#
# Inputs (both hardcoded in the script's __main__ block, edit there to change
# the policy list, target_ev_uptake=0.95, n_calls=10 or noise=0.05):
#   package/constants/base_params_endogenous_policy_single_gen.json
#   package/analysis/policy_bounds_endog_single_gen.json
# Output: one fresh results/endog_single_<timestamp> folder holding
# Data/policy_outcomes and Data/conditions. The Calibration_runs/ subfolder of
# pickled controllers is deleted at the end of a successful run.
#
# Needs scikit-optimize (skopt) -- it is a project dependency, so `uv sync`
# above installs it; do not swap in a bare `python` that skips the lockfile.
uv run python -m package.analysis.endogenous_policy_intensity_single_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.analysis.endogenous_policy_intensity_single_gen
