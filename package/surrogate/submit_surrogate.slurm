#!/bin/bash
#SBATCH --job-name=surrogate_opt
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_surrogate.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run
#SBATCH --time=24:00:00           # Phase 1 calibration + 150 Phase-2 evaluations; pad generously first time
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# The whole pipeline is single-node: joblib fans out across CPUs on this one
# machine, there is no multi-node/MPI parallelism anywhere in package/surrogate.

# Stop immediately on any failure (uv sync, run.py, best_policies.py) instead
# of ploughing on to the next command with nothing useful behind it.
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

# OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS are also defaulted
# to 1 inside run.py (before numpy is imported), so these exports are a
# redundant safety net in case run.py is ever invoked through something else.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout is redirected straight to a log file above (not a terminal), so
# Python fully block-buffers it by default -- every print() in run.py/
# sampling.py/optimisation.py (LHS/BO progress included) would otherwise sit
# in that buffer and only appear in slurm_logs/*.log once it fills or the
# process exits, well after the work it's reporting actually happened.
export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend. Every plt.show() call in
# run.py/optimisation.py/best_policies.py is already wrapped in try/except so
# a missing display can't crash the job, but this is the real fix: on Agg,
# show() is a documented no-op instead of raising at all, and it removes any
# dependence on how this cluster's Linux nodes happen to auto-detect a
# backend when $DISPLAY is unset.
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

# uv.lock is checked into the repo — use uv if available:
uv sync

# Step 1: constrained search (minimise net_cost subject to the emissions/
# utility bounds in constants/optimisation_config.json) — see run.py's module
# docstring. Writes the ranked feasible policies to
# results/surrogate_optimisation/Data/pareto.npz.
uv run python -m package.surrogate.run

# Step 2: run the cheapest feasible policies found above through the real ABM
# (full time series) and produce the dashboard + trade-off plots. Only runs
# if step 1 actually found at least one feasible policy — if it printed "No
# feasible policy found", this will fail fast with a clear error rather than
# silently producing nothing; loosen emissions_frac/utility_frac in
# optimisation_config.json or add more LHS/BO iterations and resubmit.
uv run python -m package.surrogate.best_policies

# Otherwise, activate a plain venv instead of the three uv lines above:
# source .venv/bin/activate
# python -m package.surrogate.run
# python -m package.surrogate.best_policies
