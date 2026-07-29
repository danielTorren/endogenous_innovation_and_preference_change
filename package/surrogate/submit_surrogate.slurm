#!/bin/bash
#SBATCH --job-name=surrogate_opt
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_surrogate.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run
#SBATCH --time=24:00:00           # Phase 1 calibration + 150 Phase-2 evaluations; pad generously first time
#SBATCH --partition=<your-partition>

# The whole pipeline is single-node: joblib fans out across CPUs on this one
# machine, there is no multi-node/MPI parallelism anywhere in package/surrogate.

# Stop immediately on any failure (uv sync, run.py, best_policies.py) instead
# of ploughing on to the next command with nothing useful behind it.
set -euo pipefail

module purge
module load python/3.13            # replace with whatever module gives Python 3.13 on this cluster

# OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS are also defaulted
# to 1 inside run.py (before numpy is imported), so these exports are a
# redundant safety net in case run.py is ever invoked through something else.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Force a non-interactive matplotlib backend. Every plt.show() call in
# run.py/optimisation.py/best_policies.py is already wrapped in try/except so
# a missing display can't crash the job, but this is the real fix: on Agg,
# show() is a documented no-op instead of raising at all, and it removes any
# dependence on how this cluster's Linux nodes happen to auto-detect a
# backend when $DISPLAY is unset.
export MPLBACKEND=Agg

# Resolve the repo root relative to this script's own location, so it works
# regardless of the directory `sbatch` was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
