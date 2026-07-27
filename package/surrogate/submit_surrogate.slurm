#!/bin/bash
#SBATCH --job-name=surrogate_opt
#SBATCH --output=surrogate_%j.log
#SBATCH --error=surrogate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_surrogate.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run
#SBATCH --time=24:00:00           # Phase 1 calibration + 150 Phase-2 evaluations; pad generously first time
#SBATCH --partition=<your-partition>

# The whole pipeline is single-node: joblib fans out across CPUs on this one
# machine, there is no multi-node/MPI parallelism anywhere in package/surrogate.

module purge
module load python/3.13            # replace with whatever module gives Python 3.13 on this cluster

# OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS are also defaulted
# to 1 inside run.py (before numpy is imported), so these exports are a
# redundant safety net in case run.py is ever invoked through something else.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Resolve the repo root relative to this script's own location, so it works
# regardless of the directory `sbatch` was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# uv.lock is checked into the repo — use uv if available:
uv sync
uv run python -m package.surrogate.run

# Otherwise, activate a plain venv instead of the two lines above:
# source .venv/bin/activate
# python -m package.surrogate.run
