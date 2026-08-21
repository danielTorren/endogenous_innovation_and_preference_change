#!/bin/bash
#SBATCH --job-name=fig09_10_bau_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_inputs_and_emissions.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=08:00:00           # 4 decarb x 3 elec_price = 12 combinations x 64 seeds = 768 runs,
                                   # duration_future=312 (768 steps/run) -> 12 sequential 64-wide
                                   # batches. Pad generously on the first run, then trim from seff.
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

# Runs the shared decarb x elec-price grid once, then both Figure 9 (raw
# time series) and Figure 10 (elasticity) plots off it. Prints one
# results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_<timestamp>
# folder near the top of this job's log, containing
# Plots/fig9_bau_timeseries.png (Figure 9) and, at the folder's top level,
# elasticity_comparison.png (Figure 10).
uv run python -m package.supplementary_runs.fig09_10_bau_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig09_10_bau_gen
