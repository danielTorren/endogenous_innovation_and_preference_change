#!/bin/bash
#SBATCH --job-name=figs_06_09_10
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" (64) in both base_params files this uses
#SBATCH --mem=128G                # 2G per worker at 64 workers. MEASURED peak RSS for ONE run:
                                   #   base_params_calibration        (456 steps) ~0.82 GiB  -> S6
                                   #   base_params_inputs_and_emissions (768 steps) ~1.18 GiB -> S9/S10
                                   # so 64 workers need ~52 GiB and ~75 GiB respectively for run
                                   # data alone. The 64G these jobs used to request left nothing
                                   # for the parent or the per-worker interpreter, and is what
                                   # OOM-killed the 21/08/2026 figs 13/14 jobs.
#SBATCH --time=08:00:00           # S6: 9 missing panels x 4 values x 64 seeds = 2,304 runs at 456
                                   # steps (~13 s/run) -> 36 sequential 64-wide batches, ~8 min.
                                   # S9/S10: 4 decarb x 3 price x 64 seeds = 768 runs at 768 steps
                                   # (~21 s/run) -> 12 batches, ~4 min. ~15 min total including
                                   # plotting. Padded generously; trim from seff after a real run.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# The three supplementary figures still missing after the 21/08/2026 batch:
#   S6  10-panel local sensitivity  (9 of 10 panels need running; panel a exists)
#   S9  BAU EV uptake/emissions time series
#   S10 BAU elasticity
#
# S9 and S10 share one generation run, so this is two generation steps, not
# three. Both are run and then plotted by
# package/supplementary_runs/build_figs_06_09_10.py, which feeds the folders it
# just created into build_figures via --set -- so the PNGs land in
# docs/paper/supplementary_figs/ and supplementary.tex is repointed at them
# without anything being pasted into build_figures.RUNS by hand.
#
# Sequential on purpose: each generation step already saturates all 64 workers,
# so overlapping them would only contend for the same cores and memory ceiling.
#
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

# Prints each results/ folder as it is created, then a "Building S6, S9, S10"
# banner and the copy lines putting the PNGs in place. If S6's panels do not
# all finish it says so and still builds S9/S10 rather than failing the job.
uv run python -m package.supplementary_runs.build_figs_06_09_10

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.build_figs_06_09_10
