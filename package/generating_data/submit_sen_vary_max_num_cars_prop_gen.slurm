#!/bin/bash
#SBATCH --job-name=sen_vary_max_num_cars_prop_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_vary_single.json
#SBATCH --mem=32G                 # only data_array_ev_prop is kept (5 x 64 x 600 floats); the per-run
                                   # price/margin arrays are returned but discarded unreshaped. Half the
                                   # 64G the timeseries-heavy jobs need. Trim from `seff <jobid>`.
#SBATCH --time=01:00:00           # 5 values x 64 seeds = 320 runs at 600 steps each (duration_future=144)
                                   # -> 5 sequential 64-wide batches, ~26 s per batch, so ~3 min of real
                                   # work. Padded generously against a busy node; trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Manuscript Figure 7: EV uptake under varying used car market capacity.
# Sweeps parameters_second_hand.max_num_cars_prop over [0, 0.1, 0.2, 0.5, 2]
# (constants/vary_single_max_num_cars_prop.json) against
# constants/base_params_vary_single.json. Both are the values hardcoded in the
# gen script's __main__ block -- edit the JSONs, not this file, to change them.

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

# Generation. Writes one fresh results/sen_vary_max_num_cars_prop_<timestamp>
# folder holding Data/data_array_ev_prop, Data/base_params and Data/vary_single.
uv run python -m package.generating_data.sen_vary_single_param_gen_second_hand_cars

# Plotting. The gen script's main() discards the folder name it built, so
# resolve the newest matching folder instead -- unambiguous here because the
# run above just created it. Produces Plots/ev_prop_single_plot_dated.png,
# which is what package/paper_figures/build_figures.py copies to Figure_7.png.
RESULTS_FOLDER="$(ls -td results/sen_vary_max_num_cars_prop_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"

uv run python -m package.plotting_data.sen_vary_single_param_plot_second_hand_cars "$RESULTS_FOLDER"

echo "Figure 7 source: $RESULTS_FOLDER/Plots/ev_prop_single_plot_dated.png"
echo "Put that folder in RUNS['used_car_capacity'] in package/paper_figures/build_figures.py,"
echo "then run: python -m package.paper_figures.build_figures --only 7"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.sen_vary_single_param_gen_second_hand_cars
