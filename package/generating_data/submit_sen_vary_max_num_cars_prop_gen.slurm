#!/bin/bash
#SBATCH --job-name=sen_vary_max_num_cars_prop_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # DELIBERATELY BELOW seed_repetitions=64, unlike the other gen jobs.
                                   # get_num_workers() takes this as joblib's n_jobs, so it also caps how
                                   # many model instances are alive at once -- the only lever on peak
                                   # memory here. See the --mem note below.
#SBATCH --mem=128G                 # 8G per worker. base_params_vary_single.json sets
                                   # save_timeseries_data_state=1, so every worker holds full per-step
                                   # history for 600 steps (duration_future=144, vs the 456 of the
                                   # calibration-only jobs). On top of that the sweep's top value is
                                   # max_num_cars_prop=2, i.e. a second-hand fleet of 2 x 3000 = 6000 cars
                                   # against the default 0.1 -> 300, a 20x bigger fleet held per worker.
                                   # joblib dispatches in order, so the last 64 tasks are ALL the 6000-car
                                   # case running at once. 64 workers x 32G total (0.5G each) was OOM-killed;
                                   # this is 16x the per-worker headroom. Trim from `seff <jobid>`.
#SBATCH --time=02:00:00           # 5 values x 64 seeds = 320 runs at 600 steps each -> 20 sequential
                                   # 16-wide batches at ~26 s per run, so ~9 min of real work. Padded
                                   # generously against a busy node; trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Manuscript Figure 7: EV uptake under varying used car market capacity.
# Sweeps parameters_second_hand.max_num_cars_prop over [0, 0.1, 0.2, 0.5, 2]
# (constants/vary_single_max_num_cars_prop.json) against
# constants/base_params_vary_single.json. Both are the values hardcoded in the
# gen script's __main__ block -- edit the JSONs, not this file, to change them.
#
# This job is memory-bound, not CPU-bound: the max_num_cars_prop=2 tasks each
# carry a 20x oversized second-hand fleet with full timeseries history. If it
# is OOM-killed again, halve --cpus-per-task before raising --mem -- fewer
# concurrent workers cuts peak memory proportionally and only costs wall-clock,
# of which this job uses very little. Note that save_timeseries_data_state
# cannot simply be turned off: history_mean_price_ICE_EV and
# history_mean_profit_margins_ICE, which generate_multi() returns, are recorded
# inside save_timeseries_data_social_network / _firm_manager and would come back
# empty (history_prop_EV, the only output actually saved here, is always on).

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
