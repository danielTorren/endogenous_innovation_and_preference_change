#!/bin/bash
#SBATCH --job-name=sbi_seed_av
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # Free to set independently of anything in the Python -- see the
                                   # note below. get_num_workers() reads it for joblib n_jobs.
#SBATCH --mem=128G                 # Scale this with --cpus-per-task, NOT with the run count.
                                   # MEASURED per worker on the sibling jobs: 338 MB is import
                                   # baseline alone (133 MB numpy/scipy/networkx/matplotlib, +205 MB
                                   # torch+sbi, pulled in because run_single_simulation_raw lives in
                                   # a module that imports them); settled RSS during a sim is
                                   # 395-472 MB and peak working set over several sims is ~760 MB.
                                   # So 128 workers x ~0.76 GB = ~97 GB, plus 2-4 GB parent. 128G
                                   # leaves ~25% headroom.
                                   #
                                   # Seed averaging does NOT change this. The extra parent-side array
                                   # is x_raw at (num_thetas x num_seeds x 5) float64 = 1.3 MB at the
                                   # defaults; runs are still streamed through the same 128 workers
                                   # one at a time.
#SBATCH --time=12:00:00            # DERIVED from the round-robin measurement (job 2882514): 8192 ABM
                                   # runs on 128 workers = 19.7 min, i.e. ~18.5 s per run.
                                   #
                                   #   total runs = num_rounds x num_thetas_per_round x num_seeds_per_theta
                                   #
                                   # At the defaults (2 x 4096 x 8) that is 65,536 runs = ~2.6 h of
                                   # simulation, ~3 h wall including the two train() calls and the
                                   # 500k-sample posterior draw. 12 h is ~4x headroom.
                                   #
                                   # RAISE IT if you raise any of the three factors. The one likely
                                   # to change is num_seeds_per_theta, and it is strictly linear:
                                   #   K=8  (default)   65,536 runs   ~2.6 h sim   ~3 h wall
                                   #   K=16            131,072 runs   ~5.3 h sim   ~6 h wall
                                   #   K=32            262,144 runs  ~10.5 h sim   -- RAISE --time
                                   # Training gets CHEAPER than the sibling scripts, not dearer: the
                                   # default is 4096 training pairs per round against their 8192-16384,
                                   # at 3-D theta and 5-D x rather than 4-D and 9-D.

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# How to choose --cpus-per-task:
# A round is ONE joblib fan-out of (num_thetas_per_round x num_seeds_per_theta)
# tasks, so --cpus-per-task does not have to match any Python-side count. Just
# keep the queue deep: efficiency stays above ~98% as long as round total /
# cpus-per-task is >= ~16. Seed averaging makes that EASIER than in the sibling
# scripts, because the round total is now the product of both knobs -- at the
# defaults a round is 32,768 tasks, so anything up to ~2048 cores scales close to
# linearly. Two ceilings: this is a single node, so you cannot exceed the largest
# node in the partition (check `sinfo -o "%P %D %c %m"`), and --mem must scale
# with the worker count.

set -euo pipefail

module purge
# This repo requires Python >=3.13 (see .python-version / pyproject.toml).
module load Python/3.13.5-GCCcore-14.3.0

# sbatch runs a non-interactive shell, which does not source ~/.bashrc -- so
# `uv` (installed via the astral.sh installer to ~/.local/bin) would otherwise
# be "command not found" here even though it works fine in an interactive
# terminal. Add it explicitly rather than relying on rc-file sourcing.
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL for this job specifically. sbi_seed_av_gen.py imports torch at module
# scope (and imports two sibling modules that do the same), so every one of the
# 128 joblib workers imports torch too. Left uncapped, torch and the BLAS backends
# each try to claim all 128 cores per process -- 128 workers x 128 threads is a
# 16,384-way oversubscription that thrashes the node on context switches and gets
# nowhere.
#
# Side effect worth knowing: this also pins inference.train() in the PARENT to a
# single thread. That matters even less here than on the sibling jobs -- training
# sees 8192 pairs by round 2 at 3-D theta and 5-D x -- so it is left alone. If it
# ever becomes the bottleneck, the simulation workers are idle by then, so adding
# `torch.set_num_threads(16)` immediately before the inference.train() call would
# restore multithreading for training only.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout is redirected straight to a log file above (not a terminal), so Python
# fully block-buffers it by default -- the per-round prints, the "NOT calibrated"
# kappa line and the reduced-seed-set / resample WARNINGs would otherwise sit in
# the buffer and only appear long after the work they report actually happened.
#
# Note the two streams are separate: joblib's verbose progress goes to STDERR
# (the .err file) while every print() in the gen script goes to STDOUT (the .log
# file). Grep the .log, not the .err, for "WARNING", "NOT calibrated" and
# "best_sample".
export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend. This gen script never plots, but
# package/resources/utility.py imports matplotlib.pyplot at module scope, so the
# import happens regardless; Agg removes any dependence on how this cluster's
# nodes auto-detect a backend when $DISPLAY is unset.
export MPLBACKEND=Agg

# Resolve the repo root. sbatch stages a COPY of this script in a spool directory
# before running it, so ${BASH_SOURCE[0]}-based path resolution resolves relative
# to that spool copy, not this script's real location in the repo.
# $SLURM_SUBMIT_DIR is set by Slurm to wherever `sbatch` was actually invoked
# from, which is reliable regardless of script staging; fall back to the old
# method for direct/manual (non-sbatch) invocation.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

# BEFORE LAUNCHING, check package/constants/base_params_NN.json:
#   - parameters_vehicle_user.kappa   is NOT calibrated by this script and is used
#     verbatim for every draw. The run is conditional on it. The .log prints the
#     value it read on the first line, under "NOT calibrated".
#   - parameters_ICE.rho / parameters_EV.rho / parameters_EV.max_Cost define which
#     model version is being calibrated. base_params_calibration.json has at times
#     disagreed with this file on all three, which silently invalidates any
#     comparison between a posterior fitted here and a calibration_gen run of it.

# uv.lock is checked into the repo -- use uv if available:
uv sync

# NPE calibration of a_chi / b_chi / delta against California EV stock share
# (2020-2023) and mean fleet age, on SEED-AVERAGED, LOGIT-SCALED summary
# statistics with the EV sales channel dropped. See the module docstring for the
# measurements behind each of those three choices.
#
# Writes to results/sbi_seed_av_<timestamp>/Data/ -- posterior, samples,
# best_sample, prior, x_o, base_params, inference, seed_log and run_config.
# Plot afterwards with:
#   python -m package.calibration.NN_multi_round_calibration_multi_plot
# after pointing its fileName= at the folder this run prints on startup. That
# plotting script needs no changes: the saved object names are identical and it
# loads the saved x_o tensor rather than rebuilding one from match_data (its
# reconstruct_x_o fallback does not know about the logit).
uv run python -m package.calibration.sbi_seed_av_gen

# Otherwise, activate a plain venv instead of the two uv lines above:
# source .venv/bin/activate
# python -m package.calibration.sbi_seed_av_gen
