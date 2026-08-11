#!/bin/bash
#SBATCH --job-name=NN_calib
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # Free to set independently of anything in the Python now --
                                   # see the note below. get_num_workers() reads it for joblib n_jobs.
#SBATCH --mem=256G                 # ~311 MB peak RSS per worker x 128 workers = ~40 GB, plus the
                                   # parent process and torch. Scale this WITH --cpus-per-task,
                                   # not with the simulation count.
#SBATCH --time=24:00:00           # 2 rounds x 8192 draws = 16,384 ABM runs, dispatched as 2 flat
                                   # waves of 8192 tasks over 128 workers (64 deep per worker).
                                   # Estimated 4-7 h; the deep queue means wall time now tracks the
                                   # MEAN draw cost rather than the slowest draw in each wave.

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# How to choose --cpus-per-task:
# A round is now ONE joblib fan-out of num_simulations x seed_repetitions tasks
# (128 x 64 = 8192), so --cpus-per-task no longer has to match any Python-side
# count. It used to: the old code split each round into 64 sequential waves of
# num_simulations tasks, and the two had to be equal for a wave to run as a
# single pass. That was also the worst possible ratio, because one task per
# worker leaves nothing queued behind a slow draw, and ABM cost varies ~2x with
# theta -- so ~25% of the allocation sat idle at the tail of every one of the
# 128 waves.
#
# Now just keep the queue deep. Efficiency stays above ~98% as long as the round
# total divided by cpus-per-task is >= ~16, so at 8192 draws anything up to ~512
# cores scales close to linearly. Two ceilings: this is a single node, so you
# cannot exceed the largest node in the partition (check
# `sinfo -o "%P %D %c %m"`), and --mem must scale with the worker count.

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

# CRITICAL for this job specifically. NN_multi_round_calibration_multi_gen.py
# imports torch at module scope, so every one of the 128 joblib workers imports
# torch too. Left uncapped, torch and the BLAS backends each try to claim all
# 128 cores per process -- 128 workers x 128 threads is a 16,384-way
# oversubscription that thrashes the node on context switches and gets nowhere.
#
# Side effect worth knowing: this also pins inference.train() in the PARENT to
# a single thread. That is cheap at this scale (16k samples, 4-D theta, 9-D x)
# so it is left alone here. If training ever becomes the bottleneck, the
# simulation workers are idle by then, so adding `torch.set_num_threads(16)`
# immediately before the inference.train() call would restore multithreading for
# training only, without reintroducing the oversubscription above.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout is redirected straight to a log file above (not a terminal), so Python
# fully block-buffers it by default -- the per-round "ROUND: i/n" prints and
# sbi's progress bars would otherwise sit in the buffer and only appear long
# after the work they report actually happened. On a multi-hour job that is the
# difference between being able to monitor progress and flying blind.
export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend. This gen script never plots, but
# package/resources/utility.py imports matplotlib.pyplot at module scope, so the
# import happens regardless; Agg removes any dependence on how this cluster's
# nodes auto-detect a backend when $DISPLAY is unset.
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

# Multi-round NPE calibration of a_chi / b_chi against California EV stock
# share, 2016-2023. Writes to results/NN_calibration_multi_<timestamp>/Data/
# (posterior, samples, best_sample, prior, x_o, base_params, inference).
# Plot afterwards with:
#   python -m package.calibration.NN_multi_round_calibration_multi_plot
# after pointing its fileName= at the folder this run prints on startup.
uv run python -m package.calibration.NN_multi_round_calibration_multi_gen

# Otherwise, activate a plain venv instead of the two uv lines above:
# source .venv/bin/activate
# python -m package.calibration.NN_multi_round_calibration_multi_gen
