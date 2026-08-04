#!/bin/bash
#SBATCH --job-name=NN_calib
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # MUST match "num_simulations" in the __main__ block of
                                   # NN_multi_round_calibration_multi_gen.py -- see the note below.
#SBATCH --mem=64G                 # measured ~311 MB peak RSS per worker x 64 = ~20 GB, plus
                                   # the parent process and torch; 64G leaves comfortable headroom
#SBATCH --time=24:00:00           # 3 rounds x 64 seeds x 64 sims = 12,288 ABM runs, executed as
                                   # 192 sequential waves of 64 parallel sims. Estimated 4-7 h;
                                   # padded generously because the per-wave time is set by the
                                   # SLOWEST draw in the wave and sim cost varies ~2x with theta.

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Why --cpus-per-task must equal num_simulations (both 64):
# sbi's simulate_for_sbi() splits a call into `num_simulations //
# simulation_batch_size` joblib tasks -- with simulation_batch_size=1 that is
# exactly num_simulations = 64 tasks. get_num_workers() then reads
# SLURM_CPUS_PER_TASK for n_jobs. So 64 CPUs runs each call as a single wave.
# Asking for MORE (e.g. 128) leaves half the allocation idle for the whole job;
# asking for FEWER (e.g. 32) splits every one of the 192 waves into two passes
# and roughly doubles the wall time.

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
# imports torch at module scope, so every one of the 64 joblib workers imports
# torch too. Left uncapped, torch and the BLAS backends each try to claim all
# 64 cores per process -- 64 workers x 64 threads is a 4096-way oversubscription
# that thrashes the node on context switches and gets nowhere.
#
# Side effect worth knowing: this also pins inference.train() in the PARENT to
# a single thread. That is cheap at this scale (12k samples, 2-D theta, 8-D x)
# so it is left alone here. If training ever becomes the bottleneck, the
# simulation workers have already exited by then, so adding
# `torch.set_num_threads(16)` immediately before the inference.train() call
# would restore multithreading for training only, without reintroducing the
# oversubscription above.
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
