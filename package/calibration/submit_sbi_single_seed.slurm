#!/bin/bash
#SBATCH --job-name=sbi_single_seed
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # Free to set independently of anything in the Python -- see the
                                   # note below. get_num_workers() reads it for joblib n_jobs.
#SBATCH --mem=128G                 # Scale this with --cpus-per-task, NOT with the draw count.
                                   # MEASURED per worker: 338 MB is import baseline alone (133 MB
                                   # numpy/scipy/networkx/matplotlib, +205 MB torch+sbi, pulled in
                                   # because run_single_simulation lives in a module that imports
                                   # them); settled RSS during a sim is 395-472 MB and peak working
                                   # set over several sims is ~760 MB. So 128 workers x ~0.76 GB
                                   # = ~97 GB, plus 2-4 GB parent. 128G leaves ~25% headroom.
                                   # Doubling the draws does not change this: draws are streamed
                                   # through the same 128 workers, and the parent-side arrays are
                                   # trivial (32,768 x 9 floats of x, plus 500k x 4 of samples).
#SBATCH --time=12:00:00           # MEASURED on the round-robin job (2882514): 8192 draws on 128
                                   # workers = 19.7 min of simulation per round. This script's
                                   # default is 16,384 draws x 2 rounds = 32,768 runs, so ~40 min
                                   # per round, ~1.5-2 h total including the two train() calls, the
                                   # non-finite resample passes and the 500k-sample posterior draw.
                                   # 12 h is ~6x headroom. RAISE IT if you increase
                                   # num_draws_per_round or num_rounds, or if you set
                                   # vary_seed_inputs=True -- the landscape draw changes ABM cost.
                                   # (The 24 h on submit_NN_calibration.slurm predates this
                                   # measurement and was an over-estimate.)

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# How to choose --cpus-per-task:
# A round is ONE joblib fan-out of num_draws_per_round tasks, so --cpus-per-task
# does not have to match any Python-side count. Just keep the queue deep:
# efficiency stays above ~98% as long as round total / cpus-per-task is >= ~16,
# so at 16,384 draws anything up to ~1024 cores scales close to linearly. Two
# ceilings: this is a single node, so you cannot exceed the largest node in the
# partition (check `sinfo -o "%P %D %c %m"`), and --mem must scale with the
# worker count.

set -euo pipefail

module purge
# This repo requires Python >=3.13 (see .python-version / pyproject.toml).
module load Python/3.13.5-GCCcore-14.3.0

# sbatch runs a non-interactive shell, which does not source ~/.bashrc -- so
# `uv` (installed via the astral.sh installer to ~/.local/bin) would otherwise
# be "command not found" here even though it works fine in an interactive
# terminal. Add it explicitly rather than relying on rc-file sourcing.
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL for this job specifically. sbi_single_seed_gen.py imports torch at
# module scope, so every one of the 128 joblib workers imports torch too. Left
# uncapped, torch and the BLAS backends each try to claim all 128 cores per
# process -- 128 workers x 128 threads is a 16,384-way oversubscription that
# thrashes the node on context switches and gets nowhere.
#
# Side effect worth knowing: this also pins inference.train() in the PARENT to a
# single thread. Training here sees 32,768 samples by round 2 -- double the
# round-robin script -- but at 4-D theta and 9-D x that is still cheap, so it is
# left alone. If training ever becomes the bottleneck, the simulation workers are
# idle by then, so adding `torch.set_num_threads(16)` immediately before the
# inference.train() call would restore multithreading for training only, without
# reintroducing the oversubscription above.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout is redirected straight to a log file above (not a terminal), so Python
# fully block-buffers it by default -- the per-round "ROUND: i/n" prints and the
# non-finite-draw WARNINGs would otherwise sit in the buffer and only appear long
# after the work they report actually happened.
#
# Note the two streams are separate: joblib's verbose progress goes to STDERR
# (the .err file) while every print() in the gen script goes to STDOUT (the .log
# file). Grep the .log, not the .err, for "WARNING" and "best_sample".
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

# uv.lock is checked into the repo -- use uv if available:
uv sync

# NPE calibration of a_chi / b_chi / delta / kappa against California EV stock
# share, EV sales share and mean fleet age, 2020-2023, with ONE UNIQUE ABM SEED
# PER DRAW (see the module docstring for how this differs from
# NN_multi_round_calibration_multi_gen).
#
# Writes to results/sbi_single_seed_<timestamp>/Data/ -- posterior, samples,
# best_sample, prior, x_o, base_params, inference, plus seed_log and run_config
# which are specific to this script. Plot afterwards with:
#   python -m package.calibration.NN_multi_round_calibration_multi_plot
# after pointing its fileName= at the folder this run prints on startup. That
# plotting script needs no changes: the saved object names are identical.
uv run python -m package.calibration.sbi_single_seed_gen

# Otherwise, activate a plain venv instead of the two uv lines above:
# source .venv/bin/activate
# python -m package.calibration.sbi_single_seed_gen
