#!/bin/bash
#SBATCH --job-name=endog_policy_pair_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_endogenous_policy_pair_gen.json
#SBATCH --mem=64G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=7-00:00:00         # BY FAR the longest of the four analysis jobs -- see the sizing note below
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Sizing. generate_all_policy_pairs() builds ORDERED pairs, so 5 policies give
# 5 x 4 = 20 pairs, not 10. Each pair sweeps n_steps_for_sweep=5 values of
# policy1 and runs a full Bayesian optimisation of policy2 at every step:
#
#   20 pairs x 5 steps x (n_calls=20 BO evaluations + 1 final) = 2,100
#   sequential 64-wide batches  (= 134,400 individual runs at duration_future=144)
#
# The BO loop cannot be parallelised across evaluations, so wall-clock is
# ~2,100 x (time for one 64-seed batch) and roughly 4x the vary_single job.
# The 7-day request is a guess: time ONE pair first, or watch the per-pair
# "=== Optimizing Pair:" lines in the log, before trusting it.
#
# If it will not fit inside the 10-day partition limit, split the pair list --
# endogenous_policy_intensity_pair_gen.py already carries commented-out
# `policy_pairs = policy_pairs[:10]` / `[10:]` lines just after the
# generate_all_policy_pairs() call for exactly this. Each half writes its own
# results/endog_pair_<timestamp> folder, and low_policy_intensity_gen accepts
# several such folders and merges their pairwise_outcomes -- so splitting
# costs only a duplicated 64-run calibration phase.

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout is redirected straight to a log file above (not a terminal), so
# Python fully block-buffers it by default -- every print() in the gen script
# would otherwise sit in that buffer and only appear in slurm_logs/*.log once
# it fills or the process exits. On a job this long that is the difference
# between watching progress and flying blind for days.
export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend (documented no-op on Agg instead
# of raising when $DISPLAY is unset on a compute node).
export MPLBACKEND=Agg

# Resolve the repo root. sbatch stages a COPY of this script in a spool
# directory before running it, so ${BASH_SOURCE[0]}-based path resolution
# (dirname/../..) resolves relative to that spool copy, not this script's
# real location in the repo. $SLURM_SUBMIT_DIR is set by Slurm to wherever
# `sbatch` was actually invoked from, which is reliable regardless of script
# staging; fall back to the old method for direct (non-sbatch) invocation.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

uv sync

# Submit from the repo root:
#   sbatch package/analysis/submit_endogenous_policy_intensity_pair_gen.slurm
#
# Inputs (both hardcoded in the script's __main__ block, edit there to change
# the policy list, target_ev_uptake=0.95, n_steps_for_sweep=5, n_calls=20 or
# noise=0.05):
#   package/constants/base_params_endogenous_policy_pair_gen.json
#   package/analysis/policy_bounds_vary_pair_policy_gen.json   (flat dict, no
#     "bounds_dict" wrapper -- unlike the single-policy bounds file)
# Output: one fresh results/endog_pair_<timestamp> folder holding
# Data/pairwise_outcomes and Data/conditions. Note that pairwise_outcomes is
# only saved AFTER the last pair finishes, so a job killed by the time limit
# loses everything -- another reason to split rather than to over-request.
# Feed the resulting folder to submit_low_policy_intensity_gen.slurm.
uv run python -m package.analysis.endogenous_policy_intensity_pair_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.analysis.endogenous_policy_intensity_pair_gen
