#!/bin/bash
#SBATCH --job-name=low_policy_intensity_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in the base_params saved inside the pair-gen folder
#SBATCH --mem=32G                # DOUBLE the other three jobs on purpose: this script sets
                                   # save_timeseries_data_state=1 and duration_future=312, so every one of
                                   # the 64 workers holds full per-step history arrays, and each scenario's
                                   # 64 of them are then stacked into one dict before being pickled.
                                   # Check `seff <jobid>` after the first run and trim.
#SBATCH --time=01:00:00           # Phase 1: 64 calibration runs (duration_future forced to 0)
                                   # Phase 2: BAU + ~10 low-intensity pairs + carbon tax + adoption subsidy
                                   # = ~13 sequential 64-wide batches, but at duration_future=312 rather
                                   # than 144, so each batch costs ~2x a pair-gen batch.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# REQUIRES a completed package.analysis.endogenous_policy_intensity_pair_gen
# run -- this script reads Data/pairwise_outcomes out of its results folder and
# reuses the base_params saved alongside it. Nothing is recalculated from the
# constants/ JSONs here.

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
# it fills or the process exits, well after the work it's reporting happened.
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

# PAIRWISE_FOLDERS: space-separated results/endog_pair_<timestamp> folder(s)
# from the pair-gen job, printed near the top of that job's log. Several are
# merged (main() calls .update() over their pairwise_outcomes), which is how a
# pair-gen run split in halves is put back together. Pass it at submit time:
#
#   sbatch --export=ALL,PAIRWISE_FOLDERS=results/endog_pair_XX_XX_XX__XX_XX_XXXX \
#       package/analysis/submit_low_policy_intensity_gen.slurm
#
#   sbatch --export=ALL,"PAIRWISE_FOLDERS=results/endog_pair_A results/endog_pair_B" \
#       package/analysis/submit_low_policy_intensity_gen.slurm
#
# Left unset, the script falls back to the folder hardcoded in its __main__
# block, which almost certainly is not the one you want -- so fail loudly here
# instead, minutes before `uv sync` and a wasted allocation turn into a
# FileNotFoundError.
if [ -z "${PAIRWISE_FOLDERS:-}" ]; then
    echo "ERROR: PAIRWISE_FOLDERS is not set. Pass the pair-gen results folder(s), e.g." >&2
    echo "  sbatch --export=ALL,PAIRWISE_FOLDERS=results/endog_pair_00_19_42__30_03_2026 \\" >&2
    echo "      package/analysis/submit_low_policy_intensity_gen.slurm" >&2
    exit 1
fi

for folder in $PAIRWISE_FOLDERS; do
    if [ ! -d "$folder/Data" ]; then
        echo "ERROR: '$folder/Data' does not exist (relative to $REPO_ROOT)." >&2
        exit 1
    fi
done

# Output: one fresh results/pair_low_intensity_policies_<timestamp> folder with
# Data/top_policies, Data/outputs_BAU, Data/outputs, Data/outputs_carbon_tax and
# Data/outputs_adoption_subsidy. The min/max EV uptake window (0.94-0.96) that
# selects the low-intensity pairs is hardcoded in the script's __main__ block.
# word splitting on $PAIRWISE_FOLDERS is deliberate -- each folder is a separate argv entry.
# shellcheck disable=SC2086
uv run python -m package.analysis.low_policy_intensity_gen $PAIRWISE_FOLDERS

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.analysis.low_policy_intensity_gen $PAIRWISE_FOLDERS
