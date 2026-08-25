#!/bin/bash
#SBATCH --job-name=sbc_seed_av
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # Same reasoning as submit_sbi_seed_av.slurm: this is one flat
                                   # joblib fan-out of (num_thetas x num_seeds) ABM runs, so the
                                   # count is free to set. Keep the queue deep -- efficiency stays
                                   # above ~98% while round total / cpus-per-task >= ~16.
#SBATCH --mem=128G                 # Scale with --cpus-per-task, NOT with the run count. ~0.76 GB
                                   # peak working set per worker x 128 = ~97 GB, plus 2-4 GB parent.
#SBATCH --time=04:00:00            # DERIVED: ~18.5 s per ABM run on 128 workers. The default here is
                                   # ONE round's worth of simulation, num_thetas x num_seeds, which
                                   # at 512 x 64 = 32,768 runs is ~80 min. There is no training and
                                   # no second round, so wall time is essentially the simulation
                                   # alone -- the SBC ranks themselves are a laptop-minutes job and
                                   # are NOT run here (see the two-step note at the bottom).
                                   #
                                   # RAISE IT if you raise --num-thetas. Strictly linear:
                                   #   512 thetas x K=64    32,768 runs   ~1.4 h
                                   #  1024 thetas x K=64    65,536 runs   ~2.8 h  -- RAISE --time

# WHICH RUN IS BEING VALIDATED. Everything else -- prior, pinned base_params, K,
# `central` -- is read out of this folder's own pickles, deliberately, so this job
# cannot test the posterior against a joint distribution it was not trained on.
# The only thing to set here is the path.
RUN="results/sbi_seed_av_15_31_53__18_08_2026"

set -euo pipefail

module purge
# This repo requires Python >=3.13 (see .python-version / pyproject.toml).
module load Python/3.13.5-GCCcore-14.3.0

# sbatch runs a non-interactive shell, which does not source ~/.bashrc, so `uv`
# in ~/.local/bin would otherwise be "command not found".
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL. This script pulls in sbi_seed_av_gen, which imports torch at module
# scope, so all 128 joblib workers import torch too. Left uncapped, torch and the
# BLAS backends each claim all 128 cores per process: a 16,384-way
# oversubscription that thrashes the node and gets nowhere.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# stdout goes to a file, not a terminal, so Python block-buffers it by default and
# the per-round prints and any resample WARNINGs would appear long after the work
# they report. Note the streams are split: joblib's progress goes to STDERR (.err),
# every print() goes to STDOUT (.log). Grep the .log for "WARNING".
export PYTHONUNBUFFERED=1

# package/resources/utility.py imports matplotlib.pyplot at module scope, so the
# import happens even though this job never plots.
export MPLBACKEND=Agg

# sbatch stages a COPY of this script in a spool dir, so BASH_SOURCE-based path
# resolution would resolve against the copy. $SLURM_SUBMIT_DIR is where sbatch was
# actually invoked; fall back for direct (non-sbatch) invocation.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

uv sync

# STEP 1 (this job): simulate held-out joint samples, theta ~ prior and x from the
# same seed-averaged, logit-scaled simulator the posterior was trained on, on ABM
# seeds disjoint from every seed the calibration issued.
# Writes $RUN/Data/sbc_pairs.pkl.
uv run python -m package.calibration.sbc_seed_av_gen --run "$RUN"

# STEP 2 (cheap, no ABM): rank the true theta against posterior draws, test the
# rank histograms for uniformity, and write the plots. It is run here so the whole
# check completes unattended, but it needs no cluster -- rerun it locally with
# different --num-bins or add --lc2st without repaying step 1.
uv run python -m package.calibration.sbc_seed_av --run "$RUN" --source heldout

# READ THE .log FOR, in order:
#   "WARNING: ... resampling"      thetas whose seeds mostly failed. A handful is
#                                  fine; many means theta is no longer a clean
#                                  prior draw and the SBC result needs that caveat.
#   "rank-histogram shape"         the verdict per parameter. |z| > 2 is a departure;
#                                  z_var > 0 is too narrow, z_var < 0 too wide,
#                                  z_mean != 0 is a bias.
#   "empirical coverage"           the same thing in credible-interval units.
#   "TARP"                         the JOINT check, which marginal ranks cannot do.
