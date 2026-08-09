#!/bin/bash
#SBATCH --job-name=low_policy_intensity_plot
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # plotting is single-threaded: it loads pickles and draws
#SBATCH --mem=32G                 # the outputs_* pickles hold full per-step history for 64 seeds
                                   # per scenario and are all held in memory at once. Check
                                   # `seff <jobid>` after the first run and trim.
#SBATCH --time=00:30:00
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`)

# Reads an existing results/pair_low_intensity_policies_<timestamp> folder
# produced by submit_low_policy_intensity_gen.slurm and writes PNGs into its
# Plots/ subfolder. Nothing is simulated here.

set -euo pipefail

module purge
module load Python/3.13.5-GCCcore-14.3.0

# sbatch runs a non-interactive shell that does not source ~/.bashrc, so `uv`
# would otherwise be "command not found" here.
export PATH="$HOME/.local/bin:$PATH"

export PYTHONUNBUFFERED=1

# Force a non-interactive matplotlib backend. main() ends in plt.show(), which
# raises on a compute node with no $DISPLAY unless the backend is Agg.
export MPLBACKEND=Agg

# sbatch stages a COPY of this script in a spool directory, so ${BASH_SOURCE[0]}
# resolves to the copy rather than this file's real location in the repo.
# $SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

# `python -m package.analysis...` resolves the `package` package from the
# current directory, so this cd is what makes the import work -- running the
# same command from package/analysis/ fails with ModuleNotFoundError.

uv sync

# The folder to plot is the one hardcoded in the script's __main__ block.
uv run python -m package.analysis.low_policy_intensity_plot
