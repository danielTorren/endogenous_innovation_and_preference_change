"""
Sweep parameters_vehicle_user.kappa and record HHI and fleet car age.

kappa is the logit noise scale on the vehicle choice. It is a knob that moves both
calibration targets at once: raising it flattens the choice probabilities, which
spreads sales across firms (lower HHI) and increases churn in the second-hand
market (lower mean fleet age). The two targets therefore cannot be tuned
independently through kappa, and this sweep is what shows where the two admissible
bands overlap, if they overlap at all.

The range [1e-4, 3e-4] brackets the calibrated value in
base_params_calibration.json (kappa = 2.25e-4).

Everything else -- what is run, what is saved, what is plotted -- lives in
package/generating_data/sweep_hhi_age_gen.py.

Usage:
    python -m package.generating_data.kappa_sweep_gen
    python -m package.generating_data.kappa_sweep_gen --seeds 4      # quick check
"""
import sys

import numpy as np

from package.generating_data.sweep_hhi_age_gen import parse_seeds, run_and_plot

PARAM_NAME = "kappa"
SUBDICT = "parameters_vehicle_user"
PARAM_LABEL = r"$\kappa$"

# 10 values, endpoints included.
KAPPA_LIST = np.linspace(1e-4, 3e-4, 10)

# Overrides seed_repetitions in the base JSON, so editing that file does not
# silently change the sweep. Independent of the worker count: the submit script's
# --cpus-per-task is what sets how many of the 640 runs are resident at once, and
# therefore peak memory. 64 workers in 32G is an OOM kill on this base params
# file, so do not raise the two together without checking `seff` first.
SEEDS = 64


if __name__ == "__main__":
    run_and_plot(PARAM_NAME, SUBDICT, KAPPA_LIST, PARAM_LABEL,
                 parse_seeds(sys.argv[1:], SEEDS))
