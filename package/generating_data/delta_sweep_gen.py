"""
Sweep parameters_ICE.delta and record HHI and fleet car age.

delta is the monthly depreciation rate of vehicle efficiency. It is the most
direct lever on fleet age there is: a car's driving utility carries a
(1 - delta)**age factor (Social_Network.gen_vehicle_dict_vecs), so raising delta
makes old cars worse faster, pushes owners to replace them sooner, and pulls mean
fleet age down while feeding the new-car market. Its effect on HHI runs through
that same replacement demand rather than through firm behaviour, so the two target
series need not move together, which is the point of plotting them side by side.

TWO THINGS TO KNOW BEFORE READING THE OUTPUT.

1. This is NOT an ICE-only knob despite living in parameters_ICE.
   Controller.unpack_controller_parameters copies it straight into
   parameters_EV["delta"], so one value applies to both technologies. There is no
   way to sweep ICE depreciation alone through this key.

2. delta also shifts every agent's emissions willingness to pay. gen_gamma builds
   gamma_vec proportional to (r - delta - r*delta), which shrinks towards zero as
   delta approaches r/(1+r). So a high-delta run is not a pure "faster
   depreciation" run: consumers there also care measurably less about emissions,
   and that confounds the EV share column. It does not confound HHI or fleet age
   nearly as much, which is why those two are what this script reports.

That same term is a hard feasibility limit, checked by _precheck below before any
run is launched: gen_gamma raises "r <= delta/(1-delta)" outright, and the utility
function divides by (r - delta - r*delta), so the grid must stay clear of
r/(1+r) = 0.0040576 at the calibrated r.

The range [0.001, 0.0033] is the calibration prior for delta (see the parameter
table in package/calibration/NN_multi_round_calibration_multi_gen.py), which
brackets the calibrated value in base_params_calibration.json (delta = 0.00175)
and sits well inside the feasibility limit.

Everything else -- what is run, what is saved, what is plotted -- lives in
package/generating_data/sweep_hhi_age_gen.py.

Usage:
    python -m package.generating_data.delta_sweep_gen
    python -m package.generating_data.delta_sweep_gen --seeds 4      # quick check
"""
import sys

import numpy as np

from package.generating_data.sweep_hhi_age_gen import parse_seeds, run_and_plot

PARAM_NAME = "delta"
SUBDICT = "parameters_ICE"      # copied to parameters_EV by the controller, see above
PARAM_LABEL = r"$\delta$"

# 10 values, endpoints included. The calibration prior range.
DELTA_LIST = np.linspace(0.00175, 0.0022, 5)

# Overrides seed_repetitions in the base JSON, so editing that file does not
# silently change the sweep. Independent of the worker count: the submit script's
# --cpus-per-task is what sets how many of the 640 runs are resident at once, and
# therefore peak memory. 64 workers in 32G is an OOM kill on this base params
# file, so do not raise the two together without checking `seff` first.
SEEDS = 64


def _precheck(base_params, values):
    """
    Reject an infeasible grid in one second rather than after the first worker dies.

    Controller.gen_gamma raises unless delta/(1-delta) < r, and the driving-cost
    term divides by (r - delta - r*delta), so the run is not merely wrong near the
    ceiling, it is undefined at it. Stops a whole 640-run job being burnt on a grid
    whose top value cannot run.
    """
    r = base_params["parameters_vehicle_user"]["r"]
    ceiling = r / (1 + r)
    worst = float(np.max(values))
    if worst >= ceiling:
        raise ValueError(
            f"delta grid reaches {worst:.6g}, but gen_gamma requires "
            f"delta < r/(1+r) = {ceiling:.6g} at r = {r:.6g}. Lower the grid or "
            f"raise parameters_vehicle_user.r."
        )
    # gamma_vec scales with (r - delta - r*delta); report how much of the base
    # emissions WTP survives at the top of the grid, so the EV column is read with
    # that in mind rather than taken at face value.
    base_delta = base_params["parameters_ICE"]["delta"]
    scale = lambda d: (r - d - r * d) / ((1 + r) * (1 - d))
    print(f"delta grid {values.min():.6g} to {worst:.6g}, feasibility ceiling "
          f"{ceiling:.6g}")
    print(f"emissions WTP (gamma) at grid top is "
          f"{scale(worst) / scale(base_delta):.2f}x its value at the calibrated "
          f"delta = {base_delta:.6g}\n")


if __name__ == "__main__":
    run_and_plot(PARAM_NAME, SUBDICT, DELTA_LIST, PARAM_LABEL,
                 parse_seeds(sys.argv[1:], SEEDS), precheck=_precheck)
