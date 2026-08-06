"""
FIRST THING TO RUN WHEN THE REAL POPULATION FILE ARRIVES.

Prints the marginal distributions of the five per-agent preference vectors under
the parametric draws and under the synthetic population, side by side.

Why this is the first check and not an afterthought: every other parameter in
the model (kappa, r, alpha, zeta, nu, WTP_E_mean, the NK cost/efficiency ranges)
was calibrated against the PARAMETRIC marginals. If the real population is much
tighter or much wider than those, the model moves into a regime none of those
parameters were set for, and no amount of calibrating the six free parameters
will rescue it. Two failure modes seen in development:

  - a VMT column read as monthly when it was annual: d_vec 12x too large, the
    lifetime fuel-cost term swamps the utility, EV share goes to 1.0 within a
    year, nothing errors.
  - a population that was merely too HOMOGENEOUS (VMT log-sd 0.24 against the
    parametric 0.82): no low-mileage agents means nobody stays on ICE, and EV
    share ran to ~85% instead of ~4%. Medians all matched perfectly; only the
    spreads gave it away. Hence the log-sd column below.

What to look for, in priority order:
  1. d_vec median within ~2x of 1274, and d_vec log-sd within ~0.3 of 0.82.
  2. beta_vec log-sd within ~0.2 of 0.92 (i.e. income log-sd near income_sigma).
  3. medians of all five within ~20%.
A mismatch is not necessarily wrong -- the real population is the real
population -- but it means the non-free parameters need revisiting BEFORE the
calibration, and it should be stated in the paper rather than discovered later.

Run:
    python -m package.validation.compare_populations
"""

import json
import numpy as np

from package.model.controller import Controller
from package.resources.utility import load_object

# Reference values from the parametric draws, for the guidance above.
REFERENCE = {"d_vec_median": 1274.0, "d_vec_log_sd": 0.82, "beta_vec_log_sd": 0.92}
VECTORS = ("d_vec", "beta_vec", "gamma_vec", "nu_vec", "chi_vec")


def _partial_controller(base_params, use_synth):
    """
    Build a Controller only as far as gen_users_parameters().

    Deliberately stops there: the five vectors are fully determined at that
    point, and running the remaining 456 timesteps to inspect them would make
    this diagnostic 100x slower for no extra information.
    """
    bp = json.loads(json.dumps(base_params))
    bp.setdefault("parameters_synthetic_population", {})["use"] = bool(use_synth)
    bp["calibration_data"] = load_object("package/calibration_data", "calibration_data_input")
    bp["time_steps_max"] = bp["duration_burn_in"] + bp["duration_calibration"] + bp["duration_future"]

    c = Controller.__new__(Controller)
    c.absolute_2035 = 144
    c.unpack_controller_parameters(bp)
    c.parameters_EV["delta"] = c.parameters_ICE["delta"]
    c.parameters_EV["min_Quality"] = c.parameters_ICE["min_Quality"]
    c.parameters_EV["max_Quality"] = c.parameters_ICE["max_Quality"]
    c.handle_seed()
    c.gen_time_series_calibration_scenarios_policies()
    c.gen_users_parameters()
    return c


def _stats(v):
    v = np.asarray(v, dtype=float)
    lv = np.log(np.clip(v, 1e-30, None))
    q = np.percentile(v, [1, 10, 50, 90, 99])
    return {"mean": v.mean(), "sd": v.std(), "log_sd": lv.std(),
            "p1": q[0], "p10": q[1], "p50": q[2], "p90": q[3], "p99": q[4]}


def compare(base_params_path="package/constants/base_params_NN_zip.json", verbose=True):
    with open(base_params_path) as f:
        base_params = json.load(f)

    par = _partial_controller(base_params, use_synth=False)
    syn = _partial_controller(base_params, use_synth=True)

    rows = {}
    for name in VECTORS:
        rows[name] = {"parametric": _stats(getattr(par, name)),
                      "synthetic": _stats(getattr(syn, name))}

    if verbose:
        print(f"population file: "
              f"{base_params['parameters_synthetic_population']['population_path']}")
        print(f"vmt_period     : "
              f"{base_params['parameters_synthetic_population'].get('vmt_period', 'annual')}")
        print(f"agents         : {syn.num_individuals}")
        print(f"beta anchor    : parametric {par.calc_beta_median():.6g}   "
              f"synthetic {syn.calc_beta_median():.6g}\n")

        hdr = f"  {'':<10s} {'':<11s} {'mean':>11s} {'sd':>11s} {'log-sd':>8s} " \
              f"{'p10':>11s} {'p50':>11s} {'p90':>11s}"
        print(hdr)
        for name in VECTORS:
            for side in ("parametric", "synthetic"):
                s = rows[name][side]
                print(f"  {name if side == 'parametric' else '':<10s} {side:<11s} "
                      f"{s['mean']:11.4g} {s['sd']:11.4g} {s['log_sd']:8.3f} "
                      f"{s['p10']:11.4g} {s['p50']:11.4g} {s['p90']:11.4g}")
            print()

        print("  ratio synthetic/parametric:")
        for name in VECTORS:
            p, s = rows[name]["parametric"], rows[name]["synthetic"]
            med = s["p50"] / p["p50"] if p["p50"] else np.inf
            sd = s["sd"] / p["sd"] if p["sd"] else np.inf
            print(f"    {name:<10s} median x{med:7.3f}   sd x{sd:7.3f}   "
                  f"log-sd {p['log_sd']:.3f} -> {s['log_sd']:.3f}")

        print("\n  checks:")
        _check("d_vec median", rows["d_vec"]["synthetic"]["p50"], REFERENCE["d_vec_median"], 2.0, ratio=True)
        _check("d_vec log-sd", rows["d_vec"]["synthetic"]["log_sd"], REFERENCE["d_vec_log_sd"], 0.30)
        _check("beta_vec log-sd", rows["beta_vec"]["synthetic"]["log_sd"], REFERENCE["beta_vec_log_sd"], 0.20)

    return rows


def _check(label, got, want, tol, ratio=False):
    if ratio:
        ok = (want / tol) <= got <= (want * tol)
        detail = f"got {got:.4g}, parametric {want:.4g}, tolerance x{tol:g}"
    else:
        ok = abs(got - want) <= tol
        detail = f"got {got:.3f}, parametric {want:.3f}, tolerance +/-{tol:g}"
    print(f"    [{'OK ' if ok else 'FAIL'}] {label:<18s} {detail}")


if __name__ == "__main__":
    compare()
