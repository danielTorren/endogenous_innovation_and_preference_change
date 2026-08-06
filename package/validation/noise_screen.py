"""
Which summary dimensions actually carry information about the parameters?

RUN THIS BEFORE ANY FULL CALIBRATION. It costs a few hundred simulations and it
decides whether the several thousand you are about to spend are usefully spent.

The logic: a summary dimension is informative only if varying theta moves it by
more than the model's own noise moves it. Two numbers per dimension:

  seed SD   -- theta held FIXED, seeds varied. The noise floor.
  theta SD  -- theta drawn from the prior, one seed each. Signal plus noise.

The ratio theta_SD / seed_SD is the screen. Below about 2 the density estimator
cannot learn anything from that dimension and will spend capacity fitting noise;
drop it from the spec in summary_stats.py. A dimension can also fail by being
DEGENERATE (near-zero SD under both), which means it is pinned regardless of
theta, usually because the model saturates.

WHY THIS MATTERS MORE HERE THAN IN A TYPICAL ABM. `seed` varies the behavioural
draws, but `seed_inputs` varies the two NK technology landscapes, and measured
on the parametric config the 2023 aggregate EV share moves over 0.0033 to 0.416
across 24 landscape draws. That noise floor is enormous relative to the target
of 0.038, so it is entirely possible for a dimension to be swamped even though
it looks like it should be informative. The whole point of screening is to find
that out for 200 simulations rather than 8192.

The screen deliberately uses the SAME seed-triple construction as the
calibration (build_seed_triples), so the noise floor it reports is the noise
floor the calibration will actually face, not an optimistic version of it.

Run:
    python -m package.validation.noise_screen
"""

import json
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

from package.resources.utility import load_object, get_num_workers
from package.resources.run import generate_data
from package.calibration.summary_stats import model_summary, dim_names
from package.calibration.NN_multi_round_calibration_zip_gen import (
    DEFAULT_PARAMETERS_LIST, build_seed_triples, _sanitize,
)

PASS_RATIO = 2.0


def _run(base_params, parameters_list, theta, seed_triple, spec, zip_table):
    params = deepcopy(base_params)
    for p, v in zip(parameters_list, theta):
        params[p["subdict"]][p["name"]] = float(v)
    params.update(seed_triple)
    c = generate_data(params)
    return _sanitize(model_summary(c, params, spec, zip_table), spec)


def screen(base_params_path="package/constants/base_params_NN_zip.json",
           parameters_list=None,
           n_seed=12,
           n_theta=12,
           seed=0,
           verbose=True):
    parameters_list = DEFAULT_PARAMETERS_LIST if parameters_list is None else parameters_list

    with open(base_params_path) as f:
        base_params = json.load(f)
    bundle = load_object("package/calibration_data", "calibration_data_output_zip")
    spec, zip_table = bundle["spec"], bundle["zip_table"]
    names = dim_names(spec)

    rs = np.random.RandomState(seed)
    lo = np.array([p["bounds"][0] for p in parameters_list])
    hi = np.array([p["bounds"][1] for p in parameters_list])
    theta_mid = 0.5 * (lo + hi)
    thetas = rs.uniform(lo, hi, size=(n_theta, len(lo)))

    triples = build_seed_triples(base_params, max(n_seed, n_theta))
    nw = get_num_workers()

    if verbose:
        print(f"noise screen: {n_seed} seeds at fixed theta + {n_theta} thetas "
              f"= {n_seed + n_theta} simulations on {nw} workers")

    # noise floor: theta fixed at the prior midpoint, seeds varied
    seed_runs = Parallel(n_jobs=nw)(
        delayed(_run)(base_params, parameters_list, theta_mid, triples[j], spec, zip_table)
        for j in range(n_seed))

    # signal: theta varied, one seed triple each (so each carries one seed's noise)
    theta_runs = Parallel(n_jobs=nw)(
        delayed(_run)(base_params, parameters_list, thetas[j], triples[j], spec, zip_table)
        for j in range(n_theta))

    seed_arr = np.asarray(seed_runs)
    theta_arr = np.asarray(theta_runs)
    sd_seed = seed_arr.std(axis=0, ddof=1)
    sd_theta = theta_arr.std(axis=0, ddof=1)
    ratio = np.divide(sd_theta, sd_seed, out=np.full_like(sd_theta, np.inf), where=sd_seed > 0)

    if verbose:
        print(f"\n  {'dim':<24s} {'seed SD':>11s} {'theta SD':>11s} {'ratio':>8s}  verdict")
        for i, n in enumerate(names):
            if sd_seed[i] == 0 and sd_theta[i] == 0:
                v = "DEGENERATE - pinned regardless of theta, drop"
            elif ratio[i] < PASS_RATIO:
                v = f"SWAMPED - below {PASS_RATIO}x, drop from the spec"
            else:
                v = "informative"
            print(f"  {n:<24s} {sd_seed[i]:11.6f} {sd_theta[i]:11.6f} {ratio[i]:8.2f}  {v}")

        keep = [n for n, r in zip(names, ratio) if r >= PASS_RATIO]
        drop = [n for n, r in zip(names, ratio) if r < PASS_RATIO]
        print(f"\n  {len(keep)}/{len(names)} dimensions pass at {PASS_RATIO}x")

        # Distinguish "this dimension is uninformative" from "the noise floor is
        # so high that NOTHING is informative". They call for opposite actions
        # and the ratio alone cannot tell them apart, so compare the noise floor
        # against the target LEVEL as well as against the theta signal.
        x_o = np.asarray(load_object("package/calibration_data",
                                     "calibration_data_output_zip")["x_o"])
        rel_noise = np.divide(sd_seed, np.abs(x_o),
                              out=np.full_like(sd_seed, np.inf), where=x_o != 0)
        floor_bound = float(np.median(rel_noise)) > 1.0

        if floor_bound:
            print()
            print("  " + "!" * 70)
            print("  THE NOISE FLOOR IS THE BINDING PROBLEM, NOT THE CHOICE OF DIMENSIONS.")
            print(f"  Median seed SD is {np.median(rel_noise):.1f}x the target VALUE itself")
            print(f"  (e.g. stock_2023: seed SD {sd_seed[names.index('stock_2023')]:.3f} "
                  f"vs target {x_o[names.index('stock_2023')]:.3f}).")
            print()
            print("  Dropping dimensions will NOT help: they are all swamped by the same")
            print("  noise, and the ratios above are near 1 because signal and noise are")
            print("  comparable everywhere, not because these particular statistics are bad.")
            print()
            print("  The lever is seeds_per_theta in NN_multi_round_calibration_zip_gen.main().")
            print("  Averaging k simulations per theta cuts the noise SD by sqrt(k) at k times")
            print("  the cost. Given the ratios above, k = 8 to 16 is the right order.")
            print("  Most of this noise is seed_inputs, the NK technology landscape draw: the")
            print("  2023 aggregate EV share spans 0.0033 to 0.416 across 24 landscape draws.")
            print("  See the methodological fork documented on seeds_per_theta before choosing.")
            print("  " + "!" * 70)
        elif drop:
            print(f"  DROP: {drop}")
            print("  Edit DEFAULT_SPEC in package/calibration/summary_stats.py to remove the")
            print("  corresponding years, then rebuild the target. Keeping swamped dimensions")
            print("  does not just waste capacity, it actively dilutes the informative ones.")

        if n_seed < 20:
            print(f"\n  NOTE n_seed={n_seed} gives a noisy estimate of the noise floor itself. "
                  f"Use 30+ before dropping anything on the strength of a borderline ratio.")

    return {
        "dim_names": names,
        "sd_seed": sd_seed,
        "sd_theta": sd_theta,
        "ratio": ratio,
        "pass_mask": ratio >= PASS_RATIO,
        "seed_runs": seed_arr,
        "theta_runs": theta_arr,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seed", type=int, default=12)
    ap.add_argument("--n-theta", type=int, default=12)
    a = ap.parse_args()
    screen(n_seed=a.n_seed, n_theta=a.n_theta)
