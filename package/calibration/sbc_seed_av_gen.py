"""
Fresh (theta, x) joint samples for a HELD-OUT SBC check of an sbi_seed_av run.

WHY THIS EXISTS
---------------
package.calibration.sbc_seed_av can run SBC for free on the round-0 pairs already
stored in inference.pkl, but those are the pairs the posterior was TRAINED on. The
check is then in-sample and optimistic: the density estimator has already been
pushed to reproduce those exact points, so uniform ranks there are a weaker claim
than uniform ranks on data it has never seen. This script buys the stronger claim.

WHAT IT GUARANTEES
------------------
1. theta comes from the SAVED prior of the run under test, not a re-declared one,
   so a prior-bounds edit since the run cannot silently invalidate the check.
2. base_params comes from the run's OWN base_params.pkl, so every pinned
   parameter (kappa above all) is byte-identical to what the posterior was
   trained under. SBC compares q against the posterior implied by a specific
   joint; changing the joint underneath it tests nothing.
3. x is built by the run's own build_x() at the run's own num_seeds_per_theta and
   `central`, read out of run_config.pkl. The simulator SBC validates has to be
   the simulator NPE was trained on, seed averaging and logit included.
4. ABM seeds are disjoint from every seed the calibration itself issued. The
   allocator is pre-loaded with the calibration's seed_log before it hands out
   anything, so "held out" means held out in the ABM noise as well as in theta.

COST
----
num_thetas x num_seeds ABM runs, the same arithmetic as one calibration round. At
the ~18 s per run and 128 workers measured on the cluster, 512 x 64 = 32,768 runs
is about 80 minutes. See submit_sbc_seed_av.slurm.

Fewer thetas at the same K is the right way to economise, not the reverse: K has
to stay at the calibration's value or the simulator changes, whereas num_thetas
only costs statistical power. 256 pairs still resolves a z of 2 in the rank mean.

USAGE
-----
    python -m package.calibration.sbc_seed_av_gen --run results/sbi_seed_av_...

Writes <run>/Data/sbc_pairs.pkl. Then:

    python -m package.calibration.sbc_seed_av --run <same> --source heldout
"""

import argparse
import numpy as np
import torch

from package.resources.utility import load_object, save_object
from package.calibration.sbi_single_seed_gen import SeedAllocator
from package.calibration.sbi_seed_av_gen import simulate_round


# Deliberately not the calibration's master_seed (20260816). The allocator is also
# pre-loaded with the calibration's issued seeds, so this only has to be different,
# not lucky.
DEFAULT_MASTER_SEED = 20260824

# Also deliberately not the calibration round-0 torch seed (1): the same value
# would redraw the same 512 thetas out of the same BoxUniform and the "held-out"
# set would be the training set with different ABM seeds.
DEFAULT_TORCH_SEED = 777_001


def excluded_seeds(fileName):
    """
    Every ABM seed the calibration itself used, as a set.

    Loaded from seed_log.pkl, which stores one (num_thetas, num_seeds) matrix per
    round. Feeding these to the new allocator up front makes its rejection loop
    reject them too, so no held-out x is ever produced on a seed that contributed
    to training.
    """
    seed_log = load_object(fileName + "/Data", "seed_log")
    used = set()
    for entry in seed_log:
        used.update(int(s) for s in np.asarray(entry["seed_matrix"]).reshape(-1))
    return used


def main(fileName, num_thetas=None, num_seeds=None,
         master_seed=DEFAULT_MASTER_SEED, torch_seed=DEFAULT_TORCH_SEED):
    D = fileName + "/Data"
    prior = load_object(D, "prior")
    var_dict = load_object(D, "var_dict")
    base_params = load_object(D, "base_params")
    rc = load_object(D, "run_config")

    # Defaults track the run under test rather than being restated here, so the
    # simulator cannot drift. num_seeds especially: overriding it changes what x
    # IS, and then SBC is testing the posterior against a simulator it was never
    # trained on. The flag exists for deliberate sensitivity work only.
    if num_seeds is None:
        num_seeds = rc["num_seeds_per_theta"]
    if num_thetas is None:
        num_thetas = rc["num_thetas_per_round"]

    if num_seeds != rc["num_seeds_per_theta"]:
        print(f"  WARNING: num_seeds {num_seeds} != the run's "
              f"{rc['num_seeds_per_theta']}. x is on a different noise scale than "
              "the training x, and the SBC result will not be interpretable.")

    print(f"run under test: {fileName}")
    print(f"  parameters: {', '.join(p['name'] for p in var_dict)}")
    print(f"  kappa (pinned, from the run's base_params): "
          f"{base_params['parameters_vehicle_user']['kappa']}")
    print(f"  seed_inputs (pinned): {base_params['seed_inputs']}")
    print(f"  central={rc['central']}, K={num_seeds}, thetas={num_thetas}")
    print(f"  TOTAL RUNS: {num_thetas * num_seeds}")

    used = excluded_seeds(fileName)
    allocator = SeedAllocator(master_seed)
    allocator._issued.update(used)
    print(f"  excluding {len(used)} ABM seeds already used by the calibration")

    # simulate_round draws theta from `prior` (so theta ~ prior, as SBC requires),
    # runs each at num_seeds unique seeds, and applies the same logit-then-average
    # collapse the training x went through.
    #
    # One caveat it carries over: a theta whose seeds mostly return non-finite x is
    # RESAMPLED rather than dropped. Strictly that makes theta a draw from the
    # prior truncated to the finitely-simulating region rather than from the prior
    # itself. At the rates this model shows (the sales channel, which caused nearly
    # all of them, is not in x any more) the distortion is negligible; if the run
    # log reports a non-trivial resample count, note it beside the SBC result
    # rather than ignoring it.
    theta, x, seed_matrix, x_sd = simulate_round(
        prior, base_params, var_dict,
        num_thetas=num_thetas, num_seeds=num_seeds,
        allocator=allocator, torch_seed=torch_seed, central=rc["central"],
    )

    pairs = {
        "theta": theta,
        "x": x,
        "seed_matrix": seed_matrix,
        "x_seed_sd": x_sd,
        "num_thetas": num_thetas,
        "num_seeds_per_theta": num_seeds,
        "master_seed": master_seed,
        "torch_seed": torch_seed,
        "central": rc["central"],
    }
    save_object(pairs, D, "sbc_pairs")
    print(f"\nsaved: {D}/sbc_pairs.pkl  (theta {tuple(theta.shape)}, "
          f"x {tuple(x.shape)})")
    print("now run:")
    print(f"  python -m package.calibration.sbc_seed_av --run {fileName} "
          "--source heldout")
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="simulate held-out joint samples for an SBC check")
    ap.add_argument("--run", required=True,
                    help="results/sbi_seed_av_<timestamp> folder to validate")
    ap.add_argument("--num-thetas", type=int, default=None,
                    help="default: the run's num_thetas_per_round")
    ap.add_argument("--num-seeds", type=int, default=None,
                    help="default: the run's num_seeds_per_theta. Changing it "
                         "changes the simulator; see the warning it prints.")
    ap.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    ap.add_argument("--torch-seed", type=int, default=DEFAULT_TORCH_SEED)
    a = ap.parse_args()

    main(fileName=a.run, num_thetas=a.num_thetas, num_seeds=a.num_seeds,
         master_seed=a.master_seed, torch_seed=a.torch_seed)
