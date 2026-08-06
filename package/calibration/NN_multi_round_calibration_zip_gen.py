"""
Multi-round NPE calibration against the zip-level target.

Differences from NN_multi_round_calibration_multi_gen.py, and why:

1. TARGET. 24 dims (state stock + state sales + cross-sectional gradients +
   between-stratum dispersion) instead of 8 state stock values. See
   summary_stats.py. The gradient dims are what identify eps_beta, a_rural and
   rho_pol; with only the aggregate series those four new parameters are not
   identified at all.

2. SEEDS. Each repetition varies `seed_inputs` and `seed_population` as well as
   `seed`. This is not a refinement, it is a correctness fix. Measured on the
   parametric config, 24 draws of seed_inputs (which seeds the two NK
   technology landscapes, the network and the firm manager) move the 2023 EV
   stock share over 0.0033 to 0.416, median 0.069, against a target of 0.038.
   The existing calibration holds seed_inputs at 22 for all of its runs, so its
   posterior is conditional on ONE landscape draw that happens to sit near the
   target. Varying it makes NPE marginalise over technology and population
   uncertainty instead of conditioning on one lucky world. It also widens the
   posterior, correctly: that width was always there, it was just hidden.

3. MASKS AND ZIP SUBSETS. The simulator can be restricted to a subset of the
   target dimensions (`target_mask`) and/or a subset of zips (`zip_subset`).
   That is what package/validation uses to fit on training data and score on
   held-out data, without any duplicate calibration code.

Run:
    python -m package.calibration.NN_multi_round_calibration_zip_gen
"""

import json
import numpy as np
import torch
from copy import deepcopy
from functools import partial

from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator

from package.resources.utility import (
    produce_name_datetime, save_object, createFolder, load_object, get_num_workers,
)
from package.resources.run import generate_data
from package.calibration.summary_stats import model_summary, dim_names

TARGET_ROOT = "package/calibration_data"
TARGET_NAME = "calibration_data_output_zip"

# The four new covariate-to-preference parameters plus the two existing social
# ones. Bounds are set so that the CURRENT model sits inside every interval:
# eps_beta = 1, a_rural = 0, rho_pol = 0, rho_age = 0 are all interior or
# boundary points, so the calibration can always fall back to it.
DEFAULT_PARAMETERS_LIST = [
    {"name": "a_chi",    "subdict": "parameters_social_network", "bounds": [0.8, 1.5]},
    {"name": "b_chi",    "subdict": "parameters_social_network", "bounds": [2.0, 2.7]},
    {"name": "eps_beta", "subdict": "parameters_social_network", "bounds": [0.3, 1.7]},
    {"name": "a_rural",  "subdict": "parameters_social_network", "bounds": [0.0, 3.0]},
    {"name": "rho_pol",  "subdict": "parameters_social_network", "bounds": [-0.3, 0.9]},
    {"name": "rho_age",  "subdict": "parameters_social_network", "bounds": [-0.3, 0.9]},
    # Network placement. homophily_strength = 0 is the old uniformly random
    # placement. These two are the reason moran_* is in the summary spec: fitting
    # them against cross-sectional gradients alone would leave them trading off
    # against a_chi/b_chi with nothing to separate sorting from contagion.
    {"name": "homophily_strength",       "subdict": "parameters_social_network", "bounds": [0.0, 1.0]},
    {"name": "homophily_spatial_weight", "subdict": "parameters_social_network", "bounds": [0.0, 1.0]},
]


def run_single_simulation(theta, base_params, param_list, spec, zip_table,
                         target_mask, zip_subset, seed_triple, extra_triples=()):
    """
    One simulator call -> one (masked) summary vector.

    Work on a deepcopy: writing theta into the caller's dict leaks back into the
    parent process whenever joblib runs in-process, which is the bug documented
    at length in NN_multi_round_calibration_multi_gen.run_single_simulation.

    `extra_triples` implements seeds_per_theta: the summary vector is AVERAGED
    over 1 + len(extra_triples) simulations at the same theta but different
    seeds, cutting the noise SD by sqrt(k). See the seeds_per_theta discussion
    in main() for why this is usually the binding constraint here.
    """
    params = deepcopy(base_params)
    for i, p in enumerate(param_list):
        params[p["subdict"]][p["name"]] = theta[i].item()

    acc = None
    triples = [seed_triple] + list(extra_triples)
    for seeds in triples:
        params.update(seeds)
        controller = generate_data(params)
        x = _sanitize(model_summary(controller, params, spec, zip_table,
                                    zip_subset=zip_subset), spec)
        acc = x if acc is None else acc + x

    return (acc / len(triples))[target_mask]


def _sanitize(x, spec):
    """
    NPE requires finite inputs. A dimension can legitimately come out non-finite
    when a zip subset contains no sales at all in some year, or when a
    dispersion statistic has fewer than two populated strata. Substituting 0.0
    is the right fallback (it is what "no EV activity observed" means for a
    share, a gradient and a dispersion alike) but it must not pass silently,
    because a whole dimension pinned at 0.0 across every simulation carries no
    information and should be dropped from the spec instead.
    """
    bad = ~np.isfinite(x)
    if bad.any():
        names = [n for n, b in zip(dim_names(spec), bad) if b]
        print(f"  WARNING non-finite summary dims set to 0.0: {names}")
        x = np.where(bad, 0.0, x)
    return x


def build_seed_triples(base_params, n, pin_landscape=True):
    """
    n distinct seed dicts, one per repetition.

    Four independent seeds now (see controller.handle_seed):
        seed            behavioural draws (who considers switching, logit noise)
        seed_landscape  the two NK technology landscapes
        seed_network    the Watts-Strogatz wiring
        seed_population which households are drawn

    pin_landscape=True holds seed_landscape FIXED at its base-params value while
    the other three vary. This is a deliberate modelling choice, not a default
    for convenience, and it needs stating in the paper:

      Fixing it means the posterior is CONDITIONAL on one technology landscape.
      That is defensible -- California drew one landscape, not a distribution
      over them -- but the choice is consequential, because 24 draws of the
      landscape seed move the 2023 EV stock share over 0.0033 to 0.416. So the
      particular landscape you pin matters enormously, and a robustness check
      over several pinned landscapes is not optional.

      Setting pin_landscape=False marginalises over the landscape instead,
      treating it as a nuisance. The posterior widens, correctly.

    The varying seeds advance together rather than being crossed: crossing would
    multiply the run count by n^3 for no gain, since NPE only needs an unbiased
    sample from the joint noise distribution and the diagonal already is one. The
    offsets keep the network draw and the population draw from moving in
    lockstep, which would otherwise confound them.
    """
    s0 = int(base_params.get("seed", 1))
    si0 = int(base_params.get("seed_inputs", 22))
    sl0 = int(base_params.get("seed_landscape", si0))
    sn0 = int(base_params.get("seed_network", si0))
    sp0 = int(base_params.get("seed_population", si0))
    return [{
        "seed": s0 + j,
        "seed_inputs": si0 + j,
        "seed_landscape": sl0 if pin_landscape else sl0 + j,
        "seed_network": sn0 + 500 + j,
        "seed_population": sp0 + 1000 + j,
    } for j in range(n)]


def main(parameters_list=None,
         BASE_PARAMS_LOAD="package/constants/base_params_NN_zip.json",
         TARGET_LOAD_ROOT=TARGET_ROOT,
         TARGET_LOAD_NAME=TARGET_NAME,
         num_simulations=64,
         num_rounds=2,
         seed_repetitions=None,
         seeds_per_theta=1,
         pin_landscape=True,
         target_mask=None,
         zip_subset=None,
         root="NN_calibration_zip",
         verbose=True):
    """
    Args:
        seeds_per_theta: average the summary vector over this many seed triples
            per theta before handing it to NPE. Costs k times as many
            simulations and cuts the noise SD by sqrt(k).

            WHY YOU WILL PROBABLY NEED THIS. Run
            package/validation/noise_screen.py: on the current configuration the
            seed SD of stock_2023 is about 0.35, against a target value of 0.038
            and a theta-induced SD of about 0.40. So a single simulation per
            theta gives NPE a signal-to-noise ratio near 1 on every dimension,
            and it needs an enormous number of simulations to average that out
            across theta space instead of at each theta. Raising seeds_per_theta
            to 8-16 is usually a better use of the same budget.

            A METHODOLOGICAL FORK, worth stating in the paper rather than
            deciding silently. Averaging over seed_inputs treats the NK
            technology landscape as a NUISANCE to marginalise over, so the
            posterior expresses genuine uncertainty about which landscape
            California drew. The alternative view is that the observed
            trajectory IS one landscape realisation, so one should condition on
            a landscape rather than average over it. The existing
            NN_multi_round_calibration_multi_gen.py implicitly takes the second
            position, by pinning seed_inputs at 22 -- but without saying so, and
            while presenting the resulting posterior as if it were marginal.
            Whichever you choose, choose it explicitly.

        target_mask: boolean array over the 24 target dims, or None for all.
            Used by the temporal and moment-type validation splits.
        zip_subset: boolean array over zips, or None for all. Used by the
            spatial and extrapolation validation splits; it restricts BOTH the
            model side and the observed target, including the state aggregates,
            so held-out zips leak nothing.

    Returns:
        dict with the output folder, posterior, best sample, and everything
        package/validation needs to score the fit.
    """
    parameters_list = DEFAULT_PARAMETERS_LIST if parameters_list is None else parameters_list

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    base_params_save = deepcopy(base_params)

    if seed_repetitions is not None:
        base_params["seed_repetitions"] = seed_repetitions
    n_reps = int(base_params["seed_repetitions"])

    bundle = load_object(TARGET_LOAD_ROOT, TARGET_LOAD_NAME)
    spec, zip_table = bundle["spec"], bundle["zip_table"]
    names = bundle["dim_names"]
    n_dims_full = len(names)

    # Rebuild the observed target over the requested zip subset, using the same
    # data_summary() the full target came from. Not a re-slice of x_o: a subset
    # of zips changes the state aggregates and the gradient regression too.
    if zip_subset is None:
        x_o_full = np.asarray(bundle["x_o"], dtype=np.float64)
    else:
        from package.calibration.summary_stats import data_summary
        x_o_full = data_summary(bundle["stock_arrays"], bundle["sales_arrays"],
                                spec, zip_table, zip_subset=zip_subset)
        x_o_full = _sanitize(x_o_full, spec)

    if target_mask is None:
        target_mask = np.ones(n_dims_full, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != (n_dims_full,):
        raise ValueError(f"target_mask must have shape ({n_dims_full},), got {target_mask.shape}")
    if target_mask.sum() == 0:
        raise ValueError("target_mask selects zero dimensions")

    x_o = torch.tensor(x_o_full[target_mask], dtype=torch.float32)
    fitted_names = [n for n, m in zip(names, target_mask) if m]

    seeds_per_theta = max(1, int(seeds_per_theta))
    total_runs = num_rounds * num_simulations * n_reps * seeds_per_theta
    fileName = produce_name_datetime(root)

    if verbose:
        if bundle.get("IS_FAKE_DATA"):
            print("=" * 74)
            print(" FAKE PLACEHOLDER DATA -- this is a pipeline test, not a result")
            print("=" * 74)
        print(f"fileName    : {fileName}")
        print(f"TOTAL RUNS  : {total_runs}  ({num_rounds} rounds x {num_simulations} sims "
              f"x {n_reps} seed reps x {seeds_per_theta} seeds/theta)")
        print(f"parameters  : {[p['name'] for p in parameters_list]}")
        _sl = base_params.get("seed_landscape", base_params.get("seed_inputs", 22))
        print(f"landscape   : {'PINNED at seed ' + str(_sl) if pin_landscape else 'varied (marginalised)'}")
        print(f"fitting {target_mask.sum()}/{n_dims_full} target dims: {fitted_names}")
        if zip_subset is not None:
            print(f"zip subset  : {int(np.sum(zip_subset))}/{len(zip_subset)} zips")

    low = torch.tensor([p["bounds"][0] for p in parameters_list], dtype=torch.float32)
    high = torch.tensor([p["bounds"][1] for p in parameters_list], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    inference = NPE(prior=prior)
    proposal = prior
    posteriors = []
    # n_reps * seeds_per_theta distinct triples, sliced so that repetition j's
    # averaging block never reuses a triple from another repetition
    seed_triples = build_seed_triples(base_params, n_reps * seeds_per_theta,
                                      pin_landscape=pin_landscape)
    blocks = [seed_triples[j * seeds_per_theta:(j + 1) * seeds_per_theta]
              for j in range(n_reps)]

    for i in range(num_rounds):
        if verbose:
            print(f"ROUND {i+1}/{num_rounds}")
        theta_all, x_all = [], []

        for j, block in enumerate(blocks):
            sim = partial(run_single_simulation,
                          base_params=base_params,
                          param_list=parameters_list,
                          spec=spec,
                          zip_table=zip_table,
                          target_mask=target_mask,
                          zip_subset=zip_subset,
                          seed_triple=block[0],
                          extra_triples=tuple(block[1:]))
            sim_checked = process_simulator(sim, prior, is_numpy_simulator=prior_returns_numpy)
            check_sbi_inputs(sim_checked, prior)

            # `seed` here pins sbi's OWN randomness (the theta draw and the
            # per-batch worker seeds), nothing to do with the model's seeds
            # above. It must differ per call or every repetition would draw
            # IDENTICAL thetas.
            theta, x = simulate_for_sbi(
                sim_checked, proposal,
                num_simulations=num_simulations,
                num_workers=get_num_workers(),
                simulation_batch_size=1,
                seed=int(i * 10_000 + j + 1),
            )
            theta_all.append(theta)
            x_all.append(x)

        # Append ONCE per round, not once per repetition: sbi stamps every
        # append_simulations() call carrying a non-prior proposal with
        # max(_data_round_index)+1, so appending inside the loop would make it
        # believe it was on round `n_reps` after round 2. Same reasoning as in
        # NN_multi_round_calibration_multi_gen.
        inference.append_simulations(torch.cat(theta_all), torch.cat(x_all), proposal=proposal)

        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    samples = posterior.sample((100_000,), x=x_o)
    log_prob = posterior.log_prob(samples, x=x_o)
    best_sample = samples[log_prob.argmax()]

    if verbose:
        print("\nposterior mode:")
        for p, v in zip(parameters_list, best_sample):
            print(f"  {p['name']:<10s} {v.item(): .5f}   (prior {p['bounds']})")

    out = {
        "fileName": fileName,
        "posterior": posterior,
        "prior": prior,
        "samples": samples,
        "best_sample": best_sample,
        "parameters_list": parameters_list,
        "base_params": base_params_save,
        "spec": spec,
        "dim_names": names,
        "target_mask": target_mask,
        "zip_subset": zip_subset,
        "x_o_full": x_o_full,
        "x_o_fitted": x_o.numpy(),
        "seed_triples": seed_triples,
        "IS_FAKE_DATA": bundle.get("IS_FAKE_DATA", False),
    }

    createFolder(fileName)
    for key in ("posterior", "prior", "samples", "best_sample", "parameters_list",
                "base_params", "spec", "target_mask", "zip_subset", "x_o_full", "seed_triples"):
        save_object(out[key], fileName + "/Data", key)
    save_object(inference, fileName + "/Data", "inference")

    return out


if __name__ == "__main__":
    main(num_simulations=64, num_rounds=2)
