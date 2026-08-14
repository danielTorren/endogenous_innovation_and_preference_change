"""
NPE calibration with ONE unique ABM seed per draw.

Difference from NN_multi_round_calibration_multi_gen.py
-------------------------------------------------------
That script already draws a unique theta for every simulation: a round is
num_simulations x seed_repetitions = 128 x 64 = 8192 draws, all with distinct
thetas. What it reuses is the SEED. Seeds 1..64 are cycled round-robin, so 128
different thetas share seed 1, 128 share seed 2, and so on.

Here every draw gets its own seed, never repeated within a round or across
rounds. The draw count is set directly rather than being derived from a seed
count, so it is free to be as large as the budget allows.

Why this is worth doing
-----------------------
NPE learns p(x | theta) with the ABM's internal randomness MARGINALISED OUT. In
the round-robin version that marginal is taken over a 64-point empirical
distribution of seeds, the same 64 points at every theta. Any quirk of those
particular 64 draws (an unusually calm seed, an unusually explosive one) is
baked into the learned likelihood at every theta rather than averaging away.
With a fresh seed per draw the marginal is over 8192+ independent draws, which
is the object NPE is meant to be conditioning on.

It is an improvement in kind, not a large one in size. The round-robin estimator
was already unbiased, because theta is drawn independently of the seed index.
Expect a slightly better-calibrated, marginally wider posterior, not a different
answer. The draw-count increase is the change likely to move the posterior more.

The seed that is NOT varied here by default
-------------------------------------------
params["seed"] drives the social network, the firm decisions and the second-hand
market. params["seed_inputs"] drives the landscape draw -- the initial ICE/EV
design pool and the firm starting positions -- via a separate RandomState in
Controller.handle_seed(), and it stays pinned at its base_params value (22) for
every draw, exactly as in the round-robin script.

That pin matters more than the one this file removes: the 2023 EV share moves
far more across seed_inputs than across seed. Pinning it means the posterior is
conditional on one specific landscape rather than marginal over the landscape
uncertainty, which makes it narrower than it should be.

Set vary_seed_inputs=True to marginalise over it too. That is arguably the more
correct calibration, but it is a real change of question, not a tidy-up: x
becomes much noisier at fixed theta and the posterior will widen accordingly.
Left off by default so a run here is directly comparable to the round-robin one.

Everything else -- the summary statistic layout, x_o, the prior, the resampling
of non-finite draws, and the names and folder structure of the saved objects --
is identical to NN_multi_round_calibration_multi_gen.py, which is why
NN_multi_round_calibration_multi_plot.py plots the output of this script
unchanged.
"""

import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
import json
import numpy as np
from copy import deepcopy
from package.resources.utility import (
    produce_name_datetime,
    save_object,
    createFolder,
    load_object,
    get_num_workers
)

# The x layout, the observed-data constants and the single-simulation task are
# imported rather than copied. Both scripts must build x in exactly the same
# order or a posterior trained by one cannot be read by the other's plotting
# code; a second copy of that logic is the obvious way for them to drift apart.
from package.calibration.NN_multi_round_calibration_multi_gen import (
    MATCH_NUM_YEARS,
    AGE_TARGET_RANGE,
    AGE_SCALE,
    MAX_INVALID_RESAMPLE_PASSES,
    run_single_simulation,
    _simulate_batch,
)


def run_single_simulation_seeded(theta, seed, seed_inputs, base_params, param_list):
    """
    One ABM run at (theta, seed, seed_inputs).

    Sets seed_inputs on a copy of base_params and hands the rest to
    run_single_simulation(), which sets params["seed"] and builds x. Costs one
    extra dict deepcopy per task, which is nothing against an ~18 s simulation,
    and keeps the summary-statistic construction defined in exactly one place.
    """
    params = deepcopy(base_params)
    params["seed_inputs"] = int(seed_inputs)
    return run_single_simulation(theta, seed, params, param_list)


class SeedAllocator:
    """
    Hands out ABM seeds that are unique across the whole calibration.

    Reproducible: everything comes from one np.random.default_rng(master_seed),
    so rerunning with the same master_seed replays the same seed sequence. The
    already-issued set is carried across rounds and across resample passes, so a
    seed is never used twice anywhere in the run -- including by a replacement
    draw, which gets a fresh seed rather than inheriting the one that just
    produced a non-finite x.

    Uniqueness is enforced by rejection rather than assumed. Collisions in a
    2^31 range are rare at these counts (~0.06 expected over 16384 draws) but
    they are not impossible, and "unique seed per draw" is the entire point of
    this script.
    """

    # RandomState accepts a seed in [0, 2**32-1]; the range is kept below 2**31
    # only so the values stay comfortably inside int32 wherever they are printed
    # or saved.
    SEED_MAX = 2 ** 31 - 1

    def __init__(self, master_seed):
        self._rng = np.random.default_rng(master_seed)
        self._issued = set()

    def take(self, n):
        """Return n never-before-issued seeds as an int64 array."""
        out = []
        while len(out) < n:
            candidates = self._rng.integers(1, self.SEED_MAX, size=n - len(out))
            for s in candidates:
                s = int(s)
                if s not in self._issued:
                    self._issued.add(s)
                    out.append(s)
        return np.asarray(out, dtype=np.int64)

    @property
    def total_issued(self):
        return len(self._issued)


def simulate_round(proposal, base_params, param_list, num_draws, allocator,
                   torch_seed, vary_seed_inputs):
    """
    Draw `num_draws` thetas from `proposal`, give each its own ABM seed, and
    simulate them all in ONE flat joblib fan-out. Returns (theta, x).

    Same dispatch shape as the round-robin version: all num_draws tasks go into
    a single Parallel() call with batch_size=1, so each worker pulls its next
    draw the moment it finishes one and the ~2x spread in ABM cost across theta
    averages out over the round instead of stranding workers at a barrier.
    """
    # Pins the theta draw only. Both BoxUniform and the round-2+ posterior
    # proposal sample through torch's global RNG. The ABM never touches a global
    # RNG -- it runs off the RandomStates that Controller.handle_seed() builds
    # from params["seed"] / params["seed_inputs"] -- so this is unrelated to the
    # ABM seeds, which come from `allocator`.
    torch.manual_seed(torch_seed)

    theta = proposal.sample((num_draws,))
    theta_np = np.asarray(theta.cpu().numpy(), dtype=np.float64)

    seed_per_draw = allocator.take(num_draws)
    if vary_seed_inputs:
        seed_inputs_per_draw = allocator.take(num_draws)
    else:
        seed_inputs_per_draw = np.full(
            num_draws, int(base_params["seed_inputs"]), dtype=np.int64
        )

    num_workers = get_num_workers()
    print(f"  {num_draws} draws, {num_draws} unique ABM seeds, on {num_workers} "
          f"workers ({num_draws / num_workers:.1f} tasks per worker); "
          f"seed_inputs {'varied per draw' if vary_seed_inputs else 'pinned at ' + str(base_params['seed_inputs'])}")

    x_arr = _simulate_batch_seeded(
        theta_np, seed_per_draw, seed_inputs_per_draw, base_params, param_list, num_workers
    )

    # Non-finite x must never reach append_simulations(): a single NaN row kills
    # inference.train() outright, because NPE-C's atomic loss calls
    # assert_all_finite() on the posterior evaluation.
    #
    # sbi's exclude_invalid_x=True is not the fix -- it defaults to False after
    # round 0 precisely because dropping rows breaks the atomic loss's
    # requirement that retained samples be a fair draw from the proposal.
    # Resampling keeps the slot filled instead.
    #
    # Unlike the round-robin script, the replacement gets a FRESH seed as well
    # as a fresh theta. There is no per-seed draw count to keep balanced here,
    # and reusing the seed risks retrying a seed that fails systematically.
    #
    # The dominant cause of a non-finite x is a month with zero new-car
    # purchases, which makes the EV share of new sales 0/0 (firmManager
    # update_EV_sales appends np.nan in that case). It is strongly increasing in
    # kappa, so a high kappa upper bound raises the resample rate.
    bad = ~np.isfinite(x_arr).all(axis=1)

    for attempt in range(1, MAX_INVALID_RESAMPLE_PASSES + 1):
        if not bad.any():
            break

        idx = np.flatnonzero(bad)
        print(f"  WARNING: {idx.size} of {num_draws} draws produced non-finite x "
              f"({100 * idx.size / num_draws:.2f}%); resampling them "
              f"(pass {attempt}/{MAX_INVALID_RESAMPLE_PASSES})")
        names = [p["name"] for p in param_list]
        for j in idx:
            offending = ", ".join(f"{n}={v:.6g}" for n, v in zip(names, theta_np[j]))
            print(f"    draw {j} (seed {seed_per_draw[j]}): {offending} -> x = {x_arr[j]}")

        replacement = np.asarray(
            proposal.sample((idx.size,)).cpu().numpy(), dtype=np.float64
        )
        replacement_seeds = allocator.take(idx.size)
        if vary_seed_inputs:
            replacement_seed_inputs = allocator.take(idx.size)
        else:
            replacement_seed_inputs = seed_inputs_per_draw[idx]

        replacement_x = _simulate_batch_seeded(
            replacement, replacement_seeds, replacement_seed_inputs,
            base_params, param_list, num_workers
        )

        theta_np[idx] = replacement
        x_arr[idx] = replacement_x
        seed_per_draw[idx] = replacement_seeds
        seed_inputs_per_draw[idx] = replacement_seed_inputs
        bad = ~np.isfinite(x_arr).all(axis=1)

    if bad.any():
        raise RuntimeError(
            f"{int(bad.sum())} draws still produced non-finite x after "
            f"{MAX_INVALID_RESAMPLE_PASSES} resample passes. This is no longer a "
            "rare numerical accident -- a whole region of the proposal is "
            "simulating badly. Investigate before training on it."
        )

    x = torch.as_tensor(x_arr, dtype=torch.float32)
    theta = torch.as_tensor(theta_np, dtype=torch.float32)

    return theta, x, seed_per_draw, seed_inputs_per_draw


def _simulate_batch_seeded(theta_np, seed_per_draw, seed_inputs_per_draw,
                           base_params, param_list, num_workers):
    """
    One flat joblib fan-out over a (theta, seed, seed_inputs) list.

    When seed_inputs is pinned this delegates to the round-robin script's
    _simulate_batch(), so the common case runs through exactly the same code
    path, with no extra deepcopy per task.
    """
    if len(np.unique(seed_inputs_per_draw)) == 1:
        pinned = deepcopy(base_params)
        pinned["seed_inputs"] = int(seed_inputs_per_draw[0])
        return _simulate_batch(theta_np, seed_per_draw, pinned, param_list, num_workers)

    from joblib import Parallel, delayed

    # batch_size=1 defeats joblib's auto-batching, which only groups very short
    # tasks anyway, and makes the one-task-at-a-time queue that gives the load
    # balancing explicit.
    x_list = Parallel(n_jobs=num_workers, batch_size=1, verbose=10)(
        delayed(run_single_simulation_seeded)(
            theta_np[j], seed_per_draw[j], seed_inputs_per_draw[j],
            base_params, param_list
        )
        for j in range(len(theta_np))
    )
    return np.asarray(np.stack(x_list), dtype=np.float64)


def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_draws_per_round=16384,
        num_rounds=2,
        master_seed=20260811,
        vary_seed_inputs=False,
    ) -> str:
    """
    num_draws_per_round is the TOTAL draws in a round, set directly. It is not
    a per-seed count and has no relationship to base_params["seed_repetitions"],
    which this script ignores (nothing in package/model reads it either).

    num_rounds=1 makes this a purely amortised run: every draw comes from the
    prior, and the trained posterior is valid at any x, not just x_o. num_rounds
    >= 2 spends later rounds near x_o, which sharpens the posterior there at the
    cost of that amortisation.

    master_seed pins the ABM seed sequence. Change it to get an independent set
    of seeds; keep it to reproduce a run exactly.

    vary_seed_inputs marginalises over the landscape draw as well as the
    behavioural randomness. See the module docstring -- this widens the
    posterior and changes what is being estimated.
    """

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Snapshot of the launched configuration, saved as base_params.pkl. Nothing
    # here mutates base_params -- run_single_simulation_seeded and
    # _simulate_batch_seeded both work on deepcopies -- but the snapshot is kept
    # as a guard against a future edit reintroducing in-place writes.
    base_params_save = deepcopy(base_params)

    total_runs = num_rounds * num_draws_per_round
    print(f"TOTAL RUNS: {total_runs} "
          f"({num_rounds} rounds x {num_draws_per_round} draws, one unique seed each)")

    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_stock_prop_2020_23 = EV_stock_prop_2010_23[-4:]  # last 4 years only
    EV_sales_prop_2020_23 = calibration_data_output["EV Sales Prop"]  # already just 2020-2023

    root = "sbi_single_seed"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    # Target for the age moment: the midpoint of the published range, as ONE
    # component, matched against the simulation's mean over the same years.
    age_target = np.array([np.mean(AGE_TARGET_RANGE) / AGE_SCALE])

    # 4 stock + 4 sales + 1 age = 9 components.
    # ORDER MATTERS: must match run_single_simulation().
    x_o_data = np.concatenate([
        EV_stock_prop_2020_23,
        EV_sales_prop_2020_23,
        age_target,
    ])
    assert x_o_data.size == 2 * MATCH_NUM_YEARS + 1, (
        f"x_o has {x_o_data.size} components, expected {2 * MATCH_NUM_YEARS + 1}; "
        "the observed EV series and run_single_simulation() must agree"
    )
    x_o = torch.tensor(x_o_data, dtype=torch.float32)
    print("x_o (stock 4, sales 4, age 1):", x_o_data)

    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)

    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    inference = NPE(prior=prior)

    # One allocator for the whole calibration, so no seed is reused between
    # rounds either.
    allocator = SeedAllocator(master_seed)

    posteriors = []
    proposal = prior
    seed_log = []

    for i in range(num_rounds):
        print("ROUND: ", i + 1, "/", num_rounds)

        theta, x, seeds_used, seed_inputs_used = simulate_round(
            proposal,
            base_params,
            parameters_list,
            num_draws=num_draws_per_round,
            allocator=allocator,
            # Pins the theta draw for this round only, and must differ per round
            # or every round would redraw the same thetas. Unrelated to the ABM
            # seeds, which come from `allocator`.
            torch_seed=int(i * 10_000 + 1),
            vary_seed_inputs=vary_seed_inputs,
        )
        seed_log.append({"round": i, "seed": seeds_used, "seed_inputs": seed_inputs_used})

        # Exactly ONE append_simulations() per round. sbi stamps every call
        # carrying a non-prior proposal with max(_data_round_index) + 1, so
        # appending more than once per round makes it believe it is many rounds
        # further along than it is.
        inference.append_simulations(theta, x, proposal=proposal)

        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    print(f"total unique ABM seeds issued: {allocator.total_issued}")

    createFolder(fileName)

    # Same keys as NN_multi_round_calibration_multi_gen so
    # NN_multi_round_calibration_multi_plot reads this run unchanged.
    match_data = {
        "EV_stock_prop_2020_23": EV_stock_prop_2020_23,
        "EV_sales_prop_2020_23": EV_sales_prop_2020_23,
        "extra_scalar_targets": [float(age_target[0])],
        "mean_car_age_range_months": AGE_TARGET_RANGE,
        "age_scale": AGE_SCALE,
    }
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params_save, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")

    # Extra to this script: the exact seeds every draw ran at, so a run can be
    # audited or replayed draw by draw. The plotting script ignores it.
    save_object(seed_log, fileName + "/Data", "seed_log")
    save_object(
        {
            "num_draws_per_round": num_draws_per_round,
            "num_rounds": num_rounds,
            "master_seed": master_seed,
            "vary_seed_inputs": vary_seed_inputs,
        },
        fileName + "/Data",
        "run_config",
    )

    samples = posterior.sample((500000,), x=x_o)
    log_probability_samples = posterior.log_prob(samples, x=x_o)
    max_log_prob_index = log_probability_samples.argmax()
    best_sample = samples[max_log_prob_index]
    print("best_sample", best_sample)
    save_object(samples, fileName + "/Data", "samples")
    save_object(best_sample, fileName + "/Data", "best_sample")
    save_object(inference, fileName + "/Data", "inference")

    return fileName


if __name__ == "__main__":
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [0.8, 5]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [0.8, 5]},
        {"name": "delta", "subdict": "parameters_ICE", "bounds": [0.0015, 0.0033]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [1e-4, 3e-4]},
        {"name": "max_Cost", "subdict": "parameters_EV", "bounds": [80000, 100000]}
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        # TOTAL draws per round, not per seed. At ~18 s per ABM run on 128
        # workers this is ~40 min of simulation per round.
        num_draws_per_round=8192,
        num_rounds=2,
        master_seed=20260811,
        # See the module docstring before turning this on.
        vary_seed_inputs=False,
    )
