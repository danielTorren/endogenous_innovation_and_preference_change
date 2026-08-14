import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
import json
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from package.resources.utility import (
    produce_name_datetime,
    save_object,
    createFolder,
    load_object,
    get_num_workers
)
from package.resources.run import generate_data

MATCH_START_YEAR = 2020
MATCH_END_YEAR = 2023
MATCH_NUM_YEARS = MATCH_END_YEAR - MATCH_START_YEAR + 1
STOCK_MONTH_OFFSET = 3   # APRIL index, matches the EV stock (population) data
SALES_MONTH_OFFSET = 11  # DECEMBER index, matches the EV sales data

# One extra moment beyond EV stock and EV sales: mean fleet age.
#
# It is published as a RANGE (120-144 months, i.e. 10-12 years) but NPE conditions
# on a single observation x_o, so it is matched at the interval midpoint (132) and
# the range is kept only for reporting / post-hoc filtering of posterior samples.
#
# It contributes exactly ONE component to x: the mean over the matched years
# (MATCH_START_YEAR..MATCH_END_YEAR), not one component per year. sbi z-scores x
# per dimension, so relative weight in the summary vector is carried by dimension
# COUNT rather than by scale -- one component keeps age a weak constraint (1 of 9
# dimensions) and leaves the EV stock and EV sales series (4 dimensions each)
# doing most of the calibration. The range is not resolved year by year in the
# source data anyway, so a per-year block would repeat the same target four times
# and quadruple its weight for no new information.
#
# The age component is divided by AGE_SCALE so every component of x is O(1).
#
# New-car HHI is deliberately NOT a calibration moment. Measured on this model,
# unit-share HHI is a near-deterministic function of the number of firms,
# HHI ~= 1.3/J (measured 0.084, 0.101, 0.132, 0.158 at J = 16, 12, 10, 8), and it
# does not respond to max_cars_prod (0.126-0.137 across Y+ = 2..20, less than the
# across-seed spread at fixed Y+). So it identifies J and nothing else, which is
# better handled by setting J directly: the observed 0.11-0.18 (Grieco, Murry &
# Yurukoglu 2024, Figure II panel b, a unit-sales parent-company HHI) implies a
# numbers-equivalent of 1/HHI ~= 7-9 firms, and J = 10 puts the model mid-range.
# firm_manager.history_HHI is still recorded every step for reporting.
AGE_TARGET_RANGE = (120.0, 144.0)
AGE_SCALE = 132.0


def _year_start_index(year, base_params):
    """Index of January of `year` in a monthly series that starts at the burn-in."""
    return (year - 2001) * 12 + base_params["duration_burn_in"]#ADD ON THE BURN IN PERIOD TO THE START


def convert_data(data_to_fit, base_params, month_offset, start_year=MATCH_START_YEAR, end_year=MATCH_END_YEAR):

    # Assuming `data_to_fit` is a numpy array representing monthly data from 2001 onwards.
    # Pull one month's value per year, for the last few years only, since that's
    # all we're calibrating against.
    averages = []

    for year in range(start_year, end_year + 1):
        month_idx = _year_start_index(year, base_params) + month_offset
        averages.append(data_to_fit[month_idx])

    averages_array = np.array(averages)

    return averages_array


def convert_data_annual_mean(data_to_fit, base_params, start_year=MATCH_START_YEAR, end_year=MATCH_END_YEAR):
    """
    Mean over the twelve months of each calendar year, one value per year.

    Used for mean fleet age. Age is a stock, and the published "average age of
    vehicles in operation" is an annual figure, so a 12-month mean is both the
    right comparison and far less seed-noisy than a single monthly snapshot. The
    EV stock and sales blocks do not use this: each is matched at the single month
    its source data refers to (see STOCK_MONTH_OFFSET / SALES_MONTH_OFFSET).
    """
    means = []

    for year in range(start_year, end_year + 1):
        start = _year_start_index(year, base_params)
        means.append(np.mean(data_to_fit[start:start + 12]))

    return np.array(means)


def convert_data_period_mean(annual_values):
    """
    Collapse a per-year array to the single mean over the matched years.

    Used for the mean fleet age moment, which enters x as one component (see the
    AGE comment at the top of this file). Returned as a length-1 array so it
    concatenates with the per-year blocks unchanged.
    """
    return np.array([np.mean(annual_values)])

def run_single_simulation(theta, seed, base_params, param_list):
    """
    Runs one ABM simulation at parameter vector `theta` and ABM seed `seed`.

    One task = one (theta, seed) pair. The seed is applied here rather than
    baked into a per-seed partial in the caller, which is what lets a whole
    round be dispatched as a single flat joblib fan-out (see simulate_round).
    """

    #print("theta", theta)
    # Work on a copy: writing theta straight into the caller's dict leaks back
    # into the parent process whenever joblib runs in-process (num_workers == 1),
    # leaving the last-evaluated theta -- and the whole calibration_data payload
    # that generate_data() attaches -- baked into what main() later pickles as
    # "base_params". At num_workers > 1 each task gets a pickled copy so the
    # parent happened to stay clean, i.e. the bug was invisible on the cluster
    # and only showed up on a single-core run.
    params = deepcopy(base_params)

    # ABM seed for this draw. Controller.handle_seed() turns this into the
    # RandomState driving the social network, firms and the second-hand market.
    params["seed"] = int(seed)

    # Update the parameters from theta
    for i, param in enumerate(param_list):
        subdict = param["subdict"]
        name = param["name"]
        params[subdict][name] = float(theta[i])

    # Run the market simulation
    controller = generate_data(params)

    # Compute summary statistics, all restricted to the last few years of the
    # run: EV stock proportion, EV sales proportion and mean fleet age. All three
    # series are appended once per step from social_network.next_step()/
    # firm_manager.next_step() regardless of save_timeseries_data_state, so they
    # share one index base and the same month-offset convention applies to each.
    arr_history_stock = np.asarray(controller.social_network.history_prop_EV)
    arr_history_sales = np.asarray(controller.firm_manager.history_past_new_bought_vehicles_prop_ev)
    arr_history_age = np.asarray(controller.social_network.history_mean_car_age_fleet)

    stock_data_to_fit = convert_data(arr_history_stock, params, STOCK_MONTH_OFFSET)
    sales_data_to_fit = convert_data(arr_history_sales, params, SALES_MONTH_OFFSET)

    # Age: one component, the mean over the matched years.
    age_data_to_fit = convert_data_period_mean(
        convert_data_annual_mean(arr_history_age, params)
    ) / AGE_SCALE

    # ORDER MATTERS: must match how x_o is assembled in main().
    data_to_fit = np.concatenate([
        stock_data_to_fit,
        sales_data_to_fit,
        age_data_to_fit,
    ])

    return data_to_fit


# How many replacement passes simulate_round() will spend trying to turn a
# non-finite draw into a finite one before giving up and raising. Each pass
# redraws only the offending slots, so a pass is cheap; the cap exists to stop
# a systematically-NaN region of the prior from looping forever.
MAX_INVALID_RESAMPLE_PASSES = 5


def _simulate_batch(theta_np, seed_per_draw, base_params, param_list, num_workers):
    """One flat joblib fan-out over a (theta, seed) list. Returns x as an array."""
    # batch_size=1 defeats joblib's auto-batching. Auto only groups tasks when
    # they are very short, which ABM runs are not, but pinning it makes the
    # dynamic one-task-at-a-time queue that gives the load balancing explicit.
    x_list = Parallel(n_jobs=num_workers, batch_size=1, verbose=10)(
        delayed(run_single_simulation)(
            theta_np[j], seed_per_draw[j], base_params, param_list
        )
        for j in range(len(theta_np))
    )
    return np.asarray(np.stack(x_list), dtype=np.float64)


def simulate_round(proposal, base_params, param_list, num_draws, seeds, torch_seed):
    """
    Draw `num_draws` thetas from `proposal` and simulate all of them in ONE
    flat joblib fan-out. Returns (theta, x) ready for append_simulations().

    Replaces the old per-seed loop over sbi's simulate_for_sbi(). That version
    split a round into `len(seeds)` sequential waves of `num_simulations` tasks
    each, and since --cpus-per-task was set equal to num_simulations every wave
    ran exactly one task per worker with nothing queued behind it. A wave could
    therefore not finish before its SLOWEST draw did, and ABM cost varies ~2x
    with theta, so ~25% of the allocation sat idle -- once per wave, 64 waves
    per round.

    Here all num_draws tasks go into a single Parallel() call, so with
    num_draws >> n_jobs each worker pulls its next draw the moment it finishes
    one and the runtime spread averages out across the whole round. There is
    one barrier per round instead of one per seed.

    Statistically this is identical to the old loop: those per-seed batches were
    all drawn from the same `proposal`, so they were already one round of draws
    from one distribution (which is why append_simulations() was called once per
    round on the concatenation). Every draw still gets a unique theta run at
    exactly one seed; there is no replication of a theta across seeds.

    Seeds are assigned round-robin (draw j gets seeds[j % len(seeds)]), so with
    num_draws = num_simulations x len(seeds) every seed carries exactly
    num_simulations draws, the same split as the old per-seed loop. theta is
    drawn independently of j, so round-robin induces no correlation between
    theta and seed. What has changed is only that the seed no longer gates
    DISPATCH: all num_draws tasks are in flight as one pool, rather than being
    forced through len(seeds) sequential barriers.
    """
    # Pins the theta draw only. Both the prior (BoxUniform) and the round-2+
    # proposal (a posterior with default x set) sample through torch's global
    # RNG, and the ABM itself never touches a global RNG -- it runs off the
    # np.random.RandomState objects that Controller.handle_seed() builds from
    # params["seed"] / params["seed_inputs"]. So seeding torch here is enough
    # to make a round reproducible, and it is unrelated to the ABM seeds.
    torch.manual_seed(torch_seed)

    theta = proposal.sample((num_draws,))
    theta_np = np.asarray(theta.cpu().numpy(), dtype=np.float64)

    # np.resize tiles cyclically: seeds 1..64 repeated until num_draws is filled.
    seed_per_draw = np.resize(np.asarray(seeds), num_draws)

    num_workers = get_num_workers()
    print(f"  {num_draws} draws over {len(seeds)} seeds on {num_workers} workers "
          f"({num_draws / num_workers:.1f} tasks per worker)")

    x_arr = _simulate_batch(theta_np, seed_per_draw, base_params, param_list, num_workers)

    # Non-finite x must never reach append_simulations(). A single NaN row kills
    # inference.train() outright on the multi-round path: NPE-C's atomic loss
    # calls assert_all_finite() on the posterior evaluation, so it raises
    # "NaN/Inf present in posterior eval" rather than degrading gracefully.
    #
    # sbi's own lever, exclude_invalid_x=True, is NOT the fix here. It defaults
    # to True in round 0 and False afterwards precisely because, in its words,
    # "for multi-round SNPE (atomic), discarding invalid simulations gives
    # systematically wrong results" -- the atomic loss needs the retained
    # samples to be a fair draw from the proposal, and dropping rows breaks that
    # (as well as unbalancing the per-seed counts).
    #
    # Resampling instead keeps both invariants: the slot stays filled, and it
    # keeps its original ABM seed so every seed still carries exactly
    # num_simulations draws. Formally this makes those slots draws from the
    # proposal truncated to the region that simulates finitely, which is a
    # negligible distortion at the observed rate (1 bad draw in 8192) and is
    # strictly better than the alternatives of training on NaN or dropping rows.
    bad = ~np.isfinite(x_arr).all(axis=1)

    for attempt in range(1, MAX_INVALID_RESAMPLE_PASSES + 1):
        if not bad.any():
            break

        idx = np.flatnonzero(bad)
        print(f"  WARNING: {idx.size} of {num_draws} draws produced non-finite x; "
              f"resampling them (pass {attempt}/{MAX_INVALID_RESAMPLE_PASSES})")
        for j in idx:
            names = [p["name"] for p in param_list]
            offending = ", ".join(f"{n}={v:.6g}" for n, v in zip(names, theta_np[j]))
            print(f"    draw {j} (seed {seed_per_draw[j]}): {offending} -> x = {x_arr[j]}")

        replacement = np.asarray(
            proposal.sample((idx.size,)).cpu().numpy(), dtype=np.float64
        )
        replacement_x = _simulate_batch(
            replacement, seed_per_draw[idx], base_params, param_list, num_workers
        )

        theta_np[idx] = replacement
        x_arr[idx] = replacement_x
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

    return theta, x


def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=128,
        num_rounds = 3
    ) -> str:
    """
    num_simulations is the number of draws PER SEED, so a round generates
    num_simulations x base_params["seed_repetitions"] (theta, x) training pairs
    and each of the seeds carries exactly num_simulations of them.

    Note this is a per-seed count, not a per-round one: it is the "this many
    draws for each of my seeds" knob. It no longer has any relationship to
    --cpus-per-task, because a round is dispatched as one flat fan-out of all
    num_simulations x seed_repetitions tasks (see simulate_round).
    """

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Snapshot of the launched configuration, saved as base_params.pkl. The ABM
    # seed is now set per task inside run_single_simulation() on its own deepcopy,
    # so this dict is never mutated, but the snapshot is kept as a guard against
    # a future edit reintroducing in-place writes.
    base_params_save = deepcopy(base_params)

    seeds = np.arange(1, base_params["seed_repetitions"] + 1)

    # Total draws in one round. Divides exactly by len(seeds) by construction,
    # so the round-robin seed assignment in simulate_round() is perfectly
    # balanced: num_simulations draws for every seed.
    num_draws_per_round = num_simulations * len(seeds)

    total_runs = num_rounds * num_draws_per_round
    print(f"TOTAL RUNS: {total_runs} "
          f"({num_rounds} rounds x {len(seeds)} seeds x {num_simulations} draws per seed)")

    # Load observed data
    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_stock_prop_2020_23 = EV_stock_prop_2010_23[-4:]  # last 4 years only
    EV_sales_prop_2020_23 = calibration_data_output["EV Sales Prop"]  # already just 2020-2023

    root = "NN_calibration_multi"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    # Target for the age moment: the midpoint of the published range, as ONE
    # component, matched against the simulation's mean over the same years
    # (convert_data_period_mean). A flat target needs no per-year resolution, and
    # one component keeps the moment a weak constraint -- see the AGE comment at
    # the top of this file.
    age_target = np.array([np.mean(AGE_TARGET_RANGE) / AGE_SCALE])

    # Observed data: EV stock proportion and EV sales proportion, one value per
    # year for the last 4 years, then mean fleet age (scaled), one value.
    # 4 + 4 + 1 = 9 components.
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

    # Define the prior
    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)

    # Process the prior
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Initialize inference object
    inference = NPE(prior=prior)

    posteriors = []
    proposal = prior

    for i in range(num_rounds):
        print("ROUND: ", i+1, "/", num_rounds)

        # One flat fan-out for the whole round: num_draws_per_round unique
        # thetas, ABM seeds cycled round-robin across them. See simulate_round().
        theta, x = simulate_round(
            proposal,
            base_params,
            parameters_list,
            num_draws=num_draws_per_round,
            seeds=seeds,
            # Pins the theta draw for this round only, and must differ per round
            # or every round would redraw the same thetas. Unrelated to the ABM
            # seeds, which come from `seeds`.
            torch_seed=int(i * 10_000 + 1),
        )

        # Exactly ONE append_simulations() per round. sbi stamps every call that
        # carries a non-prior proposal with max(_data_round_index) + 1, so
        # appending per seed-batch (as the old inner loop invited) made it
        # believe it was on round 64 after round 2. One call per round keeps
        # _data_round_index at [0, 1, 2], which is what discard_prior_samples
        # and the non-atomic MDN losses branch on if they are ever switched on.
        inference.append_simulations(theta, x, proposal=proposal)

        # Train the density estimator on everything collected so far
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    createFolder(fileName)

    # Save results
    match_data = {
        "EV_stock_prop_2020_23": EV_stock_prop_2020_23,
        "EV_sales_prop_2020_23": EV_sales_prop_2020_23,
        # The scalar components actually fed to NPE, in x order after the two EV
        # blocks. Named so the plotting script can rebuild x_o from match_data
        # alone (see NN_multi_round_calibration_multi_plot.reconstruct_x_o).
        "extra_scalar_targets": [float(age_target[0])],
        # The range, not just the midpoint fed to NPE: kept so posterior samples
        # can be checked against the whole interval afterwards.
        "mean_car_age_range_months": AGE_TARGET_RANGE,
        "age_scale": AGE_SCALE,
    }
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params_save, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")
    
    samples = posterior.sample((500000,), x=x_o)
    log_probability_samples = posterior.log_prob(samples, x=x_o)
    max_log_prob_index = log_probability_samples.argmax()
    best_sample = samples[max_log_prob_index]
    print("best_sample", best_sample)
    save_object(samples, fileName + "/Data", "samples")
    save_object(best_sample, fileName + "/Data", "best_sample")
    save_object(inference, fileName + "/Data", "inference")


if __name__ == "__main__":
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [0.8, 5]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [0.8, 5]},
        {"name": "delta", "subdict": "parameters_ICE", "bounds": [0.0015, 0.0033]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [1e-4, 3e-4]}
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        # Draws PER SEED. With seed_repetitions = 64 this is 8192 draws per
        # round. Raise it freely: it no longer has to relate to --cpus-per-task.
        num_simulations=128,
        num_rounds= 2
    )