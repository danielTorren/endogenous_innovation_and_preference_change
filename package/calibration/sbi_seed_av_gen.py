"""
NPE calibration on SEED-AVERAGED, LOGIT-SCALED EV stock, with sales dropped.

Difference from sbi_single_seed_gen.py / NN_multi_round_calibration_multi_gen.py
--------------------------------------------------------------------------------
Those scripts train on one ABM run per theta. This one runs every theta at
NUM_SEEDS_PER_THETA seeds and trains on the average, and it changes what goes
into x:

    x = [logit(stock 2020..2023), mean fleet age / AGE_SCALE]      5 components
        (was: stock 2020..23, sales 2020..23, age               -- 9 components)

Three changes, each with its own justification below.


1. SEED AVERAGING -- why
------------------------
Measured on NN_calibration_multi_12_34_29__14_08_2026 (16,384 draws) against the
64-seed run of its posterior mode (calibration_gen_18_04_28__14_08_2026), the
share of each x component's variance that is pure ABM seed noise rather than
response to theta:

    stock 2020   67%        sales 2020  152%        age   1%
    stock 2023   91%        sales 2023  103%

A component whose seed noise exceeds its across-prior spread carries no
information about theta at all, which is what "152%" means -- the two sales
columns were noise. NPE was being asked to condition on them anyway.

Averaging K runs at the same theta divides the noise VARIANCE by K while leaving
the theta response untouched. The averaging happens AFTER the logit transform
(see below), and in logit space the noise is far less dominant than the raw-space
numbers above suggest:

    dim         seed sd   theta sd    K for noise:signal = 1:3
    stock2020     0.880      1.352            3.8
    stock2023     0.962      1.597            3.3

so K = 8 already puts seed noise at roughly a twentieth of the signal variance.
That is where DEFAULT_NUM_SEEDS_PER_THETA comes from. Raising it further buys
little; the cost is strictly linear in it.

The budget trade this forces is thetas-per-round against seeds-per-theta at
fixed total runs. Do not push K much higher without also raising
num_thetas_per_round -- NPE needs enough distinct thetas to learn a conditional
density, and K = 32 on the old budget would leave only 512 of them.


2. LOGIT TRANSFORM -- why
-------------------------
sbi standardises every x component before the density estimator sees it
(z_score_x='independent' is the default), so what matters is how many z-units
wide the region you care about is. On the raw stock columns it is almost none:

    dim        skew   z(obs)   width of [0.5x obs, 2x obs] in z-units
    stock20    6.48    -0.18                0.35
    stock23    4.32    -0.05                0.63

A factor-of-four range around the observation occupies a third of one standard
deviation, because 78% of draws sit below 5% of the column's max while the tail
runs out to 24x the observation. After logit:

    stock20    0.68     0.36                0.99      (2.8x more resolution)
    stock23    0.59     0.70                0.85

Skew collapses and the observation lands in the body of the distribution instead
of on top of the spike. log(x + eps) does nearly as well on stock but is worse on
anything with mass at zero, and logit is the right function for a proportion
anyway since it respects the upper bound at 1 too.

Note this does NOT decorrelate the four stock years -- they are 1.01 effective
dimensions after the transform, versus 1.02 before. The transform is about
conditioning the estimator's input, not about adding information.


3. SALES DROPPED -- why
-----------------------
The correlation matrix of the eight EV components has eigenvalues
[7.21, 0.45, 0.14, 0.11, 0.09, 0.01, 0, 0] -- 1.23 effective dimensions out of
eight. The four stock years correlate 0.989 with each other and 0.85 with sales,
so the whole block was already close to a single number, and the sales half of it
was the half with no signal-to-noise (see 1). Removing it costs roughly 0.2
effective dimensions and removes four columns of noise.

Age is KEPT, and it is not a sanity check -- it is the second identifying
direction. Regressing each moment on theta and comparing gradient directions:

    stock level  ->  a_chi -0.75   b_chi +0.46   delta +0.21   kappa +0.42
    age          ->  a_chi +0.18   b_chi -0.06   delta -0.79   kappa -0.58
                     angle between them: 125 degrees

Stock reads a_chi/b_chi, age reads delta/kappa. Two moments 125 degrees apart are
measuring genuinely different things. (EV sales, and log-growth in stock, both sit
within 20 degrees of the stock level direction -- which is why neither earns its
place in x.)


Why kappa is fixed rather than fitted
-------------------------------------
Stacking [stock level, age] into a Jacobian against all four parameters leaves an
unconstrained direction, a_chi +0.56, b_chi +0.59, delta -0.29, kappa +0.50: move
theta along it and every moment stays put. kappa carries a large loading on it,
which is why its posterior has come back at 0.99x the prior width in every run
made so far -- not because nothing responds to kappa (stock and age both do) but
because its effect can always be undone by the other three.

So kappa is left out of parameters_list and three parameters are fitted; its value
comes from BASE_PARAMS_LOAD like any other uncalibrated parameter, and whatever
that file holds is recorded in base_params.pkl. Pinning it at 0.00018 costs almost
nothing: on seed-averaged x the best achievable stock fit with kappa free is
1.09x, and restricted to kappa in [1.7e-4, 1.9e-4] it is 1.12x (and on the
rho=0.5 / max_Cost=90000 model the unrestricted optimum IS 1.84e-4).

Everything else -- the flat joblib fan-out, the unique-seed allocator, the
resample-don't-drop policy for non-finite draws, and the names of the saved
objects -- follows the two existing scripts. NN_multi_round_calibration_multi_plot
reads this run unchanged: it prefers the saved x_o tensor over rebuilding one from
match_data, and takes parameter names and bounds from var_dict.
"""

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

# The observed-data constants, the year/month conventions and the per-year
# extraction helpers are imported rather than copied, so this script and the
# other two cannot drift on which month of which year they read.
from package.calibration.NN_multi_round_calibration_multi_gen import (
    MATCH_START_YEAR,
    MATCH_END_YEAR,
    MATCH_NUM_YEARS,
    STOCK_MONTH_OFFSET,
    AGE_TARGET_RANGE,
    AGE_SCALE,
    MAX_INVALID_RESAMPLE_PASSES,
    convert_data,
    convert_data_annual_mean,
    convert_data_period_mean,
)
from package.calibration.sbi_single_seed_gen import SeedAllocator


# Clip applied inside logit(). The model expresses EV stock as a share of
# num_individuals = 3000 agents, so the smallest NON-ZERO value it can produce is
# 1/3000 = 3.3e-4. A floor below that clips no real value: it only catches exact
# zeros (2.4-2.7% of draws over the full prior, in dead regions where nobody ever
# considers an EV) and maps them to a sentinel at logit(1e-4) = -9.21, just
# outside logit(1/3000) = -8.0. Dead stays distinguishable from barely-alive
# without the -inf.
LOGIT_EPS = 1e-4

# Runs per theta. See section 1 of the module docstring -- K=8 puts seed noise at
# roughly a twentieth of the signal variance in logit space. Cost is linear in it.
DEFAULT_NUM_SEEDS_PER_THETA = 8

# A theta is resampled if fewer than this fraction of its seeds returned a finite
# x. Below the threshold the surviving seeds are too few to average; at or above
# it, the average is taken over the finite ones only. Dropping the EV sales
# component removes the dominant source of non-finite x (a month with zero new-car
# purchases makes the EV share of new sales 0/0, which firmManager records as
# NaN), so this should almost never fire.
MIN_FINITE_SEED_FRACTION = 0.5

# Number of RAW moments each simulation returns, before seed averaging and before
# the logit: 4 stock years + 1 age.
NUM_RAW_MOMENTS = MATCH_NUM_YEARS + 1


def logit(p, eps=LOGIT_EPS):
    """
    Log-odds of a proportion, clipped away from 0 and 1.

    Applied to EV stock shares only. Age is already an O(1) scaled quantity with
    a near-symmetric distribution (skew about 0.3) and is left alone -- it needs
    neither the variance stabilisation nor the bounded-support handling.
    """
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def run_single_simulation_raw(theta, seed, base_params, param_list):
    """
    One ABM run at (theta, seed). Returns the RAW, untransformed moments.

    Deliberately returns raw proportions rather than the finished x: the logit is
    applied after averaging across seeds (see build_x), so it must not be applied
    here as well. Keeping the worker's job to "run the model and read four stock
    values and an age off it" also means the transform lives in exactly one place.

    Returns:
        np.ndarray, shape (NUM_RAW_MOMENTS,): EV stock share in April of
        2020..2023, then mean fleet age over the matched years divided by
        AGE_SCALE.
    """
    # Work on a copy. Writing theta straight into the caller's dict leaks back
    # into the parent whenever joblib runs in-process (num_workers == 1), which
    # bakes the last-evaluated theta -- and the whole calibration_data payload
    # generate_data() attaches -- into what main() later pickles as base_params.
    params = deepcopy(base_params)

    # ABM seed for this draw. Controller.handle_seed() turns this into the
    # RandomState driving the social network, firms and the second-hand market.
    params["seed"] = int(seed)

    for i, param in enumerate(param_list):
        params[param["subdict"]][param["name"]] = float(theta[i])

    controller = generate_data(params)

    # Both series are appended once per step from social_network.next_step()
    # regardless of save_timeseries_data_state, so they share one index base and
    # the month-offset convention applies to each.
    arr_history_stock = np.asarray(controller.social_network.history_prop_EV)
    arr_history_age = np.asarray(controller.social_network.history_mean_car_age_fleet)

    stock_raw = convert_data(arr_history_stock, params, STOCK_MONTH_OFFSET)

    # Age: one component, the mean over the matched years. A 12-month mean rather
    # than a monthly snapshot because the published figure is annual and because
    # it is far less seed-noisy; the stock block keeps its single-month reads
    # because each refers to a specific month in its source data.
    age_raw = convert_data_period_mean(
        convert_data_annual_mean(arr_history_age, params)
    ) / AGE_SCALE

    return np.concatenate([stock_raw, age_raw])


def _simulate_batch_raw(theta_np, seed_matrix, base_params, param_list, num_workers):
    """
    One flat joblib fan-out over every (theta, seed) pair.

    seed_matrix is (num_thetas, num_seeds). It is flattened so that all
    num_thetas * num_seeds tasks go into a SINGLE Parallel() call rather than one
    call per seed-wave. ABM cost varies about 2x with theta, so a per-wave barrier
    would strand workers on the slowest draw of every wave; with one deep queue
    each worker pulls its next task the moment it finishes one and the spread
    averages out over the whole round.

    batch_size=1 defeats joblib's auto-batching, which only groups very short
    tasks anyway, and makes that one-task-at-a-time queue explicit.

    Returns:
        np.ndarray, shape (num_thetas, num_seeds, NUM_RAW_MOMENTS).
    """
    num_thetas, num_seeds = seed_matrix.shape
    flat_theta_idx = np.repeat(np.arange(num_thetas), num_seeds)
    flat_seeds = seed_matrix.reshape(-1)

    x_list = Parallel(n_jobs=num_workers, batch_size=1, verbose=10)(
        delayed(run_single_simulation_raw)(
            theta_np[flat_theta_idx[t]], flat_seeds[t], base_params, param_list
        )
        for t in range(len(flat_seeds))
    )

    x_raw = np.asarray(np.stack(x_list), dtype=np.float64)
    return x_raw.reshape(num_thetas, num_seeds, NUM_RAW_MOMENTS)


def build_x(x_raw, central="mean"):
    """
    Collapse (num_thetas, num_seeds, NUM_RAW_MOMENTS) of raw moments into x.

    ORDER OF OPERATIONS MATTERS. The logit is applied per seed and the average is
    taken in logit space, NOT the other way round:

        x_stock = mean_over_seeds( logit(stock_s) )
        x_age   = mean_over_seeds( age_s )

    Averaging raw proportions first would make the result an estimate of
    E[stock | theta], an arithmetic mean over a distribution with skew above 4 --
    dominated by the handful of seeds where EV adoption goes explosive (at the
    posterior mode of the earlier calibration, mean 0.054 against median 0.040).
    Averaging in logit space instead gives a central tendency on the log-odds
    scale, which is the right notion of "typical" for a multiplicative, heavily
    right-skewed quantity, and it is the estimator whose noise the K in the module
    docstring was calculated for. Matching it against logit(observed) asks "for
    what theta is the observed share a typical outcome", which is also the more
    defensible question given the observation is a single realisation.

    Args:
        x_raw: raw per-seed moments; NaN/inf entries are tolerated and excluded.
        central: 'mean' (default) or 'median'. The logit already tames the tail,
            so the mean is both adequate and more efficient; 'median' is there for
            a fully non-parametric check and, being a monotone transform away, is
            identical to taking the median of the raw proportions.

    Returns:
        (x, n_finite, x_sd) -- x of shape (num_thetas, NUM_RAW_MOMENTS), the count
        of finite seeds per theta of shape (num_thetas,), and the ACROSS-SEED sd of
        the transformed per-seed moments, same shape as x.

        x_sd is what makes the seed averaging auditable, and it is cheap: x records
        only the average, so a finished run holds no record of how much the K seeds
        disagreed with each other. Without it the leftover scatter measured off
        (theta, x) alone confounds two things -- noise that averaging removes (a
        fresh params["seed"] per run) and noise it does not (theta perturbations
        reshuffling the pinned seed_inputs landscape, because RandomState.beta's
        rejection sampler consumes a theta-dependent number of uniforms and shifts
        every draw made after it off the same stream). x_sd / sqrt(K) is the first
        of those, so saving it separates them and says directly whether K is set
        higher than it needs to be. See simulate_round(), which prints the budget.
    """
    stock = logit(x_raw[:, :, :MATCH_NUM_YEARS])
    age = x_raw[:, :, MATCH_NUM_YEARS:]
    transformed = np.concatenate([stock, age], axis=2)

    # A seed counts as finite only if ALL of its moments are, so a partially
    # broken run never contributes to some components and not others.
    finite = np.isfinite(x_raw).all(axis=2)
    n_finite = finite.sum(axis=1)

    # np.nanmean over the masked array: seeds that failed are excluded per theta
    # rather than poisoning the average. Thetas with n_finite == 0 come out NaN
    # and are caught by the resample loop in simulate_round.
    masked = np.where(finite[:, :, None], transformed, np.nan)

    with np.errstate(invalid="ignore"):
        if central == "median":
            x = np.nanmedian(masked, axis=1)
        elif central == "mean":
            x = np.nanmean(masked, axis=1)
        else:
            raise ValueError(f"central must be 'mean' or 'median', got {central!r}")

        # ddof=1 because the K seeds are a sample, not the population. A theta with
        # one finite seed gives NaN rather than 0, which is the honest answer to
        # "how much did the seeds disagree" when there is only one of them; those
        # thetas are excluded from the medians simulate_round() prints.
        x_sd = np.nanstd(masked, axis=1, ddof=1)

    return x, n_finite, x_sd


def simulate_round(proposal, base_params, param_list, num_thetas, num_seeds,
                   allocator, torch_seed, central):
    """
    Draw `num_thetas` thetas from `proposal`, run each at `num_seeds` unique ABM
    seeds, and return the seed-averaged (theta, x) ready for append_simulations().

    Every seed is unique across the whole calibration -- no theta shares a seed
    with any other theta, and no seed is reused between rounds. That matters more
    here than it did in the round-robin script: if all thetas shared one set of K
    seeds, any quirk of those particular K draws would be common to every training
    point and would not average away across the round.

    Returns:
        (theta, x, seed_matrix)
    """
    # Pins the theta draw only. Both BoxUniform and the round-2+ posterior
    # proposal sample through torch's global RNG; the ABM never touches a global
    # RNG, it runs off the RandomStates Controller.handle_seed() builds from
    # params["seed"] / params["seed_inputs"]. So this is unrelated to the ABM
    # seeds, which come from `allocator`.
    torch.manual_seed(torch_seed)

    theta = proposal.sample((num_thetas,))
    theta_np = np.asarray(theta.cpu().numpy(), dtype=np.float64)

    seed_matrix = allocator.take(num_thetas * num_seeds).reshape(num_thetas, num_seeds)

    num_workers = get_num_workers()
    total = num_thetas * num_seeds
    print(f"  {num_thetas} thetas x {num_seeds} seeds = {total} runs on "
          f"{num_workers} workers ({total / num_workers:.1f} tasks per worker); "
          f"seed_inputs pinned at {base_params['seed_inputs']}")

    x_raw = _simulate_batch_raw(theta_np, seed_matrix, base_params, param_list, num_workers)
    x_arr, n_finite, x_sd = build_x(x_raw, central=central)

    # A theta survives if enough of its seeds were finite; the average is then
    # taken over those. Only thetas that fall below the threshold are resampled.
    #
    # Resampling rather than dropping: sbi's exclude_invalid_x defaults to False
    # after round 0 precisely because dropping rows breaks the atomic NPE-C loss's
    # requirement that retained samples be a fair draw from the proposal. Keeping
    # the slot filled preserves that. Formally the refilled slots are draws from
    # the proposal truncated to the region that simulates finitely, a negligible
    # distortion at the rates observed here.
    bad = (n_finite < MIN_FINITE_SEED_FRACTION * num_seeds) | ~np.isfinite(x_arr).all(axis=1)

    partial = (n_finite < num_seeds) & ~bad
    if partial.any():
        print(f"  note: {int(partial.sum())} thetas averaged over a reduced seed set "
              f"(min {int(n_finite[partial].min())} of {num_seeds} finite)")

    for attempt in range(1, MAX_INVALID_RESAMPLE_PASSES + 1):
        if not bad.any():
            break

        idx = np.flatnonzero(bad)
        print(f"  WARNING: {idx.size} of {num_thetas} thetas had fewer than "
              f"{MIN_FINITE_SEED_FRACTION:.0%} finite seeds; resampling them "
              f"(pass {attempt}/{MAX_INVALID_RESAMPLE_PASSES})")
        names = [p["name"] for p in param_list]
        for j in idx:
            offending = ", ".join(f"{n}={v:.6g}" for n, v in zip(names, theta_np[j]))
            print(f"    theta {j}: {offending} -> {int(n_finite[j])}/{num_seeds} finite")

        replacement = np.asarray(
            proposal.sample((idx.size,)).cpu().numpy(), dtype=np.float64
        )
        # Fresh seeds as well as a fresh theta: there is no per-seed draw count to
        # keep balanced here, and reusing a seed risks retrying one that fails
        # systematically.
        replacement_seeds = allocator.take(idx.size * num_seeds).reshape(idx.size, num_seeds)
        replacement_raw = _simulate_batch_raw(
            replacement, replacement_seeds, base_params, param_list, num_workers
        )
        replacement_x, replacement_n, replacement_sd = build_x(
            replacement_raw, central=central
        )

        theta_np[idx] = replacement
        x_arr[idx] = replacement_x
        n_finite[idx] = replacement_n
        x_sd[idx] = replacement_sd
        seed_matrix[idx] = replacement_seeds
        bad = (n_finite < MIN_FINITE_SEED_FRACTION * num_seeds) | ~np.isfinite(x_arr).all(axis=1)

    if bad.any():
        raise RuntimeError(
            f"{int(bad.sum())} thetas still failed after "
            f"{MAX_INVALID_RESAMPLE_PASSES} resample passes. This is no longer a "
            "rare numerical accident -- a whole region of the proposal is "
            "simulating badly. Investigate before training on it."
        )

    # The number that decides whether num_seeds is set sensibly, printed rather
    # than left in the pickle because it should be read while the run is still
    # cheap to abandon.
    #
    # per_seed  is how much one run wobbles at fixed theta -- the thing averaging
    #           is meant to kill.
    # of_mean   is per_seed / sqrt(K): what survives the averaging.
    # theta_sd  is the spread of x ACROSS thetas: the signal NPE learns from.
    #
    # of_mean well below theta_sd means the averaging has done its job and K is at
    # or above what it needs to be; raising it further only trades away thetas.
    # Note of_mean is a floor on the leftover scatter, not the whole of it: any
    # excess is theta-coupled and no value of K removes it (see build_x).
    with np.errstate(invalid="ignore"):
        per_seed = np.nanmedian(x_sd, axis=0)
    theta_sd = x_arr.std(axis=0)
    print("  seed-noise budget, median over thetas:")
    for i, (ps, ts) in enumerate(zip(per_seed, theta_sd)):
        of_mean = ps / np.sqrt(num_seeds)
        print(f"    moment {i}: per_seed {ps:.4f}  of_mean {of_mean:.4f}  "
              f"theta_sd {ts:.4f}  of_mean/theta_sd {of_mean / ts:.4f}")

    x = torch.as_tensor(x_arr, dtype=torch.float32)
    theta = torch.as_tensor(theta_np, dtype=torch.float32)

    return theta, x, seed_matrix, x_sd


def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_thetas_per_round=4096,
        num_seeds_per_theta=DEFAULT_NUM_SEEDS_PER_THETA,
        num_rounds=2,
        master_seed=20260816,
        central="mean",
    ) -> str:
    """
    num_thetas_per_round is the number of DISTINCT thetas in a round; the round
    costs num_thetas_per_round x num_seeds_per_theta ABM runs. Both knobs are set
    directly and neither has any relationship to base_params["seed_repetitions"],
    which this script ignores (nothing in package/model reads it either).

    The default 4096 x 8 = 32,768 runs per round is the same simulation budget as
    sbi_single_seed_gen's default, spent on 4096 well-measured thetas instead of
    32,768 noisy ones. At the ~18 s per run and 128 workers measured on the
    cluster that is ~80 min per round.

    Any parameter absent from parameters_list is held at its BASE_PARAMS_LOAD
    value for every draw, and base_params.pkl records what was used. kappa is
    absent on purpose -- see the module docstring on why it is pinned rather than
    fitted -- so set it in the JSON before launching.

    num_rounds=1 makes this a purely amortised run: every theta comes from the
    prior and the trained posterior is valid at any x. num_rounds >= 2 spends
    later rounds near x_o, sharpening the posterior there at the cost of that
    amortisation.

    central selects how the K seeds are combined, in logit space -- see build_x().
    """

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Parameters this script deliberately PINS rather than fits. Each is whatever
    # the JSON says, so a silently stale value shifts the whole posterior and the
    # run is conditional on it without saying so. Printed for that reason, and
    # cross-checked against parameters_list so a name cannot appear in both.
    #
    # b_chi joined kappa here: with only [stock level, age] as moments, a_chi and
    # b_chi push the stock level in opposite directions along a straight line
    # through the prior box, and every b_chi in [2, 5] fits equally well once a_chi
    # moves with it. b_chi is the weaker half of that pair by 3.4x per prior width,
    # so fitting it in place of a_chi would return a posterior WIDER than its prior
    # (1.56x). Fixing it costs a_chi nothing: 0.499 prior-widths fitted against
    # 0.490 pinned.
    PINNED = [
        ("parameters_social_network", "b_chi"),
        ("parameters_vehicle_user", "kappa"),
    ]
    fitted_names = {p["name"] for p in parameters_list}
    print(f"NOT calibrated, taken verbatim from {BASE_PARAMS_LOAD}:")
    for subdict, name in PINNED:
        if name in fitted_names:
            raise ValueError(
                f"{name} is in parameters_list AND in PINNED. It cannot be both: "
                "the draw would overwrite the JSON value for every theta while the "
                "log claimed it was held fixed. Remove it from one of them."
            )
        print(f"    {subdict}.{name} = {base_params[subdict][name]}")

    # Nothing below mutates base_params -- run_single_simulation_raw works on a
    # deepcopy -- but the snapshot is kept as a guard against a future edit
    # reintroducing in-place writes.
    base_params_save = deepcopy(base_params)

    total_runs = num_rounds * num_thetas_per_round * num_seeds_per_theta
    print(f"TOTAL RUNS: {total_runs} ({num_rounds} rounds x "
          f"{num_thetas_per_round} thetas x {num_seeds_per_theta} seeds)")
    print(f"training pairs: {num_rounds * num_thetas_per_round} "
          f"({num_thetas_per_round} per round)")

    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_stock_prop_2020_23 = np.asarray(EV_stock_prop_2010_23[-4:])  # last 4 years only

    root = "sbi_seed_av"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    # Target for the age moment: the midpoint of the published 120-144 month
    # range, as ONE component, matched against the simulation's mean over the same
    # years. The range itself is saved for post-hoc filtering of posterior samples.
    age_target = np.array([np.mean(AGE_TARGET_RANGE) / AGE_SCALE])

    # x_o on exactly the scale the simulations are put on in build_x(): the stock
    # block through the same logit, age untransformed.
    # ORDER MATTERS: must match run_single_simulation_raw() / build_x().
    x_o_data = np.concatenate([
        logit(EV_stock_prop_2020_23),
        age_target,
    ])
    assert x_o_data.size == NUM_RAW_MOMENTS, (
        f"x_o has {x_o_data.size} components, expected {NUM_RAW_MOMENTS}; "
        "the observed EV stock series and run_single_simulation_raw() must agree"
    )
    x_o = torch.tensor(x_o_data, dtype=torch.float32)
    print("observed EV stock 2020-23:", np.round(EV_stock_prop_2020_23, 5))
    print("x_o (logit stock 4, age 1):", np.round(x_o_data, 5))

    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    print(f"calibrating {num_parameters} parameters: "
          + ", ".join(f"{p['name']} in {p['bounds']}" for p in parameters_list))

    inference = NPE(prior=prior)

    # One allocator for the whole calibration, so no seed is reused between rounds
    # or between thetas.
    allocator = SeedAllocator(master_seed)

    posteriors = []
    proposal = prior
    seed_log = []
    x_seed_sd_log = []

    for i in range(num_rounds):
        print("ROUND: ", i + 1, "/", num_rounds)

        theta, x, seed_matrix, x_sd = simulate_round(
            proposal,
            base_params,
            parameters_list,
            num_thetas=num_thetas_per_round,
            num_seeds=num_seeds_per_theta,
            allocator=allocator,
            # Pins the theta draw for this round only, and must differ per round
            # or every round would redraw the same thetas.
            torch_seed=int(i * 10_000 + 1),
            central=central,
        )
        seed_log.append({"round": i, "seed_matrix": seed_matrix})
        # Saved alongside, not inside seed_log, so it can be loaded on its own
        # without pulling in a (num_thetas, num_seeds) integer matrix. Aligned row
        # for row with the theta/x of the same round, which come back out of
        # inference.pkl as torch.cat(inference._theta_roundwise / ._x_roundwise).
        x_seed_sd_log.append(x_sd)

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

    # Same keys as the other two gen scripts where they mean the same thing, plus
    # the ones specific to this x layout. NN_multi_round_calibration_multi_plot
    # prefers the saved x_o tensor, so it reads this run unchanged; the
    # reconstruct_x_o() fallback in that script does NOT know about the logit and
    # would rebuild a raw-scale vector, hence x_layout as an explicit marker.
    match_data = {
        "EV_stock_prop_2020_23": EV_stock_prop_2020_23,
        "extra_scalar_targets": [float(age_target[0])],
        "mean_car_age_range_months": AGE_TARGET_RANGE,
        "age_scale": AGE_SCALE,
        "x_layout": "logit_stock_4_age_1",
        "logit_eps": LOGIT_EPS,
        "match_years": (MATCH_START_YEAR, MATCH_END_YEAR),
    }
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params_save, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")

    save_object(seed_log, fileName + "/Data", "seed_log")
    # One (num_thetas, NUM_RAW_MOMENTS) array per round. See build_x() for what it
    # is for: it is the only thing that separates seed noise, which num_seeds
    # controls, from theta-coupled noise, which it cannot touch.
    save_object(x_seed_sd_log, fileName + "/Data", "x_seed_sd")
    save_object(
        {
            "num_thetas_per_round": num_thetas_per_round,
            "num_seeds_per_theta": num_seeds_per_theta,
            "num_rounds": num_rounds,
            "master_seed": master_seed,
            "central": central,
            "logit_eps": LOGIT_EPS,
            "min_finite_seed_fraction": MIN_FINITE_SEED_FRACTION,
            "kappa": base_params["parameters_vehicle_user"]["kappa"],
        },
        fileName + "/Data",
        "run_config",
    )

    samples = posterior.sample((500000,), x=x_o)
    log_probability_samples = posterior.log_prob(samples, x=x_o)
    max_log_prob_index = log_probability_samples.argmax()
    best_sample = samples[max_log_prob_index]
    print("best_sample", best_sample)
    for p, v in zip(parameters_list, best_sample):
        print(f"   {p['subdict']}.{p['name']} = {float(v):.6g}")
    save_object(samples, fileName + "/Data", "samples")
    save_object(best_sample, fileName + "/Data", "best_sample")
    save_object(inference, fileName + "/Data", "inference")

    return fileName


if __name__ == "__main__":
    # kappa is absent here on purpose. See the module docstring: it loads onto the
    # unconstrained direction of the [stock, age] Jacobian, its posterior has come
    # back at 0.99x the prior width in every run made so far, and pinning it at
    # 1.8e-4 costs ~0.03x on the achievable stock fit.
    #
    # SET parameters_vehicle_user.kappa IN base_params_NN.json BEFORE LAUNCHING.
    # It is not calibrated here, so the JSON value is used verbatim for every draw
    # and the run is conditional on it. The startup log prints what it read.
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [0.8, 2]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [2, 5]},
        {"name": "delta", "subdict": "parameters_ICE", "bounds": [0.001, 0.0033]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [2e-4, 3e-4]}
    ]

    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_thetas_per_round=1024,
        num_seeds_per_theta=64,
        num_rounds=1,
        master_seed=20260816,
        central="mean",
    )
