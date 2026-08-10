import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import json
import numpy as np
from copy import deepcopy
from functools import partial
from package.resources.utility import (
    produce_name_datetime,
    save_object,
    createFolder,
    load_object,
    get_num_workers
)
from package.resources.run import generate_data
import multiprocessing

MATCH_START_YEAR = 2020
MATCH_END_YEAR = 2023
MATCH_NUM_YEARS = MATCH_END_YEAR - MATCH_START_YEAR + 1
STOCK_MONTH_OFFSET = 3   # APRIL index, matches the EV stock (population) data
SALES_MONTH_OFFSET = 11  # DECEMBER index, matches the EV sales data

# Two extra moments beyond EV stock and EV sales.
#
# Both are published as RANGES, but NPE conditions on a single observation x_o,
# so each is matched at its interval midpoint and the range is kept only for
# reporting / post-hoc filtering of posterior samples:
#   mean fleet age    120-144 months (10-12 years) -> 132
#   new-car sales HHI 0.11-0.18                    -> 0.145
#
# The age series is divided by AGE_SCALE so every component of x is O(1). sbi
# z-scores x per dimension anyway, so this is for legibility rather than for
# the fit; what DOES change the fit is dimension count -- see main().
AGE_TARGET_RANGE = (120.0, 144.0)
HHI_TARGET_RANGE = (0.11, 0.18)
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
    right comparison and far less seed-noisy than a single monthly snapshot.
    HHI does not need this: it is already computed over a trailing 12-month
    window (see firmManager.calculate_market_concentration), so its December
    value spans exactly that calendar year.
    """
    means = []

    for year in range(start_year, end_year + 1):
        start = _year_start_index(year, base_params)
        means.append(np.mean(data_to_fit[start:start + 12]))

    return np.array(means)

def run_single_simulation(theta, base_params, param_list):
    """
    Runs a single simulation for the given parameters theta and base_params.
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

    # Update the parameters from theta
    for i, param in enumerate(param_list):
        subdict = param["subdict"]
        name = param["name"]
        params[subdict][name] = theta[i].item()

    # Run the market simulation
    controller = generate_data(params)

    # Compute summary statistics, all restricted to the last few years of the
    # run: EV stock proportion, EV sales proportion, mean fleet age and new-car
    # market concentration. All four series are appended once per step from
    # social_network.next_step()/firm_manager.next_step() regardless of
    # save_timeseries_data_state, so they share one index base and the same
    # month-offset convention applies to each.
    arr_history_stock = np.asarray(controller.social_network.history_prop_EV)
    arr_history_sales = np.asarray(controller.firm_manager.history_past_new_bought_vehicles_prop_ev)
    arr_history_age = np.asarray(controller.social_network.history_mean_car_age_fleet)
    arr_history_HHI = np.asarray(controller.firm_manager.history_HHI)

    stock_data_to_fit = convert_data(arr_history_stock, params, STOCK_MONTH_OFFSET)
    sales_data_to_fit = convert_data(arr_history_sales, params, SALES_MONTH_OFFSET)
    age_data_to_fit = convert_data_annual_mean(arr_history_age, params) / AGE_SCALE
    HHI_data_to_fit = convert_data(arr_history_HHI, params, SALES_MONTH_OFFSET)

    # ORDER MATTERS: must match how x_o is assembled in main().
    data_to_fit = np.concatenate([
        stock_data_to_fit,
        sales_data_to_fit,
        age_data_to_fit,
        HHI_data_to_fit,
    ])

    return data_to_fit

def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=100,
        num_rounds = 3
    ) -> str:

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Snapshot before the seed loop starts overwriting base_params["seed"] --
    # this is what gets saved, so base_params.pkl reflects the configuration
    # the run was launched with rather than whatever the last seed happened
    # to be (previously it always read seed = seed_repetitions).
    base_params_save = deepcopy(base_params)

    total_runs = num_rounds*num_simulations* base_params["seed_repetitions"]
    print("TOTAL RUNS: ", total_runs)
    
    # Load observed data
    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_stock_prop_2020_23 = EV_stock_prop_2010_23[-4:]  # last 4 years only
    EV_sales_prop_2020_23 = calibration_data_output["EV Sales Prop"]  # already just 2020-2023

    root = "NN_calibration_multi"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    # Targets for the two extra moments. Flat across years: neither the fleet
    # age range nor the HHI range is resolved year by year in the source data,
    # so the same midpoint is repeated. Repeating it over MATCH_NUM_YEARS rather
    # than collapsing to a single component keeps the four blocks of x the same
    # length, which keeps their relative weight in the summary vector equal --
    # sbi z-scores x per dimension, so weight is carried by dimension COUNT, not
    # by scale. If the EV moments end up dominated, collapse these two blocks to
    # one component each (a 4-year mean) rather than rescaling them.
    age_target = np.full(MATCH_NUM_YEARS, np.mean(AGE_TARGET_RANGE) / AGE_SCALE)
    HHI_target = np.full(MATCH_NUM_YEARS, np.mean(HHI_TARGET_RANGE))

    # Observed data, last 4 years: EV stock proportion, EV sales proportion,
    # mean fleet age (scaled), new-car HHI.
    # ORDER MATTERS: must match run_single_simulation().
    x_o_data = np.concatenate([
        EV_stock_prop_2020_23,
        EV_sales_prop_2020_23,
        age_target,
        HHI_target,
    ])
    x_o = torch.tensor(x_o_data, dtype=torch.float32)

    # Define the prior
    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)

    # Process the prior
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # We won't define the simulator fully yet; we will define it per seed
    # after updating base_params with that seed.

    # Initialize inference object
    inference = NPE(prior=prior)

    posteriors = []
    proposal = prior

    seeds = np.arange(1, base_params["seed_repetitions"]+1)

    for i in range(num_rounds):
        print("ROUND: ", i+1, "/", num_rounds)

        # For each round, we run multiple seeds and collect the results. These
        # are appended to `inference` ONCE, after the seed loop -- see below.
        theta_all, x_all = [], []

        for seed in seeds:

            # Update base params for this seed
            base_params["seed"] = seed

            # Create a simulator partial with these seeded params
            seeded_simulator = partial(run_single_simulation, base_params=base_params, param_list=parameters_list)

            # Process the simulator once per seed
            sim_for_seed = process_simulator(seeded_simulator, prior, is_numpy_simulator=prior_returns_numpy)
            check_sbi_inputs(sim_for_seed, prior)

            # Run simulations for this seed.
            # `seed` here pins sbi's OWN randomness (the theta draw from the
            # proposal, and the per-batch seeds it hands its workers) -- it is
            # unrelated to base_params["seed"], which is what actually drives
            # the ABM. It must differ per call: passing one value to every
            # iteration would make all 64 seed-batches draw IDENTICAL thetas.
            theta, x = simulate_for_sbi(
                sim_for_seed,
                proposal,
                num_simulations=num_simulations,
                num_workers=get_num_workers(),
                simulation_batch_size=1,
                seed=int(i * 10_000 + seed)
            )

            theta_all.append(theta)
            x_all.append(x)

        # Append ONCE per round, not once per seed. sbi stamps every
        # append_simulations() call that carries a non-prior proposal with
        # max(_data_round_index) + 1, so appending inside the seed loop made it
        # believe it was on round 64 after round 2 and round 128 after round 3.
        # All seed-batches in a round were drawn from the SAME proposal, so they
        # genuinely are one round; concatenating keeps _data_round_index at
        # [0, 1, 2]. Training is unchanged (same atomic-loss path, same
        # start_idx, same _proposal_roundwise[-1]) -- this just stops
        # discard_prior_samples / non-atomic MDN losses, which branch on
        # self._round, from silently misbehaving if they are ever switched on.
        inference.append_simulations(
            torch.cat(theta_all), torch.cat(x_all), proposal=proposal
        )

        # After collecting simulations from all seeds in this round, train the density estimator
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    createFolder(fileName)

    # Save results
    match_data = {
        "EV_stock_prop_2020_23": EV_stock_prop_2020_23,
        "EV_sales_prop_2020_23": EV_sales_prop_2020_23,
        # Ranges, not just the midpoints fed to NPE: keep them so posterior
        # samples can be checked against the whole interval afterwards.
        "mean_car_age_range_months": AGE_TARGET_RANGE,
        "HHI_range": HHI_TARGET_RANGE,
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
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [1, 5]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [1, 5]},
        {"name": "delta", "subdict": "parameters_ICE", "bounds": [0.001, 0.003]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [1e-4, 6e-4]},
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output", 
        num_simulations=256, 
        num_rounds= 2
    )