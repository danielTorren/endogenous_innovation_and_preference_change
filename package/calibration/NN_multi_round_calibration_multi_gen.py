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

# The observed EV_Sales.xlsx carries four annual values, 2020-2023 (see
# calibration_data_outputs.load_in_output_data). Same years as
# summary_stats.DEFAULT_SPEC["sales_years"], deliberately: the two calibrations
# must not disagree about what "the sales target" is.
SALES_YEARS = [2020, 2021, 2022, 2023]

BASE_YEAR = 2001   # calendar year of controller timestep 0, post burn-in


def _month_index(year, month, base_params):
    """Controller timestep for a calendar year/month. Same convention as convert_data."""
    return (year - BASE_YEAR) * 12 + base_params["duration_burn_in"] + (month - 1)


def convert_data(data_to_fit, base_params):

    # Assuming `data_to_fit` is a numpy array of size (272,) representing monthly data from 2001 to 2022
    # Define the starting and ending indices for the years 2010 to 2022
    start_year = 2016
    end_year = 2023

    # Calculate the average of the last three months of each year
    averages = []

    #print("filtered_data", filtered_data)
    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2001) * 12 + base_params["duration_burn_in"]#ADD ON THE BURN IN PERIOD TO THE START
        april_idx = year_start_index + 3  # APRIL index
        averages.append(data_to_fit[april_idx])

    averages_array = np.array(averages)

    return averages_array


def convert_sales(sn, base_params, years=SALES_YEARS, include_used=False):
    """
    Annual EV share of vehicle sales, from the ALWAYS-ON state-level flow
    counters on the social network.

    Reads history_new_sales / history_new_sales_EV, which are recorded
    regardless of save_timeseries_data_state -- see the comment next to their
    initialisation in socialNetworkUsers.__init__. The older
    history_new_car_bought / history_new_EV_cars_bought pair cannot be used
    here: those are filled from update_counters(), which only runs under
    save_timeseries_data_state = 1, and turning that on to get four numbers
    would also record a per-individual array for every quantity at every
    timestep.

    SUMMED over the twelve months of the calendar year, not sampled at a
    snapshot month like the stock series is. The observed series is annual, and
    a single month of model sales at num_individuals = 3000 would be mostly
    Monte Carlo noise.

    include_used=True adds second-hand purchases, for the case where the
    observed "sales" figure is really all registrations. The California series
    is new-vehicle sales, so the default is False.
    """
    if include_used:
        ev = np.asarray(sn.history_new_sales_EV, dtype=np.float64) + np.asarray(sn.history_used_sales_EV, dtype=np.float64)
        tot = np.asarray(sn.history_new_sales, dtype=np.float64) + np.asarray(sn.history_used_sales, dtype=np.float64)
    else:
        ev = np.asarray(sn.history_new_sales_EV, dtype=np.float64)
        tot = np.asarray(sn.history_new_sales, dtype=np.float64)

    n_hist = len(tot)
    out = []
    for year in years:
        start = _month_index(year, 1, base_params)
        if start < 0 or start + 12 > n_hist:
            raise IndexError(
                f"sales year {year} needs timesteps {start}..{start + 11} but the simulated "
                f"history is {n_hist} long. duration_calibration is too short for SALES_YEARS, "
                f"or duration_burn_in is inconsistent with BASE_YEAR."
            )
        denom = tot[start:start + 12].sum()
        # A year with no new-car sales at all should not happen at these
        # population sizes, but NPE requires finite inputs, and 0.0 is what "no
        # EV activity observed" means for a share.
        out.append(ev[start:start + 12].sum() / denom if denom > 0 else 0.0)

    return np.array(out)


def run_single_simulation(theta, base_params, param_list, include_sales=True,
                          sales_include_used=False):
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

    # Compute summary statistics
    arr_history = np.asarray(controller.social_network.history_prop_EV)
    data_to_fit = convert_data(arr_history, params)

    if include_sales:
        sales = convert_sales(controller.social_network, params,
                              include_used=sales_include_used)
        data_to_fit = np.concatenate([data_to_fit, sales])

    return data_to_fit

def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=100,
        num_rounds = 3,
        include_sales=True,
        sales_include_used=False
    ) -> str:
    """
    Args:
        include_sales: append the four annual EV sales shares (2020-2023) to the
            eight April stock shares, giving a 12-dim target. On by default: the
            sales series was already loaded by calibration_data_outputs.py and
            then discarded, and it is the flow moment, so it separates "EVs are
            accumulating because they were bought years ago" from "EVs are being
            bought now" in a way the stock series alone cannot.
        sales_include_used: count second-hand purchases in the sales share too.
            Leave False for the California new-vehicle series.
    """

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
    EV_stock_prop_2016_23 = EV_stock_prop_2010_23[6:]

    EV_sales_prop = np.asarray(calibration_data_output["EV Sales Prop"], dtype=np.float64)
    if include_sales and len(EV_sales_prop) != len(SALES_YEARS):
        raise ValueError(
            f"'EV Sales Prop' has {len(EV_sales_prop)} values but SALES_YEARS is {SALES_YEARS}. "
            f"Rebuild the target with calibration_data_outputs.py, or adjust SALES_YEARS."
        )

    root = "NN_calibration_multi"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    # Observed data
    if include_sales:
        x_o_np = np.concatenate([EV_stock_prop_2016_23, EV_sales_prop])
    else:
        x_o_np = np.asarray(EV_stock_prop_2016_23, dtype=np.float64)
    x_o = torch.tensor(x_o_np, dtype=torch.float32)

    dim_names = ([f"stock_{y}" for y in range(2016, 2024)]
                 + ([f"sales_{y}" for y in SALES_YEARS] if include_sales else []))
    print(f"fitting {len(dim_names)} dims: {dim_names}")

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
            seeded_simulator = partial(run_single_simulation, base_params=base_params,
                                       param_list=parameters_list,
                                       include_sales=include_sales,
                                       sales_include_used=sales_include_used)

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
    # match_data keeps the stock key under its old name so existing plotting
    # scripts still load; the sales entries and dim_names are additive.
    match_data = {
        "EV_stock_prop_2016_23": EV_stock_prop_2016_23,
        "EV_sales_prop": EV_sales_prop,
        "include_sales": include_sales,
        "sales_years": SALES_YEARS,
        "dim_names": dim_names,
    }
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params_save, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")
    
    samples = posterior.sample((100000,), x=x_o)
    log_probability_samples = posterior.log_prob(samples, x=x_o)
    max_log_prob_index = log_probability_samples.argmax()
    best_sample = samples[max_log_prob_index]
    print("best_sample", best_sample)
    save_object(samples, fileName + "/Data", "samples")
    save_object(best_sample, fileName + "/Data", "best_sample")
    save_object(inference, fileName + "/Data", "inference")


if __name__ == "__main__":
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [0.8, 1.5]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [2, 2.7]},
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_lower_fuel.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output", 
        num_simulations=64, 
        num_rounds= 2
    )