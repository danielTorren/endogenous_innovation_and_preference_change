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
from functools import partial
from package.resources.utility import (
    produce_name_datetime,
    save_object,
    createFolder,
    load_object
)
from package.resources.run import generate_data
import multiprocessing

def convert_data(data_to_fit, base_params):
    start_year = 2016
    end_year = 2022
    averages = []

    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2001) * 12 + base_params["duration_burn_in"]
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        last_three_months = data_to_fit[start_idx:end_idx]
        averages.append(np.mean(last_three_months))

    averages_array = np.array(averages)
    return averages_array

def run_single_simulation(theta, base_params, param_list):
    for i, param in enumerate(param_list):
        subdict = param["subdict"]
        name = param["name"]
        base_params[subdict][name] = theta[i].item()

    controller = generate_data(base_params)
    arr_history = np.asarray(controller.social_network.history_prop_EV)
    data_to_fit = convert_data(arr_history, base_params)

    return data_to_fit

def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=100,
        num_rounds=3
    ) -> str:

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    total_runs = num_rounds * num_simulations * base_params["seed_repetitions"]
    print("TOTAL RUNS: ", total_runs)

    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_stock_prop_2016_22 = EV_stock_prop_2010_23[6:]

    root = "NN_for_each_seed"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    x_o = torch.tensor(EV_stock_prop_2016_22, dtype=torch.float32)

    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)

    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    seeds = np.arange(1, base_params["seed_repetitions"] + 1)
    all_posteriors = []
    all_best_samples = []

    for seed in seeds:
        print(f"Training model for seed {seed}")
        base_params["seed"] = seed
        seeded_simulator = partial(run_single_simulation, base_params=base_params, param_list=parameters_list)
        sim_for_seed = process_simulator(seeded_simulator, prior, is_numpy_simulator=prior_returns_numpy)
        check_sbi_inputs(sim_for_seed, prior)

        inference = NPE(prior=prior)
        proposal = prior
        posteriors = []

        for i in range(num_rounds):
            print(f"ROUND {i + 1} for seed {seed}")
            theta, x = simulate_for_sbi(
                sim_for_seed,
                proposal,
                num_simulations=num_simulations,
                num_workers=multiprocessing.cpu_count(),
                simulation_batch_size=1
            )
            inference.append_simulations(theta, x, proposal=proposal)
            density_estimator = inference.train()
            posterior = inference.build_posterior(density_estimator)
            posteriors.append(posterior)
            proposal = posterior.set_default_x(x_o)

        all_posteriors.append(posteriors[-1])
        samples = posteriors[-1].sample((10000,), x=x_o)
        log_probability_samples = posteriors[-1].log_prob(samples, x=x_o)
        max_log_prob_index = log_probability_samples.argmax()
        best_sample = samples[max_log_prob_index]
        all_best_samples.append(best_sample)

    mean_best_sample = torch.mean(torch.stack(all_best_samples), dim=0)
    print("Mean best sample across seeds:", mean_best_sample)

    createFolder(fileName)
    match_data = {"EV_stock_prop_2016_22": EV_stock_prop_2016_22}
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(all_posteriors, fileName + "/Data", "all_posteriors")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")
    save_object(all_best_samples, fileName + "/Data", "all_best_samples")
    save_object(mean_best_sample, fileName + "/Data", "mean_best_sample")

if __name__ == "__main__":
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [1, 3]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [1, 4]},
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=64,
        num_rounds=3
    )