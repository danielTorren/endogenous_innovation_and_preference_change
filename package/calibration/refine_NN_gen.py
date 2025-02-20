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
    start_year = 2010
    end_year = 2022
    averages = []
    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2001) * 12 + base_params["duration_burn_in"]
        start_idx = year_start_index + 9
        end_idx = year_start_index + 12
        last_three_months = data_to_fit[start_idx:end_idx]
        averages.append(np.mean(last_three_months))
    averages_array = np.array(averages)
    return averages_array

def update_base_params_with_seed(base_params, seed):
    seed_repetitions = base_params["seed_repetitions"]
    base_params["seeds"] = {
        key: seed + i * seed_repetitions for i, key in enumerate([
            "init_tech_seed", "landscape_seed_ICE", "social_network_seed",
            "network_structure_seed", "init_vals_environmental_seed",
            "init_vals_innovative_seed", "init_vals_price_seed",
            "innovation_seed", "landscape_seed_EV", "choice_seed",
            "remove_seed", "init_vals_poisson_seed","init_vals_range_seed"
        ])
    }
    return base_params

def run_single_simulation(theta, base_params, param_list):
    for i, param in enumerate(param_list):
        subdict = param["subdict"]
        name = param["name"]
        base_params[subdict][name] = theta[i].item()
    controller = generate_data(base_params)
    arr_history = np.asarray(controller.social_network.history_prop_EV)
    data_to_fit = convert_data(arr_history, base_params)
    return data_to_fit

def refine_parameters(parameters_list, posterior, x_o, refinement_factor=0.1):
    """
    Refine parameter bounds based on the posterior distribution.
    """
    samples = posterior.sample((1000000,), x=x_o)
    for param in parameters_list:
        param_samples = samples[:, parameters_list.index(param)]
        mean = param_samples.mean().item()
        std = param_samples.std().item()
        param["bounds"][0] = max(param["bounds"][0], mean - refinement_factor * std)
        param["bounds"][1] = min(param["bounds"][1], mean + refinement_factor * std)
    return parameters_list

def main(
        LOAD_PATH="NN_calibration_multi/2023-10-01_12-34-56/Data",  # Path to saved data
        num_simulations=100,
        num_additional_rounds=2,
        refinement_factor=0.1,
        seed_multiplier=2
    ) -> str:

    # Load previously saved data
    posterior = load_object(LOAD_PATH, "posterior")
    parameters_list = load_object(LOAD_PATH, "var_dict")
    base_params = load_object(LOAD_PATH, "base_params")
    x_o = load_object(LOAD_PATH, "x_o")

    # Refine parameter bounds based on the posterior
    parameters_list = refine_parameters(parameters_list, posterior, x_o, refinement_factor)

    # Increase the number of seeds
    base_params["seed_repetitions"] *= seed_multiplier

    # Define the new prior with refined bounds
    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)

    # Process the prior
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Initialize inference object with the loaded posterior
    inference = NPE(prior=prior)
    inference.append_simulations(*posterior.sample((num_simulations,), x=x_o))

    posteriors = [posterior]
    proposal = posterior.set_default_x(x_o)

    seeds = np.arange(1, base_params["seed_repetitions"] + 1)

    for i in range(num_additional_rounds):
        print("ADDITIONAL ROUND: ", i + 1, "/", num_additional_rounds)

        for seed in seeds:
            seeded_params = update_base_params_with_seed(base_params, seed)
            seeded_simulator = partial(run_single_simulation, base_params=seeded_params, param_list=parameters_list)
            sim_for_seed = process_simulator(seeded_simulator, prior, is_numpy_simulator=prior_returns_numpy)
            check_sbi_inputs(sim_for_seed, prior)

            theta, x = simulate_for_sbi(
                sim_for_seed,
                proposal,
                num_simulations=num_simulations,
                num_workers=multiprocessing.cpu_count(),
                simulation_batch_size=1
            )

            inference.append_simulations(theta, x, proposal=proposal)
        
        # Train the density estimator and update the posterior
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

        # Further refine parameters and increase seeds for the next round
        parameters_list = refine_parameters(parameters_list, posterior, x_o, refinement_factor)
        base_params["seed_repetitions"] *= seed_multiplier

    # Save the updated results
    root = "NN_calibration_multi_refined"
    fileName = produce_name_datetime(root)
    createFolder(fileName)

    match_data = {"EV_stock_prop_2010_22": x_o.numpy()}
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")
    
    samples = posterior.sample((1000000,), x=x_o)
    log_probability_samples = posterior.log_prob(samples, x=x_o)
    max_log_prob_index = log_probability_samples.argmax()
    best_sample = samples[max_log_prob_index]
    print("best_sample", best_sample)
    save_object(samples, fileName + "/Data", "samples")
    save_object(best_sample, fileName + "/Data", "best_sample")
    save_object(inference, fileName + "/Data", "inference")


if __name__ == "__main__":
    main(
        LOAD_PATH="NN_calibration_multi/2023-10-01_12-34-56/Data",  # Update this path
        num_simulations=32,
        num_additional_rounds=2,
        refinement_factor=0.1,
        seed_multiplier=2
    )