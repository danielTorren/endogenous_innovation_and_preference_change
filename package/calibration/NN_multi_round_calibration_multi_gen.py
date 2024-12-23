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

    # Assuming `data_to_fit` is a numpy array of size (272,) representing monthly data from 2000 to 2022
    # Define the starting and ending indices for the years 2010 to 2022
    start_year = 2010
    end_year = 2022

    # Calculate the average of the last three months of each year
    averages = []

    #print("filtered_data", filtered_data)
    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2000) * 12 + base_params["duration_burn_in"]#ADD ON THE BURN IN PERIOD TO THE START
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        # Ensure the indices are within bounds
        last_three_months = data_to_fit[start_idx:end_idx]
        
        averages.append(np.mean(last_three_months))

    averages_array = np.array(averages)

    return averages_array

def update_base_params_with_seed(base_params, seed):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    seed_repetitions = base_params["seed_repetitions"]
    # VARY ALL THE SEEDS
    base_params["parameters_firm_manager"]["init_tech_seed"] = seed + seed_repetitions
    base_params["parameters_ICE"]["landscape_seed"] = seed + 2 * seed_repetitions
    base_params["parameters_EV"]["landscape_seed"] = seed + 9 * seed_repetitions
    base_params["parameters_social_network"]["social_network_seed"] = seed + 3 * seed_repetitions
    base_params["parameters_social_network"]["network_structure_seed"] = seed + 4 * seed_repetitions
    base_params["parameters_social_network"]["init_vals_environmental_seed"] = seed + 5 * seed_repetitions
    base_params["parameters_social_network"]["init_vals_innovative_seed"] = seed + 6 * seed_repetitions
    base_params["parameters_social_network"]["init_vals_price_seed"] = seed + 7 * seed_repetitions
    base_params["parameters_firm"]["innovation_seed"] = seed + 8 * seed_repetitions
    return base_params

def run_single_simulation(theta, base_params, param_list):
    """
    Runs a single simulation for the given parameters theta and base_params.
    """

    #print("theta", theta)
    # Update the parameters from theta
    for i, param in enumerate(param_list):
        subdict = param["subdict"]
        name = param["name"]
        base_params[subdict][name] = theta[i].item()

    # Run the market simulation
    controller = generate_data(base_params)

    # Compute summary statistics
    arr_history = np.asarray(controller.social_network.history_prop_EV)
    data_to_fit = convert_data(arr_history, base_params)

    return data_to_fit

def main(
        parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=100
    ) -> str:

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    # Load observed data
    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    root = "NN_calibration_multi"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    num_rounds = 3

    # Observed data
    x_o = torch.tensor(EV_stock_prop_2010_22, dtype=torch.float32)

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

        # For each round, we run multiple seeds and append results
        for seed in seeds:

            # Update base params for this seed
            seeded_params = update_base_params_with_seed(base_params, seed)

            # Create a simulator partial with these seeded params
            seeded_simulator = partial(run_single_simulation, base_params=seeded_params, param_list=parameters_list)

            # Process the simulator once per seed
            sim_for_seed = process_simulator(seeded_simulator, prior, is_numpy_simulator=prior_returns_numpy)
            check_sbi_inputs(sim_for_seed, prior)

            # Run simulations for this seed
            theta, x = simulate_for_sbi(
                sim_for_seed,
                proposal,
                num_simulations=num_simulations,
                num_workers=multiprocessing.cpu_count(),
                simulation_batch_size=1
            )

            # Append simulations and train incrementally
            inference.append_simulations(theta, x, proposal=proposal)
        
        # After collecting simulations from all seeds in this round, train the density estimator
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    createFolder(fileName)

    # Save results
    match_data = {"EV_stock_prop_2010_22": EV_stock_prop_2010_22}
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")

if __name__ == "__main__":
    parameters_list = [
        {"name": "a_innovativeness", "subdict": "parameters_social_network", "bounds": [0.05, 3]},
        {"name": "b_innovativeness", "subdict": "parameters_social_network", "bounds": [0.05, 3]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [1, 30]},
        #{"name": "SW_prob_rewire", "subdict": "parameters_social_network", "bounds": [0, 1]},
        #{"name": "SW_K", "subdict": "parameters_social_network", "bounds": [20, 500]},
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round_multi.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=128
    )
