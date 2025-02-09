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
        year_start_index = (year - 2000) * 12 + base_params["duration_burn_in"]
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        last_three_months = data_to_fit[start_idx:end_idx]
        averages.append(np.mean(last_three_months))
    return np.array(averages)

def update_base_params_with_seed(base_params, seed):
    seed_repetitions = base_params["seed_repetitions"]
    base_params["seeds"] = {
        key: seed + i * seed_repetitions for i, key in enumerate([
            "init_tech_seed", "landscape_seed_ICE", "social_network_seed",
            "network_structure_seed", "init_vals_environmental_seed",
            "init_vals_innovative_seed", "init_vals_price_seed",
            "innovation_seed", "landscape_seed_EV", "choice_seed",
            "remove_seed", "init_vals_poisson_seed"
        ])
    }
    return base_params

def run_single_simulation(theta, base_params, param_list):
    for i, param in enumerate(param_list):
        base_params[param["subdict"]][param["name"]] = theta[i].item()
    controller = generate_data(base_params)
    arr_history = np.asarray(controller.social_network.history_prop_EV)
    return convert_data(arr_history, base_params)

def run_simulation_for_seed(seed, parameters_list, prior, proposal, num_simulations, base_params):
    seeded_params = update_base_params_with_seed(base_params.copy(), seed)
    seeded_simulator = partial(run_single_simulation, base_params=seeded_params, param_list=parameters_list)
    sim_for_seed = process_simulator(seeded_simulator, prior, is_numpy_simulator=False)
    check_sbi_inputs(sim_for_seed, prior)
    theta, x = simulate_for_sbi(
        sim_for_seed,
        proposal,
        num_simulations=num_simulations,
        num_workers=max(1, multiprocessing.cpu_count() // 2)
    )
    return theta, x

def main(parameters_list, BASE_PARAMS_LOAD, OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME, num_simulations=100):
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]
    fileName = produce_name_datetime("NN_calibration_multi")
    x_o = torch.tensor(EV_stock_prop_2010_22, dtype=torch.float32)

    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)
    prior, _, _ = process_prior(prior)
    inference = NPE(prior=prior)

    seeds = np.arange(1, base_params["seed_repetitions"] + 1)
    proposal = prior

    for round_idx in range(3):
        print(f"ROUND: {round_idx + 1}/3")
        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                run_simulation_for_seed,
                [(seed, parameters_list, prior, proposal, num_simulations, base_params) for seed in seeds]
            )
        for theta, x in results:
            inference.append_simulations(theta, x, proposal=proposal)
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(x_o)

    createFolder(fileName)
    save_object(posterior, f"{fileName}/Data", "posterior")
    samples = posterior.sample((10000000,), x=x_o)
    save_object(samples, f"{fileName}/Data", "samples")

if __name__ == "__main__":
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [0.7, 3]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [0.7, 3]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [0.1, 2]},
    ]
    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=128
    )
