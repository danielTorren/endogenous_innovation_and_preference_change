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
from package.calibration.NN_multi_round_calibration_gen import convert_data
import multiprocessing

def run_single_simulation(theta, base_params, param_list):
    """
    Runs a single simulation for the given parameters theta.
    This function must be defined at the top-level so it can be pickled for parallel processing.
    """
    updated_params = base_params.copy()
    # Update the parameters from theta
    for i, param in enumerate(param_list):
        subdict = param["subdict"]
        name = param["name"]
        updated_params[subdict][name] = theta[i].item()

    # Run the market simulation
    controller = generate_data(updated_params)

    # Compute summary statistics
    data_to_fit_distance = np.median(np.asarray(controller.social_network.users_distance_vec))
    data_to_fit_age = np.median(np.asarray(controller.social_network.car_ages))
    data_to_fit_price = np.median(np.asarray([car.price for car in controller.firm_manager.cars_on_sale_all_firms]))
    arr_history = np.asarray(controller.social_network.history_prop_EV)
    data_to_fit = convert_data(arr_history)

    # Convert to tensors
    stock_tensor = torch.tensor(data_to_fit, dtype=torch.float32)
    median_distance_tensor = torch.tensor([data_to_fit_distance], dtype=torch.float32)
    median_age_tensor = torch.tensor([data_to_fit_age], dtype=torch.float32)
    median_price_tensor = torch.tensor([data_to_fit_price], dtype=torch.float32)

    # Concatenate into a single output tensor
    result = torch.cat((stock_tensor, median_distance_tensor, median_age_tensor, median_price_tensor), dim=0)
    return result


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

    # Observed statistics
    median_distance_traveled = 1400
    median_age = 120
    median_price = 30000
    match_data = {
        "median_distance_traveled": median_distance_traveled,
        "median_age": median_age,
        "median_price": median_price,
        "EV_stock_prop_2010_22": EV_stock_prop_2010_22
    }

    # Convert observed data to tensors
    EV_stock_prop_2010_22_tensor = torch.tensor(EV_stock_prop_2010_22, dtype=torch.float32)
    median_distance_traveled_tensor = torch.tensor([median_distance_traveled], dtype=torch.float32)
    median_age_tensor = torch.tensor([median_age], dtype=torch.float32)
    median_price_tensor = torch.tensor([median_price], dtype=torch.float32)
    x_o = torch.cat((EV_stock_prop_2010_22_tensor, median_distance_traveled_tensor, median_age_tensor, median_price_tensor), dim=0)

    # Define the prior
    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)

    # Process the prior
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Create the simulator function for a single parameter set
    # Using partial to "freeze" base_params and parameters_list
    simulator = partial(run_single_simulation, base_params=base_params, param_list=parameters_list)
    print("CREATED PARTIAL")

    # Process the simulator
    simulator = process_simulator(simulator, prior, is_numpy_simulator=prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    print("CREATED simulator with prior")
    # Instantiate the inference object
    inference = NPE(prior=prior)

    # Run multiple rounds
    posteriors = []
    proposal = prior

    for i in range(num_rounds):
        print("ROUND: ", i+1,"/", num_rounds)
        # Set num_workers to use parallelization
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, num_workers=multiprocessing.cpu_count())
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    createFolder(fileName)

    # Save results
    save_object(match_data, fileName + "/Data", "match_data")
    save_object(posterior, fileName + "/Data", "posterior")
    save_object(prior, fileName + "/Data", "prior")
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")


if __name__ == "__main__":
    parameters_list = [
        {"name": "a_innovativeness", "subdict": "parameters_social_network", "bounds": [0.01, 4]},
        {"name": "b_innovativeness", "subdict": "parameters_social_network", "bounds": [0.01, 4]},
        {"name": "min_Quality", "subdict": "parameters_ICE", "bounds": [0, 10]},
        {"name": "max_Quality", "subdict": "parameters_ICE", "bounds": [1, 50]},
        {"name": "min_Cost", "subdict": "parameters_ICE", "bounds": [0, 10000]},
        {"name": "max_Cost", "subdict": "parameters_ICE", "bounds": [1000, 100000]},
        {"name": "r", "subdict": "parameters_vehicle_user", "bounds": [0.01, 1]},
        {"name": "delta", "subdict": "parameters_ICE", "bounds": [1e-3, 1e-2]},
        {"name": "alpha", "subdict": "parameters_vehicle_user", "bounds": [0.1, 0.95]},
        {"name": "d_min", "subdict": "parameters_social_network", "bounds": [100, 1300]},
        {"name": "mu", "subdict": "parameters_vehicle_user", "bounds": [0.01, 1]},
    ]

    main(
        parameters_list=parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round_multi.json",
        OUTPUTS_LOAD_ROOT="package/calibration_data",
        OUTPUTS_LOAD_NAME="calibration_data_output",
        num_simulations=40
    )
