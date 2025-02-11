import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
import numpy as np
from package.resources.utility import load_object, save_object, produce_name_datetime, createFolder


def load_data_and_params(file_path):
    base_params = load_object(file_path + "/Data", "base_params")
    data_array_ev_prop = load_object(file_path + "/Data", "data_array_ev_prop")
    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]
    return base_params, data_array_ev_prop, EV_stock_prop_2010_22


def prepare_training_data(data_array_ev_prop, real_data):
    num_samples = data_array_ev_prop.shape[0] * data_array_ev_prop.shape[1] * data_array_ev_prop.shape[2]
    X = data_array_ev_prop.reshape(num_samples, -1)
    y = np.tile(real_data.flatten(), (num_samples, 1))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_neural_network(parameters_list, X, y, file_path):
    low_bounds = torch.tensor([p["bounds"][0] for p in parameters_list])
    high_bounds = torch.tensor([p["bounds"][1] for p in parameters_list])
    prior = BoxUniform(low=low_bounds, high=high_bounds)
    prior, _, _ = process_prior(prior)

    inference = NPE(prior=prior)
    inference.append_simulations(X, y, proposal=prior)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    createFolder(file_path)
    save_object(posterior, f"{file_path}/Data", "posterior")
    samples = posterior.sample((100000,), x=y)
    save_object(samples, f"{file_path}/Data", "samples")


def main(file_path):
    parameters_list = [
        {"name": "a_chi", "subdict": "parameters_social_network", "bounds": [0.7, 3]},
        {"name": "b_chi", "subdict": "parameters_social_network", "bounds": [0.7, 3]},
        {"name": "kappa", "subdict": "parameters_vehicle_user", "bounds": [0.1, 2]},
    ]

    base_params, data_array_ev_prop, EV_stock_prop_2010_22 = load_data_and_params(file_path)
    X, y = prepare_training_data(data_array_ev_prop, EV_stock_prop_2010_22)

    output_file_name = produce_name_datetime("NN_calibration_multi")
    train_neural_network(parameters_list, X, y, output_file_name)


if __name__ == "__main__":
    main("results/MAPE_ev_3D_23_56_09__10_02_2025")
