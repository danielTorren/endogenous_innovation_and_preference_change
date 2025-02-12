import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
import numpy as np
from package.resources.utility import load_object, save_object, produce_name_datetime, createFolder

def convert_data(data_to_fit, base_params):
    # Convert each sample's monthly data to yearly data
    start_year = 2010
    end_year = 2022
    num_years = end_year - start_year + 1

    # Ensure data_to_fit is 2D: (num_samples, num_months)
    if data_to_fit.ndim == 1:
        data_to_fit = data_to_fit.reshape(1, -1)

    num_samples = data_to_fit.shape[0]
    yearly_averages = np.zeros((num_samples, num_years))

    for year_idx, year in enumerate(range(start_year, end_year + 1)):
        year_start_index = (year - 2000) * 12 + base_params["duration_burn_in"]
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        
        yearly_averages[:, year_idx] = np.mean(data_to_fit[:, start_idx:end_idx], axis=1)

    return yearly_averages

def load_data_and_params(file_path):
    vary_1 = load_object(file_path + "/Data", "vary_1") 
    vary_2 = load_object(file_path + "/Data", "vary_2")
    vary_3 = load_object(file_path + "/Data", "vary_3")
    base_params = load_object(file_path + "/Data", "base_params")
    data_array_ev_prop = load_object(file_path + "/Data", "data_array_ev_prop")
    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]
    return base_params, data_array_ev_prop, EV_stock_prop_2010_22, vary_1, vary_2, vary_3

def prepare_training_data(data_array_ev_prop, real_data, base_params):
    # Reshape the data array to (num_samples, num_months)
    num_samples = np.prod(data_array_ev_prop.shape[:-1])
    num_months = data_array_ev_prop.shape[-1]
    
    X_full = data_array_ev_prop.reshape(num_samples, num_months)
    X = convert_data(X_full, base_params)

    print(f"Processed X shape: {X.shape}")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(real_data, dtype=torch.float32)

def train_neural_network(parameters_list, X, real_data, file_path):
    low_bounds = torch.tensor([p["min"] for p in parameters_list])
    high_bounds = torch.tensor([p["max"] for p in parameters_list])

    prior = BoxUniform(low=low_bounds, high=high_bounds)
    prior, _, _ = process_prior(prior)

    inference = NPE(prior=prior)
    inference.append_simulations(X, prior.sample((X.shape[0],)))  # Simulated data with corresponding parameters
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    save_object(posterior, f"{file_path}/Data", "posterior")
    save_object(X, f"{file_path}/Data", "X")
    save_object(real_data, f"{file_path}/Data", "real_data")

    # Sampling from the posterior using real data (conditioned on observation)
    samples = posterior.sample((100000,), x=real_data)
    save_object(samples, f"{file_path}/Data", "samples")

def main(file_path):
    base_params, data_array_ev_prop, EV_stock_prop_2010_22, vary_1, vary_2, vary_3 = load_data_and_params(file_path)
    print(f"Loaded data array shape: {data_array_ev_prop.shape}")

    X, real_data = prepare_training_data(data_array_ev_prop, EV_stock_prop_2010_22, base_params)

    parameters_list = [vary_1, vary_2, vary_3]

    train_neural_network(parameters_list, X, real_data, file_path)

if __name__ == "__main__":
    main("results/MAPE_ev_3D_17_21_36__11_02_2025")
