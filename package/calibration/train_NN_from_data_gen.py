import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
import numpy as np
from package.resources.utility import load_object, save_object, produce_name_datetime, createFolder

def produce_param_arr(property_dict_1, property_dict_2, property_dict_3, seed_repetitions):
    """
    Generate a arr of parameter combinations for the three parameters, including seed repetitions.
    
    Args:
        property_dict_1 (dict): Dictionary for the first parameter with keys "property_list", "min", "max", "reps".
        property_dict_2 (dict): Dictionary for the second parameter with keys "property_list", "min", "max", "reps".
        property_dict_3 (dict): Dictionary for the third parameter with keys "property_list", "min", "max", "reps".
        seed_repetitions (int): Number of seed repetitions for each parameter combination.
    
    Returns:
        list: A list of parameter combinations, where each combination is a tuple (param1, param2, param3, seed).
    """
    # Generate the parameter lists using np.linspace
    property_dict_1["property_list"] = np.linspace(property_dict_1["min"], property_dict_1["max"], property_dict_1["reps"])
    property_dict_2["property_list"] = np.linspace(property_dict_2["min"], property_dict_2["max"], property_dict_2["reps"])
    property_dict_3["property_list"] = np.linspace(property_dict_3["min"], property_dict_3["max"], property_dict_3["reps"])

    params_list = []

    # Iterate through all combinations of the three parameters
    for i in property_dict_1["property_list"]:
        for j in property_dict_2["property_list"]:
            for k in property_dict_3["property_list"]:
                # Add seed repetitions for each parameter combination
                for seed in range(1, seed_repetitions + 1):
                    params_list.append((i, j, k))

    return np.asarray(params_list)


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
    data_flat_ev_prop = load_object(file_path + "/Data", "data_flat_ev_prop")
    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    return base_params, data_flat_ev_prop, EV_stock_prop_2010_23, vary_1, vary_2, vary_3

def prepare_training_data(data_flat_ev_prop, real_data, base_params):
    # Reshape the data array to (num_samples, num_months)
    X = convert_data(data_flat_ev_prop, base_params)

    print(f"Processed X shape: {X.shape}")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(real_data, dtype=torch.float32)

def train_neural_network(params_arr, X, real_data, file_path):
    """
    Train a neural network using the given parameter array, simulated data, and real data.
    
    Args:
        params_arr (np.ndarray): Array of parameter combinations, shape (num_samples, 3).
        X (torch.Tensor): Simulated data, shape (num_samples, num_features).
        real_data (torch.Tensor): Real-world data, shape (num_features,).
        file_path (str): Path to save the trained model and results.
    """
    # Convert params_arr to a PyTorch tensor
    params_tensor = torch.tensor(params_arr, dtype=torch.float32)

    # Define the prior using the min and max values of each parameter in params_arr
    low_bounds = torch.tensor(np.min(params_arr, axis=0), dtype=torch.float32)
    print("low_bounds", low_bounds)
    high_bounds = torch.tensor(np.max(params_arr, axis=0), dtype=torch.float32)
    print("high_bounds ", high_bounds )

    prior = BoxUniform(low=low_bounds, high=high_bounds)
    prior, _, _ = process_prior(prior)

    # Initialize the inference object
    inference = NPE(prior=prior)

    # Append simulations (X and corresponding parameters)
    inference.append_simulations(X, params_tensor)

    # Train the density estimator
    density_estimator = inference.train()

    # Build the posterior
    posterior = inference.build_posterior(density_estimator)

    # Save the posterior, X, and real_data
    save_object(posterior, f"{file_path}/Data", "posterior")
    save_object(X, f"{file_path}/Data", "X")
    save_object(real_data, f"{file_path}/Data", "real_data")

    # Sample from the posterior using real data (conditioned on observation)
    samples = posterior.sample((100000,), x=real_data.unsqueeze(0))

    # Save the samples
    save_object(samples, f"{file_path}/Data", "samples")

def main(file_path):
    base_params, data_array_ev_prop_flat, EV_stock_prop_2010_23, vary_1, vary_2, vary_3 = load_data_and_params(file_path)
    # Generate the parameter list
    params_arr = produce_param_arr(vary_1, vary_2, vary_3, base_params["seed_repetitions"])
    
    print("params_arr.shape",params_arr.shape)
    print(f"Loaded data array shape: {data_array_ev_prop_flat.shape}")

    X, real_data = prepare_training_data(data_array_ev_prop_flat, EV_stock_prop_2010_23, base_params)

    train_neural_network(params_arr, X, real_data, file_path)

if __name__ == "__main__":
    main("results/MAPE_ev_3D_17_21_36__11_02_2025")
