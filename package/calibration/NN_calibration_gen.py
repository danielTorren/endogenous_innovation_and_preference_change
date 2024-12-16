import torch
from package.resources.run import generate_data
from sbi import analysis as analysis #https://github.com/sbi-dev/sbi
from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi
from sbi.analysis import pairplot
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator
)
import numpy as np
import matplotlib.pyplot as plt
import json
from package.resources.utility import (
    produce_name_datetime,
    save_object,
    createFolder
)

def convert_data(data_to_fit):

    # Assuming `data_to_fit` is a numpy array of size (272,) representing monthly data from 2000 to 2022
    # Define the starting and ending indices for the years 2010 to 2022
    start_year = 2000
    end_year = 2022

    # Calculate the average of the last three months of each year
    averages = []

    #print("filtered_data", filtered_data)
    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2000) * 12
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        # Ensure the indices are within bounds
        last_three_months = data_to_fit[start_idx:end_idx]
        
        #print(f"Year: {year}, Start Index: {start_idx}, End Index: {end_idx}, Last Three Months: {last_three_months}")

        averages.append(np.mean(last_three_months))

    averages_array = np.array(averages)
    return averages_array

def simulator_wrapper(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict):
    def simulator_base(prior_sample):
        # Update base_params with the sampled values
        base_params[param_1_subdict][param_1_name] = prior_sample[0].item()
        base_params[param_2_subdict][param_2_name] = prior_sample[1].item()

        # Run the market simulation
        controller = generate_data(base_params)

        # Compute the financial market return
        arr_history = np.asarray(controller.social_network.history_prop_EV)
        data_to_fit = convert_data(arr_history)

        return torch.tensor(data_to_fit, dtype=torch.float32)
    return simulator_base

def gen_training_data(num_simulations,base_params, prior, param_1_name, param_2_name, param_1_subdict, param_2_subdict):
    """
    Ideally this function is entirely replaced by something i write or just load in data with 
    """
    #GENERATE THE SIMULATION DATA FIRST 

    # Check and process the prior # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    #print("prior, num_parameters, prior_returns_numpy", prior, num_parameters, prior_returns_numpy)
    # Create the simulator with additional parameters
    simulator_wrapped = simulator_wrapper(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict)
    #print("simulator_wrapped", simulator_wrapped)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(simulator_wrapped, prior, is_numpy_simulator=prior_returns_numpy)
    #print("simulator", simulator)
    
    # Check inputs for consistency# Consistency check after making ready for sbi.
    check_sbi_inputs(simulator, prior)

    # Instantiate the inference object
    inference = SNPE(prior=prior)
    #print("inference", inference)

    #we need simulations, or more specifically, pairs of parameters  which we sample from the prior and corresponding simulations 
    #The sbi helper function called simulate_for_sbi allows to parallelize your code with joblib
    # Perform simulations
    theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)
    #print("theta", theta)

    #print("x", x)

    return theta, x, simulator, inference, simulator_wrapped

def build_nerual_net(theta, x, inference):

    # Both theta and x should be a torch.Tensor of type float32
    # Train the posterior
    inference = inference.append_simulations(theta, x)

    #print("INFERENCE", inference)
    # train the neural density estimator to learn the association between the simulated data (or data features) and the underlying parameters
    density_estimator = inference.train(force_first_round_loss=True)
    #print("DENSITYT estimator", density_estimator)
    # use this density estimator to build the posterior distribution , i.e., the distributions over paramters  given observation 
    posterior = inference.build_posterior(density_estimator)
    #print("POSTERIOR",  posterior)
    print("DONE BUILDIGN POSTERIOR")

    return density_estimator, posterior

def plot_results(simulator,posterior, param_1_bounds, param_2_bounds):
    # True parameters
    param_1 = 0.9  # sigma_eta
    param_2 = 1    # beta

    # Evaluate the posterior with an observation
    observation = simulator(torch.tensor([[param_1, param_2]], dtype=torch.float32))#THIS IS RUNNING THE SIMULATION WITH THE ACTUAL VALUE! SPITS OUT WHAT I THINK THE TIME SERIES IS 
    #print("OBERSERVATION", observation )
    posterior_samples = posterior.sample((100000,), x=observation)#GET A BUNCH OF GUESSES AT THE POSTERIOR BASED ON THE OBSERVATIONS, WHAT DOES IT THINK THE VALUE IS
    #print("posterior_samples", posterior_samples)
    # Plot posterior samples
    fig, ax = pairplot(
        posterior_samples,
        limits=[param_1_bounds, param_2_bounds],
        figsize=(5, 5),
        points=[np.array([param_1, param_2])],
        points_colors='r',
        labels=[r'$\sigma_{\eta}$', r'$\sigma_{\nu}$']
    )
    #plt.savefig('Figures/posterior_samples_synthetic.png')
    plt.show()

def main(
        var_dict,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        num_simulations = 100,
        
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    #seed_repetitions = base_params["seed_repetitions"]

    root = "NN_calibration"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    ####################################################################################################
    #GENERATE THE DATA
    #THIS BIT REQUIRED FOR THE PRIOR
    # Bounds for the prior
    param_1_bounds = var_dict["param_1_bounds"]
    param_2_bounds = var_dict["param_2_bounds"]
    param_1_name = var_dict["param_1_name"]
    param_2_name = var_dict["param_2_name"]
    param_1_subdict = var_dict["param_1_subdict"]
    param_2_subdict = var_dict["param_2_subdict"]
    
    # Define the prior
    prior = utils.BoxUniform(
        low=torch.tensor([param_1_bounds[0], param_2_bounds[0]]),
        high=torch.tensor([param_1_bounds[1], param_2_bounds[1]])
    )
    theta, x, simulator, inference, simulator_wrapped = gen_training_data(num_simulations, base_params, prior, param_1_name, param_2_name, param_1_subdict, param_2_subdict)

    print("DONE PRODUCING DATA")

    ########################################################################################################
    #build NN

    density_estimator, posterior = build_nerual_net(theta, x, inference)

    createFolder(fileName)

    # Save the density_estimator and posterior
    save_object( posterior, fileName + "/Data", "posterior")
    save_object( prior, fileName + "/Data", "prior")
    save_object( var_dict, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    
if __name__ == "__main__":
    var_dict = {
        "param_1_bounds" : [0.01, 10],
        "param_2_bounds" : [0.01, 10],
        "param_1_name" : "a_innovativeness",
        "param_2_name" : "b_innovativeness",
        "param_1_subdict" : "parameters_social_network",
        "param_2_subdict" : "parameters_social_network",
    }
    #EV_percentage_2010_2022 = [0.003446, 0.026368, 0.081688, 0.225396, 0.455980, 0.680997, 0.913118, 1.147275, 1.583223, 1.952829, 2.217273, 2.798319, 3.791804, 5.166498]
    main(
        var_dict = var_dict,
        BASE_PARAMS_LOAD="package/constants/base_params_NN.json",
        num_simulations = 100
        )
    