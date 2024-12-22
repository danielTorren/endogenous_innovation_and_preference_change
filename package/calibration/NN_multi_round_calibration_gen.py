import torch
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import json
from package.resources.utility import (
    produce_name_datetime,
    save_object,
    createFolder,
    load_object
)
from package.resources.run import generate_data
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

###################################################################################################################################################
#BUILD THE NEURAL NET

def convert_data(data_to_fit, base_params):

    # Assuming `data_to_fit` is a numpy array of size (272,) representing monthly data from 2000 to 2022
    # Define the starting and ending indices for the years 2010 to 2022
    start_year = 2010
    end_year = 2022

    # Calculate the average of the last three months of each year
    averages = []

    #print("filtered_data", filtered_data)
    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2000) * 12 + base_params["duration_burn_in"]
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        # Ensure the indices are within bounds
        last_three_months = data_to_fit[start_idx:end_idx]
        
        #print(f"Year: {year}, Start Index: {start_idx}, End Index: {end_idx}, Last Three Months: {last_three_months}")

        averages.append(np.mean(last_three_months))

    averages_array = np.array(averages)

    return averages_array

#####################################################################################################################################
def simulator_wrapper_single(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict):
    def simulator_base_single(prior_sample):
        # Ensure prior_sample is a 1D tensor
        if prior_sample.ndim > 1:
            raise ValueError("Batch processing is not supported. Pass one sample at a time.")

        # Update base_params with the sampled values
        updated_params = base_params.copy()
        updated_params[param_1_subdict][param_1_name] = prior_sample[0].item()
        updated_params[param_2_subdict][param_2_name] = prior_sample[1].item()

        # Run the market simulation
        controller = generate_data(updated_params)

        # Compute the financial market return
        arr_history = np.asarray(controller.social_network.history_prop_EV)
        data_to_fit = convert_data(arr_history, base_params)

        return torch.tensor(data_to_fit, dtype=torch.float32)
    
    return simulator_base_single

def simulator_wrapper_batch(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict):
    def simulator_base_batch(prior_sample):
        # Ensure prior_sample is a 2D tensor (batch_size, num_params)
        if prior_sample.ndim == 1:  # Handle single sample case
            prior_sample = prior_sample.unsqueeze(0)

        # Update base_params for each sample in the batch
        results = []
        for sample in prior_sample:
            updated_params = base_params.copy()
            updated_params[param_1_subdict][param_1_name] = sample[0].item()
            updated_params[param_2_subdict][param_2_name] = sample[1].item()

            # Run the market simulation
            controller = generate_data(updated_params)

            # Compute the financial market return
            arr_history = np.asarray(controller.social_network.history_prop_EV)
            data_to_fit = convert_data(arr_history)
            results.append(torch.tensor(data_to_fit, dtype=torch.float32))
            #print("stacked_results", len(data_to_fit))
        
        # Return a batched tensor
        stacked_results = torch.stack(results)
        
        return stacked_results
    
    return simulator_base_batch

def simulator_wrapper_batch_parallel(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict):
    def simulator_base_batch_parallel(prior_sample):
        # Ensure prior_sample is a 2D tensor (batch_size, num_params)
        if prior_sample.ndim == 1:  # Handle single sample case
            prior_sample = prior_sample.unsqueeze(0)

        def run_simulation(sample):
            # Create a copy of base_params for this sample
            updated_params = base_params.copy()
            updated_params[param_1_subdict][param_1_name] = sample[0].item()
            updated_params[param_2_subdict][param_2_name] = sample[1].item()

            # Run the market simulation
            controller = generate_data(updated_params)

            # Compute the financial market return
            arr_history = np.asarray(controller.social_network.history_prop_EV)
            data_to_fit = convert_data(arr_history)
            
            return torch.tensor(data_to_fit, dtype=torch.float32)

        # Run simulations in parallel
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose=10)(
            delayed(run_simulation)(sample) for sample in prior_sample
        )

        # Return a batched tensor
        stacked_results = torch.stack(results)
        return stacked_results

    return simulator_base_batch_parallel

#####################################################################################################################################
def gen_training_data(base_params, prior, param_1_name, param_2_name, param_1_subdict, param_2_subdict):
    """
    Ideally this function is entirely replaced by something i write or just load in data with 
    """
    #GENERATE THE SIMULATION DATA FIRST 
    
    # Check and process the prior # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    #print("prior", prior)

    # Create the simulator with additional parameters
    #simulator_wrapped = simulator_wrapper_single(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict)
    #simulator_wrapped = simulator_wrapper_batch(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict)
    simulator_wrapped = simulator_wrapper_batch(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict)
    #print("simulator_wrapped", simulator_wrapped)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(simulator_wrapped, prior, is_numpy_simulator=prior_returns_numpy)
    #print("simulator", simulator)
    
    # Check inputs for consistency# Consistency check after making ready for sbi.
    check_sbi_inputs(simulator, prior)

    # Instantiate the inference object
    inference = NPE(prior=prior)

    return inference, simulator_wrapped

###################################################################################################################################################
#RUN MULTIPLE ROUNDS
def run_simulation_rounds(num_rounds , x_o, prior, simulator_wrapped, inference, num_simulations):
    #FROM DOCUMENTEATION: Note that, for num_rounds>1, the posterior is no longer amortized: it will give good results when sampled around x=observation, but possibly bad results for other x

    posteriors = []
    proposal = prior

    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator_wrapped, proposal, num_simulations=num_simulations)
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    return posterior 

#########################################################################################################

def main(
        var_dict,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT = "package/calibration_data",
        OUTPUTS_LOAD_NAME = "calibration_data_output",
        num_simulations = 100,
        
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    #load output
    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)

    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

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
    
    # 2 rounds: first round simulates from the prior, second round simulates parameter set
    # that were sampled from the obtained posterior.
    num_rounds = 5
    # The specific observation we want to focus the inference on.
    x_o = torch.tensor(EV_stock_prop_2010_22)#THIS MAY NEED TO BE DIFFERENT
    #print("x_o", len(x_o))

    # Define the prior
    prior = BoxUniform(
        low=torch.tensor([param_1_bounds[0], param_2_bounds[0]]),
        high=torch.tensor([param_1_bounds[1], param_2_bounds[1]])
    )
    print("READIED INPUTS")
    ########################################################################################################
    #build NN
    #ADJUST THIS FOR THE MULTIPLE ROUNDS, MAYBE START WITH SOME DATA FROM THE SENSTIVITY ANALYSIS
    
    inference, simulator_wrapped = gen_training_data(base_params, prior, param_1_name, param_2_name, param_1_subdict, param_2_subdict)
    print("DONE TRAINING")
    #quit()
    ########################################################################################################
    #produce the data to update the nerual net
    posterior = run_simulation_rounds(num_rounds, x_o, prior, simulator_wrapped, inference, num_simulations)

    print("DONE PRODUCING DATA")

    createFolder(fileName)

    # Save the density_estimator and posterior
    save_object( posterior, fileName + "/Data", "posterior")
    save_object( prior, fileName + "/Data", "prior")
    save_object( var_dict, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")
    
if __name__ == "__main__":
    var_dict = {
        "param_1_bounds" : [0.01, 4],
        "param_2_bounds" : [0.01, 4],
        "param_1_name" : "a_innovativeness",
        "param_2_name" : "b_innovativeness",
        "param_1_subdict" : "parameters_social_network",
        "param_2_subdict" : "parameters_social_network",
    }
    #EV_percentage_2010_2022 = [0.003446, 0.026368, 0.081688, 0.225396, 0.455980, 0.680997, 0.913118, 1.147275, 1.583223, 1.952829, 2.217273, 2.798319, 3.791804]
    main(
        var_dict = var_dict,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round.json",
        OUTPUTS_LOAD_ROOT = "package/calibration_data",
        OUTPUTS_LOAD_NAME = "calibration_data_output",
        num_simulations = 100
        )
    