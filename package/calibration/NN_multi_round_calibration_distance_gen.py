import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi
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
from package.calibration.NN_multi_round_calibration_gen import convert_data

#####################################################################################################################################
def simulator_wrapper_batch(base_params, param_list):
    def simulator_base_batch(prior_sample):
        # Ensure prior_sample is a 2D tensor (batch_size, num_params)
        if prior_sample.ndim == 1:  # Handle single sample case
            prior_sample = prior_sample.unsqueeze(0)

        # Update base_params for each sample in the batch
        results = []
        for sample in prior_sample:
            updated_params = base_params.copy()
            # Update the parameter with the sampled value
            for i, param in enumerate(param_list):
                subdict = param["subdict"]
                name = param["name"]
                updated_params[subdict][name] = sample[i].item()

            # Run the market simulation
            controller = generate_data(updated_params)

            # Compute the financial market return
            data_to_fit_distance = np.median(np.asarray(controller.social_network.history_distance_individual[-1]))
            # Compute the financial market return
            arr_history = np.asarray(controller.social_network.history_prop_EV)
            data_to_fit = convert_data(arr_history)

            # Convert EV_stock_prop_2010_22 to a tensor (if it isn't already)
            stock_tensor = torch.tensor(data_to_fit, dtype=torch.float32)
            # Ensure median_distance_traveled is a tensor
            median_distance_tensor = torch.tensor([data_to_fit_distance], dtype=torch.float32)
            # Concatenate the two to form x_o
            result = torch.cat(( stock_tensor, median_distance_tensor), dim=0)

            results.append(result)
        
        # Return a batched tensor
        stacked_results = torch.stack(results)
        
        return stacked_results
    
    return simulator_base_batch


#####################################################################################################################################
def gen_training_data(base_params, prior, param_list):
    """
    Ideally this function is entirely replaced by something i write or just load in data with 
    """
    #GENERATE THE SIMULATION DATA FIRST 
    
    # Check and process the prior # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    #print("prior", prior)

    # Create the simulator with additional parameters
    simulator_wrapped = simulator_wrapper_batch(base_params, param_list)

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
        parameters_list,
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

    # 2 rounds: first round simulates from the prior, second round simulates parameter set
    # that were sampled from the obtained posterior.
    num_rounds = 3
    # The specific observation we want to focus the inference on.
    median_distance_traveled = 1400
    # Convert EV_stock_prop_2010_22 to a tensor (if it isn't already)
    EV_stock_prop_2010_22_tensor = torch.tensor(EV_stock_prop_2010_22, dtype=torch.float32)

    # Ensure median_distance_traveled is a tensor
    median_distance_traveled_tensor = torch.tensor([median_distance_traveled], dtype=torch.float32)

    # Concatenate the two to form x_o
    x_o = torch.cat((EV_stock_prop_2010_22_tensor, median_distance_traveled_tensor), dim=0)

    #GENERATE THE DATA
    #THIS BIT REQUIRED FOR THE PRIOR
    # Bounds for the prior
    # Prepare tensors for low and high bounds
    low_bounds = torch.tensor([param["bounds"][0] for param in parameters_list])
    high_bounds = torch.tensor([param["bounds"][1] for param in parameters_list])

    # Define the prior
    prior = BoxUniform(low=low_bounds, high=high_bounds)
    print("READIED INPUTS")
    ########################################################################################################
    #build NN
    #ADJUST THIS FOR THE MULTIPLE ROUNDS, MAYBE START WITH SOME DATA FROM THE SENSTIVITY ANALYSIS
    
    inference, simulator_wrapped = gen_training_data(base_params, prior, parameters_list)
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
    save_object(parameters_list, fileName + "/Data", "var_dict")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(x_o, fileName + "/Data", "x_o")
    
if __name__ == "__main__":
    parameters_list =  [
        {
            "name": "a_innovativeness",
            "subdict": "parameters_social_network",
            "bounds": [0.01, 4]
        },
        {
            "name": "b_innovativeness",
            "subdict": "parameters_social_network",
            "bounds": [0.01, 4]
        },
        {
            "name": "min_Quality",
            "subdict": "parameters_ICE",
            "bounds": [0, 1000]
        },
        {
            "name": "max_Quality",
            "subdict": "parameters_ICE",
            "bounds": [1, 10000]
        },
        {
            "name": "min_Cost",
            "subdict": "parameters_ICE",
            "bounds": [0, 10000]
        },
        {
            "name": "max_Cost",
            "subdict": "parameters_ICE",
            "bounds": [1000, 100000]
        },
        {
            "name": "r",
            "subdict": "parameters_vehicle_user",
            "bounds": [0.01, 1]
        },
        {
            "name": "delta",
            "subdict": "parameters_ICE",
            "bounds": [10e-4, 10e-2]
        },
        {
            "name": "alpha",
            "subdict": "parameters_vehicle_user",
            "bounds": [0.1, 0.95]
        },
        {
            "name": "d_min",
            "subdict": "parameters_social_network",
            "bounds": [100, 1300]
        },
        {
            "name": "mu",
            "subdict": "parameters_vehicle_user",
            "bounds": [0.1, 1]
        },
    ]
    #EV_percentage_2010_2022 = [0.003446, 0.026368, 0.081688, 0.225396, 0.455980, 0.680997, 0.913118, 1.147275, 1.583223, 1.952829, 2.217273, 2.798319, 3.791804]
    main(
        parameters_list = parameters_list,
        BASE_PARAMS_LOAD="package/constants/base_params_NN_multi_round_distance.json",
        OUTPUTS_LOAD_ROOT = "package/calibration_data",
        OUTPUTS_LOAD_NAME = "calibration_data_output",
        num_simulations = 100
        )
    