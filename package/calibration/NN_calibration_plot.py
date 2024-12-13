import torch
from sbi import analysis as analysis #https://github.com/sbi-dev/sbi
from sbi import utils as utils
from sbi.analysis import pairplot
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator
)
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object
)
from package.calibration.NN_calibration_gen import simulator_wrapper

def plot_results(simulator,posterior, param_1_bounds, param_2_bounds, param_1_name,param_2_name):
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
        labels=[param_1_name,param_2_name]
    )
    #plt.savefig('Figures/posterior_samples_synthetic.png')
    plt.show()

def main(
        fileName,
    ) -> str: 

    posterior = load_object(fileName + "/Data", "posterior")
    prior = load_object(fileName + "/Data", "prior")
    var_dict = load_object(fileName + "/Data", "var_dict")
    base_params = load_object(fileName + "/Data", "base_params")

    param_1_bounds = var_dict["param_1_bounds"]
    param_2_bounds = var_dict["param_2_bounds"]
    param_1_name = var_dict["param_1_name"]
    param_2_name = var_dict["param_2_name"]
    param_1_subdict = var_dict["param_1_subdict"]
    param_2_subdict = var_dict["param_2_subdict"]

    # Check and process the prior # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapped = simulator_wrapper(base_params, param_1_name, param_2_name, param_1_subdict, param_2_subdict)
    simulator = process_simulator(simulator_wrapped, prior, is_numpy_simulator=prior_returns_numpy)

    plot_results(simulator,posterior, param_1_bounds, param_2_bounds,param_1_name,param_2_name)

if __name__ == "__main__":

    #EV_percentage_2010_2022 = [0.003446, 0.026368, 0.081688, 0.225396, 0.455980, 0.680997, 0.913118, 1.147275, 1.583223, 1.952829, 2.217273, 2.798319, 3.791804, 5.166498]
    main(fileName = "results/NN_calibration_13_26_25__13_12_2024")
    