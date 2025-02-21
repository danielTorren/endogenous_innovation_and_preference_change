from sbi import analysis as analysis  # https://github.com/sbi-dev/sbi
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from package.resources.utility import load_object, save_object
from package.plotting_data.single_experiment_plot import save_and_show
import torch

def plot_results(fileName, posterior_samples, x_o, posterior, param_bounds, param_names):
    """
    Plots results for posterior samples, dynamically handling multiple parameters.
    
    Args:
        fileName (str): The output filename for saving the plot.
        x_o (torch.Tensor): Observed data tensor.
        posterior (object): The posterior object from sbi.
        param_bounds (list): List of parameter bounds, one per parameter.
        param_names (list): List of parameter names, one per parameter.
    """
    # Sample posterior
    #posterior_samples = posterior.sample((100000,), x=x_o)  # Get posterior samples based on observations

    # Generate pairplot
    fig, ax = pairplot(
        posterior_samples,
        limits=param_bounds,
        figsize=(10, 10),  # Adjust size based on number of parameters
        points_colors='r',
        labels=param_names
    )

    # Save and show plot
    save_and_show(fig, fileName, "pairplot", dpi=300)
    plt.show()

def main(fileName):
    """
    Main function to load data and plot results.
    
    Args:
        fileName (str): Base directory for data and outputs.
        OUTPUTS_LOAD_ROOT (str): Root path for loading calibration data.
        OUTPUTS_LOAD_NAME (str): File name for calibration data.
    """
    # Load observed data

    match_data = load_object(fileName + "/Data", "match_data")


    # Extract observed statistics
    EV_stock_prop_2010_22 = match_data["EV_stock_prop_2010_22"]
    #median_distance_traveled = match_data["median_distance_traveled"]
    #median_age = match_data["median_age"]
    #median_price = match_data["median_price"]

    # Convert data to tensors
    EV_stock_prop_2010_22_tensor = torch.tensor(EV_stock_prop_2010_22, dtype=torch.float32)
    #median_distance_traveled_tensor = torch.tensor([median_distance_traveled], dtype=torch.float32)
    #median_age_tensor = torch.tensor([median_age], dtype=torch.float32)
    #median_price_tensor = torch.tensor([median_price], dtype=torch.float32)

    # Reconstruct x_o by concatenating the tensors
    #x_o = torch.cat((EV_stock_prop_2010_22_tensor, 
    #                 median_distance_traveled_tensor, 
    #                 median_age_tensor, 
    #                 median_price_tensor), dim=0)
    x_o = EV_stock_prop_2010_22_tensor

    # Load posterior and variable dictionary
    posterior = load_object(fileName + "/Data", "posterior")
    var_dict = load_object(fileName + "/Data", "var_dict")
    samples = load_object(fileName + "/Data", "samples")
    #best_sample = load_object(fileName + "/Data", "best_sample")

    # Extract parameter bounds and names dynamically
    param_bounds = [p["bounds"] for p in var_dict]
    param_names = [p["name"] for p in var_dict]

    # Test posterior samples and plot results
    #samples = posterior.sample((10000000,), x=x_o)
    #log_probability_samples = posterior.log_prob(samples, x=x_o)
    #print("Log probabilities:", log_probability_samples)

    # Find sample with greatest log probability
    #max_log_prob_index = log_probability_samples.argmax()
    #best_sample = samples[max_log_prob_index]
    #print("Sample with the greatest log probability:", best_sample)
    #print("Greatest log probability:", log_probability_samples[max_log_prob_index])

    #save_object(best_sample, fileName + "/Data", "best_sample")
    
    # Plot results
    plot_results(fileName, samples, x_o, posterior, param_bounds, param_names)

if __name__ == "__main__":
    main(
        fileName="results/NN_calibration_multi_10_50_01__21_02_2025",
    )
