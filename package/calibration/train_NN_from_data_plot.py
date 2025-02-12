import torch
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from package.resources.utility import load_object, createFolder, save_object

def load_params(file_path):
    """
    Load posterior samples and parameter details from the first script's output.
    
    Args:
        file_path (str): Path to the directory containing the data.
        
    Returns:
        samples (torch.Tensor): Posterior samples.
        param_bounds (list): List of parameter bounds.
        param_names (list): List of parameter names.
    """

    
    # Load parameter variation details
    vary_1 = load_object(file_path + "/Data", "vary_1")
    vary_2 = load_object(file_path + "/Data", "vary_2")
    vary_3 = load_object(file_path + "/Data", "vary_3")
    
    param_bounds = [[vary_1["min"], vary_1["max"]],
                    [vary_2["min"], vary_2["max"]],
                    [vary_3["min"], vary_3["max"]]]
    
    param_names = [vary_1["name"], vary_2["name"], vary_3["name"]]
    
    return param_bounds, param_names


def plot_posterior_samples(samples, param_bounds, param_names, output_file_path):
    """
    Generate and save pair plot for posterior samples.
    
    Args:
        samples (torch.Tensor): Posterior samples.
        param_bounds (list): List of parameter bounds.
        param_names (list): List of parameter names.
        output_file_path (str): Path to save the output plot.
    """
    fig, ax = pairplot(
        samples,
        limits=param_bounds,
        figsize=(10, 10),
        points_colors='r',
        labels=param_names
    )

    createFolder(output_file_path)
    fig.savefig(f"{output_file_path}/pairplot.png", dpi=300)
    plt.show()


def main(file_path):
    """
    Main function to load samples and plot results.
    
    Args:
        file_path (str): Path to the directory containing the data.
    """
    real_data = load_object(file_path + "/Data", "real_data")
    posterior = load_object(file_path + "/Data", "posterior")
    X = load_object(file_path + "/Data", "X")

    conditioned_posterior = posterior.condition(x=real_data.unsqueeze(0))
    samples = conditioned_posterior.sample((100000,))
    quit()


    samples = posterior.sample((100000,), x=real_data.unsqueeze(0))

    
    
    save_object(samples, f"{file_path}/Data", "samples")

    samples, param_bounds, param_names = load_params(file_path)
    plot_posterior_samples(samples, param_bounds, param_names, file_path)


if __name__ == "__main__":
    main("results/MAPE_ev_3D_17_21_36__11_02_2025")
