from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from package.resources.utility import load_object, save_object
from package.plotting_data.single_experiment_plot import save_and_show
import torch
from torch import multiprocessing

def reconstruct_x_o(match_data):
    """
    Rebuild the observed summary statistic vector from a saved match_data dict.

    Handles the layouts still being plotted:
      - current: EV stock proportion, EV sales proportion, then the extra scalar
        moments under "extra_scalar_targets" -- the order the gen script builds
        x_o in. That is mean fleet age alone now; runs made while new-car HHI was
        also a moment carry two entries there, and both work unchanged because
        the block is concatenated whatever its length.
      - older: EV stock then EV sales only, with no extra moments.
      - legacy: a single EV stock series under whatever year range that run
        used (e.g. "EV_stock_prop_2016_23"), with no sales channel at all.

    Runs whose x layout is not recoverable this way (e.g. the intermediate
    version that repeated the age and HHI targets once per year) save the tensor
    itself as "x_o" -- load_x_o() prefers that.

    Args:
        match_data (dict): Observed data saved alongside the posterior.

    Returns:
        torch.Tensor: Observed data, matching the trained posterior's x layout.
    """
    stock_keys = sorted(k for k in match_data if k.startswith("EV_stock_prop"))
    sales_keys = sorted(k for k in match_data if k.startswith("EV_sales_prop"))

    if not stock_keys:
        raise KeyError(f"no EV stock series in match_data; keys were {sorted(match_data)}")

    parts = [match_data[k] for k in stock_keys + sales_keys]
    used = stock_keys + sales_keys

    if "extra_scalar_targets" in match_data:
        parts.append(match_data["extra_scalar_targets"])
        used.append("extra_scalar_targets")

    print("x_o built from:", used)

    return torch.cat([torch.tensor(p, dtype=torch.float32).reshape(-1) for p in parts], dim=0)


def load_x_o(fileName, match_data):
    """
    Return the run's observed vector, preferring the saved tensor.

    The gen script saves x_o directly, so use that when it is there and fall back
    to rebuilding it from match_data for runs that predate it.
    """
    try:
        x_o = load_object(fileName + "/Data", "x_o")
        print("x_o loaded from saved tensor, shape", tuple(x_o.shape))
        return x_o
    except FileNotFoundError:
        return reconstruct_x_o(match_data)

def plot_results(fileName, posterior_samples, param_bounds, param_names):
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
    base_params = load_object(fileName + "/Data", "base_params")
    print(base_params)
    

    x_o = load_x_o(fileName, match_data)

    # Load posterior and variable dictionary
    posterior = load_object(fileName + "/Data", "posterior")
    var_dict = load_object(fileName + "/Data", "var_dict")
    samples = load_object(fileName + "/Data", "samples")
    #best_sample = load_object(fileName + "/Data", "best_sample")

    # Extract parameter bounds and names dynamically
    param_bounds = [p["bounds"] for p in var_dict]
    param_names = [p["name"] for p in var_dict]

    # Test posterior samples and plot results

    # Set the number of threads for CPU parallelism
    #torch.set_num_threads(multiprocessing.cpu_count())
    # Move the posterior and data to the GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #posterior = posterior.to(device)
    #x_o = x_o.to(device)

    # Sample in parallel (PyTorch handles parallelism internally)
    #samples = posterior.sample((100000,), x=x_o)

    #samples = posterior.sample((16,), x=x_o)

    #save_object(samples, fileName + "/Data", "samples")


    log_probability_samples = posterior.log_prob(samples, x=x_o)
    #save_object(log_probability_samples, fileName + "/Data", "log_probability_samples")
    
    #print("Log probabilities:", log_probability_samples)

    # Find sample with greatest log probability
    max_log_prob_index = log_probability_samples.argmax()
    best_sample = samples[max_log_prob_index]
    print("Sample with the greatest log probability:", best_sample)
    #print("Greatest log probability:", log_probability_samples[max_log_prob_index])

    #save_object(best_sample, fileName + "/Data", "best_sample")
    
    # Plot results
    plot_results(fileName, samples, param_bounds, param_names)

if __name__ == "__main__":
    main(
        fileName="results/sbi_seed_av_08_58_38__17_08_2026",
    )
    #sbi_seed_av_08_58_38__17_08_2026
    #sbi_seed_av_10_13_07__17_08_2026 - this one has the right range and b beta fixed
    #sbi_seed_av_09_15_32__17_08_2026
    #sbi_seed_av_22_27_30__16_08_2026
    #sbi_seed_av_22_25_54__16_08_2026
    #sbi_seed_av_17_44_11__16_08_2026
    #sbi_single_seed_14_34_13__14_08_2026
    #NN_calibration_multi_14_33_36__14_08_2026
    #NN_calibration_multi_13_02_48__14_08_2026
    #sbi_single_seed_13_04_51__14_08_2026
    #sbi_single_seed_12_42_40__14_08_2026
    #NN_calibration_multi_12_34_29__14_08_2026

    #sbi_single_seed_16_43_32__12_08_2026
    #NN_calibration_multi_16_42_17__12_08_2026
    #sbi_single_seed_14_23_24__11_08_2026
    #sbi_single_seed_15_18_22__11_08_2026
    #sbi_single_seed_15_38_23__11_08_2026
    #NN_calibration_multi_12_41_07__11_08_2026 - OLD EFFICICENCY
    #NN_calibration_multi_12_25_22__11_08_2026 - UPDATED efficiency 
    #NN_calibration_multi_08_14_47__07_08_2026
#NN_calibration_multi_12_43_09__06_08_2026
#NN_calibration_multi_11_08_28__20_03_2025