import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import load_object

def plot_ev_uptake_vs_decarbonization(file_name):
    """
    Plots EV uptake against decarbonization values, distinguishing between scenarios with and without a carbon price.
    
    Parameters:
    - file_name: The base file name where the results are saved.
    """
    # Load the results
    base_params = load_object(file_name + "/Data", "base_params")
    vary_single = load_object(file_name + "/Data", "vary_single")
    results = load_object(file_name + "/Data", "results")
    results =results.reshape( vary_single["reps"], base_params["seed_repetitions"], 4)
    results_with_price = load_object(file_name + "/Data", "results_with_price")
    results_with_price = results_with_price.reshape( vary_single["reps"], base_params["seed_repetitions"], 4)

    # Extract EV uptake and decarbonization values
    ev_uptake_no_price = results[:, :, 0]  # EV uptake is the first column
    ev_uptake_with_price = results_with_price[:, :, 0]  # EV uptake is the first column
    
    mean_ev = np.mean(ev_uptake_no_price, axis = 1)
    mean_ev_with_price = np.mean(ev_uptake_with_price, axis = 1)

    # Decarbonization values (x-axis)
    electricity_decarb_vals = np.linspace(vary_single["min"], vary_single["max"], vary_single["reps"])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot EV uptake without carbon price
    plt.plot(electricity_decarb_vals, mean_ev, label="No Carbon Price", color="blue", alpha=0.7, linewidth=2)
    
    # Plot EV uptake with carbon price
    plt.plot(electricity_decarb_vals, mean_ev_with_price, label="With Carbon Price", color="red", alpha=0.7, linewidth=2)
    
    # Customize the plot
    plt.xlabel("Decarbonization Value", fontsize=12)
    plt.ylabel("EV Uptake", fontsize=12)
    plt.title("EV Uptake vs Decarbonization Value", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/ev_uptake_vs_decarbonization.png", dpi=600)


# Example usage
if __name__ == "__main__":
    file_name = "results/vary_decarb_elec_17_39_47__05_03_2025"  # Replace with the actual file name
    plot_ev_uptake_vs_decarbonization(file_name)
    plt.show()