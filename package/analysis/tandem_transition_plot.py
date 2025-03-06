from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import load_object

def plot_ev_uptake_vs_decarbonization(file_name,results, results_with_price, vary_single, dpi=600):
    """
    Plots EV uptake against Grid decarbonisation %s, distinguishing between scenarios with and without a carbon price.
    Includes shaded regions representing 1 standard deviation around the mean.
    
    Parameters:
    - file_name: The base file name where the results are saved.
    """
    # Load the results
    # Extract EV uptake and Grid decarbonisation %s
    ev_uptake_no_price = results[:, :, 0]  # EV uptake is the first column
    ev_uptake_with_price = results_with_price[:, :, 0]  # EV uptake is the first column
    
    # Calculate mean and standard deviation
    mean_ev = np.mean(ev_uptake_no_price, axis=1)

    std_ev = np.std(ev_uptake_no_price, axis=1)
    mean_ev_with_price = np.mean(ev_uptake_with_price, axis=1)
    std_ev_with_price = np.std(ev_uptake_with_price, axis=1)

    # Grid decarbonisation %s (x-axis)
    electricity_decarb_vals = np.linspace(vary_single["min"], vary_single["max"], vary_single["reps"])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot EV uptake without carbon price
    ax.plot(electricity_decarb_vals, mean_ev, label="No Carbon Price", color="blue", alpha=0.7, linewidth=2)
    ax.fill_between(electricity_decarb_vals, mean_ev - std_ev, mean_ev + std_ev, color="blue", alpha=0.2)
    
    # Plot EV uptake with carbon price
    ax.plot(electricity_decarb_vals, mean_ev_with_price, label="With Carbon Price", color="red", alpha=0.7, linewidth=2)
    ax.fill_between(electricity_decarb_vals, mean_ev_with_price - std_ev_with_price, mean_ev_with_price + std_ev_with_price, color="red", alpha=0.2)
    
    # Customize the plot
    ax.set_xlabel("Grid decarbonisation %", fontsize=12)
    ax.set_ylabel("EV Uptake", fontsize=12)
    ax.set_title("EV Uptake vs Grid decarbonisation %", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/ev_uptake_vs_decarbonization.png", dpi=dpi)

def plot_ev_uptake_vs_decarbonization_line(file_name, results, results_with_price, vary_single, dpi=600):
    """
    Plots EV uptake against Grid decarbonisation %s for the same seed, distinguishing between scenarios with and without a carbon price.
    Each seed is assigned a unique color, and the same color is used for both scenarios (solid for no carbon price, dashed for with carbon price).
    
    Parameters:
    - file_name: The base file name where the results are saved.
    - results: Results array for the scenario without a carbon price.
    - results_with_price: Results array for the scenario with a carbon price.
    - vary_single: Dictionary containing the range and repetitions for Grid decarbonisation %s.
    - dpi: Dots per inch for the saved figure.
    """
    # Extract EV uptake and Grid decarbonisation %s
    ev_uptake_no_price = results[:, :, 0]  # EV uptake is the first column
    ev_uptake_with_price = results_with_price[:, :, 0]  # EV uptake is the first column
    
    # Grid decarbonisation %s (x-axis)
    electricity_decarb_vals = np.linspace(vary_single["min"], vary_single["max"], vary_single["reps"])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the number of seeds
    num_seeds = ev_uptake_no_price.shape[1]
    
    # Create a colormap for unique colors for each seed
    colors = plt.cm.viridis(np.linspace(0, 1, num_seeds))
    
    # Plot EV uptake for each seed
    for seed_idx in range(num_seeds):
        # Assign a unique color for the current seed
        color = colors[seed_idx]
        
        # Plot EV uptake without carbon price for the current seed (solid line)
        ax.plot(electricity_decarb_vals, ev_uptake_no_price[:, seed_idx], 
                color=color, alpha=1, linewidth=1, linestyle="-", 
                label=f"Seed (No Carbon Price)" if seed_idx == 0 else "")
        
        # Plot EV uptake with carbon price for the current seed (dashed line)
        ax.plot(electricity_decarb_vals, ev_uptake_with_price[:, seed_idx], 
                color=color, alpha=1, linewidth=1, linestyle="--", 
                label=f"Seed (With Carbon Price 0.1 $/kgCO2)" if seed_idx == 0 else "")
    
    # Customize the plot
    ax.set_xlabel("Grid decarbonisation %", fontsize=12)
    ax.set_ylabel("EV Uptake", fontsize=12)
    ax.set_title("EV Uptake vs Grid decarbonisation % (Same Seed)", fontsize=14)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")  # Place legend outside the plot
    ax.grid(alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{file_name}/Plots/plot_ev_uptake_vs_decarbonization_line.png", dpi=dpi)



def main(file_name):
    base_params = load_object(file_name + "/Data", "base_params")
    vary_single = load_object(file_name + "/Data", "vary_single")
    results = load_object(file_name + "/Data", "results")
    results_with_price = load_object(file_name + "/Data", "results_with_price")
    
    #results = results.reshape(vary_single["reps"], base_params["seed_repetitions"], 4)
    #results_with_price = results_with_price.reshape(vary_single["reps"], base_params["seed_repetitions"], 4)


    plot_ev_uptake_vs_decarbonization(file_name,results, results_with_price, vary_single, dpi=300)
    plot_ev_uptake_vs_decarbonization_line(file_name,results, results_with_price, vary_single, dpi=300)
    plt.show()

# Example usage
if __name__ == "__main__":
    main(file_name = "results/vary_decarb_elec_10_32_45__06_03_2025")  # Replace with the actual file name
