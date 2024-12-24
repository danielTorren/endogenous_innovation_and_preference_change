import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.plotting_data.single_experiment_plot import save_and_show

def plot_ev_stock_multi_seed(base_params, real_data, simulated_data_seeds, fileName, dpi=600):
    # Process simulated data seeds
    processed_data_seeds = [convert_data(data, base_params) for data in simulated_data_seeds]
    
    # Convert to numpy array and ensure consistent dimensions
    processed_data_array = np.array(processed_data_seeds)
    #print("processed_data_array", processed_data_array)

    # Calculate means and confidence intervals
    means = np.mean(processed_data_array, axis=0)
    confidence_intervals = t.ppf(0.975, len(processed_data_array)-1) * sem(processed_data_array, axis=0)


    lower_bounds = means - confidence_intervals
    upper_bounds = means + confidence_intervals

    # Plot data
    fig, ax = plt.subplots(figsize=(8, 6))
    x_values = np.arange(len(means))  # Assuming time series index
    ax.plot(x_values, means, label="Simulated Data", color='blue')
    ax.fill_between(x_values, lower_bounds, upper_bounds, color='blue', alpha=0.2)
    ax.plot(x_values, real_data, label="California Data", color='orange')
    ax.set_xlabel("Months (2010-2022)")
    ax.set_ylabel("EV Stock %")
    ax.legend(loc="best")
    ax.grid(True)

    # Save and show the plot
    save_and_show(fig, fileName, "plot_ev_stock", dpi)

def plot_ev_stock_multi_seed_pair(base_params, real_data, simulated_data_seeds, fileName, dpi=600):
    # Process simulated data seeds
    processed_data_seeds = [convert_data(data, base_params) for data in simulated_data_seeds]
    
    # Convert to numpy array and ensure consistent dimensions
    processed_data_array = np.array(processed_data_seeds)
    #print("processed_data_array", processed_data_array)

    # Calculate means and confidence intervals
    means = np.mean(processed_data_array, axis=0)
    confidence_intervals = t.ppf(0.975, len(processed_data_array)-1) * sem(processed_data_array, axis=0)

    lower_bounds = means - confidence_intervals
    upper_bounds = means + confidence_intervals

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # First subplot: Mean and confidence intervals
    x_values = np.arange(len(means))  # Assuming time series index
    ax1.plot(x_values, means, label="Simulated Data (Mean)", color='blue')
    ax1.fill_between(x_values, lower_bounds, upper_bounds, color='blue', alpha=0.2)
    ax1.plot(x_values, real_data, label="California Data", color='orange')
    ax1.set_xlabel("Year (2010-2022)")
    ax1.set_ylabel("EV Stock %")
    ax1.legend(loc="best")
    ax1.grid(True)
    ax1.set_title("Mean Simulated Data with Confidence Intervals")

    # Second subplot: Individual runs
    for run in processed_data_array:
        ax2.plot(x_values, run, alpha=0.7, linewidth=0.8)
    ax2.plot(x_values, real_data, label="California Data", color='orange', linewidth=2, linestyle='--')
    ax2.set_xlabel("Months (2010-2022)")
    ax2.set_ylabel("EV Stock %")
    ax2.legend(loc="best")
    ax2.grid(True)
    ax2.set_title("Individual Simulated Runs")

    # Adjust layout and save the plot
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_ev_stock_pair", dpi)


# Sample main function
def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_ev_prop = load_object(fileName + "/Data", "data_array_ev_prop")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")

    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    #plot_ev_stock_multi_seed(base_params, EV_stock_prop_2010_22, data_array_ev_prop, fileName, dpi)
    plot_ev_stock_multi_seed_pair(base_params, EV_stock_prop_2010_22, data_array_ev_prop, fileName, dpi)
    plt.show()

if __name__ == "__main__":
    main("results/multi_seed_single_10_42_49__24_12_2024")