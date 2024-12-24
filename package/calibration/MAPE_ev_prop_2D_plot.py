import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show

def calc_mape(actual, predicted):
    #print((actual - predicted) / actual)
    #print(np.mean(np.abs((actual - predicted) / actual)))

    return np.mean(np.abs((actual - predicted) / actual))

def plot_mape_heatmap(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    # Calculate MAPE for each parameter pair and average over seeds
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    mape_values = np.zeros((num_vary_1, num_vary_2))

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            # Extract predictions for current parameter pair
            predictions = data_array_ev_prop[i, j, :, :]  # Shape: (seeds, time steps)
            #print(predictions.shape)
            # Calculate MAPE for each seed and take the mean across seeds
            seed_mape = []
            #print("NEW", param_1,param_2)
            for i, pred in enumerate(predictions):
                sim_data = convert_data(pred,base_params)
                mape_data = calc_mape(real_data, sim_data)
                #print("mape_data", mape_data)
                seed_mape.append(mape_data)
            #print("np.mean(seed_mape)", np.mean(seed_mape))
            mape_values[i, j] = np.mean(seed_mape)

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(mape_values, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label="Mean MAPE (%)")
    
    # Add axis labels and titles
    ax.set_xticks(range(num_vary_2))
    ax.set_xticklabels(vary_2["property_list"], rotation=45)
    ax.set_yticks(range(num_vary_1))
    ax.set_yticklabels(vary_1["property_list"])
    ax.set_xlabel(vary_2["property_varied"])
    ax.set_ylabel(vary_1["property_varied"])
    ax.set_title("MAPE Heatmap: Real vs. Simulated EV Stock")
    
    # Save and show the heatmap
    save_and_show(fig, fileName, "mape_heatmap", dpi)

def plot_ev_stock_all_combinations(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])

    fig, axes = plt.subplots(
        nrows=num_vary_1, ncols=num_vary_2, figsize=(20, 20), sharex=True, sharey=True
    )

    x_values = np.arange(len(real_data))  # Assuming time series index

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            ax = axes[i, j]

            # Extract predictions for current parameter pair
            predictions = data_array_ev_prop[i, j, :, :]  # Shape: (seeds, time steps)
            processed_data_seeds = [convert_data(data, base_params) for data in predictions ]
            # Convert to numpy array and ensure consistent dimensions
            processed_data_array = np.array(processed_data_seeds)

            # Calculate mean and confidence intervals across seeds
            means = np.mean(processed_data_array, axis=0)
            confidence_intervals = t.ppf(0.975, len(processed_data_array)-1) * sem(processed_data_array, axis=0)

            lower_bounds = means - confidence_intervals
            upper_bounds = means + confidence_intervals

            # Plot mean and confidence intervals
            ax.plot(x_values, means, color='blue')
            ax.fill_between(x_values, lower_bounds, upper_bounds, color='blue', alpha=0.2)
            ax.plot(x_values, real_data, color='orange', linestyle='--', linewidth=2)

            # for major ticks
            ax.set_xticks([])
            # for minor ticks
            ax.set_xticks([], minor=True)
            # for major ticks
            ax.set_yticks([])
            # for minor ticks
            ax.set_yticks([], minor=True)

            # Title with parameter combinations
            #ax.set_title(f"{round(param_1,2)}, {round(param_2,2)}")

            # Remove legends and adjust grid
            #ax.grid(True)

    # Optional: Adjust axis labels only for edge plots
    fig.supxlabel("Months (2010-2022)")
    fig.supylabel("EV Stock %")

    # Adjust layout and save the figure
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_ev_stock_combinations", dpi)

# Main function
def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_ev_prop = load_object(fileName + "/Data", "data_array_ev_prop")
        vary_1 = load_object(fileName + "/Data", "vary_1")
        vary_2 = load_object(fileName + "/Data", "vary_2")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")

    # Extract actual EV stock proportions (2010-2022)
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    # Plot MAPE heatmap
    plot_mape_heatmap(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_ev_stock_all_combinations(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plt.show()
if __name__ == "__main__":
    main("results/MAPE_ev_2D_12_03_55__24_12_2024")
