import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show

def calc_mape(actual, predicted):
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
            
            # Calculate MAPE for each seed and take the mean across seeds
            seed_mape = []
            for i, pred in enumerate(predictions):
                sim_data = convert_data(pred,base_params)
                mape_data = calc_mape(real_data, sim_data)
                seed_mape.append(mape_data)
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
    plt.show()
if __name__ == "__main__":
    main("results/MAPE_ev_2D_11_40_31__24_12_2024")
