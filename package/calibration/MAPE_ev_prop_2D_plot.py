import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import sem, t
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show

# Vectorized MAPE calculation
def calc_mape_vectorized(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual), axis=1) * 100

# Symmetric MAPE (alternative metric)
def calc_smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

# Mean Squared Error (alternative metric)
def calc_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2, axis=1)

# Root Mean Squared Error (alternative metric)
def calc_rmse(actual, predicted):
    return np.sqrt(calc_mse(actual, predicted))

# Save best parameter combination to JSON
def save_best_parameters(best_params, best_metric, metric_name, vary_1, vary_2, file_path):
    best_fit_info = {
        "best_parameters": {
            vary_1["property_varied"]: best_params[0],
            vary_2["property_varied"]: best_params[1]
        },
        f"best_{metric_name}": best_metric
    }
    with open(file_path, 'w') as f:
        json.dump(best_fit_info, f, indent=4)

# Generic function to plot heatmaps for different metrics
def plot_metric_heatmap(metric_function, metric_name, base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    metric_values = np.zeros((num_vary_1, num_vary_2))

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            predictions = data_array_ev_prop[i, j]  # Shape: (seeds, time steps)
            sim_data_array = np.array([convert_data(pred, base_params) for pred in predictions])
            metric_data = metric_function(real_data, sim_data_array)
            metric_values[i, j] = np.mean(metric_data)

    # Find best parameter combination
    best_idx = np.unravel_index(np.argmin(metric_values), metric_values.shape)
    best_params = (vary_1["property_list"][best_idx[0]], vary_2["property_list"][best_idx[1]])

    print(f"Best parameter combination: {vary_1['property_varied']} = {best_params[0]:.3g}, "
          f"{vary_2['property_varied']} = {best_params[1]:.3g} with {metric_name.upper()} = {metric_values[best_idx]:.2f}")

    # Save best parameters to a JSON file
    save_best_parameters(best_params, metric_values[best_idx], metric_name, vary_1, vary_2, f"{fileName}/best_fit_{metric_name}.json")

    # Plot static heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(metric_values, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label=f"Mean {metric_name.upper()}")

    ax.set_xticks(range(num_vary_2))
    ax.set_xticklabels([f"{val:.3g}" for val in vary_2["property_list"]], rotation=45)
    ax.set_yticks(range(num_vary_1))
    ax.set_yticklabels([f"{val:.3g}" for val in vary_1["property_list"]])
    ax.set_xlabel(f"{vary_2['property_varied']}")
    ax.set_ylabel(f"{vary_1['property_varied']}")
    ax.set_title(f"{metric_name.upper()} Heatmap: Real vs. Simulated EV Stock")

    save_and_show(fig, fileName, f"{metric_name}_heatmap", dpi)

# Plot best parameters from all metrics
def plot_best_parameters_all_metrics(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    metrics = ["mape", "smape", "mse", "rmse"]
    metric_functions = {
        "mape": calc_mape_vectorized,
        "smape": calc_smape,
        "mse": calc_mse,
        "rmse": calc_rmse
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()
    x_values = np.arange(len(real_data))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Load best parameters from JSON
        with open(f"{fileName}/best_fit_{metric}.json", 'r') as f:
            best_fit_info = json.load(f)

        param_1 = best_fit_info["best_parameters"][vary_1["property_varied"]]
        param_2 = best_fit_info["best_parameters"][vary_2["property_varied"]]

        # Find corresponding predictions
        i = np.where(vary_1["property_list"] == param_1)[0][0]
        j = np.where(vary_2["property_list"] == param_2)[0][0]
        predictions = data_array_ev_prop[i, j]  # Shape: (seeds, time steps)
        sim_data_array = np.array([convert_data(pred, base_params) for pred in predictions])

        # Plot all seed traces
        for seed_data in sim_data_array:
            ax.plot(x_values, seed_data, color='blue', alpha=0.3)

        # Plot mean across seeds
        mean_data = np.mean(sim_data_array, axis=0)
        ax.plot(x_values, mean_data, color='blue', linewidth=2, label='Mean Simulation')

        # Plot real data
        ax.plot(x_values, real_data, color='orange', linestyle='--', linewidth=2, label='Real Data')

        ax.set_title(f"Best Fit ({metric.upper()})")
        ax.set_xlabel("Time")
        ax.set_ylabel("EV Stock %")
        ax.legend()

    save_and_show(fig, fileName, "best_parameters_all_metrics", dpi)

# Main function
def main(fileName, dpi=600):
    try:
        serial_data = load_object(fileName + "/Data", "data_flat_ev_prop")
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

    # Plot heatmaps for different metrics
    plot_metric_heatmap(calc_mape_vectorized, "mape", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_metric_heatmap(calc_smape, "smape", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_metric_heatmap(calc_mse, "mse", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_metric_heatmap(calc_rmse, "rmse", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)

    # Plot best parameters from all metrics
    plot_best_parameters_all_metrics(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)

    plt.show()

if __name__ == "__main__":
    main("results/MAPE_ev_2D_20_14_43__10_02_2025")
