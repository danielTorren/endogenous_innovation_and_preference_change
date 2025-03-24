import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import sem, t
#from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show


def convert_data_short(data_to_fit, base_params):

    # Assuming `data_to_fit` is a numpy array of size (272,) representing monthly data from 2001 to 2022
    # Define the starting and ending indices for the years 2010 to 2022
    start_year = 2016
    end_year = 2022

    # Calculate the average of the last three months of each year
    averages = []

    #print("filtered_data", filtered_data)
    for year in range(start_year, end_year + 1):
        year_start_index = (year - 2001) * 12 + base_params["duration_burn_in"]#ADD ON THE BURN IN PERIOD TO THE START
        start_idx = year_start_index + 9  # October index
        end_idx = year_start_index + 12  # December index (exclusive)
        # Ensure the indices are within bounds
        last_three_months = data_to_fit[start_idx:end_idx]
        
        averages.append(np.mean(last_three_months))

    averages_array = np.array(averages)

    return averages_array

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
def plot_metric_heatmap(metric_function, metric_name, base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=300):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    metric_values = np.zeros((num_vary_1, num_vary_2))

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            predictions = data_array_ev_prop[i, j]  # Shape: (seeds, time steps)
            sim_data_array = np.array([convert_data_short(pred, base_params) for pred in predictions])
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
def plot_best_parameters_all_metrics(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=300):
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
        sim_data_array = np.array([convert_data_short(pred, base_params) for pred in predictions])

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

def plot_ev_stock_all_combinations(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=300):
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
            processed_data_seeds = [convert_data_short(data, base_params) for data in predictions]
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

            # Hide major and minor ticks
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.set_yticks([])
            ax.set_yticks([], minor=True)

            # Add parameter values as a tuple in the top left corner
            param_tuple = (round(param_1, 3), round(param_2, 3))
            ax.text(0.02, 0.95, f"{param_tuple}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

            ax.set_ylim([min(real_data), max(real_data)])

    # Add labels for the entire figure
    fig.supxlabel(vary_1["property_varied"])
    fig.supylabel(vary_2["property_varied"])

    # Adjust layout and save the figure
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_ev_stock_combinations", dpi)

def plot_price_heatmap(base_params, data_array_price_range, vary_1, vary_2, fileName, dpi=300):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    print(data_array_price_range.shape)
    data_array_price_range_mean = np.mean(data_array_price_range, axis = 2)
    print(data_array_price_range_mean.shape)
    # Plot static heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(data_array_price_range_mean, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label="Price Range ICE, $")

    ax.set_xticks(range(num_vary_2))
    ax.set_xticklabels([f"{val:.3g}" for val in vary_2["property_list"]], rotation=45)
    ax.set_yticks(range(num_vary_1))
    ax.set_yticklabels([f"{val:.3g}" for val in vary_1["property_list"]])
    ax.set_xlabel(f"{vary_2['property_varied']}")
    ax.set_ylabel(f"{vary_1['property_varied']}")

    save_and_show(fig, fileName, "price_range_heatmap", dpi)

def plot_ev_uptake_heatmap(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=300):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    ev_uptake_values = np.zeros((num_vary_1, num_vary_2))

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            predictions = data_array_ev_prop[i, j]  # Shape: (seeds, time steps)
            sim_data_array = np.array([convert_data_short(pred, base_params) for pred in predictions])
            ev_uptake_values[i, j] = np.mean(sim_data_array[:, -1])  # Take the last year's average EV uptake

    # Plot static heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(ev_uptake_values, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label="EV Uptake (%)")

    ax.set_xticks(range(num_vary_2))
    ax.set_xticklabels([f"{val:.3g}" for val in vary_2["property_list"]], rotation=45)
    ax.set_yticks(range(num_vary_1))
    ax.set_yticklabels([f"{val:.3g}" for val in vary_1["property_list"]])
    ax.set_xlabel(f"{vary_2['property_varied']}")
    ax.set_ylabel(f"{vary_1['property_varied']}")
    ax.set_title("EV Uptake Heatmap")

    save_and_show(fig, fileName, "ev_uptake_heatmap", dpi)

def plot_ev_uptake_contour(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=300):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    ev_uptake_values = np.zeros((num_vary_1, num_vary_2))

    # Calculate EV uptake for each parameter combination
    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            predictions = data_array_ev_prop[i, j]  # Shape: (seeds, time steps)
            sim_data_array = np.array([convert_data_short(pred, base_params) for pred in predictions])
            ev_uptake_values[i, j] = np.mean(sim_data_array[:, -1])  # Take the last year's average EV uptake

    # Create meshgrid for contour plot
    X, Y = np.meshgrid(vary_1["property_list"], vary_2["property_list"])

    # Plot contour plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, ev_uptake_values, levels=20, cmap='viridis')  # Filled contours
    ax.contour(X, Y, ev_uptake_values, levels=20, colors='black', linewidths=0.5)  # Contour lines

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, label="EV Uptake (%)")

    # Add labels and title
    ax.set_xlabel(f"{vary_1['property_varied']}")
    ax.set_ylabel(f"{vary_2['property_varied']}")
    ax.set_title("EV Uptake Contour Plot")

    # Save and show the plot
    save_and_show(fig, fileName, "ev_uptake_contour", dpi)

# Main function
def main(fileName, dpi=300):
    try:
        #serial_data = load_object(fileName + "/Data", "data_flat_ev_prop")
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_ev_prop = load_object(fileName + "/Data", "data_array_ev_prop")
        data_array_price_range = load_object(fileName + "/Data", "data_price_range_arr")
        vary_1 = load_object(fileName + "/Data", "vary_1")
        vary_2 = load_object(fileName + "/Data", "vary_2")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")

    # Extract actual EV stock proportions (2010-2022)
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_stock_prop_2016_22 = np.asarray(EV_stock_prop_2010_23)[6:]  
    #print("EV_stock_prop_2016_22",len(EV_stock_prop_2016_22))
    # Plot heatmaps for different metrics

    plot_price_heatmap(base_params, data_array_price_range, vary_1, vary_2, fileName, dpi)

    plt.show()
    # Plot EV uptake heatmap
    plot_ev_uptake_heatmap(base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    # Plot EV uptake contour plot
    plot_ev_uptake_contour(base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)


    plot_metric_heatmap(calc_mape_vectorized, "mape", base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_metric_heatmap(calc_smape, "smape", base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_metric_heatmap(calc_mse, "mse", base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_metric_heatmap(calc_rmse, "rmse", base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)

    # Plot best parameters from all metrics
    plot_best_parameters_all_metrics(base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)

    plot_ev_stock_all_combinations(base_params, EV_stock_prop_2016_22, data_array_ev_prop, vary_1, vary_2, fileName)

    plt.show()

if __name__ == "__main__":
    main("results/MAPE_ev_2D_14_57_35__05_03_2025")
