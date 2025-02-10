import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import sem, t
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show

def calc_mape_vectorized(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual), axis=1) * 100

def calc_smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

def calc_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2, axis=1)

def calc_rmse(actual, predicted):
    return np.sqrt(calc_mse(actual, predicted))

def save_best_parameters(best_params, best_metric, metric_name, vary_1, vary_2, vary_3, file_path):
    best_fit_info = {
        "best_parameters": {
            vary_1["property_varied"]: best_params[0],
            vary_2["property_varied"]: best_params[1],
            vary_3["property_varied"]: best_params[2]
        },
        f"best_{metric_name}": best_metric
    }
    with open(file_path, 'w') as f:
        json.dump(best_fit_info, f, indent=4)

def plot_metric_heatmap(metric_function, metric_name, base_params, real_data, data_array_ev_prop, vary_1, vary_2, vary_3, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    num_vary_3 = len(vary_3["property_list"])

    for k, param_3 in enumerate(vary_3["property_list"]):
        metric_values = np.zeros((num_vary_1, num_vary_2))

        for i, param_1 in enumerate(vary_1["property_list"]):
            for j, param_2 in enumerate(vary_2["property_list"]):
                predictions = data_array_ev_prop[i, j, k]
                sim_data_array = np.array([convert_data(pred, base_params) for pred in predictions])
                metric_data = metric_function(real_data, sim_data_array)
                metric_values[i, j] = np.mean(metric_data)

        best_idx = np.unravel_index(np.argmin(metric_values), metric_values.shape)
        best_params = (vary_1["property_list"][best_idx[0]], vary_2["property_list"][best_idx[1]], param_3)

        print(f"Best parameter combination at {vary_3['property_varied']} = {param_3:.3g}: "
              f"{vary_1['property_varied']} = {best_params[0]:.3g}, "
              f"{vary_2['property_varied']} = {best_params[1]:.3g} with {metric_name.upper()} = {metric_values[best_idx]:.2f}")

        save_best_parameters(best_params, metric_values[best_idx], metric_name, vary_1, vary_2, vary_3, f"{fileName}/best_fit_{metric_name}_{param_3}.json")

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(metric_values, cmap='viridis', origin='lower', aspect='auto')
        fig.colorbar(cax, ax=ax, label=f"Mean {metric_name.upper()}")

        ax.set_xticks(range(num_vary_2))
        ax.set_xticklabels([f"{val:.3g}" for val in vary_2["property_list"]], rotation=45)
        ax.set_yticks(range(num_vary_1))
        ax.set_yticklabels([f"{val:.3g}" for val in vary_1["property_list"]])
        ax.set_xlabel(f"{vary_2['property_varied']}")
        ax.set_ylabel(f"{vary_1['property_varied']}")
        ax.set_title(f"{metric_name.upper()} Heatmap at {vary_3['property_varied']} = {param_3:.3g}")

        save_and_show(fig, fileName, f"{metric_name}_heatmap_{param_3}", dpi)

def main(fileName, dpi=600):
    try:
        serial_data = load_object(fileName + "/Data", "data_flat_ev_prop")
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_ev_prop = load_object(fileName + "/Data", "data_array_ev_prop")
        vary_1 = load_object(fileName + "/Data", "vary_1")
        vary_2 = load_object(fileName + "/Data", "vary_2")
        vary_3 = load_object(fileName + "/Data", "vary_3")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    plot_metric_heatmap(calc_mape_vectorized, "mape", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, fileName, dpi)
    plot_metric_heatmap(calc_smape, "smape", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, fileName, dpi)
    plot_metric_heatmap(calc_mse, "mse", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, fileName, dpi)
    plot_metric_heatmap(calc_rmse, "rmse", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, fileName, dpi)

    plt.show()

if __name__ == "__main__":
    main("results/MAPE_ev_3D_20_14_43__10_02_2025")
