import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import sem, t
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def calc_mape_vectorized(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calc_smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


def calc_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2, axis=1)


def calc_rmse(actual, predicted):
    return np.sqrt(calc_mse(actual, predicted))


def save_top_parameters(top_params, metric_name, vary_1, vary_2, vary_3, file_path):
    top_fit_info = []
    for params, metric_value in top_params:
        fit_info = {
            "parameters": {
                vary_1["property_varied"]: params[0],
                vary_2["property_varied"]: params[1],
                vary_3["property_varied"]: params[2]
            },
            f"{metric_name}": metric_value
        }
        top_fit_info.append(fit_info)

    with open(file_path, 'w') as f:
        json.dump(top_fit_info, f, indent=4)


def plot_combined_best_metrics(best_params_dict, data_array_ev_prop, real_data, base_params, vary_1, vary_2, vary_3, fileName, dpi=600):
    metrics = list(best_params_dict.keys())
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for idx, metric in enumerate(metrics):
        params = best_params_dict[metric]
        print("best params", params)
        param_1_idx = np.where(vary_1["property_list"] == params[0])[0][0]
        param_2_idx = np.where(vary_2["property_list"] == params[1])[0][0]
        param_3_idx = np.where(vary_3["property_list"] == params[2])[0][0]

        predictions = data_array_ev_prop[param_1_idx, param_2_idx, param_3_idx]
        sim_data_array = np.array([convert_data(pred, base_params) for pred in predictions])

        mean_values = np.mean(sim_data_array, axis=0)
        ci = sem(sim_data_array, axis=0) * t.ppf((1 + 0.95) / 2., sim_data_array.shape[0]-1)

        years = np.arange(real_data.shape[0]) if real_data.ndim == 1 else np.arange(real_data.shape[1])

        ax = axs[idx]
        for seed_data in sim_data_array:
            ax.plot(years, seed_data, color='gray', alpha=0.3)
        ax.plot(years, mean_values, color='blue', label='Mean Prediction')
        ax.fill_between(years, mean_values - ci, mean_values + ci, color='blue', alpha=0.2, label='95% CI')
        ax.plot(years, real_data.flatten(), color='red', linestyle='--', label='Real World Data')

        ax.set_title(f"Best {metric.upper()} Fit")
        ax.set_xlabel("Year")
        ax.set_ylabel("EV Stock Proportion")
        ax.legend()

    plt.tight_layout()
    save_and_show(fig, fileName, "combined_best_metrics_timeseries", dpi)


def process_metrics(metric_function, metric_name, base_params, real_data, data_array_ev_prop, vary_1, vary_2, vary_3, best_params_dict, fileName, dpi=600):

    all_metrics = []

    for k, param_3 in enumerate(vary_3["property_list"]):
        for i, param_1 in enumerate(vary_1["property_list"]):
            for j, param_2 in enumerate(vary_2["property_list"]):
                predictions = data_array_ev_prop[i, j, k]
                sim_data_array = np.array([convert_data(pred, base_params) for pred in predictions])
                metric_data = metric_function(real_data, sim_data_array)
                all_metrics.append(((param_1, param_2, param_3), np.mean(metric_data)))

    best_params, best_metric = min(all_metrics, key=lambda x: x[1])
    best_params_dict[metric_name] = best_params

    with open(f"{fileName}/best_overall_{metric_name}.json", 'w') as f:
        json.dump({"best_parameters": best_params, f"best_{metric_name}": best_metric}, f, indent=4)



def plot_ev_uptake_over_time(data_array_ev_prop, real_data, vary_1, vary_2, vary_3, base_params, fileName, dpi=600):
    # Convert real-world data to yearly format
    num_real_time_steps = real_data.shape[0] if real_data.ndim == 1 else real_data.shape[1]
    time_steps = np.arange(num_real_time_steps)  # Ensure the correct length of time steps

    num_rows = len(vary_1["property_list"])
    num_cols = len(vary_2["property_list"])

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18,15), sharex=True, sharey=True)
    axs = np.array(axs)  # Ensure axs is always an array, even if 1D

    norm = mcolors.Normalize(vmin=min(vary_3["property_list"]), vmax=max(vary_3["property_list"]))
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            ax = axs[i, j] if num_rows > 1 and num_cols > 1 else axs[max(i, j)]
            
            for k, param_3 in enumerate(vary_3["property_list"]):
                predictions = np.array(data_array_ev_prop[i, j, k])  # Convert to NumPy array
                
                # Convert monthly data to yearly using `convert_data`
                yearly_predictions = np.array([convert_data(pred, base_params) for pred in predictions])

                if yearly_predictions.shape[1] != num_real_time_steps:
                    raise ValueError(f"Shape mismatch: converted predictions {yearly_predictions.shape[1]} vs. real data {num_real_time_steps}")

                mean_val = np.mean(yearly_predictions, axis=0)  # Compute mean over seeds
                ax.plot(time_steps, mean_val, color=cm.viridis(norm(param_3)), alpha=0.8, label=f'{param_3}')

            # Add real-world data
            #ax.plot(time_steps, real_data.flatten(), linestyle='--', color='orange', linewidth=2)

            #ax.set_title(f"{vary_1['property_varied']}={param_1}, {vary_2['property_varied']}={param_2}")
    fig.supxlabel("Time Steps (Years)")
    fig.supylabel("EV Uptake")

    fig.colorbar(sm, ax=axs, orientation='vertical', label=vary_3['property_varied'])
    #plt.tight_layout()
    save_and_show(fig, fileName, "ev_timeseries", dpi)

def main(fileName, dpi=600):
    try:
        serial_data = load_object(fileName + "/Data", "data_flat_ev_prop")
        base_params = load_object(fileName + "/Data", "base_params")
        data_flat_ev_prop = load_object(fileName + "/Data", "data_flat_ev_prop")
        data_array_ev_prop = load_object(fileName + "/Data", "data_array_ev_prop")
        vary_1 = load_object(fileName + "/Data", "vary_1")
        vary_2 = load_object(fileName + "/Data", "vary_2")
        vary_3 = load_object(fileName + "/Data", "vary_3")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    print("Max",np.max(data_array_ev_prop))

    plot_ev_uptake_over_time(data_array_ev_prop, EV_stock_prop_2010_22, vary_1, vary_2, vary_3, base_params, fileName)

    best_params_dict = {}
    process_metrics(calc_mape_vectorized, "mape", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, best_params_dict, fileName, dpi)
    process_metrics(calc_smape, "smape", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, best_params_dict, fileName, dpi)
    process_metrics(calc_mse, "mse", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, best_params_dict, fileName, dpi)
    process_metrics(calc_rmse, "rmse", base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, vary_3, best_params_dict, fileName, dpi)

    plot_combined_best_metrics(best_params_dict, data_array_ev_prop, EV_stock_prop_2010_22, base_params, vary_1, vary_2, vary_3, fileName, dpi)
    plt.show()


if __name__ == "__main__":
    main("results/MAPE_ev_3D_22_02_55__22_02_2025")
