import numpy as np
import matplotlib.pyplot as plt
#from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show



def plot_heatmap(base_params, results, measure,  vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    print(measure)
    print(results)
    data = results[measure]
    print(data.shape)
    data_mean = np.mean(data, axis = 2)
    print(data_mean.shape)

    # Plot static heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(data_mean, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label=measure)
    ax.set_xticks(range(num_vary_2))
    ax.set_xticklabels([f"{val:.3g}" for val in vary_2["property_list"]], rotation=45)
    ax.set_yticks(range(num_vary_1))
    ax.set_yticklabels([f"{val:.3g}" for val in vary_1["property_list"]])
    ax.set_xlabel(f"{vary_2['property_varied']}")
    ax.set_ylabel(f"{vary_1['property_varied']}")
    save_and_show(fig, fileName, f"heatmap_{measure}", dpi)
    plt.close()

# Main function
def main(fileName, dpi=600):

    fileName_list = load_object(fileName + "/Data", "2D_sensitivity_fileName_list")
    measure_list = [
        "ev_uptake",
        "total_cost",
        "net_cost",
        "emissions_cumulative",
        "emissions_cumulative_driving",
        "emissions_cumulative_production",
        "utility_cumulative",
        "profit_cumulative",
        "price_mean",
        "price_max",
        "price_min",
        "mean_mark_up", 
        "mean_car_age"
    ]

    # eeds to be doen seperately

    for fileName in fileName_list:
        try:
            base_params = load_object(fileName + "/Data", "base_params")
            results = load_object(fileName + "/Data", "results")
            vary_1 = load_object(fileName + "/Data", "vary_1")
            vary_2 = load_object(fileName + "/Data", "vary_2")
        except FileNotFoundError:
            print("Data files not found.")
            return
        for measure in measure_list:
            plot_heatmap(base_params, results, measure, vary_1, vary_2, fileName, dpi = 300)

    plt.show()

if __name__ == "__main__":
    main("results/sens_2d_all_13_21_16__10_03_2025")
