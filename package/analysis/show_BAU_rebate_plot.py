from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.plotting_data.single_experiment_plot import save_and_show, format_plot


def plot_ev_uptake_single(real_data, base_params, fileName, data,  data2, title, x_label, y_label, save_name, dpi=600):
    """
    Plot data across multiple seeds with mean, confidence intervals, and individual traces,
    starting from the end of the burn-in period.

    Args:
        real_data: Real-world data to compare against.
        base_params: Dictionary containing simulation parameters (e.g., duration_burn_in).
        fileName: Name of the file for saving the plot.
        data: 2D array where rows are the time series for each seed (shape: [num_seeds, num_time_steps]).
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        save_name: Name of the file to save the plot.
        dpi: Dots per inch for saving the plot.
    """
    # Determine the start of the data after the burn-in period
    burn_in_step = base_params["duration_burn_in"] + base_params["duration_calibration"]
    data_after_burn_in = data[:, burn_in_step:]
    data_after_burn_in2 = data2[:, burn_in_step:]

    time_steps = np.arange(0, data.shape[1] - burn_in_step )

    # Calculate mean and 95% confidence interval for data after burn-in
    mean_values = np.mean(data_after_burn_in, axis=0)
    median_values = np.median(data_after_burn_in, axis=0)

    mean_values2 = np.mean(data_after_burn_in2, axis=0)
    median_values2 = np.median(data_after_burn_in2, axis=0)

    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)  # 95% CI
    ci_range2 = sem(data_after_burn_in2, axis=0) * t.ppf(0.975, df=data_after_burn_in2.shape[0] - 1)  # 95% CI

    # Create subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

    # First subplot: Mean and 95% CI with real data
    #ax1.plot(time_steps_real, real_data, label="California Data", color='orange', linestyle="dotted")
    ax1.plot(time_steps, mean_values, label='Mean', color='blue')
    ax1.plot(time_steps, median_values, label="Median", color='blue', linestyle="dashed")
    median_values
    ax1.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='blue',
        alpha=0.3,
        label='95% Confidence Interval'
    )

    ax1.plot(time_steps, mean_values2, label='Mean - No rebate', color='green')
    ax1.plot(time_steps, median_values2, label="Median- No rebate", color='green', linestyle="dashed")
    median_values
    ax1.fill_between(
        time_steps,
        mean_values2 - ci_range2,
        mean_values2 + ci_range2,
        color='green',
        alpha=0.3,
        label='95% Confidence Interval- No rebate'
    )

    ax1.axvline(x=156, linestyle= "dashed", color = "red", label = "2035")
    # Second subplot: Individual traces with real data
    #for seed_data in data_after_burn_in:
    #    ax1.plot(time_steps, seed_data, color='gray', alpha=0.3, linewidth=0.8)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    #add_vertical_lines(ax1, base_params)
    ax1.legend( loc='lower right')
    ax1.grid()

    # Adjust layout and save
    plt.tight_layout()
    save_and_show(fig, fileName, save_name, dpi)


# Sample main function
def main(        
        fileName_BAU = "results/multi_seed_single_11_16_10__12_03_2025",
        fileName_BAU_no_policy = "results/multi_seed_single_11_28_37__12_03_2025", 
        dpi=300
        ):

    base_params = load_object(fileName_BAU + "/Data", "base_params")
    print(base_params)
    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")

    history_prop_EV_arr_BAU= load_object(fileName_BAU + "/Data", "history_prop_EV_arr")
    history_prop_EV_arr_BAU_no_rebate= load_object(fileName_BAU_no_policy + "/Data", "history_prop_EV_arr")

    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]

    plot_ev_uptake_single(EV_stock_prop_2010_23, base_params, fileName_BAU, history_prop_EV_arr_BAU, history_prop_EV_arr_BAU_no_rebate,
                        "Proportion of EVs Over Time", 
                        "Time Step, months", 
                        "Proportion of EVs", 
                        "history_prop_EV_PAIR")


    plt.show()

if __name__ == "__main__":
    main(
        fileName_BAU = "results/multi_seed_single_11_16_10__12_03_2025",
        fileName_BAU_no_policy = "results/multi_seed_single_11_28_37__12_03_2025"
        )