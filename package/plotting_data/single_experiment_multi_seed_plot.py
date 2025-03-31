import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.plotting_data.single_experiment_plot import save_and_show

def plot_combined_figures(base_params, fileName, dpi=300):
    """
    Create a combined 2x2 figure with all four plots.
    
    Args:
        base_params: Dictionary containing simulation parameters.
        fileName: File path for saving the figure.
        dpi: Dots per inch for the saved figure.
    """
    
    # Load all necessary data
    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    history_prop_EV_arr = load_object(fileName + "/Data", "history_prop_EV_arr")
    history_lower_percentile_price_ICE_EV_arr = load_object(fileName + "/Data", "history_lower_percentile_price_ICE_EV_arr")
    history_upper_percentile_price_ICE_EV_arr = load_object(fileName + "/Data", "history_upper_percentile_price_ICE_EV_arr")
    history_mean_price_ICE_EV_arr = load_object(fileName + "/Data", "history_mean_price_ICE_EV_arr")
    history_median_price_ICE_EV_arr = load_object(fileName + "/Data", "history_median_price_ICE_EV_arr")
    history_market_concentration_arr = load_object(fileName + "/Data", "history_market_concentration_arr")
    history_mean_car_age_arr = np.asarray(load_object(fileName + "/Data", "history_mean_car_age"))
    history_past_new_bought_vehicles_prop_ev = load_object(fileName + "/Data", "history_past_new_bought_vehicles_prop_ev")
    
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]

    EV_sales_prop_2020_23 = calibration_data_output["EV Sales Prop"]
    
    # Create a 2x2 figure
    fig, axs = plt.subplots(2, 2, figsize=(17, 10), sharex=True)
    
    # Plot 1: EV Uptake (top-left)
    ax1 = axs[0, 0]
    plot_ev_uptake(EV_stock_prop_2010_23, EV_sales_prop_2020_23, base_params, history_prop_EV_arr, 
                   history_past_new_bought_vehicles_prop_ev, ax1, 
                   annotation_height_prop=[0.45, 0.45, 0.45])
    
    # Plot 2: Mean Price (top-right)
    ax2 = axs[0, 1]
    plot_mean_price(base_params, history_mean_price_ICE_EV_arr, 
                    history_median_price_ICE_EV_arr, 
                    history_lower_percentile_price_ICE_EV_arr,
                    history_upper_percentile_price_ICE_EV_arr, ax2,
                    annotation_height_prop=[0.45, 0.45, 0.45])
    
    # Plot 3: Market Concentration (bottom-left)
    ax3 = axs[1, 0]
    plot_market_concentration(base_params, history_market_concentration_arr, ax3,
                              annotation_height_prop=[0.7, 0.7, 0.7])
    
    # Plot 4: Mean Car Age (bottom-right)
    ax4 = axs[1, 1]
    plot_mean_car_age(base_params, history_mean_car_age_arr, ax4,
                      annotation_height_prop=[0.3, 0.3, 0.3])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined figure
    plt.savefig(f"{fileName}/combined_plots.png", dpi=dpi)
    plt.show()

# Helper functions for each subplot
def plot_ev_uptake(real_data, EV_sales_prop_2020_23, base_params, data, history_past_new_bought_vehicles_prop_ev, ax, 
                   annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot EV uptake on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = data[:, burn_in_step:]
    data_buy = history_past_new_bought_vehicles_prop_ev[:, burn_in_step:]
    time_steps = np.arange(0, data_after_burn_in.shape[1])
    
    init_real = 108 + 4  # STARTS AT APRIL of THE END OF 2010
    time_steps_real = np.arange(init_real, init_real + len(real_data) * 12, 12)

    init_real_sales = 108 + 120 + 11  # STARTS AT APRIL of THE END OF 2010
    time_steps_real_sales = np.arange(init_real_sales, init_real_sales + len(EV_sales_prop_2020_23) * 12, 12)
    print(time_steps_real_sales)
    
    mean_values = np.mean(data_after_burn_in, axis=0)
    #median_values = np.median(data_after_burn_in, axis=0)
    data_buy_mean = np.mean(data_buy, axis=0)
    
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)
    ci_range_buy = sem(data_buy, axis=0) * t.ppf(0.975, df=data_buy.shape[0] - 1)
    

    ax.plot(time_steps, mean_values, label='Mean, EV Adoption', color='green')
    ax.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='green',
        alpha=0.3,
        label='95% Confidence Interval, EV Adoption'
    )
    
    ax.plot(time_steps, data_buy_mean, label='Mean, EV Sales', color='orange')
    ax.fill_between(
        time_steps,
        data_buy_mean - ci_range_buy,
        data_buy_mean + ci_range_buy,
        color='orange',
        alpha=0.3,
        label='95% Confidence Interval, EV Sales'
    )
    
    ax.plot(time_steps_real, real_data, label="California EV Adoption", color='black')
    ax.plot(time_steps_real_sales,EV_sales_prop_2020_23, label="California EV Sales", color='black', linestyle="dotted")

    #ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Proportion of EVs")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc = "upper left")

def plot_mean_price(base_params, history_mean_price_ICE_EV, history_median_price_ICE_EV, 
                    history_lower_price_ICE_EV, history_upper_price_ICE_EV, ax,
                    annotation_height_prop=[0.3, 0.3, 0.2]):
    """Plot mean price on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    
    # Extract data
    mean_new_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 0, 0]
    mean_second_hand_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 1, 0]
    mean_new_EV = history_mean_price_ICE_EV[:, burn_in_step:, 0, 1]
    mean_second_hand_EV = history_mean_price_ICE_EV[:, burn_in_step:, 1, 1]
    
    # Check for second-hand EV data cutoff
    if np.any(np.isnan(history_mean_price_ICE_EV[:, burn_in_step:, 1, 1])):
        print("Warning: Second-hand EV data contains NaN values - this may cause cutoff in plots")
    
    # Percentile data
    lower_new_ICE = history_lower_price_ICE_EV[:, burn_in_step:, 0, 0]
    lower_second_hand_ICE = history_lower_price_ICE_EV[:, burn_in_step:, 1, 0]
    lower_new_EV = history_lower_price_ICE_EV[:, burn_in_step:, 0, 1]
    lower_second_hand_EV = history_lower_price_ICE_EV[:, burn_in_step:, 1, 1]
    
    upper_new_ICE = history_upper_price_ICE_EV[:, burn_in_step:, 0, 0]
    upper_second_hand_ICE = history_upper_price_ICE_EV[:, burn_in_step:, 1, 0]
    upper_new_EV = history_upper_price_ICE_EV[:, burn_in_step:, 0, 1]
    upper_second_hand_EV = history_upper_price_ICE_EV[:, burn_in_step:, 1, 1]
    
    time_steps = np.arange(0, mean_new_ICE.shape[1])
    
    # Calculate statistics for means
    def calculate_stats(data):
        # Compute nanmean across rows
        mean = np.nanmean(data, axis=0)

        # Count non-NaN samples per column
        n = np.sum(~np.isnan(data), axis=0)
        
        # Compute standard error of the mean, ignoring NaNs
        stderr = sem(data, axis=0, nan_policy='omit')
        
        # Compute t-multiplier for 95% CI (two-sided)
        t_multiplier = t.ppf(0.975, df=n - 1)
        
        # Compute confidence intervals
        ci = t_multiplier * stderr
        
        return mean, ci
    
    # Mean statistics
    overall_mean_new_ICE, ci_new_ICE = calculate_stats(mean_new_ICE)
    overall_mean_second_hand_ICE, ci_second_hand_ICE = calculate_stats(mean_second_hand_ICE)
    overall_mean_new_EV, ci_new_EV = calculate_stats(mean_new_EV)
    overall_mean_second_hand_EV, ci_second_hand_EV = calculate_stats(mean_second_hand_EV)
    
    # 25th percentile statistics
    overall_lower_new_ICE, ci_lower_new_ICE = calculate_stats(lower_new_ICE)
    overall_lower_second_hand_ICE, ci_lower_second_hand_ICE = calculate_stats(lower_second_hand_ICE)
    overall_lower_new_EV, ci_lower_new_EV = calculate_stats(lower_new_EV)
    overall_lower_second_hand_EV, ci_lower_second_hand_EV = calculate_stats(lower_second_hand_EV)
    
    # 75th percentile statistics
    overall_upper_new_ICE, ci_upper_new_ICE = calculate_stats(upper_new_ICE)
    overall_upper_second_hand_ICE, ci_upper_second_hand_ICE = calculate_stats(upper_second_hand_ICE)
    overall_upper_new_EV, ci_upper_new_EV = calculate_stats(upper_new_EV)
    overall_upper_second_hand_EV, ci_upper_second_hand_EV = calculate_stats(upper_second_hand_EV)
    
    # Plot percentiles with original colors but without individual labels
    # 25th percentile (dash-dot line)
    ax.plot(time_steps, overall_lower_new_ICE, color="blue", linestyle=(0, (3, 1, 1, 1)), alpha=0.7)
    ax.plot(time_steps, overall_lower_second_hand_ICE, color="blue", linestyle=(0, (3, 1, 1, 1)), alpha=0.7)
    ax.plot(time_steps, overall_lower_new_EV, color="green", linestyle=(0, (3, 1, 1, 1)), alpha=0.7)
    ax.plot(time_steps, overall_lower_second_hand_EV, color="green", linestyle=(0, (3, 1, 1, 1)), alpha=0.7)
    
    # 75th percentile (dotted line)
    ax.plot(time_steps, overall_upper_new_ICE, color="blue", linestyle="dotted", alpha=0.7)
    ax.plot(time_steps, overall_upper_second_hand_ICE, color="blue", linestyle="dotted", alpha=0.7)
    ax.plot(time_steps, overall_upper_new_EV, color="green", linestyle="dotted", alpha=0.7)
    ax.plot(time_steps, overall_upper_second_hand_EV, color="green", linestyle="dotted", alpha=0.7)
    
    # Add confidence regions for percentiles (with original colors)
    # ICE (blue)
    ax.fill_between(
        time_steps,
        overall_lower_new_ICE - ci_lower_new_ICE,
        overall_lower_new_ICE + ci_lower_new_ICE,
        color="blue",
        alpha=0.1
    )
    ax.fill_between(
        time_steps,
        overall_upper_new_ICE - ci_upper_new_ICE,
        overall_upper_new_ICE + ci_upper_new_ICE,
        color="blue",
        alpha=0.1
    )
    # EV (green)
    ax.fill_between(
        time_steps,
        overall_lower_new_EV - ci_lower_new_EV,
        overall_lower_new_EV + ci_lower_new_EV,
        color="green",
        alpha=0.1
    )
    ax.fill_between(
        time_steps,
        overall_upper_new_EV - ci_upper_new_EV,
        overall_upper_new_EV + ci_upper_new_EV,
        color="green",
        alpha=0.1
    )
    
    # Plot means and CIs (original colors)
    # ICE
    ax.plot(time_steps, overall_mean_new_ICE, label="New Car Mean Price ICE", color="blue")
    ax.fill_between(
        time_steps,
        overall_mean_new_ICE - ci_new_ICE,
        overall_mean_new_ICE + ci_new_ICE,
        color="blue",
        alpha=0.2
    )
    ax.plot(time_steps, overall_mean_second_hand_ICE, label="Used Car Mean Price ICE", color="blue", linestyle="dashed")
    ax.fill_between(
        time_steps,
        overall_mean_second_hand_ICE - ci_second_hand_ICE,
        overall_mean_second_hand_ICE + ci_second_hand_ICE,
        color="blue",
        alpha=0.2
    )
    
    # EV - handle potential NaN values for second-hand
    ax.plot(time_steps, overall_mean_new_EV, label="New Car Mean Price EV", color="green")
    ax.fill_between(
        time_steps,
        overall_mean_new_EV - ci_new_EV,
        overall_mean_new_EV + ci_new_EV,
        color="green",
        alpha=0.2
    )
    
    # Only plot second-hand EV if we have valid data

    ax.plot(time_steps, overall_mean_second_hand_EV, label="Used Car Mean Price EV", color="green", linestyle="dashed")
    ax.fill_between(
        time_steps,
        overall_mean_second_hand_EV - ci_second_hand_EV,
        overall_mean_second_hand_EV + ci_second_hand_EV,
        color="green",
        alpha=0.2
    )

    # Add simplified percentile legend items (using representative lines)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="grey", linestyle=(0, (3, 1, 1, 1)), lw=2, label='25th Percentile'),
        Line2D([0], [0], color="grey", linestyle='dotted', lw=2, label='75th Percentile'),
        Line2D([0], [0], color="grey", alpha=0.1, lw=10, label='95% Confidence Interval'),
    ]
    
    # Get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Combine the legends - percentile items first
    ax.legend(handles=legend_elements + handles, fontsize="small", ncol=1)
    
    ax.set_ylabel("Price, $")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)

def plot_mean_price_old(base_params, history_mean_price_ICE_EV, history_median_price_ICE_EV, 
                    history_lower_price_ICE_EV, history_upper_price_ICE_EV, ax,
                    annotation_height_prop=[0.3, 0.3, 0.2]):
    """Plot mean price on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    
    mean_new_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 0, 0]
    mean_second_hand_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 1, 0]
    mean_new_EV = history_mean_price_ICE_EV[:, burn_in_step:, 0, 1]
    mean_second_hand_EV = history_mean_price_ICE_EV[:, burn_in_step:, 1, 1]
    
    # Calculate mean and CI for lower (25th) and upper (75th) percentiles
    lower_new_ICE = history_lower_price_ICE_EV[:, burn_in_step:, 0, 0]
    lower_second_hand_ICE = history_lower_price_ICE_EV[:, burn_in_step:, 1, 0]
    lower_new_EV = history_lower_price_ICE_EV[:, burn_in_step:, 0, 1]
    lower_second_hand_EV = history_lower_price_ICE_EV[:, burn_in_step:, 1, 1]
    
    upper_new_ICE = history_upper_price_ICE_EV[:, burn_in_step:, 0, 0]
    upper_second_hand_ICE = history_upper_price_ICE_EV[:, burn_in_step:, 1, 0]
    upper_new_EV = history_upper_price_ICE_EV[:, burn_in_step:, 0, 1]
    upper_second_hand_EV = history_upper_price_ICE_EV[:, burn_in_step:, 1, 1]
    
    time_steps = np.arange(0, mean_new_ICE.shape[1])
    
    # Calculate mean and CI for means
    overall_mean_new_ICE = np.mean(mean_new_ICE, axis=0)
    ci_new_ICE = t.ppf(0.975, df=mean_new_ICE.shape[0] - 1) * sem(mean_new_ICE, axis=0)
    
    overall_mean_second_hand_ICE = np.mean(mean_second_hand_ICE, axis=0)
    ci_second_hand_ICE = t.ppf(0.975, df=mean_second_hand_ICE.shape[0] - 1) * sem(mean_second_hand_ICE, axis=0)
    
    overall_mean_new_EV = np.mean(mean_new_EV, axis=0)
    ci_new_EV = t.ppf(0.975, df=mean_new_EV.shape[0] - 1) * sem(mean_new_EV, axis=0)
    
    overall_mean_second_hand_EV = np.mean(mean_second_hand_EV, axis=0)
    ci_second_hand_EV = t.ppf(0.975, df=mean_second_hand_EV.shape[0] - 1) * sem(mean_second_hand_EV, axis=0)
    
    # Calculate mean and CI for 25th percentiles
    overall_lower_new_ICE = np.mean(lower_new_ICE, axis=0)
    ci_lower_new_ICE = t.ppf(0.975, df=lower_new_ICE.shape[0] - 1) * sem(lower_new_ICE, axis=0)
    
    overall_lower_second_hand_ICE = np.mean(lower_second_hand_ICE, axis=0)
    ci_lower_second_hand_ICE = t.ppf(0.975, df=lower_second_hand_ICE.shape[0] - 1) * sem(lower_second_hand_ICE, axis=0)
    
    overall_lower_new_EV = np.mean(lower_new_EV, axis=0)
    ci_lower_new_EV = t.ppf(0.975, df=lower_new_EV.shape[0] - 1) * sem(lower_new_EV, axis=0)
    
    overall_lower_second_hand_EV = np.mean(lower_second_hand_EV, axis=0)
    ci_lower_second_hand_EV = t.ppf(0.975, df=lower_second_hand_EV.shape[0] - 1) * sem(lower_second_hand_EV, axis=0)
    
    # Calculate mean and CI for 75th percentiles
    overall_upper_new_ICE = np.mean(upper_new_ICE, axis=0)
    ci_upper_new_ICE = t.ppf(0.975, df=upper_new_ICE.shape[0] - 1) * sem(upper_new_ICE, axis=0)
    
    overall_upper_second_hand_ICE = np.mean(upper_second_hand_ICE, axis=0)
    ci_upper_second_hand_ICE = t.ppf(0.975, df=upper_second_hand_ICE.shape[0] - 1) * sem(upper_second_hand_ICE, axis=0)
    
    overall_upper_new_EV = np.mean(upper_new_EV, axis=0)
    ci_upper_new_EV = t.ppf(0.975, df=upper_new_EV.shape[0] - 1) * sem(upper_new_EV, axis=0)
    
    overall_upper_second_hand_EV = np.mean(upper_second_hand_EV, axis=0)
    ci_upper_second_hand_EV = t.ppf(0.975, df=upper_second_hand_EV.shape[0] - 1) * sem(upper_second_hand_EV, axis=0)
    
    # Plot 25th percentiles with confidence intervals (using dash-dot line style)
    ax.plot(time_steps, overall_lower_new_ICE, color="blue", linestyle=(0, (3, 1, 1, 1)), label="25th Percentile ICE")
    ax.fill_between(
        time_steps,
        overall_lower_new_ICE - ci_lower_new_ICE,
        overall_lower_new_ICE + ci_lower_new_ICE,
        color="blue",
        alpha=0.1
    )
    
    ax.plot(time_steps, overall_lower_second_hand_ICE, color="blue", linestyle=(0, (3, 1, 1, 1)))
    ax.fill_between(
        time_steps,
        overall_lower_second_hand_ICE - ci_lower_second_hand_ICE,
        overall_lower_second_hand_ICE + ci_lower_second_hand_ICE,
        color="blue",
        alpha=0.1
    )
    
    ax.plot(time_steps, overall_lower_new_EV, color="green", linestyle=(0, (3, 1, 1, 1)), label="25th Percentile EV")
    ax.fill_between(
        time_steps,
        overall_lower_new_EV - ci_lower_new_EV,
        overall_lower_new_EV + ci_lower_new_EV,
        color="green",
        alpha=0.1
    )
    
    ax.plot(time_steps, overall_lower_second_hand_EV, color="green", linestyle=(0, (3, 1, 1, 1)))
    ax.fill_between(
        time_steps,
        overall_lower_second_hand_EV - ci_lower_second_hand_EV,
        overall_lower_second_hand_EV + ci_lower_second_hand_EV,
        color="green",
        alpha=0.1
    )
    
    # Plot 75th percentiles with confidence intervals (using dotted line style)
    ax.plot(time_steps, overall_upper_new_ICE, color="blue", linestyle="dotted", label="75th Percentile ICE")
    ax.fill_between(
        time_steps,
        overall_upper_new_ICE - ci_upper_new_ICE,
        overall_upper_new_ICE + ci_upper_new_ICE,
        color="blue",
        alpha=0.1
    )
    
    ax.plot(time_steps, overall_upper_second_hand_ICE, color="blue", linestyle="dotted")
    ax.fill_between(
        time_steps,
        overall_upper_second_hand_ICE - ci_upper_second_hand_ICE,
        overall_upper_second_hand_ICE + ci_upper_second_hand_ICE,
        color="blue",
        alpha=0.1
    )
    
    ax.plot(time_steps, overall_upper_new_EV, color="green", linestyle="dotted", label="75th Percentile EV")
    ax.fill_between(
        time_steps,
        overall_upper_new_EV - ci_upper_new_EV,
        overall_upper_new_EV + ci_upper_new_EV,
        color="green",
        alpha=0.1
    )
    
    ax.plot(time_steps, overall_upper_second_hand_EV, color="green", linestyle="dotted")
    ax.fill_between(
        time_steps,
        overall_upper_second_hand_EV - ci_upper_second_hand_EV,
        overall_upper_second_hand_EV + ci_upper_second_hand_EV,
        color="green",
        alpha=0.1
    )
    
    # Plot means and CIs for ICE
    ax.plot(time_steps, overall_mean_new_ICE, label="New Car Mean Price ICE", color="blue")
    ax.fill_between(
        time_steps,
        overall_mean_new_ICE - ci_new_ICE,
        overall_mean_new_ICE + ci_new_ICE,
        color="blue",
        alpha=0.2
    )
    ax.plot(time_steps, overall_mean_second_hand_ICE, label="Used Car Mean Price ICE", color="blue", linestyle="dashed")
    ax.fill_between(
        time_steps,
        overall_mean_second_hand_ICE - ci_second_hand_ICE,
        overall_mean_second_hand_ICE + ci_second_hand_ICE,
        color="blue",
        alpha=0.2
    )
    
    # Plot means and CIs for EV
    ax.plot(time_steps, overall_mean_new_EV, label="New Car Mean Price EV", color="green")
    ax.fill_between(
        time_steps,
        overall_mean_new_EV - ci_new_EV,
        overall_mean_new_EV + ci_new_EV,
        color="green",
        alpha=0.2
    )
    ax.plot(time_steps, overall_mean_second_hand_EV, label="Used Car Mean Price EV", color="green", linestyle="dashed")
    ax.fill_between(
        time_steps,
        overall_mean_second_hand_EV - ci_second_hand_EV,
        overall_mean_second_hand_EV + ci_second_hand_EV,
        color="green",
        alpha=0.2
    )
    
    ax.set_ylabel("Price, $")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(fontsize="small")

def plot_market_concentration(base_params, data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot market concentration on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = data[:, burn_in_step:]
    time_steps = np.arange(0, data_after_burn_in.shape[1])
    
    mean_values = np.mean(data_after_burn_in, axis=0)
    #median_values = np.median(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)
    
    ax.plot(time_steps, mean_values, label='Mean', color='purple')
    #ax.plot(time_steps, median_values, label='Median', color='red', linestyle="--")
    ax.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='purple',
        alpha=0.3,
        label='95% Confidence Interval'
    )
    
    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Market Concentration, HHI")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc = "upper left")

def plot_mean_car_age(base_params, data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot mean car age on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = data[:, burn_in_step:]
    time_steps = np.arange(0, data_after_burn_in.shape[1])
    
    mean_values = np.mean(data_after_burn_in, axis=0)
    #median_values = np.median(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)
    
    ax.plot(time_steps, mean_values, label='Mean', color='red')
    #ax.plot(time_steps, median_values, label='Median', color='red', linestyle="--")
    ax.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='red',
        alpha=0.3,
        label='95% Confidence Interval'
    )
    
    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Car Age, months")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc = "upper left")


# Make sure to reuse the original add_vertical_lines function
def add_vertical_lines(ax, base_params, color='black', linestyle='--', annotation_height_prop=[0.2, 0.2, 0.2]):
    """
    Adds dashed vertical lines to the plot at specified steps with vertical annotations.
    """
    burn_in = base_params["duration_burn_in"]
    no_carbon_price = base_params["duration_calibration"]
    ev_production_start_time = base_params["ev_production_start_time"]

    # Determine the middle of the plot if no custom height is provided
    y_min, y_max = ax.get_ylim()

    annotation_height_0 = y_min + annotation_height_prop[0]*(y_max - y_min)
    annotation_height_1 = y_min + annotation_height_prop[1]*(y_max - y_min)
    annotation_height_2 = y_min + annotation_height_prop[2]*(y_max - y_min)

    if ev_production_start_time > 0:
        # Add vertical line with annotation
        ev_sale_start_time = ev_production_start_time
        ax.axvline(ev_sale_start_time, color="black", linestyle=':')
        ax.annotate("EV Sale Start", xy=(ev_sale_start_time, annotation_height_0),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    if base_params["EV_rebate_state"]:
        rebate_start_time = base_params["parameters_rebate_calibration"]["start_time"]
        ax.axvline(rebate_start_time, color="black", linestyle='-.')
        ax.annotate("EV Adoption Subsidy Start", xy=(rebate_start_time, annotation_height_1),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    if base_params["duration_future"] > 0:
        policy_start_time = no_carbon_price
        ax.axvline(policy_start_time, color="black", linestyle='--')
        ax.annotate("Policy Start", xy=(policy_start_time, annotation_height_2),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

# Example usage
if __name__ == "__main__":
    
    fileName = "results/multi_seed_single_17_40_32__31_03_2025"#multi_seed_single_00_03_21__27_03_2025"
    base_params = load_object(fileName + "/Data", "base_params")
    
    plot_combined_figures(base_params, fileName)