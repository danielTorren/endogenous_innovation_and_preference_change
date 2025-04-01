import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object

########################################################################################################################

def plot_combined_figures(base_params, fileName, output,save_name, dpi=300):
    #FOR SPECIFIC POLICY COMBO
    
    # Load all necessary data
    history_prop_EV_arr = output["history_prop_EV"]
    history_lower_percentile_price_ICE_EV_arr = output["history_lower_percentile_price_ICE_EV_arr"]#load_object(fileName + "/Data", "history_lower_percentile_price_ICE_EV_arr")
    history_upper_percentile_price_ICE_EV_arr = output["history_upper_percentile_price_ICE_EV_arr"]#load_object(fileName + "/Data", "history_upper_percentile_price_ICE_EV_arr")
    history_mean_price_ICE_EV_arr = output["history_mean_price_ICE_EV_arr"]#load_object(fileName + "/Data", "history_mean_price_ICE_EV_arr")
    history_median_price_ICE_EV_arr = output["history_median_price_ICE_EV_arr"]#load_object(fileName + "/Data", "history_median_price_ICE_EV_arr")
    history_market_concentration_arr = output["history_market_concentration"]#load_object(fileName + "/Data", "history_market_concentration_arr")
    history_mean_car_age_arr = output["history_mean_car_age"]#np.asarray(load_object(fileName + "/Data", "history_mean_car_age"))
    history_past_new_bought_vehicles_prop_ev = output["history_past_new_bought_vehicles_prop_ev"]#load_object(fileName + "/Data", "history_past_new_bought_vehicles_prop_ev")
    
    history_total_emissions_arr = output["history_total_emissions"]
    history_driving_emissions_arr = output["history_driving_emissions"]
    history_production_emissions_arr = output["history_production_emissions"]

    # Create a 2x2 figure
    fig, axs = plt.subplots(3, 2, figsize=(17, 14), sharex=True)

    
    # Plot 1: EV Uptake (top-left)
    ax1 = axs[0, 0]
    plot_ev_uptake(base_params, history_prop_EV_arr, 
                   history_past_new_bought_vehicles_prop_ev, ax1, 
                   annotation_height_prop=[0.45, 0.45, 0.45])
    
    # Plot 2: Mean Price (top-right)
    ax2 = axs[0, 1]
    plot_mean_price(base_params, history_mean_price_ICE_EV_arr, 
                    history_median_price_ICE_EV_arr, 
                    history_lower_percentile_price_ICE_EV_arr,
                    history_upper_percentile_price_ICE_EV_arr, ax2,
                    annotation_height_prop=[0.45, 0.45, 0.45])
    
    # Plot 3: Mean Car Age (bottom-right)
    ax4 = axs[1, 0]
    plot_mean_car_age(base_params, history_mean_car_age_arr, ax4,
                      annotation_height_prop=[0.3, 0.3, 0.3])
    
    # Plot 4: Market Concentration (bottom-left)
    plot_total_emissions(base_params, history_total_emissions_arr, axs[1, 1],
                    annotation_height_prop=[0.45, 0.45, 0.45])

    plot_emissions_produce(base_params, history_production_emissions_arr, axs[2, 0],
                    annotation_height_prop=[0.45, 0.45, 0.45])
    plot_emissions_drive(base_params, history_driving_emissions_arr, axs[2, 1],
                    annotation_height_prop=[0.45, 0.45, 0.45])

    # Adjust layout
    plt.tight_layout()
    
    # Save the combined figure
    plt.savefig(f"{fileName}/combined_plots_{save_name}.png", dpi=dpi)

def plot_total_emissions(base_params, data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    data_after_burn_in = data[:, :]
    time_steps = np.arange(data_after_burn_in.shape[1])
    
    mean_values = np.mean(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)
    
    ax.plot(time_steps, mean_values, label='Total Emissions', color='gray')
    ax.fill_between(time_steps, mean_values - ci_range, mean_values + ci_range, color='gray', alpha=0.3)
    
    ax.set_ylabel("Total Emissions, kgCO2")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)

def plot_emissions_produce(base_params, production_data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    time_steps = np.arange(production_data.shape[1])
    
    mean_production = np.mean(production_data, axis=0)
    ci_production = sem(production_data, axis=0) * t.ppf(0.975, df=production_data.shape[0] - 1)
    
    ax.plot(time_steps, mean_production, color='teal')
    ax.fill_between(time_steps, mean_production - ci_production, mean_production + ci_production, color='teal', alpha=0.3)

    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Poduction Emissions, kgCO2")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)

def plot_emissions_drive(base_params, driving_data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    time_steps = np.arange(driving_data.shape[1])
    
    mean_driving = np.mean(driving_data, axis=0)
    ci_driving = sem(driving_data, axis=0) * t.ppf(0.975, df=driving_data.shape[0] - 1)
    
    ax.plot(time_steps, mean_driving, color='brown')
    ax.fill_between(time_steps, mean_driving - ci_driving, mean_driving + ci_driving, color='brown', alpha=0.3)

    ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Driving Emissions, kgCO2")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)

# Helper functions for each subplot
def plot_ev_uptake(base_params, data, history_past_new_bought_vehicles_prop_ev, ax, 
                   annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot EV uptake on the provided axes."""
    
    data_after_burn_in = data[:,-311:]
    data_buy = history_past_new_bought_vehicles_prop_ev
    time_steps = np.arange(0, data_after_burn_in.shape[1])
    
    mean_values = np.nanmean(data_after_burn_in, axis=0)
    #median_values = np.median(data_after_burn_in, axis=0)
    data_buy_mean = np.nanmean(data_buy, axis=0)
    
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

    #ax.set_xlabel("Time Step, months")
    ax.set_ylabel("Proportion of EVs")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc = "lower right")

def plot_mean_price(base_params, history_mean_price_ICE_EV, history_median_price_ICE_EV, 
                    history_lower_price_ICE_EV, history_upper_price_ICE_EV, ax,
                    annotation_height_prop=[0.3, 0.3, 0.2]):
    """Plot mean price on the provided axes."""
    
    # Extract data
    mean_new_ICE = history_mean_price_ICE_EV[:, :, 0, 0]
    mean_second_hand_ICE = history_mean_price_ICE_EV[:, :, 1, 0]
    mean_new_EV = history_mean_price_ICE_EV[:, :, 0, 1]
    mean_second_hand_EV = history_mean_price_ICE_EV[:, :, 1, 1]
    
    # Check for second-hand EV data cutoff
    if np.any(np.isnan(history_mean_price_ICE_EV[:, :, 1, 1])):
        print("Warning: Second-hand EV data contains NaN values - this may cause cutoff in plots")
    
    # Percentile data
    lower_new_ICE = history_lower_price_ICE_EV[:, :, 0, 0]
    lower_second_hand_ICE = history_lower_price_ICE_EV[:, :, 1, 0]
    lower_new_EV = history_lower_price_ICE_EV[:, :, 0, 1]
    lower_second_hand_EV = history_lower_price_ICE_EV[:, :, 1, 1]
    
    upper_new_ICE = history_upper_price_ICE_EV[:, :, 0, 0]
    upper_second_hand_ICE = history_upper_price_ICE_EV[:, :, 1, 0]
    upper_new_EV = history_upper_price_ICE_EV[:, :, 0, 1]
    upper_second_hand_EV = history_upper_price_ICE_EV[:, :, 1, 1]
    
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
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.set_ylabel("Price, $")

def plot_mean_price_old(base_params, history_mean_price_ICE_EV, history_median_price_ICE_EV, 
                    history_lower_price_ICE_EV, history_upper_price_ICE_EV, ax,
                    annotation_height_prop=[0.3, 0.3, 0.2]):
    """Plot mean price on the provided axes."""

    mean_new_ICE = history_mean_price_ICE_EV[:, :, 0, 0]
    mean_second_hand_ICE = history_mean_price_ICE_EV[:, :, 1, 0]
    mean_new_EV = history_mean_price_ICE_EV[:, :, 0, 1]
    mean_second_hand_EV = history_mean_price_ICE_EV[:, :, 1, 1]
    
    # Calculate mean and CI for lower (25th) and upper (75th) percentiles
    lower_new_ICE = history_lower_price_ICE_EV[:, :, 0, 0]
    lower_second_hand_ICE = history_lower_price_ICE_EV[:, :, 1, 0]
    lower_new_EV = history_lower_price_ICE_EV[:, :, 0, 1]
    lower_second_hand_EV = history_lower_price_ICE_EV[:, :, 1, 1]
    
    upper_new_ICE = history_upper_price_ICE_EV[:, :, 0, 0]
    upper_second_hand_ICE = history_upper_price_ICE_EV[:, :, 1, 0]
    upper_new_EV = history_upper_price_ICE_EV[:, :, 0, 1]
    upper_second_hand_EV = history_upper_price_ICE_EV[:, :, 1, 1]
    
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

def plot_mean_car_age(base_params, data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot mean car age on the provided axes."""
    data_after_burn_in = data[:, :]
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
    
    ax.set_ylabel("Car Age, months")
    add_vertical_lines(ax, base_params, annotation_height_prop=annotation_height_prop)


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
    # Add vertical line with annotation
    ev_sale_start_time = 144
    ax.axvline(ev_sale_start_time, color="black", linestyle=':')
    ax.annotate("Policy end", xy=(ev_sale_start_time, annotation_height_0),
                rotation=90, verticalalignment='center', horizontalalignment='right',
                fontsize=12, color='black')

        
##########################################################################################################################
def plot_policy_results_ev(base_params, fileName, outputs_BAU, outputs, top_policies, x_label, y_label, prop_name, dpi=300):
    
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] - 1#-1 is because i cant keep track of tiem correnctly
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Business-as-Usual (BAU) case in black
    bau_mean = np.mean(outputs_BAU[prop_name], axis=0)[start:]
    bau_ci = (sem(outputs_BAU[prop_name], axis=0) * t.ppf(0.975, df=outputs_BAU[prop_name].shape[0] - 1))[start:]
    ax.plot(time_steps, bau_mean, color='black', label='Business-as-Usual')
    ax.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        intensity_1 = top_policies[(policy1, policy2)]['policy1_value']
        intensity_2 = top_policies[(policy1, policy2)]['policy2_value']

        policy_label = f"{policy1} ({round(intensity_1, 3)}), {policy2} ({round(intensity_2, 3)})"
        mean_values = np.mean(data[prop_name], axis=0)[start:]
        ci_values = (sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1))[start:]
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_path = f'{fileName}/Plots/{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)
    print("Done", prop_name)


def plot_policy_results(fileName, outputs_BAU, outputs, top_policies, x_label, y_label, prop_name, dpi=300):

    time_steps = np.arange(outputs_BAU[prop_name].shape[1])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Business-as-Usual (BAU) case in black
    bau_mean = np.mean(outputs_BAU[prop_name], axis=0)
    bau_ci = sem(outputs_BAU[prop_name], axis=0) * t.ppf(0.975, df=outputs_BAU[prop_name].shape[0] - 1)
    ax.plot(time_steps, bau_mean, color='black', label='Business-as-Usual')
    ax.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        intensity_1 = top_policies[(policy1, policy2)]['policy1_value']
        intensity_2 = top_policies[(policy1, policy2)]['policy2_value']

        policy_label = f"{policy1} ({round(intensity_1, 3)}), {policy2} ({round(intensity_2, 3)})"
        mean_values = np.mean(data[prop_name], axis=0)
        ci_values = sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1)
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_path = f'{fileName}/Plots/{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)

    print("Done", prop_name)

def plot_policy_results_cum(fileName, outputs_BAU, outputs, top_policies, x_label, y_label, prop_name, dpi=300):

    time_steps = np.arange(outputs_BAU[prop_name].shape[1])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Business-as-Usual (BAU) case in black
    data_bau_measure = np.cumsum(outputs_BAU[prop_name], axis = 1)
    bau_mean = np.mean(data_bau_measure, axis=0)
    bau_ci = sem(data_bau_measure, axis=0) * t.ppf(0.975, df=data_bau_measure.shape[0] - 1)
    ax.plot(time_steps, bau_mean, color='black', label='Business-as-Usual')
    ax.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        intensity_1 = top_policies[(policy1, policy2)]['policy1_value']
        intensity_2 = top_policies[(policy1, policy2)]['policy2_value']

        policy_label = f"{policy1} ({round(intensity_1, 3)}), {policy2} ({round(intensity_2, 3)})"
        data_measure = np.cumsum(data[prop_name], axis = 1)
        mean_values = np.mean(data_measure, axis=0)
        ci_values = sem(data_measure, axis=0) * t.ppf(0.975, df=data_measure.shape[0] - 1)
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_path = f'{fileName}/Plots/cum_{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)

    print("Done", prop_name)



def main(fileName):
    base_params = load_object(fileName + "/Data", "base_params")
    outputs_BAU = load_object(fileName + "/Data", "outputs_BAU")
    outputs = load_object(fileName + "/Data", "outputs")
    top_policies = load_object(fileName + "/Data", "top_policies")

    plot_combined_figures(base_params, fileName, outputs[('Carbon_price', 'Electricity_subsidy')], "c_e", dpi=300)
    plot_combined_figures(base_params, fileName, outputs[('Carbon_price', 'Adoption_subsidy_used')],"c_a", dpi=300)
    plot_combined_figures(base_params, fileName, outputs[('Electricity_subsidy', 'Production_subsidy')],"e_p", dpi=300)

    plot_policy_results_cum(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Emissions Cumulative, kgCO2",
        "history_total_emissions"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Net cost, $", 
        "history_policy_net_cost"
    )

    plot_policy_results_ev(
        base_params,
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "EV uptake proportion", 
        "history_prop_EV"
    )

    plot_policy_results(
        fileName, 
        outputs_BAU,
        outputs,
        top_policies,
        "Time Step, months",
        "Total Emissions, kgCO2",
        "history_total_emissions"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies, 
        "Time Step, months", 
        "Driving Emissions, kgCO2", 
        "history_driving_emissions"
        )
    
    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies, 
        "Time Step, months", 
        "Production Emissions, kgCO2", 
        "history_production_emissions"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Total Utility", 
        "history_total_utility"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Market Concentration, HHI", 
        "history_market_concentration"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Total Profit, $", 
        "history_total_profit"
    )

    plt.show()

if __name__ == "__main__":
    main("results/2D_low_intensity_policies_17_34_40__01_04_2025")
