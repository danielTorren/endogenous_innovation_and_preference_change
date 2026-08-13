"""
All plots for a multi-seed calibration run, from the one `outputs` dict written
by package/generating_data/calibration_gen.py.

Figures produced, all into <fileName>/Plots:
  calibration_fit           - simulated vs Californian EV share, prices, HHI, car age
  multi_seed_dashboard      - EV share, EV price, emissions, utility
  multi_seed_dashboard_extra- emissions split, HHI, profit, car age, price spread
  multi_seed_2d_scatter     - cars on sale, range vs price, against real vehicles
  multi_seed_2d_contour     - the same, as a density map
  calibration_targets       - the four target variables against their observed ranges
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import sem, t, gaussian_kde

from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import info_real_cars

CALIBRATION_END_YEAR = 2024  # The end of calibration is Jan 2024
CALIBRATION_START_YEAR = 2001  # The end of burn-in

EV_COLOR = "#2E8B57"   # SeaGreen
ICE_COLOR = "#4169E1"  # RoyalBlue

# Observed target ranges, from Table "Target variable ranges" in the manuscript.
# New-car prices are TRANSACTION prices (Grieco, Murry & Yurukoglu 2024, Fig II),
# which is why record_calibration_targets() records prices paid rather than list
# prices. There is no published target for the used-market quality spread; the
# comparator there is the model's own used price spread, see
# plot_calibration_targets.
TARGET_NEW_PRICE_P25 = 32359.41
TARGET_NEW_PRICE_P75 = 57784.66
TARGET_HHI = (0.11, 0.18)
TARGET_FLEET_AGE_YEARS = (10, 12)

# Transaction-volume and channel-mix targets, built from US light-vehicle data.
# 2023: 15.46M new (NADA Market Beat), 19.0M used RETAIL and 35.9M used in total
# including wholesale (Cox Automotive), 286M vehicles in operation at Jan 2024
# (S&P Global Mobility).
#
# Which used figure applies is a modelling choice, and it moves the target a lot.
# The model has one merchant intermediating every used sale, and each transaction
# puts a car with a new end user, so wholesale dealer-to-dealer moves should be
# excluded while private-party sales should be included. Retail-only gives a used
# share of 19.0/(19.0+15.46) = 0.55; all used transactions give 0.70; the paper's
# own cited figure (74%) sits at the broad end. The band spans that disagreement
# and the midpoint assumes retail plus roughly 12M private-party sales.
TARGET_USED_SHARE = (0.55, 0.74)
TARGET_USED_SHARE_MID = 0.67

# P(a given car changes owner in a year). Transactions over fleet: (15.46 + 31)/286
# = 0.16 on the mid definition, 0.12 retail-only, 0.18 counting wholesale. It also
# agrees with the ownership-duration route, since S&P Global Mobility report
# ownership of five years or less for nearly two thirds of Americans, i.e. a mean
# near six years, giving 1/6 = 0.17.
TARGET_PROB_BUY = (0.12, 0.18)
TARGET_PROB_BUY_MID = 0.16

# Okabe-Ito, as used by the dashboards below.
C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
C_TARGET = "#666666"


####################################################################################
# Shared helpers
####################################################################################

def get_shape_safe(data):
    """Safely get shape of data, handling different types"""
    if hasattr(data, 'shape'):
        return data.shape
    elif isinstance(data, (list, tuple)):
        return (len(data),)
    elif isinstance(data, (int, float, np.number)):
        return "scalar"
    else:
        return "unknown type"


def add_vertical_lines(ax, base_params, annotation_height_prop=[0.2, 0.2, 0.2, 0.2]):
    """
    Dashed vertical lines at the policy milestones, in ABSOLUTE time steps.

    Use this on axes whose x data is still absolute steps (the dashboards).
    """
    ev_production_start_time = base_params["ev_production_start_time"]
    # Policies start at the end of calibration, which is burn-in + calibration
    # steps into the run (duration_calibration excludes the burn-in)
    calibration_end_step = base_params["duration_burn_in"] + base_params["duration_calibration"]

    y_min, y_max = ax.get_ylim()
    heights = [y_min + prop * (y_max - y_min) for prop in annotation_height_prop]

    # EV Sale Start
    if ev_production_start_time > 0:
        ax.axvline(ev_production_start_time, color="black", linestyle=':')
        ax.annotate("EV Sale Start", xy=(ev_production_start_time, heights[0]),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    # EV Adoption Subsidy Start
    if base_params["EV_rebate_state"]:
        rebate_start_time = base_params["parameters_rebate_calibration"]["start_time"]
        ax.axvline(rebate_start_time, color="black", linestyle='-.')
        ax.annotate("EV Adoption Subsidy Start", xy=(rebate_start_time, heights[1]),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    # Policy Start
    if base_params["duration_future"] > 0:
        ax.axvline(calibration_end_step, color="black", linestyle='--')
        ax.annotate("Policy Start", xy=(calibration_end_step, heights[2]),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

        # Policy End
        if base_params["duration_future"] >= 144:
            policy_end_time = calibration_end_step + 144
            ax.axvline(policy_end_time, color="black", linestyle='--')
            ax.annotate("Policy End", xy=(policy_end_time, heights[3]),
                        rotation=90, verticalalignment='center', horizontalalignment='right',
                        fontsize=8, color='black')


def add_vertical_lines_from_burn_in(ax, base_params, annotation_height_prop=[0.2, 0.2, 0.2]):
    """
    The same milestones, for axes re-based so that step 0 is the END OF BURN-IN.

    The calibration-fit figure plots time from the end of burn-in, so the policy
    start sits at duration_calibration rather than burn-in + duration_calibration.
    """
    y_min, y_max = ax.get_ylim()
    heights = [y_min + prop * (y_max - y_min) for prop in annotation_height_prop]

    ev_production_start_time = base_params["ev_production_start_time"]
    if ev_production_start_time > 0:
        ax.axvline(ev_production_start_time, color="black", linestyle=':')
        ax.annotate("EV Sale Start", xy=(ev_production_start_time, heights[0]),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    if base_params["EV_rebate_state"]:
        rebate_start_time = base_params["parameters_rebate_calibration"]["start_time"]
        ax.axvline(rebate_start_time, color="black", linestyle='-.')
        ax.annotate("EV Adoption Subsidy Start", xy=(rebate_start_time, heights[1]),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    if base_params["duration_future"] > 0:
        policy_start_time = base_params["duration_calibration"]
        ax.axvline(policy_start_time, color="black", linestyle='--')
        ax.annotate("Policy Start", xy=(policy_start_time, heights[2]),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')


def get_plot_window(base_params, total_steps):
    """
    Work out which steps to plot, and which step corresponds to Jan 2024.

    Runs with a future period show only that period. Calibration-only runs
    (duration_future == 0) have nothing after the end of calibration, so they
    show the calibration period from the end of burn-in instead - otherwise
    every panel slices an empty array and comes out blank.
    """
    calibration_end_step = base_params["duration_burn_in"] + base_params["duration_calibration"]

    if base_params["duration_future"] > 0 and calibration_end_step < total_steps:
        start_step = calibration_end_step
    else:
        start_step = base_params["duration_burn_in"]

    # Never leave a window with no room in it (e.g. truncated output)
    start_step = max(0, min(start_step, total_steps - 1))
    return start_step, calibration_end_step


def step_to_year(step, calibration_end_step):
    return CALIBRATION_END_YEAR + (step - calibration_end_step) / 12


def format_year_axis(axs, start_step, total_steps, calibration_end_step, spacing=5):
    """Label the x axis in calendar years, ticking every `spacing` years."""
    start_year = step_to_year(start_step, calibration_end_step)
    end_year = step_to_year(total_steps, calibration_end_step)

    first_tick = np.ceil(start_year / spacing) * spacing
    tick_years = np.arange(first_tick, end_year + spacing, spacing)
    tick_positions = calibration_end_step + (tick_years - CALIBRATION_END_YEAR) * 12

    visible = (tick_positions >= start_step) & (tick_positions <= total_steps)
    tick_positions = tick_positions[visible]
    tick_labels = [str(int(y)) for y in tick_years[visible]]

    for ax in axs:
        ax.set_xlim(start_step, total_steps)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Year", fontsize=14)

    return f"{start_year:.0f}-{end_year:.0f}"


def plot_series_with_ci(ax, data, time_steps, start_step, color, linestyle='-', label=None, safe=False):
    """
    Mean across seeds with a 95% CI band, over the steps from `start_step` on.

    `data` can be a tuple/list of per-seed series as well as an array, so it is
    coerced first. With safe=True a failure is annotated on the axis rather than
    raised, which the extra dashboard relies on for its optional metrics.
    """
    try:
        sliced_data = np.asarray(data, dtype=float)[:, start_step:]
        mean = np.nanmean(sliced_data, axis=0)
        ci = sem(sliced_data, axis=0, nan_policy='omit') * t.ppf(0.975, df=sliced_data.shape[0] - 1)
        ax.plot(time_steps, mean, color=color, linestyle=linestyle, label=label)
        ax.fill_between(time_steps, mean - ci, mean + ci, color=color, alpha=0.2)
        return True
    except Exception as e:
        if not safe:
            raise
        print(f"  [!] Failed to plot {label}: {str(e)}")
        ax.text(0.5, 0.5, f'{label} unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, alpha=0.7)
        return False


def mean_and_ci(data):
    """Nan-tolerant mean and 95% CI across seeds (axis 0)."""
    data = np.asarray(data, dtype=float)
    mean = np.nanmean(data, axis=0)
    n = np.sum(~np.isnan(data), axis=0)
    ci = t.ppf(0.975, df=n - 1) * sem(data, axis=0, nan_policy='omit')
    return mean, ci


def print_data_shapes(outputs, base_params):
    """Print shapes of all relevant data arrays for debugging with robust error handling"""
    print("\n" + "="*60)
    print("DATA SHAPES FOR DEBUGGING")
    print("="*60)

    for key in sorted(outputs.keys()):
        try:
            print(f"[ok] {key}: {get_shape_safe(outputs[key])}")
        except Exception as e:
            print(f"[!] {key}: Error accessing - {str(e)}")

    # Print simulation timeline info
    print("\n" + "="*60)
    print("--- Simulation Timeline ---")
    try:
        total_steps = np.asarray(outputs["history_prop_EV"]).shape[1]
        start_step, calibration_end_step = get_plot_window(base_params, total_steps)
        print(f"Total steps in simulation: {total_steps}")
        print(f"Burn-in: {base_params['duration_burn_in']}, calibration: "
              f"{base_params['duration_calibration']}, future: {base_params['duration_future']}")
        print(f"End of calibration (Jan {CALIBRATION_END_YEAR}) is step {calibration_end_step}")
        if base_params["duration_future"] > 0:
            print(f"Future period: {total_steps - calibration_end_step} steps "
                  f"({(total_steps - calibration_end_step) / 12:.1f} years)")
        else:
            print("No future period - plotting the calibration period instead")
        print(f"Plotting steps {start_step}-{total_steps} "
              f"({step_to_year(start_step, calibration_end_step):.0f}-"
              f"{step_to_year(total_steps, calibration_end_step):.0f})")
    except Exception as e:
        print(f"Error determining timeline: {str(e)}")

    print("="*60 + "\n")


####################################################################################
# Figure 1: calibration fit against the Californian data
####################################################################################

def plot_calibration_fit(base_params, fileName, outputs, dpi=200):
    """2x2 figure: EV uptake, prices, market concentration and mean car age."""

    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")
    EV_stock_prop_2010_23 = calibration_data_output["EV Prop"]
    EV_sales_prop_2020_23 = calibration_data_output["EV Sales Prop"]

    fig, axs = plt.subplots(2, 2, figsize=(17, 10), sharex=True)

    # Plot 1: EV Uptake (top-left)
    plot_ev_uptake(EV_stock_prop_2010_23, EV_sales_prop_2020_23, base_params,
                   outputs["history_prop_EV"],
                   outputs["history_past_new_bought_vehicles_prop_ev"], axs[0, 0],
                   annotation_height_prop=[0.9, 0.9, 0.9])

    # Plot 2: Mean Price (top-right)
    plot_mean_price(base_params, outputs["history_mean_price_ICE_EV"],
                    outputs["history_median_price_ICE_EV"],
                    outputs["history_lower_percentile_price_ICE_EV"],
                    outputs["history_upper_percentile_price_ICE_EV"], axs[0, 1],
                    annotation_height_prop=[0.1, 0.1, 0.1])

    # Plot 3: Market Concentration (bottom-left)
    plot_market_concentration(base_params, outputs["history_market_concentration"], axs[1, 0],
                              annotation_height_prop=[0.3, 0.3, 0.3])

    # Plot 4: Mean Car Age (bottom-right)
    plot_mean_car_age(base_params, np.asarray(outputs["history_mean_car_age"]), axs[1, 1],
                      annotation_height_prop=[0.3, 0.3, 0.3])

    #########################################################################
    # Set x-axis ticks every 5 years starting at the end of burn-in, stopping at
    # the last year of data. Time is re-based so step 0 is the end of burn-in.
    tick_interval_years = 5
    months_per_year = 12

    max_time_step = max(
        ax.get_lines()[0].get_xdata().max() for ax in [axs[1, 0], axs[1, 1]]
    )
    max_year = CALIBRATION_START_YEAR + int(max_time_step // months_per_year)
    last_tick_year = max_year - (max_year - CALIBRATION_START_YEAR) % tick_interval_years
    tick_years = np.arange(CALIBRATION_START_YEAR, last_tick_year + 1, tick_interval_years)

    tick_positions = (tick_years - CALIBRATION_START_YEAR) * months_per_year
    tick_labels = [str(year) for year in tick_years]

    for ax in axs[1]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Year")
    #########################################################################

    plt.tight_layout(rect=[0.01, 0.0, 0.98, 1])  # Leaves space at the bottom
    plt.subplots_adjust(wspace=0.15)  # increase spacing between columns

    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/calibration_fit.png", dpi=dpi)
    fig.savefig(f"{save_path}/calibration_fit.eps", dpi=dpi)
    print(f"Saved to {save_path}/calibration_fit.png")


def plot_ev_uptake(real_data, EV_sales_prop_2020_23, base_params, data,
                   history_past_new_bought_vehicles_prop_ev, ax,
                   annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot EV uptake on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = np.asarray(data)[:, burn_in_step:]
    data_buy = np.asarray(history_past_new_bought_vehicles_prop_ev)[:, burn_in_step:]
    time_steps = np.arange(0, data_after_burn_in.shape[1])

    init_real = 108 + 4  # STARTS AT APRIL of THE END OF 2010
    time_steps_real = np.arange(init_real, init_real + len(real_data) * 12, 12)

    init_real_sales = 108 + 120 + 11  # STARTS AT APRIL of THE END OF 2010
    time_steps_real_sales = np.arange(init_real_sales, init_real_sales + len(EV_sales_prop_2020_23) * 12, 12)

    mean_values = np.mean(data_after_burn_in, axis=0)
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
    ax.plot(time_steps_real_sales, EV_sales_prop_2020_23, label="California EV Sales",
            color='black', linestyle="dotted")

    ax.set_ylabel("EV Proportion", fontsize=16)
    add_vertical_lines_from_burn_in(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc="upper left", fontsize=12)


def plot_mean_price(base_params, history_mean_price_ICE_EV, history_median_price_ICE_EV,
                    history_lower_price_ICE_EV, history_upper_price_ICE_EV, ax,
                    annotation_height_prop=[0.3, 0.3, 0.2]):
    """Plot mean price on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]

    history_mean_price_ICE_EV = np.asarray(history_mean_price_ICE_EV)
    history_lower_price_ICE_EV = np.asarray(history_lower_price_ICE_EV)
    history_upper_price_ICE_EV = np.asarray(history_upper_price_ICE_EV)

    # Extract data
    mean_new_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 0, 0]
    mean_second_hand_ICE = history_mean_price_ICE_EV[:, burn_in_step:, 1, 0]
    mean_new_EV = history_mean_price_ICE_EV[:, burn_in_step:, 0, 1]
    mean_second_hand_EV = history_mean_price_ICE_EV[:, burn_in_step:, 1, 1]

    # Check for second-hand EV data cutoff
    if np.any(np.isnan(mean_second_hand_EV)):
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

    # Mean statistics
    overall_mean_new_ICE, ci_new_ICE = mean_and_ci(mean_new_ICE)
    overall_mean_second_hand_ICE, ci_second_hand_ICE = mean_and_ci(mean_second_hand_ICE)
    overall_mean_new_EV, ci_new_EV = mean_and_ci(mean_new_EV)
    overall_mean_second_hand_EV, ci_second_hand_EV = mean_and_ci(mean_second_hand_EV)

    # 25th percentile statistics
    overall_lower_new_ICE, ci_lower_new_ICE = mean_and_ci(lower_new_ICE)
    overall_lower_second_hand_ICE, _ = mean_and_ci(lower_second_hand_ICE)
    overall_lower_new_EV, ci_lower_new_EV = mean_and_ci(lower_new_EV)
    overall_lower_second_hand_EV, _ = mean_and_ci(lower_second_hand_EV)

    # 75th percentile statistics
    overall_upper_new_ICE, ci_upper_new_ICE = mean_and_ci(upper_new_ICE)
    overall_upper_second_hand_ICE, _ = mean_and_ci(upper_second_hand_ICE)
    overall_upper_new_EV, ci_upper_new_EV = mean_and_ci(upper_new_EV)
    overall_upper_second_hand_EV, _ = mean_and_ci(upper_second_hand_EV)

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
    ax.plot(time_steps, overall_mean_second_hand_ICE, label="Used Car Mean Price ICE",
            color="blue", linestyle="dashed")
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

    ax.plot(time_steps, overall_mean_second_hand_EV, label="Used Car Mean Price EV",
            color="green", linestyle="dashed")
    ax.fill_between(
        time_steps,
        overall_mean_second_hand_EV - ci_second_hand_EV,
        overall_mean_second_hand_EV + ci_second_hand_EV,
        color="green",
        alpha=0.2
    )

    # Add simplified percentile legend items (using representative lines)
    legend_elements = [
        Line2D([0], [0], color="grey", linestyle=(0, (3, 1, 1, 1)), lw=2, label='25th Percentile'),
        Line2D([0], [0], color="grey", linestyle='dotted', lw=2, label='75th Percentile'),
        Line2D([0], [0], color="grey", alpha=0.1, lw=10, label='95% Confidence Interval'),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_elements + handles, ncol=1, fontsize=10)

    ax.set_ylabel("Price, $", fontsize=16)
    add_vertical_lines_from_burn_in(ax, base_params, annotation_height_prop=annotation_height_prop)


def plot_market_concentration(base_params, data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot market concentration on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = np.asarray(data)[:, burn_in_step:]
    time_steps = np.arange(0, data_after_burn_in.shape[1])

    mean_values = np.mean(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)

    ax.plot(time_steps, mean_values, label='Mean', color='purple')
    ax.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='purple',
        alpha=0.3,
        label='95% Confidence Interval'
    )

    ax.set_xlabel("Time Step, months", fontsize=16)
    ax.set_ylabel("Market Concentration, HHI", fontsize=16)
    add_vertical_lines_from_burn_in(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc="upper left", fontsize=12)


def plot_mean_car_age(base_params, data, ax, annotation_height_prop=[0.8, 0.8, 0.8]):
    """Plot mean car age on the provided axes."""
    burn_in_step = base_params["duration_burn_in"]
    data_after_burn_in = np.asarray(data)[:, burn_in_step:]
    time_steps = np.arange(0, data_after_burn_in.shape[1])

    mean_values = np.mean(data_after_burn_in, axis=0)
    ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)

    ax.plot(time_steps, mean_values, label='Mean', color='red')
    ax.fill_between(
        time_steps,
        mean_values - ci_range,
        mean_values + ci_range,
        color='red',
        alpha=0.3,
        label='95% Confidence Interval'
    )

    ax.set_xlabel("Time Step, months", fontsize=16)
    ax.set_ylabel("Car Age, months", fontsize=16)
    add_vertical_lines_from_burn_in(ax, base_params, annotation_height_prop=annotation_height_prop)
    ax.legend(loc="upper left", fontsize=12)


####################################################################################
# The four calibration target variables, in one figure
####################################################################################

def _target_band(ax, lo, hi, label):
    """Shade an observed target range. Recessive: it is a reference, not a series."""
    ax.axhspan(lo, hi, color=C_TARGET, alpha=0.15, zorder=0)
    for y in (lo, hi):
        ax.axhline(y, color=C_TARGET, linestyle="--", linewidth=1, zorder=1)
    ax.plot([], [], color=C_TARGET, linestyle="--", linewidth=1, label=label)


def _final_mean(series, months=12):
    """Mean of the last `months` steps of a (T,) series, ignoring NaN."""
    tail = np.asarray(series, dtype=float)[-months:]
    return float(np.nanmean(tail)) if np.any(~np.isnan(tail)) else np.nan


def plot_calibration_targets(base_params, fileName, outputs, dpi=200):
    """
    2x2 figure: each of the four calibration target variables against the range
    it is supposed to land in, so a run can be judged at a glance.

    Panels are new-car transaction prices, the used-market quality spread, HHI,
    and whole-fleet mean age. Each panel title carries the final-year mean and
    the target, so "how is it doing" is readable without measuring the axes.

    The quality-spread panel has no published target. It plots the dollar value
    of the used stock's quality range against the used stock's own price range,
    both in dollars on one axis. When the quality spread is far below the price
    spread, trading down to a cheaper car costs little utility and the fleet
    churns; that ratio is annotated.
    """
    burn_in = base_params["duration_burn_in"]

    new_q = np.asarray(outputs["history_new_car_price_quantiles"], dtype=float)[:, burn_in:, :]
    used_q = np.asarray(outputs["history_used_car_price_quantiles"], dtype=float)[:, burn_in:, :]
    qual_spread = np.asarray(outputs["history_used_stock_quality_spread"], dtype=float)[:, burn_in:]
    hhi = np.asarray(outputs["history_market_concentration"], dtype=float)[:, burn_in:]
    age_years = np.asarray(outputs["history_mean_car_age_fleet"], dtype=float)[:, burn_in:] / 12

    # Purchase counts -> the two ratio targets. Rolling 12-month RATIO OF SUMS,
    # not a mean of per-step ratios: months with few transactions would otherwise
    # get the same weight as busy ones, and a month with zero purchases has an
    # undefined used share.
    counts = np.asarray(outputs["history_purchase_counts"], dtype=float)[:, burn_in:, :]
    win = np.ones(12)
    def roll(x):
        return np.apply_along_axis(lambda v: np.convolve(v, win, mode="valid"), 1, x)
    r_new, r_used, r_opp = roll(counts[:, :, 0]), roll(counts[:, :, 1]), roll(counts[:, :, 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        used_share = r_used/(r_new + r_used)
        # Transactions per vehicle per year is the OBSERVABLE, so that is what the
        # target band applies to. Dividing by switch opportunities instead would
        # compare against a quantity that only exists inside the model, since the
        # opportunity rate IS the prob_switch_car assumption. The two coincide at
        # prob_switch_car = 1/12 (one opportunity per agent per year) and diverge
        # as soon as it is recalibrated, which is the whole point of the panel.
        turnover = (r_new + r_used)/base_params["parameters_social_network"]["num_individuals"]
        prob_buy = (r_new + r_used)/r_opp
    ratio_steps = np.arange(len(win) - 1, len(win) - 1 + used_share.shape[1])

    time_steps = np.arange(new_q.shape[1])
    fig, axs = plt.subplots(3, 2, figsize=(16, 13))

    # ---- 1. New-car transaction prices vs the observed p25-p75 -------------
    ax = axs[0, 0]
    _target_band(ax, TARGET_NEW_PRICE_P25, TARGET_NEW_PRICE_P75, "Observed p25-p75")
    for idx, (lbl, colour, style) in enumerate(
            [("p25", C_BLUE, "--"), ("median", C_BLUE, "-"), ("p75", C_BLUE, ":")]):
        mean, ci = mean_and_ci(new_q[:, :, idx])
        ax.plot(time_steps, mean, color=colour, linestyle=style, label=lbl)
        ax.fill_between(time_steps, mean - ci, mean + ci, color=colour, alpha=0.15)
    model_iqr = _final_mean(np.nanmean(new_q[:, :, 2] - new_q[:, :, 0], axis=0))
    target_iqr = TARGET_NEW_PRICE_P75 - TARGET_NEW_PRICE_P25
    # Dollar signs are escaped throughout: matplotlib reads a bare $ as a
    # mathtext delimiter, which silently italicises the rest of the title.
    ax.set_title(f"New-car transaction prices\nfinal-year IQR \\${model_iqr:,.0f} "
                 f"vs target \\${target_iqr:,.0f}  ({model_iqr/target_iqr:.2f}x)", fontsize=13)
    ax.set_ylabel("Price, \\$", fontsize=14)
    # Clip the burn-in transient, which is otherwise wide enough to flatten the
    # rest of the series, and never show negative prices.
    ax.set_ylim(0, max(TARGET_NEW_PRICE_P75*1.6, np.nanpercentile(new_q[:, :, 2], 90)))

    # ---- 2. Used market: is quality worth as much as price? ----------------
    ax = axs[0, 1]
    mean_qual, ci_qual = mean_and_ci(qual_spread)
    ax.plot(time_steps, mean_qual, color=C_GREEN, label=r"quality spread, $\beta_{med}\cdot sd(Q^\alpha)$")
    ax.fill_between(time_steps, mean_qual - ci_qual, mean_qual + ci_qual, color=C_GREEN, alpha=0.15)
    used_width = used_q[:, :, 2] - used_q[:, :, 0]
    mean_width, ci_width = mean_and_ci(used_width)
    ax.plot(time_steps, mean_width, color=C_ORANGE, label="used price spread, p90 - p10")
    ax.fill_between(time_steps, mean_width - ci_width, mean_width + ci_width, color=C_ORANGE, alpha=0.15)
    q_end, p_end = _final_mean(mean_qual), _final_mean(mean_width)
    ax.set_title(f"Used market: quality spread vs price spread\n"
                 f"final year \\${q_end:,.0f} vs \\${p_end:,.0f}  "
                 f"(ratio {q_end/p_end:.2f}, want order 1)", fontsize=13)
    ax.set_ylabel("Dollars", fontsize=14)
    ax.set_ylim(0, max(np.nanpercentile(mean_width, 95), np.nanpercentile(mean_qual, 95))*1.6)

    # ---- 3. HHI ------------------------------------------------------------
    ax = axs[1, 0]
    _target_band(ax, *TARGET_HHI, "Observed range")
    mean, ci = mean_and_ci(hhi)
    ax.plot(time_steps, mean, color=C_PINK, label="HHI, unit shares")
    ax.fill_between(time_steps, mean - ci, mean + ci, color=C_PINK, alpha=0.2)
    ax.set_title(f"Market concentration\nfinal-year {_final_mean(mean):.3f} "
                 f"vs target {TARGET_HHI[0]}-{TARGET_HHI[1]}", fontsize=13)
    ax.set_ylabel("HHI", fontsize=14)
    ax.set_ylim(0, max(0.4, np.nanpercentile(mean, 95)*1.4))

    # ---- 4. Whole-fleet mean age ------------------------------------------
    ax = axs[1, 1]
    _target_band(ax, *TARGET_FLEET_AGE_YEARS, "Observed range")
    mean, ci = mean_and_ci(age_years)
    ax.plot(time_steps, mean, color=C_BLUE, label="mean age, whole fleet")
    ax.fill_between(time_steps, mean - ci, mean + ci, color=C_BLUE, alpha=0.2)
    ax.set_title(f"Fleet mean age\nfinal-year {_final_mean(mean):.1f} yr "
                 f"vs target {TARGET_FLEET_AGE_YEARS[0]}-{TARGET_FLEET_AGE_YEARS[1]} yr", fontsize=13)
    ax.set_ylabel("Age, years", fontsize=14)
    ax.set_ylim(0, max(TARGET_FLEET_AGE_YEARS[1]*1.3, np.nanpercentile(mean, 95)*1.3))

    # ---- 5. Used share of purchases -----------------------------------------
    ax = axs[2, 0]
    _target_band(ax, *TARGET_USED_SHARE, "Observed range (definition-dependent)")
    ax.axhline(TARGET_USED_SHARE_MID, color=C_TARGET, linewidth=1.5, alpha=0.8)
    mean, ci = mean_and_ci(used_share)
    ax.plot(ratio_steps, mean, color=C_ORANGE, label="used / (new + used), 12-mo")
    ax.fill_between(ratio_steps, mean - ci, mean + ci, color=C_ORANGE, alpha=0.2)
    ax.set_title(f"Used share of purchases\nfinal-year {_final_mean(mean):.3f} "
                 f"vs target {TARGET_USED_SHARE_MID:.2f} "
                 f"({TARGET_USED_SHARE[0]:.2f}-{TARGET_USED_SHARE[1]:.2f})", fontsize=13)
    ax.set_ylabel("Share of purchases", fontsize=14)
    ax.set_ylim(0, 1)

    # ---- 6. Transaction volume ---------------------------------------------
    # The banded series is the observable. P(buy | opportunity) is drawn alongside
    # as a diagnostic, unbanded, because it is what the choice model can actually
    # move: turnover = 12 * prob_switch_car * P(buy | opportunity), so if P(buy)
    # bottoms out well above target, the residual has to come from the
    # opportunity rate.
    ax = axs[2, 1]
    _target_band(ax, *TARGET_PROB_BUY, "Observed range")
    ax.axhline(TARGET_PROB_BUY_MID, color=C_TARGET, linewidth=1.5, alpha=0.8)
    mean, ci = mean_and_ci(turnover)
    ax.plot(ratio_steps, mean, color=C_GREEN, label="transactions per vehicle per year")
    ax.fill_between(ratio_steps, mean - ci, mean + ci, color=C_GREEN, alpha=0.2)
    mean_pb, _ = mean_and_ci(prob_buy)
    ax.plot(ratio_steps, mean_pb, color=C_GREEN, linestyle=":", alpha=0.7,
            label="P(buy | opportunity), diagnostic")
    ax.set_title(f"Transaction volume\nfinal-year {_final_mean(mean):.3f} per vehicle/yr "
                 f"vs target {TARGET_PROB_BUY_MID:.2f} "
                 f"({TARGET_PROB_BUY[0]:.2f}-{TARGET_PROB_BUY[1]:.2f})", fontsize=13)
    ax.set_ylabel("Per vehicle per year", fontsize=14)
    ax.set_ylim(0, 1)

    for ax in axs.flat:
        add_vertical_lines_from_burn_in(ax, base_params, annotation_height_prop=[0.9, 0.9, 0.9])
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)

    # Year ticks, same rebasing as plot_calibration_fit: step 0 is end of burn-in.
    tick_years = np.arange(CALIBRATION_START_YEAR,
                           CALIBRATION_START_YEAR + len(time_steps)//12 + 1, 5)
    tick_positions = (tick_years - CALIBRATION_START_YEAR)*12
    visible = tick_positions <= time_steps[-1]
    for ax in axs.flat:
        ax.set_xticks(tick_positions[visible])
        ax.set_xticklabels([str(y) for y in tick_years[visible]])
        ax.set_xlabel("Year", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/calibration_targets.png", dpi=dpi)
    print(f"Saved to {save_path}/calibration_targets.png")

    # Same four numbers as text, so a headless run still reports them.
    print("\n--- calibration targets, final-year mean over seeds ---")
    print(f"  new-car price IQR      ${model_iqr:>10,.0f}   target ${target_iqr:,.0f} "
          f"({model_iqr/target_iqr:.2f}x)")
    print(f"  used quality spread    ${q_end:>10,.0f}   vs used price spread ${p_end:,.0f} "
          f"(ratio {q_end/p_end:.2f})")
    print(f"  HHI                     {_final_mean(mean_and_ci(hhi)[0]):>10.3f}   target {TARGET_HHI}")
    print(f"  fleet mean age, years   {_final_mean(mean_and_ci(age_years)[0]):>10.1f}   target {TARGET_FLEET_AGE_YEARS}")
    print(f"  used share of purchases {_final_mean(mean_and_ci(used_share)[0]):>10.3f}   target {TARGET_USED_SHARE_MID} {TARGET_USED_SHARE}")
    turnover_now = _final_mean(mean_and_ci(turnover)[0])
    prob_buy_now = _final_mean(mean_and_ci(prob_buy)[0])
    print(f"  transactions/vehicle/yr {turnover_now:>10.3f}   target {TARGET_PROB_BUY_MID} {TARGET_PROB_BUY}")
    print(f"    P(buy | opportunity)  {prob_buy_now:>10.3f}   -> prob_switch_car implied by the target: "
          f"{TARGET_PROB_BUY_MID/(12*prob_buy_now):.4f} "
          f"(currently {base_params['parameters_social_network']['prob_switch_car']})")


####################################################################################
# Figures 2 and 3: the multi-seed dashboards
####################################################################################

def plot_multi_seed_dashboard(base_params, fileName, outputs, dpi=200):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    total_steps = np.asarray(outputs["history_prop_EV"]).shape[1]
    start_step, calibration_end_step = get_plot_window(base_params, total_steps)
    time_steps = np.arange(start_step, total_steps)

    def plot_line_with_ci(ax, data, color, linestyle='-', label=None):
        plot_series_with_ci(ax, data, time_steps, start_step, color, linestyle, label)

    # --- EV Adoption and Sales Share
    ax = axs[0, 0]
    plot_line_with_ci(ax, outputs["history_prop_EV"], '#0072B2', '-', 'EV Adoption')
    plot_line_with_ci(ax, outputs["history_past_new_bought_vehicles_prop_ev"], '#0072B2', '--', 'EV Sales')
    ax.set_ylabel("EV Share", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.6, 0.2, 0.2, 0.2])
    ax.legend(loc='lower right', fontsize='small')

    # --- EV Price (New and Used)
    ax = axs[0, 1]
    mean_price = np.asarray(outputs["history_mean_price_ICE_EV"])
    plot_line_with_ci(ax, mean_price[:, :, 0, 1], '#0072B2', '-', 'New EV')
    plot_line_with_ci(ax, mean_price[:, :, 1, 1], '#0072B2', '--', 'Used EV')
    ax.set_ylabel("EV Sale Price, $", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.9, 0.2, 0.2, 0.2])
    ax.legend(loc='upper right', fontsize='small')

    # --- Flow Emissions
    ax = axs[1, 0]
    plot_line_with_ci(ax, outputs["history_total_emissions"] * 1e-9, '#D55E00', '-', 'Flow Emissions')
    ax.set_ylabel("Flow Emissions, MTCO2", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Cumulative Emissions (Calculated on full history, then sliced)
    ax = axs[1, 1]
    cum_emissions = np.cumsum(outputs["history_total_emissions"], axis=1) * 1e-9
    plot_line_with_ci(ax, cum_emissions, '#D55E00', '-', 'Cumulative Emissions')
    ax.set_ylabel("Cumulative Emissions, MTCO2", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.2, 0.2, 0.2, 0.2])

    # --- Flow Utility
    ax = axs[2, 0]
    plot_line_with_ci(ax, outputs["history_total_utility"] * 1e-9, '#009E73', '-', 'Flow Utility')
    ax.set_ylabel("Flow Utility, bn $", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.2, 0.2, 0.2, 0.2])

    # --- Cumulative Net Cost or Utility
    ax = axs[2, 1]
    if "history_policy_net_cost" in outputs:
        cum_cost = np.cumsum(outputs["history_policy_net_cost"], axis=1) * 1e-9
        plot_line_with_ci(ax, cum_cost, '#009E73', '-', 'Cumulative Net Cost')
        ax.set_ylabel("Cumulative Net Cost, bn $", fontsize=14)
    else:
        cum_utility = np.cumsum(outputs["history_total_utility"], axis=1) * 1e-9
        plot_line_with_ci(ax, cum_utility, '#009E73', '-', 'Cumulative Utility')
        ax.set_ylabel("Cumulative Utility, bn $", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.2, 0.2, 0.2, 0.2])

    # --- X axis formatting, in calendar years
    window = format_year_axis(axs[2], start_step, total_steps, calibration_end_step)

    fig.suptitle(f"Multi-Seed Single Run Dashboard ({window})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/multi_seed_dashboard.png", dpi=dpi)
    print(f"Saved to {save_path}/multi_seed_dashboard.png")


def plot_multi_seed_dashboard_extra(base_params, fileName, outputs, dpi=200):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    total_steps = np.asarray(outputs["history_prop_EV"]).shape[1]
    start_step, calibration_end_step = get_plot_window(base_params, total_steps)
    time_steps = np.arange(start_step, total_steps)

    def plot_line_with_ci(ax, data, color, linestyle='-', label=None):
        return plot_series_with_ci(ax, data, time_steps, start_step, color, linestyle, label, safe=True)

    print("\n--- Plotting Extra Dashboard ---")

    # --- Driving vs Production Emissions
    ax = axs[0, 0]
    has_data = False

    if "history_driving_emissions" in outputs:
        print(f"[ok] Plotting history_driving_emissions (shape: {get_shape_safe(outputs['history_driving_emissions'])})")
        if plot_line_with_ci(ax, outputs["history_driving_emissions"] * 1e-9, '#D55E00', '-', 'Driving'):
            has_data = True
    else:
        print("[--] history_driving_emissions not found")

    if "history_production_emissions" in outputs:
        print(f"[ok] Plotting history_production_emissions (shape: {get_shape_safe(outputs['history_production_emissions'])})")
        if plot_line_with_ci(ax, outputs["history_production_emissions"] * 1e-9, '#E69F00', '--', 'Production'):
            has_data = True
    else:
        print("[--] history_production_emissions not found")

    ax.set_ylabel("Flow Emissions, MTCO2", fontsize=14)
    if has_data:
        ax.legend(loc='upper right', fontsize='small')
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Cumulative Driving vs Production Emissions
    ax = axs[0, 1]
    has_data = False

    if "history_driving_emissions" in outputs:
        cum_driving = np.cumsum(outputs["history_driving_emissions"], axis=1) * 1e-9
        print(f"[ok] Plotting cumulative driving emissions (shape after cumsum: {cum_driving.shape})")
        if plot_line_with_ci(ax, cum_driving, '#D55E00', '-', 'Driving'):
            has_data = True
    else:
        print("[--] Cannot plot cumulative driving emissions - data not found")

    if "history_production_emissions" in outputs:
        cum_prod = np.cumsum(outputs["history_production_emissions"], axis=1) * 1e-9
        print(f"[ok] Plotting cumulative production emissions (shape after cumsum: {cum_prod.shape})")
        if plot_line_with_ci(ax, cum_prod, '#E69F00', '--', 'Production'):
            has_data = True
    else:
        print("[--] Cannot plot cumulative production emissions - data not found")

    ax.set_ylabel("Cumulative Emissions, MTCO2", fontsize=14)
    if has_data:
        ax.legend(loc='upper left', fontsize='small')
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Market Concentration (HHI)
    ax = axs[1, 0]
    if "history_market_concentration" in outputs:
        print(f"[ok] Plotting history_market_concentration (shape: {get_shape_safe(outputs['history_market_concentration'])})")
        plot_line_with_ci(ax, outputs["history_market_concentration"], '#0072B2', '-', 'HHI')
    else:
        print("[--] history_market_concentration not found")
        ax.text(0.5, 0.5, 'HHI Data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

    ax.set_ylabel("Market Concentration (HHI)", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Total Profit
    ax = axs[1, 1]
    has_data = False

    if "history_total_profit" in outputs:
        print(f"[ok] Plotting history_total_profit (shape: {get_shape_safe(outputs['history_total_profit'])})")
        if plot_line_with_ci(ax, outputs["history_total_profit"] * 1e-9, '#0072B2', '-', 'Total Profit'):
            has_data = True
    else:
        print("[--] history_total_profit not found")

    ax.set_ylabel("Total Profit, bn $", fontsize=14)

    # Margins are fractions, so they get their own axis to be visible alongside
    # profit in billions
    ax_margin = ax.twinx()
    for key, colour, linestyle, label in (
        ("history_mean_profit_margins_ICE", '#D55E00', '--', 'ICE Margin'),
        ("history_mean_profit_margins_EV", '#009E73', ':', 'EV Margin'),
    ):
        if key not in outputs:
            print(f"[--] {key} not found")
            continue
        print(f"[ok] Plotting {key} (shape: {get_shape_safe(outputs[key])})")
        if plot_line_with_ci(ax_margin, outputs[key], colour, linestyle, label):
            has_data = True
    ax_margin.set_ylabel("Mean Profit Margin", fontsize=12)

    if has_data:
        handles = ax.get_legend_handles_labels()[0] + ax_margin.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc='upper left', fontsize='small')
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Mean Car Age
    ax = axs[2, 0]
    if "history_mean_car_age" in outputs:
        print(f"[ok] Plotting history_mean_car_age (shape: {get_shape_safe(outputs['history_mean_car_age'])})")
        plot_line_with_ci(ax, outputs["history_mean_car_age"], '#CC79A7', '-', 'Mean Car Age')
    else:
        print("[--] history_mean_car_age not found")
        ax.text(0.5, 0.5, 'Mean Car Age Data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

    ax.set_ylabel("Mean Car Age, months", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- EV Price Spread
    ax = axs[2, 1]
    has_data = False

    for key, linestyle, label in (
        ("history_upper_percentile_price_ICE_EV", '-', 'Upper Percentile EV'),
        ("history_lower_percentile_price_ICE_EV", '--', 'Lower Percentile EV'),
    ):
        if key not in outputs:
            print(f"[--] {key} not found")
            continue

        arr = np.asarray(outputs[key], dtype=float)
        print(f"[ok] Plotting {key} (shape: {arr.shape})")
        if arr.ndim == 4:
            # (n_seeds, n_steps, new/used, ICE/EV) - take new EV, matching the
            # mean price panel on the main dashboard
            series = arr[:, :, 0, 1]
            print(f"  -> Using 4D indexing [:, :, 0, 1] (new EV), resulting shape: {series.shape}")
        elif arr.ndim == 3:
            # (n_seeds, n_steps, ICE/EV)
            series = arr[:, :, 1]
            print(f"  -> Using 3D indexing [:, :, 1], resulting shape: {series.shape}")
        else:
            series = arr
            print(f"  -> Using data as is, shape: {series.shape}")

        if plot_line_with_ci(ax, series * 1e-3, '#009E73', linestyle, label):
            has_data = True

    ax.set_ylabel("New EV Price Percentiles, k$", fontsize=14)
    if has_data:
        ax.legend(loc='upper right', fontsize='small')
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- X axis formatting, in calendar years
    window = format_year_axis(axs[2], start_step, total_steps, calibration_end_step)

    fig.suptitle(f"Additional Metrics ({window})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/multi_seed_dashboard_extra.png", dpi=dpi)
    print(f"\nSaved to {save_path}/multi_seed_dashboard_extra.png")


####################################################################################
# Figures 4 and 5: the cars on sale, against real-world vehicles
####################################################################################

def extract_car_attributes_from_objects(cars_list, base_params):
    """
    Extract price, range, efficiency, and type from car objects.
    Removes duplicates by tracking unique_id.

    Units: the model does ALL fuel accounting in kWh, not litres -- gas prices
    and gasoline emissions are converted to a per-kWh basis in
    calibration_data_inputs.py at 33.41 kWh per US gallon (AFDC gallon
    equivalent). So ICE efficiency is km/kWh and fuel_tank is kWh, exactly like
    the EV's km/kWh and battery kWh. Range is efficiency * capacity either way.
    """
    fuel_tank_kWh = base_params["parameters_ICE"]["fuel_tank"]

    seen_ids = set()
    records = []

    for car in cars_list:
        uid = getattr(car, "unique_id", id(car))
        if uid in seen_ids:
            continue
        seen_ids.add(uid)

        attrs = car.attributes_fitness
        price = getattr(car, "price", car.ProdCost_t * 1.0)

        if car.transportType == 3:  # EV
            efficiency = attrs[1]  # km/kWh
            battery_kwh = attrs[3]  # kWh
            range_km = efficiency * battery_kwh
            vtype = "EV"
        else:  # ICE
            efficiency = attrs[1]  # km/kWh (gasoline on a kWh basis, not km/L)
            range_km = efficiency * fuel_tank_kWh
            vtype = "ICE"

        records.append({
            "price": price,
            "range_km": range_km,
            "efficiency": efficiency,
            "vtype": vtype
        })

    return records


def _pool_cars(outputs, base_params):
    """All unique cars on sale at the end of the run, pooled over seeds."""
    cars_on_sale_per_seed = outputs.get("cars_on_sale", [])
    if not cars_on_sale_per_seed:
        return None, None

    all_cars = []
    for seed_idx, seed_cars in enumerate(cars_on_sale_per_seed):
        car_attrs = extract_car_attributes_from_objects(seed_cars, base_params)
        all_cars.extend(car_attrs)
        print(f"  Seed {seed_idx}: {len(seed_cars)} raw -> {len(car_attrs)} unique cars")

    return all_cars, cars_on_sale_per_seed


def plot_multi_seed_2d_scatter(fileName, outputs, base_params, dpi=200):
    """
    Simple 2D scatter plot: Driving Range (km) vs Price (USD)
    Flattens all cars from all seeds and plots them together.
    """

    MILES_PER_KM, KM_PER_MILE, MPGE_TO_KM_KWH, MPG_TO_KM_L, REAL_WORLD_VEHICLES = info_real_cars()

    all_cars, cars_on_sale_per_seed = _pool_cars(outputs, base_params)
    if not all_cars:
        print("No cars_on_sale found in outputs.")
        return None

    print(f"Processing {len(cars_on_sale_per_seed)} seeds...")

    # Split by type
    sim_ev = [c for c in all_cars if c["vtype"] == "EV"]
    sim_ice = [c for c in all_cars if c["vtype"] == "ICE"]

    # Real-world vehicles
    real_ev = [v for v in REAL_WORLD_VEHICLES if v["type"] == "EV"]
    real_ice = [v for v in REAL_WORLD_VEHICLES if v["type"] in ["ICE", "PHEV"]]

    print(f"\nTotal unique cars across all seeds: {len(all_cars)}")
    print(f"  EVs: {len(sim_ev)}, ICEs: {len(sim_ice)}")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot simulated EVs (green circles)
    if sim_ev:
        ev_ranges = np.array([c["range_km"] for c in sim_ev])
        ev_prices = np.array([c["price"] for c in sim_ev])
        ax.scatter(ev_ranges, ev_prices, marker="o", s=30, alpha=0.4,
                  c=EV_COLOR, edgecolors="darkgreen", linewidths=0.3,
                  label=f"Simulated EV (n={len(sim_ev)})", zorder=2)

    # Plot simulated ICEs (blue squares)
    if sim_ice:
        ice_ranges = np.array([c["range_km"] for c in sim_ice])
        ice_prices = np.array([c["price"] for c in sim_ice])
        ax.scatter(ice_ranges, ice_prices, marker="s", s=30, alpha=0.4,
                  c=ICE_COLOR, edgecolors="darkblue", linewidths=0.3,
                  label=f"Simulated ICE (n={len(sim_ice)})", zorder=2)

    # Plot real-world EVs (green diamonds)
    for v in real_ev:
        ax.scatter(v["range_km"], v["price_usd"], marker="D", s=200,
                  c=EV_COLOR, edgecolors="darkgreen", linewidths=1.5, zorder=5)
        ax.annotate(v["label"], xy=(v["range_km"], v["price_usd"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, fontweight="bold", color=EV_COLOR,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=EV_COLOR, alpha=0.8))

    # Plot real-world ICE (blue triangles)
    for v in real_ice:
        ax.scatter(v["range_km"], v["price_usd"], marker="^", s=200,
                  c=ICE_COLOR, edgecolors="darkblue", linewidths=1.5, zorder=5)
        ax.annotate(v["label"], xy=(v["range_km"], v["price_usd"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, fontweight="bold", color=ICE_COLOR,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=ICE_COLOR, alpha=0.8))

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=EV_COLOR,
               markeredgecolor="darkgreen", markersize=8, label=f"Simulated EV ({len(sim_ev)})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ICE_COLOR,
               markeredgecolor="darkblue", markersize=8, label=f"Simulated ICE ({len(sim_ice)})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=EV_COLOR,
               markeredgecolor="darkgreen", markersize=8, label="Real-world EV"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=ICE_COLOR,
               markeredgecolor="darkblue", markersize=8, label="Real-world ICE"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_xlabel("Driving Range (km)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USD)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")

    # Add some padding
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.05)

    plt.tight_layout()

    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/multi_seed_2d_scatter.png", dpi=dpi, bbox_inches="tight")
    print(f"\nSaved to {save_path}/multi_seed_2d_scatter.png")

    return fig, ax


def _density_levels(kde, sample_xy, mass_fractions):
    """
    Convert "contour enclosing X% of the simulated cars" into KDE density levels.

    Evaluates the KDE at the sample points themselves and takes the (1-X)
    quantile of those densities, so the level enclosing e.g. 50% of the mass is
    the density value that 50% of the cars sit above.
    """
    dens = kde(sample_xy)
    # Largest mass fraction -> lowest density level, so sorting the fractions
    # descending gives strictly increasing levels to pair the labels against
    pairs = [(float(np.quantile(dens, 1.0 - f)), f)
             for f in sorted(mass_fractions, reverse=True)]

    # contour/contourf require strictly increasing levels
    kept = []
    for level, frac in pairs:
        if not kept or level > kept[-1][0]:
            kept.append((level, frac))
    return kept


def plot_multi_seed_2d_contour(fileName, outputs, base_params, dpi=200,
                               mass_fractions=(0.95, 0.8, 0.5), grid_n=200):
    """
    Real-world vehicles on top of a 2D density contour map of the simulated cars.

    The simulated cloud (all unique cars on sale, pooled across seeds) is turned
    into a Gaussian KDE over (Driving Range, Price) and drawn as nested contours
    enclosing 95 / 80 / 50 % of the simulated cars. EV and ICE get their own
    density, so the two populations can be compared against their real-world
    counterparts separately.

    Note this is a SNAPSHOT of the final time step, not a time series -- see
    generate_multi_seed in package/resources/run.py.
    """

    MILES_PER_KM, KM_PER_MILE, MPGE_TO_KM_KWH, MPG_TO_KM_L, REAL_WORLD_VEHICLES = info_real_cars()

    all_cars, cars_on_sale_per_seed = _pool_cars(outputs, base_params)
    if not all_cars:
        print("No cars_on_sale found in outputs.")
        return None

    sim_ev = [c for c in all_cars if c["vtype"] == "EV"]
    sim_ice = [c for c in all_cars if c["vtype"] == "ICE"]

    real_ev = [v for v in REAL_WORLD_VEHICLES if v["type"] == "EV"]
    real_ice = [v for v in REAL_WORLD_VEHICLES if v["type"] in ["ICE", "PHEV"]]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Grid spans simulated and real data so nothing is clipped out of frame
    all_x = [c["range_km"] for c in all_cars] + [v["range_km"] for v in REAL_WORLD_VEHICLES]
    all_y = [c["price"] for c in all_cars] + [v["price_usd"] for v in REAL_WORLD_VEHICLES]
    x_pad = 0.08 * (max(all_x) - min(all_x))
    y_pad = 0.08 * (max(all_y) - min(all_y))
    xx, yy = np.meshgrid(
        np.linspace(min(all_x) - x_pad, max(all_x) + x_pad, grid_n),
        np.linspace(min(all_y) - y_pad, max(all_y) + y_pad, grid_n),
    )
    grid_xy = np.vstack([xx.ravel(), yy.ravel()])

    for sim, colour, cmap, name in (
        (sim_ice, ICE_COLOR, "Blues", "ICE"),
        (sim_ev, EV_COLOR, "Greens", "EV"),
    ):
        if len(sim) < 5:
            print(f"Skipping {name} density: only {len(sim)} cars.")
            continue

        sample_xy = np.vstack([
            np.array([c["range_km"] for c in sim]),
            np.array([c["price"] for c in sim]),
        ])
        try:
            kde = gaussian_kde(sample_xy)
        except np.linalg.LinAlgError:
            print(f"Skipping {name} density: degenerate covariance (no spread in the data).")
            continue

        zz = kde(grid_xy).reshape(xx.shape)
        level_pairs = _density_levels(kde, sample_xy, mass_fractions)
        levels = [lv for lv, _ in level_pairs]
        if len(levels) < 2 or zz.max() <= levels[-1]:
            print(f"Skipping {name} density: could not form distinct contour levels.")
            continue

        # Filled bands give the "where the mass is" read, lines give the boundary
        ax.contourf(xx, yy, zz, levels=levels + [zz.max()], cmap=cmap,
                    alpha=0.35, zorder=1)
        cs = ax.contour(xx, yy, zz, levels=levels, colors=colour,
                        linewidths=1.5, zorder=2)
        ax.clabel(cs, inline=True, fontsize=8,
                  fmt={lv: f"{int(round(f * 100))}%" for lv, f in level_pairs})

        print(f"{name}: {len(sim)} cars, contour levels {levels}")

    # Real-world vehicles on top, with a white ring so they read over the fills.
    # Labels are fanned out vertically within each type, cheapest first, so the
    # tightly-clustered real ICE models don't overprint each other.
    for group, marker, size, colour in (
        (real_ice, "^", 180, ICE_COLOR),
        (real_ev, "D", 160, EV_COLOR),
    ):
        for rank, v in enumerate(sorted(group, key=lambda c: c["price_usd"])):
            ax.scatter(v["range_km"], v["price_usd"], marker=marker, s=size,
                       c=colour, edgecolors="white", linewidths=1.8, zorder=5)
            ax.annotate(
                v["label"], xy=(v["range_km"], v["price_usd"]),
                xytext=(10, 8 + 13 * (rank % len(group))), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#222222",
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.7,
                                shrinkA=0, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=colour, alpha=0.85), zorder=6,
            )

    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=EV_COLOR,
               markeredgecolor="white", markersize=9, label="Real-world EV"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=ICE_COLOR,
               markeredgecolor="white", markersize=10, label="Real-world ICE"),
        Line2D([0], [0], color=EV_COLOR, linewidth=1.5,
               label=f"Simulated EV density (n={len(sim_ev)})"),
        Line2D([0], [0], color=ICE_COLOR, linewidth=1.5,
               label=f"Simulated ICE density (n={len(sim_ice)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_xlabel("Driving Range (km)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USD)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Simulated cars on sale (contours enclose 50 / 80 / 95% of cars, pooled over "
        f"{len(cars_on_sale_per_seed)} seeds) vs real-world vehicles",
        fontsize=11
    )
    ax.grid(alpha=0.25, linestyle="--", zorder=0)

    plt.tight_layout()

    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/multi_seed_2d_contour.png", dpi=dpi, bbox_inches="tight")
    print(f"Saved to {save_path}/multi_seed_2d_contour.png")

    return fig, ax


####################################################################################

def main(fileName, dpi=200):
    base_params = load_object(fileName + "/Data", "base_params")
    outputs = load_object(fileName + "/Data", "outputs")

    print_data_shapes(outputs, base_params)

    # Fit against the Californian data
    plot_calibration_fit(base_params, fileName, outputs, dpi=dpi)

    # The four target variables against their observed ranges. Skipped on runs
    # saved before record_calibration_targets() existed.
    if "history_new_car_price_quantiles" in outputs:
        plot_calibration_targets(base_params, fileName, outputs, dpi=dpi)
    else:
        print("[!] No calibration-target series in this run, skipping calibration_targets.")

    # Time-series dashboards
    plot_multi_seed_dashboard(base_params, fileName, outputs, dpi=dpi)
    plot_multi_seed_dashboard_extra(base_params, fileName, outputs, dpi=dpi)

    # Cars on sale at the end of the run, against real vehicles
    if "cars_on_sale" in outputs:
        print(f"\nNumber of seeds: {len(outputs['cars_on_sale'])}")
        plot_multi_seed_2d_scatter(fileName, outputs, base_params, dpi=dpi)
        plot_multi_seed_2d_contour(fileName, outputs, base_params, dpi=dpi)
    else:
        print("\n[!] No 'cars_on_sale' found in outputs.")

    plt.show()


if __name__ == "__main__":
    main(fileName="results/calibration_gen_09_56_46__07_08_2026")
