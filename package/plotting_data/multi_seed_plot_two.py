import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
from matplotlib.lines import Line2D
import os

def add_vertical_lines(ax, base_params, color='black', linestyle='--', annotation_height_prop=[0.2, 0.2, 0.2, 0.2]):
    burn_in = base_params["duration_burn_in"]
    no_carbon_price = base_params["duration_calibration"]
    ev_production_start_time = base_params["ev_production_start_time"]

    y_min, y_max = ax.get_ylim()
    annotation_height_0 = y_min + annotation_height_prop[0] * (y_max - y_min)
    annotation_height_1 = y_min + annotation_height_prop[1] * (y_max - y_min)
    annotation_height_2 = y_min + annotation_height_prop[2] * (y_max - y_min)
    annotation_height_3 = y_min + annotation_height_prop[3] * (y_max - y_min)

    # EV Sale Start
    ax.axvline(ev_production_start_time, color="black", linestyle=':')
    ax.annotate("EV Sale Start", xy=(ev_production_start_time, annotation_height_0),
                rotation=90, verticalalignment='center', horizontalalignment='right',
                fontsize=8, color='black')

    # EV Adoption Subsidy Start
    if base_params["EV_rebate_state"]:
        rebate_start_time = base_params["parameters_rebate_calibration"]["start_time"]
        ax.axvline(rebate_start_time, color="black", linestyle='-.')
        ax.annotate("EV Adoption Subsidy Start", xy=(rebate_start_time, annotation_height_1),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

    # Policy Start
    if base_params["duration_future"] > 0:
        policy_start_time = no_carbon_price
        ax.axvline(policy_start_time, color="black", linestyle='--')
        ax.annotate("Policy Start", xy=(policy_start_time, annotation_height_2),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

        # Policy End
        if base_params["duration_future"] >= 144:
            policy_end_time = no_carbon_price + 144
            ax.axvline(policy_end_time, color="black", linestyle='--')
            ax.annotate("Policy End", xy=(policy_end_time, annotation_height_3),
                        rotation=90, verticalalignment='center', horizontalalignment='right',
                        fontsize=8, color='black')


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


def print_data_shapes(outputs, label=""):
    """Print shapes of all relevant data arrays for debugging with robust error handling"""
    print(f"\n{'='*60}")
    print(f"DATA SHAPES FOR DEBUGGING {label}")
    print(f"{'='*60}")
    
    # Main dashboard data
    print("\n--- Main Dashboard Data ---")
    
    data_keys = [
        "history_prop_EV",
        "history_past_new_bought_vehicles_prop_ev",
        "history_mean_price_ICE_EV_arr",
        "history_total_emissions",
        "history_total_utility",
        "history_policy_net_cost"
    ]
    
    for key in data_keys:
        try:
            if key in outputs:
                shape = get_shape_safe(outputs[key])
                print(f"✓ {key}: {shape}")
            else:
                print(f"✗ {key}: NOT FOUND")
        except Exception as e:
            print(f"⚠ {key}: Error accessing - {str(e)}")
    
    # Extra dashboard data
    print("\n--- Extra Dashboard Data ---")
    
    extra_keys = [
        "history_driving_emissions",
        "history_production_emissions",
        "history_market_concentration",
        "history_total_profit",
        "history_mean_profit_margins_ICE",
        "history_mean_profit_margins_EV",
        "history_mean_car_age",
        "history_upper_percentile_price_ICE_EV_arr",
        "history_lower_percentile_price_ICE_EV_arr"
    ]
    
    for key in extra_keys:
        try:
            if key in outputs:
                shape = get_shape_safe(outputs[key])
                if hasattr(outputs[key], 'shape') and len(outputs[key].shape) >= 3:
                    print(f"✓ {key}: {shape}")
                    if "percentile" in key:
                        print(f"  → Likely dimensions: (n_seeds, n_steps, n_percentiles or n_types)")
                else:
                    print(f"✓ {key}: {shape}")
            else:
                print(f"✗ {key}: NOT FOUND")
        except Exception as e:
            print(f"⚠ {key}: Error accessing - {str(e)}")
    
    # Also check for any other potentially useful data
    print("\n--- Additional Available Keys ---")
    all_keys = list(outputs.keys())
    standard_keys = set(data_keys + extra_keys)
    other_keys = [k for k in all_keys if k not in standard_keys]
    
    if other_keys:
        for key in other_keys[:10]:  # Show first 10 additional keys
            try:
                shape = get_shape_safe(outputs[key])
                print(f"  • {key}: {shape}")
            except:
                print(f"  • {key}: (accessible but shape unknown)")
        if len(other_keys) > 10:
            print(f"  ... and {len(other_keys) - 10} more")
    else:
        print("  No additional keys found")
    
    # Print simulation timeline info
    print("\n" + "="*60)
    print("--- Simulation Timeline ---")
    try:
        if "history_prop_EV" in outputs:
            total_steps = outputs["history_prop_EV"].shape[1]
            print(f"Total steps in simulation: {total_steps}")
            calibration_end_step = 456
            print(f"Steps in future period (from step {calibration_end_step}): {total_steps - calibration_end_step}")
            print(f"Future period years: {(total_steps - calibration_end_step) / 12:.1f} years")
            print(f"Start year of future period: 2024")
        else:
            print("Cannot determine timeline - history_prop_EV not found")
    except Exception as e:
        print(f"Error determining timeline: {str(e)}")
    
    print("="*60 + "\n")


def plot_comparison_dashboard(base_params1, base_params2, fileName1, fileName2, outputs1, outputs2, labels, dpi=200):
    """Plot comparison dashboard for two multi-seed runs"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    
    # Define start index for future period (after calibration)
    calibration_end_step = 456
    total_steps1 = outputs1["history_prop_EV"].shape[1]
    total_steps2 = outputs2["history_prop_EV"].shape[1]
    total_steps = min(total_steps1, total_steps2)  # Use minimum steps for comparison
    time_steps = np.arange(calibration_end_step, total_steps)
    
    start_year = 2024

    def plot_line_with_ci(ax, data, color, linestyle='-', label=None, alpha=0.2):
        # Slice data from future period start
        sliced_data = data[:, calibration_end_step:total_steps]
        mean = np.nanmean(sliced_data, axis=0)
        ci = sem(sliced_data, axis=0, nan_policy='omit') * t.ppf(0.975, df=sliced_data.shape[0] - 1)
        ax.plot(time_steps, mean, color=color, linestyle=linestyle, label=label, linewidth=2)
        ax.fill_between(time_steps, mean - ci, mean + ci, color=color, alpha=alpha)

    # Color schemes for two runs
    colors = ['#0072B2', '#D55E00']  # Blue and Orange
    
    # --- EV Adoption and Sales Share
    ax = axs[0, 0]
    plot_line_with_ci(ax, outputs1["history_prop_EV"], colors[0], '-', f'{labels[0]} - EV Adoption')
    plot_line_with_ci(ax, outputs1["history_past_new_bought_vehicles_prop_ev"], colors[0], '--', f'{labels[0]} - EV Sales')
    plot_line_with_ci(ax, outputs2["history_prop_EV"], colors[1], '-', f'{labels[1]} - EV Adoption')
    plot_line_with_ci(ax, outputs2["history_past_new_bought_vehicles_prop_ev"], colors[1], '--', f'{labels[1]} - EV Sales')
    ax.set_ylabel("EV Share", fontsize=14)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.6, 0.2, 0.2, 0.2])
    ax.legend(loc='lower right', fontsize='small', ncol=2)

    # --- EV Price (New and Used)
    ax = axs[0, 1]
    plot_line_with_ci(ax, outputs1["history_mean_price_ICE_EV_arr"][:, :, 0, 1], colors[0], '-', f'{labels[0]} - New EV')
    plot_line_with_ci(ax, outputs1["history_mean_price_ICE_EV_arr"][:, :, 1, 1], colors[0], '--', f'{labels[0]} - Used EV')
    plot_line_with_ci(ax, outputs2["history_mean_price_ICE_EV_arr"][:, :, 0, 1], colors[1], '-', f'{labels[1]} - New EV')
    plot_line_with_ci(ax, outputs2["history_mean_price_ICE_EV_arr"][:, :, 1, 1], colors[1], '--', f'{labels[1]} - Used EV')
    ax.set_ylabel("EV Sale Price, $", fontsize=14)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.9, 0.2, 0.2, 0.2])
    ax.legend(loc='upper right', fontsize='small', ncol=2)

    # --- Flow Emissions
    ax = axs[1, 0]
    plot_line_with_ci(ax, outputs1["history_total_emissions"] * 1e-9, colors[0], '-', f'{labels[0]} - Flow Emissions')
    plot_line_with_ci(ax, outputs2["history_total_emissions"] * 1e-9, colors[1], '-', f'{labels[1]} - Flow Emissions')
    ax.set_ylabel("Flow Emissions, MTCO2", fontsize=14)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])
    ax.legend(loc='upper right', fontsize='small')

    # --- Cumulative Emissions
    ax = axs[1, 1]
    cum_emissions1 = np.cumsum(outputs1["history_total_emissions"], axis=1) * 1e-9
    cum_emissions2 = np.cumsum(outputs2["history_total_emissions"], axis=1) * 1e-9
    plot_line_with_ci(ax, cum_emissions1, colors[0], '-', f'{labels[0]} - Cumulative Emissions')
    plot_line_with_ci(ax, cum_emissions2, colors[1], '-', f'{labels[1]} - Cumulative Emissions')
    ax.set_ylabel("Cumulative Emissions, MTCO2", fontsize=14)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.2, 0.2, 0.2, 0.2])
    ax.legend(loc='upper left', fontsize='small')

    # --- Flow Utility
    ax = axs[2, 0]
    plot_line_with_ci(ax, outputs1["history_total_utility"] * 1e-9, colors[0], '-', f'{labels[0]} - Flow Utility')
    plot_line_with_ci(ax, outputs2["history_total_utility"] * 1e-9, colors[1], '-', f'{labels[1]} - Flow Utility')
    ax.set_ylabel("Flow Utility, bn $", fontsize=14)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.2, 0.2, 0.2, 0.2])
    ax.legend(loc='upper right', fontsize='small')

    # --- Cumulative Net Cost or Utility
    ax = axs[2, 1]
    if "history_policy_net_cost" in outputs1 and "history_policy_net_cost" in outputs2:
        cum_cost1 = np.cumsum(outputs1["history_policy_net_cost"], axis=1) * 1e-9
        cum_cost2 = np.cumsum(outputs2["history_policy_net_cost"], axis=1) * 1e-9
        plot_line_with_ci(ax, cum_cost1, colors[0], '-', f'{labels[0]} - Cumulative Net Cost')
        plot_line_with_ci(ax, cum_cost2, colors[1], '-', f'{labels[1]} - Cumulative Net Cost')
        ax.set_ylabel("Cumulative Net Cost, bn $", fontsize=14)
    else:
        cum_utility1 = np.cumsum(outputs1["history_total_utility"], axis=1) * 1e-9
        cum_utility2 = np.cumsum(outputs2["history_total_utility"], axis=1) * 1e-9
        plot_line_with_ci(ax, cum_utility1, colors[0], '-', f'{labels[0]} - Cumulative Utility')
        plot_line_with_ci(ax, cum_utility2, colors[1], '-', f'{labels[1]} - Cumulative Utility')
        ax.set_ylabel("Cumulative Utility, bn $", fontsize=14)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.2, 0.2, 0.2, 0.2])
    ax.legend(loc='upper left', fontsize='small')

    # --- X axis formatting (now showing years from 2024 onwards)
    future_duration_years = (total_steps - calibration_end_step) / 12
    all_tick_years = np.arange(start_year, start_year + future_duration_years + 5, 5)
    all_tick_positions = calibration_end_step + (all_tick_years - start_year) * 12
    
    # Filter to only show ticks within the visible range
    visible_mask = all_tick_positions <= total_steps
    tick_positions = all_tick_positions[visible_mask]
    tick_labels = [str(y) for y in all_tick_years[visible_mask]]

    for ax in axs[2]:
        ax.set_xlim(calibration_end_step, total_steps)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Year", fontsize=14)

    # Create output directory if it doesn't exist
    output_dir = "results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.suptitle(f"Multi-Seed Comparison: {labels[0]} vs {labels[1]} (Future Period from 2024)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{output_dir}/comparison_dashboard.png", dpi=dpi)
    print(f"Saved to {output_dir}/comparison_dashboard.png")


def plot_comparison_extra(base_params1, base_params2, fileName1, fileName2, outputs1, outputs2, labels, dpi=200):
    """Plot comparison extra dashboard for two multi-seed runs"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    
    # Define start index for future period (after calibration)
    calibration_end_step = 456
    total_steps1 = outputs1["history_prop_EV"].shape[1]
    total_steps2 = outputs2["history_prop_EV"].shape[1]
    total_steps = min(total_steps1, total_steps2)
    time_steps = np.arange(calibration_end_step, total_steps)
    
    start_year = 2024
    colors = ['#0072B2', '#D55E00']

    def plot_line_with_ci(ax, data, color, linestyle='-', label=None, alpha=0.2):
        try:
            sliced_data = data[:, calibration_end_step:total_steps]
            mean = np.nanmean(sliced_data, axis=0)
            ci = sem(sliced_data, axis=0, nan_policy='omit') * t.ppf(0.975, df=sliced_data.shape[0] - 1)
            ax.plot(time_steps, mean, color=color, linestyle=linestyle, label=label, linewidth=2)
            ax.fill_between(time_steps, mean - ci, mean + ci, color=color, alpha=alpha)
            return True
        except Exception as e:
            print(f"  ⚠ Failed to plot {label}: {str(e)}")
            return False

    print("\n--- Plotting Extra Comparison Dashboard ---")
    
    # --- Driving vs Production Emissions
    ax = axs[0, 0]
    has_data = False
    
    if "history_driving_emissions" in outputs1:
        if plot_line_with_ci(ax, outputs1["history_driving_emissions"] * 1e-9, colors[0], '-', f'{labels[0]} - Driving'):
            has_data = True
    if "history_driving_emissions" in outputs2:
        if plot_line_with_ci(ax, outputs2["history_driving_emissions"] * 1e-9, colors[1], '-', f'{labels[1]} - Driving'):
            has_data = True
    if "history_production_emissions" in outputs1:
        if plot_line_with_ci(ax, outputs1["history_production_emissions"] * 1e-9, colors[0], '--', f'{labels[0]} - Production'):
            has_data = True
    if "history_production_emissions" in outputs2:
        if plot_line_with_ci(ax, outputs2["history_production_emissions"] * 1e-9, colors[1], '--', f'{labels[1]} - Production'):
            has_data = True
    
    ax.set_ylabel("Flow Emissions, MTCO2", fontsize=14)
    if has_data:
        ax.legend(loc='upper right', fontsize='small', ncol=2)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Cumulative Driving vs Production Emissions
    ax = axs[0, 1]
    has_data = False
    
    if "history_driving_emissions" in outputs1:
        cum_driving1 = np.cumsum(outputs1["history_driving_emissions"], axis=1) * 1e-9
        if plot_line_with_ci(ax, cum_driving1, colors[0], '-', f'{labels[0]} - Driving'):
            has_data = True
    if "history_driving_emissions" in outputs2:
        cum_driving2 = np.cumsum(outputs2["history_driving_emissions"], axis=1) * 1e-9
        if plot_line_with_ci(ax, cum_driving2, colors[1], '-', f'{labels[1]} - Driving'):
            has_data = True
    if "history_production_emissions" in outputs1:
        cum_prod1 = np.cumsum(outputs1["history_production_emissions"], axis=1) * 1e-9
        if plot_line_with_ci(ax, cum_prod1, colors[0], '--', f'{labels[0]} - Production'):
            has_data = True
    if "history_production_emissions" in outputs2:
        cum_prod2 = np.cumsum(outputs2["history_production_emissions"], axis=1) * 1e-9
        if plot_line_with_ci(ax, cum_prod2, colors[1], '--', f'{labels[1]} - Production'):
            has_data = True
    
    ax.set_ylabel("Cumulative Emissions, MTCO2", fontsize=14)
    if has_data:
        ax.legend(loc='upper left', fontsize='small', ncol=2)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Market Concentration (HHI)
    ax = axs[1, 0]
    if "history_market_concentration" in outputs1:
        plot_line_with_ci(ax, outputs1["history_market_concentration"], colors[0], '-', f'{labels[0]} - HHI')
    if "history_market_concentration" in outputs2:
        plot_line_with_ci(ax, outputs2["history_market_concentration"], colors[1], '-', f'{labels[1]} - HHI')
    ax.set_ylabel("Market Concentration (HHI)", fontsize=14)
    ax.legend(loc='upper right', fontsize='small')
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Total Profit
    ax = axs[1, 1]
    has_data = False
    
    if "history_total_profit" in outputs1:
        if plot_line_with_ci(ax, outputs1["history_total_profit"] * 1e-9, colors[0], '-', f'{labels[0]} - Total Profit'):
            has_data = True
    if "history_total_profit" in outputs2:
        if plot_line_with_ci(ax, outputs2["history_total_profit"] * 1e-9, colors[1], '-', f'{labels[1]} - Total Profit'):
            has_data = True
    
    ax.set_ylabel("Total Profit, bn $", fontsize=14)
    if has_data:
        ax.legend(loc='upper left', fontsize='small')
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Mean Car Age
    ax = axs[2, 0]
    if "history_mean_car_age" in outputs1:
        plot_line_with_ci(ax, outputs1["history_mean_car_age"], colors[0], '-', f'{labels[0]} - Mean Car Age')
    if "history_mean_car_age" in outputs2:
        plot_line_with_ci(ax, outputs2["history_mean_car_age"], colors[1], '-', f'{labels[1]} - Mean Car Age')
    ax.set_ylabel("Mean Car Age, months", fontsize=14)
    ax.legend(loc='upper right', fontsize='small')
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- EV Price Spread
    ax = axs[2, 1]
    has_data = False
    
    # Try to plot upper and lower percentiles for both runs
    for i, (outputs, label, color) in enumerate([(outputs1, labels[0], colors[0]), (outputs2, labels[1], colors[1])]):
        if "history_upper_percentile_price_ICE_EV_arr" in outputs:
            try:
                if len(outputs["history_upper_percentile_price_ICE_EV_arr"].shape) == 3:
                    upper_data = outputs["history_upper_percentile_price_ICE_EV_arr"][:, :, 1] * 1e-3
                else:
                    upper_data = outputs["history_upper_percentile_price_ICE_EV_arr"] * 1e-3
                if plot_line_with_ci(ax, upper_data, color, '-', f'{label} - Upper Percentile'):
                    has_data = True
            except Exception as e:
                print(f"  ⚠ Error processing upper percentile for {label}: {str(e)}")
        
        if "history_lower_percentile_price_ICE_EV_arr" in outputs:
            try:
                if len(outputs["history_lower_percentile_price_ICE_EV_arr"].shape) == 3:
                    lower_data = outputs["history_lower_percentile_price_ICE_EV_arr"][:, :, 1] * 1e-3
                else:
                    lower_data = outputs["history_lower_percentile_price_ICE_EV_arr"] * 1e-3
                if plot_line_with_ci(ax, lower_data, color, '--', f'{label} - Lower Percentile'):
                    has_data = True
            except Exception as e:
                print(f"  ⚠ Error processing lower percentile for {label}: {str(e)}")
    
    ax.set_ylabel("EV Price Percentiles, k$", fontsize=14)
    if has_data:
        ax.legend(loc='upper right', fontsize='small', ncol=2)
    add_vertical_lines(ax, base_params1, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- X axis formatting (future period from 2024 onwards)
    future_duration_years = (total_steps - calibration_end_step) / 12
    all_tick_years = np.arange(start_year, start_year + future_duration_years + 5, 5)
    all_tick_positions = calibration_end_step + (all_tick_years - start_year) * 12
    
    visible_mask = all_tick_positions <= total_steps
    tick_positions = all_tick_positions[visible_mask]
    tick_labels = [str(y) for y in all_tick_years[visible_mask]]

    for ax in axs[2]:
        ax.set_xlim(calibration_end_step, total_steps)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Year", fontsize=14)

    # Create output directory if it doesn't exist
    output_dir = "results/comparison_plus_one"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.suptitle(f"Additional Metrics Comparison: {labels[0]} vs {labels[1]} (Future Period from 2024)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{output_dir}/comparison_extra.png", dpi=dpi)
    print(f"\nSaved to {output_dir}/comparison_extra.png")


def main():
    # Hardcode the two file names
    fileName1 = "results/multi_seed_19_48_57__29_04_2026"  # First run
    fileName2 = "results/multi_seed_18_49_55__29_04_2026"  # Second run (change this to your second file)
    
    # Set labels for the runs
    label1 = "Production Subsidy, $20000"
    label2 = "New Car Rebate, $20000"
    
    # Load data
    print(f"Loading data from {fileName1}...")
    base_params1 = load_object(fileName1 + "/Data", "base_params")
    outputs1 = load_object(fileName1 + "/Data", "outputs")
    
    print(f"Loading data from {fileName2}...")
    base_params2 = load_object(fileName2 + "/Data", "base_params")
    outputs2 = load_object(fileName2 + "/Data", "outputs")
    
    # Print data shapes for debugging
    print_data_shapes(outputs1, f"- {label1}")
    print_data_shapes(outputs2, f"- {label2}")
    
    # Plot comparison dashboards
    plot_comparison_dashboard(base_params1, base_params2, fileName1, fileName2, outputs1, outputs2, [label1, label2], dpi=200)
    plot_comparison_extra(base_params1, base_params2, fileName1, fileName2, outputs1, outputs2, [label1, label2], dpi=200)
    
    print("\nComparison complete! Check the 'results/comparison' folder for output plots.")
    plt.show()


if __name__ == "__main__":
    main()