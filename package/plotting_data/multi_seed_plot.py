import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
from matplotlib.lines import Line2D


def add_vertical_lines(ax, base_params, color='black', linestyle='--', annotation_height_prop=[0.2, 0.2, 0.2, 0.2]):
    ev_production_start_time = base_params["ev_production_start_time"]
    # Policies start at the end of calibration, which is burn-in + calibration
    # steps into the run (duration_calibration excludes the burn-in)
    calibration_end_step = base_params["duration_burn_in"] + base_params["duration_calibration"]

    y_min, y_max = ax.get_ylim()
    annotation_height_0 = y_min + annotation_height_prop[0] * (y_max - y_min)
    annotation_height_1 = y_min + annotation_height_prop[1] * (y_max - y_min)
    annotation_height_2 = y_min + annotation_height_prop[2] * (y_max - y_min)
    annotation_height_3 = y_min + annotation_height_prop[3] * (y_max - y_min)

    # EV Sale Start
    if ev_production_start_time > 0:
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
        policy_start_time = calibration_end_step
        ax.axvline(policy_start_time, color="black", linestyle='--')
        ax.annotate("Policy Start", xy=(policy_start_time, annotation_height_2),
                    rotation=90, verticalalignment='center', horizontalalignment='right',
                    fontsize=8, color='black')

        # Policy End
        if base_params["duration_future"] >= 144:
            policy_end_time = calibration_end_step + 144
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


CALIBRATION_END_YEAR = 2024  # The end of calibration is Jan 2024


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
        print(f"  ⚠ Failed to plot {label}: {str(e)}")
        ax.text(0.5, 0.5, f'{label} unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, alpha=0.7)
        return False


def print_data_shapes(outputs, base_params):
    """Print shapes of all relevant data arrays for debugging with robust error handling"""
    print("\n" + "="*60)
    print("DATA SHAPES FOR DEBUGGING")
    print("="*60)
    
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
                # For multi-dimensional arrays, show more detail
                if hasattr(outputs[key], 'shape') and len(outputs[key].shape) >= 3:
                    print(f"✓ {key}: {shape}")
                    # Show interpretation of dimensions
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
        else:
            print("Cannot determine timeline - history_prop_EV not found")
    except Exception as e:
        print(f"Error determining timeline: {str(e)}")
    
    print("="*60 + "\n")


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
    plot_line_with_ci(ax, outputs["history_mean_price_ICE_EV_arr"][:, :, 0, 1], '#0072B2', '-', 'New EV')
    plot_line_with_ci(ax, outputs["history_mean_price_ICE_EV_arr"][:, :, 1, 1], '#0072B2', '--', 'Used EV')
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
    fig.savefig(f"{fileName}/Plots/multi_seed_dashboard.png", dpi=dpi)
    print(f"Saved to {fileName}/Plots/multi_seed_dashboard.png")


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
        print(f"✓ Plotting history_driving_emissions (shape: {get_shape_safe(outputs['history_driving_emissions'])})")
        if plot_line_with_ci(ax, outputs["history_driving_emissions"] * 1e-9, '#D55E00', '-', 'Driving'):
            has_data = True
    else:
        print("✗ history_driving_emissions not found")
    
    if "history_production_emissions" in outputs:
        print(f"✓ Plotting history_production_emissions (shape: {get_shape_safe(outputs['history_production_emissions'])})")
        if plot_line_with_ci(ax, outputs["history_production_emissions"] * 1e-9, '#E69F00', '--', 'Production'):
            has_data = True
    else:
        print("✗ history_production_emissions not found")
    
    ax.set_ylabel("Flow Emissions, MTCO2", fontsize=14)
    if has_data:
        ax.legend(loc='upper right', fontsize='small')
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Cumulative Driving vs Production Emissions
    ax = axs[0, 1]
    has_data = False
    
    if "history_driving_emissions" in outputs:
        cum_driving = np.cumsum(outputs["history_driving_emissions"], axis=1) * 1e-9
        print(f"✓ Plotting cumulative driving emissions (shape after cumsum: {cum_driving.shape})")
        if plot_line_with_ci(ax, cum_driving, '#D55E00', '-', 'Driving'):
            has_data = True
    else:
        print("✗ Cannot plot cumulative driving emissions - data not found")
    
    if "history_production_emissions" in outputs:
        cum_prod = np.cumsum(outputs["history_production_emissions"], axis=1) * 1e-9
        print(f"✓ Plotting cumulative production emissions (shape after cumsum: {cum_prod.shape})")
        if plot_line_with_ci(ax, cum_prod, '#E69F00', '--', 'Production'):
            has_data = True
    else:
        print("✗ Cannot plot cumulative production emissions - data not found")
    
    ax.set_ylabel("Cumulative Emissions, MTCO2", fontsize=14)
    if has_data:
        ax.legend(loc='upper left', fontsize='small')
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Market Concentration (HHI)
    ax = axs[1, 0]
    if "history_market_concentration" in outputs:
        print(f"✓ Plotting history_market_concentration (shape: {get_shape_safe(outputs['history_market_concentration'])})")
        plot_line_with_ci(ax, outputs["history_market_concentration"], '#0072B2', '-', 'HHI')
    else:
        print("✗ history_market_concentration not found")
        ax.text(0.5, 0.5, 'HHI Data not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax.set_ylabel("Market Concentration (HHI)", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- Total Profit
    ax = axs[1, 1]
    has_data = False
    
    if "history_total_profit" in outputs:
        print(f"✓ Plotting history_total_profit (shape: {get_shape_safe(outputs['history_total_profit'])})")
        if plot_line_with_ci(ax, outputs["history_total_profit"] * 1e-9, '#0072B2', '-', 'Total Profit'):
            has_data = True
    else:
        print("✗ history_total_profit not found")
    
    ax.set_ylabel("Total Profit, bn $", fontsize=14)

    # Margins are fractions, so they get their own axis to be visible alongside
    # profit in billions
    ax_margin = ax.twinx()
    for key, colour, linestyle, label in (
        ("history_mean_profit_margins_ICE", '#D55E00', '--', 'ICE Margin'),
        ("history_mean_profit_margins_EV", '#009E73', ':', 'EV Margin'),
    ):
        if key not in outputs:
            print(f"✗ {key} not found")
            continue
        print(f"✓ Plotting {key} (shape: {get_shape_safe(outputs[key])})")
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
        print(f"✓ Plotting history_mean_car_age (shape: {get_shape_safe(outputs['history_mean_car_age'])})")
        plot_line_with_ci(ax, outputs["history_mean_car_age"], '#CC79A7', '-', 'Mean Car Age')
    else:
        print("✗ history_mean_car_age not found")
        ax.text(0.5, 0.5, 'Mean Car Age Data not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax.set_ylabel("Mean Car Age, months", fontsize=14)
    add_vertical_lines(ax, base_params, annotation_height_prop=[0.5, 0.2, 0.2, 0.2])

    # --- EV Price Spread
    ax = axs[2, 1]
    has_data = False
    
    for key, linestyle, label in (
        ("history_upper_percentile_price_ICE_EV_arr", '-', 'Upper Percentile EV'),
        ("history_lower_percentile_price_ICE_EV_arr", '--', 'Lower Percentile EV'),
    ):
        if key not in outputs:
            print(f"✗ {key} not found")
            continue

        arr = np.asarray(outputs[key], dtype=float)
        print(f"✓ Plotting {key} (shape: {arr.shape})")
        if arr.ndim == 4:
            # (n_seeds, n_steps, new/used, ICE/EV) - take new EV, matching the
            # mean price panel on the main dashboard
            series = arr[:, :, 0, 1]
            print(f"  → Using 4D indexing [:, :, 0, 1] (new EV), resulting shape: {series.shape}")
        elif arr.ndim == 3:
            # (n_seeds, n_steps, ICE/EV)
            series = arr[:, :, 1]
            print(f"  → Using 3D indexing [:, :, 1], resulting shape: {series.shape}")
        else:
            series = arr
            print(f"  → Using data as is, shape: {series.shape}")

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
    fig.savefig(f"{fileName}/Plots/multi_seed_dashboard_extra.png", dpi=dpi)
    print(f"\nSaved to {fileName}/Plots/multi_seed_dashboard_extra.png")


def main(fileName):
    base_params = load_object(fileName + "/Data", "base_params")
    outputs = load_object(fileName + "/Data", "outputs")
    
    # Print all data shapes before plotting
    print_data_shapes(outputs, base_params)
    
    # Plot dashboards
    plot_multi_seed_dashboard(base_params, fileName, outputs, dpi=200)
    plot_multi_seed_dashboard_extra(base_params, fileName, outputs, dpi=200)
    
    plt.show()


if __name__ == "__main__":
    main(fileName="results/multi_seed_15_49_15__04_08_2026")