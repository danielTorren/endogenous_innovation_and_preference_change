import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
from package.resources.utility import load_object

# --- Global Formatting ---
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def format_correlation_label(val):
    """
    Extracts the last element from the rho vector [1, 0, 0, x] 
    and returns it as a correlation label.
    """
    # Ensure we are looking at the last element if it's a list/array
    if isinstance(val, (list, np.ndarray)):
        corr_val = val[-1]
    else:
        corr_val = val
        
    return f"Correlation: {corr_val}"

def plot_single_parameter_sensitivity(results_folder):
    # 1. Load Data
    data_ev = load_object(f"{results_folder}/Data", "data_ev")
    data_em = load_object(f"{results_folder}/Data", "data_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    property_list = metadata["property_list"]
    # Clean up the variable name for the label
    property_name = metadata["property_varied"].replace("_", " ").title()
    
    # 2. Time Logic
    start_step = 456
    time_indices = np.arange(start_step, data_ev.shape[2])
    dates = [datetime(2024, 1, 1) + timedelta(days=30.44 * (i - start_step)) for i in time_indices]

    # 3. Setup Figure (2 Rows, 1 Column)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 16), sharex=True)
    
    # Using Viridis for correlation intensity
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(property_list)))

    # 4. Plotting Loop
    for i, val in enumerate(property_list):
        # Data slices [Scenario, Seed, Time]
        ev_scen = data_ev[i, :, start_step:]
        em_scen = data_em[i, :, start_step:]
        
        # Mean and 95% Confidence Interval
        mean_ev = np.mean(ev_scen, axis=0)
        ci_ev = stats.sem(ev_scen, axis=0) * 1.96
        
        mean_em = np.mean(em_scen, axis=0)
        ci_em = stats.sem(em_scen, axis=0) * 1.96

        # Create label using only the changing correlation value
        label_str = format_correlation_label(val)

        # Top Plot: EV Proportion
        ax_top.plot(dates, mean_ev, color=colors[i], label=label_str, lw=4)
        ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=colors[i], alpha=0.15)
        
        # Bottom Plot: Emissions Flow
        ax_bot.plot(dates, mean_em, color=colors[i], lw=4)
        ax_bot.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=colors[i], alpha=0.15)

    # 5. Labels and Axis Formatting
    ax_top.set_ylabel("EV Proportion of Fleet", labelpad=20)
    ax_bot.set_ylabel("Monthly Emissions Flow", labelpad=20)
    ax_bot.set_xlabel("Year", labelpad=20)

    # Format dates to show only years
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())

    # Legend on the top plot only to avoid duplication
    ax_top.legend(loc='upper left', frameon=True, fontsize=18)
    
    plt.tight_layout()
    
    # 6. Save result
    save_path = f"{results_folder}/sensitivity_{metadata['property_varied']}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    

if __name__ == "__main__":
    # Ensure PATH points to the directory containing the 'Data' folder
    PATH = "results/battery_corr_rho_12_28_15__23_04_2026"
    plot_single_parameter_sensitivity(PATH)

    plt.show()
