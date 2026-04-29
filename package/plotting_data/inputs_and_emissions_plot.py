import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
from package.resources.utility import load_object

# --- Global Formatting ---
plt.rcParams.update({
    'font.size': 16,          # Base font size
    'axes.titlesize': 18,     # Subplot titles
    'axes.labelsize': 18,     # X and Y labels
    'xtick.labelsize': 14,    # X tick labels
    'ytick.labelsize': 14,    # Y tick labels
    'legend.fontsize': 15,    # Legend text
    'axes.grid': True         # Enable grid lines globally
})

def format_pct_label(val):
    """Formats labels as % change/reduction."""
    if np.isclose(val, 1.0):
        return "0% Change"
    elif val < 1.0:
        return f"{int(round((1 - val) * 100))}% Reduction"
    else:
        return f"{int(round(val * 100))}% Increase"

def plot_fixed_gas_at_one(results_folder):
    data_ev = load_object(f"{results_folder}/Data", "data_phys_duo_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_duo_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")
    
    # Get the varying parameters
    decarb_vals = metadata[0]["property_list"]  
    elec_prices = metadata[1]["property_list"]   
    
    print(f"Data shapes: EV {data_ev.shape}, Emissions {data_em.shape}")
    print(f"Decarb values: {decarb_vals}")
    print(f"Electricity prices: {elec_prices}")
    
    start_step = 456
    time_length = data_ev.shape[3]  # Time is the 4th dimension (index 3)
    
    if start_step >= time_length:
        raise ValueError(f"start_step ({start_step}) >= time_length ({time_length})")
    
    time_indices = np.arange(start_step, time_length)
    dates = [datetime(2024, 1, 1) + timedelta(days=30.44 * (i - start_step)) for i in time_indices]
    
    # Create subplots: rows = 2 (EV uptake + Emissions), cols = len(decarb_vals)
    fig, axes = plt.subplots(2, len(decarb_vals), figsize=(7 * len(decarb_vals), 12), sharex=True, sharey='row')
    
    # Handle case where there's only 1 decarb value (axes might not be 2D)
    if len(decarb_vals) == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(elec_prices)))
    
    for d_idx, d_val in enumerate(decarb_vals):
        ax_top, ax_bot = axes[0, d_idx], axes[1, d_idx]
        
        for e_idx, e_val in enumerate(elec_prices):
            # Correct indexing for 4D array: (decarb, elec, seeds, time)
            ev_data = data_ev[d_idx, e_idx, :, start_step:]  # Shape: (seeds, time_slice)
            em_data = data_em[d_idx, e_idx, :, start_step:]  # Shape: (seeds, time_slice)
            
            mean_ev = np.mean(ev_data, axis=0)
            ci_ev = stats.sem(ev_data, axis=0) * 1.96
            
            mean_em = np.mean(em_data, axis=0)
            ci_em = stats.sem(em_data, axis=0) * 1.96
            
            lbl = f"Electricity Price: {format_pct_label(e_val)}"
            ax_top.plot(dates, mean_ev, color=colors[e_idx], label=lbl if d_idx == 0 else "", lw=2.5)
            ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=colors[e_idx], alpha=0.1)
            
            ax_bot.plot(dates, mean_em, color=colors[e_idx], lw=2.5)
            ax_bot.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=colors[e_idx], alpha=0.1)
        
        # Subplot Titles
        ax_top.set_title(f"Electricity Emissions Intensity:\n{format_pct_label(d_val)}", pad=20, fontweight='bold')
        
        # Grid lines configuration
        ax_top.grid(True, linestyle='--', alpha=0.6)
        ax_bot.grid(True, linestyle='--', alpha=0.6)
        
        if d_idx == 0:
            ax_top.set_ylabel("EV Uptake Proportion", labelpad=15)
            ax_bot.set_ylabel(r"Monthly Emissions, kg$CO_2$", labelpad=15)
    
    # Add legend only if there are multiple electricity prices to show
    if len(elec_prices) > 1:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(len(elec_prices), 4), frameon=True)
    
    # Format x-axis - show year labels every 5 years
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Set major ticks every 5 years (5 * 365 days)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        # Optional: add minor ticks every year
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.subplots_adjust(top=0.9, bottom=0.2, hspace=0.2, wspace=0.25)
    plt.savefig(f"{results_folder}/impact_elec_and_intensity.png", dpi=300, bbox_inches='tight')

    print(f"Plot saved to: {results_folder}/impact_elec_and_intensity.png")

if __name__ == "__main__":
    # Point to your results folder from the duo run
    RESULTS_PATH = "results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_20_59_52__28_04_2026"
    
    plot_fixed_gas_at_one(RESULTS_PATH)

    plt.show()