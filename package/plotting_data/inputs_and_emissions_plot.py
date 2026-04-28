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
    data_ev = load_object(f"{results_folder}/Data", "data_phys_trio_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_trio_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    decarb_vals = metadata[0]["property_list"]  
    gas_prices = metadata[1]["property_list"]   
    elec_prices = metadata[2]["property_list"]  
    
    g_idx = next((i for i, v in enumerate(gas_prices) if np.isclose(v, 1.0)), len(gas_prices)//2)

    start_step = 456
    time_indices = np.arange(start_step, data_ev.shape[4])
    dates = [datetime(2024, 1, 1) + timedelta(days=30.44 * (i - start_step)) for i in time_indices]

    fig, axes = plt.subplots(2, len(decarb_vals), figsize=(7 * len(decarb_vals), 12), sharex=True, sharey='row')
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(elec_prices)))

    for d_idx, d_val in enumerate(decarb_vals):
        ax_top, ax_bot = axes[0, d_idx], axes[1, d_idx]
        for e_idx, e_val in enumerate(elec_prices):
            ev_data = data_ev[d_idx, g_idx, e_idx, :, start_step:]
            em_data = data_em[d_idx, g_idx, e_idx, :, start_step:]
            
            mean_ev, ci_ev = np.mean(ev_data, axis=0), stats.sem(ev_data, axis=0) * 1.96
            mean_em, ci_em = np.mean(em_data, axis=0), stats.sem(em_data, axis=0) * 1.96

            lbl = f"Electricity Price: {format_pct_label(e_val)}"
            ax_top.plot(dates, mean_ev, color=colors[e_idx], label=lbl if d_idx == 0 else "", lw=2.5)
            ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=colors[e_idx], alpha=0.1)
            
            ax_bot.plot(dates, mean_em, color=colors[e_idx], lw=2.5)
            ax_bot.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=colors[e_idx], alpha=0.1)

        # Subplot Titles (Now Electricity Emissions Intensity)
        ax_top.set_title(f"Electricity Emissions Intensity:\n{format_pct_label(d_val)}", pad=20, fontweight='bold')
        
        # Grid lines configuration
        ax_top.grid(True, linestyle='--', alpha=0.6)
        ax_bot.grid(True, linestyle='--', alpha=0.6)

        if d_idx == 0:
            ax_top.set_ylabel("EV Uptake Proportion", labelpad=15)
            ax_bot.set_ylabel(r"Monthly Emissions, kg$C0_2$", labelpad=15)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(elec_prices), frameon=True)
    plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.2, wspace=0.25)
    plt.savefig(f"{results_folder}/impact_elec_and_intensity.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Point to your results folder
    RESULTS_PATH = "results/phys_trio_Grid_emissions_intensity_vs_Gas_price_vs_Electricity_price_14_59_12__16_04_2026"

    plot_fixed_gas_at_one(RESULTS_PATH)

    plt.show()