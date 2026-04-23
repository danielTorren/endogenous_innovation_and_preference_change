import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
from matplotlib.lines import Line2D
from package.resources.utility import load_object

def plot_trio_time_series_grid(results_folder):
    # 1. Load Data
    data_ev = load_object(f"{results_folder}/Data", "data_phys_trio_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_trio_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    decarb_vals = metadata[0]["property_list"]  
    gas_prices = metadata[1]["property_list"]   
    elec_prices = metadata[2]["property_list"]  
    
    start_step = 456
    num_seeds = data_ev.shape[3]
    total_steps = data_ev.shape[4]
    
    # 2. Handle Time/Date logic
    time_indices = np.arange(start_step, total_steps)
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=30.44 * (step - start_step)) for step in time_indices]

    # 3. Setup Figure: sharey=True ensures the Y-axis scale is identical for all columns in a row
    num_cols = len(elec_prices)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharex=True, sharey='row')
    
    if num_cols == 1: 
        axes = axes.reshape(2, 1)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(gas_prices)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    # 4. Plotting Loop
    for c_idx, e_val in enumerate(elec_prices):
        ax_top, ax_bot = axes[0, c_idx], axes[1, c_idx]
        
        for g_idx, g_val in enumerate(gas_prices):
            for d_idx, d_val in enumerate(decarb_vals):
                # Correct Indexing: [Decarb, Gas, Elec, Seeds, Time]
                seeds_ev = data_ev[d_idx, g_idx, c_idx, :, start_step:]
                seeds_em = data_em[d_idx, g_idx, c_idx, :, start_step:]
                
                mean_ev = np.mean(seeds_ev, axis=0)
                ci_ev = stats.sem(seeds_ev, axis=0) * stats.t.ppf(0.975, num_seeds - 1)
                
                mean_em = np.mean(seeds_em, axis=0)
                ci_em = stats.sem(seeds_em, axis=0) * stats.t.ppf(0.975, num_seeds - 1)

                m_style = markers[d_idx % len(markers)]
                m_freq = max(1, len(dates)//12)

                # Row 0: EV Proportion
                ax_top.plot(dates, mean_ev, color=colors[g_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, 
                                    color=colors[g_idx], alpha=0.1)
                
                # Row 1: Emissions
                ax_bot.plot(dates, mean_em, color=colors[g_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_bot.fill_between(dates, mean_em - ci_em, mean_em + ci_em, 
                                    color=colors[g_idx], alpha=0.1)

        ax_top.set_title(f"Elec Price: {e_val}")
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_bot.grid(True, linestyle='--', alpha=0.3)
        
        if c_idx == 0:
            ax_top.set_ylabel("EV Proportion")
            ax_bot.set_ylabel("Total Emissions")

    # 5. Legend Proxy Artists
    color_handles = [Line2D([0], [0], color=colors[i], lw=3, label=f"Gas: {val}") 
                     for i, val in enumerate(gas_prices)]
    
    marker_handles = [Line2D([0], [0], color='gray', marker=markers[i], linestyle='None',
                             markersize=10, label=f"Decarb: {val}") 
                      for i, val in enumerate(decarb_vals)]

    # 6. Optimized Legend Placement
    # We create one large legend area at the very top
    leg1 = fig.legend(handles=color_handles, title="Gas Price (Color)", 
                      loc='upper center', bbox_to_anchor=(0.3, 0.96), 
                      ncol=len(gas_prices), frameon=True)
    
    leg2 = fig.legend(handles=marker_handles, title="Grid Intensity (Marker)", 
                      loc='upper center', bbox_to_anchor=(0.7, 0.96), 
                      ncol=len(decarb_vals), frameon=True)

    fig.add_artist(leg1)

    # 7. Final Polish
    for ax in axes[1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    fig.autofmt_xdate()
    
    # Increase the top margin significantly to prevent legend overlap
    plt.subplots_adjust(top=0.82, hspace=0.3, wspace=0.15)
    
    plt.savefig(f"{results_folder}/trio_comparable_grid.png", dpi=300, bbox_inches='tight')


def plot_pct_trio_time_series_grid(results_folder):
    # 1. Load Data
    data_ev = load_object(f"{results_folder}/Data", "data_phys_trio_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_trio_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    decarb_vals = metadata[0]["property_list"]  
    gas_prices = metadata[1]["property_list"]   
    elec_prices = metadata[2]["property_list"]  
    
    start_step = 456
    num_seeds = data_ev.shape[3]
    total_steps = data_ev.shape[4]
    
    # 2. Handle Time/Date logic
    time_indices = np.arange(start_step, total_steps)
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=30.44 * (step - start_step)) for step in time_indices]

    # 3. Setup Figure
    num_cols = len(elec_prices)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharex=True, sharey='row')
    if num_cols == 1: axes = axes.reshape(2, 1)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(gas_prices)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    # 4. Plotting Loop
    for c_idx, e_val in enumerate(elec_prices):
        ax_top, ax_bot = axes[0, c_idx], axes[1, c_idx]
        
        for g_idx, g_val in enumerate(gas_prices):
            for d_idx, d_val in enumerate(decarb_vals):
                # Data Slicing
                seeds_ev = data_ev[d_idx, g_idx, c_idx, :, start_step:]
                
                # --- Emissions Percentage Change Logic ---
                # Grab step 0 for all seeds in this specific parameter set
                # Shape: (num_seeds, 1)
                initial_emissions = data_em[d_idx, g_idx, c_idx, :, 456].reshape(-1, 1)
                
                # Extract emissions from start_step onwards
                raw_em_seeds = data_em[d_idx, g_idx, c_idx, :, start_step:]
                
                # Calculate % change: ((Current - Initial) / Initial) * 100
                seeds_em_pct = ((raw_em_seeds - initial_emissions) / initial_emissions) * 100
                
                # Statistics
                def get_stats(data):
                    mean = np.mean(data, axis=0)
                    ci = stats.sem(data, axis=0) * stats.t.ppf(0.975, num_seeds - 1)
                    return mean, ci

                mean_ev, ci_ev = get_stats(seeds_ev)
                mean_em_pct, ci_em_pct = get_stats(seeds_em_pct)

                m_style = markers[d_idx % len(markers)]
                m_freq = max(1, len(dates)//12)

                # Row 0: EV Proportion
                ax_top.plot(dates, mean_ev, color=colors[g_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, 
                                    color=colors[g_idx], alpha=0.1)
                
                # Row 1: Emissions % Change
                ax_bot.plot(dates, mean_em_pct, color=colors[g_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_bot.fill_between(dates, mean_em_pct - ci_em_pct, mean_em_pct + ci_em_pct, 
                                    color=colors[g_idx], alpha=0.1)

        ax_top.set_title(f"Electricity Price Multiplier: {e_val}")
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_bot.grid(True, linestyle='--', alpha=0.3)
        
        # Add a horizontal line at 0 for emissions change
        ax_bot.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
        
        if c_idx == 0:
            ax_top.set_ylabel("EV Proportion")
            ax_bot.set_ylabel("Emissions % Change\n(Relative to Jan 2024)")

    # 5. Legend Proxy Artists
    color_handles = [Line2D([0], [0], color=colors[i], lw=3, label=f"Gas: {val}") 
                     for i, val in enumerate(gas_prices)]
    marker_handles = [Line2D([0], [0], color='gray', marker=markers[i], linestyle='None',
                             markersize=10, label=f"Decarb: {val}") 
                      for i, val in enumerate(decarb_vals)]

    leg1 = fig.legend(handles=color_handles, title="Gas Price Multiplier", 
                      loc='upper center', bbox_to_anchor=(0.3, 0.98), 
                      ncol=len(gas_prices), frameon=True)
    leg2 = fig.legend(handles=marker_handles, title="Electricity Decarbonisation Multiplier", 
                      loc='upper center', bbox_to_anchor=(0.7, 0.98), 
                      ncol=len(decarb_vals), frameon=True)
    fig.add_artist(leg1)

    # 6. Final Polish
    for ax in axes[1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    fig.autofmt_xdate()

# Move the super x-label UP by setting the 'y' parameter. 
    # Adjust y=0.08 based on your preference; smaller numbers move it down.
    fig.supxlabel("Year", fontsize=12, y=0.1)

    # hspace=0 makes the rows touch. 
    # wspace=0.05 keeps columns very close.
    # bottom=0.18 ensures the Year label and dates aren't cut off by the edge.
    plt.subplots_adjust(top=0.87, bottom=0.18, hspace=0, wspace=0.05)
    
    plt.savefig(f"{results_folder}/emissions_pct_change_grid.png", dpi=300, bbox_inches='tight')
    
def plot_pct_trio_time_series_grid_EI(results_folder):
    # 1. Load Data
    data_ev = load_object(f"{results_folder}/Data", "data_phys_trio_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_trio_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    decarb_vals = metadata[0]["property_list"]  # Now Columns
    gas_prices = metadata[1]["property_list"]   # Now Markers
    elec_prices = metadata[2]["property_list"]  # Now Colors
    
    start_step = 456
    num_seeds = data_ev.shape[3]
    total_steps = data_ev.shape[4]
    
    # 2. Handle Time/Date logic
    time_indices = np.arange(start_step, total_steps)
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=30.44 * (step - start_step)) for step in time_indices]

    # 3. Setup Figure: Columns = Decarb Vals
    num_cols = len(decarb_vals)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharex=True, sharey='row')
    if num_cols == 1: axes = axes.reshape(2, 1)

    # Colors represent Electricity Price
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(elec_prices)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    # 4. Plotting Loop
    # Outer loop defines the Columns (Decarb Intensity)
    for d_idx, d_val in enumerate(decarb_vals):
        ax_top, ax_bot = axes[0, d_idx], axes[1, d_idx]
        
        # Mid loop defines Color (Elec Price)
        for e_idx, e_val in enumerate(elec_prices):
            # Inner loop defines Markers (Gas Price)
            for g_idx, g_val in enumerate(gas_prices):
                
                # Correct Data Slicing: [Decarb, Gas, Elec, Seeds, Time]
                seeds_ev = data_ev[d_idx, g_idx, e_idx, :, start_step:]
                initial_emissions = data_em[d_idx, g_idx, e_idx, :, 456].reshape(-1, 1)
                raw_em_seeds = data_em[d_idx, g_idx, e_idx, :, start_step:]
                
                seeds_em_pct = ((raw_em_seeds - initial_emissions) / initial_emissions) * 100
                
                def get_stats(data):
                    mean = np.mean(data, axis=0)
                    ci = stats.sem(data, axis=0) * stats.t.ppf(0.975, num_seeds - 1)
                    return mean, ci

                mean_ev, ci_ev = get_stats(seeds_ev)
                mean_em_pct, ci_em_pct = get_stats(seeds_em_pct)

                m_style = markers[g_idx % len(markers)] # Gas price determines marker
                m_freq = max(1, len(dates)//12)

                # Row 0: EV Proportion
                ax_top.plot(dates, mean_ev, color=colors[e_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, 
                                    color=colors[e_idx], alpha=0.1)
                
                # Row 1: Emissions % Change
                ax_bot.plot(dates, mean_em_pct, color=colors[e_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_bot.fill_between(dates, mean_em_pct - ci_em_pct, mean_em_pct + ci_em_pct, 
                                    color=colors[e_idx], alpha=0.1)

        ax_top.set_title(f"Grid Intensity Multiplier: {d_val}")
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_bot.grid(True, linestyle='--', alpha=0.3)
        ax_bot.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
        
        if d_idx == 0:
            ax_top.set_ylabel("EV Proportion")
            ax_bot.set_ylabel("Emissions % Change\n(Relative to Jan 2024)")

    # 5. Legend Proxy Artists (Updated Labels)
    color_handles = [Line2D([0], [0], color=colors[i], lw=3, label=f"Elec: {val}") 
                     for i, val in enumerate(elec_prices)]
    
    marker_handles = [Line2D([0], [0], color='gray', marker=markers[i], linestyle='None',
                             markersize=10, label=f"Gas: {val}") 
                      for i, val in enumerate(gas_prices)]

    leg1 = fig.legend(handles=color_handles, title="Electricity Price Multiplier", 
                      loc='upper center', bbox_to_anchor=(0.3, 0.98), 
                      ncol=len(elec_prices), frameon=True)
    
    leg2 = fig.legend(handles=marker_handles, title="Gasoline Price (Marker)", 
                      loc='upper center', bbox_to_anchor=(0.7, 0.98), 
                      ncol=len(gas_prices), frameon=True)
    fig.add_artist(leg1)

    # 6. Final Polish
    for ax in axes[1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    fig.autofmt_xdate()
    fig.supxlabel("Year", fontsize=12, y=0.1)
    plt.subplots_adjust(top=0.87, bottom=0.18, hspace=0, wspace=0.05)
    
    plt.savefig(f"{results_folder}/emissions_intensity_columns.png", dpi=300, bbox_inches='tight')


def plot_cumulative_emissions_grid(results_folder):
    # 1. Load Data
    data_ev = load_object(f"{results_folder}/Data", "data_phys_trio_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_trio_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    decarb_vals = metadata[0]["property_list"]  # Columns
    gas_prices = metadata[1]["property_list"]   # Markers
    elec_prices = metadata[2]["property_list"]  # Colors
    
    start_step = 456
    num_seeds = data_ev.shape[3]
    total_steps = data_ev.shape[4]
    
    # 2. Handle Time/Date logic
    time_indices = np.arange(start_step, total_steps)
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=30.44 * (step - start_step)) for step in time_indices]

    # 3. Setup Figure
    num_cols = len(decarb_vals)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharex=True, sharey='row')
    if num_cols == 1: axes = axes.reshape(2, 1)

    colors = plt.cm.plasma(np.linspace(0, 0.8, len(elec_prices)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    # 4. Plotting Loop
    for d_idx, d_val in enumerate(decarb_vals):
        ax_top, ax_bot = axes[0, d_idx], axes[1, d_idx]
        
        for e_idx, e_val in enumerate(elec_prices):
            for g_idx, g_val in enumerate(gas_prices):
                
                # --- Data Extraction ---
                seeds_ev = data_ev[d_idx, g_idx, e_idx, :, start_step:]
                raw_em_seeds = data_em[d_idx, g_idx, e_idx, :, start_step:]
                
                # --- Cumulative Emissions Calculation ---
                # np.cumsum across the time axis (axis=1)
                seeds_em_cum = np.cumsum(raw_em_seeds, axis=1)
                
                def get_stats(data):
                    mean = np.mean(data, axis=0)
                    ci = stats.sem(data, axis=0) * stats.t.ppf(0.975, num_seeds - 1)
                    return mean, ci

                mean_ev, ci_ev = get_stats(seeds_ev)
                mean_em_cum, ci_em_cum = get_stats(seeds_em_cum)

                m_style = markers[g_idx % len(markers)] 
                m_freq = max(1, len(dates)//12)

                # Row 0: EV Proportion
                ax_top.plot(dates, mean_ev, color=colors[e_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, 
                                    color=colors[e_idx], alpha=0.1)
                
                # Row 1: Cumulative Emissions (Raw)
                ax_bot.plot(dates, mean_em_cum, color=colors[e_idx], marker=m_style, 
                            markevery=m_freq, alpha=0.7, linewidth=1.5)
                ax_bot.fill_between(dates, mean_em_cum - ci_em_cum, mean_em_cum + ci_em_cum, 
                                    color=colors[e_idx], alpha=0.1)

        ax_top.set_title(f"Grid Intensity: {d_val}")
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_bot.grid(True, linestyle='--', alpha=0.3)
        
        if d_idx == 0:
            ax_top.set_ylabel("EV Proportion")
            ax_bot.set_ylabel("Cumulative Emissions\n(Raw Units)")

    # 5. Legend
    color_handles = [Line2D([0], [0], color=colors[i], lw=3, label=f"Elec: {val}") 
                     for i, val in enumerate(elec_prices)]
    marker_handles = [Line2D([0], [0], color='gray', marker=markers[i], linestyle='None',
                             markersize=10, label=f"Gas: {val}") 
                      for i, val in enumerate(gas_prices)]

    leg1 = fig.legend(handles=color_handles, title="Electricity Price Multiplier", 
                      loc='upper center', bbox_to_anchor=(0.3, 0.98), 
                      ncol=len(elec_prices), frameon=True)
    leg2 = fig.legend(handles=marker_handles, title="Gasoline Price (Marker Shape)", 
                      loc='upper center', bbox_to_anchor=(0.7, 0.98), 
                      ncol=len(gas_prices), frameon=True)
    fig.add_artist(leg1)

    # 6. Formatting
    for ax in axes[1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    fig.autofmt_xdate()
    fig.supxlabel("Year", fontsize=12, y=0.1)
    plt.subplots_adjust(top=0.87, bottom=0.18, hspace=0, wspace=0.05)
    
    plt.savefig(f"{results_folder}/cumulative_emissions_grid.png", dpi=300, bbox_inches='tight')


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

def plot_fixed_elec_at_one(results_folder):
    data_ev = load_object(f"{results_folder}/Data", "data_phys_trio_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_trio_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")

    decarb_vals = metadata[0]["property_list"]  
    gas_prices = metadata[1]["property_list"]   
    elec_prices = metadata[2]["property_list"]  
    
    e_idx = next((i for i, v in enumerate(elec_prices) if np.isclose(v, 1.0)), len(elec_prices)//2)

    start_step = 456
    time_indices = np.arange(start_step, data_ev.shape[4])
    dates = [datetime(2024, 1, 1) + timedelta(days=30.44 * (i - start_step)) for i in time_indices]

    fig, axes = plt.subplots(2, len(decarb_vals), figsize=(7 * len(decarb_vals), 12), sharex=True, sharey='row')
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(gas_prices)))

    for d_idx, d_val in enumerate(decarb_vals):
        ax_top, ax_bot = axes[0, d_idx], axes[1, d_idx]
        for g_idx, g_val in enumerate(gas_prices):
            ev_data = data_ev[d_idx, g_idx, e_idx, :, start_step:]
            em_data = data_em[d_idx, g_idx, e_idx, :, start_step:]
            
            mean_ev, ci_ev = np.mean(ev_data, axis=0), stats.sem(ev_data, axis=0) * 1.96
            mean_em, ci_em = np.mean(em_data, axis=0), stats.sem(em_data, axis=0) * 1.96

            lbl = f"Gasoline Price: {format_pct_label(g_val)}"
            ax_top.plot(dates, mean_ev, color=colors[g_idx], label=lbl if d_idx == 0 else "", lw=2.5)
            ax_top.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=colors[g_idx], alpha=0.1)
            
            ax_bot.plot(dates, mean_em, color=colors[g_idx], lw=2.5)
            ax_bot.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=colors[g_idx], alpha=0.1)

        ax_top.set_title(f"Electricity Emissions Intensity:\n{format_pct_label(d_val)}", pad=20, fontweight='bold')
        ax_top.grid(True, linestyle='--', alpha=0.6)
        ax_bot.grid(True, linestyle='--', alpha=0.6)

    axes[0, 0].set_ylabel("EV Proportion", labelpad=15)
    axes[1, 0].set_ylabel(r"Monthly Emissions, kg$C0_2$", labelpad=15)
    
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(gas_prices), frameon=True)
    plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.2, wspace=0.25)
    plt.savefig(f"{results_folder}/impact_gas_and_intensity.png", dpi=300, bbox_inches='tight')
if __name__ == "__main__":
    # Point to your results folder
    RESULTS_PATH = "results/phys_trio_Grid_emissions_intensity_vs_Gas_price_vs_Electricity_price_14_59_12__16_04_2026"
    #plot_trio_time_series_grid(RESULTS_PATH)
    #plot_pct_trio_time_series_grid(RESULTS_PATH)
    #plot_pct_trio_time_series_grid_EI(RESULTS_PATH)
    #plot_cumulative_emissions_grid(RESULTS_PATH)
    plot_fixed_gas_at_one(RESULTS_PATH)
    plot_fixed_elec_at_one(RESULTS_PATH)

    plt.show()