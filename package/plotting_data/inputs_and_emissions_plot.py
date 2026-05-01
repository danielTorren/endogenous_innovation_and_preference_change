import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
from package.resources.utility import load_object

# --- Global Formatting ---
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
    'axes.grid': True
})

def plot_elasticity_comparison(results_folder):
    """
    Compare the effect of a 1% change in electricity price vs grid emissions intensity
    on EV uptake and CO2 emissions.
    
    WHAT THIS PLOT SHOWS:
    - Positive values = a 1% increase in the input leads to an X% INCREASE in the output
    - Negative values = a 1% increase in the input leads to an X% DECREASE in the output
    - The steeper the line, the more sensitive the system is to that factor
    
    INTERPRETATION:
    - EV Uptake Elasticity: How responsive EV adoption is to changes in electricity price vs grid cleanliness
    - Emissions Elasticity: How responsive CO2 emissions are to changes in electricity price vs grid cleanliness
    """
    
    # Load data
    data_ev = load_object(f"{results_folder}/Data", "data_phys_duo_ev")
    data_em = load_object(f"{results_folder}/Data", "data_phys_duo_emissions")
    metadata = load_object(f"{results_folder}/Data", "vary_metadata")
    
    # Get the varying parameters
    grid_intensities = metadata[0]["property_list"]  # Grid emissions intensity (0.1, 0.5, 1.0)
    elec_prices = metadata[1]["property_list"]        # Electricity prices (0.5, 1.0, 1.5)
    
    print(f"Grid intensities tested: {grid_intensities}")
    print(f"Electricity prices tested: {elec_prices}")
    
    # Time setup (from 2024 onwards)
    start_step = 456
    time_length = data_ev.shape[3]
    time_indices = np.arange(start_step, time_length)
    dates = [datetime(2024, 1, 1) + timedelta(days=30.44 * (i - start_step)) for i in time_indices]
    
    # Find baseline (where both factors = 1.0, i.e., "normal" conditions)
    baseline_grid_idx = np.where(np.array(grid_intensities) == 1.0)[0][0]
    baseline_price_idx = np.where(np.array(elec_prices) == 1.0)[0][0]
    
    print(f"\nBaseline: Grid intensity = {grid_intensities[baseline_grid_idx]}, Electricity price = {elec_prices[baseline_price_idx]}")
    
    # Create figure: 2 rows (EV uptake, Emissions) x 2 columns (Price effect, Grid effect)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for different perturbation magnitudes
    colors_price = plt.cm.Blues(np.linspace(0.3, 0.9, len(elec_prices)))  # Blue shades for price
    colors_grid = plt.cm.Reds(np.linspace(0.3, 0.9, len(grid_intensities)))  # Red shades for grid
    
    # =========================================================================
    # LEFT COLUMN: Effect of changing ELECTRICITY PRICE
    # (holding grid intensity constant at baseline)
    # =========================================================================
    
    # Get baseline data
    baseline_ev = data_ev[baseline_grid_idx, baseline_price_idx, :, start_step:]
    baseline_em = data_em[baseline_grid_idx, baseline_price_idx, :, start_step:]
    
    # Plot EV uptake response (top-left)
    ax_ev_price = axes[0, 0]
    # Plot emissions response (bottom-left)
    ax_em_price = axes[1, 0]
    
    for i, price in enumerate(elec_prices):
        if price == 1.0:
            continue  # Skip baseline
            
        # Get data for this electricity price
        ev_data = data_ev[baseline_grid_idx, i, :, start_step:]
        em_data = data_em[baseline_grid_idx, i, :, start_step:]
        
        # Calculate % change in input (electricity price)
        pct_change_input = (price - 1.0) / 1.0 * 100
        
        # Calculate % change in output (EV uptake and emissions)
        pct_change_ev = (ev_data - baseline_ev) / baseline_ev * 100
        pct_change_em = (em_data - baseline_em) / baseline_em * 100
        
        # Elasticity = % change in output / % change in input
        # This tells us the effect of a 1% change in input
        elasticity_ev = pct_change_ev / pct_change_input
        elasticity_em = pct_change_em / pct_change_input
        
        # Calculate mean and confidence intervals across seeds
        mean_ev = np.mean(elasticity_ev, axis=0)
        ci_ev = stats.sem(elasticity_ev, axis=0) * 1.96
        
        mean_em = np.mean(elasticity_em, axis=0)
        ci_em = stats.sem(elasticity_em, axis=0) * 1.96
        
        # Plot
        label = f"Elec price: {price:.1f}x ({pct_change_input:.0f}% change)"
        ax_ev_price.plot(dates, mean_ev, color=colors_price[i], label=label, lw=2.5)
        ax_ev_price.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=colors_price[i], alpha=0.2)
        
        ax_em_price.plot(dates, mean_em, color=colors_price[i], label=label, lw=2.5)
        ax_em_price.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=colors_price[i], alpha=0.2)
    
    # Add baseline reference line at 0
    ax_ev_price.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline (0% change)')
    ax_em_price.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Labels and titles for left column
    ax_ev_price.set_title('Effect of 1% ↑ in Electricity Price\non EV Uptake', fontweight='bold')
    ax_ev_price.set_ylabel('% Change in EV Uptake', labelpad=15)
    ax_ev_price.legend(loc='best', fontsize=12)
    ax_ev_price.grid(True, linestyle='--', alpha=0.6)
    
    ax_em_price.set_title('Effect of 1% ↑ in Electricity Price\non CO2 Emissions', fontweight='bold')
    ax_em_price.set_ylabel('% Change in Monthly Emissions', labelpad=15)
    ax_em_price.legend(loc='best', fontsize=12)  # ADDED LEGEND
    ax_em_price.grid(True, linestyle='--', alpha=0.6)
    
    # =========================================================================
    # RIGHT COLUMN: Effect of changing GRID INTENSITY
    # (holding electricity price constant at baseline)
    # =========================================================================
    
    # Use same baseline data (grid intensity fixed at 1.0, price fixed at 1.0)
    ax_ev_grid = axes[0, 1]
    ax_em_grid = axes[1, 1]
    
    for i, grid_intensity in enumerate(grid_intensities):
        if grid_intensity == 1.0:
            continue  # Skip baseline
            
        # Get data for this grid intensity
        ev_data = data_ev[i, baseline_price_idx, :, start_step:]
        em_data = data_em[i, baseline_price_idx, :, start_step:]
        
        # Calculate % change in input (grid intensity)
        pct_change_input = (grid_intensity - 1.0) / 1.0 * 100
        
        # Calculate % change in output (EV uptake and emissions)
        pct_change_ev = (ev_data - baseline_ev) / baseline_ev * 100
        pct_change_em = (em_data - baseline_em) / baseline_em * 100
        
        # Elasticity
        elasticity_ev = pct_change_ev / pct_change_input
        elasticity_em = pct_change_em / pct_change_input
        
        # Calculate mean and confidence intervals
        mean_ev = np.mean(elasticity_ev, axis=0)
        ci_ev = stats.sem(elasticity_ev, axis=0) * 1.96
        
        mean_em = np.mean(elasticity_em, axis=0)
        ci_em = stats.sem(elasticity_em, axis=0) * 1.96
        
        # Plot
        label = f"Grid intensity: {grid_intensity:.1f}x ({pct_change_input:.0f}% change)"
        ax_ev_grid.plot(dates, mean_ev, color=colors_grid[i], label=label, lw=2.5)
        ax_ev_grid.fill_between(dates, mean_ev - ci_ev, mean_ev + ci_ev, color=colors_grid[i], alpha=0.2)
        
        ax_em_grid.plot(dates, mean_em, color=colors_grid[i], label=label, lw=2.5)
        ax_em_grid.fill_between(dates, mean_em - ci_em, mean_em + ci_em, color=colors_grid[i], alpha=0.2)
    
    # Add baseline reference line at 0
    ax_ev_grid.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline (0% change)')
    ax_em_grid.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Labels and titles for right column
    ax_ev_grid.set_title('Effect of 1% ↑ in Grid Intensity\non EV Uptake', fontweight='bold')
    ax_ev_grid.legend(loc='best', fontsize=12)  # ADDED LEGEND
    ax_ev_grid.grid(True, linestyle='--', alpha=0.6)
    
    ax_em_grid.set_title('Effect of 1% ↑ in Grid Intensity\non CO2 Emissions', fontweight='bold')
    #ax_em_grid.set_ylabel('% Change in Monthly Emissions', labelpad=15)
    ax_em_grid.legend(loc='best', fontsize=12)  # ADDED LEGEND
    ax_em_grid.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis for all subplots
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    
    plt.tight_layout()
    plt.savefig(f"{results_folder}/elasticity_comparison.png", dpi=300, bbox_inches='tight')
    
    print(f"\nPlot saved to: {results_folder}/elasticity_comparison.png")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("ELASTICITY SUMMARY (Last 5 years average)")
    print("="*70)
    
    last_5_years = 60  # months
    
    print("\n--- ELECTRICITY PRICE EFFECTS ---")
    for i, price in enumerate(elec_prices):
        if price == 1.0:
            continue
        ev_data = data_ev[baseline_grid_idx, i, :, start_step:]
        em_data = data_em[baseline_grid_idx, i, :, start_step:]
        
        pct_change_input = (price - 1.0) / 1.0 * 100
        elasticity_ev_final = np.mean(((ev_data - baseline_ev) / baseline_ev * 100)[:, -last_5_years:] / pct_change_input)
        elasticity_em_final = np.mean(((em_data - baseline_em) / baseline_em * 100)[:, -last_5_years:] / pct_change_input)
        
        print(f"\nElectricity Price {price}x (+{pct_change_input:.0f}%):")
        print(f"  EV Elasticity: {elasticity_ev_final:+.2f}% change in EV uptake per 1% price increase")
        print(f"  Emissions Elasticity: {elasticity_em_final:+.2f}% change in emissions per 1% price increase")
    
    print("\n--- GRID INTENSITY EFFECTS ---")
    for i, grid_intensity in enumerate(grid_intensities):
        if grid_intensity == 1.0:
            continue
        ev_data = data_ev[i, baseline_price_idx, :, start_step:]
        em_data = data_em[i, baseline_price_idx, :, start_step:]
        
        pct_change_input = (grid_intensity - 1.0) / 1.0 * 100
        elasticity_ev_final = np.mean(((ev_data - baseline_ev) / baseline_ev * 100)[:, -last_5_years:] / pct_change_input)
        elasticity_em_final = np.mean(((em_data - baseline_em) / baseline_em * 100)[:, -last_5_years:] / pct_change_input)
        
        print(f"\nGrid Intensity {grid_intensity}x (+{pct_change_input:.0f}%):")
        print(f"  EV Elasticity: {elasticity_ev_final:+.2f}% change in EV uptake per 1% intensity increase")
        print(f"  Emissions Elasticity: {elasticity_em_final:+.2f}% change in emissions per 1% intensity increase")
    
    return fig

if __name__ == "__main__":
    RESULTS_PATH = "results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_16_18_40__30_04_2026"
    plot_elasticity_comparison(RESULTS_PATH)
    plt.show()