import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from package.resources.utility import load_object

def load_ev_data(folder):
    base_params = load_object(folder + "/Data", "base_params")
    ev_prop = load_object(folder + "/Data", "data_array_ev_prop")
    vary_single = load_object(folder + "/Data", "vary_single")
    property_list = vary_single["property_list"]
    property_name = vary_single["property_varied"]
    return base_params, ev_prop, property_list, property_name

def plot_single_ev_prop(folder, real_data, dpi=200):
    # Load data
    base_params, ev_prop, property_list, property_name = load_ev_data(folder)
    
    # NEW: Define start step and base date
    start_step = 180
    base_date = datetime(2001, 1, 1)
    
    # Setup Figure
    plt.figure(figsize=(10, 6))
    
    # 1. Handle Time/Date logic for simulation
    # Slice the data from step 180 onwards
    total_steps = ev_prop.shape[2]
    time_indices = np.arange(start_step, total_steps)
    
    # Convert steps to datetime objects (1 step = 1 month approx)
    dates = [base_date + timedelta(days=30.44 * (step - start_step)) for step in time_indices]

    # 2. Process Simulation Data
    num_deltas = ev_prop.shape[0]
    num_seeds = ev_prop.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_deltas))

    for i in range(num_deltas):
        # Slice simulation data to start at step 180
        data_from_start = ev_prop[i, :, start_step:]
        mean_data = np.mean(data_from_start, axis=0)
        
        # Calculate 95% Confidence Interval
        sem_data = stats.sem(data_from_start, axis=0)
        ci_range = sem_data * stats.t.ppf(0.975, num_seeds - 1)

        # --- UPDATED LEGEND LOGIC ---
        # Multiplies the value by 3000 and formats it
        market_size = property_list[i] * 3000
        label_name = f"Used Car Market Size: {market_size:,.0f}"
        
        # Plot mean and CI ribbon using 'dates' as X
        plt.plot(dates, mean_data, label=label_name, color=colors[i])
        plt.fill_between(dates, 
                         mean_data - ci_range, mean_data + ci_range,
                         color=colors[i], alpha=0.2)

    # 3. Handle Real-World Data (Optional: Adjusting to match date axis)
    # If real_data starts at a different date, you'll need to create a specific 
    # date range for it. Assuming it aligns with the calibration logic:
    # (Leaving this as standard steps or you can convert time_steps_real to dates too)

    # Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Tick every 2 years for clarity
    
    plt.xlabel("Year")
    plt.ylabel("EV Uptake Proportion")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Tilt dates
    plt.gcf().autofmt_xdate()
    
    # Save and Show
    save_path = f"{folder}/Plots/ev_prop_single_plot_dated.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()

def main():
    # Update this path to your specific results folder
    target_folder = "results/sen_vary_max_num_cars_prop_00_13_31__17_04_2026"
    
    real_data = load_object("package/calibration_data", "calibration_data_output")["EV Prop"]
    
    plot_single_ev_prop(target_folder, real_data)

if __name__ == "__main__":
    main()