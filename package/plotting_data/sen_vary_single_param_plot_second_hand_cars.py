import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
    
    # Setup Figure
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Real-World Data
    burn_in = base_params["duration_burn_in"]
    init_index = burn_in + 120
    time_steps_real = np.arange(init_index, init_index + len(real_data) * 12, 12)
    plt.plot(time_steps_real, real_data, label="California Data 2010-23", 
             color='orange', linestyle="--", linewidth=2)

    # 2. Process Simulation Data
    num_deltas = ev_prop.shape[0]
    num_seeds = ev_prop.shape[1]
    time_series = np.arange(ev_prop.shape[2])
    colors = plt.cm.viridis(np.linspace(0, 1, num_deltas))

    for i in range(num_deltas):
        data_after_burn_in = ev_prop[i, :, burn_in:]
        mean_data = np.mean(data_after_burn_in, axis=0)
        
        # Calculate 95% Confidence Interval
        sem_data = stats.sem(data_after_burn_in, axis=0)
        ci_range = sem_data * stats.t.ppf(0.975, num_seeds - 1)

        label_name = f"{property_name} = {property_list[i]:.1e}"
        
        # Plot mean and CI ribbon
        plt.plot(time_series[burn_in:], mean_data, label=label_name, color=colors[i])
        plt.fill_between(time_series[burn_in:], 
                         mean_data - ci_range, mean_data + ci_range,
                         color=colors[i], alpha=0.2)

    # Formatting
    plt.xlabel("Time Step")
    plt.ylabel("EV Uptake Proportion")
    plt.title(f"Sensitivity Analysis: Varying {property_name}")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save and Show
    save_path = f"{folder}/Plots/ev_prop_single_plot.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()

def main():
    # Update this path to your specific results folder
    target_folder = "results/sen_vary_max_num_cars_prop_14_29_51__26_03_2026"
    
    real_data = load_object("package/calibration_data", "calibration_data_output")["EV Prop"]
    
    plot_single_ev_prop(target_folder, real_data)

if __name__ == "__main__":
    main()