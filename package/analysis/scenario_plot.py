import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import os
import matplotlib.cm as cm

def plot_policy_results_ev_triple(base_params, fileName, outputs, x_label, y_label, prop_name, property_dicts, dpi=300):
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] - 1
    time_steps = np.arange(base_params["duration_future"])

    # Extract the unique values
    gas_prices = sorted(set(k[0] for k in outputs.keys()))
    electricity_prices = sorted(set(k[1] for k in outputs.keys()))
    emissions_intensities = sorted(set(k[2] for k in outputs.keys()))

    num_elec_prices = len(electricity_prices)
    fig, axs = plt.subplots(1, num_elec_prices, figsize=(10,6), sharex=True, sharey=True)

    # Visual mappings
    color_map = plt.get_cmap('Set1', 10)
    gas_color_map = {val: color_map(i) for i, val in enumerate(gas_prices)}

    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*']
    ei_marker_map = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(emissions_intensities)}

    for idx, elec_price in enumerate(electricity_prices):
        ax = axs[idx] if num_elec_prices > 1 else axs  # Handle case with only one electricity price

        for (gas_price, electricity_price, emissions_intensity), data in outputs.items():
            if electricity_price != elec_price:
                continue  # Skip if this subplot is for a different electricity price

            mean_values = np.mean(data[prop_name], axis=0)[start:]
            ci_values = (sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1))[start:]

            color = gas_color_map[gas_price]
            marker = ei_marker_map[emissions_intensity]
            linestyle = '-'  # Fixed style for clarity

            label = f"Gas {gas_price}, EI {emissions_intensity}"

            ax.plot(time_steps, mean_values, linestyle=linestyle, color=color, marker=marker, markevery=32,
                    markerfacecolor="black", markeredgecolor=color, markersize=8, label=label)
            ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.2, color=color)

        ax.set_title(f"Electricity Price: {elec_price}", fontsize=14)

        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True)

    axs[0].set_ylabel(y_label)
    fig.supxlabel(x_label)
    plt.tight_layout()

    os.makedirs(f'{fileName}/Plots', exist_ok=True)
    save_path = f'{fileName}/Plots/{prop_name}_by_electricity_price_triple.png'
    plt.savefig(save_path, dpi=dpi)
    print("Saved plot to", save_path)

def main(fileName):
    # Load data
    base_params = load_object(fileName + "/Data", "base_params")
    if "duration_calibration" not in base_params:
        base_params["duration_calibration"] = base_params["duration_no_carbon_price"]

    outputs = load_object(fileName + "/Data", "outputs")
    
    # Load property dictionaries
    property_dicts = []
    i = 1
    while True:
        try:
            prop_dict = load_object(fileName + "/Data", f"property_dict_{i}")
            property_dicts.append(prop_dict)
            i += 1
        except FileNotFoundError:
            break
    
    print(f"Loaded {len(property_dicts)} property dictionaries")

    plot_policy_results_ev_triple(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "EV uptake proportion", 
        "history_prop_EV",
        property_dicts
    )
    plt.show()


if __name__ == "__main__":
    main("results/scenario_gen_17_06_21__10_04_2025")