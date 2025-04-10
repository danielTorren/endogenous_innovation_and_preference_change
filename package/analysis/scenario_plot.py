import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import os
import matplotlib.cm as cm

def plot_policy_results_ev(base_params, fileName, outputs, x_label, y_label, prop_name, property_dicts, dpi=300):
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] - 1
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract the unique values
    gas_prices = sorted(set(k[0] for k in outputs.keys()))
    electricity_prices = sorted(set(k[1] for k in outputs.keys()))
    emissions_intensities = sorted(set(k[2] for k in outputs.keys()))

    # Set up visual mappings
    color_map = cm.get_cmap("viridis", len(gas_prices))
    gas_color_map = {val: color_map(i) for i, val in enumerate(gas_prices)}

    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*']
    elec_marker_map = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(electricity_prices)}

    line_styles = ['-', '--', '-.', ':']
    emissions_style_map = {val: line_styles[i % len(line_styles)] for i, val in enumerate(emissions_intensities)}

    # Plot each policy combination
    for (gas_price, electricity_price, emissions_intensity), data in outputs.items():
        mean_values = np.mean(data[prop_name], axis=0)[start:]
        ci_values = (sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1))[start:]

        # Visual style
        color = gas_color_map[gas_price]
        marker = elec_marker_map[electricity_price]
        linestyle = emissions_style_map[emissions_intensity]

        label = f"Gas {gas_price}, Elec {electricity_price}, EI {emissions_intensity}"

        # Plot line and scatter
        ax.plot(time_steps, mean_values, label=label, color=color, linestyle=linestyle)
        ax.scatter(time_steps, mean_values, color=color, marker=marker, s=20)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.2, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()

    os.makedirs(f'{fileName}/Plots', exist_ok=True)
    save_path = f'{fileName}/Plots/{prop_name}.png'
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

    plot_policy_results_ev(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "EV uptake proportion", 
        "history_prop_EV",
        property_dicts
    )


if __name__ == "__main__":
    main("results/scenario_tests_gen_13_36_46__12_03_2025")