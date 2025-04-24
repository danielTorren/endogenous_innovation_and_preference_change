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
    fig, axs = plt.subplots(2, num_elec_prices, figsize=(5 * num_elec_prices, 10), sharex=True)

    # Visual mappings
    color_map = plt.get_cmap('Set1', 10)
    gas_color_map = {val: color_map(i) for i, val in enumerate(gas_prices)}

    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*']
    ei_marker_map = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(emissions_intensities)}

    for idx, elec_price in enumerate(electricity_prices):
        ax_ev = axs[0, idx] if num_elec_prices > 1 else axs[0]
        ax_emissions = axs[1, idx] if num_elec_prices > 1 else axs[1]

        for (gas_price, electricity_price, emissions_intensity), data in outputs.items():
            if electricity_price != elec_price:
                continue

            color = gas_color_map[gas_price]
            marker = ei_marker_map[emissions_intensity]
            linestyle = '-'
            label = f"Gas {gas_price}, EI {emissions_intensity}"

            # Plot EV Adoption
            ev_mean = np.mean(data["history_prop_EV"], axis=0)[start:]
            ev_ci = (sem(data["history_prop_EV"], axis=0) * t.ppf(0.975, df=data["history_prop_EV"].shape[0] - 1))[start:]

            ax_ev.plot(time_steps, ev_mean, linestyle=linestyle, color=color, marker=marker, markevery=32,
                       markerfacecolor="black", markeredgecolor=color, markersize=8, label=label)
            ax_ev.fill_between(time_steps, ev_mean - ev_ci, ev_mean + ev_ci, alpha=0.2, color=color)

            # Plot Total Emissions
            em_mean = np.mean(data["history_total_emissions"], axis=0)
            em_ci = (sem(data["history_total_emissions"], axis=0) * t.ppf(0.975, df=data["history_total_emissions"].shape[0] - 1))

            ax_emissions.plot(time_steps[:-1], em_mean, linestyle=linestyle, color=color, marker=marker, markevery=32,
                              markerfacecolor="black", markeredgecolor=color, markersize=8, label=label)
            ax_emissions.fill_between(time_steps[:-1], em_mean - em_ci, em_mean + em_ci, alpha=0.2, color=color)

        ax_ev.set_title(f"Electricity Price: {elec_price}", fontsize=14)
        ax_ev.legend(loc='upper left', fontsize=9)
        ax_ev.grid(True)

        ax_emissions.set_xlabel(x_label)
        ax_emissions.grid(True)

    axs[0, 0].set_ylabel("EV Adoption Proportion")
    axs[1, 0].set_ylabel("Total Emissions")
    fig.suptitle("EV Adoption and Emissions by Electricity Price", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(f'{fileName}/Plots', exist_ok=True)
    save_path = f'{fileName}/Plots/ev_and_emissions_by_electricity_price_triple.png'
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
        "EV Adoption Proportion", 
        "history_prop_EV",
        property_dicts
    )
    plt.show()


if __name__ == "__main__":
    main("results/scenario_gen_17_06_21__10_04_2025")
