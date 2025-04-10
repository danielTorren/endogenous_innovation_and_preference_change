import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import os


def plot_policy_results_ev(base_params, fileName, outputs, x_label, y_label, prop_name, property_dicts, dpi=300):
    
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] - 1  # -1 is because i cant keep track of time correctly
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Generate labels for each parameter
    param_labels = [d["property_varied"] for d in property_dicts]

    # Plot each policy combination
    for policy_combo, data in outputs.items():
        # Create label combining all policy parameters
        policy_label = ", ".join([f"{param_labels[i]} {val}" for i, val in enumerate(policy_combo)])
        
        mean_values = np.mean(data[prop_name], axis=0)[start:]
        ci_values = (sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1))[start:]
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    
    # Ensure the Plots directory exists
    os.makedirs(f'{fileName}/Plots', exist_ok=True)
    save_path = f'{fileName}/Plots/{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)
    print("Done", prop_name)


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