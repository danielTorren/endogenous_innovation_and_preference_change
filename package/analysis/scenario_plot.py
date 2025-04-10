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


def plot_policy_results(base_params, fileName, outputs, x_label, y_label, prop_name, property_dicts, dpi=300):
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Generate labels for each parameter
    param_labels = [d["property_varied"] for d in property_dicts]

    # Plot each policy combination
    for policy_combo, data in outputs.items():
        # Create label combining all policy parameters
        policy_label = ", ".join([f"{param_labels[i]} {val}" for i, val in enumerate(policy_combo)])
        
        mean_values = np.mean(data[prop_name], axis=0)
        ci_values = sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1)
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


def plot_policy_results_cum(base_params, fileName, outputs, x_label, y_label, prop_name, property_dicts, dpi=300):
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Generate labels for each parameter
    param_labels = [d["property_varied"] for d in property_dicts]

    # Plot each policy combination
    for policy_combo, data in outputs.items():
        data_measure = np.cumsum(data[prop_name], axis=1)
        
        # Create label combining all policy parameters
        policy_label = ", ".join([f"{param_labels[i]} {val}" for i, val in enumerate(policy_combo)])
        
        mean_values = np.mean(data_measure, axis=0)
        ci_values = sem(data_measure, axis=0) * t.ppf(0.975, df=data_measure.shape[0] - 1)
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    
    # Ensure the Plots directory exists
    os.makedirs(f'{fileName}/Plots', exist_ok=True)
    save_path = f'{fileName}/Plots/cum_{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)
    print("Done", prop_name)


def plot_grid_search_heatmap(base_params, fileName, outputs, prop_name, property_dicts, 
                             time_index=-1, aggregator=np.mean, dpi=300):
    """
    Create heatmaps for different values of the third parameter.
    
    Args:
        time_index: Which time step to plot (-1 for final step)
        aggregator: Function to aggregate over seeds (default: mean)
    """
    # Extract unique values for each parameter
    unique_values = []
    for prop_dict in property_dicts:
        unique_values.append(prop_dict["property_list"])
    
    # Create heatmaps for each value of the third parameter
    for z_value in unique_values[2]:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix to hold values
        matrix = np.zeros((len(unique_values[0]), len(unique_values[1])))
        
        # Fill matrix with values
        for policy_combo, data in outputs.items():
            if policy_combo[2] == z_value:  # Only include points with the current z value
                # Find indices for x and y values
                x_idx = unique_values[0].index(policy_combo[0])
                y_idx = unique_values[1].index(policy_combo[1])
                
                # Calculate value (mean across seeds at specified time)
                values = data[prop_name][:, time_index]
                matrix[x_idx, y_idx] = aggregator(values)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='viridis', origin='lower')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(prop_name)
        
        # Set labels
        ax.set_xlabel(property_dicts[1]["property_varied"])
        ax.set_ylabel(property_dicts[0]["property_varied"])
        ax.set_title(f'{property_dicts[2]["property_varied"]} = {z_value}')
        
        # Set ticks
        ax.set_xticks(np.arange(len(unique_values[1])))
        ax.set_yticks(np.arange(len(unique_values[0])))
        ax.set_xticklabels(unique_values[1])
        ax.set_yticklabels(unique_values[0])
        
        # Ensure the Plots directory exists
        os.makedirs(f'{fileName}/Plots', exist_ok=True)
        save_path = f'{fileName}/Plots/heatmap_{prop_name}_{property_dicts[2]["property_varied"]}_{z_value}.png'
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)
    
    print(f"Done heatmaps for {prop_name}")


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
    
    # Regular line plots
    plot_policy_results_cum(
        base_params,
        fileName, 
        outputs,
        "Time Step, months",
        "Emissions Cumulative, kgCO2",
        "history_total_emissions",
        property_dicts
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Net cost, $", 
        "history_policy_net_cost",
        property_dicts
    )

    plot_policy_results_ev(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "EV uptake proportion", 
        "history_prop_EV",
        property_dicts
    )

    plot_policy_results(
        base_params,
        fileName, 
        outputs,
        "Time Step, months",
        "Total Emissions, kgCO2",
        "history_total_emissions",
        property_dicts
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Driving Emissions, kgCO2", 
        "history_driving_emissions",
        property_dicts
    )
    
    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Production Emissions, kgCO2", 
        "history_production_emissions",
        property_dicts
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Total Utility", 
        "history_total_utility",
        property_dicts
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Market Concentration, HHI", 
        "history_market_concentration",
        property_dicts
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Total Profit, $", 
        "history_total_profit",
        property_dicts
    )
    
    # Heatmap plots for key metrics
    if len(property_dicts) >= 3:  # Only create heatmaps if we have at least 3 dimensions
        plot_grid_search_heatmap(
            base_params,
            fileName,
            outputs,
            "history_total_emissions",
            property_dicts,
            time_index=-1  # Final time step
        )
        
        plot_grid_search_heatmap(
            base_params,
            fileName,
            outputs,
            "history_prop_EV",
            property_dicts,
            time_index=-1  # Final time step
        )
        
        # Cumulative emissions
        def cum_sum_final(arr):
            return np.mean(np.sum(arr, axis=1))
        
        plot_grid_search_heatmap(
            base_params,
            fileName,
            outputs,
            "history_total_emissions",
            property_dicts,
            aggregator=cum_sum_final
        )

    print("All plots completed")


if __name__ == "__main__":
    main("results/scenario_tests_gen_13_36_46__12_03_2025")