import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object


def plot_policy_results_ev(base_params, fileName, outputs, x_label, y_label, prop_name, dpi=600):
    
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] - 1#-1 is because i cant keep track of tiem correnctly
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        policy_label = f"Gas multiplier {policy1}, Electricity multiplier {policy2}"

        mean_values = np.mean(data[prop_name], axis=0)[start:]
        ci_values = (sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1))[start:]
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_path = f'{fileName}/Plots/{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)
    print("Done", prop_name)


def plot_policy_results(base_params,fileName,  outputs,  x_label, y_label, prop_name, dpi=600):

    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))


    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        policy_label = f"Gas multiplier {policy1}, Electricity multiplier {policy2}"

        mean_values = np.mean(data[prop_name], axis=0)
        ci_values = sem(data[prop_name], axis=0) * t.ppf(0.975, df=data[prop_name].shape[0] - 1)
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_path = f'{fileName}/Plots/{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)

    print("Done", prop_name)

def plot_policy_results_cum(base_params,fileName,  outputs,  x_label, y_label, prop_name, dpi=600):

    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))


    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        data_measure = np.cumsum(data[prop_name], axis = 1)
        policy_label = f"Gas multiplier {policy1}, Electricity multiplier {policy2}"
        mean_values = np.mean(data_measure, axis=0)
        ci_values = sem(data_measure, axis=0) * t.ppf(0.975, df=data_measure.shape[0] - 1)
        ax.plot(time_steps, mean_values, label=policy_label)
        ax.fill_between(time_steps, mean_values - ci_values, mean_values + ci_values, alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_path = f'{fileName}/Plots/cum_{prop_name}.png'
    plt.savefig(save_path, dpi=dpi)

    print("Done", prop_name)

def main(fileName):
    base_params = load_object(fileName + "/Data", "base_params")
    #base_params["duration_calibration"] = base_params["duration_no_carbon_price"]
    outputs = load_object(fileName + "/Data", "outputs")

    plot_policy_results_cum(
        base_params,
        fileName, 
        outputs,
        "Time Step, months",
        "Emissions Cumulative, kgCO2",
        "history_total_emissions"
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Net cost, $", 
        "history_policy_net_cost"
    )


    plot_policy_results_ev(
        base_params,
        fileName,

        outputs, 
        "Time Step, months", 
        "EV uptake proportion", 
        "history_prop_EV"
    )

    plot_policy_results(
        base_params,
        fileName, 

        outputs,
        "Time Step, months",
        "Total Emissions, kgCO2",
        "history_total_emissions"
    )


    plot_policy_results(
        base_params,
        fileName,

        outputs, 
        "Time Step, months", 
        "Driving Emissions, kgCO2", 
        "history_driving_emissions"
        )
    
    plot_policy_results(
        base_params,
        fileName,

        outputs, 
        "Time Step, months", 
        "Production Emissions, kgCO2", 
        "history_production_emissions"
    )

    plot_policy_results(
        base_params,
        fileName,

        outputs, 
        "Time Step, months", 
        "Total Utility", 
        "history_total_utility"
    )

    plot_policy_results(
        base_params,
        fileName,

        outputs, 
        "Time Step, months", 
        "Market Concentration, HHI", 
        "history_market_concentration"
    )

    plot_policy_results(
        base_params,
        fileName,
        outputs, 
        "Time Step, months", 
        "Total Profit, $", 
        "history_total_profit"
    )




    plt.show()

if __name__ == "__main__":
    main("results/sceanrio_tests_gen_13_36_46__12_03_2025")
