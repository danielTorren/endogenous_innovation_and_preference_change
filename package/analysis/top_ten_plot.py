import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object


def plot_policy_results_ev(base_params, fileName, outputs_BAU, outputs, top_policies, x_label, y_label, prop_name, dpi=600):
    
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] - 1#-1 is because i cant keep track of tiem correnctly
    time_steps = np.arange(base_params["duration_future"])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Business-as-Usual (BAU) case in black
    bau_mean = np.mean(outputs_BAU[prop_name], axis=0)[start:]
    bau_ci = (sem(outputs_BAU[prop_name], axis=0) * t.ppf(0.975, df=outputs_BAU[prop_name].shape[0] - 1))[start:]
    ax.plot(time_steps, bau_mean, color='black', label='Business-as-Usual')
    ax.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        intensity_1 = top_policies[(policy1, policy2)]['policy1_value']
        intensity_2 = top_policies[(policy1, policy2)]['policy2_value']

        policy_label = f"{policy1} ({round(intensity_1, 3)}), {policy2} ({round(intensity_2, 3)})"
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


def plot_policy_results(fileName, outputs_BAU, outputs, top_policies, x_label, y_label, prop_name, dpi=600):

    time_steps = np.arange(outputs_BAU[prop_name].shape[1])
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Business-as-Usual (BAU) case in black
    bau_mean = np.mean(outputs_BAU[prop_name], axis=0)
    bau_ci = sem(outputs_BAU[prop_name], axis=0) * t.ppf(0.975, df=outputs_BAU[prop_name].shape[0] - 1)
    ax.plot(time_steps, bau_mean, color='black', label='Business-as-Usual')
    ax.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    # Plot each policy combination
    for (policy1, policy2), data in outputs.items():
        
        intensity_1 = top_policies[(policy1, policy2)]['policy1_value']
        intensity_2 = top_policies[(policy1, policy2)]['policy2_value']

        policy_label = f"{policy1} ({round(intensity_1, 3)}), {policy2} ({round(intensity_2, 3)})"
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


def main(fileName):
    base_params = load_object(fileName + "/Data", "base_params")
    base_params["duration_calibration"] = base_params["duration_no_carbon_price"]
    outputs_BAU = load_object(fileName + "/Data", "outputs_BAU")
    outputs = load_object(fileName + "/Data", "outputs")
    outputs = load_object(fileName + "/Data", "outputs")
    top_policies = load_object(fileName + "/Data", "top_policies")

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Net cost, $", 
        "history_policy_net_cost"
    )


    plot_policy_results_ev(
        base_params,
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "EV uptake proportion", 
        "history_prop_EV"
    )

    plot_policy_results(
        fileName, 
        outputs_BAU,
        outputs,
        top_policies,
        "Time Step, months",
        "Total Emissions, kgCO2",
        "history_total_emissions"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies, 
        "Time Step, months", 
        "Driving Emissions, kgCO2", 
        "history_driving_emissions"
        )
    
    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies, 
        "Time Step, months", 
        "Production Emissions, kgCO2", 
        "history_production_emissions"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Total Utility", 
        "history_total_utility"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Market Concentration, HHI", 
        "history_market_concentration"
    )

    plot_policy_results(
        fileName,
        outputs_BAU,
        outputs, 
        top_policies,
        "Time Step, months", 
        "Total Profit, $", 
        "history_total_profit"
    )




    plt.show()

if __name__ == "__main__":
    main("results/top_ten_10_45_15__12_03_2025")
