import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import save_object, load_object
from package.plotting_data.single_experiment_plot import save_and_show

def add_vertical_lines(ax, base_params, color='black', linestyle='--'):
    burn_in = base_params["duration_burn_in"]
    no_carbon_price = base_params["duration_no_carbon_price"]
    ev_production_start_time = base_params["ev_production_start_time"]

    ax.axvline(burn_in + ev_production_start_time, color="red", linestyle=':', label="EV sale start")

    if base_params.get("EV_rebate_state", False):
        rebate_start_time = burn_in + base_params["parameters_rebate_calibration"]["start_time"]
        ax.axvline(rebate_start_time, color="red", linestyle='-.', label="EV adoption subsidy start")
    
    if base_params["duration_future"] > 0:
        policy_start_time = burn_in + no_carbon_price
        ax.axvline(policy_start_time, color="red", linestyle='--', label="Policy start")


def plot_all_policies_time_series(outputs, base_params, fileName, variable_key, y_label, save_name, dpi=600):
    """
    General function to plot a time series variable for all top 10 policy combinations across multiple seeds.
    Each subplot is a different policy combination.
    """
    burn_in_step = base_params["duration_burn_in"]
    time_steps = np.arange(burn_in_step, burn_in_step + base_params["duration_no_carbon_price"] + base_params["duration_future"])
    time_steps = np.arange(burn_in_step,  burn_in_step + base_params["duration_no_carbon_price"] + base_params["duration_future"])
   

    num_policies = len(outputs)
    fig, axes = plt.subplots(5, 2, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    policy_colors = plt.cm.get_cmap('tab10', num_policies)

    for idx, ((policy1, policy2), data) in enumerate(outputs.items()):
        ax = axes[idx]

        history = np.asarray(data[variable_key])  # shape: [seeds, time_steps]
        data_after_burn_in = history[:, burn_in_step:]

        mean_values = np.mean(data_after_burn_in, axis=0)
        median_values = np.median(data_after_burn_in, axis=0)
        ci_range = sem(data_after_burn_in, axis=0) * t.ppf(0.975, df=data_after_burn_in.shape[0] - 1)

        color = policy_colors(idx)

        for seed_data in data_after_burn_in:
            ax.plot(time_steps, seed_data, color=color, alpha=0.3, linewidth=0.8)

        ax.plot(time_steps, mean_values, label='Mean', color=color, linewidth=2)
        ax.plot(time_steps, median_values, label='Median', color=color, linestyle="dashed", linewidth=1)
        ax.fill_between(
            time_steps,
            mean_values - ci_range,
            mean_values + ci_range,
            color=color,
            alpha=0.2,
            label='95% CI'
        )

        add_vertical_lines(ax, base_params)

        policy1_value = data.get("policy1_value", "N/A")
        policy2_value = data.get("policy2_value", "N/A")
        ax.set_title(f"{policy1}={policy1_value}, {policy2}={policy2_value}")

        ax.set_xlabel("Time Step (months)")
        ax.set_ylabel(y_label)

        if idx == 0:
            ax.legend()

    plt.tight_layout()
    save_and_show(fig, fileName, save_name, dpi)


def plot_ev_uptake_top10_policies(outputs, base_params, fileName, dpi=600):
    plot_all_policies_time_series(
        outputs, base_params, fileName,
        variable_key="history_prop_EV",
        y_label="Proportion of EVs",
        save_name="ev_uptake_top10_policies",
        dpi=dpi
    )


def plot_emissions_top10_policies(outputs, base_params, fileName, dpi=600):
    plot_all_policies_time_series(
        outputs, base_params, fileName,
        variable_key="history_total_emissions",
        y_label="Total Emissions (kg CO2)",
        save_name="emissions_top10_policies",
        dpi=dpi
    )


def plot_profit_top10_policies(outputs, base_params, fileName, dpi=600):
    plot_all_policies_time_series(
        outputs, base_params, fileName,
        variable_key="history_total_profit",
        y_label="Total Profit ($)",
        save_name="profit_top10_policies",
        dpi=dpi
    )


def plot_margins_top10_policies(outputs, base_params, fileName, dpi=600):
    burn_in_step = base_params["duration_burn_in"]
    time_steps = np.arange(burn_in_step, burn_in_step + base_params["duration_no_carbon_price"] + base_params["duration_future"])

    fig, axes = plt.subplots(5, 2, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    policy_colors = plt.cm.get_cmap('tab10', len(outputs))

    for idx, ((policy1, policy2), data) in enumerate(outputs.items()):
        ax = axes[idx]
        color = policy_colors(idx)

        margins_ICE = np.asarray(data["history_mean_profit_margins_ICE"])[:, burn_in_step:]
        margins_EV = np.asarray(data["history_mean_profit_margins_EV"])[:, burn_in_step:]

        mean_ICE = np.mean(margins_ICE, axis=0)
        mean_EV = np.mean(margins_EV, axis=0)

        ci_ICE = sem(margins_ICE, axis=0) * t.ppf(0.975, df=margins_ICE.shape[0] - 1)
        ci_EV = sem(margins_EV, axis=0) * t.ppf(0.975, df=margins_EV.shape[0] - 1)

        # Plot ICE profit margins
        for seed_data in margins_ICE:
            ax.plot(time_steps, seed_data, color=color, alpha=0.3, linewidth=0.8)

        ax.plot(time_steps, mean_ICE, label=f'ICE Mean', color=color, linewidth=2)
        ax.fill_between(time_steps, mean_ICE - ci_ICE, mean_ICE + ci_ICE, color=color, alpha=0.2)

        # Plot EV profit margins (shift color a bit darker)
        darker_color = tuple(c * 0.6 for c in color)
        for seed_data in margins_EV:
            ax.plot(time_steps, seed_data, color=darker_color, alpha=0.3, linewidth=0.8)

        ax.plot(time_steps, mean_EV, label=f'EV Mean', color=darker_color, linewidth=2, linestyle="dashed")
        ax.fill_between(time_steps, mean_EV - ci_EV, mean_EV + ci_EV, color=darker_color, alpha=0.2)

        add_vertical_lines(ax, base_params)

        policy1_value = data.get("policy1_value", "N/A")
        policy2_value = data.get("policy2_value", "N/A")
        ax.set_title(f"{policy1}={policy1_value}, {policy2}={policy2_value}")

        ax.set_xlabel("Time Step (months)")
        ax.set_ylabel("Profit Margin")

        if idx == 0:
            ax.legend()

    plt.tight_layout()
    save_and_show(fig, fileName, "profit_margins_top10_policies", dpi)


def main(fileName, dpi=300):
    """
    Master plotter function for Top 10 Policy Combination Results.
    """
    base_params = load_object(fileName + "/Data", "base_params")
    outputs = load_object(fileName + "/Data", "outputs")

    # EV Uptake Plot
    plot_ev_uptake_top10_policies(outputs, base_params, fileName, dpi)

    # Emissions Plot
    plot_emissions_top10_policies(outputs, base_params, fileName, dpi)

    # Total Profit Plot
    plot_profit_top10_policies(outputs, base_params, fileName, dpi)

    # Profit Margins Plot (dual ICE and EV per policy)
    plot_margins_top10_policies(outputs, base_params, fileName, dpi)


if __name__ == "__main__":
    main("results/endogenous_policy_intensity_19_30_46__06_03_2025")
