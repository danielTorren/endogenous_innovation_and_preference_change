import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from package.resources.utility import load_object

policy_titles = {
    "BAU": "BAU",
    "Carbon_price": "Carbon Price",
    "Electricity_subsidy": "Electricity Subsidy",
    "Adoption_subsidy": "New Car Rebate",
    "Adoption_subsidy_used": "Used Car Rebate",
    "Production_subsidy": "Production Subsidy",
}

MEASURES = {
    "EV Uptake": {
        "mean_key": "mean_EV_uptake",
        "array_key": "ev_uptake",
        "sd_key": "sd_ev_uptake",
        "scale": 1,
        "ylabel": "EV Adoption Proportion",
    },
    "Net Cost": {
        "mean_key": "mean_net_cost",
        "array_key": "net_cost",
        "sd_key": None,
        "scale": 1e-9,
        "ylabel": "Cumulative Net Cost, bn $",
    },
    "Cumulative Emissions": {
        "mean_key": "mean_emissions_cumulative",
        "array_key": ("emissions_cumulative_driving", "emissions_cumulative_production"),
        "sd_key": None,
        "scale": 1e-9,
        "ylabel": "Cumulative Emissions, MTCO2",
    },
    "Cumulative Utility": {
        "mean_key": "mean_utility_cumulative",
        "array_key": "utility_cumulative",
        "sd_key": None,
        "scale": 1e-9,
        "ylabel": "Cumulative Utility, bn $",
    },
    "Cumulative Profit": {
        "mean_key": "mean_profit_cumulative",
        "array_key": "profit_cumulative",
        "sd_key": None,
        "scale": 1e-9,
        "ylabel": "Cumulative Profit, bn $",
    },
}


def print_base_params(base_params):
    print("=" * 80)
    print("BASE PARAMS")
    print("=" * 80)
    print(json.dumps(base_params, indent=2, default=str))
    print("=" * 80)


def get_mean_ci(entry, measure, seed_repetitions):
    """
    Mean + 95% CI for one policy entry and one measure. Falls back from the
    per-seed array (present for optimized policies) to sd_ev_uptake (present
    for BAU's EV uptake only) to a zero-width CI when neither is available.
    """
    scale = measure["scale"]
    array_key = measure["array_key"]

    if isinstance(array_key, tuple):
        if all(k in entry for k in array_key):
            arr = sum(np.asarray(entry[k]) for k in array_key)
        else:
            arr = None
    else:
        arr = np.asarray(entry[array_key]) if array_key in entry else None

    if arr is not None:
        arr = arr * scale
        n = len(arr)
        return np.mean(arr), 1.96 * np.std(arr) / np.sqrt(n)

    mean = entry[measure["mean_key"]] * scale
    sd_key = measure["sd_key"]
    if sd_key is not None and sd_key in entry:
        ci = 1.96 * (entry[sd_key] * scale) / np.sqrt(seed_repetitions)
    else:
        ci = 0.0
    return mean, ci


def plot_policy_outcomes_bar(base_params, policy_outcomes, conditions, file_name, dpi=300):
    policy_list = ["BAU"] + list(conditions["policy_list"])
    seed_repetitions = base_params["seed_repetitions"]
    selected_measures = list(MEASURES.keys())

    okabe_ito_colors = ['#000000', '#E69F00', '#009E73', '#56B4E9', '#F0E442',
                         '#0072B2', '#D55E00', '#CC79A7']
    color_map = ListedColormap(okabe_ito_colors)
    policy_colors = {p: color_map(i) for i, p in enumerate(policy_list)}

    fig, axes = plt.subplots(1, len(selected_measures), figsize=(4 * len(selected_measures), 5))

    for ax, measure_name in zip(axes, selected_measures):
        measure = MEASURES[measure_name]
        means, cis, colors, labels = [], [], [], []
        for policy in policy_list:
            entry = policy_outcomes[policy]
            mean, ci = get_mean_ci(entry, measure, seed_repetitions)
            means.append(mean)
            cis.append(ci)
            colors.append(policy_colors[policy])
            labels.append(policy_titles.get(policy, policy))

        x = np.arange(len(policy_list))
        ax.bar(x, means, yerr=cis, capsize=4, color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(measure["ylabel"], fontsize=11)
        ax.set_title(measure_name, fontsize=12)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(f"{file_name}/Plots/endogenous_policy_intensity_single", exist_ok=True)
    fig.savefig(f"{file_name}/Plots/endogenous_policy_intensity_single/policy_outcomes_bar.png", dpi=dpi)


def plot_optimized_intensity(base_params, policy_outcomes, conditions, file_name, dpi=300):
    policy_list = list(conditions["policy_list"])

    okabe_ito_colors = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
                         '#0072B2', '#D55E00', '#CC79A7', '#000000']
    color_map = ListedColormap(okabe_ito_colors)

    fig, axes = plt.subplots(1, len(policy_list), figsize=(3.5 * len(policy_list), 4.5))
    if len(policy_list) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policy_list):
        intensity = policy_outcomes[policy]["optimized_intensity"]
        ax.bar([policy_titles.get(policy, policy)], [intensity], color=color_map(policy_list.index(policy)), edgecolor="black")
        ax.set_title(policy_titles.get(policy, policy), fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis='x', labelrotation=0)

    axes[0].set_ylabel("Optimized Intensity\n(target EV uptake = {:.0%})".format(conditions["target_ev_uptake"]), fontsize=11)

    fig.tight_layout()
    os.makedirs(f"{file_name}/Plots/endogenous_policy_intensity_single", exist_ok=True)
    fig.savefig(f"{file_name}/Plots/endogenous_policy_intensity_single/optimized_intensity_bar.png", dpi=dpi)


def print_policy_outcomes(policy_outcomes, conditions):
    print("=" * 80)
    print("CONDITIONS")
    print("=" * 80)
    print(json.dumps(conditions, indent=2, default=str))

    print("=" * 80)
    print("POLICY OUTCOMES")
    print("=" * 80)
    for policy, entry in policy_outcomes.items():
        print(f"\n--- {policy_titles.get(policy, policy)} ---")
        for key, value in entry.items():
            if hasattr(value, "__len__") and not isinstance(value, str):
                continue
            print(f"  {key}: {value}")


def main(file_name):
    base_params = load_object(file_name + "/Data", "base_params")
    policy_outcomes = load_object(file_name + "/Data", "policy_outcomes")
    conditions = load_object(file_name + "/Data", "conditions")

    print_base_params(base_params)
    print_policy_outcomes(policy_outcomes, conditions)

    plot_policy_outcomes_bar(base_params, policy_outcomes, conditions, file_name, dpi=300)
    plot_optimized_intensity(base_params, policy_outcomes, conditions, file_name, dpi=300)

    plt.show()


if __name__ == "__main__":
    main(file_name="results/endog_single_19_34_26__11_08_2026")
