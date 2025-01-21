from package.resources.utility import (
    load_object
)
import matplotlib.pyplot as plt

from package.plotting_data.single_experiment_plot import save_and_show

def plot_total_cost_all_policies(runs_data, fileName, dpi = 600):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy_name, data in runs_data.items():
        iterations = range(1, len(data) + 1)
        total_costs = [entry[2] for entry in data]
        ax.plot(iterations, total_costs, marker='o', label=f"{policy_name}")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Cost Over Iterations for All Policies")
    ax.grid()
    ax.legend()
    save_and_show(fig, fileName, "plot_total_cost_all_policies", dpi)


def plot_intensity_levels_all_policies(runs_data, fileName, dpi = 600):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy_name, data in runs_data.items():
        iterations = range(1, len(data) + 1)
        intensities = [entry[6] for entry in data]
        ax.plot(iterations, intensities, marker='o', label=f"{policy_name}")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Intensity Level")
    ax.set_title("Policy Intensity Levels Over Iterations for All Policies")
    ax.grid()
    ax.legend()
    save_and_show(fig, fileName, "plot_intensity_levels_all_policies", dpi)


def plot_intensity_levels_excluding_adoption_subsidy(runs_data, fileName, dpi=600):
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    ax1,ax2 = axes[0], axes[1]
    for policy_name, data in runs_data.items():
        iterations = range(1, len(data) + 1)
        intensities = [entry[6] for entry in data]
        
        if policy_name == "Adoption_subsidy":
            ax2.plot(iterations, intensities, marker='o', label=f"{policy_name}")
        else:
            ax1.plot(iterations, intensities, marker='o', label=f"{policy_name}")
    
    # Top subplot for all policies except "Adoption Subsidy"
    ax1.set_ylabel("Intensity Level")
    ax1.set_title("Policy Intensity Levels (Excluding Adoption Subsidy)")
    ax1.grid()
    ax1.legend()
    
    # Bottom subplot for "Adoption Subsidy"
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Intensity Level")
    ax2.set_title("Policy Intensity Levels for Adoption Subsidy")
    ax2.grid()
    ax2.legend()
    
    # Save and show the figure
    save_and_show(fig, fileName, "plot_intensity_levels_excluding_adoption_subsidy", dpi)


def plot_ev_uptake_with_confidence_all_policies(runs_data, fileName, dpi = 600):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy_name, data in runs_data.items():
        iterations = range(1, len(data) + 1)
        ev_uptakes = [entry[1] for entry in data]
        lower_conf = [entry[3][0] for entry in data]
        upper_conf = [entry[3][1] for entry in data]
        
        ax.plot(iterations, ev_uptakes, marker='o', label=f"{policy_name}")
        ax.fill_between(iterations, lower_conf, upper_conf, alpha=0.2, label=f"{policy_name} CI")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("EV Uptake")
    ax.set_title("EV Uptake with Confidence Intervals for All Policies")
    ax.grid()
    ax.legend()
    save_and_show(fig, fileName, "plot_ev_uptake_with_confidence_all_policies", dpi)


def plot_policy_summary_all(runs_data, fileName):

    plot_total_cost_all_policies(runs_data, fileName)
    plot_intensity_levels_all_policies(runs_data, fileName)
    plot_ev_uptake_with_confidence_all_policies(runs_data, fileName)
    plot_intensity_levels_excluding_adoption_subsidy(runs_data, fileName)

def main(fileName):
    # Load observed data

    base_params = load_object(fileName + "/Data", "base_params")
    policy_outcomes = load_object(fileName + "/Data", "policy_outcomes")
    runs_data = load_object(fileName + "/Data", "runs_data")
    
    plot_policy_summary_all(runs_data, fileName)

    print("policy_outcomes", policy_outcomes)

    plt.show()

if __name__ == "__main__":
    main(
        fileName="results/endogenous_policy_intensity_single_13_05_42__19_01_2025",
    )
