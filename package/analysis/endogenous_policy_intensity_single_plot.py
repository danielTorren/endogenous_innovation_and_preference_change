from package.resources.utility import (
    load_object
)
import matplotlib.pyplot as plt


def plot_ev_uptake_all_policies(runs_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy_name, data in runs_data.items():
        iterations = range(1, len(data) + 1)
        ev_uptakes = [entry[1] for entry in data]
        ax.plot(iterations, ev_uptakes, marker='o', label=f"{policy_name}")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("EV Uptake")
    ax.set_title("EV Uptake Over Iterations for All Policies")
    ax.grid()
    ax.legend()




def plot_total_cost_all_policies(runs_data):
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


  

def plot_intensity_levels_all_policies(runs_data):
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



def plot_ev_uptake_with_confidence_all_policies(runs_data):
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




def plot_policy_summary_all(runs_data):
    plot_ev_uptake_all_policies(runs_data)
    plot_total_cost_all_policies(runs_data)
    plot_intensity_levels_all_policies(runs_data)
    plot_ev_uptake_with_confidence_all_policies(runs_data)



def main(fileName):
    # Load observed data

    base_params = load_object(fileName + "/Data", "base_params")
    policy_outcomes = load_object(fileName + "/Data", "policy_outcomes")
    runs_data = load_object(fileName + "/Data", "runs_data")
    
    plot_policy_summary_all(runs_data)

    print("policy_outcomes", policy_outcomes)

    plt.show()

if __name__ == "__main__":
    main(
        fileName="results/endogenous_policy_intensity_single_13_05_42__19_01_2025",
    )
