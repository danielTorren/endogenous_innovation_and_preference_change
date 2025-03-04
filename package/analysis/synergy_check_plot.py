from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np

def calculate_synergy(pairwise_outcomes, individual_policy_results, EV_uptake_no_policy):
    """
    Calculate synergy for each policy combination using the provided equation:
    synergy = EV_uptake(both policies) - (EV_uptake(policy1) + EV_uptake(policy2)) - EV_uptake(no policy)
    """
    synergy_data = {}

    for (policy1, policy2), data in pairwise_outcomes.items():
        synergy_values = []
        for entry in data:
            EV_uptake_both = entry["mean_ev_uptake"]

            # Extract EV uptake for individual policies
            if "mean_ev_uptake_policy1" in entry and "mean_ev_uptake_policy2" in entry:
                EV_uptake_policy1 = entry["mean_ev_uptake_policy1"]
                EV_uptake_policy2 = entry["mean_ev_uptake_policy2"]
            else:
                # Fallback: Calculate EV uptake for individual policies using individual_policy_results
                EV_uptake_policy1 = next(
                    (x["mean_ev_uptake"] for x in individual_policy_results[policy1] 
                     if x["policy_value"] == entry["policy1_value"]), None
                )
                EV_uptake_policy2 = next(
                    (x["mean_ev_uptake"] for x in individual_policy_results[policy2] 
                     if x["policy_value"] == entry["policy2_value"]), None
                )

                if EV_uptake_policy1 is None or EV_uptake_policy2 is None:
                    raise ValueError(f"Could not find EV uptake for {policy1} or {policy2} in individual_policy_results")

            synergy = EV_uptake_both - (EV_uptake_policy1 + EV_uptake_policy2) - EV_uptake_no_policy
            synergy_values.append(synergy)

        synergy_data[(policy1, policy2)] = {
            "policy1_values": [entry["policy1_value"] for entry in data],
            "policy2_values": [entry["policy2_value"] for entry in data],
            "synergy_values": synergy_values
        }

    return synergy_data

def plot_synergy_scatter(synergy_data):
    """
    Plot a scatter plot for each policy combination, with synergy as the color.
    """
    # Create subplots for each policy combination
    num_pairs = len(synergy_data)
    fig, axes = plt.subplots(1, num_pairs, figsize=(20, 5), sharey=True)
    if num_pairs == 1:
        axes = [axes]  # Ensure axes is iterable even for a single subplot

    for ax, ((policy1, policy2), data) in zip(axes, synergy_data.items()):
        scatter = ax.scatter(
            data["policy1_values"],
            data["policy2_values"],
            c=data["synergy_values"],
            cmap="viridis",
            s=100,
            edgecolor='k'
        )
        ax.set_xlabel(f"{policy1} Intensity", fontsize=4)
        
        #ax.set_title(f"{policy1} vs {policy2}")
        ax.grid(True)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        #cbar.set_label("Synergy")
    #print(axes)
    #quit()

    axes[0].set_ylabel(f"{policy2} Intensity")
    #plt.tight_layout()


def main(
        fileName_policies, 
        fileName_synergy,
        EV_uptake_no_policy
        ):
    """
    Main function to load data, calculate synergy, and plot results.
    """
    # Load observed data
    base_params = load_object(fileName_policies + "/Data", "base_params")
    pairwise_outcomes_complied = load_object(fileName_policies + "/Data", "pairwise_outcomes_complied")
    individual_policy_results = load_object(fileName_synergy + "/Data", "individual_policy_results")

    # Exclude the ('Discriminatory_corporate_tax', 'Carbon_price') combination
    if ('Discriminatory_corporate_tax', 'Carbon_price') in pairwise_outcomes_complied:
        del pairwise_outcomes_complied[('Discriminatory_corporate_tax', 'Carbon_price')]

    # Calculate synergy
    synergy_data = calculate_synergy(pairwise_outcomes_complied, individual_policy_results, EV_uptake_no_policy)

    # Plot the synergy scatter plot
    plot_synergy_scatter(synergy_data)

    plt.show()
    
if __name__ == "__main__":
    # Run the main function
    main(
         "results/endogenous_policy_intensity_17_21_20__28_02_2025", 
         "results/endogenous_policy_intensity_00_58_15__04_03_2025",
         0.3  # Replace with the actual EV uptake for the "no policy" scenario
         )