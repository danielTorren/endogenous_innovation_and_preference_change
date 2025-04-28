from package.resources.utility import (
    load_object
)
import matplotlib.pyplot as plt

# Function to divide specific keys by 1e9
def divide_by_billion(d):
    for key, value in d.items():
        if isinstance(value, dict):
            divide_by_billion(value)  # Recursively process nested dictionaries
        else:
            if key in ['mean_total_cost', 'mean_net_cost', 'mean_emissions_cumulative', 
                    'mean_emissions_cumulative_driving', 'mean_emissions_cumulative_production', 
                    'mean_utility_cumulative', 'mean_profit_cumulative']:
                d[key] = value/1e9  # Divide by 1 billion
            d[key] = round(d[key], 4)#4sf

    return d

def main(fileName):
    # Load observed data

    #base_params = load_object(fileName + "/Data", "base_params")
    policy_outcomes = load_object(fileName + "/Data", "outcomes")
    policy_outcomes_carbon_tax = load_object(fileName + "/Data", "outcomes_carbon_tax")
    policy_outcomes_adoption_subsidy = load_object(fileName + "/Data", "outcomes_adoption_subsidy")
    
    # Get the keys up to and including 'mean_profit_cumulative'
    all_keys = list(policy_outcomes.keys())

    cutoff_index = all_keys.index("mean_profit_cumulative") + 1
    selected_keys = all_keys[:cutoff_index]


    # Loop through each policy and print the selected outcomes
    print("BAU")
    for key in selected_keys:
        if key in ["mean_utility_cumulative","mean_utility_cumulative_30"]:
            print(f"  {key}: {(policy_outcomes.get(key))*12}")
        else:
            print(f"  {key}: {policy_outcomes.get(key)}")

    # Loop through each policy and print the selected outcomes
    print("CARBON TAX")
    for key in selected_keys:
        if key in ["mean_utility_cumulative","mean_utility_cumulative_30"]:
            print(f"  {key}: {(policy_outcomes_carbon_tax.get(key))*12}")
        else:
            print(f"  {key}: {policy_outcomes_carbon_tax.get(key)}")

    # Loop through each policy and print the selected outcomes
    print("ADOPTION SUBSIDY")
    for key in selected_keys:
        if key in ["mean_utility_cumulative","mean_utility_cumulative_30"]:
            print(f"  {key}: {(policy_outcomes_adoption_subsidy.get(key))*12}")
        else:
            print(f"  {key}: {policy_outcomes_adoption_subsidy.get(key)}")

    # Apply the function to the data
    #policy_outcomes = divide_by_billion(policy_outcomes)
    #policy_outcomes = divide_by_billion(policy_outcomes_carbon_tax)
    #policy_outcomes = divide_by_billion(policy_outcomes_adoption_subsidy)

    plt.show()

if __name__ == "__main__":
    main(
        fileName="results/BAU_runs_19_23_01__24_04_2025"
    )
