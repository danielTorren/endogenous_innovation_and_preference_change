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
    policy_outcomes = load_object(fileName + "/Data", "policy_outcomes")
    #runs_data = load_object(fileName + "/Data", "runs_data")
    print("policy_outcomes", list(policy_outcomes.keys()))

    # Apply the function to the data
    policy_outcomes = divide_by_billion(policy_outcomes)

    plt.show()

if __name__ == "__main__":
    main(
        fileName="results/endogenous_policy_intensity_18_17_27__06_03_2025"
    )
