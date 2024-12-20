from copy import deepcopy
import itertools
import json
from package.resources.run import emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
import numpy as np

def generate_scenario_combinations(base_params, key):
    """
    Generate all combinations for a specific set of scenarios or policies.
    """
    param_states = base_params[key]["States"]
    param_values = base_params[key]["Values"]

    # Generate all possible combinations for the given parameters
    keys, states = zip(*param_states.items())
    value_sets = [list(param_values[k].keys()) for k in keys]

    combinations = list(itertools.product(*value_sets))

    return keys, combinations

def update_parameters(base_params, key, param_keys, combination):
    """
    Update base_params with a specific combination of scenario or policy states.
    """
    base_param_copy = deepcopy(base_params)
    for idx, state in enumerate(combination):
        base_param_copy[key]["States"][param_keys[idx]] = state
    return base_param_copy

def expand_scenarios_with_seeds(scenarios, seed_list):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    seed_repetitions = len(seed_list)
    expanded_scenarios = []
    for scenario in scenarios:
        scenario_seeds = []
        for seed in seed_list:

            scenario_copy = deepcopy(scenario)

            # VARY ALL THE SEEDS
            scenario_copy["parameters_firm_manager"]["init_tech_seed"] = seed + seed_repetitions
            scenario_copy["parameters_ICE"]["landscape_seed"] = seed + 2 * seed_repetitions
            scenario_copy["parameters_EV"]["landscape_seed"] = seed + 9 * seed_repetitions
            scenario_copy["parameters_social_network"]["social_network_seed"] = seed + 3 * seed_repetitions
            scenario_copy["parameters_social_network"]["network_structure_seed"] = seed + 4 * seed_repetitions
            scenario_copy["parameters_social_network"]["init_vals_environmental_seed"] = seed + 5 * seed_repetitions
            scenario_copy["parameters_social_network"]["init_vals_innovative_seed"] = seed + 6 * seed_repetitions
            scenario_copy["parameters_social_network"]["init_vals_price_seed"] = seed + 7 * seed_repetitions
            scenario_copy["parameters_firm"]["innovation_seed"] = seed + 8 * seed_repetitions

            scenario_seeds.append(scenario_copy)
        expanded_scenarios.append(scenario_seeds)
    return expanded_scenarios

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    seed_repetitions = base_params["seed_repetitions"]

    root = "scenarios_combo"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    # Get all combinations of scenario and policy parameters
    scenario_keys, scenario_combinations = generate_scenario_combinations(base_params, "parameters_scenarios")
    scenario_reps = len(scenario_combinations)

    base_scenarios = []
    for scenario_combination in scenario_combinations:
        # Update scenario parameters
        base_scenarios.append(update_parameters(base_params, "parameters_scenarios", scenario_keys, scenario_combination))

    # Expand scenarios with seeds
    seed_list = np.arange(1,1 + seed_repetitions)
    params_list = expand_scenarios_with_seeds(base_scenarios, seed_list)
    
    #print(params_list, len(params_list), len())
    #quit()
    # Flatten the parameters list for parallel execution
    flattened_params_list = [scenario for scenario_group in params_list for scenario in scenario_group]

    # Run the simulation with the current combination in parallel!
    print("TOTAL RUNS: ", len(flattened_params_list))

    data_flat = emissions_parallel_run(flattened_params_list) 

    # Reshape data into 2D structure: rows for scenarios, columns for seed values
    data_array = data_flat.reshape(scenario_reps,seed_repetitions, len(data_flat[0]))
    
    data_reshaped = [data_flat[i * seed_repetitions:(i + 1) * seed_repetitions] for i in range(len(base_scenarios))]
    #print("data_reshaped", data_reshaped)

    createFolder(fileName)

    save_object(data_flat, fileName + "/Data", "data_flat")
    save_object(data_reshaped, fileName + "/Data", "data_reshaped")
    save_object(data_array, fileName + "/Data", "data_array")
    save_object(params_list, fileName + "/Data", "params_list_flat")
    save_object(base_params, fileName + "/Data", "base_params")

    print("Done")
    return params_list

if __name__ == "__main__":
    results = main(BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json")