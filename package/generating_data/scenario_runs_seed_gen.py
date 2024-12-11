from copy import deepcopy
import itertools
import json
from package.resources.run import  emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

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


def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_run_scenario_seeds.json",
        ) -> str: 

    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    
    
    root = "scenarios_combo"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    createFolder(fileName)

    # Get all combinations of scenario and policy parameters
    scenario_keys, scenario_combinations = generate_scenario_combinations(base_params, "parameters_scenarios")
    
    results = []
    params_list = []
    for scenario_combination in scenario_combinations:
        # Update scenario parameters
        params_list.append(update_parameters(base_params, "parameters_scenarios", scenario_keys, scenario_combination))

    # Run the simulation with the current combination in parallel!
    print("TOTAL RUNS: ", len(params_list))
    data_flat = emissions_parallel_run(params_list) 
    
    save_object(data_flat, fileName + "/Data", "data_flat")
    save_object(params_list, fileName + "/Data", "params_list_flat")
    save_object(base_params, fileName + "/Data", "base_params")
    
    return results

if __name__ == "__main__":
    results = main(BASE_PARAMS_LOAD = "package/constants/base_params_run_scenario_seeds.json")
