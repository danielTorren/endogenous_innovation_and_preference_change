import json
import numpy as np
from SALib.sample import saltelli
import numpy.typing as npt
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime
)
from copy import deepcopy
from package.resources.run import parallel_run_sa_ev

def generate_problem_seeds(seed_keys, N_samples, calc_second_order):
    """
    Generate the saltelli sample for seed values instead of model parameters.
    """
    D_vars = len(seed_keys)  # Number of different seeds being tested
    
    if calc_second_order:
        samples = N_samples * (2 * D_vars + 2)
    else:
        samples = N_samples * (D_vars + 2)
    
    print("samples: ", samples)
    
    problem = {
        "num_vars": D_vars,
        "names": seed_keys,
        "bounds": [[1, 10000]] * D_vars  # Assume seeds are sampled within this range
    }
    
    param_values = saltelli.sample(problem, N_samples, calc_second_order=calc_second_order)
    param_values = np.round(param_values).astype(int)  # Ensure seeds are integers
    
    return problem, param_values

def stochastic_produce_param_list_SA(param_values, base_params, seed_keys):
    """
    Generate list of parameter dictionaries, varying only the seeds.
    """
    params_list = []
    for X in param_values:
        base_params_copy = deepcopy(base_params)
        
        for i, seed_key in enumerate(seed_keys):
            base_params_copy["seeds"][seed_key] = X[i]
        
        params_list.append(base_params_copy)
    
    return params_list

def main(
        N_samples=1024,
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        calc_second_order=False
):
    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)
    
    # Identify which seeds to vary
    seed_keys = list(base_params["seeds"].keys())
    
    problem, param_values = generate_problem_seeds(seed_keys, N_samples, calc_second_order)
    
    params_list_sa = stochastic_produce_param_list_SA(param_values, base_params, seed_keys)
    
    print("Total runs: ", len(params_list_sa))
    
    root = "sensitivity_analysis_seeds"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)
    
    Y_ev = parallel_run_sa_ev(params_list_sa)
    
    createFolder(fileName)
    
    # Save results
    save_object(Y_ev, fileName + "/Data", "Y_ev")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(problem, fileName + "/Data", "problem")
    save_object(N_samples, fileName + "/Data", "N_samples")
    save_object(calc_second_order, fileName + "/Data", "calc_second_order")
    
    return fileName

if __name__ == '__main__':
    fileName_Figure = main(
        N_samples=16,  # Adjust as needed
        BASE_PARAMS_LOAD="package/constants/base_params_SA_seeds.json",
        calc_second_order=False
    )
