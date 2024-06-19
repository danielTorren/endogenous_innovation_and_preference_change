"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 

Created: 10/10/2022
"""

# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import  parallel_run
from copy import deepcopy
from package.resources.utility import generate_vals,save_data_csv_2D

def produce_param_list_stochastic(params: dict) -> list[dict]:
    params_list = []
    for v in range(params["seed_reps"]):
        params["parameters_firm_manager"]["landscape_seed"] = int(v+1)#change landscape seed
        params_list.append(deepcopy( params))#aparently copy doesnt work here and i have literally no idea why
    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        print_simu = 1
        ) -> str: 


    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "stochastic_runs"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    #Gen params lists
    params_list = produce_param_list_stochastic(params)

    #quit()
    print("Total runs: ",len(params_list))
    
    #RESULTS
    data_list = parallel_run(params_list)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    #quit()
    ##################################
    #save data

    createFolder(fileName)

    save_object(data_list, fileName + "/Data", "data_list")
    save_object(params, fileName + "/Data", "base_params")

    #############################
    #SAVE AS CSV for miquel
    #save_data_csv_2D(emissions_cumulative, fileName + "/Data", "emissions_cumulative")

    return fileName

if __name__ == "__main__":
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_stochastic.json"
    )