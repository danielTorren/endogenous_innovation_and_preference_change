"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 

Created: 10/10/2022
"""

# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import  preferences_parallel_run
from copy import deepcopy
from package.resources.utility import generate_vals,save_data_csv_2D

def produce_param_list_scenarios(params: dict, property_list: list, property: str, property_section:str) -> list[dict]:
    params_list = []

    for i in property_list:
        #params[property_section][property] = i#, ive changed where i set the carbon price
        params[property_section][property] = i
        for v in range(params["seed_reps"]):
            params["parameters_firm_manager"]["landscape_seed"] = int(v+1)#change landscape seed
            #params_list.append(params.copy())
            params_list.append(deepcopy( params))#aparently copy doesnt work here and i have literally no idea why
    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau_vary.json",
        print_simu = 1,
        scenarios = [1]
        ) -> str: 

    scenario_reps = len(scenarios)

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]# #"A to Omega ratio"
    property_section = var_params["property_section"]

    property_values_list = generate_vals(
        var_params
    )
    #print("property_list",property_values_list )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "tax_sweep"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    #Gen params lists
    params_list = produce_param_list_scenarios(params,property_values_list,property_varied,property_section)

    #quit()
    print("Total runs: ",len(params_list))
    
    #RESULTS
    preference_distribution_flat = preferences_parallel_run(params_list)
    preference_distribution = preference_distribution_flat.reshape(property_reps,params["seed_reps"],params["parameters_social_network"]["num_individuals"] )

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    #quit()
    ##################################
    #save data

    createFolder(fileName)

    save_object(preference_distribution, fileName + "/Data", "preference_distribution")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(scenarios, fileName + "/Data", "scenarios")

    #############################
    #SAVE AS CSV for miquel
    #save_data_csv_2D(emissions_cumulative, fileName + "/Data", "emissions_cumulative")

    return fileName

if __name__ == "__main__":
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_preference_innovation.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_carbon_price_preference_innovation.json"
    )