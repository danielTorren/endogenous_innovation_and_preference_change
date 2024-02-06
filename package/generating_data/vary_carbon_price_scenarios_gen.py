"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 

Created: 10/10/2022
"""

# imports
import time
from copy import deepcopy
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import emissions_intensities_parallel_run
from copy import deepcopy
from package.resources.utility import generate_vals,save_data_csv_2D

def produce_param_list_scenarios(params: dict, property_list: list, property: str, property_section:str) -> list[dict]:
    params_list = []

    for i in property_list:
        params[property_section][property] = i
        for v in range(params["seed_reps"]):
            params["parameters_firm_manager"]["landscape_seed"] = int(v+1)#change landscape seed
            #params_list.append(params.copy())
            params_list.append(deepcopy( params))#aparently copy doesnt work here and i have literally no idea why
    return params_list

def arrange_scenarios_tax(base_params, carbon_tax_vals,scenarios,property_varied, property_section):
    base_params_copy = deepcopy(base_params)
    params_list = []

    ###### WITHOUT CARBON TAX

    # 1. Run with no preference change, no innovation
    if 1 in scenarios:
        base_params_copy_1 = deepcopy(base_params_copy)
        base_params_copy_1["parameters_social_network"]["nu_change_state"] = "fixed_preferences"
        base_params_copy_1["parameters_firm"]["static_tech_state"] = 1
        #print(" base_params_copy_1", base_params_copy_1)
        #quit()
        params_sub_list_1 = produce_param_list_scenarios(base_params_copy_1, carbon_tax_vals,property_varied, property_section)
        params_list.extend(params_sub_list_1)

    # 2. Run with preference change, no innovation
    if 2 in scenarios:
        base_params_copy_2 = deepcopy(base_params_copy)
        base_params_copy_2["parameters_social_network"]["nu_change_state"] = "dynamic_multi_sector_weights"
        base_params_copy_2["parameters_firm"]["static_tech_state"] = 1
        #print(" base_params_copy_2", base_params_copy_2)
        params_sub_list_2 = produce_param_list_scenarios(base_params_copy_2, carbon_tax_vals,property_varied,property_section)
        params_list.extend(params_sub_list_2)

    # 3. Run with no preference change, with innovation
    if 3 in scenarios:
        base_params_copy_3 = deepcopy(base_params_copy)
        base_params_copy_3["parameters_social_network"]["nu_change_state"] =  "fixed_preferences"
        base_params_copy_3["parameters_firm"]["static_tech_state"] = 0
        #print(" base_params_copy_3", base_params_copy_3)
        params_sub_list_3 = produce_param_list_scenarios(base_params_copy_3, carbon_tax_vals,property_varied,property_section)
        params_list.extend(params_sub_list_3)

    # 4. Run with preference change, with innovation
    if 4 in scenarios:
        base_params_copy_4 = deepcopy(base_params_copy)
        base_params_copy_4["parameters_social_network"]["nu_change_state"] = "dynamic_multi_sector_weights"
        base_params_copy_4["parameters_firm"]["static_tech_state"] = 0
        #print(" base_params_copy_4", base_params_copy_4)
        params_sub_list_4 = produce_param_list_scenarios(base_params_copy_4, carbon_tax_vals,property_varied,property_section)
        params_list.extend(params_sub_list_4)
    #quit()
    #for i in params_list:
    #    print(i["parameters_social_network"]["nu_change_state"],i["parameters_firm"]["static_tech_state"],i["parameters_social_network"]["carbon_price"], i["parameters_firm_manager"]["landscape_seed"])
    #quit()
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
    params_list = arrange_scenarios_tax(params,property_values_list,scenarios,property_varied, property_section)

    #quit()
    print("Total runs: ",len(params_list))
    
    #RESULTS
    emissions_cumulative_flat, weighted_emissions_intensities_flat = emissions_intensities_parallel_run(params_list)

    #unpack_results into scenarios and seeds
    emissions_cumulative = emissions_cumulative_flat.reshape(scenario_reps,property_reps,params["seed_reps"])
    weighted_emissions_intensities = weighted_emissions_intensities_flat.reshape(scenario_reps,property_reps,params["seed_reps"])

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    #quit()
    ##################################
    #save data

    createFolder(fileName)

    save_object(emissions_cumulative, fileName + "/Data", "emissions_cumulative")
    save_object(weighted_emissions_intensities, fileName + "/Data", "weighted_emissions_intensities")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(scenarios, fileName + "/Data", "scenarios")

    #############################
    #SAVE AS CSV for miquel
    save_data_csv_2D(emissions_cumulative, fileName + "/Data", "emissions_cumulative")
    save_data_csv_2D(weighted_emissions_intensities, fileName + "/Data", "weighted_emissions_intensities")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_carbon_price_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_carbon_price_vary.json",
        scenarios = [1,2,3,4]
    )