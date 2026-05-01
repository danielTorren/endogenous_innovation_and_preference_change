from copy import deepcopy
import json
from package.resources.run import parallel_run_multi_seed_cars
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    params_list_with_seed
)
from package.plotting_data.multi_seed_plot_cars import main as plotting_main

def main(
        BASE_PARAMS_LOAD="package/constants/base_params_run_scenario_seeds.json",
    ) -> str: 

    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    root = "multi_seed_cars"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = params_list_with_seed(base_params)
    print("TOTAL RUNS: ", len(params_list))

    cars_on_sale = parallel_run_multi_seed_cars(params_list)

    createFolder(fileName)

    outputs = {
        "cars_on_sale": cars_on_sale
    }

    save_object(outputs, fileName + "/Data", "outputs")
    save_object(base_params, fileName + "/Data", "base_params")

    print(fileName)
    return fileName

if __name__ == "__main__":
    fileName = main(BASE_PARAMS_LOAD="package/constants/base_params_multi_seed_cars.json")

    """
    Will also plot stuff at the same time for convieniency
    """
    RUN_PLOT = 1
    print("fileName",fileName)
    if RUN_PLOT:
        plotting_main(fileName = fileName)