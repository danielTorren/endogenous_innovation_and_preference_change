import json
import numpy as np
from itertools import product
from package.resources.run import ev_prop_emissions_parallel_run
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime,
    params_list_with_seed
)

def produce_physical_param_list(base_params, var_configs):
    value_grids = [v["property_list"] for v in var_configs]
    final_list = []
    for values in product(*value_grids):
        current_params = json.loads(json.dumps(base_params))
        for i, val in enumerate(values):
            v_cfg = var_configs[i]
            current_params[v_cfg["subdict"]][v_cfg["property_varied"]] = val
        final_list.extend(params_list_with_seed(current_params))
    return final_list

def run_physical_duo(BASE_PARAMS_PATH, VAR_PATHS):
    # 1. Load Files
    with open(BASE_PARAMS_PATH) as f:
        base_params = json.load(f)
    var_configs = []
    for path in VAR_PATHS:
        with open(path) as f:
            var_configs.append(json.load(f))

    # 2. Metadata
    dims = [len(v["property_list"]) for v in var_configs]
    seed_reps = base_params.get("seed_repetitions", 1)
    var_names = "_vs_".join([v["property_varied"] for v in var_configs])
    folder_name = produce_name_datetime(f"phys_duo_{var_names}")

    # 3. Generate Parameters
    params_list = produce_physical_param_list(base_params, var_configs)
    print(f"Structure: {' x '.join(map(str, dims))} (total combinations: {np.prod(dims)})")
    print(f"Total individual runs (including seeds): {len(params_list)}")

    # 4. Execute
    data_ev, data_emissions = ev_prop_emissions_parallel_run(params_list)

    # 5. Reshape — Shape: (Decarb, ElecPrice, Seeds, Time)
    data_array_ev = data_ev.reshape(*dims, seed_reps, -1)
    data_array_emissions = data_emissions.reshape(*dims, seed_reps, -1)

    # 6. Save
    createFolder(folder_name)
    save_object(data_array_ev, folder_name + "/Data", "data_phys_duo_ev")
    save_object(data_array_emissions, folder_name + "/Data", "data_phys_duo_emissions")
    save_object(base_params, folder_name + "/Data", "base_params")
    save_object(var_configs, folder_name + "/Data", "vary_metadata")

    print(f"Success! Data saved to: {folder_name}")
    return folder_name

if __name__ == "__main__":
    PHYSICAL_VARS = [
        "package/constants/vary_sen_decarb.json",
        "package/constants/vary_sen_elec_price.json"
    ]
    run_physical_duo(
        BASE_PARAMS_PATH="package/constants/base_params_inputs_and_emissions.json",
        VAR_PATHS=PHYSICAL_VARS
    )