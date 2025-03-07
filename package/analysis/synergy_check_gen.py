import json
import numpy as np
from joblib import Parallel, delayed, load
import multiprocessing
from package.resources.run import load_in_controller
from package.resources.utility import save_object, load_object
import shutil  # Cleanup
from pathlib import Path  # Path handling
from package.analysis.endogenous_policy_intensity_single_gen import set_up_calibration_runs

def update_policy_intensity(params, policy_name, intensity_level):
    """
    Update the policy intensity in the parameter dictionary.
    """
    params["parameters_policies"]["States"][policy_name] = 1

    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity_level
    return params


def single_policy_simulation(params, controller_file):
    """
    Run a single simulation and return EV uptake and policy distortion.
    """
    controller = load(controller_file)  # Load fresh controller
    data = load_in_controller(controller, params)
    return (
        data.calc_EV_prop(),
        data.calc_total_policy_distortion(),
        data.social_network.emissions_cumulative,
        data.social_network.emissions_cumulative_driving,
        data.social_network.emissions_cumulative_production,
        data.social_network.utility_cumulative,
        data.firm_manager.profit_cumulative
    )

def single_policy_with_seeds(params, controller_files):
    """
    Run policy scenarios using pre-saved controllers for consistency.
    """
    num_cores = multiprocessing.cpu_count()

    res = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(single_policy_simulation)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )

    EV_uptake_arr, total_cost_arr, emissions_cumulative_arr, emissions_cumulative_driving_arr, emissions_cumulative_production_arr, utility_cumulative_arr, profit_cumulative_arr = zip(*res)
    
    return (
        np.asarray(EV_uptake_arr),
        np.asarray(total_cost_arr),
        np.asarray(emissions_cumulative_arr),
        np.asarray(emissions_cumulative_driving_arr),
        np.asarray(emissions_cumulative_production_arr),
        np.asarray(utility_cumulative_arr),
        np.asarray(profit_cumulative_arr)
    )

def run_individual_policies(base_params, controller_files, policy_values):
    """
    Run simulations for individual policies using the provided policy values.
    """
    individual_policy_results = {}

    for policy_name, values in policy_values.items():
        print(f"Running simulations for {policy_name}...")
        policy_results = []
        print("num simulatiosn in policy",len(values))
        for value in values:
            params = update_policy_intensity(base_params.copy(), policy_name, value)
            EV_uptake_arr, total_cost_arr, emissions_cumulative_arr, emissions_cumulative_driving_arr, emissions_cumulative_production_arr, utility_cumulative_arr, profit_cumulative_arr = single_policy_with_seeds(params, controller_files)

            policy_results.append({
                "policy_value": value,
                "mean_ev_uptake": np.mean(EV_uptake_arr),
                "mean_total_cost": np.mean(total_cost_arr),
                "mean_emissions_cumulative": np.mean(emissions_cumulative_arr),
                "mean_emissions_cumulative_driving": np.mean(emissions_cumulative_driving_arr),
                "mean_emissions_cumulative_production": np.mean(emissions_cumulative_production_arr),
                "mean_utility_cumulative": np.mean(utility_cumulative_arr),
                "mean_profit_cumulative": np.mean(profit_cumulative_arr)
            })

        individual_policy_results[policy_name] = policy_results

    return individual_policy_results

def extract_policy_values(data):
    """
    Extract policy values from the dataset.

    Args:
        data (dict): The dataset containing combined policy runs.

    Returns:
        dict: A dictionary where keys are policy names and values are lists of policy values.
    """
    policy_values = {}

    for policy_pair, runs in data.items():
        policy1_name, policy2_name = policy_pair

        # Initialize lists for each policy if not already present
        if policy1_name not in policy_values:
            policy_values[policy1_name] = []
        if policy2_name not in policy_values:
            policy_values[policy2_name] = []

        # Extract values for each run
        for run in runs:
            policy_values[policy1_name].append(run["policy1_value"])
            policy_values[policy2_name].append(run["policy2_value"])

    # Remove duplicates (if any) and sort the values
    for policy_name in policy_values:
        print("runs: ",len(sorted(list(set(policy_values[policy_name])))))

        policy_values[policy_name] = sorted(list(set(policy_values[policy_name])))
    #quit()
    return policy_values

def main(
        fileName = "results/endogenous_policy_intensity_22_02_22__28_02_2025"
):
    # Load base parameters

    base_params = load_object(fileName + "/Data", "base_params")
    #print("base_params", base_params)
    pairwise_outcomes_complied = load_object(fileName + "/Data", "pairwise_outcomes_complied")
    del pairwise_outcomes_complied[('Discriminatory_corporate_tax', 'Carbon_price')]

    # Set up calibration runs
    controller_files, base_params, file_name = set_up_calibration_runs(base_params,"synergy_check")

    # Extract policy values from the dataset
    policy_values = extract_policy_values(pairwise_outcomes_complied)
    print("policy_values",policy_values)

    # Run individual policy simulations
    individual_policy_results = run_individual_policies(base_params, controller_files, policy_values)

    # Save results
    save_object(individual_policy_results, file_name + "/Data", "individual_policy_results")

    print("Individual policy simulations complete!")

    # Cleanup
    shutil.rmtree(Path(file_name) / "Calibration_runs", ignore_errors=True)

if __name__ == "__main__":
    main(
        fileName = "results/endogenous_policy_intensity_17_21_20__28_02_2025"
    )