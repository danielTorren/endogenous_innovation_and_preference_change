import json
import numpy as np
from copy import deepcopy
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from package.resources.utility import save_object
from package.analysis.endogenous_policy_intensity_single_gen import (
    set_up_calibration_runs,
    single_policy_with_seeds
)

emissions_BAU = 0.12e9
utility_BAU = 87e9

def update_policy_intensity(params, policy_name, intensity_level):
    params["parameters_policies"]["States"][policy_name] = 1
    if policy_name == "Carbon_price":
        params["parameters_policies"]["Values"][policy_name]["Carbon_price"] = intensity_level
    else:
        params["parameters_policies"]["Values"][policy_name] = intensity_level
    return params


def custom_cost_function(ev_uptake, emissions, utility):
    """
    Step cost:
    - Huge penalty if EV uptake is outside 0.945–0.955
    - Reward low emissions otherwise
    """
    global emissions_BAU
    global utility_BAU

    if ev_uptake < 0.945 or ev_uptake > 0.955:
        return 1e6 + abs(ev_uptake - 0.95) * 1e5
    emissions_ratio = (emissions/ emissions_BAU)
    utility_ratio = (utility_BAU/utility)
    print(emissions_ratio,utility_ratio)
    return 0.3*emissions_ratio + 0.7*utility_ratio


def simulate_policy_scenario(sim_params, controller_files):
    EV_uptake_arr, _, _, emissions_cumulative_arr, _, _, _, _, _ = single_policy_with_seeds(sim_params, controller_files)
    mean_ev_uptake = np.mean(EV_uptake_arr)
    mean_emissions = np.mean(emissions_cumulative_arr)
    return mean_ev_uptake, mean_emissions


def optimize_three_policies_BO(base_params, controller_files, policy_names, bounds_dict, n_calls=30):
    # Define search space
    dimensions = [
        Real(bounds_dict[p][0], bounds_dict[p][1], name=p)
        for p in policy_names
    ]

    @use_named_args(dimensions)
    def objective(**params_dict):
        params = deepcopy(base_params)
        for policy, value in params_dict.items():
            params = update_policy_intensity(params, policy, value)

        try:
            ev_uptake, emissions = simulate_policy_scenario(params, controller_files)
        except Exception as e:
            print(f"[ERROR] Simulation failed: {e}")
            return 1e10

        cost = custom_cost_function(ev_uptake, emissions)
        print(f"[TEST] {params_dict} -> Uptake: {ev_uptake:.4f}, Emissions: {emissions:.2e}, Cost: {cost:.2f}")
        return cost

    result = gp_minimize(
        objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=8, # 8–10 random samples before modeling
        random_state=42,
        acq_func="EI"
    )

    best_values = result.x
    best_cost = result.fun
    best_combo = dict(zip(policy_names, best_values))

    return {
        "policy_names": policy_names,
        "best_intensities": best_combo,
        "best_cost": best_cost,
        "result": result
    }


def main(
    BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
    BOUNDS_LOAD="package/analysis/policy_bounds_vary_pair_policy_gen.json",
    policy_names=["Carbon_price", "Adoption_subsidy", "Production_subsidy"],
    n_calls=30,
    emissions_BAU = 1e9,
    utility_BAU = 1e9
):
    # Load parameters
    with open(BASE_PARAMS_LOAD) as f:
        base_params = json.load(f)

    with open(BOUNDS_LOAD) as f:
        bounds_dict = json.load(f)

    # Setup calibration and controller files
    controller_files, base_params, file_name = set_up_calibration_runs(base_params, "triple_policy_BO")

    # Run Bayesian Optimization
    result = optimize_three_policies_BO(base_params, controller_files, policy_names, bounds_dict,n_calls=n_calls)

    print("\n✅ Optimization Complete")
    print("Best Policy Intensities:", result["best_intensities"])
    print("Final Cost:", result["best_cost"])
    print("Results saved in:", file_name)


    # Save results
    save_object(result, file_name + "/Data", "triple_policy_optimization_result")
    save_object(policy_names, file_name + "/Data", "optimized_policy_names")



if __name__ == "__main__":
    main(
        BASE_PARAMS_LOAD="package/constants/base_params_endogenous_policy_pair_gen.json",
        BOUNDS_LOAD="package/analysis/policy_bounds_vary_pair_policy_gen.json",
        policy_names=["Carbon_price", "Adoption_subsidy_used", "Production_subsidy"],
        n_calls=30
    )
