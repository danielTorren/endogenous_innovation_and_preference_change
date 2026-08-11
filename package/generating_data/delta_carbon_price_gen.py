"""Sweep ICE wear/depreciation rate `delta` under two scenarios, 12 seeds each:

  1. Calibration period only, at zero carbon price (duration_future = 0).
  2. Calibration + a policy period with a flat carbon price of 0.8.

For every delta value, all seeds for that scenario run together (one
parallel_run_multi_seed call per scenario, covering every delta * seed
combination at once) and the results are then split back out per delta into
their own results folder. The calibration scenario gets the full calibration
fit figure (EV uptake, prices, HHI, car age); the policy scenario only needs
EV uptake and sales.
"""
import json
from copy import deepcopy

from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder,
    save_object,
    load_object,
    produce_name_datetime,
    params_list_with_seed,
)
from package.plotting_data.calibration_plot import (
    plot_calibration_fit,
    plot_ev_uptake_sales_only,
)

SAVED_KEYS = (
    "history_driving_emissions",
    "history_production_emissions",
    "history_total_emissions",
    "history_prop_EV",
    "history_lower_percentile_price_ICE_EV",
    "history_upper_percentile_price_ICE_EV",
    "history_mean_price_ICE_EV",
    "history_median_price_ICE_EV",
    "history_total_utility",
    "history_market_concentration",
    "history_total_profit",
    "history_mean_profit_margins_ICE",
    "history_mean_profit_margins_EV",
    "history_mean_car_age",
    "history_past_new_bought_vehicles_prop_ev",
    "cars_on_sale",
)

DELTA_VALUES = [0.00175, 0.00176, 0.00177, 0.00178, 0.00179, 0.0018]


def delta_folder_name(root, delta):
    return f"{root}/delta_{delta:.5f}"


def run_delta_sweep(base_params, delta_values, seed_repetitions, root_name):
    base_params = deepcopy(base_params)
    base_params["seed_repetitions"] = seed_repetitions

    root = produce_name_datetime(root_name)
    print("fileName:", root)
    createFolder(root)

    params_list = []
    for delta in delta_values:
        params = deepcopy(base_params)
        params["parameters_ICE"]["delta"] = delta
        params_list.extend(params_list_with_seed(params))

    print(f"TOTAL RUNS: {len(params_list)} ({len(delta_values)} delta values x {seed_repetitions} seeds)")
    run_outputs = parallel_run_multi_seed(params_list)

    for i, delta in enumerate(delta_values):
        start = i * seed_repetitions
        end = start + seed_repetitions

        delta_params = deepcopy(base_params)
        delta_params["parameters_ICE"]["delta"] = delta

        delta_outputs = {key: run_outputs[key][start:end] for key in SAVED_KEYS}

        sub_name = delta_folder_name(root, delta)
        createFolder(sub_name)
        save_object(delta_outputs, sub_name + "/Data", "outputs")
        save_object(delta_params, sub_name + "/Data", "base_params")

    return root


def main(
        CALIBRATION_BASE_PARAMS_LOAD="package/constants/base_params_calibration.json",
        POLICY_BASE_PARAMS_LOAD="package/constants/base_params_vary_single_carbon_tax.json",
        delta_values=DELTA_VALUES,
        seed_repetitions=12,
        policy_carbon_price=0.8,
    ):

    with open(CALIBRATION_BASE_PARAMS_LOAD) as f:
        calibration_base_params = json.load(f)

    with open(POLICY_BASE_PARAMS_LOAD) as f:
        policy_base_params = json.load(f)
    policy_base_params["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price"] = policy_carbon_price

    calibration_root = run_delta_sweep(
        calibration_base_params, delta_values, seed_repetitions, "delta_sweep_calibration"
    )
    policy_root = run_delta_sweep(
        policy_base_params, delta_values, seed_repetitions, "delta_sweep_policy"
    )

    return calibration_root, policy_root


if __name__ == "__main__":
    calibration_root, policy_root = main()

    print("calibration_root", calibration_root)
    print("policy_root", policy_root)

    RUN_PLOT = 1
    if RUN_PLOT:
        for delta in DELTA_VALUES:
            calib_sub = delta_folder_name(calibration_root, delta)
            base_params = load_object(calib_sub + "/Data", "base_params")
            outputs = load_object(calib_sub + "/Data", "outputs")
            plot_calibration_fit(base_params, calib_sub, outputs)

            policy_sub = delta_folder_name(policy_root, delta)
            base_params = load_object(policy_sub + "/Data", "base_params")
            outputs = load_object(policy_sub + "/Data", "outputs")
            plot_ev_uptake_sales_only(base_params, policy_sub, outputs)
