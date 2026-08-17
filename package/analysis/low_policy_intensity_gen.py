from package.resources.utility import load_object, save_object
import numpy as np
from package.analysis.endogenous_policy_intensity_single_gen import update_policy_intensity, set_up_calibration_runs
from package.resources.utility import (
    save_object,
    load_object,
    get_num_workers,
)
import shutil  # Cleanup
import sys
from operator import attrgetter
from pathlib import Path  # Path handling
from copy import deepcopy
from package.resources.run import load_in_controller
from joblib import Parallel, delayed, load
import multiprocessing


# Histories collected from every scenario run, as
#   saved output key -> attribute path on the finished controller.
#
# This is the whole contract between the workers and everything downstream:
# each worker returns exactly these, keyed by name, and single_policy_with_seeds
# stacks them across seeds under the same names. Add an entry to collect more.
#
# It is deliberately NOT everything the model records. The per-entity histories
# (history_car_age, which appends all 3000 car ages EVERY step, plus the
# quality / efficiency / production-cost ones, which append a per-car-on-sale
# array each step) used to be returned too and then thrown away by both callers.
# At 768 steps and 64 seeds that was gigabytes shipped from the workers back to
# the parent for nothing, and it OOM-killed the job. Do not re-add one unless
# something actually reads it.
SCENARIO_HISTORIES = {
    "history_driving_emissions": "social_network.history_driving_emissions",
    "history_production_emissions": "social_network.history_production_emissions",
    "history_total_emissions": "social_network.history_total_emissions",
    "history_prop_EV": "social_network.history_prop_EV",
    "history_lower_percentile_price_ICE_EV_arr": "social_network.history_lower_percentile_price_ICE_EV",
    "history_upper_percentile_price_ICE_EV_arr": "social_network.history_upper_percentile_price_ICE_EV",
    "history_mean_price_ICE_EV_arr": "social_network.history_mean_price_ICE_EV",
    "history_median_price_ICE_EV_arr": "social_network.history_median_price_ICE_EV",
    "history_total_utility": "social_network.history_total_utility",
    "history_total_utility_bottom": "social_network.history_total_utility_bottom",
    "history_ev_adoption_rate_bottom": "social_network.history_ev_adoption_rate_bottom",
    "history_mean_car_age": "social_network.history_mean_car_age",
    "history_market_concentration": "firm_manager.history_market_concentration",
    "history_total_profit": "firm_manager.history_total_profit",
    "history_mean_profit_margins_ICE": "firm_manager.history_mean_profit_margins_ICE",
    "history_mean_profit_margins_EV": "firm_manager.history_mean_profit_margins_EV",
    "history_past_new_bought_vehicles_prop_ev": "firm_manager.history_past_new_bought_vehicles_prop_ev",
    "history_policy_net_cost": "history_policy_net_cost",
}


def single_policy_simulation(params, controller_file):
    """
    Run a single simulation and return the SCENARIO_HISTORIES time series.
    """
    controller = load(controller_file)  # Load fresh controller
    data = load_in_controller(controller, params)
    return {name: attrgetter(path)(data) for name, path in SCENARIO_HISTORIES.items()}


def single_policy_with_seeds(params, controller_files):
    """
    Run one scenario across the pre-saved controllers (one per seed, for
    consistency) and stack each history across seeds.

    Returns {history name: array of shape (n_seeds, n_steps, ...)} -- the exact
    dict that gets pickled as outputs_BAU / outputs[pair] / outputs_<policy>.
    """
    num_cores = get_num_workers()
    res = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(single_policy_simulation)(params, controller_files[i % len(controller_files)])
        for i in range(len(controller_files))
    )

    return {name: np.asarray([seed_result[name] for seed_result in res])
            for name in SCENARIO_HISTORIES}


# Single-policy reference trajectories drawn on top of the pairs in Figure 5.
# policy name -> the Data/<name> the plotting script loads it back from. Their
# intensities are NOT hardcoded here: they are read from the pair-gen run's
# single_policy_outcomes, so they always match the endogenous single-policy
# solution of the very run whose pairs are being plotted.
SINGLE_POLICY_REFERENCES = {
    "Carbon_price": "outputs_carbon_tax",
    "Adoption_subsidy": "outputs_adoption_subsidy",
}

# Used only for pair-gen folders too old to carry single_policy_outcomes. These
# are the values that were hardcoded here before, from a long-superseded run.
FALLBACK_SINGLE_POLICY_INTENSITIES = {
    "Carbon_price": 0.910,
    "Adoption_subsidy": 36875.57,
}


def load_single_policy_intensities(fileName_load):
    """
    Endogenous single-policy intensities from the pair-gen run, i.e. the
    intensity of each instrument on its own that hits the EV uptake target.
    """
    try:
        single_policy_outcomes = load_object(f"{fileName_load}/Data", "single_policy_outcomes")
    except FileNotFoundError:
        print(f"[!] No single_policy_outcomes in {fileName_load}/Data -- falling back to the old "
              f"hardcoded intensities {FALLBACK_SINGLE_POLICY_INTENSITIES}. These do NOT match this "
              f"run and the single-policy lines in Figure 5 will be inconsistent with its pairs.")
        return dict(FALLBACK_SINGLE_POLICY_INTENSITIES)

    intensities = {}
    for policy in SINGLE_POLICY_REFERENCES:
        if policy in single_policy_outcomes:
            intensities[policy] = single_policy_outcomes[policy]["optimized_intensity"]
        else:
            intensities[policy] = FALLBACK_SINGLE_POLICY_INTENSITIES[policy]
            print(f"[!] '{policy}' missing from single_policy_outcomes -- falling back to "
                  f"{intensities[policy]}")
    return intensities


def calc_low_intensities(pairwise_outcomes_complied, min_val, max_val):
    # Collect all policies and initialize min/max ranges
    policy_ranges = {}


    for (p1, p2), entries in pairwise_outcomes_complied.items():
        for entry in entries:
            for policy, key in zip((p1, p2), ("policy1_value", "policy2_value")):
                policy_ranges.setdefault(policy, {"min": float('inf'), "max": float('-inf')})
                policy_ranges[policy]["min"] = min(policy_ranges[policy]["min"], entry[key])
                policy_ranges[policy]["max"] = max(policy_ranges[policy]["max"], entry[key])

    # Merge symmetric pairs
    merged = {}
    for (p1, p2), entries in pairwise_outcomes_complied.items():
        key = tuple(sorted((p1, p2)))
        merged.setdefault(key, []).extend([entry | {"original_order": (p1, p2)} for entry in entries])

    #print(list(merged.keys()), len(list(merged.keys())))

    best_entries = {}
    for (pa, pb), entries in merged.items():
        best = None
        min_intensity = float('inf')
        for entry in entries:
            if (min_val <= entry["mean_ev_uptake"] <= max_val):
                p1, p2 = entry["original_order"]
                v1 = entry["policy1_value"]
                v2 = entry["policy2_value"]

                norm1 = (v1 - policy_ranges[p1]["min"]) / (policy_ranges[p1]["max"] - policy_ranges[p1]["min"] or 1)
                norm2 = (v2 - policy_ranges[p2]["min"]) / (policy_ranges[p2]["max"] - policy_ranges[p2]["min"] or 1)

                if (m := max(norm1, norm2)) < min_intensity:
                    min_intensity = m
                    best = (v1, v2, entry["mean_ev_uptake"], entry["original_order"])

            if best:
                v1, v2, uptake, (orig_p1, orig_p2) = best
                # Flip if needed to match key order
                best_entries[(pa, pb)] = {
                    "policy1_value": v1 if (pa, pb) == (orig_p1, orig_p2) else v2,
                    "policy2_value": v2 if (pa, pb) == (orig_p1, orig_p2) else v1,
                    "mean_ev_uptake": uptake,
                    "original_order": (pa, pb)
                }

    return best_entries


def main(fileNames,
        min_ev_uptake = 0.945,
        max_ev_uptake = 0.955
        ):

    pairwise_outcomes_complied = {}
    #pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")
    if len(fileNames) == 1:
        fileName = fileNames[0]
        pairwise_outcomes_complied = load_object(f"{fileName}/Data", "pairwise_outcomes")
    else:
        for fileName in fileNames:
            pairwise_outcomes = load_object(f"{fileName}/Data", "pairwise_outcomes")
            pairwise_outcomes_complied.update(pairwise_outcomes)

    fileName_load = fileNames[0]
    #pairwise_outcomes_complied = load_object(f"{fileName_load}/Data", "pairwise_outcomes")

    top_policies = calc_low_intensities(pairwise_outcomes_complied,  min_ev_uptake, max_ev_uptake)
    
    print("top_policies", list(top_policies.keys()), len( list(top_policies.keys())))

    ##########################################################################################

    base_params_calibration = load_object(fileName_load + "/Data", "base_params")

    base_params_calibration["duration_future"] = 312#564#2050#144#SET UP FUTURE

    if "duration_calibration" not in base_params_calibration:
        base_params_calibration["duration_calibration"] = base_params_calibration["duration_no_carbon_price"]

    base_params_calibration["parameters_policies"]["States"] = {
        "Carbon_price": 0,
        "Targeted_research_subsidy": 0,
        "Electricity_subsidy": 0,
        "Adoption_subsidy": 0,
        "Adoption_subsidy_used": 0,
        "Production_subsidy": 0,
        "Research_subsidy": 0
    }

    controller_files, base_params, root_folder  = set_up_calibration_runs(base_params_calibration, "pair_low_intensity_policies")
    print("DONE calibration")
    #
    #NOW SAVE
    save_object(top_policies, root_folder + "/Data", "top_policies")
    save_object(min_ev_uptake, root_folder + "/Data", "min_ev_uptake")
    save_object(max_ev_uptake, root_folder + "/Data", "max_ev_uptake")

    ###########################################################################################

    #base_params["parameters_scenarios"]["Grid_emissions_intensity"] = 1
    base_params["save_timeseries_data_state"] = 1

    #RESET TO B SURE
    #RUN BAU
    outputs_BAU = single_policy_with_seeds(base_params, controller_files)

    save_object(outputs_BAU, root_folder + "/Data", "outputs_BAU")
    print("DONE BAU")

    ##############################################################################################################################

    print("TOTAL RUNS", len(top_policies)*base_params["seed_repetitions"])
    outputs = {}

    for (policy1, policy2), welfare_data in top_policies.items():

        policy1_value = welfare_data["policy1_value"]
        policy2_value = welfare_data["policy2_value"]

        print(f"Running time series for {policy1},{policy1_value} & {policy2},{policy2_value}")

        params_policy = deepcopy(base_params)
        params_policy = update_policy_intensity(params_policy, policy1, policy1_value)
        params_policy = update_policy_intensity(params_policy, policy2, policy2_value)

        outputs[(policy1, policy2)] = single_policy_with_seeds(params_policy, controller_files)

    save_object(outputs, root_folder + "/Data", "outputs")
    save_object(base_params, root_folder + "/Data", "base_params")
    print(f"All top 10 policies processed and saved in '{root_folder}'")

    ######################################################################################################
    #SINGLE POLICIES
    #
    # Reference trajectory for each instrument on its own, at the intensity the
    # pair-gen run endogenously solved for. Previously these two intensities were
    # hardcoded here (and again in low_policy_intensity_plot.py), so they silently
    # went stale every time the pair-gen was re-run against new calibration
    # parameters -- the single-policy lines in Figure 5 then described a different
    # model than the pairs drawn beside them.

    single_policy_intensities = load_single_policy_intensities(fileName_load)
    save_object(single_policy_intensities, root_folder + "/Data", "single_policy_intensities")

    for policy, save_name in SINGLE_POLICY_REFERENCES.items():
        intensity = single_policy_intensities[policy]
        print(f"Running single policy {policy} at intensity {intensity}")

        params_single = update_policy_intensity(deepcopy(base_params), policy, intensity)
        save_object(single_policy_with_seeds(params_single, controller_files),
                    root_folder + "/Data", save_name)
        print(f"DONE SINGLE {policy}")

    #######################################################################################################
    #DELETE CALIBRATION RUNS
    shutil.rmtree(Path(root_folder) / "Calibration_runs", ignore_errors=True)

if __name__ == "__main__":
    # fileNames = the results/endog_pair_<timestamp> folder(s) produced by
    # package.analysis.endogenous_policy_intensity_pair_gen, whose
    # Data/pairwise_outcomes files get merged here. Pass one or more on the
    # command line to override the hardcoded default below -- that default is
    # just the folder last used interactively and does not exist on a fresh
    # checkout, so submit_low_policy_intensity_gen.slurm always passes
    # $PAIRWISE_FOLDERS explicitly.
    file_names = sys.argv[1:] or ["results/endog_pair_15_10_05__13_08_2026"]
    print("Loading pairwise outcomes from:", file_names)
    main(
        fileNames=file_names,
        min_ev_uptake = 0.94,
        max_ev_uptake = 0.96#0.96
    )