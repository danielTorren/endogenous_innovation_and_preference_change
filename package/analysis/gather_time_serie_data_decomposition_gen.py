from package.resources.utility import load_object, save_object
import numpy as np

def main(fileName):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        controller = load_object(fileName + "/Data", "controller")
    except FileNotFoundError:
        print("Data files not found.")
        return

    social_network = controller.social_network
    policy_duration = base_params["duration_future"]

    outputs = {
        "history_ev_prop": np.array(social_network.history_EV_users[-policy_duration:]) / social_network.num_individuals,
        "history_mean_distance_EV": np.nanmean(np.asarray(social_network.history_distance_individual_EV[-policy_duration:]).T, axis=0),
        "history_mean_distance_ICE": np.nanmean(np.asarray(social_network.history_distance_individual_ICE[-policy_duration:]).T, axis=0),
        "history_e_EV": controller.electricity_emissions_intensity_vec[-policy_duration:],
        "history_e_ICE": np.asarray([controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]] * policy_duration),
        "history_new_ICE_cars_bought": np.asarray(social_network.history_new_ICE_cars_bought[-policy_duration:]),
        "history_new_EV_cars_bought": np.asarray(social_network.history_new_EV_cars_bought[-policy_duration:]),
        "history_num_individuals": np.asarray([social_network.num_individuals] * policy_duration),
        "history_production_emissions_ICE_parameter": np.asarray([controller.parameters_ICE["production_emissions"]] * policy_duration),
        "history_production_emissions_EV_parameter": np.asarray([controller.parameters_EV["production_emissions"]] * policy_duration),
        "history_driving_emissions_ICE": np.array(social_network.history_driving_emissions_ICE[-policy_duration:]),
        "history_driving_emissions_EV": np.array(social_network.history_driving_emissions_EV[-policy_duration:]),
        "history_total_emissions": np.asarray(social_network.history_total_emissions[-policy_duration:])
    }

    save_object(outputs, fileName + "/Data", "outputs")

if __name__ == "__main__":
    main("results/single_experiment_18_02_03__28_12_2024")