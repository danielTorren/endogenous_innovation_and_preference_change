import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object
import matplotlib.pyplot as plt
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.plotting_data.single_experiment_plot import save_and_show

def plot_ev_stock_multi_seed(base_params, real_data, simulated_data_seeds, fileName, dpi=600):
    
    data_truncated_seeds = []
    for data in simulated_data_seeds:
        data_truncated = convert_data(data, base_params)
        data_truncated_seeds.append(data_truncated)

    data_truncated_seeds_arr = np.asarray(data_truncated_seeds)

    mean = np.mean(data_truncated_seeds_arr, axis = 1)
    confidence = t.ppf(0.975, len(data_truncated_seeds_arr)-1) * sem(data_truncated_seeds_arr)
        
    lower_bounds = mean - confidence
    upper_bounds = mean + confidence

    # Create a grid of subplots (4x4 layout)
    fig, ax = plt.subplots(nrows=1,ncols=1,  figsize=(6, 6))
    ax.plot(mean, label="Simulated data", color='blue')
    ax.fill_between(lower_bounds, upper_bounds, color='blue', alpha=0.2)
    ax.plot(real_data, label = "California data")
    ax.set_xlabel("Months, 2010-2022")
    ax.set_ylabel("EV stock %")
    ax.legend(loc="best")
    save_and_show(fig, fileName, "plot_ev_stock", dpi)    


# Sample main function
def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_emissions = load_object(fileName + "/Data", "data_array_emissions")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object( "package/calibration_data", "calibration_data_output")
    #print(calibration_data_output)
    #quit()
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    plot_ev_stock_multi_seed(base_params, EV_stock_prop_2010_22, data_array_emissions, fileName, dpi)
    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_10_09_08__24_12_2024")