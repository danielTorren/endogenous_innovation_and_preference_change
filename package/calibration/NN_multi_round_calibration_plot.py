from sbi import analysis as analysis #https://github.com/sbi-dev/sbi
from sbi import utils as utils
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object
)
from package.plotting_data.single_experiment_plot import save_and_show

def plot_results(fileName, x_o,posterior, param_1_bounds, param_2_bounds, param_1_name,param_2_name):
    posterior_samples = posterior.sample((100000,), x=x_o)#GET A BUNCH OF GUESSES AT THE POSTERIOR BASED ON THE OBSERVATIONS, WHAT DOES IT THINK THE VALUE IS
    # Plot posterior samples
    fig, ax = pairplot(
        posterior_samples,
        limits=[param_1_bounds, param_2_bounds],
        figsize=(5, 5),
        points_colors='r',
        labels=[param_1_name,param_2_name]
    )
    save_and_show(fig, fileName, "pairplot", dpi=300)
    #plt.savefig('Figures/posterior_samples_synthetic.png')
    plt.show()

def main(
        fileName,
        OUTPUTS_LOAD_ROOT, 
        OUTPUTS_LOAD_NAME
    ) -> str: 

    calibration_data_output = load_object(OUTPUTS_LOAD_ROOT, OUTPUTS_LOAD_NAME)
    EV_stock_percentage_2010_22 = calibration_data_output["EV Percentage"]
    x_o = EV_stock_percentage_2010_22

    posterior = load_object(fileName + "/Data", "posterior")
    var_dict = load_object(fileName + "/Data", "var_dict")

    param_1_bounds = var_dict["param_1_bounds"]
    param_2_bounds = var_dict["param_2_bounds"]
    param_1_name = var_dict["param_1_name"]
    param_2_name = var_dict["param_2_name"]

    ########################################################################################################
    #TEST NN

    plot_results(fileName, x_o,posterior, param_1_bounds, param_2_bounds, param_1_name,param_2_name)

if __name__ == "__main__":

    #EV_percentage_2010_2022 = [0.003446, 0.026368, 0.081688, 0.225396, 0.455980, 0.680997, 0.913118, 1.147275, 1.583223, 1.952829, 2.217273, 2.798319, 3.791804, 5.166498]
    main(
        fileName = "results/NN_calibration_11_57_19__16_12_2024",
        OUTPUTS_LOAD_ROOT = "package/calibration_data",
        OUTPUTS_LOAD_NAME = "calibration_data_output"
        )
    