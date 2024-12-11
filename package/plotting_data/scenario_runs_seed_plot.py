# imports
from xml.dom import EMPTY_PREFIX
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object
)

def plot_emissions_flow(
    fileName, emissions_time_series
):
    # First figure: only the first column
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), sharey=True)

    for i,emissions in enumerate(emissions_time_series):
        ax1.plot(emissions, label = i)

    ax1.set_title("Scenarios: Emissions flow", fontsize="12")
    ax1.set_ylabel(r"Flow carbon emissions, $E_{F,t}$", fontsize="12")
    ax1.set_xlabel(r"Time", fontsize="12")
    ax1.legend()

    # Save the first figure
    plotName = fileName + "/Plots"
    f1 = plotName + "/plot_emissions_flow"
    fig1.savefig(f1 + ".png", dpi=600, format="png")

def plot_emissions_cum(
    fileName, emissions_time_series
):
    # First figure: only the first column
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), sharey=True)

    for i,emissions in enumerate(emissions_time_series):
        emissions_cum = np.cumsum(emissions)
        ax1.plot(emissions_cum, label = i)

    ax1.set_title("Scenarios: Cumulative emissions", fontsize="12")
    ax1.set_ylabel(r"Cumulative carbon emissions, $E_{C,t}$", fontsize="12")
    ax1.set_xlabel(r"Time", fontsize="12")
    ax1.legend()

    # Save the first figure
    plotName = fileName + "/Plots"
    f1 = plotName + "/plot_emissions_cum"
    fig1.savefig(f1 + ".png", dpi=600, format="png")

def main(
    fileName
) -> None:

    emissions_time_series = load_object(fileName + "/Data","data_flat")    
    base_params = load_object(fileName + "/Data","base_params")
    
    plot_emissions_flow(fileName, emissions_time_series)
    plot_emissions_cum(fileName, emissions_time_series)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/scenarios_combo_22_49_42__10_12_2024",
    )