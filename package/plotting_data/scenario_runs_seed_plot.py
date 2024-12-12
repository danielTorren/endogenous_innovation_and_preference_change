# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.resources.utility import load_object

def plot_with_confidence_interval(ax, data, label, confidence=0.95):
    """
    Plot the mean and shaded confidence interval for the given data.
    """
    data = np.array(data)
    mean = data.mean(axis=0)
    error = sem(data, axis=0) * t.ppf((1 + confidence) / 2, df=data.shape[0] - 1)
    
    ax.plot(mean, label=label)
    ax.fill_between(range(len(mean)), mean - error, mean + error, alpha=0.3)

def plot_emissions_flow(fileName, emissions_time_series_reshaped):
    """
    Plot emissions flow with mean and 95% confidence interval.
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), sharey=True)

    for i, emissions_group in enumerate(emissions_time_series_reshaped):
        plot_with_confidence_interval(ax, emissions_group, label=f"Scenario {i}")

    ax.set_title("Scenarios: Emissions Flow", fontsize="12")
    ax.set_ylabel(r"Flow carbon emissions, $E_{F,t}$", fontsize="12")
    ax.set_xlabel(r"Time", fontsize="12")
    ax.legend()

    # Save the figure
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_flow"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_emissions_cum(fileName, emissions_time_series_reshaped):
    """
    Plot cumulative emissions with mean and 95% confidence interval.
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), sharey=True)

    for i, emissions_group in enumerate(emissions_time_series_reshaped):
        cumulative_emissions = np.cumsum(emissions_group, axis=1)
        plot_with_confidence_interval(ax, cumulative_emissions, label=f"Scenario {i}")

    ax.set_title("Scenarios: Cumulative Emissions", fontsize="12")
    ax.set_ylabel(r"Cumulative carbon emissions, $E_{C,t}$", fontsize="12")
    ax.set_xlabel(r"Time", fontsize="12")
    ax.legend()

    # Save the figure
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_cum"
    fig.savefig(f + ".png", dpi=600, format="png")

def main(fileName) -> None:
    #emissions_time_series_reshaped = load_object(fileName + "/Data", "data_reshaped")
    emissions_time_series_reshaped = load_object(fileName + "/Data", "data_array")
    #base_params = load_object(fileName + "/Data", "base_params")
    
    plot_emissions_flow(fileName, emissions_time_series_reshaped)
    plot_emissions_cum(fileName, emissions_time_series_reshaped)
    plt.show()

if __name__ == "__main__":
    plots = main(
        fileName="results/scenarios_combo_16_07_43__12_12_2024",
    )
