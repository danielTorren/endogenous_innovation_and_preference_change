"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    save_object,
    calc_bounds
)
from matplotlib.cm import rainbow
import matplotlib.pyplot as plt

def plot_scatter_end_points_emissions_scatter(
    fileName, emissions, scenarios_titles, property_vals, title, save_name
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    #print(len(emissions))
    colors = iter(rainbow(np.linspace(0, 1,len(emissions))))

    for i in range(len(emissions)):
        
        color = next(colors)#set color for whole scenario?
        data = emissions[i].T#its now seed then tax
        #print("data",data)
        for j in range(len(data)):
            ax.scatter(property_vals,  data[j], color = color, label=scenarios_titles[i] if j == 0 else "")

    ax.legend()
    ax.set_xlabel(r"Carbon Tax, $\tau$")
    ax.set_ylabel(title)

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/scatter_carbon_tax_emissions_" + save_name
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_means_end_points_emissions(
    fileName, emissions, scenarios_titles, property_vals, title, save_name
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)

    colors = iter(rainbow(np.linspace(0, 1, len(emissions))))

    for i in range(len(emissions)):
        color = next(colors)#set color for whole scenario?
        Data = emissions[i]
        #print("Data", Data.shape)
        mu_emissions, min_emissions, max_emissions = calc_bounds(Data, 0.95)
        #mu_emissions =  Data.mean(axis=1)
        #min_emissions =  Data.min(axis=1)
        #max_emissions=  Data.max(axis=1)

        #print("mu_emissions",mu_emissions)
        ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
        ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.5)

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(title)

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_means_end_points_emissions_" + save_name
    fig.savefig(f+ ".png", dpi=600, format="png") 


def plot_emissions_lines(
    fileName, emissions, scenarios_titles, property_vals, title, save_name
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)

    colors = iter(rainbow(np.linspace(0, 1, len(emissions))))

    for i in range(len(emissions)):
        color = next(colors)#set color for whole scenario?
        Data = emissions[i]
        #print("Data", Data.shape)
        mu_emissions, min_emissions, max_emissions = calc_bounds(Data, 0.95)
        #mu_emissions =  Data.mean(axis=1)
        #min_emissions =  Data.min(axis=1)
        #max_emissions=  Data.max(axis=1)

        #print("mu_emissions",mu_emissions)
        ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
        #ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.5)
        for j, data_seed in enumerate(Data.T):
            ax.plot(property_vals, data_seed, c= color, alpha = 0.2)

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(title)

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_lines_" + save_name
    fig.savefig(f+ ".png", dpi=600, format="png") 


def main(
    fileName = "results/",
) -> None:
        
    emissions_cumulative = load_object(fileName + "/Data","emissions_cumulative")
    #print()
    #print("emissions_cumulative",emissions_cumulative,emissions_cumulative.shape)
    #quit()
    #weighted_emissions_intensities = load_object(fileName + "/Data","weighted_emissions_intensities")


    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    print(base_params)
    scenarios = [1,2]#load_object(fileName + "/Data", "scenarios")
    #print(scenarios)

    scenarios_titles = ["No preference change, No innovation","Preference change, No innovation","No preference change, Innovation","Preference change, Innovation"]

    seed_reps = base_params["seed_reps"]
    #emissions plot
    plot_emissions_lines(fileName, emissions_cumulative, scenarios_titles ,property_values_list,"Cumulative carbon emissions, E", "carbon_emissions")
    plot_scatter_end_points_emissions_scatter(fileName, emissions_cumulative, scenarios_titles ,property_values_list, "Cumulative carbon emissions, E", "carbon_emissions")
    plot_means_end_points_emissions(fileName, emissions_cumulative, scenarios_titles ,property_values_list,"Cumulative carbon emissions, E", "carbon_emissions")

    #
    #plot_scatter_end_points_emissions_scatter(fileName, weighted_emissions_intensities, scenarios_titles ,property_values_list,"Weighted emissions intensities, EI", "weighted_emissions_intensities")
    #plot_means_end_points_emissions(fileName, weighted_emissions_intensities, scenarios_titles ,property_values_list,"Weighted emissions intensities, EI", "weighted_emissions_intensities")

    plt.show()
if __name__ == "__main__":
    plots = main(
        fileName = "results/tax_sweep_00_32_24__19_06_2024",
    )