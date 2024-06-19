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

def plot_data(fileName, preferences, carbon_price_vals, stochastic_seed_reps , number_of_individuals):
    # Flatten the data for scatter plot
    carbon_price_reps = len(carbon_price_vals)
    carbon_prices = np.repeat(carbon_price_vals, stochastic_seed_reps * number_of_individuals)
    preferences_flat = preferences.flatten()

    """
    # Plot scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x=carbon_prices, y=preferences_flat, alpha=0.6)
    ax.set_xlabel('Carbon Price')
    ax.set_ylabel('Preference')
    ax.set_title('Scatter Plot of Preferences vs. Carbon Price')
    """
    # Calculate mean and 95% confidence intervals
    means = np.mean(preferences, axis=(1, 2))
    std_devs = np.std(preferences, axis=(1, 2))
    confidence_intervals = 1.96 * std_devs / np.sqrt(stochastic_seed_reps * number_of_individuals)

    # Plot mean and 95% confidence intervals
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.errorbar(carbon_price_vals, means, yerr=confidence_intervals, fmt='o', capsize=5)
    ax1.set_xlabel('Carbon Price')
    ax1.set_ylabel('Mean Preference')
    ax1.set_title('Mean Preferences and 95% Confidence Intervals')

    #plotName = fileName + "/Plots"
    #f = plotName + "/scatter_tau" 
    #fig.savefig(f+ ".png", dpi=600, format="png")  

    plotName = fileName + "/Plots"
    f = plotName + "/scatter_tau_density" 
    fig1.savefig(f+ ".png", dpi=600, format="png")  
             
def main(
    fileName = "results/",
) -> None:
        
    preferences_distribution = load_object(fileName + "/Data","preference_distribution")

    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 

    seed_reps = base_params["seed_reps"]
    number_of_individuals = base_params["parameters_social_network"]["num_individuals"]
    #emissions plot

    plot_data(fileName, preferences_distribution, property_values_list, seed_reps , number_of_individuals)

    plt.show()
if __name__ == "__main__":
    plots = main(
        fileName = "results/tax_sweep_16_11_26__19_06_2024",
    )