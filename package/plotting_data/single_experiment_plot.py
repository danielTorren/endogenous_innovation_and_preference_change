"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from package.model.firm_manager import Firm_Manager
from package.resources.utility import ( 
    load_object,
)
import pandas as pd
from package.resources.plot import (
    plot_low_carbon_preferences_timeseries,
    plot_total_flow_carbon_emissions_timeseries,
    plot_network_timeseries
)

def plot_emissions_individuals(fileName, data, dpi_save):

    y_title = r"Individuals' emissions flow"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    for v in range(data.num_individuals):
        data_indivdiual = data.agent_list[v]
        ax.plot(data.history_time_social_network,data_indivdiual.history_flow_carbon_emissions)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Plots"

    f = plotName + "/indi_emisisons_flow_timeseries_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_low_carbon_preferences_timeseries(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = r"Low carbon preference"

    fig, ax = plt.subplots(figsize=(10, 6))
    data_list = []
    for v in range(data.num_individuals):
        data_indivdiual = np.asarray(data.agent_list[v].history_low_carbon_preference)
        data_list.append(data_indivdiual)
        #print(data.history_time_social_network,data_indivdiual)
        ax.plot(data.history_time_social_network,data_indivdiual)

    data_list_array_t = np.asarray(data_list).T#t,n
    mean_data = np.mean(data_list_array_t, axis = 1)
    median_data = np.median(data_list_array_t, axis = 1)

    ax.plot(
            np.asarray(data.history_time_social_network),
            mean_data,
            label= "mean",
            linestyle="dotted"
        )
    ax.plot(
            np.asarray(data.history_time_social_network),
            median_data,
            label= "median",
            linestyle="dashed"
        )
    ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"

    f = plotName + "/timeseries_preference"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_firm_manager_timeseries(
    fileName: str, Data: Firm_Manager, y_title: str, property: str, dpi_save: int,latex_bool = False
):

    fig, ax = plt.subplots(figsize=(10,6))
    data = eval("Data.%s" % property)

    # bodge
    ax.plot(Data.history_time_firm_manager, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_firm_market_share(fileName: str, Data, dpi_save: int):

    y_title = "Firm market share"
    property = "history_market_share_vec"
    plot_firm_manager_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_emissions_intensity_firm(fileName: str, Data, dpi_save: int):

    y_title = "Firm emissions intensities"
    property = "history_emissions_intensities_vec"
    plot_firm_manager_timeseries(fileName, Data, y_title, property, dpi_save)

def  plot_frim_price(fileName: str, Data, dpi_save: int):

    y_title = "Frim prices (No carbon price)"
    property = "history_prices_vec"
    plot_firm_manager_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_firm_cost(fileName: str, Data, dpi_save: int):

    y_title = "Firm cost"
    property = "history_cost_vec"
    plot_firm_manager_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_firm_budget(fileName: str, Data, dpi_save: int):

    y_title = "Firm budget"
    property = "history_budget_vec"
    plot_firm_manager_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_firm_expected_carbon_premium_vec(fileName: str, Data, dpi_save: int):

    y_title = "Firm expected carbon premium vec"
    property = "history_expected_carbon_premium_vec"
    plot_firm_manager_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_demand_firm(fileName: str, Data, dpi_save: int):

    y_title = "Demand firms"
    property = "history_consumed_quantities_vec_firms"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_demand_individuals(fileName: str, Data, dpi_save: int):

    y_title = "Demand individuals"
    property = "history_consumed_quantities_vec"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_expenditure_individuals(fileName, data, dpi_save):

    y_title = r"Expenditure individuals"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    for v in range(data.num_individuals):
        data_indivdiual = data.agent_list[v]
        ax.plot(data.history_time_social_network,data_indivdiual.history_expenditure)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Plots"

    f = plotName + "/indi_expenditure"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_carbon_dividend_individuals(fileName, data, dpi_save):

    y_title = r"Carbon dividend individuals"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    for v in range(data.num_individuals):
        data_indivdiual = data.agent_list[v]
        ax.plot(data.history_time_social_network,data_indivdiual.history_carbon_dividend)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Plots"

    f = plotName + "/indi_carbon_dividend"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_search_range(fileName, data, dpi_save):

    y_title = r"search_range"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    for v in range(data.J):
        data_indivdiual = data.firms_list[v]
        ax.plot(data.history_time_firm_manager[:-1],data_indivdiual.history_search_range)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Plots"

    f = plotName + "/indi_search_range"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_firm_proft(fileName, data, dpi_save):

    y_title = r"firm profit"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    for v in range(data.J):
        data_indivdiual = data.firms_list[v]
        ax.plot(data.history_time_firm_manager[:-1],data_indivdiual.history_profit)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Plots"

    f = plotName + "/indi_firm_profit"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_flow_emissions_firm(fileName: str, data_social_network, data_firm_manager):

    y_title = r"Firm Emissions flow"

    fig, ax = plt.subplots(figsize=(10,6))

    data_demand = np.asarray(data_social_network.history_consumed_quantities_vec_firms)
    data_emisisons_intensity = np.asarray(data_firm_manager.history_emissions_intensities_vec)

    data = data_demand*data_emisisons_intensity
    # bodge
    ax.plot(data_social_network.history_time_social_network, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/" + "emissions_firm_flow_timeseries"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_cumulative_emissions_firm(fileName: str, data_social_network, data_firm_manager):

    y_title = r"Firm Emissions cumulative"

    fig, ax = plt.subplots(figsize=(10,6))

    data_demand = np.asarray(data_social_network.history_consumed_quantities_vec_firms)
    data_emisisons_intensity = np.asarray(data_firm_manager.history_emissions_intensities_vec)

    flow_data = data_demand*data_emisisons_intensity
    #print(flow_data.shape)
    data = np.cumsum(flow_data, axis=0)#sum along time axis
    #print(data,data.shape)
    #quit()

    # bodge
    ax.plot(data_social_network.history_time_social_network, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/" + "emissions_cumsum_timeseries"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_len_indices_higher(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = r"len indices higher"

    fig, ax = plt.subplots(figsize=(10, 6))
    data_list = []
    for v in range(data.J):
        data_indivdiual = np.asarray(data.firms_list[v].history_indices_higher)
        data_list.append(data_indivdiual)
        #print(data.history_time_social_network,data_indivdiual)
        ax.plot(data.history_time_firm_manager[:-1],data_indivdiual)
      
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"

    f = plotName + "/timeseries_len_indices_higher"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    social_plots = 1,
    firm_plots = 1
    ) -> None: 

    data_social_network = load_object(fileName + "/Data", "social_network")
    data_firm_manager = load_object(fileName + "/Data", "firm_manager")

    if social_plots:
        ###SOCIAL NETWORK PLOTS
        #THERES A BUINCH MORE IN PLOT.PY BUT PUT THEM HERE FOR NOW JUST TO SEPERATE
        plot_low_carbon_preferences_timeseries(fileName, data_social_network, dpi_save)
        #plot_emissions_individuals(fileName, data_social_network, dpi_save)
        #plot_total_flow_carbon_emissions_timeseries(fileName, data_social_network, dpi_save)
        #plot_demand_individuals(fileName, data_social_network, dpi_save)
        #plot_expenditure_individuals(fileName, data_social_network, dpi_save)
        #plot_carbon_dividend_individuals(fileName, data_social_network, dpi_save)
        

    if firm_plots:
        ##FIRM PLOTS
        #plot_len_indices_higher(fileName, data_firm_manager, dpi_save)
        plot_firm_market_share(fileName, data_firm_manager, dpi_save)
        #plot_frim_price(fileName, data_firm_manager, dpi_save)
        plot_firm_cost(fileName, data_firm_manager, dpi_save)
        #plot_search_range(fileName, data_firm_manager, dpi_save)
        plot_firm_proft(fileName, data_firm_manager, dpi_save)
        plot_firm_budget(fileName, data_firm_manager, dpi_save)
        plot_firm_expected_carbon_premium_vec(fileName, data_firm_manager, dpi_save)
        #plot_demand_firm(fileName, data_social_network, dpi_save)
        plot_emissions_intensity_firm(fileName, data_firm_manager, dpi_save)
        #plot_flow_emissions_firm(fileName, data_social_network,data_firm_manager)
        #plot_cumulative_emissions_firm(fileName, data_social_network, data_firm_manager)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_15_48_13__13_02_2024",
    )


