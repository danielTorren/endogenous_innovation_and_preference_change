"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.animation as animation 
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

def burn_in(ax,data):
    if data.burn_in_no_OD > 0:
        ax.axvline(x = data.burn_in_no_OD,ls='--',  color='black')#OD
    
    if data.burn_in_duration_no_policy > 0:
        ax.axvline(x = (data.burn_in_no_OD+data.burn_in_duration_no_policy),ls='--',  color='black')#OD

def plot_emissions_individuals(fileName, data, dpi_save):

    y_title = r"Individuals' emissions flow"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    burn_in(ax,data)

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
    
    burn_in(ax,data)

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


def plot_outward_social_influence_timeseries(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = r"Outward social influence"

    fig, ax = plt.subplots(figsize=(10, 6))
    
    burn_in(ax,data)

    data_list = []
    for v in range(data.num_individuals):
        data_indivdiual = np.asarray(data.agent_list[v].history_outward_social_influence)
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

    f = plotName + "/plot_outward_social_influence_timeseries"
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

def plot_firm_segment_index_max_profit(fileName, data, dpi_save):

    y_title = r"firm segment"

    fig, ax = plt.subplots(constrained_layout=True,figsize=(10, 6))

    for v in range(data.J):
        data_indivdiual = data.firms_list[v]
        ax.plot(data.history_time_firm_manager[:-1],data_indivdiual.history_segment_index_max_profit)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Plots"

    f = plotName + "/indi_firm_history_segment_index_max_profit"
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

def final_scatter_price_EI(
    fileName, 
    data, 
    dpi_save,
    ):

    fig, ax = plt.subplots(figsize=(10,6)) 

    # bodge
    scatter = ax.scatter(data.history_prices_vec[-1], data.history_emissions_intensities_vec[-1],  c=data.history_market_share_vec[-1], cmap='viridis', alpha=0.9, edgecolors='black')

    # Add colorbar to indicate market share
    cbar = fig.colorbar(scatter)
    cbar.set_label('Market Share')

    ax.set_xlabel(r"Emissions intensities, ei")
    ax.set_ylabel(r"Price, p")
    ax.set_title("Price-Emissions intensity correlation $\\rho$ = %s" % (data.rho))

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/scatter_ei_price_market_share"
    #fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def ani_scatter_price_EI(fileName, data, dpi_save):
    fig, ax = plt.subplots(figsize=(10,6)) 
        # Calculate the min and max of market share data
    
    min_price, max_price = np.min(data.history_prices_vec), np.max(data.history_prices_vec)
    min_ei, max_ei = np.min(data.history_emissions_intensities_vec), np.max(data.history_emissions_intensities_vec)

    #min_price, max_price = np.floor(np.min(data.history_market_share_vec)), np.ceil(np.min(data.history_market_share_vec))
    #min_ei, max_ei = np.floor(np.min(data.history_market_share_vec)), np.ceil(np.min(data.history_market_share_vec))

    min_market_share = round(0.95*np.min(data.history_market_share_vec), 3)
    max_market_share = round(1.05*np.max(data.history_market_share_vec), 3)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=get_cmap("plasma"), norm=Normalize(vmin=min_market_share, vmax=max_market_share)), ax=ax
        )
    cbar.set_label('Market Share')

    def update(frame):
        ax.clear()
        ax.set_xlim(min_price, max_price)
        ax.set_ylim(min_ei, max_ei)
        scatter = ax.scatter(data.history_prices_vec[frame], data.history_emissions_intensities_vec[frame],  c=data.history_market_share_vec[frame], cmap='viridis', alpha=0.9, edgecolors='black')
        ax.set_xlabel(r"Price, p")
        ax.set_ylabel(r"Emissions intensities, ei")
        ax.set_title("Price-Emissions intensity correlation $\\rho$ = %s" % (data.rho))
        
        # Clear the color bar axis
        #cbar = fig.colorbar(scatter, ax=ax)
        #cbar.remove()
        # Set colorbar limits
        #scatter.set_clim(vmin=min_market_share, vmax=max_market_share)
        #fig.colorbar(scatter, ax=ax).set_label('Market Share')

        # Add step counter
        ax.annotate('Step: %d/%s' % (data.history_time_firm_manager[frame], data.history_time_firm_manager[-1]), xy=(0.75, 0.95), xycoords='axes fraction', fontsize=12, color='black')

    ani = animation.FuncAnimation(fig, update, frames=len(data.history_time_firm_manager), interval=1)
    # saving to m4 using ffmpeg writer 
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save(fileName + '/Animations/scatter_price_EI.mp4', writer=writervideo) 
    #ni.save(fileName + '/Animations/scatter_ei_price_market_share_animation.gif', dpi=dpi_save)
    print("done")
    #plt.close()
    return ani
    

# Example usage:
# final_scatter_price_EI('path_to_save', your_data_object, 100)



def main(
    fileName = "results/single_experiment_15_05_51__26_02_2024",
    dpi_save = 600,
    social_plots = 1,
    firm_plots = 1
    ) -> None: 

    #social_plots = 0
    #firm_plots = 0

    data_social_network = load_object(fileName + "/Data", "social_network")
    data_firm_manager = load_object(fileName + "/Data", "firm_manager")

    if social_plots:
        ###SOCIAL NETWORK PLOTS
        #THERES A BUINCH MORE IN PLOT.PY BUT PUT THEM HERE FOR NOW JUST TO SEPERATE
        plot_low_carbon_preferences_timeseries(fileName, data_social_network, dpi_save)
        plot_outward_social_influence_timeseries(fileName, data_social_network, dpi_save)
        #plot_emissions_individuals(fileName, data_social_network, dpi_save)
        #plot_total_flow_carbon_emissions_timeseries(fileName, data_social_network, dpi_save)
        plot_demand_individuals(fileName, data_social_network, dpi_save)
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
        #plot_firm_expected_carbon_premium_vec(fileName, data_firm_manager, dpi_save)
        #plot_demand_firm(fileName, data_social_network, dpi_save)
        plot_emissions_intensity_firm(fileName, data_firm_manager, dpi_save)
        #plot_firm_segment_index_max_profit(fileName, data_firm_manager, dpi_save)
        #plot_flow_emissions_firm(fileName, data_social_network,data_firm_manager)
        #plot_cumulative_emissions_firm(fileName, data_social_network, data_firm_manager)

    #final_scatter_price_EI(fileName, data_firm_manager, dpi_save)
    #ani_1 = ani_scatter_price_EI(fileName, data_firm_manager, dpi_save)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_17_23_48__14_02_2024",
    )


