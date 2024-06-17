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
    plot_network_timeseries
)
import matplotlib.lines as mlines

def burn_in(ax,data):
    if data.duration_no_OD_no_stock_no_policy > 0:
        ax.axvline(x = data.duration_no_OD_no_stock_no_policy,ls="--",  color="black")#OD
    
    if data.duration_OD_no_stock_no_policy > 0:
        ax.axvline(x = (data.duration_no_OD_no_stock_no_policy+data.duration_OD_no_stock_no_policy),ls="--",  color="black")#OD

    if data.duration_OD_stock_no_policy > 0:
        ax.axvline(x = (data.duration_no_OD_no_stock_no_policy+data.duration_OD_no_stock_no_policy + data.duration_OD_stock_no_policy),ls="--",  color="black")#OD
    
    #if data.duration_OD_stock_policy > 0:
    #    ax.axvline(x = (data.duration_no_OD_no_stock_no_policy+data.duration_OD_no_stock_no_policy + data.duration_OD_stock_no_policy + data.duration_OD_stock_policy ),ls="--",  color="black")#OD

def plot_emissions_individuals(fileName, data, dpi_save):

    y_title = r"Individuals emissions flow"

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

def plot_low_carbon_preference(fileName, data):
    fig, ax = plt.subplots(figsize=(10, 6))
    burn_in(ax,data)
    #burn_in(ax,data)
    data_t = np.asarray(data.history_preference_list).T
    for data_indivdiual in data_t:
        ax.plot(data.history_time_social_network,data_indivdiual)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Low carbon preference")

    plotName = fileName + "/Plots"

    f = plotName + "/timeseries_preference"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_car_utility(fileName, data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #burn_in(ax,data)
    data_t = np.asarray(data.history_utility_vec).T
    for data_indivdiual in data_t:
        ax.plot(data.history_time_social_network,data_indivdiual)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Owned car utility")

    plotName = fileName + "/Plots"

    f = plotName + "/timeseries_car_util"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


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

def plot_emissions_firm(fileName: str, Data, dpi_save: int):

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
    scatter = ax.scatter(data.history_prices_vec[-1], data.history_emissions_intensities_vec[-1],  c=data.history_market_share_vec[-1], cmap="viridis", alpha=0.9, edgecolors="black")

    # Add colorbar to indicate market share
    cbar = fig.colorbar(scatter)
    cbar.set_label("Market Share")

    ax.set_xlabel(r"Price, p")
    ax.set_ylabel(r"Emissions intensities, ei")
    ax.set_title("Price-Emissions intensity correlation $\\rho$ = %s" % (data.rho))

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/scatter_ei_price_market_share"
    #fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def final_scatter_price_EI_alt(
    fileName, 
    data, 
    dpi_save,
    ):

    fig, ax = plt.subplots(figsize=(10,6)) 

    # bodge
    scatter = ax.scatter(data.history_market_share_vec[-1], data.history_emissions_intensities_vec[-1],  c=data.history_prices_vec[-1], cmap="plasma", alpha=0.9, edgecolors="black")

    # Add colorbar to indicate market share
    cbar = fig.colorbar(scatter)
    cbar.set_label(r"Price, p")

    ax.set_xlabel("Market Share")
    ax.set_ylabel(r"Emissions intensities, ei")
    ax.set_title("Price-Emissions intensity correlation $\\rho$ = %s" % (data.rho))

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/scatter_ei_price_market_share_alt"
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
    cbar.set_label("Market Share")

    def update(frame):
        ax.clear()
        ax.set_xlim(min_price, max_price)
        ax.set_ylim(min_ei, max_ei)
        scatter = ax.scatter(data.history_prices_vec[frame], data.history_emissions_intensities_vec[frame],  c=data.history_market_share_vec[frame], cmap="viridis", alpha=0.9, edgecolors="black")
        ax.set_xlabel(r"Price, p")
        ax.set_ylabel(r"Emissions intensities, ei")
        ax.set_title("Price-Emissions intensity correlation $\\rho$ = %s" % (data.rho))
        
        # Clear the color bar axis
        #cbar = fig.colorbar(scatter, ax=ax)
        #cbar.remove()
        # Set colorbar limits
        #scatter.set_clim(vmin=min_market_share, vmax=max_market_share)
        #fig.colorbar(scatter, ax=ax).set_label("Market Share")

        # Add step counter
        ax.annotate("Step: %d/%s" % (data.history_time_firm_manager[frame], data.history_time_firm_manager[-1]), xy=(0.75, 0.95), xycoords="axes fraction", fontsize=12, color="black")

    ani = animation.FuncAnimation(fig, update, frames=len(data.history_time_firm_manager), interval=1)
    # saving to m4 using ffmpeg writer 
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save(fileName + "/Animations/scatter_price_EI.mp4", writer=writervideo) 
    #ni.save(fileName + "/Animations/scatter_ei_price_market_share_animation.gif", dpi=dpi_save)
    print("done")
    #plt.close()
    return ani
    

# Example usage:
# final_scatter_price_EI("path_to_save", your_data_object, 100)

def plot_arbon_price_AR1(
    fileName, 
    data, 
    ):

    fig, ax = plt.subplots(figsize=(10,6)) 
    ax.plot(data.carbon_price_AR1)

    plotName = fileName + "/Plots"
    f = plotName + "/ar1 time_series"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_firm_count(fileName,data_social_network):
    # Initialize a dictionary to store the data
    data = {}

    # Extract data
    for time_point, snapshot in enumerate(data_social_network.history_firm_count):
        for firm, cars in snapshot.items():
            if firm not in data:
                data[firm] = []
            total_cars_sold = sum(cars.values())
            data[firm].append((time_point, total_cars_sold))

    # Convert to a DataFrame
    df = pd.DataFrame()
    for firm, sales in data.items():
        times, car_sales = zip(*sales)
        df[firm] = pd.Series(car_sales, index=times)

    # Fill missing values with 0 (if any firm didn't sell cars at some time points)
    df = df.fillna(0)

    #print(df)

    # Plot the data
    fig, ax = plt.subplots(figsize=(10,6)) 
    for firm in df.columns:
        ax.scatter(df.index, df[firm], marker='o')

    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Cars Sold')
    ax.set_title('Number of Cars Sold by Each Firm Over Time')
    #ax.legend()
    ax.grid(True)
    #ax.set_xlim(0,1000)
    
    plotName = fileName + "/Plots"
    f = plotName + "/firm_count"
    fig.savefig(f + ".png", dpi=600, format="png")

def weighted_bought_average_plots(fileName,data_social_network,data_firm_manager):

    time_series_demand = data_social_network.history_firm_count
    time_series_cars = data_firm_manager.history_cars_on_sale_all_firms

    # Initialize data structures to hold weighted averages over time
    weighted_averages = {
        'environmental_score': [],
        'cost': [],
        'quality': []
    }

    # Calculate weighted averages for each time step
    for demand_snapshot, car_snapshot in zip(time_series_demand, time_series_cars):
        total_demand = sum(sum(cars.values()) for cars in demand_snapshot.values())
        if total_demand == 0:
            weighted_averages['environmental_score'].append(None)
            weighted_averages['cost'].append(None)
            weighted_averages['quality'].append(None)
            continue

        weighted_emissions_sum = 0
        weighted_cost_sum = 0
        weighted_quality_sum = 0

        for car in car_snapshot:
            car_demand = sum(demand_snapshot[firm].get(car.id, 0) for firm in demand_snapshot)
            if car_demand > 0:
                weighted_emissions_sum += car.environmental_score * car_demand
                weighted_cost_sum += car.cost * car_demand
                weighted_quality_sum += car.quality * car_demand

        weighted_averages['environmental_score'].append(weighted_emissions_sum / total_demand)
        weighted_averages['cost'].append(weighted_cost_sum / total_demand)
        weighted_averages['quality'].append(weighted_quality_sum / total_demand)

    # Convert to DataFrame for easier plotting
    df_weighted_averages = pd.DataFrame(weighted_averages)

    #print(df_weighted_averages)

    # Plot the data using subfigures
    fig, axes = plt.subplots(nrows= 1, ncols = 3,figsize=(10,6)) 

    subfig1 = axes[0]
    subfig2 = axes[1]
    subfig3 = axes[2]

    # Subfigure for emissions
    subfig1.scatter(df_weighted_averages.index, df_weighted_averages['environmental_score'], marker='o')
    subfig1.set_xlabel('Time')
    subfig1.set_ylabel('Weighted Average Emissions - Bought')
    subfig1.set_title('Environmental Score')
    subfig1.grid(True)
    #subfig1.set_xlim(0,1000)

    # Subfigure for cost
    subfig2.scatter(df_weighted_averages.index, df_weighted_averages['cost'], marker='*')
    subfig2.set_xlabel('Time')
    subfig2.set_ylabel('Weighted Average Cost - Bought')
    subfig2.set_title('Cost')
    subfig2.grid(True)
    #subfig2.set_xlim(0,1000)

    # Subfigure for quality
    subfig3.scatter(df_weighted_averages.index, df_weighted_averages['quality'], marker='x')
    subfig3.set_xlabel('Time')
    subfig3.set_ylabel('Weighted Average Quality - Bought')
    subfig3.set_title('Quality')
    subfig3.grid(True)
    #subfig3.set_xlim(0,1000)

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/bought_weighted_vals"
    fig.savefig(f + ".png", dpi=600, format="png")

def weighted_owned_average_plots(fileName,data_social_network):

    time_series_cars = data_social_network.history_car_owned_vec

    # Initialize data structures to hold weighted averages over time
    weighted_averages = {
        'environmental_score': [],
        'cost': [],
        'quality': []
    }

    for snapshot in time_series_cars:
        attributes_matrix = np.asarray([car.attributes_fitness  for car in snapshot])
        weighted_averages['environmental_score'].append(np.mean(attributes_matrix[:,1]))
        weighted_averages['cost'].append(np.mean(attributes_matrix[:,0]))
        weighted_averages['quality'].append(np.mean(attributes_matrix[:,2]))

    # Convert to DataFrame for easier plotting
    df_weighted_averages = pd.DataFrame(weighted_averages)

    # Plot the data using subfigures
    fig, axes = plt.subplots(nrows= 1, ncols = 3,figsize=(10,6)) 

    subfig1 = axes[0]
    subfig2 = axes[1]
    subfig3 = axes[2]

    # Subfigure for emissions
    subfig1.scatter(data_social_network.history_time_social_network, df_weighted_averages['environmental_score'], marker='o')
    subfig1.set_xlabel('Time')
    subfig1.set_ylabel('Weighted Average Emissions - Owned')
    subfig1.set_title('Emissions')
    subfig1.grid(True)

    # Subfigure for cost
    subfig2.scatter(data_social_network.history_time_social_network, df_weighted_averages['cost'], marker='*')
    subfig2.set_xlabel('Time')
    subfig2.set_ylabel('Weighted Average Cost - Owned')
    subfig2.set_title('Cost')
    subfig2.grid(True)

    # Subfigure for quality
    subfig3.scatter(data_social_network.history_time_social_network, df_weighted_averages['quality'], marker='x')
    subfig3.set_xlabel('Time')
    subfig3.set_ylabel('Weighted Average Quality - Owned')
    subfig3.set_title('Quality')
    subfig3.grid(True)

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/owned_weighted_vals"
    fig.savefig(f + ".png", dpi=600, format="png")

def offered_plots(fileName,data_firm_manager, data_social_network):

    time_series_cars = data_firm_manager.history_cars_on_sale_all_firms

    # Initialize data structures to hold weighted averages over time
    weighted_averages_em = []
    weighted_averages_cost = []
    weighted_averages_quality = []
    for snapshot in time_series_cars:
        attributes_matrix = np.asarray([car.attributes_fitness  for car in snapshot])
        weighted_averages_em.append(attributes_matrix[:,1])
        weighted_averages_cost.append(attributes_matrix[:,0])
        weighted_averages_quality.append(attributes_matrix[:,2])

    arr_e = np.asarray(weighted_averages_em).T
    arr_c = np.asarray(weighted_averages_cost).T
    arr_q = np.asarray(weighted_averages_quality).T

    # Plot the data using subfigures
    fig, axes = plt.subplots(nrows= 1, ncols = 3,figsize=(10,6)) 

    subfig1 = axes[0]
    subfig2 = axes[1]
    subfig3 = axes[2]

    for i, vec_e  in enumerate(arr_e):
        # Subfigure for emissions
        subfig1.scatter(data_social_network.history_time_social_network, arr_e[i], marker='o')
        subfig1.set_xlabel('Time')
        subfig1.set_ylabel('Emissions - Offered')
        subfig1.set_title('Emissions')
        subfig1.grid(True)

        # Subfigure for cost
        subfig2.scatter(data_social_network.history_time_social_network, arr_c[i], marker='*')
        subfig2.set_xlabel('Time')
        subfig2.set_ylabel('Average Cost - Offered')
        subfig2.set_title('Cost')
        subfig2.grid(True)

        # Subfigure for quality
        subfig3.scatter(data_social_network.history_time_social_network, arr_q[i], marker='x')
        subfig3.set_xlabel('Time')
        subfig3.set_ylabel(' Average Quality - Offered')
        subfig3.set_title('Quality')
        subfig3.grid(True)

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/offered_vals"
    fig.savefig(f + ".png", dpi=600, format="png")

def researched_tech(fileName,data_firm_manager, data_social_network):

    time_series_cars = data_firm_manager.history_researched_tech

    # Initialize data structures to hold weighted averages over time
    weighted_averages_em = []
    weighted_averages_cost = []
    weighted_averages_quality = []

    for snapshot in time_series_cars:
        row_e = []
        row_c = []
        row_q = []
        for car in snapshot:
            attributes_matrix = car.attributes_fitness
            print()
            #print("attributes_matrix", attributes_matrix)
            row_e.append(attributes_matrix[1])
            row_c.append(attributes_matrix[0])
            row_q.append(attributes_matrix[2])
        if len(snapshot) < data_firm_manager.J:
            row_e.extend([None]*(data_firm_manager.J - len(snapshot)))
            row_c.extend([None]*(data_firm_manager.J - len(snapshot)))
            row_q.extend([None]*(data_firm_manager.J - len(snapshot)))
        #print(len(row_e))
        weighted_averages_em.append(row_e)
        weighted_averages_cost.append(row_c)
        weighted_averages_quality.append(row_q)


    arr_e = np.asarray(weighted_averages_em).T
    arr_c = np.asarray(weighted_averages_cost).T
    arr_q = np.asarray(weighted_averages_quality).T
    print(arr_e.shape)

    # Plot the data using subfigures
    fig, axes = plt.subplots(nrows= 1, ncols = 3,figsize=(10,6)) 

    subfig1 = axes[0]
    subfig2 = axes[1]
    subfig3 = axes[2]

    for i, vec_e  in enumerate(arr_e):
        # Subfigure for emissions
        subfig1.scatter(data_social_network.history_time_social_network, arr_e[i], marker='o')
        subfig1.set_xlabel('Time')
        subfig1.set_ylabel('Emissions - Researched')
        subfig1.set_title('Emissions')
        subfig1.grid(True)

        # Subfigure for cost
        subfig2.scatter(data_social_network.history_time_social_network, arr_c[i], marker='*')
        subfig2.set_xlabel('Time')
        subfig2.set_ylabel('Average Cost - Researched')
        subfig2.set_title('Cost')
        subfig2.grid(True)

        # Subfigure for quality
        subfig3.scatter(data_social_network.history_time_social_network, arr_q[i], marker='x')
        subfig3.set_xlabel('Time')
        subfig3.set_ylabel(' Average Quality - Researched')
        subfig3.set_title('Quality')
        subfig3.grid(True)

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/reserach_tech"
    fig.savefig(f + ".png", dpi=600, format="png")


def scatter_trace_plots(fileName, data_social_network, x_param, y_param):

    time_series_cars = data_social_network.history_car_owned_vec

    # Initialize lists to hold values over time
    time_points = []
    x_values = []
    y_values = []

    if x_param == "cost":
        x_param_index = 0
        x_param_title = "Cost"
    elif x_param == "environmental_score":
        x_param_index = 1
        x_param_title = "Environmental score"
    elif x_param == "quality": 
        x_param_index = 2
        x_param_title = "Quality"

    if y_param == "cost":
        y_param_index = 0
        y_param_title = "Cost"
    elif y_param == "environmental_score":
        y_param_index = 1
        y_param_title = "Environmental score"
    elif y_param == "quality": 
        y_param_index = 2
        y_param_title = "Quality"

    for t, snapshot in enumerate(time_series_cars):
        for car in snapshot:
            attributes = car.attributes_fitness
            time_points.append(t)
            x_values.append(attributes[x_param_index])
            y_values.append(attributes[y_param_index])

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'time': time_points,
        'x': x_values,
        'y': y_values
    })
    #print(df['x'])
    #quit()

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(df['x'], df['y'], c=df['time'], cmap='viridis', alpha=1, edgecolors='b', linewidth=0.4)
    ax.set_xlabel(x_param_title)
    ax.set_ylabel(y_param_title)
    ax.set_title(f'{x_param_title} vs {y_param_title} Over Time')
    ax.grid(True)

    # Add colorbar to show the time evolution
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time')

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + f"/scatter_{x_param}_vs_{y_param}"
    fig.savefig(f + ".png", dpi=600, format="png")

    #plt.show()

def scatter_trace_plots_offered(fileName, data_firm_manager, x_param, y_param):
    time_series_cars = data_firm_manager.history_cars_on_sale_all_firms

    # Initialize lists to hold values over time
    time_points = []
    x_values = []
    y_values = []
    firm_ids = []

    param_map = {'cost': 0, 'environmental_score': 1, 'quality': 2}
    x_param_index = param_map[x_param]
    y_param_index = param_map[y_param]

    if x_param == "cost":
        x_param_index = 0
        x_param_title = "Cost"
    elif x_param == "environmental_score":
        x_param_index = 1
        x_param_title = "Environmental score"
    elif x_param == "quality": 
        x_param_index = 2
        x_param_title = "Quality"

    if y_param == "cost":
        y_param_index = 0
        y_param_title = "Cost"
    elif y_param == "environmental_score":
        y_param_index = 1
        y_param_title = "Environmental score"
    elif y_param == "quality": 
        y_param_index = 2
        y_param_title = "Quality"


    for t, snapshot in enumerate(time_series_cars):
        for car in snapshot:
            attributes = car.attributes_fitness
            time_points.append(t)
            x_values.append(attributes[x_param_index])
            y_values.append(attributes[y_param_index])
            firm_ids.append(car.firm_id)

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'time': time_points,
        'x': x_values,
        'y': y_values,
        'firm_id': firm_ids
    })

    # Assign a unique marker for each firm
    unique_firms = df['firm_id'].unique()
    markers = list(mlines.Line2D.markers.keys())
    marker_map = {firm_id: markers[i % len(markers)] for i, firm_id in enumerate(unique_firms)}

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    for firm_id in unique_firms:
        firm_data = df[df['firm_id'] == firm_id]
        ax.scatter(firm_data['x'], firm_data['y'], label=f'Firm {firm_id}', 
                   marker=marker_map[firm_id], alpha=0.6, edgecolors='w', linewidth=0.5)

    ax.set_xlabel(x_param_title)
    ax.set_ylabel(y_param_title)
    ax.set_title(f'{x_param_title} vs {y_param_title} Over Time - Firms')
    ax.grid(True)
    ax.legend(title='Firms', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + f"/scatter_{x_param}_vs_{y_param}_offered"
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_raw_util(fileName, data):

    fig, ax = plt.subplots(figsize=(10, 6))
    
    #burn_in(ax,data)
    data_t = np.asarray(data.histor_raw_utility_buy_0).T
    for data_indivdiual in data_t:
        ax.plot(data.history_time_social_network,data_indivdiual)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Raw Utility")
    ax.set_title("Utility of different offered cars from 0th individuals")

    plotName = fileName + "/Plots"

    f = plotName + "/raw_util"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_len_n(fileName, data_firm_manager, data_social_network):
    fig, ax = plt.subplots(figsize=(10, 6))
    burn_in(ax, data_social_network)
    #burn_in(ax,data)
    data_t = np.asarray(data_firm_manager.history_len_n).T
    for data_indivdiual in data_t:
        ax.plot(data_social_network.history_time_social_network,data_indivdiual)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Size of neighbouring technologies")

    plotName = fileName + "/Plots"

    f = plotName + "/len_n"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_len_alt(fileName, data_firm_manager, data_social_network):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    burn_in(ax,data_social_network)

    data_t = np.asarray(data_firm_manager.history_len_alt).T
    for data_indivdiual in data_t:
        ax.plot(data_social_network.history_time_social_network,data_indivdiual)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Size of alternative technologies")

    plotName = fileName + "/Plots"

    f = plotName + "/len_alt"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_emissions_stock(fileName, data_social_network):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_social_network.history_time_social_network, data_social_network.history_cumulative_carbon_emissions)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Cumulative carbon emmissions, E")

    plotName = fileName + "/Plots"

    f = plotName + "/cum_e"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_public_transport_count(fileName, data_social_network):

    fig, ax = plt.subplots(figsize=(10, 6))
    burn_in(ax,data_social_network)
    ax.plot(data_social_network.history_time_social_network, data_social_network.history_public_transport_prop)

    #ax.legend()          
    #ax.tight_layout()
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Public transport usage proportion")

    plotName = fileName + "/Plots"

    f = plotName + "/public_trans"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def weighted_owned_average_plots_no_public(fileName, data_social_network):

    time_series_cars = data_social_network.history_car_owned_vec

    # Initialize data structures to hold weighted averages over time
    weighted_averages = {
        'environmental_score': [],
        'cost': [],
        'quality': []
    }

    for snapshot in time_series_cars:
        # Filter out None values
        valid_cars = [car for car in snapshot if car is not None]
        
        if valid_cars:
            attributes_matrix = np.asarray([car.attributes_fitness for car in valid_cars])
            weighted_averages['environmental_score'].append(np.mean(attributes_matrix[:, 1]))
            weighted_averages['cost'].append(np.mean(attributes_matrix[:, 0]))
            weighted_averages['quality'].append(np.mean(attributes_matrix[:, 2]))
        else:
            # If no valid cars in snapshot, append NaN to maintain the time series length
            weighted_averages['environmental_score'].append(np.nan)
            weighted_averages['cost'].append(np.nan)
            weighted_averages['quality'].append(np.nan)

    # Convert to DataFrame for easier plotting
    df_weighted_averages = pd.DataFrame(weighted_averages)

    # Plot the data using subfigures
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

    subfig1 = axes[0]
    subfig2 = axes[1]
    subfig3 = axes[2]

    burn_in(subfig1,data_social_network)
    burn_in(subfig2,data_social_network)
    burn_in(subfig3,data_social_network)

    # Subfigure for emissions
    subfig1.scatter(data_social_network.history_time_social_network, df_weighted_averages['environmental_score'], marker='o')
    subfig1.set_xlabel('Time')
    subfig1.set_ylabel('Weighted Average Emissions - Owned - No public')
    subfig1.set_title('Emissions')
    subfig1.grid(True)

    # Subfigure for cost
    subfig2.scatter(data_social_network.history_time_social_network, df_weighted_averages['cost'], marker='*')
    subfig2.set_xlabel('Time')
    subfig2.set_ylabel('Weighted Average Cost - Owned - No public')
    subfig2.set_title('Cost')
    subfig2.grid(True)

    # Subfigure for quality
    subfig3.scatter(data_social_network.history_time_social_network, df_weighted_averages['quality'], marker='x')
    subfig3.set_xlabel('Time')
    subfig3.set_ylabel('Weighted Average Quality - Owned - No public')
    subfig3.set_title('Quality')
    subfig3.grid(True)

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/owned_weighted_vals_NO_PUBLIC"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_firm_count_and_market_concentration(fileName, data_social_network):
    # Initialize a dictionary to store the data
    data = {}

    

    # Extract data
    for time_point, snapshot in enumerate(data_social_network.history_firm_count):
        for firm, cars in snapshot.items():
            if firm not in data:
                data[firm] = []
            total_cars_sold = sum(cars.values())
            data[firm].append((time_point, total_cars_sold))

    # Convert to a DataFrame
    df = pd.DataFrame()
    for firm, sales in data.items():
        times, car_sales = zip(*sales)
        df[firm] = pd.Series(car_sales, index=times)

    # Fill missing values with 0 (if any firm didn't sell cars at some time points)
    df = df.fillna(np.nan)

    # Calculate market concentration (Herfindahl-Hirschman Index)
    market_concentration = []
    for time_point in df.index:
        total_cars = df.loc[time_point].sum()
        if total_cars > 0:
            market_shares = df.loc[time_point] / total_cars
            concentration = (market_shares ** 2).sum()
        else:
            concentration = np.nan
        market_concentration.append(concentration)

    # Plot the number of cars sold by each firm over time
    fig, ax1 = plt.subplots(figsize=(10, 6)) 

    burn_in(ax1,data_social_network)
    #for firm in df.columns:
    #    ax1.scatter(df.index, df[firm], marker='o', label=firm)

    #ax1.set_xlabel('Time')
    #ax1.set_ylabel('Number of Cars Sold')
    #ax1.set_title('Number of Cars Sold by Each Firm Over Time')
    #ax1.grid(True)
    #ax1.legend()

    # Plot the market concentration on a secondary y-axis
    #ax2 = ax1.twinx()
    ax1.plot(df.index, market_concentration, color='red', marker='x', linestyle='-', linewidth=2, label='Market Concentration')
    ax1.set_ylabel('Market Concentration (HHI)')
    #ax1.legend(loc='upper right')

    # Adjust layout and show plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/firm_count_and_market_concentration"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_research_clean_count(fileName, data_firm_manager,data_social_network):
    time_series = data_firm_manager.history_green_research_bools
    
    # Initialize lists to store counts
    green_counts = []
    dirty_counts = []
    time_steps = range(len(time_series))

    # Count the number of 1s and 0s at each time step
    for snapshot in time_series:
        green_count = sum(1 for x in snapshot if x == 1)
        dirty_count = sum(1 for x in snapshot if x == 0)
        
        if green_count == 0 and dirty_count == 0:
            green_counts.append(None)
            dirty_counts.append(None)
        else:
            green_counts.append(green_count)
            dirty_counts.append(dirty_count)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    burn_in(ax,data_social_network)
    ax.scatter(time_steps, green_counts, label='Green', marker='o', color='g')
    ax.scatter(time_steps, dirty_counts, label='Dirty', marker='x', color='r')

    # Labeling the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title('Count of Green and Dirty Over Time')
    ax.legend()
    ax.grid(True)

    # Show the plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/count_green"
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_carbon_price(fileName, data_controller):
    time_series = data_controller.history_carbon_price
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(time_series)), time_series)

    # Labeling the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Carbon price')

    ax.grid(True)

    # Show the plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/carbon_price"
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_em_flow(fileName, data_social_network):
    time_series = data_social_network.history_flow_carbon_emissions
    fig, ax = plt.subplots(figsize=(10, 6))
    burn_in(ax,data_social_network)
    ax.plot(range(len(time_series)), time_series)

    # Labeling the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Emission flow, $E_F$')

    ax.grid(True)

    # Show the plot
    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/em_flow"
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName = "results/single_experiment_15_05_51__26_02_2024",
    dpi_save = 600,
    social_plots = 1,
    firm_plots = 1
    ) -> None: 

    social_plots = 1
    firm_plots = 1

    base_params = load_object(fileName + "/Data", "base_params")
    data_controller= load_object(fileName + "/Data", "controller")
    data_social_network = data_controller.social_network
    data_firm_manager = data_controller.firm_manager

    plot_carbon_price(fileName,data_controller)

    if social_plots:
        ###SOCIAL NETWORK PLOTS
        plot_low_carbon_preference(fileName, data_social_network)
        plot_em_flow(fileName, data_social_network)
        #plot_emissions_stock(fileName, data_social_network)
        weighted_owned_average_plots_no_public(fileName, data_social_network)
        #weighted_owned_average_plots(fileName, data_social_network)
        #scatter_trace_plots(fileName, data_social_network, 'environmental_score', 'cost')
        #scatter_trace_plots(fileName, data_social_network, 'environmental_score', 'quality')
        #scatter_trace_plots(fileName, data_social_network, 'quality', 'cost')
        if base_params["parameters_social_network"]["init_public_transport_state"]:
            plot_public_transport_count(fileName, data_social_network)
        
    #print(len(list(mlines.Line2D.markers.keys())))
    #quit()

    if firm_plots:
        ##FIRM PLOTS
        #plot_firm_count(fileName, data_social_network)
        #plot_firm_count_and_market_concentration(fileName, data_social_network)
        plot_research_clean_count(fileName, data_firm_manager,data_social_network)
        #weighted_bought_average_plots(fileName, data_social_network, data_firm_manager)
        """THIS DOESNT QUITE WORK"""
        #scatter_trace_plots_offered(fileName, data_firm_manager, 'environmental_score', 'cost')
        #offered_average_plots(fileName,data_firm_manager, data_social_network)#FIX THIS PLOT
        #offered_properties(fileName,data_firm_manager, data_social_network)
        #offered_plots(fileName,data_firm_manager, data_social_network)
        #researched_tech(fileName,data_firm_manager, data_social_network)
        plot_len_n(fileName,data_firm_manager, data_social_network)
        plot_len_alt(fileName,data_firm_manager, data_social_network)


    #final_scatter_price_EI(fileName, data_firm_manager, dpi_save)
    #final_scatter_price_EI_alt(fileName, data_firm_manager, dpi_save)
    #ani_1 = ani_scatter_price_EI(fileName, data_firm_manager, dpi_save)

    plt.show()

if __name__ == "__main__":
    plots = main(
        fileName = "results/single_experiment_15_42_44__13_06_2024",
    )


