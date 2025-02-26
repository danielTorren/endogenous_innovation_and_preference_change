import json
import numpy as np
import matplotlib.pyplot as plt
from SALib.analyze import sobol
from package.resources.utility import (
    load_object,
    produce_name_datetime
)

def analyze_ev_uptake(problem, Y_ev_uptake, calc_second_order):
    """
    Perform sobol sensitivity analysis on EV uptake results.
    """
    Si_ev_uptake = sobol.analyze(
        problem,
        Y_ev_uptake,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )
    return Si_ev_uptake

def get_plot_data_ev(problem, Y_ev_uptake, calc_second_order):
    """
    Extract sobol indices for EV uptake.
    """
    Si_ev_uptake = analyze_ev_uptake(problem, Y_ev_uptake, calc_second_order)
    
    if calc_second_order:
        total_ev, first_ev, second_ev = Si_ev_uptake.to_df()
    else:
        total_ev, first_ev = Si_ev_uptake.to_df()
    
    total_data_ev, total_yerr_ev = get_data_bar_chart(total_ev)
    first_data_ev, first_yerr_ev = get_data_bar_chart(first_ev)
    
    data_sa_dict_total = {"ev_uptake": {"data": total_data_ev, "yerr": total_yerr_ev}}
    data_sa_dict_first = {"ev_uptake": {"data": first_data_ev, "yerr": first_yerr_ev}}
    
    if calc_second_order:
        return data_sa_dict_total, data_sa_dict_first, second_ev
    else:
        return data_sa_dict_total, data_sa_dict_first, calc_second_order

def get_data_bar_chart(Si_df):
    """
    Extract index values and their errors for plotting.
    """
    conf_cols = Si_df.columns.str.contains("_conf")
    confs = Si_df.loc[:, conf_cols]
    confs.columns = [c.replace("_conf", "") for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]
    return Sis, confs

def Merge_dict_SA(data_sa_dict: dict, plot_dict: dict) -> dict:
    """
    Merge the dictionaries used to create the data with the plotting dictionaries for easy of plotting later on so that its drawing from
    just one dictionary. This way I seperate the plotting elements from the data generation allowing easier re-plotting. I think this can be
    done with some form of join but I have not worked out how to so far
    Parameters
    ----------
    data_sa_dict: dict
        Dictionary of dictionaries of data associated with each output measure from the sensitivity analysis for a specific sobol index
    plot_dict: dict
        data structure that contains specifics about how a plot should look for each output measure from the sensitivity analysis

    Returns
    -------
    data_sa_dict: dict
        the joined dictionary of dictionaries
    """
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            data_sa_dict[i][v] = plot_dict[i][v]
    return data_sa_dict

def main(
    fileName="results/sensitivity_analysis_ev_uptake",
    plot_outputs=['ev_uptake'],
    plot_dict={
        "ev_uptake": {"title": "EV Uptake", "colour": "blue", "linestyle": "-"},
    }
):
    """
    Perform and visualize sensitivity analysis for EV uptake.
    """
    problem = load_object(fileName + "/Data", "problem")
    Y_ev_uptake = load_object(fileName + "/Data", "Y_ev")
    
    N_samples = load_object(fileName + "/Data", "N_samples")
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")
    
    data_sa_dict_total, data_sa_dict_first, _ = get_plot_data_ev(problem, Y_ev_uptake, calc_second_order)
    
    #data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    
    plot_sensitivity_results(fileName, data_sa_dict_total, plot_outputs,)

def plot_sensitivity_results(fileName, data_sa_dict_total, plot_outputs):
    """
    Generate and save sensitivity plots.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    data_dict = data_sa_dict_total[plot_outputs[0]]
    ax.errorbar(
        data_dict["data"]["ST"].tolist(),
        data_dict["data"].index,
        xerr=data_dict["yerr"]["ST"].tolist(),
        fmt="o",
        ecolor="k",
        color=data_dict["colour"]
    )
    ax.set_title("Total Order Sensitivity - EV Uptake")
    ax.set_xlim(left=0)
    plt.savefig(fileName + "/Plots/ev_uptake_sensitivity.png")
    plt.show()

if __name__ == '__main__':
    main(fileName="results/sensitivity_analysis_seeds_11_15_26__26_02_2025")
