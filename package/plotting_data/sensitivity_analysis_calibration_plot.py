# imports
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from package.resources.utility import (
    load_object
)
import numpy as np

def get_plot_data(
    problem: dict,
    Y_emissions_stock: npt.NDArray,
    calc_second_order: bool,
) -> tuple[dict, dict]:

    Si_emissions_stock = analyze_results(problem,Y_emissions_stock,calc_second_order) 

    if calc_second_order:
        total_emissions_stock, first_emissions_stock, second_emissions_stock = Si_emissions_stock.to_df()
    else:
        total_emissions_stock, first_emissions_stock = Si_emissions_stock.to_df()


    total_data_sa_emissions_stock, total_yerr_emissions_stock = get_data_bar_chart(total_emissions_stock)
    first_data_sa_emissions_stock, first_yerr_emissions_stock= get_data_bar_chart(first_emissions_stock)

    data_sa_dict_total = {
        "emissions_stock": {
            "data": total_data_sa_emissions_stock,
            "yerr": total_yerr_emissions_stock,
        },
    }
    data_sa_dict_first = {
        "emissions_stock": {
            "data": first_data_sa_emissions_stock,
            "yerr": first_yerr_emissions_stock,
        },
    }

    if calc_second_order:
        return data_sa_dict_total, data_sa_dict_first, second_emissions_stock
    else:
        return data_sa_dict_total, data_sa_dict_first, calc_second_order#return nothing for second order
    
    

def get_data_bar_chart(Si_df):
    """
    Taken from: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html
    Reduce the sobol index dataframe down to just the bits I want for easy plotting of sobol index and its error

    Parameters
    ----------
    Si_df: pd.DataFrame,
        Dataframe of sensitivity results.
    Returns
    -------
    Sis: pd.Series
        the value of the index
    confs: pd.Series
        the associated error with index
    """

    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains("_conf")
    confs = Si_df.loc[:, conf_cols]  # select all those that ARE in conf_cols!
    confs.columns = [c.replace("_conf", "") for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]  # select all those that ARENT in conf_cols!

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

def analyze_results(
    problem: dict,
    Y_emissions_stock: npt.NDArray,
    calc_second_order: bool,
) -> tuple:
    """
    Perform sobol analysis on simulation results
    """
    #print(problem)

    #print(Y_emissions_stock.shape, )
    #print(calc_second_order)
    #N_samples = 64

    #expected_size = (2 * len(problem["names"]) + 2) * N_samples if calc_second_order else (len(problem["names"]) + 2) * N_samples
    #print(f"Expected Y size: {expected_size}, Actual: {len(Y_emissions_stock)}")


    Si_emissions_stock = sobol.analyze(
        problem,
        Y_emissions_stock,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )
    
    return Si_emissions_stock

def multi_output_sensitivity_analysis_plot(
    fileName, data_dict, output_keys, param_names, N_samples, order, latex_bool=False
):
    """
    Creates a horizontally stacked sensitivity analysis plot with each column as a different output variable.
    Y-axis (parameter names) is shown for each subplot.
    """
    num_outputs = len(output_keys)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_outputs,
        figsize=(3.8 * num_outputs, 6),
        sharey=True,
        constrained_layout=True
    )

    if num_outputs == 1:
        axes = [axes]  # ensure iterable

    for i, output in enumerate(output_keys):
        ax = axes[i]
        data = data_dict[output]["data"]
        yerr = data_dict[output]["yerr"]
        color = data_dict[output]["colour"]
        title = data_dict[output]["title"]

        if order == "First":
            x = data["S1"].tolist()
            xerr = yerr["S1"].tolist()
        else:
            x = data["ST"].tolist()
            xerr = yerr["ST"].tolist()

        ax.errorbar(
            x,
            param_names,
            xerr=xerr,
            fmt="o",
            ecolor="k",
            color=color,
        )

        ax.set_title(title, fontsize=10)
        ax.set_xlim(left=0)
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names, fontsize=8)  # ✅ y-labels on every subplot
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.supylabel("Parameter", fontsize=12)
    fig.supxlabel(r"%s order Sobol index" % (order), fontsize=12)

    plotName = fileName + "/Prints"
    f = f"{plotName}/{len(param_names)}_{N_samples}_{order}_multi_output_sensitivity_plot_horizontal.eps"
    f_png = f.replace(".eps", ".png")
    fig.savefig(f, dpi=300, format="eps")
    fig.savefig(f_png, dpi=300, format="png")


def replace_nan_with_interpolation(arr):
    # Get the indices where the values are not NaN
    not_nan_indices = np.where(~np.isnan(arr))[0]
    
    # Get the indices where the values are NaN
    nan_indices = np.where(np.isnan(arr))[0]
    
    # Interpolate the NaN indices based on the not NaN values
    arr[nan_indices] = np.interp(nan_indices, not_nan_indices, arr[not_nan_indices])
    
    return arr

import seaborn as sns
import pandas as pd

def plot_second_order_heatmap(
    fileName: str,
    second_order_df: pd.DataFrame,
    param_names: list[str],
    output_name: str,
    cmap: str = "coolwarm",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Plots a heatmap of second-order Sobol indices.
    
    Parameters
    ----------
    fileName : str
        Root directory for saving the plot.
    second_order_df : pd.DataFrame
        DataFrame containing second-order Sobol indices as returned by SALib.
    param_names : list
        List of parameter names.
    output_name : str
        Name of the model output this analysis corresponds to.
    cmap : str
        Color map used in heatmap.
    vmin : float
        Minimum value for color scale.
    vmax : float
        Maximum value for color scale.
    """

    # Clean the DataFrame to isolate just the values
    S2_matrix = pd.DataFrame(
        data=0.0,
        index=param_names,
        columns=param_names
    )

    # Populate with second-order values (it's a flat format: 'S2_ij')
    for col in second_order_df.columns:
        if col.startswith("S2_"):
            i, j = col.replace("S2_", "").split(",")
            i, j = int(i), int(j)
            val = second_order_df[col].values[0]
            S2_matrix.iloc[i, j] = val
            S2_matrix.iloc[j, i] = val  # symmetrical

    plt.figure(figsize=(10, 8))
    sns.heatmap(S2_matrix, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", square=True,
                cbar_kws={'label': 'Second-order Sobol index'})
    plt.title(f"Second-order Sobol Indices: {output_name}", fontsize=14)
    plt.tight_layout()

    save_path_eps = f"{fileName}/Prints/second_order_{output_name}.eps"
    save_path_png = save_path_eps.replace(".eps", ".png")

    plt.savefig(save_path_eps, format="eps", dpi=300)
    plt.savefig(save_path_png, format="png", dpi=300)


def main(
    fileName="results/sensitivity_analysis_14_42_24__07_04_2025",
    plot_outputs=['emissions_stock', 'ev_uptake', 'total_firm_profit', 'market_concentration', "utility", "car_age"],
    plot_dict={
        "emissions_stock": {"title": r"Cumulative Emissions, $E$", "colour": "red", "linestyle": "--"},
        "ev_uptake": {"title": r"EV Adoption Proportion", "colour": "blue", "linestyle": "."},
        "total_firm_profit": {"title": r"Firm Profit, $\$$", "colour": "orange", "linestyle": "-."},
        "market_concentration": {"title": r"Market Concentration HHI", "colour": "green", "linestyle": "--"},
        "utility": {"title": r"Utility, $\$$", "colour": "yellow", "linestyle": "--"},
        "car_age": {"title": r"Mean Car Age", "colour": "purple", "linestyle": "."}
    },
    titles=["K_ICE", "K_EV", "delta", "lambda", "a_chi", "b_chi", "kappa", "mu", "r", "alpha"],
    latex_bool=False
):
    # Load shared inputs
    problem = load_object(fileName + "/Data", "problem")
    problem["names"] = ["K_ICE", "K_EV", "delta", "lambda", "a_chi", "b_chi", "kappa", "mu", "r", "alpha"]

    N_samples = load_object(fileName + "/Data", "N_samples")
    calc_second_order = bool(load_object(fileName + "/Data", "calc_second_order"))

    # Mapping from file keys to internal variable names
    output_files = {
        "emissions_stock": "Y_emissions_stock",
        "ev_uptake": "Y_ev_uptake",
        "total_firm_profit": "Y_total_firm_profit",
        "market_concentration": "Y_market_concentration",
        "utility": "Y_total_utility",
        "car_age": "Y_mean_car_age"
    }

    data_sa_dict_total_all = {}
    data_sa_dict_first_all = {}

    # Loop through all desired outputs
    for key in plot_outputs:
        #print(f"\nProcessing output: {key}")
        y_data = load_object(fileName + "/Data", output_files[key])
        y_data = replace_nan_with_interpolation(y_data)

        # Analyze
        Si = analyze_results(problem, y_data, calc_second_order)

        if calc_second_order:
            total_df, first_df, second_df = Si.to_df()
            plot_second_order_heatmap(
                fileName=fileName,
                second_order_df=second_df,
                param_names=problem["names"],
                output_name=key
            )

        if calc_second_order:
            total_df, first_df, _ = Si.to_df()
        else:
            total_df, first_df = Si.to_df()

        total_data, total_yerr = get_data_bar_chart(total_df)
        first_data, first_yerr = get_data_bar_chart(first_df)

        data_sa_dict_total_all[key] = {"data": total_data, "yerr": total_yerr}
        data_sa_dict_first_all[key] = {"data": first_data, "yerr": first_yerr}

    # Merge plot info
    data_sa_dict_total_all = Merge_dict_SA(data_sa_dict_total_all, plot_dict)
    data_sa_dict_first_all = Merge_dict_SA(data_sa_dict_first_all, plot_dict)

    # Plot
    multi_output_sensitivity_analysis_plot(
        fileName, data_sa_dict_first_all, plot_outputs, titles, N_samples, "First", latex_bool
    )
    multi_output_sensitivity_analysis_plot(
        fileName, data_sa_dict_total_all, plot_outputs, titles, N_samples, "Total", latex_bool
    )

    plt.show()

if __name__ == '__main__':

    plots = main(
        fileName="results/sensitivity_analysis_14_42_24__07_04_2025",
        plot_outputs = ['emissions_stock', 'ev_uptake', 'total_firm_profit', 'market_concentration', "utility", "car_age"],#,'emissions_flow','var',"emissions_change"
        plot_dict = {
            "emissions_stock": {"title": r"Cumulative Emissions, $E$", "colour": "red", "linestyle": "--"},
            "ev_uptake": {"title": r"EV Adoption Proportion", "colour": "blue", "linestyle": "."},
            "total_firm_profit": {"title": r"Firm Profit, $\$$", "colour": "orange", "linestyle": "-."},
            "market_concentration": {"title": r"Market Concentration HHI", "colour": "green", "linestyle": "--"},
            "utility": {"title": r"Utility, $\$$", "colour": "yellow", "linestyle": "--"},
            "car_age": {"title": r"Mean Car Age", "colour": "purple", "linestyle": "."}
        },
        titles =[
            "K_ICE",
            "K_EV",
            "delta",
            "lambda",
            "a_chi",
            "b_chi",
            "kappa",
            "mu",
            "r",
            "alpha"
        ]
    )

