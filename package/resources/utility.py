"""Contains functions that are not crucial to the simulation itself and are shared amongst files.
A module that aides in preparing folders, saving, loading and generating data for plots.

Created: 10/10/2022
"""

# imports
from copy import deepcopy
import pickle
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
from scipy import stats

# modules
def produce_name_datetime(root):
    fileName = "results/" + root +  "_" + datetime.datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    return fileName

def check_other_folder():
        # make prints folder:
    plotsName = "results/Other"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)
        
def get_cmap_colours(n, name="plasma"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def createFolder(fileName: str) -> str:
    """
    Check if folders exist and if they dont create results folder in which place Data, Plots, Animations
    and Prints folders

    Parameters
    ----------
    fileName:
        name of file where results may be found

    Returns
    -------
    None
    """

    # print(fileName)
    # check for resutls folder
    if str(os.path.exists("results")) == "False":
        os.mkdir("results")

    # check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    # make data folder:#
    dataName = fileName + "/Data"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)

    # make Calibration_data folder:#
    dataName = fileName + "/Calibration_runs"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)
        
    # make plots folder:
    plotsName = fileName + "/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make animation folder:
    plotsName = fileName + "/Animations"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make prints folder:
    plotsName = fileName + "/Prints"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

def save_object(data, fileName, objectName):
    """save single object as a pickle object

    Parameters
    ----------
    data: object,
        object to be saved
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    objectName: str
        what name to give the saved object

    Returns
    -------
    None
    """
    with open(fileName + "/" + objectName + ".pkl", "wb") as f:
        pickle.dump(data, f)

def is_2d_list(lst):
    # Check if the list is not empty
    if not lst:
        return False

    # Check if all elements are of type list
    return all(isinstance(sublist, list) for sublist in lst)

def save_data_csv_firms(firm_manager_firms_list,save_data_csv_list_firm_manager, fileName):

    for title in save_data_csv_list_firm_manager:
        for firm_id in [1,2,3]: 
            list_history = getattr(firm_manager_firms_list[firm_id], title)
            csv_file_path = fileName + "/" + title + "_firm_id_" + str(firm_id) + ".csv"
            #print("csv_file_path", csv_file_path)
            if is_2d_list(list_history):
                # Use "newline="" to ensure that newline characters are handled properly across different platforms
                
                with open(csv_file_path, "w", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)

                    # Iterate through each list in your data and write it to the CSV file
                    for row in list_history:
                        csv_writer.writerow(row)
            else:
                with open(csv_file_path, "w", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(list_history)
                """
                list_history = getattr(firm_manager_firms_list[firm_id], title)
                #print("LIST",title +"_firm_id_" + str(firm_id),list_history)
                array_history = np.asarray(list_history)
                #print("ARRAY",title +"_firm_id_" + str(firm_id),array_history)
                file_label = fileName + "/" + title +"_firm_id_" + str(firm_id) + ".csv"
                print("file_label",file_label)
                np.savetxt(file_label, array_history, delimiter=",")
                """
            
def save_data_csv_2D(data,fileName,title):

    csv_file_path = fileName + "/" + title + ".csv"
    with open(csv_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Iterate through each list in your data and write it to the CSV file
        for row in data:
            csv_writer.writerow(row)

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"]== "geo":
        property_list = np.geomspace(variable_parameters_dict["property_min"],variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_list 


def load_object(fileName, objectName) -> dict:
    """load single pickle file

    Parameters
    ----------
    fileName: str
        where to load it from e.g in the results folder in data folder
    objectName: str
        what name of the object to load is

    Returns
    -------
    data: object
        the pickle file loaded
    """
    with open(fileName + "/" + objectName + ".pkl", "rb") as f:
        data = pickle.load(f)
    return data

def calc_bounds(data, confidence_level):
    # Calculate mean and standard deviation across rows
    ys_mean = data.mean(axis=1)
    ys_std = data.std(axis=1)

    # Calculate the standard error of the mean (SEM)
    n = data.shape[1]  # Number of samples
    ys_sem = ys_std / np.sqrt(n)

    # Calculate margin of error
    z_score = np.abs(stats.norm.ppf((1 - confidence_level) / 2))  # For a two-tailed test
    margin_of_error = z_score * ys_sem

    # Calculate confidence intervals
    lower_bound = ys_mean - margin_of_error
    upper_bound = ys_mean + margin_of_error

    return ys_mean,lower_bound, upper_bound

def params_list_with_seed(base_params):
    """
    Expand the list of scenarios by varying the seed parameters.
    """
    base_params_list = []
    seed_repetitions = base_params["seed_repetitions"]

    for seed in range(1, seed_repetitions + 1):
        base_params_copy = deepcopy(base_params)
        base_params_copy["seed"] = seed
        base_params_copy["seed_inputs"] = seed + 66
        base_params_list.append( base_params_copy)

    return base_params_list