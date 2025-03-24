import matplotlib.pyplot as plt
from package.resources.utility import load_object,save_object
import numpy as np
from package.plotting_data.single_experiment_plot import add_vertical_lines, save_and_show, plot_emissions
def calculate_time_series_derivatives(outputs):
    """
    Calculate the derivatives of CO2 emissions for each time step.

    Args:
        outputs (dict): Time series data for all variables.

    Returns:
        dict: Time series of derivatives for each variable.
    """
    # Unpack outputs
    history_ev_prop = outputs["history_ev_prop"]
    history_mean_distance_EV = outputs["history_mean_distance_EV"]
    history_mean_distance_ICE = outputs["history_mean_distance_ICE"]
    history_e_EV = outputs["history_e_EV"]
    history_e_ICE = outputs["history_e_ICE"]
    history_new_ICE_cars_bought = outputs["history_new_ICE_cars_bought"]
    history_new_EV_cars_bought = outputs["history_new_EV_cars_bought"]
    history_num_individuals = outputs["history_num_individuals"]
    history_production_emissions_ICE = outputs["history_production_emissions_ICE_parameter"]
    history_production_emissions_EV = outputs["history_production_emissions_EV_parameter"]

    # Initialize derivatives dictionary
    derivatives = {
        "dCO2_dpi_ICE_t": [],
        "dCO2_dpi_EV_t": [],
        "dCO2_dSh_EV_t": [],
        "dCO2_dD_ICE_t": [],
        "dCO2_dD_EV_t": [],
        "dCO2_domega_ICE_t": [],
        "dCO2_domega_EV_t": [],
        "dCO2_de_EV_t": []
    }

    # Iterate over time steps
    for t in range(len(history_ev_prop)):
        Sh_EV_t = history_ev_prop[t]
        D_ICE_t = history_mean_distance_ICE[t]
        D_EV_t = history_mean_distance_EV[t]
        e_EV_t = history_e_EV[t]
        e_ICE = history_e_ICE[t]
        pi_ICE_t = history_new_ICE_cars_bought[t]
        pi_EV_t = history_new_EV_cars_bought[t]
        I = history_num_individuals[t]
        E_ICE = history_production_emissions_ICE[t]
        E_EV = history_production_emissions_EV[t]

        omega_ICE_t = D_ICE_t / e_ICE if e_ICE > 0 else np.nan
        omega_EV_t = D_EV_t / e_EV_t if e_EV_t > 0 else np.nan

        # Compute derivatives
        derivatives["dCO2_dpi_ICE_t"].append(E_ICE)
        derivatives["dCO2_dpi_EV_t"].append(E_EV)
        derivatives["dCO2_dSh_EV_t"].append(
            I * ((D_EV_t * e_EV_t) / omega_EV_t - (D_ICE_t * e_ICE) / omega_ICE_t)
        )
        derivatives["dCO2_dD_ICE_t"].append(
            I * (1 - Sh_EV_t) * (e_ICE / omega_ICE_t)
        )
        derivatives["dCO2_dD_EV_t"].append(
            I * Sh_EV_t * (e_EV_t / omega_EV_t)
        )
        derivatives["dCO2_domega_ICE_t"].append(
            -I * (1 - Sh_EV_t) * (D_ICE_t * e_ICE) / (omega_ICE_t ** 2) if omega_ICE_t > 0 else np.nan
        )
        derivatives["dCO2_domega_EV_t"].append(
            -I * Sh_EV_t * (D_EV_t * e_EV_t) / (omega_EV_t ** 2) if omega_EV_t > 0 else np.nan
        )
        derivatives["dCO2_de_EV_t"].append(
            I * Sh_EV_t * (D_EV_t / omega_EV_t) if omega_EV_t > 0 else np.nan
        )

    return derivatives


def plot_time_series_derivatives(fileName, base_params, derivatives, time_steps,  whole_simulation):
    """
    Plot the derivatives over time using a single set of axes.

    Args:
        fileName (str): File name for saving the figure.
        base_params (dict): Base parameters containing relevant metadata.
        derivatives (dict): Time series of derivatives for each variable.
        time_steps (list or np.array): Time steps corresponding to the derivatives.
    """
    # Create figure and a single axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each derivative on the same axis
    for key, values in derivatives.items():
        ax.plot(time_steps, values, label=key)

    # Add labels and title
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Derivative Value")
    ax.set_title("Time Series of Derivatives of CO2 Emissions")

    # Add legend
    ax.legend(loc="upper right")

    # Add vertical lines if required by `base_params`
    if  whole_simulation:
        add_vertical_lines(ax,base_params)

    # Enable grid
    ax.grid()

    # Adjust layout and display the plot
    plt.tight_layout()



    save_and_show(fig, fileName, "plot_time_series_derivatives", dpi=300)

def compute_relative_impacts(derivatives):
    """
    Compute the relative impact of each derivative over the entire period.
    
    Args:
        derivatives (dict): Dictionary of derivative lists over time.
            e.g. {
               "dCO2_dpi_ICE_t": [val_t1, val_t2, ...],
               "dCO2_dpi_EV_t": [...],
               ...
            }
            
    Returns:
        dict: Dictionary of the same keys with fractional contributions (0-1).
              To get a percentage, multiply by 100.
    """
    # 1. Convert each derivative list to numpy array for easier math
    derivative_arrays = {k: np.array(v) for k, v in derivatives.items()}
    
    # 2. Compute total absolute sum for each derivative
    totals = {}
    for key, arr in derivative_arrays.items():
        # Use np.nansum in case some derivatives contain NaN values
        totals[key] = np.nansum(np.abs(arr))
        
    # 3. Compute grand total (sum of all totals)
    grand_total = sum(totals.values())
    
    # 4. Compute fraction for each derivative
    #    If grand_total is zero (edge case), set all impacts to 0
    if grand_total == 0:
        relative_impacts = {key: 0.0 for key in totals}
    else:
        relative_impacts = {key: totals[key] / grand_total for key in totals}
    
    return relative_impacts

def plot_normalized_time_series(outputs, fileName, base_params, whole_simulation) -> None:
    """
    Plot each time series variable in outputs on the same figure,
    normalized to its own maximum (or other chosen factor).
    
    Args:
        outputs (dict): Dictionary of time series arrays to be plotted.
        fileName (str): Filename prefix for saving figures.
        base_params (dict): Base parameters (for vertical lines or metadata).
        whole_simulation (bool): Whether to plot vertical lines from base_params.
    """
    # Create figure and a single axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate x-axis range
    first_key = next(iter(outputs))
    time_steps = np.arange(len(outputs[first_key]))
    
    for key, values in outputs.items():
        # Convert to NumPy array for easier calculations
        arr = np.array(values, dtype=float)
        
        # Handle edge case if all zero or NaN
        # E.g. if max is zero or array is entirely NaN
        max_val = np.nanmax(arr)
        if np.isnan(max_val) or max_val == 0:
            # Skip plot or plot zeros
            print(f"Warning: Skipping '{key}' due to zero/NaN values.")
            continue
        
        # Normalize by max_val
        norm_arr = arr / max_val
        
        # Plot the normalized array
        if len(time_steps) != len(norm_arr):
            ax.plot(time_steps, norm_arr[:-1], label=f"{key} (normalized)")
        else:
            ax.plot(time_steps, norm_arr, label=f"{key} (normalized)")
        
    # Add labels, title, legend
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Normalized Time Series Comparison")
    ax.legend(loc="upper right")
    
    # Optionally add vertical lines
    if whole_simulation:
        add_vertical_lines(ax, base_params)
    
    # Enable grid
    ax.grid()
    
    # Layout and save/show
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_normalized_time_series", dpi=300)

def unify_time_series_lengths(outputs):
    """
    Ensures all arrays in 'outputs' have the same length by slicing each 
    to the minimum length across all arrays.

    Args:
        outputs (dict): Dictionary of time-series arrays (lists or NumPy arrays).
                       e.g. {
                           "history_mean_distance_ICE": [...],
                           "history_e_ICE": [...],
                           ...
                       }

    Returns:
        dict: A new dictionary with each time series truncated to the same min length.
    """
    # Convert everything to NumPy arrays first
    arrays = {k: np.array(v, dtype=float) for k, v in outputs.items()}

    # Find the smallest length among all arrays
    min_len = min(len(arr) for arr in arrays.values())

    # Slice each array to min_len
    unified = {k: arr[:min_len] for k, arr in arrays.items()}

    return unified


def calculate_time_series_elasticities(outputs, derivatives):
    """
    Calculate elasticities over time: E_X(t) = (dCO2 / dX)(t) * [X(t) / CO2(t)]
    """
    # 1. Force same length for all time-series arrays in outputs
    outputs = unify_time_series_lengths(outputs)

    # 2. Convert the relevant arrays
    CO2_arr = np.array(outputs["history_total_emissions"], dtype=float)

    # Mapping from derivative keys -> variable array in 'outputs'
    variable_map = {
        "dCO2_dpi_ICE_t": "history_new_ICE_cars_bought",
        "dCO2_dpi_EV_t":  "history_new_EV_cars_bought",
        "dCO2_dSh_EV_t":  "history_ev_prop",
        "dCO2_dD_ICE_t":  "history_mean_distance_ICE",
        "dCO2_dD_EV_t":   "history_mean_distance_EV",
        "dCO2_domega_ICE_t": None,  # We handle these separately
        "dCO2_domega_EV_t":  None,
        "dCO2_de_EV_t":   "history_e_EV"
    }

    # Example: build or slice arrays for distances / intensities
    D_ICE_arr = outputs["history_mean_distance_ICE"]
    D_EV_arr  = outputs["history_mean_distance_EV"]
    e_ICE_arr = outputs["history_e_ICE"]
    e_EV_arr  = outputs["history_e_EV"]

    # Safely compute omega arrays
    with np.errstate(invalid="ignore", divide="ignore"):
        omega_ICE_arr = np.where(e_ICE_arr > 0, D_ICE_arr / e_ICE_arr, np.nan)
        omega_EV_arr  = np.where(e_EV_arr  > 0, D_EV_arr  / e_EV_arr,  np.nan)

    # Convert derivatives to arrays (and optionally unify them too)
    derivatives_np = {}
    for k, v in derivatives.items():
        arr_v = np.array(v, dtype=float)
        # Optionally unify derivative array length with CO2_arr
        min_len_deriv = min(len(arr_v), len(CO2_arr))
        arr_v = arr_v[:min_len_deriv]
        # Also ensure CO2_arr is truncated if needed
        CO2_arr = CO2_arr[:min_len_deriv]

        derivatives_np[k] = arr_v

    # Build elasticity results
    elasticities = {}
    for deriv_key, dCO2_dX_arr in derivatives_np.items():
        elasticity_key = deriv_key.replace("dCO2_d", "elasticity_")

        # Identify the variable array that corresponds
        if deriv_key in ("dCO2_domega_ICE_t", "dCO2_domega_EV_t"):
            X_arr = omega_ICE_arr if deriv_key == "dCO2_domega_ICE_t" else omega_EV_arr
            # Make sure X_arr also gets truncated if needed
            min_len_elastic = min(len(X_arr), len(dCO2_dX_arr))
            X_arr = X_arr[:min_len_elastic]
            dCO2_dX_arr = dCO2_dX_arr[:min_len_elastic]
            local_CO2_arr = CO2_arr[:min_len_elastic]
        else:
            var_name = variable_map[deriv_key]
            X_arr = outputs[var_name]
            # Truncate if needed
            min_len_elastic = min(len(X_arr), len(dCO2_dX_arr), len(CO2_arr))
            X_arr = X_arr[:min_len_elastic]
            dCO2_dX_arr = dCO2_dX_arr[:min_len_elastic]
            local_CO2_arr = CO2_arr[:min_len_elastic]

        # E_X(t) = dCO2/dX(t) * [X(t) / CO2(t)]
        elasticity_t = np.where(
            local_CO2_arr != 0,
            dCO2_dX_arr * (X_arr / local_CO2_arr),
            np.nan
        )

        elasticities[elasticity_key] = elasticity_t

    return elasticities

def plot_time_series_elasticities(
    fileName, base_params, elasticities, time_steps, whole_simulation
):
    """
    Plot the time series of elasticities for each variable on a single axis.

    Args:
        fileName (str): Prefix/path for saving figure.
        base_params (dict): Contains relevant metadata (e.g., for vertical lines).
        elasticities (dict): Dictionary of elasticity arrays, e.g.:
            {
                "elasticity_pi_ICE_t": [...],
                "elasticity_pi_EV_t": [...],
                ...
            }
        time_steps (np.array): Array of time indices.
        whole_simulation (bool): If True, add vertical lines, etc.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each elasticity curve
    for key, arr in elasticities.items():
        ax.plot(time_steps, arr, label=key)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Elasticity (dCO2/dX * X/CO2)")
    ax.set_title("Time Series of CO2 Elasticities")
    ax.legend(loc="upper right")

    if whole_simulation:
        add_vertical_lines(ax, base_params)

    ax.grid()
    plt.tight_layout()

    save_and_show(fig, fileName, "plot_time_series_elasticities", dpi=300)

def plot_stacked_contributions_with_net(
    derivatives,
    time_steps,
    fileName,
    base_params,
    whole_simulation=True,
    title="Stacked Contributions to CO2 Changes"
):
    """
    Produce a stacked bar plot of contributions from each derivative over time.
    Positive values are stacked above zero; negative values are stacked below zero.
    Also plots the net value (sum of all derivatives) as a line on top.

    Args:
        derivatives (dict):
            Keys are derivative names (e.g. "dCO2_dpi_ICE_t")
            Values are arrays/lists of length T with contributions.
        time_steps (array-like):
            The x-values (time indices). Should match the length of each derivative array.
        fileName (str):
            Used for saving the figure.
        base_params (dict):
            Used for vertical lines or metadata.
        whole_simulation (bool):
            If True, add vertical lines from base_params.
        title (str):
            Title of the figure.
    """
    # Convert each derivative to a NumPy array
    derivative_arrays = {
        key: np.array(val, dtype=float)
        for key, val in derivatives.items()
    }

    # Ensure consistent length across all derivative arrays
    # using the length of 'time_steps' as reference
    T = len(time_steps)
    for k, arr in derivative_arrays.items():
        if len(arr) != T:
            derivative_arrays[k] = arr[:T]  # truncate if needed

    # Separate positive and negative parts for each derivative
    positives = []
    negatives = []
    derivative_keys = list(derivative_arrays.keys())

    # Prepare an array for net contributions at each time step
    net_contribution = np.zeros(T)

    for key in derivative_keys:
        arr = derivative_arrays[key]
        pos_part = np.where(arr > 0, arr, 0.0)   # keep positives, zero out negatives
        neg_part = np.where(arr < 0, arr, 0.0)   # keep negatives, zero out positives
        positives.append(pos_part)
        negatives.append(neg_part)

        # Accumulate the full array into net
        net_contribution += arr

    # Create stacked arrays: shape (num_derivatives, T)
    positives = np.vstack(positives)  # shape (D, T)
    negatives = np.vstack(negatives)  # shape (D, T)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot stacked bars for positives
    bottom = np.zeros(T)  # starts at 0
    for i, key in enumerate(derivative_keys):
        ax.bar(
            time_steps,
            positives[i],
            bottom=bottom,
            label=f"{key} (+)" if i == 0 else None,  # label only once to avoid legend clutter
            width=0.8,
            color=f"C{i}",  # so each derivative has a unique color
            edgecolor="white"
        )
        bottom += positives[i]

    # Plot stacked bars for negatives
    bottom = np.zeros(T)  # starts at 0 again
    for i, key in enumerate(derivative_keys):
        ax.bar(
            time_steps,
            negatives[i],
            bottom=bottom,
            label=f"{key} (-)" if i == 0 else None,  # label only once
            width=0.8,
            color=f"C{i}",  # use the same color as the positive counterpart
            edgecolor="white"
        )
        bottom += negatives[i]

    # Plot the net contribution as a line on top
    ax.plot(time_steps, net_contribution, "k--", linewidth=2, label="Net Contribution")

    # Customize plot
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Contribution to CO2 (partial derivative value)")
    ax.set_title(title)



    ax.grid(True)

    # We handle legend so that we don't repeat derivative labels for (+) and (-) multiple times
    legend_elements = [
        plt.Line2D([0], [0], color="k", linestyle="--", label="Net Contribution")
    ]
    legend_labels = set()  # store label strings to avoid repetition
    for i, key in enumerate(derivative_keys):
        label_plus = f"{key} (+)"
        label_minus = f"{key} (-)"
        if label_plus not in legend_labels:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=f"C{i}", edgecolor="white", label=key)
            )
            legend_labels.add(label_plus)
            legend_labels.add(label_minus)
    ax.legend(handles=legend_elements, loc="best")
    
    if whole_simulation:
        add_vertical_lines(ax, base_params)

    plt.tight_layout()
    save_and_show(fig, fileName, "plot_stacked_contributions_with_net", dpi=300)


def main(fileName):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        controller = load_object(fileName + "/Data", "controller")
    except FileNotFoundError:
        print("Data files not found.")
        return
    
    time_series = controller.time_series

    social_network = controller.social_network
    policy_duration = base_params["duration_future"]
    
    whole_simulation = False

    if whole_simulation:
        time_steps_max  = len(controller.electricity_emissions_intensity_vec)#base_params["duration_burn_in"] + base_params["duration_calibration"] + base_params["duration_future"]
        outputs = {
            "history_ev_prop": np.array(social_network.history_EV_users) / social_network.num_individuals,
            "history_mean_distance_EV": np.nanmean(np.asarray(social_network.history_distance_individual_EV).T, axis=0),
            "history_mean_distance_ICE": np.nanmean(np.asarray(social_network.history_distance_individual_ICE).T, axis=0),
            "history_e_EV": controller.electricity_emissions_intensity_vec,
            "history_e_ICE": np.asarray([controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]] * time_steps_max),
            "history_new_ICE_cars_bought": np.asarray(social_network.history_new_ICE_cars_bought),
            "history_new_EV_cars_bought": np.asarray(social_network.history_new_EV_cars_bought),
            "history_num_individuals": np.asarray([social_network.num_individuals] * time_steps_max),
            "history_production_emissions_ICE_parameter": np.asarray([controller.parameters_ICE["production_emissions"]] * time_steps_max),
            "history_production_emissions_EV_parameter": np.asarray([controller.parameters_EV["production_emissions"]] * time_steps_max),
            "history_driving_emissions_ICE": np.array(social_network.history_driving_emissions_ICE),
            "history_driving_emissions_EV": np.array(social_network.history_driving_emissions_EV),
            "history_total_emissions": np.asarray(social_network.history_total_emissions)
        }
    else:
        outputs = {
            "history_ev_prop": np.array(social_network.history_EV_users[-policy_duration:]) / social_network.num_individuals,
            "history_mean_distance_EV": np.nanmean(np.asarray(social_network.history_distance_individual_EV[-policy_duration:]).T, axis=0),
            "history_mean_distance_ICE": np.nanmean(np.asarray(social_network.history_distance_individual_ICE[-policy_duration:]).T, axis=0),
            "history_e_EV": controller.electricity_emissions_intensity_vec[-policy_duration:],
            "history_e_ICE": np.asarray([controller.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]] * policy_duration),
            "history_new_ICE_cars_bought": np.asarray(social_network.history_new_ICE_cars_bought[-policy_duration:]),
            "history_new_EV_cars_bought": np.asarray(social_network.history_new_EV_cars_bought[-policy_duration:]),
            "history_num_individuals": np.asarray([social_network.num_individuals] * policy_duration),
            "history_production_emissions_ICE_parameter": np.asarray([controller.parameters_ICE["production_emissions"]] * policy_duration),
            "history_production_emissions_EV_parameter": np.asarray([controller.parameters_EV["production_emissions"]] * policy_duration),
            "history_driving_emissions_ICE": np.array(social_network.history_driving_emissions_ICE[-policy_duration:]),
            "history_driving_emissions_EV": np.array(social_network.history_driving_emissions_EV[-policy_duration:]),
            "history_total_emissions": np.asarray(social_network.history_total_emissions[-policy_duration:])
        }
    #need average efficieny of ice and ev beign DRIVEn, USED

    save_object(outputs, fileName + "/Data", "outputs")

    # Calculate time series derivatives
    derivatives = calculate_time_series_derivatives(outputs)

    # Get time steps
    time_steps = np.arange(len(outputs["history_ev_prop"]))

    # Plot derivatives
    plot_time_series_derivatives(fileName,base_params, derivatives, time_steps, whole_simulation)
    
    plot_normalized_time_series(outputs, fileName, base_params, whole_simulation)

    plot_emissions(social_network, time_series, fileName)
    # 2. Compute relative impacts
    relative_impacts = compute_relative_impacts(derivatives)

    # 4. Plot elasticities
    elasticities = calculate_time_series_elasticities(outputs, derivatives)
    plot_time_series_elasticities(fileName, base_params, elasticities, time_steps, whole_simulation)

    #5 stacked
    plot_stacked_contributions_with_net(
        derivatives=derivatives,
        time_steps=time_steps,
        fileName=fileName,
        base_params=base_params,
        whole_simulation=whole_simulation,
        title="Stacked First-Order Contributions to CO2 with Net"
    )

    print("Relative impacts (fractions of total):")
    for key, frac in relative_impacts.items():
        print(f"{key}: {frac*100:.2f}%")

    plt.show()

if __name__ == "__main__":
    main("results/single_experiment_13_04_47__04_01_2025")
