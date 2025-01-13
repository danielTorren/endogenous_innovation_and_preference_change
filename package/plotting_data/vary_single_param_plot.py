import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t
from package.resources.utility import load_object

def plot_distance(data_array, property_values_list, fileName, name_property, property_save, dpi=600):
    """
    Plots multiple subplots of mean user distance over time for different delta values.
    Each subplot corresponds to one delta value and contains multiple lines for different seeds.
    
    Parameters:
    -----------
    data_array : ndarray
        Data array of shape [num_deltas, num_seeds, time_steps, num_individuals].
    property_values_list : list or array
        The list of delta values corresponding to the first dimension of data_array.
    fileName : str
        The directory name or base filename where the plot should be saved.
    dpi : int
        Dots per inch for the saved figure.
    """

    num_deltas = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]

    # Create time series array if you don't have one
    time_series = np.arange(time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_deltas, figsize=(5 * num_deltas, 5), sharey=True)

    # If there's only one delta, axes might not be an array
    if num_deltas == 1:
        axes = [axes]

    for i, delta in enumerate(property_values_list):
        ax = axes[i]

        # For each seed, compute the mean over individuals and plot
        for seed in range(num_seeds):
            # data for this delta and seed: shape (time_steps, num_individuals)
            data = data_array[i, seed, :, :]  # shape (time_steps, num_individuals)

            # Mean over individuals at each time step
            mean_distance = np.mean(data, axis=1)
            standard_error = sem(data, axis=1)
            confidence_interval = t.ppf(0.975, df=data.shape[0] - 1) * standard_error
            
            # Plot mean and capture the line object
            line = ax.plot(time_series, mean_distance, label=f"Seed {seed+1}", alpha=0.7)[0]

            # Use the line color for the confidence interval
            ax.fill_between(
                time_series, 
                mean_distance - confidence_interval, 
                mean_distance + confidence_interval, 
                color=line.get_color(), 
                alpha=0.2,
            )

        # Format each subplot
        ax.set_title(f"{delta}")
        #ax.set_xlabel("Time Step")
        #if i == 0:
        #    ax.set_ylabel("Mean User Distance")

        # Add a legend if desired (or only in one subplot)
        #ax.legend()

    fig.supxlabel("Time Step")
    fig.supylabel("Mean User Distance")
    # Adjust layout
    #plt.tight_layout()

    # Save and show
    fig.savefig(f"{fileName}/user_distance_multi_{property_save}.png", dpi=dpi)
    #plt.show()

def plot_ev_prop(data_array, property_values_list, fileName, name_property, property_save, dpi=600):


    num_deltas = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]

    # Create time series array if you don't have one
    time_series = np.arange(time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_deltas, figsize=(5 * num_deltas, 5), sharey=True)

    # If there's only one delta, axes might not be an array
    if num_deltas == 1:
        axes = [axes]

    for i, delta in enumerate(property_values_list):
        ax = axes[i]

        # For each seed, compute the mean over individuals and plot
        for seed in range(num_seeds):
            # data for this delta and seed: shape (time_steps, num_individuals)
            data = data_array[i, seed, :]  # shape (time_steps, num_individuals)
            # Plot mean and capture the line object
            ax.plot(time_series, data , label=f"Seed {seed+1}", alpha=0.7)

        # Format each subplot
        ax.set_title(f"{delta}")
        #ax.set_xlabel("Time Step")
        #if i == 0:
        #    ax.set_ylabel("EV prop")

        ax.grid()

        # Add a legend if desired (or only in one subplot)
        #ax.legend()

    fig.supxlabel("Time Step")
    fig.supylabel("EV prop")
    # Adjust layout
    #plt.tight_layout()

    # Save and show
    fig.savefig(f"{fileName}/user_ev_prop_multi_{property_save}.png", dpi=dpi)

def plot_age(data_array, property_values_list, fileName, name_property, property_save, dpi=600):
    """
    Plots multiple subplots of mean user distance over time for different delta values.
    Each subplot corresponds to one delta value and contains multiple lines for different seeds.
    
    Parameters:
    -----------
    data_array : ndarray
        Data array of shape [num_deltas, num_seeds, time_steps, num_individuals].
    property_values_list : list or array
        The list of delta values corresponding to the first dimension of data_array.
    fileName : str
        The directory name or base filename where the plot should be saved.
    dpi : int
        Dots per inch for the saved figure.
    """

    num_deltas = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]

    # Create time series array if you don't have one
    time_series = np.arange(time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_deltas, figsize=(5 * num_deltas, 5), sharey=True)

    # If there's only one delta, axes might not be an array
    if num_deltas == 1:
        axes = [axes]

    for i, delta in enumerate(property_values_list):
        ax = axes[i]

        # For each seed, compute the mean over individuals and plot
        for seed in range(num_seeds):
            # data for this delta and seed: shape (time_steps, num_individuals)
            data = data_array[i, seed, :, :]  # shape (time_steps, num_individuals)

            # Mean over individuals at each time step
            mean_distance = np.mean(data, axis=1)
            standard_error = sem(data, axis=1)
            confidence_interval = t.ppf(0.975, df=data.shape[0] - 1) * standard_error
            
            # Plot mean and capture the line object
            line = ax.plot(time_series, mean_distance, label=f"Seed {seed+1}", alpha=0.7)[0]

            # Use the line color for the confidence interval
            ax.fill_between(
                time_series, 
                mean_distance - confidence_interval, 
                mean_distance + confidence_interval, 
                color=line.get_color(), 
                alpha=0.2,
            )

        # Format each subplot
        ax.set_title(f"{delta}")
        #ax.set_xlabel("Time Step")
        #if i == 0:
        #    ax.set_ylabel("Car Age")

        # Add a legend if desired (or only in one subplot)
        #ax.legend()

    fig.supxlabel("Time Step")
    fig.supylabel("Car Age")

    # Adjust layout
    #plt.tight_layout()

    # Save and show
    fig.savefig(f"{fileName}/user_age_multi_{property_save}.png", dpi=dpi)
    #plt.show()

def plot_emissions(data_array, property_values_list, fileName, name_property, property_save, dpi=600):

    num_deltas = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]

    # Create time series array if you don't have one
    time_series = np.arange(time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_deltas, figsize=(5 * num_deltas, 5), sharey=True)

    # If there's only one delta, axes might not be an array
    if num_deltas == 1:
        axes = [axes]

    for i, delta in enumerate(property_values_list):
        ax = axes[i]

        # For each seed, compute the mean over individuals and plot
        for seed in range(num_seeds):
            # data for this delta and seed: shape (time_steps, num_individuals)
            data = data_array[i, seed, :]  # shape (time_steps, num_individuals)
            # Plot mean and capture the line object
            ax.plot(time_series, data , label=f"Seed {seed+1}", alpha=0.7)

        # Format each subplot
        ax.set_title(f"{delta}")
        #ax.set_xlabel("Time Step")
        #if i == 0:
        #    ax.set_ylabel("Emisisons")

        # Add a legend if desired (or only in one subplot)
        #ax.legend()

    fig.supxlabel("Time Step")
    fig.supylabel("Emisisons")

    # Adjust layout
    #plt.tight_layout()

    # Save and show
    fig.savefig(f"{fileName}/emissions_{property_save}.png", dpi=dpi)

def plot_price(data_array, property_values_list, fileName, name_property, property_save, dpi=600):

    num_deltas = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]

    # Create time series array if you don't have one
    time_series = np.arange(time_steps)

    # Create subplots with 2 rows per delta
    fig, axes = plt.subplots(nrows=2, ncols=num_deltas, figsize=(5 * num_deltas, 10), sharex=True, sharey=False)

    # If there's only one delta, axes might not be an array, ensure it's consistent
    if num_deltas == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, delta in enumerate(property_values_list):
        ax_new = axes[0, i]  # Upper row for new prices
        ax_second_hand = axes[1, i]  # Lower row for second-hand prices

        # For each seed, compute the mean over individuals and plot
        for seed in range(num_seeds):
            # data for this delta and seed: shape (time_steps, num_individuals)
            data = data_array[i, seed, :, :]  # shape (time_steps, num_individuals)
            data_new = data[:, 0]
            data_second_hand = data[:, 1]

            # Plot new prices on the upper row
            ax_new.plot(time_series, data_new, label=f"Seed {seed+1}", alpha=0.7)

            # Plot second-hand prices on the lower row
            ax_second_hand.plot(time_series, data_second_hand, label=f"Seed {seed+1}", alpha=0.7)

        # Format each subplot
        ax_new.set_title(f"{name_property} = {delta} (New)")
        ax_second_hand.set_title(f"{name_property} = {delta} (Second-hand)")

        if i == 0:
            ax_new.set_ylabel("Price (New)")
            ax_second_hand.set_ylabel("Price (Second-hand)")

        #ax_second_hand.set_xlabel("Time Step")

        # Add a legend if desired (or only in one subplot)
        #ax_new.legend()
        #ax_second_hand.legend()
    fig.supxlabel("Time Step")

    # Adjust layout
    ##plt.tight_layout()

    # Save and show
    fig.savefig(f"{fileName}/price_{property_save}.png", dpi=dpi)
################################################################################################

def plot_efficiency(data_array, property_values_list, fileName, name_property, property_save, dpi=600):

    num_deltas = data_array.shape[0]
    num_seeds = data_array.shape[1]
    time_steps = data_array.shape[2]

    # Create time series array if you don't have one
    time_series = np.arange(time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_deltas, figsize=(5 * num_deltas, 5), sharey=True)

    # If there's only one delta, axes might not be an array
    if num_deltas == 1:
        axes = [axes]

    for i, delta in enumerate(property_values_list):
        ax = axes[i]

        # For each seed, compute the mean over individuals and plot
        for seed in range(num_seeds):
            # data for this delta and seed: shape (time_steps, num_individuals)
            data = data_array[i, seed, :]  # shape (time_steps, num_individuals)
            # Plot mean and capture the line object
            ax.plot(time_series, data , label=f"Seed {seed+1}", alpha=0.7)

        # Format each subplot
        ax.set_title(f"{delta}")
        #ax.set_xlabel("Time Step")
        #if i == 0:
        #    ax.set_ylabel("Emisisons")

        # Add a legend if desired (or only in one subplot)
        #ax.legend()

    fig.supxlabel("Time Step")
    fig.supylabel("Efficiency ICE")

    # Adjust layout
    #plt.tight_layout()

    # Save and show
    fig.savefig(f"{fileName}/eff_{property_save}.png", dpi=dpi)


# Sample main function
def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_distance = load_object(fileName + "/Data", "data_array_distance")
        data_array_EV_prop = load_object(fileName + "/Data", "data_array_EV_prop")
        data_array_age =  load_object(fileName + "/Data", "data_array_age")
        data_array_price =  load_object(fileName + "/Data", "data_array_price")
        data_array_emissions = load_object(fileName + "/Data", "data_array_emissions")
        data_array_efficiency = load_object(fileName + "/Data", "data_array_efficiency")
        vary_single = load_object(fileName + "/Data", "vary_single")
        
    except FileNotFoundError:
        print("Data files not found.")
        return
    
    property_values_list = vary_single["property_values_list"]
    name_property = vary_single["property_varied"] 
    property_save = vary_single["property_varied"]

    plot_distance(data_array_distance, property_values_list, fileName, name_property, property_save, 600)
    plot_ev_prop(data_array_EV_prop, property_values_list, fileName, name_property, property_save, 600)
    plot_age(data_array_age, property_values_list, fileName, name_property, property_save, 600)
    plot_price(data_array_price , property_values_list, fileName, name_property, property_save, 600)
    plot_emissions(data_array_emissions , property_values_list, fileName, name_property, property_save, 600)
    plot_efficiency(data_array_efficiency , property_values_list, fileName, name_property, property_save, 600)

    plt.show()

if __name__ == "__main__":
    main("results/single_param_vary_13_47_46__13_01_2025")