import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from package.calibration.NN_multi_round_calibration_multi_gen import convert_data
from package.resources.utility import load_object
from package.plotting_data.single_experiment_plot import save_and_show

def calc_mape(actual, predicted):
    #print((actual - predicted) / actual)
    #print(np.mean(np.abs((actual - predicted) / actual)))

    return np.mean(np.abs((actual - predicted) / actual))*100

def plot_mape_heatmap(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    # Calculate MAPE for each parameter pair and average over seeds
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    mape_values = np.zeros((num_vary_1, num_vary_2))

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            # Extract predictions for current parameter pair
            predictions = data_array_ev_prop[i, j]  # Shape: (seeds, time steps)
            #print(predictions.shape)
            # Calculate MAPE for each seed and take the mean across seeds
            seed_mape = []
            
            #print("NEW", param_1,param_2)
            for i, pred in enumerate(predictions):
                sim_data = convert_data(pred,base_params)
                mape_data = calc_mape(real_data, sim_data)
                #print("mape_data", mape_data)
                seed_mape.append(mape_data)
            #print("seed_mape", seed_mape)
            #print("np.mean(seed_mape)", np.mean(seed_mape))
            mape_values[i, j] = np.mean(seed_mape)

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(mape_values, cmap='viridis', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=ax, label="Mean MAPE (%)")
    
    # Add axis labels and titles
    ax.set_xticks(range(num_vary_2))
    ax.set_xticklabels(vary_2["property_list"], rotation=45)
    ax.set_yticks(range(num_vary_1))
    ax.set_yticklabels(vary_1["property_list"])
    ax.set_xlabel(vary_2["property_varied"])
    ax.set_ylabel(vary_1["property_varied"])
    ax.set_title("MAPE Heatmap: Real vs. Simulated EV Stock")
    
    # Save and show the heatmap
    save_and_show(fig, fileName, "mape_heatmap", dpi)

def plot_ev_stock_all_combinations(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])

    fig, axes = plt.subplots(
        nrows=num_vary_1, ncols=num_vary_2, figsize=(20, 20), sharex=True, sharey=True
    )

    x_values = np.arange(len(real_data))  # Assuming time series index

    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            ax = axes[i, j]

            # Extract predictions for current parameter pair
            predictions = data_array_ev_prop[i, j, :, :]  # Shape: (seeds, time steps)
            processed_data_seeds = [convert_data(data, base_params) for data in predictions ]
            # Convert to numpy array and ensure consistent dimensions
            processed_data_array = np.array(processed_data_seeds)

            # Calculate mean and confidence intervals across seeds
            means = np.mean(processed_data_array, axis=0)
            confidence_intervals = t.ppf(0.975, len(processed_data_array)-1) * sem(processed_data_array, axis=0)

            lower_bounds = means - confidence_intervals
            upper_bounds = means + confidence_intervals

            # Plot mean and confidence intervals
            ax.plot(x_values, means, color='blue')
            ax.fill_between(x_values, lower_bounds, upper_bounds, color='blue', alpha=0.2)
            ax.plot(x_values, real_data, color='orange', linestyle='--', linewidth=2)

            # for major ticks
            ax.set_xticks([])
            # for minor ticks
            ax.set_xticks([], minor=True)
            # for major ticks
            ax.set_yticks([])
            # for minor ticks
            ax.set_yticks([], minor=True)

            # Title with parameter combinations
            #ax.set_title(f"{round(param_1,2)}, {round(param_2,2)}")

            # Remove legends and adjust grid
            #ax.grid(True)

            ax.set_ylim([min(real_data), max(real_data)])

    # Optional: Adjust axis labels only for edge plots
    fig.supxlabel("Months (2010-2022)")
    fig.supylabel("EV Stock %")

    # Adjust layout and save the figure
    plt.tight_layout()
    save_and_show(fig, fileName, "plot_ev_stock_combinations", dpi)

def plot_mape_heatmaps_per_seed(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    num_seeds = data_array_ev_prop.shape[2]  # Number of seeds

    # Arrange subplots in 2 rows
    ncols = (num_seeds + 1) // 2
    nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), constrained_layout=True)

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Compute the MAPE heatmaps and plot them
    mape_values_list = []
    for seed_index in range(num_seeds):
        # Initialize MAPE values for this seed
        mape_values = np.zeros((num_vary_1, num_vary_2))

        for i, param_1 in enumerate(vary_1["property_list"]):
            for j, param_2 in enumerate(vary_2["property_list"]):
                # Extract predictions for the current parameter pair and seed
                predictions = data_array_ev_prop[i, j, seed_index, :]  # Shape: (time steps)
                
                # Convert data and calculate MAPE
                sim_data = convert_data(predictions, base_params)
                mape_data = calc_mape(real_data, sim_data)
                mape_values[i, j] = mape_data
        
        # Store MAPE values for global color bar scaling
        mape_values_list.append(mape_values)

        # Plot heatmap for the current seed
        ax = axes[seed_index]
        cax = ax.imshow(mape_values, cmap='viridis', origin='lower', aspect='auto')

        # Set titles
        ax.set_title(f"Seed {seed_index + 1}", fontsize=10)

        # Add x-axis labels only for the last row
        if seed_index >= ncols:
            ax.set_xticks(range(num_vary_2))
            ax.set_xticklabels(np.round(vary_2["property_list"],2), rotation=45)

        # Add y-axis labels only for the first column
        if seed_index % ncols == 0:
            ax.set_yticks(range(num_vary_1))
            ax.set_yticklabels(np.round(vary_1["property_list"],2))

        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    fig.supxlabel(vary_2["property_varied"])
    fig.supylabel(vary_1["property_varied"])

    # Adjust and add a single color bar for all subplots
    mape_values_all = np.vstack(mape_values_list)  # Combine all MAPE values for consistent scaling
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=mape_values_all.min(), vmax=mape_values_all.max())),
        ax=axes,
        orientation='vertical',
        shrink=0.8
    )
    cbar.set_label("MAPE (%)")

    # Remove unused subplots
    for ax in axes[num_seeds:]:
        ax.axis("off")

    # Save and show the figure
    save_and_show(fig, fileName, "mape_heatmaps_all_seeds", dpi)

def plot_ev_stock_per_seed(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    num_seeds = data_array_ev_prop.shape[2]  # Number of seeds

    # Arrange subplots in 2 rows
    ncols = (num_seeds + 1) // 2
    nrows = 2

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), constrained_layout=True, sharex=True, sharey=True
    )

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    x_values = np.arange(len(real_data))  # Assuming time series index
    colors = plt.cm.tab20(np.linspace(0, 1, num_vary_1 * num_vary_2))  # Unique colors for each line

    # Loop through seeds
    for seed_index in range(num_seeds):
        ax = axes[seed_index]

        # Loop through parameter combinations
        for i, param_1 in enumerate(vary_1["property_list"]):
            for j, param_2 in enumerate(vary_2["property_list"]):
                predictions = data_array_ev_prop[i, j, seed_index, :]  # Shape: (time steps)
                sim_data = convert_data(predictions, base_params)

                # Generate label for the first seed only
                label = f"{vary_1['property_varied']}: {round(param_1, 2)}, {vary_2['property_varied']}: {round(param_2, 2)}" if seed_index == 0 else None
                
                # Plot each parameter combination
                ax.plot(x_values, sim_data, label=label, color=colors[i * num_vary_2 + j], alpha=0.8)

        # Plot real data for reference
        ax.plot(x_values, real_data, color='black', linestyle='--', linewidth=2, label="Actual Data" if seed_index == 0 else None)

        # Set titles
        ax.set_title(f"Seed {seed_index + 1}", fontsize=10)

    # Add x-axis labels only for the bottom row
    for ax in axes[-ncols:]:
        ax.set_xlabel("Months (2010-2022)")

    # Add y-axis labels only for the first column
    for ax in axes[::ncols]:
        ax.set_ylabel("EV Stock %")

    # Hide unused subplots
    for ax in axes[num_seeds:]:
        ax.axis("off")

    # Add a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", fontsize=8, title="Parameter Combinations", title_fontsize=10, frameon=False
    )

    # Save and show the figure
    save_and_show(fig, fileName, "plot_ev_stock_per_seed", dpi)

def plot_ev_stock_per_seed_mape(base_params, real_data, data_array_ev_prop, vary_1, vary_2, fileName, dpi=600):
    num_vary_1 = len(vary_1["property_list"])
    num_vary_2 = len(vary_2["property_list"])
    num_seeds = data_array_ev_prop.shape[2]  # Number of seeds

    # Arrange subplots in 2 rows
    ncols = (num_seeds + 1) // 2
    nrows = 2

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), constrained_layout=True, sharex=True, sharey=True
    )

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    x_values = np.arange(len(real_data))  # Assuming time series index

    # Compute MAPE for all combinations
    mape_values = np.zeros((num_vary_1, num_vary_2, num_seeds))
    for i, param_1 in enumerate(vary_1["property_list"]):
        for j, param_2 in enumerate(vary_2["property_list"]):
            for seed_index in range(num_seeds):
                predictions = data_array_ev_prop[i, j, seed_index, :]  # Shape: (time steps)
                sim_data = convert_data(predictions, base_params)
                mape_values[i, j, seed_index] = calc_mape(real_data, sim_data)

    # Normalize MAPE values for color mapping
    norm = plt.Normalize(vmin=mape_values.min(), vmax=mape_values.max())
    cmap = plt.cm.viridis

    # Loop through seeds
    for seed_index in range(num_seeds):
        ax = axes[seed_index]

        # Loop through parameter combinations
        for i, param_1 in enumerate(vary_1["property_list"]):
            for j, param_2 in enumerate(vary_2["property_list"]):
                predictions = data_array_ev_prop[i, j, seed_index, :]  # Shape: (time steps)
                sim_data = convert_data(predictions, base_params)
                mape = mape_values[i, j, seed_index]

                # Plot each parameter combination, color-coded by MAPE
                ax.plot(x_values, sim_data, color=cmap(norm(mape)), alpha=0.8)

        # Plot real data for reference
        ax.plot(x_values, real_data, color='orange', linestyle='--', linewidth=2)

        # Set titles
        ax.set_title(f"Seed {seed_index + 1}", fontsize=10)

    # Add x-axis labels only for the bottom row
    for ax in axes[-ncols:]:
        ax.set_xlabel("Months (2010-2022)")

    # Add y-axis labels only for the first column
    for ax in axes[::ncols]:
        ax.set_ylabel("EV Stock %")

    # Hide unused subplots
    for ax in axes[num_seeds:]:
        ax.axis("off")

    # Add a single color bar for MAPE
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for the color bar
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.8)
    cbar.set_label("MAPE (%)")

    # Save and show the figure
    save_and_show(fig, fileName, "plot_ev_stock_per_seed_mape", dpi)

# Main function
def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_array_ev_prop = load_object(fileName + "/Data", "data_array_ev_prop")
        vary_1 = load_object(fileName + "/Data", "vary_1")
        vary_2 = load_object(fileName + "/Data", "vary_2")
    except FileNotFoundError:
        print("Data files not found.")
        return

    calibration_data_output = load_object("package/calibration_data", "calibration_data_output")

    # Extract actual EV stock proportions (2010-2022)
    EV_stock_prop_2010_22 = calibration_data_output["EV Prop"]

    # Plot MAPE heatmap
    #plot_mape_heatmap(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    #plot_mape_heatmaps_per_seed(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plot_ev_stock_all_combinations(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    #plot_ev_stock_per_seed(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    #plot_ev_stock_per_seed_mape(base_params, EV_stock_prop_2010_22, data_array_ev_prop, vary_1, vary_2, fileName, dpi)
    plt.show()

if __name__ == "__main__":
    main("results/MAPE_ev_2D_13_55_58__10_02_2025")
