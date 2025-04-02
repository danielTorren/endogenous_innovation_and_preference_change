import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from sympy import N
from package.resources.utility import load_object
from matplotlib.lines import Line2D  # Add this at the top of your file if not already imported

# Make sure to reuse the original add_vertical_lines function
def add_vertical_lines(ax, base_params, color='black', linestyle='--', annotation_height_prop=[0.2, 0.2, 0.2]):
    """
    Adds dashed vertical lines to the plot at specified steps with vertical annotations.
    """
    # Determine the middle of the plot if no custom height is provided
    y_min, y_max = ax.get_ylim()

    annotation_height_0 = y_min + annotation_height_prop[0]*(y_max - y_min)
    # Add vertical line with annotation
    ev_sale_start_time = 144
    ax.axvline(ev_sale_start_time, color="black", linestyle=':')
    ax.annotate("Policy end", xy=(ev_sale_start_time, annotation_height_0),
                rotation=90, verticalalignment='center', horizontalalignment='right',
                fontsize=8, color='black')
    
########################################################################################################################
def plot_combined_policy_figures(base_params, fileName, outputs, outputs_BAU, outputs_f2, outputs_BAU_f2, top_policies, save_name="combined_policy", dpi=300):
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    time_steps = np.arange(base_params["duration_future"] - 1)
    start = base_params["duration_burn_in"] + base_params["duration_calibration"] 
    # Colors for consistent styling
    colors = plt.cm.tab20(np.linspace(0, 1, len(outputs)))

    # Get all unique policies from keys
    policy_pairs = outputs.keys()
    all_policies = sorted(list(set([p for pair in policy_pairs for p in pair])))

    # Use Set1 for high-contrast distinct colors
    color_map = plt.get_cmap('Set1', 10)
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}

    # Choose some distinct markers to cycle through (if needed)
    marker_list = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'H', '1']  # distinct markers
    policy_markers = {policy: marker_list[i % len(marker_list)] for i, policy in enumerate(all_policies)}


    # --- 1. EV Uptake
    ax1 = axs[0, 0]
    bau_ev = outputs_BAU["history_prop_EV"][:, start:]
    bau_ev_mean = np.mean(bau_ev, axis=0)
    bau_ev_ci = sem(bau_ev, axis=0) * t.ppf(0.975, df=bau_ev.shape[0] - 1)
    ax1.plot(time_steps, bau_ev_mean, color='black', label='BAU', linewidth=2)
    ax1.fill_between(time_steps, bau_ev_mean - bau_ev_ci, bau_ev_mean + bau_ev_ci, color='black', alpha=0.2)

    bau_ev = outputs_BAU_f2["history_prop_EV"][:, start:]
    bau_ev_mean = np.mean(bau_ev, axis=0)
    bau_ev_ci = sem(bau_ev, axis=0) * t.ppf(0.975, df=bau_ev.shape[0] - 1)
    ax1.plot(time_steps[144:], bau_ev_mean[144:], color='black', label='Ciruclar', linewidth=2, linestyle = ":")
    ax1.fill_between(time_steps[144:], bau_ev_mean[144:] - bau_ev_ci[144:], bau_ev_mean[144:] + bau_ev_ci[144:], color='black', alpha=0.2)


    for idx, ((policy1, policy2), output) in enumerate(outputs.items()):
        #label = f"{policy_abbr.get(policy1, policy1)} ({round(top_policies[(policy1, policy2)]['policy1_value'], 2)}), {policy_abbr.get(policy2, policy2)} ({round(top_policies[(policy1, policy2)]['policy2_value'], 2)})"
        label = f"{policy1} ({round(top_policies[(policy1, policy2)]['policy1_value'], 2)}), {policy2} ({round(top_policies[(policy1, policy2)]['policy2_value'], 2)})"
        data = output["history_prop_EV"][:, start:]

        mean = np.mean(data, axis=0)
        ci = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax1.plot(time_steps, mean, label=label, color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4)
        ax1.fill_between(time_steps, mean - ci, mean + ci, color=color1, alpha=0.2)

    for ((policy1, policy2), output_f2) in outputs_f2.items():
        data_f2 = output_f2["history_prop_EV"][:, start:]
        mean_f2 = np.mean(data_f2, axis=0)
        ci_f2 = sem(data_f2, axis=0) * t.ppf(0.975, df=data_f2.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax1.plot(
            time_steps[144:], mean_f2[144:], 
            color=color1, linestyle="dashed", linewidth=1,
            marker=marker_shape, markevery=32, markerfacecolor=color2, markeredgecolor=color2, markersize=4
        )
        ax1.fill_between(time_steps[144:], mean_f2[144:] - ci_f2[144:], mean_f2[144:] + ci_f2[144:], color=color1, alpha=0.2)


    #ax1.legend(fontsize="x-small")
    ax1.set_ylabel("EV Adoption")
    add_vertical_lines(ax1, base_params, annotation_height_prop=[0.5, 0.45, 0.45])
    
    # --- 2. Mean EV Prices (only)
    ax2 = axs[0, 1]
    bau_price = outputs_BAU["history_mean_price_ICE_EV_arr"]
    bau_new_EV = bau_price[:, :, 0, 1]
    bau_used_EV = bau_price[:, :, 1, 1]

    bau_new_mean = np.nanmean(bau_new_EV, axis=0)
    bau_new_ci = sem(bau_new_EV, axis=0, nan_policy='omit') * t.ppf(0.975, df=bau_new_EV.shape[0] - 1)
    ax2.plot(time_steps, bau_new_mean, color='black', label='BAU New', linewidth=2)
    ax2.fill_between(time_steps, bau_new_mean - bau_new_ci, bau_new_mean + bau_new_ci, color='black', alpha=0.2)

    bau_used_mean = np.nanmean(bau_used_EV, axis=0)
    bau_used_ci = sem(bau_used_EV, axis=0, nan_policy='omit') * t.ppf(0.975, df=bau_used_EV.shape[0] - 1)
    ax2.plot(time_steps, bau_used_mean, color='black', linestyle='--', label='BAU Used', linewidth=2)
    ax2.fill_between(time_steps, bau_used_mean - bau_used_ci, bau_used_mean + bau_used_ci, color='black', alpha=0.2)

    ################################################
    bau_price = outputs_BAU_f2["history_mean_price_ICE_EV_arr"]
    bau_new_EV = bau_price[:, :, 0, 1]
    bau_used_EV = bau_price[:, :, 1, 1]

    bau_new_mean = np.nanmean(bau_new_EV, axis=0)
    bau_new_ci = sem(bau_new_EV, axis=0, nan_policy='omit') * t.ppf(0.975, df=bau_new_EV.shape[0] - 1)
    ax2.plot(time_steps, bau_new_mean, color='black', label='BAU New', linewidth=2, linestyle = ":")
    ax2.fill_between(time_steps, bau_new_mean - bau_new_ci, bau_new_mean + bau_new_ci, color='black', alpha=0.2)

    bau_used_mean = np.nanmean(bau_used_EV, axis=0)
    bau_used_ci = sem(bau_used_EV, axis=0, nan_policy='omit') * t.ppf(0.975, df=bau_used_EV.shape[0] - 1)
    ax2.plot(time_steps[144:], bau_used_mean[144:], color='black', linestyle='-.', label='BAU Used', linewidth=2)
    ax2.fill_between(time_steps[144:], bau_used_mean[144:] - bau_used_ci[144:], bau_used_mean[144:] + bau_used_ci[144:], color='black', alpha=0.2)


    for idx, ((policy1, policy2), output) in enumerate(outputs.items()):
        label = f"{policy1} ({round(top_policies[(policy1, policy2)]['policy1_value'], 2)}), {policy2} ({round(top_policies[(policy1, policy2)]['policy2_value'], 2)})"
        mean_price = output["history_mean_price_ICE_EV_arr"]
        new_EV = mean_price[:, :, 0, 1]
        used_EV = mean_price[:, :, 1, 1]

        mean_new, ci_new = np.nanmean(new_EV, axis=0), sem(new_EV, axis=0, nan_policy='omit') * t.ppf(0.975, df=new_EV.shape[0] - 1)
        mean_used, ci_used = np.nanmean(used_EV, axis=0), sem(used_EV, axis=0, nan_policy='omit') * t.ppf(0.975, df=used_EV.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax2.plot(time_steps, mean_new, color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4)
        ax2.fill_between(time_steps, mean_new - ci_new, mean_new + ci_new, color=color1, alpha=0.2)

        ax2.plot(time_steps, mean_used, color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4, linestyle=":")
        ax2.fill_between(time_steps, mean_used - ci_used, mean_used + ci_used, color=color1, alpha=0.2)

    for ((policy1, policy2), output_f2) in outputs_f2.items():
        mean_price_f2 = output_f2["history_mean_price_ICE_EV_arr"]
        mean_new_f2 = np.nanmean(mean_price_f2[:, :, 0, 1], axis=0)
        mean_used_f2 = np.nanmean(mean_price_f2[:, :, 1, 1], axis=0)
        ci_new_f2 =  sem(mean_price_f2[:, :, 0, 1], axis=0, nan_policy='omit') * t.ppf(0.975, df=mean_price_f2[:, :, 0, 1].shape[0] - 1)
        ci_used_f2 = sem(mean_price_f2[:, :, 1, 1], axis=0, nan_policy='omit') * t.ppf(0.975, df=mean_price_f2[:, :, 1, 1].shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax2.plot(
            time_steps[144:], mean_new_f2[144:], 
            color=color1, linestyle=":", marker=marker_shape, markevery=32,
            markerfacecolor=color2, markeredgecolor=color2, markersize=4
        )
        ax2.fill_between(time_steps[144:], mean_new_f2[144:] - ci_new_f2[144:], mean_new_f2[144:] + ci_new_f2[144:], color=color1, alpha=0.2)
        ax2.plot(
            time_steps[144:], mean_used_f2[144:], 
            color=color1, linestyle="-.", marker=marker_shape, markevery=32,
            markerfacecolor=color2, markeredgecolor=color2, markersize=4
        )
        ax2.fill_between(time_steps[144:], mean_used_f2[144:] - ci_used_f2[144:], mean_used_f2[144:] + ci_used_f2[144:], color=color1, alpha=0.2)


    ax2.set_ylabel("EV Price, $")
    # Add after plotting all the policy lines in ax2
    custom_legend = [
        Line2D([0], [0], color="black", linestyle='-', label='New'),
        Line2D([0], [0], color="black", linestyle='--', label='Used'),
        Line2D([0], [0], color="black", linestyle=':', label='New Circular'),
        Line2D([0], [0], color="black", linestyle='-.', label='Used Circular')
    ]
    ax2.legend(handles=custom_legend, loc='lower right', fontsize="small", ncols = 2)
    add_vertical_lines(ax2, base_params, annotation_height_prop=[0.88, 0.45, 0.45])

    # --- 3. Production Emissions
    ax3 = axs[1, 0]
    
    bau_emissions = np.cumsum(outputs_BAU["history_total_emissions"], axis = 1)*1e-9
    bau_mean = np.mean(bau_emissions, axis=0)
    bau_ci = sem(bau_emissions, axis=0) * t.ppf(0.975, df=bau_emissions.shape[0] - 1)
    ax3.plot(time_steps, bau_mean, color='black', label='BAU', linewidth=2)
    ax3.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    bau_emissions = np.cumsum(outputs_BAU_f2["history_total_emissions"], axis = 1)*1e-9
    bau_mean = np.mean(bau_emissions, axis=0)
    bau_ci = sem(bau_emissions, axis=0) * t.ppf(0.975, df=bau_emissions.shape[0] - 1)
    ax3.plot(time_steps[144:], bau_mean[144:], color='black', label='BAU Circular', linewidth=2, linestyle=":")
    ax3.fill_between(time_steps[144:], bau_mean[144:] - bau_ci[144:], bau_mean[144:] + bau_ci[144:], color='black', alpha=0.2)

    for idx, ((policy1, policy2), output) in enumerate(outputs.items()):
        label = f"{policy1} ({round(top_policies[(policy1, policy2)]['policy1_value'], 2)}), {policy2} ({round(top_policies[(policy1, policy2)]['policy2_value'], 2)})"
        data = np.cumsum(output["history_total_emissions"], axis = 1)*1e-9
        mean = np.mean(data, axis=0)
        ci = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax3.plot(time_steps, mean, color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4, label = label)
        ax3.fill_between(time_steps, mean - ci, mean + ci, color=color1, alpha=0.2)
    for idx, ((policy1, policy2), output) in enumerate(outputs_f2.items()):
        data = np.cumsum(output["history_total_emissions"], axis = 1)*1e-9
        mean = np.mean(data, axis=0)
        ci = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax3.plot(time_steps[144:], mean[144:], color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4, linestyle=":")
        ax3.fill_between(time_steps[144:], mean[144:] - ci[144:], mean[144:] + ci[144:], color=color1, alpha=0.2)

    ax3.set_ylabel("Total Emissions, MTCO2")
    ax3.set_xlabel("Time Step, months")
    #ax3.legend(fontsize="x-small", loc = "lower right", ncols = 2)
    add_vertical_lines(ax3, base_params, annotation_height_prop=[0.8, 0.45, 0.45])
    
    # --- 4. Mean Car Age
    ax4 = axs[1, 1]
    bau_age = outputs_BAU["history_mean_car_age"]
    bau_mean = np.mean(bau_age, axis=0)
    bau_ci = sem(bau_age, axis=0) * t.ppf(0.975, df=bau_age.shape[0] - 1)
    ax4.plot(time_steps, bau_mean, color='black', label='BAU', linewidth=2)
    ax4.fill_between(time_steps, bau_mean - bau_ci, bau_mean + bau_ci, color='black', alpha=0.2)

    bau_age = outputs_BAU_f2["history_mean_car_age"]
    bau_mean = np.mean(bau_age, axis=0)
    bau_ci = sem(bau_age, axis=0) * t.ppf(0.975, df=bau_age.shape[0] - 1)
    ax4.plot(time_steps[144:], bau_mean[144:], color='black', label='BAU', linewidth=2, linestyle=":")
    ax4.fill_between(time_steps[144:], bau_mean[144:] - bau_ci[144:], bau_mean[144:] + bau_ci[144:], color='black', alpha=0.2)

    for idx, ((policy1, policy2), output) in enumerate(outputs.items()):
        data = output["history_mean_car_age"][:, :]
        mean = np.mean(data, axis=0)
        ci = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax4.plot(time_steps, mean, color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4)
        ax4.fill_between(time_steps, mean - ci, mean + ci, color=color1, alpha=0.2)
    for idx, ((policy1, policy2), output) in enumerate(outputs_f2.items()):
        data = output["history_mean_car_age"][:, :]
        mean = np.mean(data, axis=0)
        ci = sem(data, axis=0) * t.ppf(0.975, df=data.shape[0] - 1)

        color1 = policy_colors[policy1]
        color2 = policy_colors[policy2]
        marker_shape = policy_markers[policy2]

        ax4.plot(time_steps[144:], mean[144:], color=color1, marker=marker_shape, markevery=32,
        markerfacecolor=color2, markeredgecolor=color2, markersize=4, linestyle= ":")
        ax4.fill_between(time_steps[144:], mean[144:] - ci[144:], mean[144:] + ci[144:], color=color1, alpha=0.2)

    ax4.set_ylabel("Car Age, months")
    ax4.set_xlabel("Time Step, months")
    add_vertical_lines(ax4, base_params, annotation_height_prop=[0.5, 0.3, 0.3])
   # Gather handles/labels from ax1 (or combine from others if needed)
    handles, labels = ax1.get_legend_handles_labels()

    # Create figure-wide legend below the subplots
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.00))

    plt.tight_layout(rect=[0.01, 0.08, 0.98, 1])  # Leaves space at the bottom
    plt.subplots_adjust(wspace=0.15)  # increase spacing between columns
    plt.savefig(f"{fileName}/Plots/DUAL_combined_policy_dashboard.png", dpi=dpi)

def main(fileName, f2):
    base_params = load_object(fileName + "/Data", "base_params")
    outputs_BAU = load_object(fileName + "/Data", "outputs_BAU")
    outputs_BAU_f2 = load_object(f2 + "/Data", "outputs_BAU")
    outputs = load_object(fileName + "/Data", "outputs")
    outputs_f2 = load_object(f2 + "/Data", "outputs")
    top_policies = load_object(fileName + "/Data", "top_policies")

    plot_combined_policy_figures(base_params, fileName, outputs, outputs_BAU, outputs_f2, outputs_BAU_f2, top_policies, save_name="combined_policy", dpi=300)
    plt.show()
    

if __name__ == "__main__":
    main(fileName = "results/2D_low_intensity_policies_11_52_46__02_04_2025", f2= "results/2D_low_intensity_policies_12_14_55__02_04_2025")
