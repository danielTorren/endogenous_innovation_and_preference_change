import copy
import sys

from package.resources.utility import load_object, save_object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import os
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

# Marker helper functions
def half_circle_marker(start, end, offset=-45):
    angles = np.linspace(np.radians(start + offset), np.radians(end + offset), 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([verts, [0, 0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

# Intensity-to-marker-size scaling
def scale_marker_size(value, policy, policy_ranges, scale_marker):
    min_val = policy_ranges[policy]["min"]
    max_val = policy_ranges[policy]["max"]
    if max_val - min_val == 0:
        return scale_marker  # Fixed size if no variation
    norm = (value - min_val) / (max_val - min_val)
    return np.maximum(0, norm * scale_marker)

def full_circle_marker():
    angles = np.linspace(0, 2 * np.pi, 100)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([verts, [verts[0]]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    return Path(verts, codes)



def plot_emissions_tradeoffs_from_outcomes(
        base_params, 
        pairwise_outcomes_complied, 
        single_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.9,
        max_ev_uptake=1.0,
        dpi=300,
        insets=True,
        plot_name="emissions_tradeoff"
        ):

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # --- Setup
    okabe_ito_colors = ['#E69F00','#009E73', '#56B4E9', '#F0E442', 
                    '#0072B2', '#D55E00', '#CC79A7', '#000000']
    
    color_map = ListedColormap(okabe_ito_colors)
    all_policies = sorted({p for pair in pairwise_outcomes_complied for p in pair})
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}
    policy_ranges = {policy: {"min": 0, "max": 0} for policy in all_policies}

    plotted_points = []
    scale_marker = 350
    scale_small = 1
    ########################################################################################

    # --- Zoom windows
    e_min = 0.125
    e_max = 0.15#0.145
    c_min = -0.08
    c_max = 0.3

    e_min2 = 0.131
    e_max2 = 0.135
    u_min = 72
    u_max = 78

    axins = axins2 = None
    if insets:
        # --- Create zoom-in inset axes for the top plot
        axins = inset_axes(ax_top, width="70%", height="45%",
                        bbox_to_anchor=(0.01, 0.52, 1, 1),
                        bbox_transform=ax_top.transAxes,
                        loc='lower left'
                        )

        axins.set_xlim( e_min, e_max)
        axins.set_ylim(c_min, c_max)

        axins.set_xticks([])
        axins.set_yticks([])

        # --- Create zoom-in inset axes for the bottom plot
        axins2 = inset_axes(ax_bottom, width="30%", height="30%",
                        bbox_to_anchor=(0.15, 0.01, 1, 1),
                        bbox_transform=ax_bottom.transAxes,
                        loc='lower left'
                        )

        axins2.set_xlim(e_min2, e_max2)
        axins2.set_ylim(u_min, u_max)

        axins2.set_xticks([])
        axins2.set_yticks([])

    ##########################################################################################
    
    #axins.tick_params(axis='both', which='major', labelsize=8)

    # --- Gather data from pairs
    for (policy1, policy2), results in pairwise_outcomes_complied.items():
        for entry in results:
            ev = entry["mean_ev_uptake"]
            if (min_ev_uptake <= ev <= max_ev_uptake):

                e = entry["mean_emissions_cumulative"] * 1e-9
                u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                c = entry["mean_net_cost"] * 1e-9

                p1_val = entry["policy1_value"]
                p2_val = entry["policy2_value"]

                policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], p1_val)
                policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], p2_val)

                plotted_points.append((e, u, c, policy1, policy2, p1_val, p2_val))

    # --- BAU
    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
    bau_cost = outcomes_BAU["mean_net_cost"] * 1e-9

    ax_top.scatter(bau_em, bau_cost, s=scale_marker, color='black', edgecolor='black', label="BAU")
    ax_bottom.scatter(bau_em, bau_ut, s=scale_marker, color='black', edgecolor='black')
    # Also plot BAU on the inset axes
    if insets and e_min <= bau_em <= e_max and c_min <= bau_cost <= c_max :
        axins.scatter(bau_em, bau_cost, s=scale_marker*scale_small, color='black', edgecolor='black')
    if insets and e_min2 <= bau_em <= e_max2 and u_min <= bau_ut <= u_max :
        axins2.scatter(bau_em, bau_ut, s=scale_marker*scale_small, color='black', edgecolor='black')

    for (policy1, policy2), results in pairwise_outcomes_complied.items():
        for entry in results:
            print( (policy1, policy2),entry["mean_ev_uptake"])
            if (min_ev_uptake <= entry["mean_ev_uptake"] <= max_ev_uptake):
                e = entry["mean_emissions_cumulative"] * 1e-9
                u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                c = entry["mean_net_cost"] * 1e-9

                p1_val = entry["policy1_value"]
                p2_val = entry["policy2_value"]

                entry["emissions_cumulative"] = entry["emissions_cumulative_driving"] + entry["emissions_cumulative_production"]
                emissions_array = np.array(entry["emissions_cumulative"]) * 1e-9
                utility_array = np.array(entry["utility_cumulative"]) / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
                cost_array = np.array(entry["net_cost"]) * 1e-9

                n_seeds = len(emissions_array)
                e_err = 1.96 * np.std(emissions_array) / np.sqrt(n_seeds)
                u_err = 1.96 * np.std(utility_array) / np.sqrt(n_seeds)
                c_err = 1.96 * np.std(cost_array) / np.sqrt(n_seeds)

                policy_ranges[policy1]["max"] = max(policy_ranges[policy1]["max"], p1_val)
                policy_ranges[policy2]["max"] = max(policy_ranges[policy2]["max"], p2_val)

                color1 = policy_colors[policy1]
                color2 = policy_colors[policy2]
                size1 = scale_marker_size(p1_val, policy1, policy_ranges, scale_marker)
                size2 = scale_marker_size(p2_val, policy2, policy_ranges, scale_marker)

                # --- Top Panel: Net Cost vs Emissions
                ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                ax_top.scatter(e, c, s=scale_marker, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle = "--", alpha = 0.5)
                ax_top.scatter(e, c, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                ax_top.scatter(e, c, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)
                
                # --- Plot in inset axes if within zoom range
                if insets and e_min <= e <= e_max and c_min <= c <= c_max :
                    axins.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                    axins.scatter(e, c, s=scale_marker*scale_small, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)
                    axins.scatter(e, c, s=size1*scale_small, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                    axins.scatter(e, c, s=size2*scale_small, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)

                # --- Plot in inset axes if within zoom range
                if insets and e_min2 <= e <= e_max2 and u_min <= u <= u_max :
                    axins2.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                    axins2.scatter(e, u, s=scale_marker*scale_small, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)
                    axins2.scatter(e, u, s=size1*scale_small, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                    axins2.scatter(e, u, s=size2*scale_small, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)

                # --- Bottom Panel: Utility vs Emissions
                ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                ax_bottom.scatter(e, u, s=scale_marker, marker=full_circle_marker(), facecolor='none', edgecolor='black', linewidth=1, linestyle = "--", alpha = 0.5)
                ax_bottom.scatter(e, u, s=size1, marker=half_circle_marker(0, 180), color=color1, edgecolor="black", zorder=2)
                ax_bottom.scatter(e, u, s=size2, marker=half_circle_marker(180, 360), color=color2, edgecolor="black", zorder=2)

    # --- Plot Single Policy Outcomes
    for policy, entry in single_outcomes.items():
        print(policy,entry["mean_EV_uptake"])
        ev = entry["mean_EV_uptake"]
        if (min_ev_uptake <= ev <= max_ev_uptake):
            e = entry["mean_emissions_cumulative"] * 1e-9
            u = entry["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
            c = entry["mean_net_cost"] * 1e-9


            entry["emissions_cumulative"] = entry["emissions_cumulative_driving"] + entry["emissions_cumulative_production"]
            emissions_array = np.array(entry["emissions_cumulative"]) * 1e-9
            utility_array = np.array(entry["utility_cumulative"]) / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
            cost_array = np.array(entry["net_cost"]) * 1e-9

            n_seeds = len(emissions_array)
            e_err = 1.96 * np.std(emissions_array) / np.sqrt(n_seeds)
            u_err = 1.96 * np.std(utility_array) / np.sqrt(n_seeds)
            c_err = 1.96 * np.std(cost_array) / np.sqrt(n_seeds)

            color = policy_colors.get(policy, 'gray')
            size = scale_marker

            # --- Top Panel: Net Cost vs Emissions
            ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
            ax_top.scatter(e, c, s=size, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)
            
            # --- Plot in inset axes if within zoom range
            if insets and e_min <= e <= e_max and c_min <= c <= c_max :
                axins.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                axins.scatter(e, c, s=size*scale_small, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)

            # --- Plot in inset axes if within zoom range
            if insets and e_min2 <= e <= e_max2 and u_min <= u <= u_max :
                axins2.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
                axins2.scatter(e, u, s=size*scale_small, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)

            # --- Bottom Panel: Utility vs Emissions
            ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
            ax_bottom.scatter(e, u, s=size, marker=full_circle_marker(), color=color, edgecolor="black", zorder=2)

    # --- Configure zoom area


    #axins.set_title('Zoom', fontsize=10)
    
    # Draw box in main plot showing zoom area
    if insets:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax_top, axins, loc1=3, loc2=4, fc="none", ec="0.5")
        mark_inset(ax_bottom, axins2, loc1=2, loc2=3, fc="none", ec="0.5")


    # --- Labels
    ax_top.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
    ax_bottom.set_ylabel("Cumulative Utility, bn $", fontsize=16)
    ax_bottom.set_xlabel("Cumulative Emissions, MTCO2", fontsize=16)

    policy_titles = {
        "Carbon_price": "Carbon Price",
        "Electricity_subsidy": "Electricity Subsidy",
        "Adoption_subsidy": "New Car Rebate",
        "Adoption_subsidy_used": "Used Car Rebate",
        "Production_subsidy": "Production Subsidy"
    }
    # --- Legend
    legend_elements = [Patch(facecolor=policy_colors[policy], edgecolor='black',
                 label=f"{policy_titles[policy]} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
                       for policy in all_policies]

    # Custom intensity markers
    legend_elements += [Patch(facecolor='black', edgecolor='black', label='BAU')]

    low_proxy = plt.Line2D([0], [0], marker=half_circle_marker(0, 180),
                           color='gray', markerfacecolor='gray', markeredgecolor='black',
                           linestyle='None', label='Low Intensity', markersize=8)

    high_proxy = plt.Line2D([0], [0], marker=half_circle_marker(0, 180),
                            color='gray', markerfacecolor='gray', markeredgecolor='black',
                            linestyle='None', label='High Intensity', markersize=12)

    confidence = plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval')
    legend_elements += [confidence, low_proxy, high_proxy]

    ax_bottom.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # --- Save
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{file_name}/Plots/emissions_tradeoffs/{plot_name}.png", dpi=dpi)


POLICY_TITLES = {
    "Carbon_price": "Carbon Price",
    "Electricity_subsidy": "Electricity Subsidy",
    "Adoption_subsidy": "New Car Rebate",
    "Adoption_subsidy_used": "Used Car Rebate",
    "Production_subsidy": "Production Subsidy"
}


def _collect_records(base_params, pairwise_outcomes_complied, single_outcomes,
                     min_ev_uptake, max_ev_uptake):
    """Collect every plotted point once, plus the per-policy intensity ranges."""
    prob_switch = base_params["parameters_social_network"]["prob_switch_car"]
    records = []
    policy_ranges = {}

    def _bump(policy, value):
        r = policy_ranges.setdefault(policy, {"min": 0, "max": 0})
        r["max"] = max(r["max"], value)

    for (policy1, policy2), results in pairwise_outcomes_complied.items():
        policy_ranges.setdefault(policy1, {"min": 0, "max": 0})
        policy_ranges.setdefault(policy2, {"min": 0, "max": 0})
        for entry in results:
            if not (min_ev_uptake <= entry["mean_ev_uptake"] <= max_ev_uptake):
                continue

            entry["emissions_cumulative"] = entry["emissions_cumulative_driving"] + entry["emissions_cumulative_production"]
            emissions_array = np.array(entry["emissions_cumulative"]) * 1e-9
            utility_array = np.array(entry["utility_cumulative"]) / prob_switch * 1e-9
            cost_array = np.array(entry["net_cost"]) * 1e-9
            n_seeds = len(emissions_array)

            _bump(policy1, entry["policy1_value"])
            _bump(policy2, entry["policy2_value"])

            records.append(dict(
                kind="pair",
                e=entry["mean_emissions_cumulative"] * 1e-9,
                u=entry["mean_utility_cumulative"] / prob_switch * 1e-9,
                c=entry["mean_net_cost"] * 1e-9,
                e_err=1.96 * np.std(emissions_array) / np.sqrt(n_seeds),
                u_err=1.96 * np.std(utility_array) / np.sqrt(n_seeds),
                c_err=1.96 * np.std(cost_array) / np.sqrt(n_seeds),
                policy1=policy1, policy2=policy2,
                p1_val=entry["policy1_value"], p2_val=entry["policy2_value"],
            ))

    for policy, entry in single_outcomes.items():
        if not (min_ev_uptake <= entry["mean_EV_uptake"] <= max_ev_uptake):
            continue

        entry["emissions_cumulative"] = entry["emissions_cumulative_driving"] + entry["emissions_cumulative_production"]
        emissions_array = np.array(entry["emissions_cumulative"]) * 1e-9
        utility_array = np.array(entry["utility_cumulative"]) / prob_switch * 1e-9
        cost_array = np.array(entry["net_cost"]) * 1e-9
        n_seeds = len(emissions_array)

        records.append(dict(
            kind="single",
            e=entry["mean_emissions_cumulative"] * 1e-9,
            u=entry["mean_utility_cumulative"] / prob_switch * 1e-9,
            c=entry["mean_net_cost"] * 1e-9,
            e_err=1.96 * np.std(emissions_array) / np.sqrt(n_seeds),
            u_err=1.96 * np.std(utility_array) / np.sqrt(n_seeds),
            c_err=1.96 * np.std(cost_array) / np.sqrt(n_seeds),
            policy1=policy, policy2=policy,
            p1_val=None, p2_val=None,
        ))

    return records, policy_ranges


def _draw_record(ax, rec, y_key, policy_colors, policy_ranges, scale_marker, scale=1.0,
                 x_key="e"):
    """Draw one record on ax, with x_key on x and y_key ('c' or 'u') on y."""
    x, y = rec[x_key], rec[y_key]
    y_err = rec[f"{y_key}_err"]

    ax.errorbar(x, y, xerr=rec[f"{x_key}_err"], yerr=y_err, fmt='none',
                ecolor='gray', alpha=0.5, zorder=1)

    if rec["kind"] == "single":
        color = policy_colors.get(rec["policy1"], 'gray')
        ax.scatter(x, y, s=scale_marker * scale, marker=full_circle_marker(),
                   color=color, edgecolor="black", zorder=2)
        return

    size1 = scale_marker_size(rec["p1_val"], rec["policy1"], policy_ranges, scale_marker) * scale
    size2 = scale_marker_size(rec["p2_val"], rec["policy2"], policy_ranges, scale_marker) * scale
    ax.scatter(x, y, s=scale_marker * scale, marker=full_circle_marker(), facecolor='none',
               edgecolor='black', linewidth=1, linestyle="--", alpha=0.5)
    ax.scatter(x, y, s=size1, marker=half_circle_marker(0, 180),
               color=policy_colors[rec["policy1"]], edgecolor="black", zorder=2)
    ax.scatter(x, y, s=size2, marker=half_circle_marker(180, 360),
               color=policy_colors[rec["policy2"]], edgecolor="black", zorder=2)


def plot_emissions_tradeoffs_zoom(
        base_params,
        pairwise_outcomes_complied,
        single_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.9,
        max_ev_uptake=1.0,
        dpi=300,
        e_lims=(0.12, 0.17),
        c_lims=(-0.05, 0.3),
        u_lims=None,
        inset_policies=("Production_subsidy", "Adoption_subsidy"),
        inset_bbox_top=(0.04, 0.76, 0.20, 0.20),
        inset_bbox_bottom=(0.04, 0.78, 0.20, 0.18),
        plot_name="emissions_tradeoff_zoom"
        ):
    """
    Zoomed version of the trade-off figure. The main axes are cropped to
    e_lims / c_lims (utility left on autoscale unless u_lims is given), and each
    panel carries an inset that shows the inset_policies pair points which fall
    outside that crop. The inset is a callout of off-scale data, not a
    magnification, so it keeps its own tick labels.
    """
    okabe_ito_colors = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7', '#000000']
    color_map = ListedColormap(okabe_ito_colors)
    all_policies = sorted({p for pair in pairwise_outcomes_complied for p in pair})
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}

    scale_marker = 350
    scale_inset = 0.35

    records, policy_ranges = _collect_records(
        base_params, pairwise_outcomes_complied, single_outcomes,
        min_ev_uptake, max_ev_uptake)
    for policy in all_policies:
        policy_ranges.setdefault(policy, {"min": 0, "max": 0})

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # --- Main panels
    for rec in records:
        _draw_record(ax_top, rec, "c", policy_colors, policy_ranges, scale_marker)
        _draw_record(ax_bottom, rec, "u", policy_colors, policy_ranges, scale_marker)

    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / base_params["parameters_social_network"]["prob_switch_car"] * 1e-9
    bau_cost = outcomes_BAU["mean_net_cost"] * 1e-9
    ax_top.scatter(bau_em, bau_cost, s=scale_marker, color='black', edgecolor='black', label="BAU")
    ax_bottom.scatter(bau_em, bau_ut, s=scale_marker, color='black', edgecolor='black')

    ax_top.set_xlim(*e_lims)
    ax_top.set_ylim(*c_lims)
    ax_bottom.set_xlim(*e_lims)
    if u_lims is not None:
        ax_bottom.set_ylim(*u_lims)

    # --- Off-scale callout insets
    inset_set = set(inset_policies)
    inset_recs = [r for r in records
                  if r["kind"] == "pair" and {r["policy1"], r["policy2"]} == inset_set]

    def _out_of_window(rec, y_key, y_lims):
        inside_x = e_lims[0] <= rec["e"] <= e_lims[1]
        inside_y = True if y_lims is None else (y_lims[0] <= rec[y_key] <= y_lims[1])
        return not (inside_x and inside_y)

    def _add_inset(ax, y_key, y_lims, bbox):
        selected = [r for r in inset_recs if _out_of_window(r, y_key, y_lims)]
        if not selected:
            return None
        axins = ax.inset_axes(bbox)
        for rec in selected:
            _draw_record(axins, rec, y_key, policy_colors, policy_ranges,
                         scale_marker, scale=scale_inset)

        # Limits cover the means and their confidence intervals, so the CI bars
        # stay inside the box instead of reading as crosshairs.
        xs = [r["e"] for r in selected]
        ys = [r[y_key] for r in selected]
        y_errs = [r["c_err"] if y_key == "c" else r["u_err"] for r in selected]
        x_lo = min(r["e"] - r["e_err"] for r in selected)
        x_hi = max(r["e"] + r["e_err"] for r in selected)
        y_lo = min(y - err for y, err in zip(ys, y_errs))
        y_hi = max(y + err for y, err in zip(ys, y_errs))
        x_pad = max(0.15 * (x_hi - x_lo), 0.002)
        y_pad = max(0.15 * (y_hi - y_lo), 0.01 if y_key == "c" else 0.2)
        axins.set_xlim(x_lo - x_pad, x_hi + x_pad)
        axins.set_ylim(y_lo - y_pad, y_hi + y_pad)
        # Two ticks at the data extremes rather than at the means: in a box this
        # small, means sit too close together and their labels overlap.
        axins.set_xticks([round(x_lo, 3), round(x_hi, 3)])
        axins.set_yticks([round(y_lo, 2), round(y_hi, 2)] if y_key == "c"
                         else [round(y_lo, 1), round(y_hi, 1)])
        axins.tick_params(axis='both', labelsize=6, pad=1.5, length=2)
        for spine in axins.spines.values():
            spine.set_edgecolor("0.4")
            spine.set_linestyle((0, (4, 2)))
        return axins

    _add_inset(ax_top, "c", c_lims, inset_bbox_top)
    _add_inset(ax_bottom, "u", u_lims, inset_bbox_bottom)

    # --- Labels
    ax_top.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
    ax_bottom.set_ylabel("Cumulative Utility, bn $", fontsize=16)
    ax_bottom.set_xlabel("Cumulative Emissions, MTCO2", fontsize=16)

    # --- Legend
    legend_elements = [Patch(facecolor=policy_colors[policy], edgecolor='black',
                             label=f"{POLICY_TITLES[policy]} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
                       for policy in all_policies]
    legend_elements += [Patch(facecolor='black', edgecolor='black', label='BAU')]
    legend_elements += [
        plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval'),
        plt.Line2D([0], [0], marker=half_circle_marker(0, 180), color='gray',
                   markerfacecolor='gray', markeredgecolor='black', linestyle='None',
                   label='Low Intensity', markersize=8),
        plt.Line2D([0], [0], marker=half_circle_marker(0, 180), color='gray',
                   markerfacecolor='gray', markeredgecolor='black', linestyle='None',
                   label='High Intensity', markersize=12),
    ]
    ax_bottom.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # --- Save
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{file_name}/Plots/emissions_tradeoffs/{plot_name}.png", dpi=dpi)


def plot_emissions_tradeoffs_bau_inset(
        base_params,
        pairwise_outcomes_complied,
        single_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.9,
        max_ev_uptake=1.0,
        dpi=300,
        pad_frac=0.10,
        headroom_frac=0.35,
        inset_bbox_top=(0.70, 0.68, 0.28, 0.29),
        inset_bbox_bottom=(0.70, 0.68, 0.28, 0.29),
        plot_name="emissions_tradeoff_bau_inset"
        ):
    """
    All policy combinations at full scale, with BAU relegated to an inset.

    BAU has far higher emissions than any policy point, so keeping it on the
    main axes squashes the whole policy cloud into the left edge. Here the main
    axes are scaled to the policy points only, and each panel carries a small
    overview inset holding the full range: the policy cloud as grey dots inside
    a dashed box marking the main window, and BAU as the black marker off to
    the right.
    """
    okabe_ito_colors = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7', '#000000']
    color_map = ListedColormap(okabe_ito_colors)
    all_policies = sorted({p for pair in pairwise_outcomes_complied for p in pair})
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}

    scale_marker = 350

    records, policy_ranges = _collect_records(
        base_params, pairwise_outcomes_complied, single_outcomes,
        min_ev_uptake, max_ev_uptake)
    for policy in all_policies:
        policy_ranges.setdefault(policy, {"min": 0, "max": 0})

    prob_switch = base_params["parameters_social_network"]["prob_switch_car"]
    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / prob_switch * 1e-9
    bau_cost = outcomes_BAU["mean_net_cost"] * 1e-9

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for rec in records:
        _draw_record(ax_top, rec, "c", policy_colors, policy_ranges, scale_marker)
        _draw_record(ax_bottom, rec, "u", policy_colors, policy_ranges, scale_marker)

    # --- Main window: scaled to the means, not to the confidence intervals. The
    # emissions CIs are wide enough that including them would push every marker
    # into the middle third of the axes; the bars are allowed to run off instead.
    # headroom_frac adds empty space at the top for the inset to sit in.
    def _limits(values, headroom=True):
        lo, hi = min(values), max(values)
        span = hi - lo if hi > lo else abs(hi) or 1.0
        return lo - pad_frac * span, hi + (headroom_frac if headroom else pad_frac) * span

    e_lims = _limits([r["e"] for r in records], headroom=False)
    c_lims = _limits([r["c"] for r in records])
    u_lims = _limits([r["u"] for r in records])

    ax_top.set_xlim(*e_lims)
    ax_top.set_ylim(*c_lims)
    ax_bottom.set_xlim(*e_lims)
    ax_bottom.set_ylim(*u_lims)

    # --- Overview insets carrying BAU
    def _add_bau_inset(ax, y_key, y_lims, bau_y, bbox):
        axins = ax.inset_axes(bbox)

        xs = [r["e"] for r in records]
        ys = [r[y_key] for r in records]
        axins.scatter(xs, ys, s=8, color="0.55", edgecolor="none", zorder=2)
        axins.scatter(bau_em, bau_y, s=70, color="black", edgecolor="black", zorder=3)
        axins.annotate("BAU", (bau_em, bau_y), textcoords="offset points",
                       xytext=(0, -12), ha="center", va="top", fontsize=7)

        # Dashed box = the window shown on the main axes.
        axins.add_patch(plt.Rectangle(
            (e_lims[0], y_lims[0]), e_lims[1] - e_lims[0], y_lims[1] - y_lims[0],
            fill=False, edgecolor="0.35", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1))

        x_lo = min(min(xs), bau_em)
        x_hi = max(max(xs), bau_em)
        y_lo = min(min(ys), bau_y, y_lims[0])
        y_hi = max(max(ys), bau_y, y_lims[1])
        x_pad = 0.12 * (x_hi - x_lo)
        y_pad = 0.16 * (y_hi - y_lo)
        axins.set_xlim(x_lo - x_pad, x_hi + x_pad)
        axins.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # Few, auto-placed ticks: hand-picked ticks at the data extremes land on
        # the callout box corners and collide with it in a box this small.
        axins.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        axins.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        axins.tick_params(axis='both', labelsize=6, pad=1.5, length=2)
        axins.set_facecolor("white")
        for spine in axins.spines.values():
            spine.set_edgecolor("0.4")
        return axins

    _add_bau_inset(ax_top, "c", c_lims, bau_cost, inset_bbox_top)
    _add_bau_inset(ax_bottom, "u", u_lims, bau_ut, inset_bbox_bottom)

    # --- Labels
    ax_top.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
    ax_bottom.set_ylabel("Cumulative Utility, bn $", fontsize=16)
    ax_bottom.set_xlabel("Cumulative Emissions, MTCO2", fontsize=16)

    # --- Legend
    legend_elements = [Patch(facecolor=policy_colors[policy], edgecolor='black',
                             label=f"{POLICY_TITLES[policy]} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
                       for policy in all_policies]
    legend_elements += [Patch(facecolor='black', edgecolor='black', label='BAU (inset)')]
    legend_elements += [
        plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval'),
        plt.Line2D([0], [0], marker=half_circle_marker(0, 180), color='gray',
                   markerfacecolor='gray', markeredgecolor='black', linestyle='None',
                   label='Low Intensity', markersize=8),
        plt.Line2D([0], [0], marker=half_circle_marker(0, 180), color='gray',
                   markerfacecolor='gray', markeredgecolor='black', linestyle='None',
                   label='High Intensity', markersize=12),
    ]
    # Below the panels rather than inside one: with every pair drawn, an in-axes
    # legend sits on top of the low-utility cluster.
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    # --- Save
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(f"{file_name}/Plots/emissions_tradeoffs/{plot_name}.png", dpi=dpi)


def plot_emissions_tradeoffs_reference_lines(
        base_params,
        pairwise_outcomes_complied,
        single_outcomes,
        outcomes_BAU,
        file_name,
        min_ev_uptake=0.9,
        max_ev_uptake=1.0,
        dpi=300,
        utility_as_pct_of_bau=False,
        emissions_as_pct_of_bau=False,
        show_zero_net_cost_line=True,
        pad_frac=0.10,
        plot_name="emissions_tradeoff_reference_lines"
        ):
    """
    Policy points at full scale with the benchmarks drawn as dashed lines
    rather than as a point: zero net cost on the top panel and BAU utility on
    the bottom one.

    BAU sits well to the right in emissions, so its marker cannot share the axes
    with the policy cloud without squashing it (see
    plot_emissions_tradeoffs_bau_inset). A horizontal line carries the part of
    BAU that matters for the comparison and costs no x-range.

    utility_as_pct_of_bau rescales the bottom panel to utility as a percentage
    of BAU utility; the benchmark line is dropped there, since the axis itself
    already carries BAU.
    emissions_as_pct_of_bau does the same to the shared x axis. No 100% line is
    drawn there: the axis stays zoomed on the policy points, and BAU emissions
    are read off the axis scale instead.
    show_zero_net_cost_line=False drops the zero net cost line from the top
    panel; the top axis is then scaled to the policy points alone.
    """
    okabe_ito_colors = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7', '#000000']
    color_map = ListedColormap(okabe_ito_colors)
    all_policies = sorted({p for pair in pairwise_outcomes_complied for p in pair})
    policy_colors = {policy: color_map(i) for i, policy in enumerate(all_policies)}

    scale_marker = 350

    records, policy_ranges = _collect_records(
        base_params, pairwise_outcomes_complied, single_outcomes,
        min_ev_uptake, max_ev_uptake)
    for policy in all_policies:
        policy_ranges.setdefault(policy, {"min": 0, "max": 0})

    prob_switch = base_params["parameters_social_network"]["prob_switch_car"]
    bau_ut = outcomes_BAU["mean_utility_cumulative"] / prob_switch * 1e-9

    if utility_as_pct_of_bau:
        for rec in records:
            rec["u_pct"] = rec["u"] / bau_ut * 100
            rec["u_pct_err"] = rec["u_err"] / bau_ut * 100
        y_bottom = "u_pct"
        # No benchmark line: the axis is already in units of BAU, so a line at
        # 100% only repeats what the scale says.
        bottom_ref = None
        bottom_label = "Cumulative Utility, % of BAU"
        bottom_ref_label = None
    else:
        y_bottom = "u"
        bottom_ref = bau_ut
        bottom_label = "Cumulative Utility, bn $"
        bottom_ref_label = "BAU utility"

    bau_em = outcomes_BAU["mean_emissions_cumulative"] * 1e-9
    if emissions_as_pct_of_bau:
        for rec in records:
            rec["e_pct"] = rec["e"] / bau_em * 100
            rec["e_pct_err"] = rec["e_err"] / bau_em * 100
        x_key = "e_pct"
        x_label = "Cumulative Emissions, % of BAU"
    else:
        x_key = "e"
        x_label = "Cumulative Emissions, MTCO2"

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for rec in records:
        _draw_record(ax_top, rec, "c", policy_colors, policy_ranges, scale_marker, x_key=x_key)
        _draw_record(ax_bottom, rec, y_bottom, policy_colors, policy_ranges, scale_marker, x_key=x_key)

    # --- Reference lines, labelled in-panel: the two lines mean different things
    # per panel, so one shared legend entry could not name both.
    ref_kwargs = dict(color="0.35", linestyle=(0, (5, 3)), linewidth=1.4, zorder=0)
    top_ref = 0.0 if show_zero_net_cost_line else None
    any_ref = top_ref is not None or bottom_ref is not None
    for ax, ref, label in ((ax_top, top_ref, "Zero net cost"),
                           (ax_bottom, bottom_ref, bottom_ref_label)):
        if ref is None:
            continue
        ax.axhline(ref, **ref_kwargs)
        ax.annotate(label, xy=(0.995, ref), xycoords=ax.get_yaxis_transform(),
                    xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=11, color="0.25")

    # --- Limits: scaled to the means, with the reference line kept in view even
    # if every policy point sits on one side of it. Confidence bars are allowed
    # to run off the axes, as in the other full-scale figure.
    def _limits(values, reference):
        refs = [] if reference is None else [reference]
        lo, hi = min(values + refs), max(values + refs)
        span = hi - lo if hi > lo else abs(hi) or 1.0
        return lo - pad_frac * span, hi + pad_frac * span

    # x is always scaled to the policy points alone: BAU emissions sit far to
    # the right (100% when rescaled), and reserving room for them would leave
    # most of the axis empty.
    e_vals = [r[x_key] for r in records]
    e_lo, e_hi = min(e_vals), max(e_vals)
    e_span = e_hi - e_lo or abs(e_hi) or 1.0
    ax_top.set_xlim(e_lo - pad_frac * e_span, e_hi + pad_frac * e_span)
    ax_top.set_ylim(*_limits([r["c"] for r in records], top_ref))
    ax_bottom.set_ylim(*_limits([r[y_bottom] for r in records], bottom_ref))

    # --- Labels
    ax_top.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
    ax_bottom.set_ylabel(bottom_label, fontsize=16)
    ax_bottom.set_xlabel(x_label, fontsize=16)

    # --- Legend
    legend_elements = [Patch(facecolor=policy_colors[policy], edgecolor='black',
                             label=f"{POLICY_TITLES[policy]} ({policy_ranges[policy]['min']:.2f} - {policy_ranges[policy]['max']:.2f})")
                       for policy in all_policies]
    if any_ref:
        legend_elements += [
            plt.Line2D([0], [0], color="0.35", linestyle=(0, (5, 3)), linewidth=1.4,
                       label="Benchmark"),
        ]
    legend_elements += [
        plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval'),
        plt.Line2D([0], [0], marker=half_circle_marker(0, 180), color='gray',
                   markerfacecolor='gray', markeredgecolor='black', linestyle='None',
                   label='Low Intensity', markersize=8),
        plt.Line2D([0], [0], marker=half_circle_marker(0, 180), color='gray',
                   markerfacecolor='gray', markeredgecolor='black', linestyle='None',
                   label='High Intensity', markersize=12),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    # --- Save
    os.makedirs(f"{file_name}/Plots/emissions_tradeoffs", exist_ok=True)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(f"{file_name}/Plots/emissions_tradeoffs/{plot_name}.png", dpi=dpi)


def main(fileNames):
    """
    fileNames : list of endog_pair folders. The FIRST one must be a folder made
        by the current endogenous_policy_intensity_pair_gen, so it also holds
        base_params, outcomes_BAU and single_policy_outcomes. Extra folders
        contribute their pairwise_outcomes only.
    """
    fileName = fileNames[0]

    base_params = load_object(f"{fileName}/Data", "base_params")

    #BAU AND ENDOGENOUS SINGLE POLICY, from the same run as the pairs
    outcomes_BAU = load_object(f"{fileName}/Data", "outcomes_BAU")
    single_policy_outcomes = load_object(f"{fileName}/Data", "single_policy_outcomes")


    # Plots (and the compiled pairwise data) go into the first run folder rather
    # than a separate all_policies folder.
    file_name = fileName

    pairwise_outcomes_complied = {}
    
    for folder in fileNames:
        pairwise_outcomes = load_object(f"{folder}/Data", "pairwise_outcomes")
        pairwise_outcomes_complied.update(pairwise_outcomes)

    #pairwise_outcomes_complied = {k: v for k, v in pairwise_outcomes_complied.items() if set(k) == {"Electricity_subsidy", "Adoption_subsidy"}}


    min_ev_uptake = 0.94
    max_ev_uptake = 0.96

    # With zoom insets
    plot_emissions_tradeoffs_from_outcomes(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                            file_name,
                                            min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                            insets=True, plot_name="emissions_tradeoff")

    # Same figure, no zoom insets
    plot_emissions_tradeoffs_from_outcomes(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                            file_name,
                                            min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                            insets=False, plot_name="emissions_tradeoff_no_inset")

    # Zoomed figure with an off-scale callout inset for Production Subsidy + New Car Rebate
    plot_emissions_tradeoffs_zoom(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                  file_name,
                                  min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                  e_lims=(0.12, 0.17), c_lims=(-0.05, 0.3), u_lims=None,
                                  inset_policies=("Production_subsidy", "Adoption_subsidy"),
                                  plot_name="emissions_tradeoff_zoom")

    # All policy combinations at full scale, BAU moved into an overview inset
    plot_emissions_tradeoffs_bau_inset(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                       file_name,
                                       min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                       plot_name="emissions_tradeoff_bau_inset")

    # Benchmarks as dashed lines instead of the off-scale BAU marker
    plot_emissions_tradeoffs_reference_lines(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                             file_name,
                                             min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                             utility_as_pct_of_bau=False,
                                             plot_name="emissions_tradeoff_reference_lines")

    # Same, with utility rescaled to a percentage of BAU utility
    plot_emissions_tradeoffs_reference_lines(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                             file_name,
                                             min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                             utility_as_pct_of_bau=True,
                                             plot_name="emissions_tradeoff_utility_pct_bau")

    # Both axes as a percentage of BAU
    plot_emissions_tradeoffs_reference_lines(base_params, pairwise_outcomes_complied, single_policy_outcomes, outcomes_BAU,
                                             file_name,
                                             min_ev_uptake=min_ev_uptake, max_ev_uptake=max_ev_uptake, dpi=300,
                                             utility_as_pct_of_bau=True, emissions_as_pct_of_bau=True,
                                             show_zero_net_cost_line=False,
                                             plot_name="emissions_tradeoff_pct_bau")

    # Saved under its own name so the folder's own pairwise_outcomes (and the
    # BAU/single-policy/params objects already sitting there) stay untouched.
    save_object(pairwise_outcomes_complied, file_name + "/Data", "pairwise_outcomes_complied")

    plt.show()


if __name__ == "__main__":
    # The gen script calls main() itself, so the default below is only the folder
    # last used interactively; pass one or more folders to plot other runs.
    fileNames = sys.argv[1:] or ["results/endog_pair_14_08_10__19_08_2026"]
    print("Plotting:", fileNames)
    main(fileNames=fileNames)