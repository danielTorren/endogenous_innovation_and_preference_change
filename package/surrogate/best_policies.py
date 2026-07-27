"""
best_policies.py — Run the surrogate's cheapest feasible policies through the real ABM.

run.py only ever queries the GP surrogate — it never confirms a result
against the actual model. This module closes that loop:

  1. Loads the ranked feasible policies saved by run.py (results_dir/Data/pareto.npz)
     — sorted by ascending net_cost, since run.py now does a constrained
     single-objective search (minimise cost subject to emissions/utility
     bounds), not a multi-objective Pareto front.
  2. Takes the best N (cheapest feasible) policies plus a BAU baseline, and
     runs each through the real ABM — full time series, N_seeds future-period
     runs each (calibration is NOT re-run), in the same way as
     package/analysis/low_policy_intensity_gen.py.
  3. Produces a combined dashboard (EV share, EV price, emissions, utility,
     car age, net cost), in the same style as
     package/analysis/low_policy_intensity_plot.py.
  4. Produces a trade-off scatter plot (net cost / utility vs emissions)
     using the same "split ball" marker style as
     package/analysis/endogenous_policy_intensity_pair_plot.py — but each
     marker is now split into 5 wedges (one per policy instrument, sized by
     that policy's intensity relative to its search bounds) instead of the
     2-policy half-circles, since the surrogate optimises all 5 policies
     jointly rather than in pairs.

`run_final_abm()` (previously in run.py) lives here now, since this is where
it's actually used.

Usage (from repo root, after running `python -m package.surrogate.run`):
    python -m package.surrogate.best_policies
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.stats import sem, t

from package.resources.utility import (
    load_object, save_object, createFolder, produce_name_datetime,
)
from package.analysis.endogenous_policy_intensity_single_gen import update_policy_intensity
from package.analysis.low_policy_intensity_gen import single_policy_with_seeds
from package.analysis.endogenous_policy_intensity_pair_plot import (
    full_circle_marker, half_circle_marker, scale_marker_size,
)

from .sampling import PolicyBounds, load_policy_bounds
from .run import (
    get_or_create_calibration, _resolve_calib_folder,
    BASE_PARAMS_PATH, BOUNDS_PATH, RESULTS_DIR,
)

N_BEST = 10

POLICY_TITLES = {
    "Carbon_price": "Carbon Price",
    "Electricity_subsidy": "Electricity Subsidy",
    "Adoption_subsidy": "New Car Rebate",
    "Adoption_subsidy_used": "Used Car Rebate",
    "Production_subsidy": "Production Subsidy",
}

OKABE_ITO_COLORS = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
                    '#0072B2', '#D55E00', '#CC79A7', '#000000']

RANK_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']


# ---------------------------------------------------------------------------
# Full ABM run for a chosen policy (moved from run.py)
# ---------------------------------------------------------------------------

def run_final_abm(
    policy_vector: np.ndarray,
    base_params: dict,
    controller_files: list,
    bounds: PolicyBounds = None,
    results_dir: str = RESULTS_DIR,
    save: bool = True,
    tag: str = "optimal_policy",
) -> dict:
    """
    Run the full ABM (with time-series output) for a chosen policy vector.

    WHAT THIS RUNS:
      N_seeds future-period runs (Phase 2 only — calibration NOT re-run).
      Returns a dict matching the format used by low_policy_intensity_plot.py.

    policy_vector : one row from X_ranked, or all-zeros for the BAU baseline
    controller_files : from get_or_create_calibration() — reuse the same ones
    bounds : PolicyBounds used during optimisation; defaults to loading from JSON
    save : whether to pickle the output/policy_dict to results_dir/Data
    tag : filename prefix used when save=True — pass a distinct tag per call
          when running several policies so files don't overwrite each other
    """
    if bounds is None:
        bounds = load_policy_bounds(BOUNDS_PATH)
    policy_dict = dict(zip(bounds.names, policy_vector))
    print(f"\nRunning final ABM ({len(controller_files)} seeds, future period only):")
    active = {k: v for k, v in policy_dict.items() if v > 0}
    if active:
        for k, v in active.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("  (BAU — no policy active)")

    params = deepcopy(base_params)
    # The surrogate's base_params has this off (0) for speed during LHS/BO,
    # which only reads final scalar values — single_policy_with_seeds() needs
    # the full history_* time series, so it must be turned on here.
    params["save_timeseries_data_state"] = 1
    for key in params["parameters_policies"]["States"]:
        params["parameters_policies"]["States"][key] = 0
    for name, intensity in policy_dict.items():
        if intensity > 0:
            params = update_policy_intensity(params, name, intensity)

    results = single_policy_with_seeds(params, controller_files)

    output = {
        "history_driving_emissions":              results[0],
        "history_production_emissions":           results[1],
        "history_total_emissions":                results[2],
        "history_prop_EV":                        results[3],
        "history_total_utility":                  results[9],
        "history_mean_price_ICE_EV_arr":          results[7],
        "history_policy_net_cost":                results[22],
        "history_mean_car_age":                   results[20],
        "history_past_new_bought_vehicles_prop_ev": results[21],
    }

    if save:
        save_object(output,      f"{results_dir}/Data", f"{tag}_output")
        save_object(policy_dict, f"{results_dir}/Data", f"{tag}_dict")
        print(f"Saved to {results_dir}/Data/{tag}_*.pkl")
    return output


# ---------------------------------------------------------------------------
# Select and run the top (cheapest feasible) policies
# ---------------------------------------------------------------------------

def select_top_policies(X_ranked: np.ndarray, Y_ranked: np.ndarray, n_best: int = N_BEST) -> tuple:
    """
    Top n_best cheapest feasible policies (ascending net_cost, Y column 3).
    run.py already saves pareto.npz pre-sorted this way, so this is mostly
    just a top-N slice — the explicit sort here is defensive, in case this
    is ever called on unsorted data from elsewhere.
    """
    n_best = min(n_best, len(X_ranked))
    if n_best < N_BEST:
        print(f"Only {n_best} feasible point(s) available (requested {N_BEST}).")
    order = np.argsort(Y_ranked[:, 3])
    idx = order[:n_best]
    return X_ranked[idx], Y_ranked[idx]


def run_top_policies(
    n_best: int = N_BEST,
    results_dir: str = RESULTS_DIR,
    bounds_path: str = BOUNDS_PATH,
    base_params_path: str = BASE_PARAMS_PATH,
    existing_calib_folder: str = None,
) -> tuple:
    """
    Load the ranked feasible policies saved by run.py, run the BAU baseline
    and the top n_best (cheapest feasible) policies through the real ABM, and
    save the aggregated results — same pattern as low_policy_intensity_gen.py.

    Returns: (out_folder, base_params, outputs, outputs_BAU, policy_dicts, Y_top, bounds)
    """
    ranked_path = f"{results_dir}/Data/pareto.npz"
    data = np.load(ranked_path)
    X_ranked, Y_ranked = data["X"], data["Y"]
    if len(X_ranked) == 0:
        raise ValueError(f"No feasible policies found in {ranked_path} — run package.surrogate.run first.")

    bounds = load_policy_bounds(bounds_path)

    calib_folder = _resolve_calib_folder(results_dir, existing_calib_folder)
    controller_files, base_params, calib_folder = get_or_create_calibration(
        base_params_path, calib_folder
    )

    X_top, Y_top = select_top_policies(X_ranked, Y_ranked, n_best)
    print(f"Running BAU + top {len(X_top)} cheapest feasible policies through the real ABM "
          f"({len(controller_files)} seeds each)...")

    outputs_BAU = run_final_abm(
        np.zeros(bounds.n), base_params, controller_files, bounds=bounds, save=False,
    )

    outputs, policy_dicts = {}, {}
    for rank, x in enumerate(X_top):
        print(f"\n--- Policy {rank + 1}/{len(X_top)} ---")
        outputs[rank] = run_final_abm(
            x, base_params, controller_files, bounds=bounds, save=False,
        )
        policy_dicts[rank] = dict(zip(bounds.names, x))

    out_folder = produce_name_datetime("surrogate_top_policies")
    createFolder(out_folder)
    save_object(outputs,      out_folder + "/Data", "outputs")
    save_object(outputs_BAU,  out_folder + "/Data", "outputs_BAU")
    save_object(policy_dicts, out_folder + "/Data", "policy_dicts")
    save_object(Y_top,        out_folder + "/Data", "Y_top")
    save_object(base_params,  out_folder + "/Data", "base_params")
    print(f"\nAll {len(X_top)} top policies + BAU processed and saved in '{out_folder}'")

    return out_folder, base_params, outputs, outputs_BAU, policy_dicts, Y_top, bounds


# ---------------------------------------------------------------------------
# Dashboard plot — same structure as package/analysis/low_policy_intensity_plot.py,
# generalised from 1-2 active policies per key to N (rank-keyed, up to 5 active).
# ---------------------------------------------------------------------------

def _label_for_rank(rank: int, policy_dict: dict, y_row: np.ndarray) -> str:
    active = [f"{POLICY_TITLES.get(k, k)} ({v:.2g})" for k, v in policy_dict.items() if v > 0]
    tag = ", ".join(active) if active else "no active policy"
    return f"#{rank + 1} (cost {y_row[3]:.2g}, EV {y_row[0]:.0%}): {tag}"


def plot_top_policies_dashboard(
    base_params, fileName, outputs, outputs_BAU, policy_dicts, Y_top, dpi=300,
):
    time_steps = np.arange(base_params["duration_future"] - 1)
    start = base_params["duration_burn_in"] + base_params["duration_calibration"]

    ranks = sorted(outputs.keys())
    cmap = plt.get_cmap("tab10")
    rank_style = {r: (cmap(r % 10), RANK_MARKERS[r % len(RANK_MARKERS)]) for r in ranks}
    rank_label = {r: _label_for_rank(r, policy_dicts[r], Y_top[r]) for r in ranks}

    def plot_line_with_ci(ax, data, color, marker, linestyle='-', label=None):
        mean = np.nanmean(data, axis=0)
        ci = sem(data, axis=0, nan_policy='omit') * t.ppf(0.975, df=data.shape[0] - 1)
        ax.plot(time_steps, mean, color=color, marker=marker, markevery=32,
                 markerfacecolor=color, markeredgecolor=color, markersize=4,
                 linestyle=linestyle, label=label)
        ax.fill_between(time_steps, mean - ci, mean + ci, color=color, alpha=0.2)

    def _add_vline(ax, annotation_height_prop=(0.2, 0.2, 0.2)):
        y_min, y_max = ax.get_ylim()
        annotation_height_0 = y_min + annotation_height_prop[0] * (y_max - y_min)
        ax.axvline(143, color='black', linestyle=':')
        ax.annotate("Policy end", xy=(143, annotation_height_0),
                     rotation=90, verticalalignment='center', horizontalalignment='right',
                     fontsize=8, color='black')

    def panel_ev_share(ax, add_labels=True):
        plot_line_with_ci(ax, outputs_BAU["history_prop_EV"][:, start:], 'black', 'o', '-', 'BAU - EV Adoption')
        plot_line_with_ci(ax, outputs_BAU["history_past_new_bought_vehicles_prop_ev"], 'black', 'o', '--', 'BAU - EV Sales')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, outputs[r]["history_prop_EV"][:, start:], color, marker, '-', None)
            plot_line_with_ci(ax, outputs[r]["history_past_new_bought_vehicles_prop_ev"], color, marker, '--', lbl)
        ax.set_ylabel("EV Share", fontsize=16)
        _add_vline(ax, annotation_height_prop=(0.6, 0.2, 0.2))
        ax.legend(handles=[
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='EV Adoption'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='EV Sales'),
        ], loc='lower right', fontsize='small', ncols=2)

    def panel_ev_price(ax, add_labels=False):
        for i, label_txt, linestyle in [(0, 'New', '-'), (1, 'Used', '--')]:
            plot_line_with_ci(ax, outputs_BAU["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                                'black', 'o', linestyle, f"BAU - {label_txt}")
        for r in ranks:
            color, marker = rank_style[r]
            for i, linestyle in [(0, '-'), (1, '--')]:
                lbl = rank_label[r] if (add_labels and i == 0) else None
                plot_line_with_ci(ax, outputs[r]["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                                    color, marker, linestyle, lbl)
        ax.set_ylabel("EV Sale Price, $", fontsize=16)
        ax.legend(handles=[
            Line2D([0], [0], color="black", linestyle='-', label='New'),
            Line2D([0], [0], color="black", linestyle='--', label='Used'),
        ], loc='lower right', fontsize="small", ncols=2)
        _add_vline(ax, annotation_height_prop=(0.9, 0.2, 0.2))

    def panel_emissions(ax, cumulative=False, add_labels=True):
        transform = (lambda x: np.cumsum(x, axis=1) * 1e-9) if cumulative else (lambda x: x * 1e-9)
        ylabel = "Cumulative Emissions, MTCO2" if cumulative else "Flow Emissions, MTCO2"
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_emissions"]), 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, transform(outputs[r]["history_total_emissions"]), color, marker, '-', lbl)
        ax.set_ylabel(ylabel, fontsize=16)
        _add_vline(ax, annotation_height_prop=(0.5, 0.2, 0.2))

    def panel_utility(ax, cumulative=False, add_labels=False):
        # Raw utility flow/sum — not the log-utility metric the surrogate
        # actually optimises against (see sampling.compute_log_utility_metric).
        transform = (lambda x: np.cumsum(x, axis=1) * 1e-9) if cumulative else (lambda x: x * 1e-9)
        ylabel = "Cumulative Utility (raw), bn $" if cumulative else "Flow Utility (raw), bn $"
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_utility"]), 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, transform(outputs[r]["history_total_utility"]), color, marker, '-', lbl)
        ax.set_ylabel(ylabel, fontsize=16)
        _add_vline(ax, annotation_height_prop=(0.2, 0.2, 0.2))

    def panel_car_age(ax, add_labels=False):
        plot_line_with_ci(ax, outputs_BAU["history_mean_car_age"], 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, outputs[r]["history_mean_car_age"], color, marker, '-', lbl)
        ax.set_ylabel("Car Age, months", fontsize=16)
        _add_vline(ax, annotation_height_prop=(0.5, 0.2, 0.2))

    def panel_net_cost(ax, add_labels=True):
        plot_line_with_ci(ax, outputs_BAU["history_policy_net_cost"] * 1e-9, 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, outputs[r]["history_policy_net_cost"] * 1e-9, color, marker, '-', lbl)
        ax.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
        _add_vline(ax, annotation_height_prop=(0.5, 0.2, 0.2))

    start_year = 2024
    tick_years = np.arange(start_year, start_year + (time_steps[-1] // 12) + 5, 5)
    tick_positions = (tick_years - start_year) * 12

    def set_xaxis(ax):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(year) for year in tick_years])
        ax.set_xlabel("Year", fontsize=16)

    fig, axs = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    panel_ev_share(axs[0, 0], add_labels=True)
    panel_ev_price(axs[0, 1], add_labels=False)
    panel_emissions(axs[1, 0], cumulative=False, add_labels=True)
    panel_emissions(axs[1, 1], cumulative=True, add_labels=False)
    panel_utility(axs[2, 0], cumulative=False, add_labels=False)
    panel_utility(axs[2, 1], cumulative=True, add_labels=False)
    panel_car_age(axs[3, 0], add_labels=False)
    panel_net_cost(axs[3, 1], add_labels=False)

    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.00), fontsize=9)

    for ax in axs[3]:
        set_xaxis(ax)

    fig.tight_layout(rect=[0.01, 0.11, 0.98, 1])
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(f"{fileName}/Plots/top_policies_dashboard.png", dpi=dpi)

    individual_panels = [
        ("ev_share",             lambda ax: panel_ev_share(ax, add_labels=True)),
        ("ev_price",             lambda ax: panel_ev_price(ax, add_labels=True)),
        ("flow_emissions",       lambda ax: panel_emissions(ax, cumulative=False, add_labels=True)),
        ("cumulative_emissions", lambda ax: panel_emissions(ax, cumulative=True,  add_labels=True)),
        ("flow_utility",         lambda ax: panel_utility(ax,   cumulative=False, add_labels=True)),
        ("cumulative_utility",   lambda ax: panel_utility(ax,   cumulative=True,  add_labels=True)),
        ("car_age",              lambda ax: panel_car_age(ax,   add_labels=True)),
        ("net_cost",             lambda ax: panel_net_cost(ax,  add_labels=True)),
    ]
    for name, plot_fn in individual_panels:
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        plot_fn(ax_ind)
        set_xaxis(ax_ind)
        handles_ind, labels_ind = ax_ind.get_legend_handles_labels()
        if handles_ind:
            fig_ind.legend(handles_ind, labels_ind, loc='lower center', ncol=1,
                            bbox_to_anchor=(0.5, 0.00), fontsize=7)
            fig_ind.tight_layout(rect=[0.01, 0.28, 0.98, 1])
        else:
            fig_ind.tight_layout()
        fig_ind.savefig(f"{fileName}/Plots/individual_{name}.png", dpi=dpi)
        plt.close(fig_ind)


# ---------------------------------------------------------------------------
# Trade-off "split ball" plot — same style as
# package/analysis/endogenous_policy_intensity_pair_plot.py, split 5 ways.
# ---------------------------------------------------------------------------

def plot_top_policies_tradeoff(
    base_params, fileName, outputs, outputs_BAU, policy_dicts, bounds, dpi=300,
):
    policy_names = sorted(bounds.names)
    n_policies = len(policy_names)
    wedge_width = 360 / n_policies
    policy_ranges = {name: {"min": lo, "max": hi} for name, (lo, hi) in bounds.bounds.items()}

    color_map = ListedColormap(OKABE_ITO_COLORS)
    policy_colors = {p: color_map(i) for i, p in enumerate(policy_names)}

    scale_marker = 350

    def _point(output):
        emissions = np.cumsum(output["history_total_emissions"], axis=1)[:, -1] * 1e-9
        utility = np.cumsum(output["history_total_utility"], axis=1)[:, -1] * 1e-9
        cost = output["history_policy_net_cost"][:, -1] * 1e-9
        n_seeds = len(emissions)
        z = 1.96
        return (
            np.mean(emissions), np.mean(utility), np.mean(cost),
            z * np.std(emissions) / np.sqrt(n_seeds),
            z * np.std(utility) / np.sqrt(n_seeds),
            z * np.std(cost) / np.sqrt(n_seeds),
        )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # --- BAU
    e_bau, u_bau, c_bau, e_err, u_err, c_err = _point(outputs_BAU)
    ax_top.errorbar(e_bau, c_bau, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
    ax_top.scatter(e_bau, c_bau, s=scale_marker, color='black', edgecolor='black', label="BAU")
    ax_bottom.errorbar(e_bau, u_bau, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
    ax_bottom.scatter(e_bau, u_bau, s=scale_marker, color='black', edgecolor='black')

    # --- Top-N policies, each split into n_policies wedges
    for rank in sorted(outputs.keys()):
        e, u, c, e_err, u_err, c_err = _point(outputs[rank])
        policy_dict = policy_dicts[rank]

        ax_top.errorbar(e, c, xerr=e_err, yerr=c_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
        ax_bottom.errorbar(e, u, xerr=e_err, yerr=u_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)

        # Outline showing the full-circle envelope for this point
        ax_top.scatter(e, c, s=scale_marker, marker=full_circle_marker(), facecolor='none',
                        edgecolor='black', linewidth=1, linestyle='--', alpha=0.5)
        ax_bottom.scatter(e, u, s=scale_marker, marker=full_circle_marker(), facecolor='none',
                            edgecolor='black', linewidth=1, linestyle='--', alpha=0.5)

        for i, policy in enumerate(policy_names):
            value = policy_dict[policy]
            size = scale_marker_size(value, policy, policy_ranges, scale_marker)
            wedge = half_circle_marker(i * wedge_width, (i + 1) * wedge_width)
            ax_top.scatter(e, c, s=size, marker=wedge, color=policy_colors[policy], edgecolor="black", zorder=2)
            ax_bottom.scatter(e, u, s=size, marker=wedge, color=policy_colors[policy], edgecolor="black", zorder=2)

    ax_top.set_ylabel("Cumulative Net Cost, bn $", fontsize=16)
    ax_bottom.set_ylabel("Cumulative Utility, bn $\n(raw sum — not the log-utility optimisation metric)", fontsize=13)
    ax_bottom.set_xlabel("Cumulative Emissions, MTCO2", fontsize=16)

    legend_elements = [
        Patch(facecolor=policy_colors[p], edgecolor='black',
              label=f"{POLICY_TITLES.get(p, p)} ({policy_ranges[p]['min']:.2f} - {policy_ranges[p]['max']:.2f})")
        for p in policy_names
    ]
    legend_elements += [Patch(facecolor='black', edgecolor='black', label='BAU')]
    small_proxy = plt.Line2D([0], [0], marker=half_circle_marker(0, wedge_width),
                              color='gray', markerfacecolor='gray', markeredgecolor='black',
                              linestyle='None', label='Low intensity (small wedge)', markersize=8)
    large_proxy = plt.Line2D([0], [0], marker=half_circle_marker(0, wedge_width),
                               color='gray', markerfacecolor='gray', markeredgecolor='black',
                               linestyle='None', label='High intensity (large wedge)', markersize=12)
    confidence = plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle='-', label='95% Confidence Interval')
    legend_elements += [confidence, small_proxy, large_proxy]
    ax_bottom.legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{fileName}/Plots/top_policies_tradeoff.png", dpi=dpi)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    n_best: int = N_BEST,
    results_dir: str = RESULTS_DIR,
    existing_calib_folder: str = None,
):
    out_folder, base_params, outputs, outputs_BAU, policy_dicts, Y_top, bounds = run_top_policies(
        n_best=n_best, results_dir=results_dir, existing_calib_folder=existing_calib_folder,
    )
    plot_top_policies_dashboard(base_params, out_folder, outputs, outputs_BAU, policy_dicts, Y_top, dpi=200)
    plot_top_policies_tradeoff(base_params, out_folder, outputs, outputs_BAU, policy_dicts, bounds, dpi=300)
    print(f"\nPlots saved to {out_folder}/Plots/")
    plt.show()


if __name__ == "__main__":
    main()
