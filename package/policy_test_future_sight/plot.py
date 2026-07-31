"""
package/policy_test_future_sight/plot.py — naive vs. forward-looking, side by side.

One PNG per metric (EV share, EV price, flow/cumulative emissions,
flow/cumulative utility, car age, net cost), each with two panels sharing a
y-axis: left = naive agents, right = forward-looking agents. Same policy
gets the same colour/marker in both panels, so the only thing that differs
left-to-right is the expectation_mode — directly comparable at a glance.

Panel logic (slicing, CI bands, "Policy end" line, x-axis) mirrors
package.surrogate.best_policies.plot_top_policies_dashboard exactly; only the
side-by-side layout is new.

Usage (regenerate plots from already-saved data, no ABM re-run):
    python -m package.policy_test_future_sight.plot --folder results/policy_test_future_sight_<timestamp>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import sem, t

from package.resources.utility import load_object
from package.surrogate.best_policies import _label_for_rank, RANK_MARKERS

MODE_TITLES = {"naive": "Naive", "forward_looking": "Forward-looking"}


def plot_future_sight_comparison(
    base_params, fileName, outputs_by_mode, outputs_bau_by_mode, policy_dicts, Y_top, dpi=300,
):
    # duration_future, not duration_future - 1: load_in_controller now
    # actually simulates the full requested duration (see its fixed
    # off-by-one loop in package.resources.run).
    time_steps = np.arange(base_params["duration_future"])
    start = base_params["duration_burn_in"] + base_params["duration_calibration"]

    modes = list(outputs_by_mode.keys())
    ranks = sorted(next(iter(outputs_by_mode.values())).keys())
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

    # --- Panel functions: take explicit outputs/outputs_BAU so the same logic
    # can be called once per expectation_mode into different axes ---

    def panel_ev_share(ax, outputs, outputs_BAU, add_labels=True):
        plot_line_with_ci(ax, outputs_BAU["history_prop_EV"][:, start:], 'black', 'o', '-', 'BAU - EV Adoption')
        plot_line_with_ci(ax, outputs_BAU["history_past_new_bought_vehicles_prop_ev"], 'black', 'o', '--', 'BAU - EV Sales')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, outputs[r]["history_prop_EV"][:, start:], color, marker, '-', None)
            plot_line_with_ci(ax, outputs[r]["history_past_new_bought_vehicles_prop_ev"], color, marker, '--', lbl)
        ax.set_ylabel("EV Share", fontsize=14)
        _add_vline(ax, annotation_height_prop=(0.6, 0.2, 0.2))

    def panel_ev_price(ax, outputs, outputs_BAU, add_labels=False):
        for i, label_txt, linestyle in [(0, 'New', '-'), (1, 'Used', '--')]:
            plot_line_with_ci(ax, outputs_BAU["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                                'black', 'o', linestyle, f"BAU - {label_txt}")
        for r in ranks:
            color, marker = rank_style[r]
            for i, linestyle in [(0, '-'), (1, '--')]:
                lbl = rank_label[r] if (add_labels and i == 0) else None
                plot_line_with_ci(ax, outputs[r]["history_mean_price_ICE_EV_arr"][:, :, i, 1],
                                    color, marker, linestyle, lbl)
        ax.set_ylabel("EV Sale Price, $", fontsize=14)
        _add_vline(ax, annotation_height_prop=(0.9, 0.2, 0.2))

    def panel_emissions(ax, outputs, outputs_BAU, cumulative=False, add_labels=True):
        transform = (lambda x: np.cumsum(x, axis=1) * 1e-9) if cumulative else (lambda x: x * 1e-9)
        ylabel = "Cumulative Emissions, MTCO2" if cumulative else "Flow Emissions, MTCO2"
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_emissions"]), 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, transform(outputs[r]["history_total_emissions"]), color, marker, '-', lbl)
        ax.set_ylabel(ylabel, fontsize=14)
        _add_vline(ax, annotation_height_prop=(0.5, 0.2, 0.2))

    def panel_utility(ax, outputs, outputs_BAU, cumulative=False, add_labels=False):
        # Raw utility flow/sum — not the log-utility metric the surrogate
        # actually optimises against (see sampling.compute_log_utility_metric).
        transform = (lambda x: np.cumsum(x, axis=1) * 1e-9) if cumulative else (lambda x: x * 1e-9)
        ylabel = "Cumulative Utility (raw), bn $" if cumulative else "Flow Utility (raw), bn $"
        plot_line_with_ci(ax, transform(outputs_BAU["history_total_utility"]), 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, transform(outputs[r]["history_total_utility"]), color, marker, '-', lbl)
        ax.set_ylabel(ylabel, fontsize=14)
        _add_vline(ax, annotation_height_prop=(0.2, 0.2, 0.2))

    def panel_car_age(ax, outputs, outputs_BAU, add_labels=False):
        plot_line_with_ci(ax, outputs_BAU["history_mean_car_age"], 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, outputs[r]["history_mean_car_age"], color, marker, '-', lbl)
        ax.set_ylabel("Car Age, months", fontsize=14)
        _add_vline(ax, annotation_height_prop=(0.5, 0.2, 0.2))

    def panel_net_cost(ax, outputs, outputs_BAU, add_labels=True):
        plot_line_with_ci(ax, outputs_BAU["history_policy_net_cost"] * 1e-9, 'black', 'o', '-', 'BAU')
        for r in ranks:
            color, marker = rank_style[r]
            lbl = rank_label[r] if add_labels else None
            plot_line_with_ci(ax, outputs[r]["history_policy_net_cost"] * 1e-9, color, marker, '-', lbl)
        ax.set_ylabel("Cumulative Net Cost, bn $", fontsize=14)
        _add_vline(ax, annotation_height_prop=(0.5, 0.2, 0.2))

    start_year = 2024
    tick_years = np.arange(start_year, start_year + (time_steps[-1] // 12) + 5, 5)
    tick_positions = (tick_years - start_year) * 12

    def set_xaxis(ax):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(year) for year in tick_years])
        ax.set_xlabel("Year", fontsize=14)

    individual_panels = [
        ("ev_share",             lambda ax, o, ob: panel_ev_share(ax, o, ob, add_labels=True)),
        ("ev_price",             lambda ax, o, ob: panel_ev_price(ax, o, ob, add_labels=True)),
        ("flow_emissions",       lambda ax, o, ob: panel_emissions(ax, o, ob, cumulative=False, add_labels=True)),
        ("cumulative_emissions", lambda ax, o, ob: panel_emissions(ax, o, ob, cumulative=True,  add_labels=True)),
        ("flow_utility",         lambda ax, o, ob: panel_utility(ax,   o, ob, cumulative=False, add_labels=True)),
        ("cumulative_utility",   lambda ax, o, ob: panel_utility(ax,   o, ob, cumulative=True,  add_labels=True)),
        ("car_age",              lambda ax, o, ob: panel_car_age(ax,   o, ob, add_labels=True)),
        ("net_cost",             lambda ax, o, ob: panel_net_cost(ax,  o, ob, add_labels=True)),
    ]

    for name, plot_fn in individual_panels:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
        for ax, mode in zip(axs, modes):
            plot_fn(ax, outputs_by_mode[mode], outputs_bau_by_mode[mode])
            ax.set_title(MODE_TITLES.get(mode, mode), fontsize=15)
            set_xaxis(ax)

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0.00), fontsize=8)
        fig.tight_layout(rect=[0.01, 0.22, 0.98, 1])
        fig.subplots_adjust(wspace=0.08)
        fig.savefig(f"{fileName}/Plots/individual_{name}_naive_vs_forward_looking.png", dpi=dpi)
        plt.close(fig)


def main(folder: str):
    base_params = load_object(folder + "/Data", "base_params")
    outputs_by_mode = load_object(folder + "/Data", "outputs_by_mode")
    outputs_bau_by_mode = load_object(folder + "/Data", "outputs_bau_by_mode")
    policy_dicts = load_object(folder + "/Data", "policy_dicts")
    Y_top = load_object(folder + "/Data", "Y_top")

    plot_future_sight_comparison(
        base_params, folder, outputs_by_mode, outputs_bau_by_mode, policy_dicts, Y_top, dpi=200,
    )
    print(f"Plots saved to {folder}/Plots/")
    plt.show()


if __name__ == "__main__":
    import sys
    _folder = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--folder="):
            _folder = _arg.split("=", 1)[1]
    if _folder is None:
        raise SystemExit("Usage: python -m package.policy_test_future_sight.plot --folder=results/policy_test_future_sight_<timestamp>")
    main(_folder)
