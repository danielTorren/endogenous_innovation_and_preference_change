"""
package/variable_carbon_price/plot.py — time-series plots for gen.py's
rising-carbon-price vs. flat-carbon-tax, naive-vs-forward-looking results.

Produces ONE FIGURE PER METRIC (cost / utility / emissions / ev_uptake /
sales), each saved as its own PNG under <save_dir>/. Within a figure, columns
= expectation_mode (naive / forward_looking), so comparing the two panels
side by side shows exactly what forward_looking_expectations changes for
that metric. Every panel shows BAU (black) plus every ramp (viridis, by
end-price) and every flat (autumn, by level) scenario as a mean line with a
95% CI band across seeds.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from package.resources.utility import load_object

RESULTS_DIR = "results/variable_carbon_price"

METRICS = ("cost", "utility", "emissions", "ev_uptake", "sales")
METRIC_LABELS = {
    "cost": "Cumulative net policy cost ($)",
    "utility": "Monthly aggregate utility (flow, not cumulative)",
    "emissions": "Monthly emissions, driving + production (kg CO2, not cumulative)",
    "ev_uptake": "EV share of current fleet (stock)",
    "sales": "EV share of new car sales (flow)",
}
EXPECTATION_TITLES = {
    "naive": "Naive / permanent-policy agents",
    "forward_looking": "Forward-looking agents",
}


def _calc_bounds_nan(data, confidence_level=0.95):
    """
    Like package.resources.utility.calc_bounds, but NaN-aware (the "sales"
    metric is NaN for any seed/month where nobody bought a new car that
    month — see gen.py) so one NaN seed doesn't blank out the whole month's
    mean/CI for every other seed.
    """
    mean = np.nanmean(data, axis=1)
    std = np.nanstd(data, axis=1)
    n = np.sum(~np.isnan(data), axis=1)
    sem = std / np.sqrt(np.maximum(n, 1))
    z_score = np.abs(stats.norm.ppf((1 - confidence_level) / 2))
    margin = z_score * sem
    return mean, mean - margin, mean + margin


def _plot_scenario(ax, months, data, color, linestyle, label, linewidth=1.6):
    """data: shape (n_months, n_seeds). Shades a 95% CI band across seeds."""
    mean, lower, upper = _calc_bounds_nan(data, confidence_level=0.95)
    ax.plot(months, mean, color=color, linestyle=linestyle, label=label, linewidth=linewidth)
    ax.fill_between(months, lower, upper, color=color, alpha=0.15, linewidth=0)


def _plot_metric(metric, results, expectation_modes, ramp_values, flat_values,
                  ramp_colors, flat_colors, save_dir):
    fig, axes = plt.subplots(
        1, len(expectation_modes),
        figsize=(7 * len(expectation_modes), 5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for col, mode in enumerate(expectation_modes):
        ax = axes[col]

        bau = results[(mode, "BAU")]
        _plot_scenario(ax, bau["months"], bau[metric], color="black", linestyle="-",
                       label="BAU", linewidth=2.2)

        for v, color in zip(ramp_values, ramp_colors):
            r = results[(mode, f"ramp_{v:.3f}")]
            _plot_scenario(ax, r["months"], r[metric], color=color, linestyle="-",
                           label=f"ramp → {v:.2f}")

        for v, color in zip(flat_values, flat_colors):
            r = results[(mode, f"flat_{v:.3f}")]
            _plot_scenario(ax, r["months"], r[metric], color=color, linestyle="--",
                           label=f"flat = {v:.2f}")

        ax.set_title(EXPECTATION_TITLES.get(mode, mode))
        ax.set_xlabel("Months since policy start (2023)")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{METRIC_LABELS[metric]} — mean ± 95% CI across seeds", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = f"{save_dir}/variable_carbon_price_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        # Raises on headless cluster nodes with no display backend; the
        # figure is already saved above, so it's safe to skip silently.
        plt.show()
    except Exception:
        pass
    return fig


def plot_time_series(results: dict, scenarios: list, save_dir: str = f"{RESULTS_DIR}/Plots"):
    """
    Saves one PNG per metric in METRICS to save_dir. Returns
    {metric: matplotlib.figure.Figure}.
    """
    os.makedirs(save_dir, exist_ok=True)

    expectation_modes = sorted({s["expectation_mode"] for s in scenarios})
    ramp_values = sorted({s["value"] for s in scenarios if s["kind"] == "ramp"})
    flat_values = sorted({s["value"] for s in scenarios if s["kind"] == "flat"})

    ramp_colors = plt.cm.viridis(np.linspace(0, 1, max(len(ramp_values), 1)))
    flat_colors = plt.cm.autumn(np.linspace(0, 1, max(len(flat_values), 1)))

    figs = {}
    for metric in METRICS:
        figs[metric] = _plot_metric(
            metric, results, expectation_modes, ramp_values, flat_values,
            ramp_colors, flat_colors, save_dir,
        )
    return figs


def main(results_dir: str = RESULTS_DIR):
    results = load_object(f"{results_dir}/Data", "variable_carbon_price_results")
    scenarios = load_object(f"{results_dir}/Data", "scenarios")
    plot_time_series(results, scenarios, save_dir=f"{results_dir}/Plots")
    print(f"Saved one PNG per metric ({', '.join(METRICS)}) to {results_dir}/Plots/")


if __name__ == "__main__":
    main()
