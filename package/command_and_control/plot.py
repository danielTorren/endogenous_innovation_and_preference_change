"""
package/command_and_control/plot.py — time-series plots for gen.py's
command-and-control ICE phase-out results (research ban vs. research+sales
ban vs. research+sales+driving ban, at a 2030 vs. a 2035 start date, run out
to 2050).

Produces ONE FIGURE PER METRIC (cost / utility / emissions / ev_uptake /
sales), each saved as its own PNG under <save_dir>/. Within a figure, columns
= ban_year (2030 / 2035), so comparing the two panels side by side shows how
much starting five years earlier changes the outcome. Every panel shows BAU
(black, no bans) plus every stringency level (viridis, by strictness) as a
mean line with a 95% CI band across seeds.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from package.resources.utility import load_object

METRICS = ("cost", "utility", "emissions", "ev_uptake", "sales")
METRIC_LABELS = {
    "cost": "Cumulative net policy cost ($)",
    "utility": "Monthly aggregate utility (flow, not cumulative)",
    "emissions": "Monthly emissions, driving + production (kg CO2, not cumulative)",
    "ev_uptake": "EV share of current fleet (stock)",
    "sales": "EV share of new car sales (flow)",
}
STRINGENCY_LEVELS = ("research_ban", "research_sales_ban", "research_sales_driving_ban")
STRINGENCY_LABELS = {
    "research_ban": "Research ban only",
    "research_sales_ban": "+ sales ban",
    "research_sales_driving_ban": "+ sales + driving ban",
}


def _calc_bounds_nan(data, confidence_level=0.95):
    """
    NaN-aware mean/CI across seeds (data: shape (n_months, n_seeds)) — the
    "sales" metric is NaN for any seed/month where nobody bought a new car
    that month (see gen.py), so one NaN seed shouldn't blank out the whole
    month's mean/CI for every other seed.
    """
    mean = np.nanmean(data, axis=1)
    std = np.nanstd(data, axis=1)
    n = np.sum(~np.isnan(data), axis=1)
    sem = std / np.sqrt(np.maximum(n, 1))
    z_score = np.abs(stats.norm.ppf((1 - confidence_level) / 2))
    margin = z_score * sem
    return mean, mean - margin, mean + margin


def _plot_scenario(ax, months, data, color, linestyle, label, linewidth=1.6):
    mean, lower, upper = _calc_bounds_nan(data, confidence_level=0.95)
    ax.plot(months, mean, color=color, linestyle=linestyle, label=label, linewidth=linewidth)
    ax.fill_between(months, lower, upper, color=color, alpha=0.15, linewidth=0)


def _plot_metric(metric, results, ban_years, stringency_levels, stringency_colors, save_dir):
    fig, axes = plt.subplots(
        1, len(ban_years),
        figsize=(7 * len(ban_years), 5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for col, year in enumerate(ban_years):
        ax = axes[col]

        bau = results["BAU"]
        _plot_scenario(ax, bau["months"], bau[metric], color="black", linestyle="-",
                       label="BAU (no ban)", linewidth=2.2)

        for stringency, color in zip(stringency_levels, stringency_colors):
            r = results[f"{year}_{stringency}"]
            _plot_scenario(ax, r["months"], r[metric], color=color, linestyle="-",
                           label=STRINGENCY_LABELS.get(stringency, stringency))

        ax.set_title(f"Ban start: {year}")
        ax.set_xlabel("Months since policy start (2024)")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=7, ncol=1)

    fig.suptitle(f"{METRIC_LABELS[metric]} — mean ± 95% CI across seeds", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = f"{save_dir}/command_and_control_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        # Raises on headless cluster nodes with no display backend; the
        # figure is already saved above, so it's safe to skip silently.
        plt.show()
    except Exception:
        pass
    return fig


def plot_time_series(results: dict, scenarios: list, save_dir: str):
    """
    Saves one PNG per metric in METRICS to save_dir. Returns
    {metric: matplotlib.figure.Figure}.
    """
    os.makedirs(save_dir, exist_ok=True)

    ban_years = sorted({s["ban_year"] for s in scenarios if s["kind"] == "ban"})
    stringency_levels = [s for s in STRINGENCY_LEVELS if any(sc["stringency"] == s for sc in scenarios)]

    stringency_colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(stringency_levels), 1)))

    figs = {}
    for metric in METRICS:
        figs[metric] = _plot_metric(metric, results, ban_years, stringency_levels, stringency_colors, save_dir)
    return figs


def main(results_dir: str):
    """
    results_dir : REQUIRED — the exact self-contained, timestamped folder
        gen.main() printed/returned (e.g.
        "results/command_and_control_14_53_52__30_07_2026"). There's no
        shared fixed folder to fall back to any more — see gen.py's module
        docstring.
    """
    results = load_object(f"{results_dir}/Data", "command_and_control_results")
    scenarios = load_object(f"{results_dir}/Data", "scenarios")
    plot_time_series(results, scenarios, save_dir=f"{results_dir}/Plots")
    print(f"Saved one PNG per metric ({', '.join(METRICS)}) to {results_dir}/Plots/")


if __name__ == "__main__":
    import sys
    _results_dir = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--results_dir="):
            _results_dir = _arg.split("=", 1)[1]
    if _results_dir is None:
        raise SystemExit(
            "package.command_and_control.plot requires --results_dir=PATH -- the exact "
            "folder gen.main() printed/returned, e.g.:\n"
            "  python -m package.command_and_control.plot "
            "--results_dir=results/command_and_control_14_53_52__30_07_2026"
        )
    main(_results_dir)
