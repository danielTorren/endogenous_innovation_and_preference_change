"""
Figures for any one-parameter HHI-and-car-age sweep folder written by
package/generating_data/sweep_hhi_age_gen.py.

  <slug>_sweep_timeseries.png - HHI and fleet mean age against time, one line per
                                parameter value, against the observed target bands
  <slug>_sweep_response.png   - the 2023 mean of each series against the parameter,
                                with a 95% CI over seeds, so the admissible window
                                can be read straight off the x axis

Everything parameter-specific (the axis label, the slug, the x scale) comes out of
the folder's own Data, so this module never needs editing to cover a new sweep.

Arrays arrive with burn-in ALREADY trimmed, so index 0 is Jan 2001. Age is stored
in months and plotted in years, because both target bands are stated in years.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t

from package.plotting_data.calibration_plot import (
    CALIBRATION_START_YEAR,
    TARGET_FLEET_AGE_YEARS,
    TARGET_HHI,
    add_vertical_lines_from_burn_in,
)
from package.resources.utility import createFolder, load_object

C_TARGET = "#666666"
CMAP = "viridis"


def _ci(data_2d):
    """Seed mean and half-width of the 95% CI, over axis 0."""
    mean = np.nanmean(data_2d, axis=0)
    n = data_2d.shape[0]
    half = sem(data_2d, axis=0, nan_policy="omit") * t.ppf(0.975, df=n - 1)
    return mean, half


def _target_band(ax, low, high, label):
    ax.axhspan(low, high, color=C_TARGET, alpha=0.18, zorder=0)
    ax.axhline(low, color=C_TARGET, lw=0.8, ls="--", zorder=0)
    ax.axhline(high, color=C_TARGET, lw=0.8, ls="--", zorder=0, label=label)


def _year_ticks(ax, n_steps, interval_years=5):
    max_year = CALIBRATION_START_YEAR + int((n_steps - 1) // 12)
    last = max_year - (max_year - CALIBRATION_START_YEAR) % interval_years
    years = np.arange(CALIBRATION_START_YEAR, last + 1, interval_years)
    ax.set_xticks((years - CALIBRATION_START_YEAR) * 12)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year", fontsize=14)


def plot_timeseries(fileName, slug, label, values, hhi, age_yr, base_params, dpi=200):
    """
    One line per parameter value, seed-mean only.

    No confidence ribbons: ten overlapping bands are unreadable, and the
    per-value uncertainty is what the response figure is for.
    """
    n_steps = hhi.shape[2]
    steps = np.arange(n_steps)
    cmap = plt.get_cmap(CMAP)
    colours = [cmap(i / max(len(values) - 1, 1)) for i in range(len(values))]

    fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for i, value in enumerate(values):
        axs[0].plot(steps, np.nanmean(hhi[i], axis=0), color=colours[i], lw=1.4,
                    label=f"{value:.2e}")
        axs[1].plot(steps, np.nanmean(age_yr[i], axis=0), color=colours[i], lw=1.4)

    _target_band(axs[0], *TARGET_HHI, "Observed range")
    _target_band(axs[1], *TARGET_FLEET_AGE_YEARS, "Observed range")

    axs[0].set_ylabel("Market concentration, HHI", fontsize=14)
    axs[1].set_ylabel("Mean fleet car age, years", fontsize=14)
    for ax in axs:
        add_vertical_lines_from_burn_in(ax, base_params,
                                       annotation_height_prop=[0.3, 0.3, 0.3])
    _year_ticks(axs[1], n_steps)

    # Legend OUTSIDE the axes. Ten series plus the target band is a tall key, and
    # inside the panel it covered the HHI lines it was there to identify.
    axs[0].legend(title=label, fontsize=9, title_fontsize=10,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.suptitle(f"HHI and fleet age against {label}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])

    path = f"{fileName}/Plots/{slug}_sweep_timeseries.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_response(fileName, slug, label, values, hhi, age_yr, dpi=200):
    """2023 mean of each series against the parameter, with a 95% CI over seeds."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, ylabel, band in (
        (axs[0], hhi, "Market concentration, HHI", TARGET_HHI),
        (axs[1], age_yr, "Mean fleet car age, years", TARGET_FLEET_AGE_YEARS),
    ):
        # Per-seed 2023 mean first, THEN the CI over seeds. Averaging the 12
        # months before taking the CI is what makes the interval a statement
        # about seed spread rather than about within-year wobble.
        per_seed = np.nanmean(data[:, :, -12:], axis=2)   # (n_values, seeds)
        mean, half = _ci(per_seed.T)
        ax.errorbar(values, mean, yerr=half, marker="o", capsize=3,
                    color="#0072B2", lw=1.4)
        _target_band(ax, *band, "Observed range")
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=10)

    fig.suptitle(f"2023 mean against {label} (95% CI over seeds)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = f"{fileName}/Plots/{slug}_sweep_response.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main(fileName):
    D = fileName + "/Data"
    base_params = load_object(D, "base_params")
    values = np.asarray(load_object(D, "param_list"), dtype=float)
    slug = load_object(D, "param_name")
    label = load_object(D, "param_label")

    hhi = np.asarray(load_object(D, "data_array_hhi"), dtype=float)
    age_yr = np.asarray(load_object(D, "data_array_age_fleet"), dtype=float) / 12

    if not os.path.exists(fileName + "/Plots"):
        createFolder(fileName)

    plot_timeseries(fileName, slug, label, values, hhi, age_yr, base_params)
    plot_response(fileName, slug, label, values, hhi, age_yr)
    return fileName


if __name__ == "__main__":
    # Folder on argv, so this does not need editing between runs.
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m package.plotting_data.sweep_hhi_age_plot "
                         "results/<sweep folder>")
    main(sys.argv[1])
