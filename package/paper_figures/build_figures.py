"""
One-stop figure builder for the manuscript and the supplementary material.

Drop the results folder names into RUNS below, then:

    python -m package.paper_figures.build_figures

It does four things:

1. Re-plots each figure from its results folder (existing plotting code only;
   no simulation, no duplicated plotting logic).
2. Copies the produced PNG into docs/paper/, named by its number in the paper:
   main figures become docs/paper/Figure_<n>.png, supplementary figures become
   docs/paper/supplementary_figs/Supp_Figure_<n>.png.
3. Rewrites the \\includegraphics paths in manuscript.tex and supplementary.tex
   so they point at those files. Figures are matched by their \\label, not by
   position, so reordering the .tex is safe.
4. Rebuilds the numbers in Tables 3 and 4 from the same results folders and
   rewrites the matching tabular block in manuscript.tex. Tables are matched by
   their label too, and their keys on the command line are T3 and T4.

Useful flags:

    --list              show the manifest and what is missing, change nothing
    --only 3,S6,T3      build a subset (S prefix = supplementary, T = table)
    --skip-plot         reuse the PNGs already in the results folders
    --no-tex            copy figures but leave the .tex files alone
    --no-tables         build the figures only, leave the tables alone
    --dry-run           print what would happen

Figures that are not model output (main Figure 1, supplementary Figures 2-4)
are not regenerated. Put those source images in docs/paper/static_figs/ under
the filenames listed in STATIC_SOURCES and they get copied and renumbered with
everything else.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The plotting modules all end in plt.show(); harmless under Agg, but silencing
# it keeps the log clean and avoids any backend that does decide to block.
plt.show = lambda *args, **kwargs: None


# ---------------------------------------------------------------------------
# 1. RESULTS FOLDERS -- this is the only part you normally edit
# ---------------------------------------------------------------------------

RUNS = {
    # ---- main manuscript -------------------------------------------------
    # Figure 2 (validation dashboard) and supplementary Figure 5 (car qualities)
    # both come out of the same calibration run.
    "calibration": "results/calibration_gen_15_46_01__19_08_2026",
    # Figure 3 -- package/analysis/vary_single_policy_gen.py
    "single_policies": "results/vary_single_policy_gen_20_01_30__18_08_2026",
    # Figure 4 -- package/analysis/endogenous_policy_intensity_pair_gen.py
    "policy_pairs": "results/endog_pair_20_22_47__18_08_2026",
    # Table 3 -- package/analysis/endogenous_policy_intensity_single_gen.py.
    # The endogenous single-policy solve: one optimised intensity per policy, of
    # which only those landing in the target uptake band get a column.
    "single_policy_endog": "results/endog_single_13_47_10__19_08_2026",
    # Figure 5 and Table 4 -- package/analysis/low_policy_intensity_gen.py
    "low_intensity": "pair_low_intensity_policies_11_46_52__19_08_2026",
    # Figure 6 -- package/generating_data/single_experiment_gen.py
    # No single_experiment_* run in results/ yet.
    "single_experiment": "single_experiment_13_58_56__19_08_2026",
    # Figure 7 -- vary_single_param_gen over max_num_cars_prop
    "used_car_capacity": "results/sen_vary_max_num_cars_prop_20_27_28__18_08_2026",

    # ---- supplementary material -----------------------------------------
    # Figure 1 -- package/calibration/sbi_single_seed_gen.py
    "posterior": "results/sbi_seed_av_15_31_53__18_08_2026",
    # Figure 6 -- ten vary_single runs, in panel order a-j. Order matters:
    # a alpha, b r, c mu, d kappa, e b_chi, f a_chi, g lambda, h delta,
    # i K_EV, j K_ICE (see supplementary_runs/fig06_local_sensitivity_gen.PANELS).
    "local_sensitivity": [
        "results/single_param_vary_20_27_49__18_08_2026",  # a  alpha
        "results/single_param_vary_20_30_11__18_08_2026",  # b  r
        "results/single_param_vary_20_32_19__18_08_2026",  # c  mu
        "results/single_param_vary_20_34_32__18_08_2026",  # d  kappa
        "results/single_param_vary_20_36_44__18_08_2026",  # e  b_chi
        "results/single_param_vary_20_40_39__18_08_2026",  # f  a_chi
        "results/single_param_vary_20_42_55__18_08_2026",  # g  lambda
        "results/single_param_vary_20_45_13__18_08_2026",  # h  delta
        "results/single_param_vary_20_47_26__18_08_2026",  # i  K_EV
        "results/single_param_vary_20_49_48__18_08_2026",  # j  K_ICE
    ],
    # Figures 7 and 8 -- package/supplementary_runs/fig07_08_sobol_gen.py
    "sobol": "results/sensitivity_analysis_20_28_14__18_08_2026",
    # Figures 9 and 10 -- package/supplementary_runs/fig09_10_bau_gen.py
    "bau_grid": "",
    # Figures 11-14 -- package/supplementary_runs/fig11_14_policy_grid_gen.py
    "grid_beta_carbon": "results/cross_beta_multiplier_vs_Carbon_price_20_27_49__18_08_2026",
    # No cross_beta_multiplier_vs_Adoption_subsidy_* run in results/ yet.
    "grid_beta_rebate": "",
    "grid_achi_carbon": "results/cross_a_chi_vs_Carbon_price_20_32_04__18_08_2026",
    "grid_achi_rebate": "results/cross_a_chi_vs_Adoption_subsidy_20_52_15__18_08_2026",
    # The BAU baseline for Figures 11-14 has no policy in it, so it depends only
    # on the physical parameter: one BAU run serves both policies on that axis.
    # A cross_* run writes data_cross_bau.pkl of its own; point these at whichever
    # run actually has that file (leave empty to use each figure's own folder).
    # None of the current cross_* runs has data_cross_bau.pkl, so both are empty
    # and the heatmaps come out without the BAU contour.
    "grid_bau_beta": "",
    "grid_bau_achi": "",
}

# N_samples used for the Sobol run, which is baked into its output filename.
SOBOL_N_SAMPLES = 256

# The EV uptake band a run has to land in to count as reaching the 95% target,
# used by Figure 4 and by Table 3 when it picks its columns.
MIN_EV_UPTAKE = 0.94
MAX_EV_UPTAKE = 0.96

# Scale applied to cumulative emissions in the tables. This keeps the magnitude
# of the hand-typed Table 3, whose BAU row was of order 100 "MTCO2". The figure
# code multiplies the same quantity by 1e-9 instead, so the table and the figure
# axis have never agreed; flip this to 1e-9 to make the figures the reference.
EMISSIONS_SCALE = 1e-6

# Figures the model does not produce. Put these files in docs/paper/static_figs/.
STATIC_SOURCES = {
    "diagram": "Figure_1.png",
    "Costs_Electricity_Gasoline": "electricity_vs_gasoline_prices.png",
    "Emissions_Electricity_Gasoline": "electricity_vs_gasoline_emissions.png",
    "NK": "insights_on_NK.png",
}


# ---------------------------------------------------------------------------
# 2. PATHS
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER_DIR = os.path.join(REPO_ROOT, "docs", "paper")
SUPP_DIR = os.path.join(PAPER_DIR, "supplementary_figs")
STATIC_DIR = os.path.join(PAPER_DIR, "static_figs")
MANUSCRIPT_TEX = os.path.join(PAPER_DIR, "manuscript.tex")
SUPPLEMENTARY_TEX = os.path.join(PAPER_DIR, "supplementary.tex")


def _abs(path):
    """
    Absolute path of a configured folder. A bare run name with no separator in
    it means results/<name>, the same rule the gen scripts use, so a folder name
    pasted into RUNS works with or without the results/ prefix.
    """
    if os.path.isabs(path):
        return path
    if "/" not in path.replace(chr(92), "/"):
        path = "results/" + path
    return os.path.join(REPO_ROOT, path)


# ---------------------------------------------------------------------------
# 3. RE-PLOTTING -- every function here calls existing code, nothing new
# ---------------------------------------------------------------------------

def _replot_calibration(folder):
    from package.plotting_data.calibration_plot import main as plot_calibration

    plot_calibration(fileName=_abs(folder))


def _replot_single_policies(folder):
    from package.analysis.vary_single_policy_plot import main as plot_single_policies

    plot_single_policies(file_name=_abs(folder))


def _replot_policy_pairs(folder):
    """
    endogenous_policy_intensity_pair_plot.main() writes into a fresh
    all_policies_<timestamp> folder, which we cannot address afterwards, so
    call the plotting function it wraps and point it at the run folder itself.
    """
    from package.analysis.endogenous_policy_intensity_pair_plot import (
        plot_emissions_tradeoffs_from_outcomes,
    )
    from package.resources.utility import load_object

    path = _abs(folder)
    base_params = load_object(f"{path}/Data", "base_params")
    outcomes_BAU = load_object(f"{path}/Data", "outcomes_BAU")
    single_policy_outcomes = load_object(f"{path}/Data", "single_policy_outcomes")
    pairwise_outcomes = load_object(f"{path}/Data", "pairwise_outcomes")

    plot_emissions_tradeoffs_from_outcomes(
        base_params,
        pairwise_outcomes,
        single_policy_outcomes,
        outcomes_BAU,
        path,
        min_ev_uptake=MIN_EV_UPTAKE,
        max_ev_uptake=MAX_EV_UPTAKE,
        dpi=300,
        insets=True,
        plot_name="emissions_tradeoff",
    )


def _replot_low_intensity(folder):
    from package.analysis.low_policy_intensity_plot import main as plot_low_intensity

    plot_low_intensity(fileName=_abs(folder))


def _replot_preferences(folder):
    from package.plotting_data.single_experiment_plot import plot_preferences
    from package.resources.utility import load_object

    path = _abs(folder)
    controller = load_object(f"{path}/Data", "controller")
    plot_preferences(controller.social_network, path, dpi=300)


def _replot_used_car_capacity(folder):
    from package.plotting_data.sen_vary_single_param_plot_second_hand_cars import (
        plot_single_ev_prop,
    )
    from package.resources.utility import load_object

    real_data = load_object("package/calibration_data", "calibration_data_output")["EV Prop"]
    plot_single_ev_prop(_abs(folder), real_data)


def _replot_posterior(folder):
    from package.supplementary_runs.fig01_posterior_plot import plot_fig1_posterior

    plot_fig1_posterior(results_folder=_abs(folder))


def _local_sensitivity_output_folder(folders):
    """Same derivation as fig06_local_sensitivity_gen.run_fig6."""
    return "results/" + os.path.basename(folders[0].rstrip("/")) + "_fig6_combined"


def _replot_local_sensitivity(folders):
    from package.supplementary_runs.fig06_local_sensitivity_plot import plot_fig6_combined

    letters = "abcdefghij"
    panel_folders = [(letters[i], _abs(f)) for i, f in enumerate(folders)]
    plot_fig6_combined(
        panel_folders,
        output_folder=_abs(_local_sensitivity_output_folder(folders)),
    )


def _replot_sobol(folder):
    from package.plotting_data.sensitivity_analysis_calibration_plot import main as plot_sobol

    plot_sobol(
        fileName=_abs(folder),
        plot_outputs=[
            "emissions_stock",
            "ev_uptake",
            "total_firm_profit",
            "market_concentration",
            "utility",
            "car_age",
        ],
        plot_dict={
            "emissions_stock": {"title": r"Cumulative Emissions, $E$", "colour": "red", "linestyle": "--"},
            "ev_uptake": {"title": r"EV Adoption Proportion", "colour": "blue", "linestyle": "."},
            "total_firm_profit": {"title": r"Firm Profit, $\$$", "colour": "orange", "linestyle": "-."},
            "market_concentration": {"title": r"Market Concentration HHI", "colour": "green", "linestyle": "--"},
            "utility": {"title": r"Utility, $\$$", "colour": "yellow", "linestyle": "--"},
            "car_age": {"title": r"Mean Car Age", "colour": "purple", "linestyle": "."},
        },
        titles=["K_ICE", "K_EV", "delta", "lambda", "a_chi", "b_chi", "kappa", "mu", "r", "alpha"],
    )


def _replot_bau_timeseries(folder):
    from package.supplementary_runs.fig09_bau_timeseries_plot import plot_fig9_bau_timeseries

    plot_fig9_bau_timeseries(_abs(folder))


def _replot_bau_elasticity(folder):
    from package.supplementary_runs.fig10_bau_elasticity_plot import plot_fig10

    plot_fig10(_abs(folder))


def _policy_grid_replotter(bau_key):
    """
    load_and_plot_combined needs data_cross_bau, which a cross_* run only holds
    if its BAU sweep finished. bau_key names the RUNS entry to borrow it from;
    empty means the figure's own folder already has it.
    """
    def replot(folder):
        from package.plotting_data.policy_sensitivity_plot import load_and_plot_combined

        bau = RUNS.get(bau_key, "") if bau_key else ""
        load_and_plot_combined(
            _abs(folder),
            bau_results_folder=_abs(bau) if bau else None,
        )

    replot.__name__ = f"_replot_policy_grid<bau={bau_key or 'own folder'}>"
    return replot


# ---------------------------------------------------------------------------
# 4. THE MANIFEST -- figure number -> label, results folder, artifact, plotter
# ---------------------------------------------------------------------------

class Figure:
    def __init__(self, number, supplementary, label, run_key, artifact, replot,
                 caption, out_folder=None):
        self.number = number
        self.supplementary = supplementary
        self.label = label            # \label{fig:<label>} in the .tex
        self.run_key = run_key        # key into RUNS, or None for static figures
        self.artifact = artifact      # path inside the run folder, or a callable
        self.replot = replot          # callable(folder) -> None, or None
        self.caption = caption        # short human description, for --list
        self.out_folder = out_folder  # callable(folders) -> folder holding artifact

    @property
    def key(self):
        return ("S" if self.supplementary else "") + str(self.number)

    @property
    def dest_name(self):
        if self.supplementary:
            return f"Supp_Figure_{self.number}.png"
        return f"Figure_{self.number}.png"

    @property
    def dest_path(self):
        return os.path.join(SUPP_DIR if self.supplementary else PAPER_DIR, self.dest_name)

    @property
    def tex_path(self):
        """Path as written into the .tex, relative to docs/paper/."""
        if self.supplementary:
            return f"supplementary_figs/{self.dest_name}"
        return self.dest_name

    @property
    def is_static(self):
        return self.run_key is None

    def folders(self):
        """Configured results folder(s) for this figure, [] if unset."""
        if self.is_static:
            return []
        return _run_folders(self.run_key)

    def source_path(self):
        """Absolute path of the PNG this figure is copied from, or None."""
        if self.is_static:
            source = STATIC_SOURCES.get(self.label)
            candidate = os.path.join(STATIC_DIR, source) if source else None
            if candidate and os.path.isfile(candidate):
                return candidate
            # Nothing in static_figs/, but the image may already sit at its
            # destination (main Figure 1 does) -- leave it where it is.
            if os.path.isfile(self.dest_path):
                return self.dest_path
            return candidate
        folders = self.folders()
        if not folders:
            return None
        base = self.out_folder(folders) if self.out_folder else folders[0]
        return os.path.join(_abs(base), self.artifact)


def _run_folders(run_key):
    """The RUNS entry as a list of folders, [] if unset. One string or a list."""
    value = RUNS.get(run_key, "")
    if isinstance(value, str):
        return [value] if value else []
    return [f for f in value if f]


def _sobol_artifact(order):
    return os.path.join(
        "Prints",
        f"10_{SOBOL_N_SAMPLES}_{order}_multi_output_sensitivity_plot_2row.png",
    )


FIGURES = [
    # ---- manuscript ------------------------------------------------------
    Figure(1, False, "diagram", None, None, None,
           "Model structure diagram (hand-drawn, not model output)"),
    Figure(2, False, "validation_plots", "calibration",
           "Plots/calibration_fit.png", _replot_calibration,
           "Calibration validation dashboard, 2001-2023"),
    Figure(3, False, "individual_policy_multiintensity1", "single_policies",
           "Plots/policy_intensity_effects_means_0234567.png", _replot_single_policies,
           "Key model outputs for single policies"),
    Figure(4, False, "policy_pair", "policy_pairs",
           "Plots/emissions_tradeoffs/emissions_tradeoff.png", _replot_policy_pairs,
           "Policy pair emissions/utility/cost trade-offs"),
    Figure(5, False, "policy_low_intensity", "low_intensity",
           "Plots/combined_policy_dashboard_with_utility_flow_cost_both.png",
           _replot_low_intensity,
           "Policy mix trajectories to 2050"),
    Figure(6, False, "preferences", "single_experiment",
           "Plots/preferences.png", _replot_preferences,
           "Histograms of consumer preference parameters"),
    Figure(7, False, "used_car", "used_car_capacity",
           "Plots/ev_prop_single_plot_dated.png", _replot_used_car_capacity,
           "EV uptake vs used car market capacity"),

    # ---- supplementary ---------------------------------------------------
    Figure(1, True, "inference", "posterior",
           "Plots/fig1_posterior_a_chi_b_chi.png", _replot_posterior,
           "NPE posterior density for a_chi, b_chi"),
    Figure(2, True, "Costs_Electricity_Gasoline", None, None, None,
           "EIA electricity vs gasoline prices (external data)"),
    Figure(3, True, "Emissions_Electricity_Gasoline", None, None, None,
           "EIA electricity vs gasoline emissions (external data)"),
    Figure(4, True, "NK", None, None, None,
           "NK landscape value distribution (external)"),
    Figure(5, True, "calibration_cars", "calibration",
           "Plots/multi_seed_2d_scatter.png", _replot_calibration,
           "Simulated vs real ICE/EV price and range"),
    Figure(6, True, "sensitivity", "local_sensitivity",
           "Plots/fig6_local_sensitivity_combined.png", _replot_local_sensitivity,
           "10-panel local sensitivity of EV uptake",
           out_folder=_local_sensitivity_output_folder),
    Figure(7, True, "sobol_first", "sobol",
           _sobol_artifact("First"), _replot_sobol,
           "Sobol first-order indices"),
    Figure(8, True, "sobol_total", "sobol",
           _sobol_artifact("Total"), _replot_sobol,
           "Sobol total-order indices"),
    Figure(9, True, "BAU_sen_elec", "bau_grid",
           "Plots/fig9_bau_timeseries.png", _replot_bau_timeseries,
           "BAU sensitivity to grid intensity and electricity price"),
    Figure(10, True, "BAU_sen_elec_elas", "bau_grid",
           "elasticity_comparison.png", _replot_bau_elasticity,
           "BAU elasticities to grid intensity and electricity price"),
    Figure(11, True, "WTP_sensitivity_carbon_price", "grid_beta_carbon",
           "policy_surface_heatmap_with_BAU.png", _policy_grid_replotter("grid_bau_beta"),
           "EV uptake, beta multiplier x carbon price"),
    Figure(12, True, "WTP_sensitivity_adoption_subsidy", "grid_beta_rebate",
           "policy_surface_heatmap_with_BAU.png", _policy_grid_replotter("grid_bau_beta"),
           "EV uptake, beta multiplier x new car rebate"),
    Figure(13, True, "a_chi_carbon_price", "grid_achi_carbon",
           "policy_surface_heatmap_with_BAU.png", _policy_grid_replotter("grid_bau_achi"),
           "EV uptake, a_chi x carbon price"),
    Figure(14, True, "a_chi_adoption_subsidy", "grid_achi_rebate",
           "policy_surface_heatmap_with_BAU.png", _policy_grid_replotter("grid_bau_achi"),
           "EV uptake, a_chi x new car rebate"),
]


# ---------------------------------------------------------------------------
# 4b. TABLES -- numbers taken out of the same run folders as the figures
# ---------------------------------------------------------------------------

TABLE_ENV = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)
TABLE_LABEL = re.compile(r"\\label\{tab:([^}]+)\}")
TABULAR = re.compile(r"[ \t]*\\begin\{tabular\}\{[^}]*\}.*?\\end\{tabular\}", re.DOTALL)


def _tabular(column_spec, header, rows):
    """Assemble a booktabs tabular block out of already-formatted cells."""
    lines = [r"\begin{tabular}{%s}" % column_spec, r"\toprule",
             (" & ".join(header) + r" \\").lstrip(), r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _intensity_cell(policy, value):
    """
    Policy intensities are stored in the units the model uses: the carbon price
    in $/kgCO2, the electricity subsidy as a fraction, the rest in dollars.
    """
    if policy == "Carbon_price":
        return r"%.0f $\$/TCO_2$" % (value * 1000)
    if policy == "Electricity_subsidy":
        return r"$%.1f\%%$" % (value * 100)
    return r"$\$%.2f$" % value


def _table_policy_outcomes(folder):
    """
    Table 3 -- outcomes of each single policy at the endogenously found minimum
    intensity, for the policies whose mean EV uptake lands in the target band.
    Reads policy_outcomes.pkl from an endog_single_* run.
    """
    from package.analysis.endogenous_policy_intensity_single_plot import policy_titles
    from package.resources.utility import load_object

    outcomes = load_object(_abs(folder) + "/Data", "policy_outcomes")
    if "BAU" not in outcomes:
        raise ValueError("policy_outcomes has no BAU entry")

    on_target, off_target = [], []
    for policy, entry in outcomes.items():
        if policy == "BAU":
            continue
        if MIN_EV_UPTAKE <= entry["mean_EV_uptake"] <= MAX_EV_UPTAKE:
            on_target.append((policy, entry))
        else:
            off_target.append((policy, entry["mean_EV_uptake"]))

    for policy, uptake in off_target:
        print("    [drop] %s: EV uptake %.3f outside %.2f-%.2f"
              % (policy_titles.get(policy, policy), uptake, MIN_EV_UPTAKE, MAX_EV_UPTAKE))
    if not on_target:
        raise ValueError("no policy in this run reached %.2f-%.2f EV uptake"
                         % (MIN_EV_UPTAKE, MAX_EV_UPTAKE))
    for policy, entry in on_target:
        print("    [keep] %s: EV uptake %.3f at intensity %s"
              % (policy_titles.get(policy, policy), entry["mean_EV_uptake"],
                 _intensity_cell(policy, entry["optimized_intensity"])))

    def uptake_cell(policy, entry):
        return "%.3f (%.3f)" % (entry["mean_EV_uptake"], entry["sd_ev_uptake"])

    def intensity_cell(policy, entry):
        if policy == "BAU":
            return "-"
        return _intensity_cell(policy, entry["optimized_intensity"])

    def scaled(key, scale, digits):
        def cell(policy, entry):
            return "%.*f" % (digits, entry[key] * scale)
        return cell

    row_spec = [
        (r"\textbf{EV Adoption Proportion, ($\sigma$)}", uptake_cell),
        (r"\textbf{Intensity}", intensity_cell),
        (r"\textbf{Cumulative Net Cost, bn USD}", scaled("mean_net_cost", 1e-9, 3)),
        (r"\textbf{Cumulative Emissions, MT$CO_2$}",
         scaled("mean_emissions_cumulative", EMISSIONS_SCALE, 2)),
        (r"\textbf{Cumulative Emissions (Driving), MT$CO_2$}",
         scaled("mean_emissions_cumulative_driving", EMISSIONS_SCALE, 2)),
        (r"\textbf{Cumulative Emissions (Production), MT$CO_2$}",
         scaled("mean_emissions_cumulative_production", EMISSIONS_SCALE, 2)),
        (r"\textbf{Cumulative Profit, bn USD}", scaled("mean_profit_cumulative", 1e-9, 3)),
        (r"\textbf{Cumulative Utility (2030), bn USD}",
         scaled("mean_utility_cumulative_30", 1e-9, 3)),
        (r"\textbf{Cumulative Utility (2035), bn USD}",
         scaled("mean_utility_cumulative", 1e-9, 3)),
    ]

    columns = [("BAU", outcomes["BAU"])] + on_target
    header = [""] + [r"\textbf{%s}" % policy_titles.get(p, p) for p, _ in columns]
    rows = [[label] + [cell(p, entry) for p, entry in columns] for label, cell in row_spec]
    return _tabular("l" + "c" * len(columns), header, rows)


def _table_low_intensity_emissions(folder):
    """
    Table 4 -- cumulative emissions of each policy mix at the last time step
    (2050), as a percentage change against BAU. Same quantity the low intensity
    dashboard prints, from the same pickles, in the same policy ordering.
    """
    import numpy as np

    from package.analysis.low_policy_intensity_plot import flip_policy_pair, policy_titles
    from package.resources.utility import load_object

    path = _abs(folder) + "/Data"
    outputs = load_object(path, "outputs")
    outputs_BAU = load_object(path, "outputs_BAU")

    # Same reordering the plot module applies before labelling anything.
    flip_policy_pair(outputs, "Adoption_subsidy_used", "Carbon_price")
    flip_policy_pair(outputs, "Adoption_subsidy", "Carbon_price")

    # The dashboard draws the two single policies alongside the pairs; the table
    # lists them too, so add them the same way.
    outputs[("Carbon_price",)] = load_object(path, "outputs_carbon_tax")
    outputs[("Adoption_subsidy",)] = load_object(path, "outputs_adoption_subsidy")

    def cumulative_final(output):
        return np.nanmean(np.cumsum(output["history_total_emissions"], axis=1)[:, -1])

    bau_final = cumulative_final(outputs_BAU)

    changes = []
    for key, output in outputs.items():
        label = " + ".join(policy_titles.get(p, p) for p in key)
        changes.append((label, (cumulative_final(output) - bau_final) / bau_final * 100))
    changes.sort(key=lambda row: row[1])

    header = [r"\textbf{Policy Combination}", r"\textbf{Change (\%)}"]
    rows = [[label, r"$%s$%.2f" % ("-" if change < 0 else "+", abs(change))]
            for label, change in changes]
    return _tabular("lc", header, rows)


class Table:
    def __init__(self, number, label, run_key, build, caption):
        self.number = number
        self.label = label            # \label{tab:<label>} in manuscript.tex
        self.run_key = run_key        # key into RUNS
        self.build = build            # callable(folder) -> list of tex lines
        self.caption = caption        # short human description, for --list

    @property
    def key(self):
        return "T%d" % self.number

    def folders(self):
        return _run_folders(self.run_key)

    def source_folder(self):
        folders = self.folders()
        return folders[0] if folders else None


TABLES = [
    Table(3, "policy_outcomes", "single_policy_endog", _table_policy_outcomes,
          "Single policy outcomes at the minimum intensity reaching the uptake band"),
    Table(4, "emissions", "low_intensity", _table_low_intensity_emissions,
          "Cumulative emissions change against BAU at the final time step"),
]


def build_tables(tables, dry_run=False, write_tex=True):
    """Recompute each table and rewrite its tabular block. Returns (built, skipped)."""
    built, skipped = [], []

    for table in tables:
        folder = table.source_folder()
        if not folder:
            print("  [skip] Table %s: RUNS['%s'] is empty" % (table.key, table.run_key))
            skipped.append(table)
            continue
        if not os.path.isdir(_abs(folder)):
            print("  [skip] Table %s: folder not found: %s" % (table.key, folder))
            skipped.append(table)
            continue

        print("  [build] Table %s <- %s" % (table.key, folder))
        try:
            lines = table.build(folder)
        except Exception as error:  # keep going; report at the end
            print("  [FAIL] Table %s: %s" % (table.key, error))
            skipped.append(table)
            continue

        for line in lines:
            print("      " + line)
        built.append((table, lines))

    if write_tex and built:
        print("\nUpdating LaTeX tables")
        rewrite_tables(MANUSCRIPT_TEX, built, dry_run=dry_run)

    return [table for table, _ in built], skipped


def rewrite_tables(tex_path, built, dry_run=False):
    """Replace the tabular block of each rebuilt table, matched by its label."""
    if not os.path.isfile(tex_path):
        print("  [skip] %s not found" % os.path.basename(tex_path))
        return 0

    by_label = {table.label: lines for table, lines in built}
    with open(tex_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    changed = []

    def fix_environment(match):
        block = match.group(0)
        label_match = TABLE_LABEL.search(block)
        if not label_match:
            return block
        lines = by_label.get(label_match.group(1))
        if lines is None:
            return block
        tabular = TABULAR.search(block)
        if tabular is None:
            print("  [warn] tab:%s: no tabular block to replace" % label_match.group(1))
            return block
        indent = re.match(r"[ \t]*", tabular.group(0)).group(0)
        inner = "\t" if "\t" in indent else "    "
        replacement = "\n".join(
            indent + ("" if line.startswith(r"\begin{tabular}")
                      or line.startswith(r"\end{tabular}") else inner) + line
            for line in lines)
        if replacement == tabular.group(0):
            return block
        changed.append(label_match.group(1))
        return block[:tabular.start()] + replacement + block[tabular.end():]

    updated = TABLE_ENV.sub(fix_environment, original)

    name = os.path.basename(tex_path)
    if not changed:
        print("  %s: tables already up to date" % name)
        return 0

    for label in changed:
        print("  %s: tab:%s rebuilt" % (name, label))

    if not dry_run:
        # rewrite_tex may have written the backup already, in which case keep
        # it: it is the version from before this run touched anything.
        if not os.path.isfile(tex_path + ".bak"):
            shutil.copyfile(tex_path, tex_path + ".bak")
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write(updated)
        print("  %s: rewritten (%d table(s), backup at %s.bak)" % (name, len(changed), name))

    return len(changed)


def check_unmapped_tables(tex_path, tables):
    """Warn about table environments in the .tex the manifest does not cover."""
    if not os.path.isfile(tex_path):
        return
    known = {table.label for table in tables}
    with open(tex_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    for block in TABLE_ENV.findall(content):
        label_match = TABLE_LABEL.search(block)
        if label_match and label_match.group(1) not in known:
            print("  [note] %s: tab:%s is not rebuilt from a run"
                  % (os.path.basename(tex_path), label_match.group(1)))


# ---------------------------------------------------------------------------
# 5. BUILD
# ---------------------------------------------------------------------------

def _plot_group_key(figure):
    """Figures sharing a run and a plotter (e.g. Sobol 7/8) only replot once."""
    return (figure.run_key, getattr(figure.replot, "__name__", None))


def build_figures(figures, skip_plot=False, dry_run=False):
    """Re-plot and copy. Returns (copied, skipped) lists of Figure."""
    copied, skipped = [], []
    already_plotted = set()

    for figure in figures:
        folders = figure.folders()

        if not figure.is_static and not folders:
            print(f"  [skip] Figure {figure.key}: RUNS['{figure.run_key}'] is empty")
            skipped.append(figure)
            continue

        missing = [f for f in folders if not os.path.isdir(_abs(f))]
        if missing:
            print(f"  [skip] Figure {figure.key}: folder not found: {missing[0]}")
            skipped.append(figure)
            continue

        if not figure.is_static and not skip_plot:
            group = _plot_group_key(figure)
            if group not in already_plotted:
                already_plotted.add(group)
                print(f"  [plot] Figure {figure.key}: {figure.replot.__name__}")
                if not dry_run:
                    try:
                        argument = folders if figure.run_key == "local_sensitivity" else folders[0]
                        figure.replot(argument)
                    except Exception as error:  # keep going; report at the end
                        print(f"  [FAIL] Figure {figure.key} plotting: {error}")
                        skipped.append(figure)
                        continue
                    finally:
                        plt.close("all")

        source = figure.source_path()
        if source is None or not os.path.isfile(source):
            where = source or f"docs/paper/static_figs/<{figure.label}>"
            print(f"  [skip] Figure {figure.key}: no PNG at {os.path.relpath(where, REPO_ROOT)}")
            skipped.append(figure)
            continue

        if os.path.abspath(source) == os.path.abspath(figure.dest_path):
            print(f"  [keep] Figure {figure.key}: already at {figure.tex_path}")
        else:
            print(f"  [copy] Figure {figure.key} <- {os.path.relpath(source, REPO_ROOT)}")
            if not dry_run:
                os.makedirs(os.path.dirname(figure.dest_path), exist_ok=True)
                shutil.copyfile(source, figure.dest_path)
        copied.append(figure)

    return copied, skipped


# ---------------------------------------------------------------------------
# 6. TEX REWRITING
# ---------------------------------------------------------------------------

FIGURE_ENV = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL)
LABEL = re.compile(r"\\label\{fig:([^}]+)\}")
INCLUDEGRAPHICS = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]*)(\})")


def rewrite_tex(tex_path, figures, dry_run=False):
    """Point each \\includegraphics at its figure's new path, matched by \\label."""
    if not os.path.isfile(tex_path):
        print(f"  [skip] {os.path.basename(tex_path)} not found")
        return 0

    by_label = {figure.label: figure for figure in figures}
    with open(tex_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    changes = []

    def fix_environment(match):
        block = match.group(0)
        label_match = LABEL.search(block)
        if not label_match:
            return block
        figure = by_label.get(label_match.group(1))
        if figure is None:
            return block

        def fix_include(include_match):
            old = include_match.group(2)
            new = figure.tex_path
            if old != new:
                changes.append((label_match.group(1), old, new))
            return include_match.group(1) + new + include_match.group(3)

        return INCLUDEGRAPHICS.sub(fix_include, block)

    updated = FIGURE_ENV.sub(fix_environment, original)

    name = os.path.basename(tex_path)
    if not changes:
        print(f"  {name}: paths already correct")
        return 0

    for label, old, new in changes:
        print(f"  {name}: fig:{label}  {old}  ->  {new}")

    if not dry_run:
        shutil.copyfile(tex_path, tex_path + ".bak")
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write(updated)
        print(f"  {name}: rewritten ({len(changes)} paths, backup at {name}.bak)")

    return len(changes)


def check_unmapped(tex_path, figures):
    """Warn about figure environments in the .tex that the manifest misses."""
    if not os.path.isfile(tex_path):
        return
    known = {figure.label for figure in figures}
    with open(tex_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    for block in FIGURE_ENV.findall(content):
        label_match = LABEL.search(block)
        if not label_match:
            print(f"  [warn] {os.path.basename(tex_path)}: figure with no \\label{{fig:...}}")
        elif label_match.group(1) not in known:
            print(f"  [warn] {os.path.basename(tex_path)}: fig:{label_match.group(1)} not in the manifest")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def print_manifest():
    for supplementary, heading in ((False, "MANUSCRIPT"), (True, "SUPPLEMENTARY")):
        print(f"\n{heading}")
        print(f"  {'Fig':<5}{'status':<10}{'destination':<38}source")
        for figure in (f for f in FIGURES if f.supplementary == supplementary):
            source = figure.source_path()
            if source is None:
                if figure.is_static:
                    status = "no source"
                    shown = f"put {STATIC_SOURCES.get(figure.label, '?')} in docs/paper/static_figs/"
                else:
                    status, shown = "unset", f"RUNS['{figure.run_key}'] is empty"
            elif os.path.isfile(source):
                status, shown = "ready", os.path.relpath(source, REPO_ROOT)
            else:
                folders = figure.folders()
                exists = all(os.path.isdir(_abs(f)) for f in folders) if folders else False
                status = "to plot" if exists else "missing"
                shown = os.path.relpath(source, REPO_ROOT)
            print(f"  {figure.key:<5}{status:<10}{figure.tex_path:<38}{shown}")
            print(f"  {'':<5}{'':<10}{figure.caption}")

    print("\nTABLES (manuscript.tex)")
    print(f"  {'Tab':<5}{'status':<10}{'label':<38}source")
    for table in TABLES:
        folder = table.source_folder()
        if not folder:
            status, shown = "unset", f"RUNS['{table.run_key}'] is empty"
        elif os.path.isdir(_abs(folder)):
            status, shown = "ready", folder
        else:
            status, shown = "missing", folder
        print(f"  {table.key:<5}{status:<10}{'tab:' + table.label:<38}{shown}")
        print(f"  {'':<5}{'':<10}{table.caption}")


def parse_selection(text):
    """Split a --only list into the figures and the tables it names."""
    wanted = {token.strip().upper() for token in text.split(",") if token.strip()}
    figures = [figure for figure in FIGURES if figure.key.upper() in wanted]
    tables = [table for table in TABLES if table.key.upper() in wanted]
    known = {figure.key.upper() for figure in figures} | {table.key.upper() for table in tables}
    unknown = wanted - known
    if unknown:
        sys.exit(f"Unknown figure(s)/table(s): {', '.join(sorted(unknown))}")
    return figures, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the manifest and exit")
    parser.add_argument("--only", default=None, help="comma-separated figure keys, e.g. 3,S6,S11")
    parser.add_argument("--skip-plot", action="store_true", help="reuse PNGs already in the results folders")
    parser.add_argument("--no-tex", action="store_true", help="do not touch the .tex files")
    parser.add_argument("--no-tables", action="store_true", help="build the figures only")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    if args.list:
        print_manifest()
        return

    figures, tables = parse_selection(args.only) if args.only else (FIGURES, TABLES)
    if args.no_tables:
        tables = []

    copied, skipped = [], []
    if figures:
        print("Building figures")
        copied, skipped = build_figures(figures, skip_plot=args.skip_plot, dry_run=args.dry_run)

    if figures and not args.no_tex:
        print("\nUpdating LaTeX paths")
        main_figures = [f for f in copied if not f.supplementary]
        supp_figures = [f for f in copied if f.supplementary]
        rewrite_tex(MANUSCRIPT_TEX, main_figures, dry_run=args.dry_run)
        rewrite_tex(SUPPLEMENTARY_TEX, supp_figures, dry_run=args.dry_run)
        check_unmapped(MANUSCRIPT_TEX, [f for f in FIGURES if not f.supplementary])
        check_unmapped(SUPPLEMENTARY_TEX, [f for f in FIGURES if f.supplementary])

    built_tables, skipped_tables = [], []
    if tables:
        print("\nBuilding tables")
        built_tables, skipped_tables = build_tables(
            tables, dry_run=args.dry_run, write_tex=not args.no_tex)
        check_unmapped_tables(MANUSCRIPT_TEX, TABLES)

    print(f"\n{len(copied)} figure(s) in place, {len(skipped)} skipped.")
    if tables:
        print(f"{len(built_tables)} table(s) rebuilt, {len(skipped_tables)} skipped.")
    if skipped or skipped_tables:
        print("Skipped: " + ", ".join(item.key for item in skipped + skipped_tables))
        print("Fill in the matching RUNS entry (or docs/paper/static_figs/) and re-run.")


if __name__ == "__main__":
    main()
