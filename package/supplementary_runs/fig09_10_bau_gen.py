"""
Reproduces Figures 9 and 10 of the supplementary material from a single
generation run: BAU EV uptake/emissions over the extended policy period
(2024-2050) crossed against grid-decarbonisation and electricity-price
scenarios.

Generation is package/generating_data/inputs_and_emissions_gen.py::run_physical_duo,
unchanged, run against:
  - package/constants/vary_sen_decarb.json: Grid_emissions_intensity in
    [0.1, 0.5, 0.75, 1] (90% / 50% / 25% / 0% reduction -- the first three
    are Figure 9's columns, the last is the baseline Figure 10's elasticity
    is measured relative to)
  - package/constants/vary_sen_elec_price.json: Electricity_price in
    [0.5, 1, 1.5] (50% decrease / no change / 50% increase -- Figure 9's
    three line colors, and Figure 10's baseline + both price magnitudes)
One run of this grid serves both figures; the two plotting scripts just
select different subsets of it.
"""
from package.generating_data.inputs_and_emissions_gen import run_physical_duo
from package.supplementary_runs.fig09_bau_timeseries_plot import plot_fig9_bau_timeseries
from package.supplementary_runs.fig10_bau_elasticity_plot import plot_fig10

PHYSICAL_VARS = [
    "package/constants/vary_sen_decarb.json",
    "package/constants/vary_sen_elec_price.json",
]
BASE_PARAMS_LOAD = "package/constants/base_params_inputs_and_emissions.json"


def run_fig9_10(base_params_load=BASE_PARAMS_LOAD, physical_vars=PHYSICAL_VARS):
    folder_name = run_physical_duo(BASE_PARAMS_PATH=base_params_load, VAR_PATHS=physical_vars)
    plot_fig9_bau_timeseries(folder_name)
    plot_fig10(folder_name)
    return folder_name


if __name__ == "__main__":
    run_fig9_10()
