# Driving in the wrong direction? Modelling policy mixes for EV adoption

Replication code for an agent-based model (ABM) of the transition from internal
combustion engine vehicles (ICEVs) to electric vehicles (EVs), calibrated on
California, 2001–2023.

The model couples four submodules so that consumer adoption and firm innovation
co-evolve:

- **Discrete choice consumption**: heterogeneous car users compare the lifecycle
  utility (quality, range, lifecycle costs, lifecycle emissions) of their own car
  against new and used alternatives, and choose via a logit model.
- **Social imitation**: users only *consider* EVs once the share of EV owners in
  their Watts–Strogatz network neighbourhood passes an idiosyncratic
  innovativeness threshold.
- **Directed innovation**: manufacturers search parallel NK landscapes for ICEVs
  and EVs, choosing research direction and product mix by expected profit across
  consumer segments.
- **Used car market**: a single consolidated dealer prices used cars off the most
  similar new car with age-based depreciation, and scraps cars below scrap value.

Policy experiments evaluate five market-based instruments (carbon price,
electricity subsidy, new EV rebate, used EV rebate, and production subsidy)
individually and in pairs, over 2024–2035, with a post-policy projection to 2050.

---

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python
3.13+.

```bash
uv sync
```

This installs a CPU-only build of PyTorch (only `sbi` needs it, for the
simulation-based-inference calibration) rather than the default CUDA build; see
the comments in `pyproject.toml`.

## Running

All scripts resolve config paths relative to the repository root and use absolute
imports, so run them as modules from the root:

```bash
uv run python -m package.generating_data.calibration_gen
```

Each `*_gen.py` script writes a timestamped directory under `results/` and most
call their matching `*_plot.py` immediately afterwards. `results/` is
git-ignored, so a fresh clone reproduces outputs by re-running the generators.

To explore a single run interactively, use `model_playground.ipynb`: load a
`base_params` dict from `package/constants/`, call `generate_data(base_params)`,
and inspect the returned controller.

---

## Reproducing the paper

Reproduction is two steps. First run the generator for a figure, which leaves a
timestamped folder under `results/`. Then paste that folder name into the `RUNS`
dict at the top of `package/paper_figures/build_figures.py` and run the builder,
which re-plots from the results folder, renumbers each PNG to its number in the
paper, and recomputes the two results tables:

```bash
uv run python -m package.paper_figures.build_figures --list          # what is ready, what is missing
uv run python -m package.paper_figures.build_figures --only 3,S6     # build a subset
```

`--list` is the quickest way to see which `RUNS` entries are still empty. The
builder writes into the manuscript directory (`docs/paper/`), which is not part
of this repository — without it, use `--no-tex` to produce the renumbered PNGs
alone, or take each figure straight from its results folder. See
`package/paper_figures/README.md` for the flags and the full figure manifest.

Runs use 64 Monte Carlo seeds and are parallelised over available cores with
`joblib`; the policy experiments are the expensive ones. Every `*.slurm` file
next to a generator is the cluster version of the same command.

### Manuscript figures and tables

| Paper output | Generate | Plot | Config |
|---|---|---|---|
| Fig. 2: calibration 2001–2023 (EV uptake and sales, prices, HHI, car age) | `generating_data.calibration_gen` | `plotting_data.calibration_plot` | `base_params_calibration.json` |
| Fig. 3: single-instrument grid search (100 intensities per instrument) | `analysis.vary_single_policy_gen` | `analysis.vary_single_policy_plot` | `base_params_vary_single_policy_gen.json`, `analysis/policy_bounds_vary_single_policy_gen.json` |
| Table 3: minimum single-policy intensity reaching 95% uptake, and its outcomes | `analysis.endogenous_policy_intensity_single_gen` | `analysis.endogenous_policy_intensity_single_plot` | `base_params_endogenous_policy_single_gen.json`, `analysis/policy_bounds_endog_single_gen.json` |
| Fig. 4: policy pairs achieving 94–96% uptake | `analysis.endogenous_policy_intensity_pair_gen` | `analysis.endogenous_policy_intensity_pair_plot` | `base_params_endogenous_policy_pair_gen.json`, `analysis/policy_bounds_vary_pair_policy_gen.json` |
| Fig. 5, Table 4: trajectories to 2050 after policy removal | `analysis.low_policy_intensity_gen` | `analysis.low_policy_intensity_plot` | reads pair- and single-analysis results folders (see note below) |
| Fig. 6: parameter distribution histograms | `generating_data.single_experiment_gen` | `plotting_data.single_experiment_plot` | inline `base_params` dict in the script |
| Fig. 7: EV uptake vs used car market capacity | `generating_data.vary_single_param_gen` | `plotting_data.vary_single_param_plot` | `base_params_vary_single.json`, `vary_single_max_num_cars_prop.json` |
| BAU reference outcomes | `analysis.BAU_outcomes_gen` | n/a | `base_params_endogenous_policy_pair_gen.json` |

Figure 1 is a hand-drawn model diagram, not model output.

### Supplementary figures

`package/supplementary_runs/` reproduces the supplementary material against the
current `package/constants/base_params_calibration.json`. Every script there is a
thin wrapper around generation and plotting code that already exists elsewhere in
`package/`; each file's docstring names the function it calls.

| Figure | What it shows | Script | Config |
|---|---|---|---|
| S1 | NPE posterior density for `a_chi`, `b_chi` | `calibration.sbi_single_seed_gen`, plotted by `supplementary_runs.fig01_posterior_plot` | `base_params_NN.json` |
| S5 | Simulated vs real-world ICE/EV price and range | `supplementary_runs.fig05_calibration_cars_gen` | `base_params_calibration.json` |
| S6 | 10-panel local sensitivity of EV uptake | `supplementary_runs.fig06_panels` (resumable) or `fig06_local_sensitivity_gen`, then `fig06_local_sensitivity_plot` | `base_params_vary_single.json`, `vary_single_*.json` |
| S7, S8 | Sobol first- and total-order indices, 6 outputs × 10 parameters | `supplementary_runs.fig07_08_sobol_gen` | `base_params_SA.json`, `variable_parameters_dict_SA.json` |
| S9, S10 | BAU EV uptake, emissions and elasticities over grid decarbonisation × electricity price | `supplementary_runs.fig09_10_bau_gen`, then `fig09_bau_timeseries_plot` and `fig10_bau_elasticity_plot` | `base_params_inputs_and_emissions.json`, `vary_sen_decarb.json`, `vary_sen_elec_price.json` |
| S11–S14 | EV uptake in 2035 over a behavioural parameter × a policy intensity | `supplementary_runs.fig11_14_policy_grid_gen <11\|12\|13\|14>` | `base_params_vary_policy_joint.json`, `vary_policy_*.json` |

Figures S2–S4 are external data and NK-landscape illustrations, not model output.
`package/supplementary_runs/README.md` has the run counts, wall-clock estimates,
memory sizing and the SLURM jobs, including how to resume a partial Figure S6 and
how to reuse one BAU sweep across a pair of policy grids.

### Ordering note

The policy stages are sequential: the single-instrument analysis produces the
intensity bounds used by the pair analysis, and `low_policy_intensity_gen`
consumes the output folders of both. Paste those folder names into the
`ENDOG_PAIR` and `ENDOG_SINGLE` constants at the top of
`package/analysis/low_policy_intensity_gen.py`, or pass them on the command line:

```bash
uv run python -m package.analysis.low_policy_intensity_gen \
    endog_pair_<timestamp> --single-policy endog_single_<timestamp>
```

Policy intensities are found by Bayesian optimisation over a Gaussian-process
surrogate (`skopt.gp_minimize`), maximising expected improvement against the 95%
uptake target with a 1% tolerance.

Note that `single_experiment_gen.py` defines its parameters as an inline dict
rather than loading a JSON config; it is the most convenient place to read off the
full parameter set of Appendix B in one piece.

### Not tied to a paper output

`generating_data/` also holds exploratory sweeps kept for reference:
`policy_sensitivity_gen`, `sen_vary_single_param_gen` (and its
`_second_hand_cars` variant), `battery_cost_sen_gen`, `sweep_hhi_age_gen`,
`ablation_gen`, `burn_in_ablation_gen`, `delta_carbon_price_gen`, and the
single-parameter sweeps `a_chi_sweep_gen`, `b_chi_grid_search_gen`,
`delta_sweep_gen`, `kappa_sweep_gen` and `seed_inputs_sweep_gen`.
`analysis/policy_dominance.py` compares policy pairs across result folders, and
`calibration/NN_multi_round_calibration_multi_gen.py` is the earlier multi-round
form of the SBI calibration that `sbi_single_seed_gen.py` replaced.

---

## Repository layout

```text
├── model_playground.ipynb              # Interactive single-run exploration
├── pyproject.toml / uv.lock            # Dependencies (uv)
├── docs/
│   ├── code_narrative.tex / .pdf       # Extended walkthrough of the model code
│   └── forward_looking/                # Note on the forward-looking expectations extension
└── package/
    ├── model/                          # Core agent-based model
    ├── analysis/                       # Policy experiments (paper Section 4)
    ├── generating_data/                # Calibration, sensitivity and sweep runners
    ├── plotting_data/                  # Figure scripts for the above
    ├── paper_figures/                  # Renumbers figures and tables into the manuscript
    ├── supplementary_runs/             # Reproduces the supplementary figures
    ├── calibration/                    # Empirical data loading and SBI calibration
    ├── calibration_data/               # California input data (see Data sources)
    ├── constants/                      # base_params_*.json and vary_*.json configs
    └── resources/                      # Run harness and I/O helpers
```

### `package/model/`

| File | Role |
|---|---|
| `controller.py` | Orchestrates the monthly update sequence, exogenous input paths, and policy application |
| `socialNetworkUsers.py` | Car users: choice set construction, logit choice, imitation threshold update |
| `firm.py` | A single manufacturer: product mix, pricing, NK innovation |
| `firmManager.py` | Firm population and market segmentation |
| `nkModel_ICE.py`, `nkModel_EV.py` | NK technology landscapes for each drivetrain |
| `carModel.py` | A car design offered for sale (attribute vector plus price) |
| `personalCar.py` | An owned vehicle and its accumulated state |
| `secondHandMerchant.py` | Used car pricing, stock limits and scrapping |
| `VehicleUser.py` | Base user attributes |
| `centralizedIdGenerator.py` | Unique IDs across the simulation |

### `package/resources/`

- `run.py`: `generate_data()` builds and steps a controller for one parameter
  set; `load_in_controller()` resumes a calibrated controller for the policy
  period, so the 2001–2023 burn-in and calibration are computed once and reused
  across policy scenarios.
- `utility.py`: object save/load, run naming, directory creation, worker counts.

### Configuration

Each experiment loads a `base_params_*.json` from `package/constants/`. Runs are
divided into phases by timestep count: `duration_burn_in` (180, ICEV-only),
`duration_calibration` (276, 2001–2023) and `duration_future` (144, the 2024–2035
policy period, extended to 2050 for the stability analysis, and 0 for the
calibration and sensitivity configs that stop in 2023). All configurations use
`seed_repetitions` 64.

Policies live under `parameters_policies`, with `States` switching each instrument
on or off and `Values` giving its intensity. The `vary_*.json` files describe
parameter sweeps consumed by the sensitivity runners.

---

## Extensions present in the code but not used in the paper

The model contains three optional mechanisms that are **disabled by default and
switched off in every configuration shipped in this repository**. They produce no
result reported in the paper, and are retained as a starting point for follow-up
work on command-and-control policy and policy anticipation. Each is guarded, so
omitting its parameter reproduces the published behaviour exactly.

### ICE bans (command-and-control)

The paper deliberately restricts itself to market-based instruments and excludes
command-and-control policy. Three independent ban levers are nevertheless
implemented, each taking effect a given number of months after the burn-in ends
(the same convention as `ev_production_start_time`), and each defaulting to
`None`, meaning "never":

| Parameter | Effect |
|---|---|
| `ICE_research_ban_time` | Firms may no longer research or improve ICE technology. Existing ICE designs stay in firm memory and remain sellable. |
| `ICE_sales_ban_time` | Firms may no longer offer *new* ICE cars. Already-sold ICEVs remain drivable and resellable on the used market. |
| `ICE_driving_ban_time`, `ICE_driving_ban_penalty` | A per-unit cost shock added to effective ICE fuel cost from the ban date onward, reaching second-hand ICEVs too. |

They are additive, so research-only, research+sales, and research+sales+driving
scenarios can all be configured. The driving ban is modelled as a cost shock
rather than a hard prohibition, which means it flows through the existing
utility-driven choice mechanism and needs no new decision rule; the sales and
research bans are hard constraints on firms' choice sets. All three validate that
EV research or production has already begun before the ban bites.

### Forward-looking expectations

`forward_looking_expectations` (default `False`) switches agents away from the
naive expectations used in the paper, under which the current energy price and
emissions intensity are assumed to persist indefinitely and the discounted sum
collapses to the closed-form geometric series of Appendix A. With the flag on,
agents instead discount the actual known future path of fuel costs, electricity
prices and grid carbon intensity.

`controller.compute_discounted_indices()` precomputes present-value indices by
backward recursion once per run, so the switch costs nothing per utility
evaluation. Two subtleties are handled there: the carbon price is extended past
the simulated horizon using its own schedule (so a temporary tax is not
anticipated as a permanent one at its peak rate), while base energy prices and the
decarbonisation trend are extended by holding their final value. With the flag
off the indices are still computed but never read, so behaviour is unchanged.

This is the channel through which forward-looking agents would anticipate an
announced ban or a scheduled carbon price ahead of its arrival, over a horizon
implied by the model's own discount and depreciation rates rather than a separate
anticipation parameter. `docs/forward_looking/` works through the derivation.

### Carbon price ramp and research subsidy

`calculate_growth()` supports `flat`, `linear`, `quadratic` and `exponential`
carbon price paths via `Carbon_price_state`. Every configuration here uses `flat`,
since the paper applies policies at full intensity from January 2024 and holds
them constant, but the ramp is the hook for time-varying schemes such as the EU
ETS.
The ramp is coded to *end* at the close of the policy period rather than persist
at its final value.

A sixth policy lever, `Research_subsidy`, is present in the policy state
dictionary but is not among the five instruments analysed in the paper.

---

## Data sources

`package/calibration_data/` holds the California series used for calibration:
vehicle population and EV sales (California Energy Commission), gasoline and
residential electricity prices, grid emissions intensity, and CPI for conversion
to 2020 US dollars. `calibration_data_inputs.py` assembles these into the pickled
input object the model reads; `calibration_data_outputs.py` formats the observed
targets used for indirect calibration.

Full parameter values, units and sources are tabulated in the paper's Appendix B.
