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
uv run python -m package.generating_data.multi_seed_gen
```

Each `*_gen.py` script writes a timestamped directory under `results/` and most
call their matching `*_plot.py` immediately afterwards. `results/` is
git-ignored, so a fresh clone reproduces outputs by re-running the generators.

To explore a single run interactively, use `model_playground.ipynb`: load a
`base_params` dict from `package/constants/`, call `generate_data(base_params)`,
and inspect the returned controller.

---

## Reproducing the paper

Run from the repository root. Multi-seed experiments use 64 Monte Carlo
replications and are parallelised over available cores via `joblib`; the policy
experiments are the expensive ones.

| Paper output | Generate | Plot | Config |
|---|---|---|---|
| Fig. 2: calibration 2001–2023 (EV uptake and sales, prices, HHI, car age) | `generating_data.calibration_gen` | `plotting_data.calibration_plot` | `base_params_multi_seed.json` |
| Fig. 3: single-instrument grid search (100 intensities per instrument) | `analysis.vary_single_policy_gen` | `analysis.vary_single_policy_plot` | `base_params_vary_single_policy_gen.json`, `analysis/policy_bounds_vary_single_policy_gen.json` |
| Table 3: BAU baseline | `analysis.BAU_outcomes_gen` | n/a | `base_params_endogenous_policy_pair_gen.json` |
| Table 3: minimum single-policy intensity reaching 95% uptake | `analysis.endogenous_policy_intensity_single_gen` | n/a | `base_params_endogenous_policy_single_gen.json`, `analysis/policy_bounds_endog_single_gen.json` |
| Fig. 4: policy pairs achieving 94–96% uptake | `analysis.endogenous_policy_intensity_pair_gen` | `analysis.endogenous_policy_intensity_pair_plot` | `base_params_endogenous_policy_pair_gen.json`, `analysis/policy_bounds_vary_pair_policy_gen.json` |
| Fig. 5, Table 4: trajectories to 2050 after policy removal | `analysis.low_policy_intensity_gen` | `analysis.low_policy_intensity_plot` | reads a pair-analysis results directory (see note below) |
| Fig. 6: parameter distribution histograms | `generating_data.single_experiment_gen` | `plotting_data.single_experiment_plot` | inline `base_params` dict in the script |
| Fig. 7: EV uptake vs used car market capacity | `generating_data.sen_vary_single_param_gen_second_hand_cars` | `plotting_data.sen_vary_single_param_plot_second_hand_cars` | `base_params_vary_single.json`, `vary_single_max_num_cars_prop.json` |
| SM: Sobol global sensitivity analysis | `generating_data.sensitivity_analysis_calibration_gen` | `plotting_data.sensitivity_analysis_calibration_plot` | `base_params_SA.json`, `variable_parameters_dict_SA.json` |
| SM: BAU sensitivity (grid decarbonisation, electricity prices) | `generating_data.inputs_and_emissions_gen` | `plotting_data.inputs_and_emissions_plot` | `base_params_inputs_and_emissions.json`, `vary_sen_decarb.json`, `vary_sen_elec_price.json` |
| SM: single-parameter robustness sweeps | `generating_data.sen_vary_single_param_gen` | `plotting_data.sen_vary_single_param_plot` | `base_params_vary_single.json`, `vary_single_*.json` |
| Calibration of the innovativeness-threshold beta distribution (SBI) | `calibration.NN_multi_round_calibration_multi_gen` | `calibration.NN_multi_round_calibration_multi_plot` | `base_params_NN.json` |
| Distance-driven distribution fit | `calibration.fit_distance` | n/a | `package/calibration_data/` |

**Ordering note.** The stages are sequential: the single-instrument analysis
produces the intensity bounds used by the pair analysis, and
`low_policy_intensity_gen` consumes a pair-analysis output directory. Its
`fileNames` argument is a hardcoded local `results/` path, so set it to your own
pair-analysis directory before running it.

Policy intensities are found by Bayesian optimisation over a Gaussian-process
surrogate (`skopt.gp_minimize`), maximising expected improvement against the 95%
uptake target with a 1% tolerance.

Note that `single_experiment_gen.py` defines its parameters as an inline dict
rather than loading a JSON config; it is the most convenient place to read off the
full parameter set of Appendix B in one piece.

Not tied to a paper figure: `multi_seed_gen`/`multi_seed_plot` and
`multi_seed_gen_cars`/`multi_seed_plot_cars` are multi-seed diagnostic dashboards
and car-attribute scatter/contour plots, `policy_sensitivity_gen`,
`vary_single_param_gen` and `battery_cost_sen_gen` are further sweeps
(behavioural parameters under policy, carbon tax, battery cost correlation).

---

## Repository layout

```text
├── model_playground.ipynb              # Interactive single-run exploration
├── pyproject.toml / uv.lock            # Dependencies (uv)
├── docs/code_narrative.tex / .pdf      # Extended walkthrough of the model code
└── package/
    ├── model/                          # Core agent-based model
    ├── analysis/                       # Policy experiments (paper Section 4)
    ├── generating_data/                # Calibration, sensitivity and sweep runners
    ├── plotting_data/                  # Figure scripts for the above
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
divided into three phases by timestep count: `duration_burn_in` (180, ICEV-only),
`duration_calibration` (276, 2001–2023) and `duration_future` (144, the 2024–2035
policy period, extended to 2050 for the stability analysis).

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
anticipation parameter.

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

