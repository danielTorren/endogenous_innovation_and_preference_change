
# 🚗 EV Transition Simulation Framework

This repository simulates vehicle market transitions under various policy, technology, and consumer behavior scenarios. It models the interaction of consumers, firms, and second-hand markets in the adoption of electric vehicles (EVs) using agent-based modeling and NK fitness landscapes.

---

## 📁 Project Structure

```text
├── model_playground.ipynb        # Jupyter notebook for testing simulation interactively
├── requirements.txt              # Python dependencies
├── package/
│   ├── analysis/                 # Scripts for batch simulation generation and plotting policy scenarios
│   ├── model/                    # Core agent-based models: vehicles, users, network, firms
│   ├── plotting_data/           # Plotting utilities for time series and simulation outcomes
│   ├── resources/               # Simulation entry points and helpers
│   └── calibration/             # Calibration scripts and data for empirical matching
```

---

## 🔍 File Descriptions by Folder

### `package/analysis/`
Scripts for running and visualizing multiple policy scenarios.

- `endogenous_policy_intensity_pair_gen.py`: Runs experiments varying two policy levers jointly.
- `endogenous_policy_intensity_pair_plot.py`: Plots results from paired policy simulations.
- `endogenous_policy_intensity_single_gen.py`: Simulates one adaptive policy at a time.
- `low_policy_intensity_gen.py`: Baseline run with minimal policy intensity.
- `low_policy_intensity_plot.py`: Visualizes minimal policy intensity outputs.
- `vary_single_policy_gen.py`: Sweeps one policy variable to isolate its effect.
- `vary_single_policy_plot.py`: Plots the results of those single-policy sweeps.

### `package/calibration/`
Files for calibrating the model to match real-world EV adoption.

- `calibration_data_inputs.py`: Loads empirical EV data.
- `calibration_data_outputs.py`: Outputs formatted calibration results.
- `fit_distance.py`: Calculates the fit for distance driven distribution from empirical data.
- `NN_multi_round_calibration_multi_gen.py`: Runs iterative NN-based calibration.
- `NN_multi_round_calibration_multi_plot.py`: Plots results of calibration.

### `package/generating_data/`
Scripts to create and manage different simulation types.

- `sensitivity_analysis_calibration_gen.py`: Sensitivity analysis for calibration params.
- `sen_vary_single_param_gen.py`: Changes one parameter at a time to test robustness.
- `single_experiment_gen.py`: Runs a one-off simulation with chosen params.
- `single_experiment_multi_seed_gen.py`: Repeats the same config with varied random seeds.

### `package/model/`
Core agent-based simulation logic.

- `carModel.py`: Defines firm-created car templates.
- `centralizedIdGenerator.py`: Provides unique IDs across the simulation.
- `controller.py`: Master orchestrator for simulation state.
- `firm.py`: Describes firm behavior like production/innovation.
- `firmManager.py`: Controls firms.
- `nkModel_EV.py` / `nkModel_ICE.py`: EV/ICE-specific NK landscape implementations.
- `personalCar.py`: Tracks attributes of owned vehicles.
- `secondHandMerchant.py`: Handles pricing and turnover in resale markets.
- `socialNetworkUsers.py`: Models peer influence on EV consideration adn car selection.
- `VehicleUser.py`: Holds basic user information.

### `package/plotting_data/`
Reusable visualizations for model results.

- `sensitivity_analysis_calibration_plot.py`: Sensitivity output visualizer.
- `sen_vary_single_param_plot.py`: Graphs from single-parameter tests.
- `single_experiment_multi_seed_plot.py`: Aggregates plots from multi-seed runs.
- `single_experiment_plot.py`: Main plotting script for a single run.

### `package/resources/`
Helpers and entry points.

- `run.py`: Entrypoint that defines the `generate_data()` function.
- `utility.py`: General-purpose tools (saving, naming, directories).

---

## 🧪 Running a Simulation

You can run a simulation using the included Jupyter notebook `model_playground.ipynb`:

1. Load and optionally modify `base_params`.
2. Call `generate_data(base_params)` to simulate.
4. Review EV adoption, emissions, utility, and more.

Or use one of the `*_gen.py` scripts for batch or automated experiments, with the associated `*_plot.py` file.

## 📦 Requirements

Install with:

```bash
pip install -r requirements.txt
```
