# Supplementary material reproduction

Reproduces every figure in the paper's supplementary material (1, 5-14)
against the **current** `package/constants/base_params_calibration.json`
(and the other base-params files derived from it).

Every script here is a thin wrapper around existing generation/plotting code
elsewhere in `package/` -- see each file's docstring for exactly which
existing function it calls. Nothing here duplicates simulation logic.

## Figure -> script map

| Figure | What it shows | Script(s) | Slurm |
|---|---|---|---|
| 1 | Posterior density for `a_chi`, `b_chi` | `fig01_posterior_plot.py` | `submit_fig01_posterior_plot.slurm` |
| 5 | Simulated vs. real-world ICE/EV price & range | `fig05_calibration_cars_gen.py` | `submit_fig05_calibration_cars_gen.slurm` |
| 6 | 10-panel local sensitivity (EV uptake) | `fig06_local_sensitivity_gen.py` + `fig06_local_sensitivity_plot.py` | `submit_fig06_local_sensitivity_gen.slurm` |
| 7 | Sobol first-order sensitivity, 6 outputs x 10 params | `fig07_08_sobol_gen.py` | `submit_fig07_08_sobol_gen.slurm` |
| 8 | Sobol total-order sensitivity, 6 outputs x 10 params | `fig07_08_sobol_gen.py` | (same job as Figure 7) |
| 9 | BAU EV uptake/emissions, decarb x elec-price, time series | `fig09_10_bau_gen.py` + `fig09_bau_timeseries_plot.py` | `submit_fig09_10_bau_gen.slurm` |
| 10 | BAU elasticity, decarb x elec-price | `fig09_10_bau_gen.py` + `fig10_bau_elasticity_plot.py` | (same job as Figure 9) |
| 11 | EV uptake 2035, beta multiplier x carbon price | `fig11_14_policy_grid_gen.py "11"` | `submit_fig11_beta_carbon_gen.slurm` |
| 12 | EV uptake 2035, beta multiplier x new car rebate | `fig11_14_policy_grid_gen.py "12"` | `submit_fig12_beta_rebate_gen.slurm` |
| 13 | EV uptake 2035, a_chi x carbon price | `fig11_14_policy_grid_gen.py "13"` | `submit_fig13_achi_carbon_gen.slurm` |
| 14 | EV uptake 2035, a_chi x new car rebate | `fig11_14_policy_grid_gen.py "14"` | `submit_fig14_achi_rebate_gen.slurm` |

Figure 1 has no generation script here: NPE calibration
(`package/calibration/sbi_single_seed_gen.py`, submitted via
`package/calibration/submit_sbi_single_seed.slurm`) is a separate, much
longer-running job that's already been run against the current calibration
parameters (see `results/sbi_single_seed_15_38_23__11_08_2026/`). Re-run that
job only if you need a fresh posterior; otherwise just point
`fig01_posterior_plot.py` at whichever `results/sbi_single_seed_*` folder you
want to plot.

## Submitting

Run everything at once with:

```bash
bash package/supplementary_runs/submit_all.sh
```

which just loops over `sbatch` for the eight jobs below. All eight are
independent (no shared state, no dependencies between them), so they queue
and can run concurrently on the cluster. Figure 1 is deliberately NOT in this
batch -- it does no new simulation, just re-plots an already-existing SBI
posterior, so submit it separately (it finishes in seconds):

```bash
sbatch package/supplementary_runs/submit_fig01_posterior_plot.slurm   # separate, not part of submit_all.sh
sbatch package/supplementary_runs/submit_fig05_calibration_cars_gen.slurm
sbatch package/supplementary_runs/submit_fig06_local_sensitivity_gen.slurm
sbatch package/supplementary_runs/submit_fig07_08_sobol_gen.slurm
sbatch package/supplementary_runs/submit_fig09_10_bau_gen.slurm
sbatch package/supplementary_runs/submit_fig11_beta_carbon_gen.slurm
sbatch package/supplementary_runs/submit_fig12_beta_rebate_gen.slurm
sbatch package/supplementary_runs/submit_fig13_achi_carbon_gen.slurm
sbatch package/supplementary_runs/submit_fig14_achi_rebate_gen.slurm
```

Each job prints its own fresh `results/<name>_<timestamp>` folder near the
top of its log -- that's where the figure PNGs land (see each script's
docstring for the exact filename; most are under `Plots/`, but the Figures
10-14 plotting functions save to the results folder's top level instead,
since that's the existing convention in `policy_sensitivity_plot.py` /
`inputs_and_emissions_plot.py`).

## Runtime / cost

Total ABM run count across all eight jobs (Figure 1 does no new runs -- it
only re-plots an existing posterior):

| Job | Runs | Arithmetic |
|---|---|---|
| Fig 5 | 64 | 1 combo x 64 seeds |
| Fig 6 | 2,560 | 10 params x 4 values x 64 seeds |
| Fig 7/8 | 196,608 | N_samples=256 x (D+2)=12 x 64 seeds, calc_second_order=False |
| Fig 9/10 | 192 | 4 decarb x 3 price x 16 seeds |
| Fig 11 | 3,584 | (8 beta x 6 carbon x 64 seeds) + (8 x 64 BAU) |
| Fig 12 | 3,584 | (8 beta x 6 rebate x 64 seeds) + (8 x 64 BAU) |
| Fig 13 | 3,584 | (8 a_chi x 6 carbon x 64 seeds) + (8 x 64 BAU) |
| Fig 14 | 3,584 | (8 a_chi x 6 rebate x 64 seeds) + (8 x 64 BAU) |
| **Total** | **213,760** | |

At ~20 s/run (measured during smoke-testing -- 8 full-length, 456-step runs
finished in ~21 s on 16 workers, i.e. ~20 s each when there's a free core per
run) that's **~1,188 core-hours** of total compute, almost all of it (~1,092
core-hours) the Sobol job. What that means for wall-clock time depends on how
many jobs the cluster schedules at once, since each job parallelises
internally across its own `--cpus-per-task`:

- **Per-job wall-clock**, i.e. `ceil(runs / cpus-per-task) x 20 s`:
  Fig 5 ~20 s, Fig 6 ~13 min, Fig 7/8 ~8.5 h (128 cores), Fig 9/10 ~4 min,
  each of Fig 11-14 ~19 min.
- **If all 8 jobs get scheduled at once**: wall-clock for the whole batch is
  set by the slowest job, i.e. **Fig 7/8's ~8.5 h**.
- **Worst case, jobs queue one after another**: sum of all eight, ~10.1 h.

Either way this comfortably finishes overnight -- the Sobol job (Fig 7/8) is
now the long pole by a wide margin, everything else finishes in under 20 min.
The `--time` values inside each `.slurm` file are padded generously above
these estimates (up to 24 h) as safe upper bounds against a slow/busy node,
not the expected runtime; trim them from `seff <jobid>` after the first real
run, per this repo's existing convention (see e.g.
`package/car_ban/submit_car_ban.slurm`).

## Figures 7-8: Sobol global sensitivity

Generation and plotting code for these already existed elsewhere in the
repo, over the exact same 10 parameters as Figure 6 (K_ICE, K_EV, delta,
lambda, a_chi, b_chi, kappa, mu, r, alpha) -- `fig07_08_sobol_gen.py` here is
a thin wrapper pinning the settings this run uses:

- Generation: `package/generating_data/sensitivity_analysis_calibration_gen.py`,
  using `SALib.sample.saltelli` against
  `package/constants/variable_parameters_dict_SA.json` (bounds for all 10
  params) and `package/constants/base_params_SA.json` (`seed_repetitions=64`,
  same burn-in/calibration duration as everything else here).
- Plotting: `package/plotting_data/sensitivity_analysis_calibration_plot.py`,
  which calls `SALib.analyze.sobol` and draws the first-order (Figure 7) /
  total-order (Figure 8) error-bar layout, one panel per output (cumulative
  emissions, EV adoption, firm profit, market concentration, utility, mean
  car age).

The run count is dominated entirely by `N_samples`:

```
runs = N_samples x (D + 2) x seed_repetitions      [D=10 params, calc_second_order=False]
     = N_samples x 12 x 64
```

`N_samples=256` and `calc_second_order=False` were chosen deliberately over
the underlying gen script's own defaults (`N_samples=512`, `calc_second_order=True`,
which together would need `2D+2=22` rather than `D+2=12` runs per sample and
cost ~2.6 days on 64 cores) since the paper's Figures 7/8 only show
first-/total-order indices, not the second-order interaction terms, and 256
samples still gives reasonable convergence at `--cpus-per-task=128`:

| N_samples | Total runs | Wall-clock @ 20s/run, 128 cores |
|---|---|---|
| 512 | 393,216 | ~17.1 h |
| **256 (chosen)** | **196,608** | **~8.5 h** |
| 128 | 98,304 | ~4.3 h |
| 64 | 49,152 | ~2.1 h |

For comparison, Figure 6 (the *local* sensitivity grid) is 2,560 runs, ~13
min -- three orders of magnitude cheaper. They're kept as two separate jobs
(rather than folded into one script) since that cost gap means combining
them wouldn't save any wall-clock time, only remove one `sbatch` call.

## Changes made to shared code

- `package/constants/vary_sen_decarb.json`: `Grid_emissions_intensity`
  property list changed from `[0.1, 0.5, 1]` to `[0.1, 0.5, 0.75, 1]`, so the
  same generation run serves both Figure 9's three reduction columns (90% /
  50% / 25%) and Figure 10's baseline + two perturbation magnitudes (which
  need the `1.0` "no change" value that Figure 9 doesn't display).
- `package/plotting_data/inputs_and_emissions_plot.py::plot_elasticity_comparison`:
  added optional `grid_intensities_to_plot` / `elec_prices_to_plot` filters
  (default `None` = plot every non-baseline value, i.e. unchanged behaviour)
  so Figure 10 can restrict itself to the two magnitudes per axis the paper
  shows, from a dataset that also carries Figure 9's extra 25%-reduction
  value.
- `package/generating_data/vary_single_param_gen.py::main`: now returns the
  results folder name instead of the internal `params_list` (nothing else in
  the repo called or used that return value) -- needed so
  `fig06_local_sensitivity_gen.py` can chain its 10 generation calls into the
  combined plot without re-deriving timestamps.
