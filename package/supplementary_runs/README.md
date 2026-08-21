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
| 6 | 10-panel local sensitivity (EV uptake) | `fig06_panels.py` (resumable) or `fig06_local_sensitivity_gen.py`, both + `fig06_local_sensitivity_plot.py` | `submit_fig06_panels.slurm` |
| 7 | Sobol first-order sensitivity, 6 outputs x 10 params | `fig07_08_sobol_gen.py` | `submit_fig07_08_sobol_gen.slurm` |
| 8 | Sobol total-order sensitivity, 6 outputs x 10 params | `fig07_08_sobol_gen.py` | (same job as Figure 7) |
| 9 | BAU EV uptake/emissions, decarb x elec-price, time series | `fig09_10_bau_gen.py` + `fig09_bau_timeseries_plot.py` | `submit_fig09_10_bau_gen.slurm`, or `submit_figs_06_09_10.slurm` to build S6/S9/S10 in one job |
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

## Figure 6: resuming a partial run

`fig06_local_sensitivity_gen.run_fig6()` does all ten parameter sweeps in one
unbroken sequence, and the folders it makes are all called
`results/single_param_vary_<timestamp>` with nothing in the name to say which
parameter each holds. If the job dies (or hits its `--time` limit) part-way, the
panels that did finish are unusable in practice and the whole thing gets re-run.

`fig06_panels.py` is the resumable form of the same work. It matches each
existing folder to its parameter by reading `Data/vary_single.pkl`, so it only
runs what is missing, and a sweep that fails no longer takes the other nine with
it:

```bash
python -m package.supplementary_runs.fig06_panels list      # which panels exist
python -m package.supplementary_runs.fig06_panels run-all   # run the missing ones, then combine
python -m package.supplementary_runs.fig06_panels run e     # just panel e (b_chi)
python -m package.supplementary_runs.fig06_panels combine   # re-combine, no simulation
```

`run-all` ends by printing the ten folders in panel order, ready to paste into
`RUNS["local_sensitivity"]` in `package/paper_figures/build_figures.py`. On the
cluster, `submit_fig06_panels.slurm` runs exactly that, and re-submitting it
after a failure picks up where it stopped. A folder whose value grid no longer
matches its `vary_single_*.json` is reported and not reused, so editing a config
forces that panel to be re-run rather than quietly plotting the old grid.

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
sbatch package/supplementary_runs/submit_fig06_panels.slurm
sbatch package/supplementary_runs/submit_fig07_08_sobol_gen.slurm
sbatch package/supplementary_runs/submit_fig09_10_bau_gen.slurm
sbatch package/supplementary_runs/submit_fig11_beta_carbon_gen.slurm
sbatch package/supplementary_runs/submit_fig12_beta_rebate_gen.slurm
sbatch package/supplementary_runs/submit_fig13_achi_carbon_gen.slurm
sbatch package/supplementary_runs/submit_fig14_achi_rebate_gen.slurm
```

To re-run only Figures 9, 10, 13 and 14 -- the four whose
`\includegraphics` in `docs/paper/supplementary.tex` still point at the old
`pics/` PNGs rather than `supplementary_figs/Supp_Figure_N.png` -- use the
narrower submitter instead (three jobs: 9+10 share one generation run):

```bash
bash package/supplementary_runs/submit_figs_09_10_13_14.sh
```

### Memory sizing (why the 21/08/2026 figs 13/14 jobs were OOM-killed)

One full-length run of the Figures 11-14 config (600 steps, 3,000 individuals)
peaks at **~1.0 GiB RSS**, measured single-process. The original `--mem=64G`
with `--cpus-per-task=64` therefore gave each worker exactly 1.0 GiB and no
headroom for the parent process, the per-worker interpreter, or allocator
fragmentation. Symptom: a worker died mid-way through the policy phase
(joblib logged "A worker stopped while some jobs were given to the executor"),
the phase still finished, and then the BAU phase was OOM-killed the moment it
started. Two things were wrong, both now fixed:

- `--mem` is 128G for figs 11-14, i.e. 2 GiB per worker at 64 workers.
- `run_cross_variation` now frees the policy phase's arrays and calls
  `_release_workers()` before the BAU phase. joblib's loky backend keeps its
  workers warm between `Parallel(...)` calls, and CPython does not return freed
  memory to the OS, so the BAU phase was inheriting 64 processes already at
  their ~1 GiB high-water mark and allocating on top of that. Shutting the
  executor down gives the BAU phase fresh workers at baseline RSS.

### Recovering a run whose BAU phase died

`run_cross_variation` saves the policy grid (`Data/data_cross_ev`) **before**
starting the BAU sweep, so an OOM in the BAU phase costs only the BAU sweep --
the 3,072-run policy grid on disk is still good. The BAU sweep is 512 runs
(~3 min) and, because every policy is switched off in it, depends only on the
*physical* parameter -- so one sweep serves both figures sharing that axis
(13 and 14, or 11 and 12):

```bash
sbatch package/supplementary_runs/submit_fig13_14_achi_bau_gen.slurm
```

Paste the `results/cross_a_chi_vs_Carbon_price_BAU_<ts>` folder it prints into
`RUNS["grid_bau_achi"]` in `package/paper_figures/build_figures.py`, then build
the two figures with no further simulation:

```bash
python -m package.paper_figures.build_figures --only S13,S14
```

The same script also has direct BAU-only and plot-only modes:

```bash
python -m package.supplementary_runs.fig11_14_policy_grid_gen 13 bau
python -m package.supplementary_runs.fig11_14_policy_grid_gen 13 plot <policy_folder> [bau_folder]
```

### Figures 6, 9 and 10 in one command

These three were the remainder after the 21/08/2026 batch: S6's other nine
local-sensitivity panels never came back, and the decarb x elec-price grid
behind S9/S10 was never produced. One job runs both generation steps and builds
all three figures:

```bash
sbatch package/supplementary_runs/submit_figs_06_09_10.slurm
```

It runs `build_figs_06_09_10.py`, which takes each folder from the generating
function's return value and feeds it to `build_figures` through the new `--set`
flag, so the PNGs reach `docs/paper/supplementary_figs/` and `supplementary.tex`
is repointed without anything being pasted into `RUNS` by hand. ~15 min:

| Step | Runs | Time |
|---|---|---|
| S6 -- 9 missing panels x 4 values x 64 seeds, 456 steps | 2,304 | ~8 min |
| S9 + S10 -- 4 decarb x 3 price x 64 seeds, 768 steps | 768 | ~4 min |

Sequential on purpose: each step already saturates all 64 workers. If S6's
panels do not all finish, it says so and still builds S9/S10 rather than failing
the whole job. It exits non-zero unless each expected PNG was actually
*rewritten* -- `build_figures` exits 0 even when it skips a figure, and
`Supp_Figure_6.png` is already on disk from an earlier run, so existence alone
would not distinguish "built" from "left alone".

`--set` is generally useful, not just here -- it overrides any `RUNS` entry for
one invocation:

```bash
python -m package.paper_figures.build_figures --only S9,S10 \
    --set bau_grid=results/phys_duo_Grid_emissions_intensity_vs_Electricity_price_<ts>
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
| Fig 9/10 | 768 | 4 decarb x 3 price x 64 seeds |
| Fig 11 | 3,584 | (8 beta x 6 carbon x 64 seeds) + (8 x 64 BAU) |
| Fig 12 | 3,584 | (8 beta x 6 rebate x 64 seeds) + (8 x 64 BAU) |
| Fig 13 | 3,584 | (8 a_chi x 6 carbon x 64 seeds) + (8 x 64 BAU) |
| Fig 14 | 3,584 | (8 a_chi x 6 rebate x 64 seeds) + (8 x 64 BAU) |

Each of Figs 11-14 above includes its own 512-run BAU sweep, which is
redundant within a pair: BAU has all policies off, so 11/12 compute the same
beta sweep twice and 13/14 the same a_chi sweep twice. Running one pair member
full and the other's BAU from `grid_bau_*` saves 512 runs per pair.
| **Total** | **214,336** | |

At ~20 s/run (measured during smoke-testing -- 8 full-length, 456-step runs
finished in ~21 s on 16 workers, i.e. ~20 s each when there's a free core per
run) that's **~1,188 core-hours** of total compute, almost all of it (~1,092
core-hours) the Sobol job. What that means for wall-clock time depends on how
many jobs the cluster schedules at once, since each job parallelises
internally across its own `--cpus-per-task`:

- **Per-job wall-clock**, i.e. `ceil(runs / cpus-per-task) x 20 s`:
  Fig 5 ~20 s, Fig 6 ~13 min, Fig 7/8 ~8.5 h (128 cores), Fig 9/10 ~5 min
  (12 batches of 64, ~25 s/run at 768 steps), each of Fig 11-14 ~19 min.
- **If all 8 jobs get scheduled at once**: wall-clock for the whole batch is
  set by the slowest job, i.e. **Fig 7/8's ~8.5 h**.
- **Worst case, jobs queue one after another**: sum of all eight, ~10.1 h.

Either way this comfortably finishes overnight -- the Sobol job (Fig 7/8) is
now the long pole by a wide margin, everything else finishes in under 20 min.
The `--time` values inside each `.slurm` file are padded generously above
these estimates (up to 24 h) as safe upper bounds against a slow/busy node,
not the expected runtime; trim them from `seff <jobid>` after the first real
run, per this repo's existing convention (see e.g.
`package/analysis/submit_vary_single_policy_gen.slurm`).

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
