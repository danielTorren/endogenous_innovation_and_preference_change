# Validation

Out-of-sample tests of the zip-level model. Everything here currently runs on
**fake placeholder data**; see
`package/calibration_data/FAKE_zip_data/FAKE_DATA_README.md` for the column
contract the real data has to meet.

## Order of operations

```bash
# 0. generate the placeholder data (once; skip when the real data is in place)
python -m package.generating_data.fake_zip_data_gen

# 1. build the calibration target from the zip data
python -m package.calibration.calibration_data_outputs_zip

# 2. FIRST CHECK when a real population file arrives: are its marginals in the
#    regime the model's non-free parameters were calibrated for?
python -m package.validation.compare_populations

# 2b. do the homophily parameters do what you think to THIS population?
python -m package.validation.homophily_diagnostics

# 3. how well is the target actually known? sets the tolerance band
python -m package.validation.bootstrap_targets

# 4. which summary dimensions carry theta signal above the noise floor?
#    ~200 simulations, and it can save you thousands
python -m package.validation.noise_screen --n-seed 30 --n-theta 30

# 5. calibrate
python -m package.calibration.NN_multi_round_calibration_zip_gen

# 6. validate
python -m package.validation.run_validation              # smoke, minutes
python -m package.validation.run_validation --production # cluster
```

Steps 2-4 are not optional preliminaries. Each of them caught a real problem
during development, described below.

## The four splits

`splits.py`. Every split simulates the **full** population and holds out only in
the loss, because firms, the two NK landscapes, the second-hand merchant and the
network are state-wide and coupled: you cannot remove zips or years from the
simulation itself.

| split | what is held out | severity | what it tells you |
|---|---|---|---|
| `temporal` | 2022-2023, all dimensions | high | whether the diffusion dynamics extrapolate through the bend in the curve |
| `moment_type` | the sales flow | medium | whether stock and turnover are both right, not just the accumulation |
| `random_zip` | a random 30% of zips | low | near-automatic; only a failure means anything |
| `extrapolate_ruralness` | the most rural 25% of zips | highest | whether the covariate-to-preference map generalises |

**Read `extrapolate_ruralness` first.** It is the only split that tests the
assumption every policy counterfactual rests on. A charging-infrastructure or
rebate counterfactual is precisely a claim about parts of the covariate space
you did not fit; if `a_rural` fitted on urban zips does not predict rural zips,
that counterfactual has no support. Swap the covariate for
`political_leaning` or `median_income` to interrogate `rho_pol` or `eps_beta`
the same way.

**Be honest about `random_zip`.** The model maps zip covariates to preferences
through a smooth parametric function, so a randomly held-out zip is an
interpolation between trained zips. A pass is close to automatic and is not
evidence.

## The score

```
z = (model mean - observed) / sqrt(model seed variance + observation variance)
```

`|z| < 2` means the data cannot distinguish the model from correct. Both terms
in the denominator are measured, not assumed:

- **model seed variance** from re-simulating across seed triples, which vary the
  behavioural seed, the NK landscape seed and the population-draw seed together.
- **observation variance** from `bootstrap_targets.py`, which runs two different
  bootstraps because the right notion of uncertainty differs by dimension: a
  binomial-within-zip resample for the state aggregates (the data covers every
  zip, so there is no zip sampling error, only counting error), and a
  zip-cluster resample for the gradients and dispersion (whose standard error is
  driven by residual zip heterogeneity, not counting). The larger of the two is
  used, which is the conservative choice.

On the fake data the counting noise is about 0.3% relative while the zip-cluster
noise is 8-30%, so the gradients are known to roughly 20-30% relative precision.
Do not tune against gaps smaller than that.

**A large test `|z|` with a large model sd is a noise problem, not a model
problem.** Raise `seed_repetitions` and `n_eval_seed` before drawing a
conclusion. A large test `|z|` with a small model sd is a real miss.

## Three things that bit during development

Recorded because they will bite again with the real data.

### 1. The landscape seed dominates everything, and is now pinnable

`seed_inputs` used to seed the NK landscapes, the network wiring, the firm
manager and the agent draws all at once, so none could be varied independently.
It is now split into `seed_landscape`, `seed_network` and `seed_population`, each
defaulting to `seed_inputs` so every existing config is unchanged.

`build_seed_triples(..., pin_landscape=True)` is the default: the landscape is
held fixed while the other three vary. **That makes the posterior conditional on
one technology landscape.** It is a defensible position — California drew one
landscape, not a distribution over them — but it is consequential, because:

Measured on `base_params_NN_lower_fuel.json`, 24 draws of `seed_inputs` with
everything else fixed: **2023 EV stock share ranges 0.0033 to 0.416, median
0.069**, against a target of 0.038. 38% of draws exceed 0.10.

So *which* landscape you pin matters enormously, and a robustness check across
several pinned landscapes is not optional. The existing calibration
(`NN_multi_round_calibration_multi_gen.py`) pins `seed_inputs` at 22 for all
8,192 runs and varies only `seed`. Seed 22 gives 0.026, close to the target — so
that posterior is conditional on one favourable draw, while being presented as if
it were marginal. Pinning is fine; pinning silently is not.

Pass `pin_landscape=False` to marginalise over the landscape instead. The
posterior widens, correctly: that width was always there, it was hidden.

Practical consequence: **a 10x shift in EV share is inside landscape noise.**
Before attributing any change to a mechanism, check it against a seed sweep.
This is how the synthetic-population path was cleared of a suspected 20x bug —
matched sweeps gave parametric median 0.069 against synthetic 0.066.

### 2. A too-homogeneous population silently breaks the model

A first version of the fake population had VMT log-sd 0.24 against the
parametric 0.82. Every median matched to within 1%, and EV share ran to 85%
instead of 4%: with no low-mileage agents, nobody has a reason to stay on ICE.
Only the spreads gave it away, which is why `compare_populations.py` prints
log-sd and not just medians.

### 3. `d_vec` is monthly, per vehicle

The model steps monthly and `d_vec` is a per-timestep distance (parametric
median 1,274 miles). A real synthetic population will report **annual household**
VMT, which needs dividing by 12 *and* by vehicles per household. A 12x too large
`d_vec` makes the lifetime fuel-cost term swamp the utility and drives EV share
to 1.0 within a year with nothing raised. `SyntheticPopulation.gen_d_vec` warns
when the monthly median is more than 3x from the parametric value; do not ignore
it, because every other parameter in the model was calibrated against that
distance scale.

## Network homophily

Two parameters, both calibrated, both nesting the previous behaviour at their
defaults. Because the network is a Watts-Strogatz ring over agent *index*, index
adjacency **is** network adjacency, so homophily is implemented as the choice of
which household sits at which ring position
(`SyntheticPopulation._homophilous_order`).

| parameter | 0 | 1 |
|---|---|---|
| `homophily_strength` (h) | uniformly random placement | placement fully determined by similarity |
| `homophily_spatial_weight` (w) | similarity = income only | similarity = physical proximity only |

`h` interpolates via a Gaussian copula, `key = h·z_sim + √(1−h²)·ε`, so **h is
exactly the rank correlation between ring position and similarity**. That makes it
scale-free and comparable across `w`, which an ad-hoc "reshuffle a fraction 1−h"
scheme would not be. Verified: h = 0.3/0.6/0.9 gives measured Spearman
0.255/0.559/0.886.

Proximity is measured by a **Hilbert space-filling curve** index over zip
centroids, not by sorting on latitude or by a Morton curve. The ring position is
the network neighbourhood, so the 1-D ordering must preserve 2-D locality
properly: single-axis sorting fails entirely in the other axis, and Morton order
has periodic long jumps that would wire distant zips together — costly here
because K = 150 neighbours, so each jump contaminates many edges.

Measured on the ring at `num_individuals = 3000` (`homophily_diagnostics.py`):

| h | w | income assort. | spatial assort. | mean pair distance | same-zip |
|---|---|---|---|---|---|
| 0.0 | – | 0.002 | 0.003 | 406 km | 0.4% |
| 1.0 | 0.0 | **0.830** | 0.001 | 404 km | 0.5% |
| 1.0 | 0.5 | 0.421 | 0.440 | 333 km | 0.9% |
| 1.0 | 1.0 | 0.050 | **0.927** | **79 km** | 8.0% |

The income column stays near zero at w = 1 even though rich zips cluster
spatially, because within-zip income spread (log-sd ≈ 0.81) dwarfs the
between-zip component (≈ 0.45). **That separation is what makes `w`
identifiable.** If a real population turns out to have income and geography more
collinear than this, `w` will be weakly identified — check with
`homophily_diagnostics.py` before trusting a fitted value.

**`h` saturates, and the ceiling is `prob_rewire`, not the parameter bound.**
Rewiring is applied after placement, so ~10% of every agent's edges stay
uniformly random long-range bridges at any `h`. That is deliberate — it preserves
the small-world property the network exists for — but it means h = 1 is strongly
assortative local structure, not disconnected cliques. Lower `SW_prob_rewire` if
you need segregation to bite harder, and **report realised assortativity, not h**.

### Moran's I, and why it had to come with the homophily parameters

`moran_2018/2021/2023` are now in the summary vector. They were added in the same
change as the homophily parameters, because the two cannot be separated
afterwards.

Once agents are placed by similarity, spatially clustered EV adoption has two
explanations that look identical in a single cross-section: similar people were
placed together and share preferences (**sorting**: `homophily_*`), or adoption
spread locally through the network (**contagion**: `a_chi`/`b_chi`). Fitting both
to cross-sectional gradients alone leaves them trading off with nothing to
distinguish them.

Measured model response, three seeds each:

| case | Moran 2018 | 2021 | 2023 |
|---|---|---|---|
| h = 0 (random) | −0.021 | −0.034 | 0.025 |
| h = 1, w = 0 (income) | 0.053 | 0.039 | 0.053 |
| h = 1, w = 1 (spatial) | **0.478** | **0.426** | **0.381** |
| h = 1, w = 1, chi × 0.3 | 0.007 | −0.011 | 0.000 |

Rows 1–3 are the clean result: Moran's I is near zero under random placement,
still small under pure income homophily, and large only under spatial homophily.
So **Moran's I identifies `w`**, which the gradients alone cannot.

Row 4 is suggestive but **confounded and should not be quoted**: that run reached
96% EV share, so there was almost no cross-sectional variance left to correlate
spatially. Whether Moran's I cleanly separates contagion from sorting needs
re-testing once the model is calibrated into a non-saturated regime. The
theoretical argument (sorting gives a spatial pattern fixed in shape from t = 0,
contagion gives autocorrelation that grows, so the *trajectory* discriminates)
is not yet demonstrated here: in these saturated runs the Moran trend is a weak
discriminator and the **level** does all the work. Treat the level as identifying
`w` today, and revisit the trend later.

## What is still deliberately absent

Nothing in the summary vector is missing by design any more. The remaining known
gap is that `SW_prob_rewire` is fixed rather than calibrated, which caps
achievable homophily; if fitted `homophily_strength` piles up at its upper bound,
that is the constraint to relax next.
