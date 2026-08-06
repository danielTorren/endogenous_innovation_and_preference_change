"""
Run the four validation designs end to end.

For each split:
  1. Calibrate using ONLY the fit side (fit dimensions, fit zips).
  2. Draw parameters from the resulting posterior.
  3. Re-simulate at those parameters over several seed triples.
  4. Score the TEST side, which the calibration never saw.

THE SCORE. A raw model-minus-data gap is meaningless on its own, because both
sides are noisy. So every test dimension is standardised:

    z = (model mean - observed) / sqrt(model seed variance + observation variance)

model seed variance comes from re-simulating across seed triples (which vary the
behavioural seed, the NK landscape seed AND the population-draw seed, so it
includes all three of the model's noise sources). observation variance comes
from package/validation/bootstrap_targets.py.

|z| < 2 means the model is inside the combined noise band, i.e. the data cannot
distinguish it from correct. That is a pass. It is deliberately not a claim that
the model is right: with the landscape noise floor as wide as it is (2023
aggregate EV share spans 0.0033 to 0.416 across 24 landscape draws), the band is
generous, and the report prints the band width alongside the score so a pass
achieved only by huge model noise is visible rather than flattering.

COST. Full settings are expensive: 4 splits x num_rounds x num_simulations x
seed_repetitions plus the evaluation runs. The defaults here are a SMOKE
configuration that finishes in minutes and proves the wiring; the production
numbers are in the argparse help and in README.md. A smoke run's z-scores are
not evidence about the model.

Run:
    python -m package.validation.run_validation                 # smoke
    python -m package.validation.run_validation --production
    python -m package.validation.run_validation --split temporal
"""

import argparse
import json
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

from package.resources.utility import load_object, save_object, createFolder, \
    produce_name_datetime, get_num_workers
from package.resources.run import generate_data
from package.calibration.summary_stats import model_summary, data_summary, dim_names
from package.calibration.NN_multi_round_calibration_zip_gen import (
    main as calibrate, DEFAULT_PARAMETERS_LIST, build_seed_triples, _sanitize,
)
from package.validation.splits import all_splits
from package.validation.bootstrap_targets import target_uncertainty

BASE_PARAMS = "package/constants/base_params_NN_zip.json"


# ===========================================================================
# Evaluation: re-simulate at posterior parameters and score the held-out side
# ===========================================================================

def _eval_run(base_params, parameters_list, theta, seed_triple, spec, zip_table,
              fit_zips, test_zips):
    """
    One simulation, returning the summary over BOTH zip sides.

    Both sides come from the SAME simulation, which is the whole reason the
    zip-holdout splits are cheap: the model always has to simulate every agent
    anyway (firms, the second-hand market and the network are state-wide and
    coupled), so restricting to a zip subset is a post-hoc aggregation, not a
    separate run.
    """
    params = deepcopy(base_params)
    for p, v in zip(parameters_list, theta):
        params[p["subdict"]][p["name"]] = float(v)
    params.update(seed_triple)

    c = generate_data(params)
    fit = _sanitize(model_summary(c, params, spec, zip_table, zip_subset=fit_zips), spec)
    test = _sanitize(model_summary(c, params, spec, zip_table, zip_subset=test_zips), spec)
    return fit, test


def evaluate(split, fitted, bundle, base_params, n_eval_theta, n_eval_seed, verbose=True):
    spec, zip_table = bundle["spec"], bundle["zip_table"]
    names = dim_names(spec)
    parameters_list = fitted["parameters_list"]

    # posterior draws: the mode plus a spread, so the report shows whether the
    # held-out prediction is robust across the posterior or only at its peak
    samples = fitted["samples"].numpy()
    rs = np.random.RandomState(0)
    thetas = [fitted["best_sample"].numpy()]
    if n_eval_theta > 1:
        thetas += [samples[i] for i in rs.choice(len(samples), n_eval_theta - 1, replace=False)]

    triples = build_seed_triples(base_params, n_eval_seed)
    jobs = [(t, tr) for t in thetas for tr in triples]
    nw = get_num_workers()

    if verbose:
        print(f"  evaluating: {len(thetas)} posterior draws x {len(triples)} seed triples "
              f"= {len(jobs)} simulations")

    res = Parallel(n_jobs=nw)(
        delayed(_eval_run)(base_params, parameters_list, t, tr, spec, zip_table,
                           split.fit_zips, split.test_zips)
        for t, tr in jobs)

    model_fit = np.asarray([r[0] for r in res])
    model_test = np.asarray([r[1] for r in res])

    # observed target on each side, rebuilt (not sliced) because a zip subset
    # changes the aggregates and the gradient regression, not just the rows
    obs_fit = _sanitize(data_summary(bundle["stock_arrays"], bundle["sales_arrays"],
                                    spec, zip_table, zip_subset=split.fit_zips), spec)
    obs_test = _sanitize(data_summary(bundle["stock_arrays"], bundle["sales_arrays"],
                                     spec, zip_table, zip_subset=split.test_zips), spec)

    obs_sd = bundle["_obs_sd"]

    return {
        "dim_names": names,
        "model_fit": model_fit, "model_test": model_test,
        "obs_fit": obs_fit, "obs_test": obs_test,
        "obs_sd": obs_sd,
        "thetas": np.asarray(thetas),
        "seed_triples": triples,
        "score_fit": _score(model_fit, obs_fit, obs_sd, split.fit_mask, names),
        "score_test": _score(model_test, obs_test, obs_sd, split.test_mask, names),
    }


def _score(model_runs, obs, obs_sd, mask, names):
    """Standardised error on the masked dimensions."""
    m = np.asarray(mask, dtype=bool)
    mu = model_runs.mean(axis=0)
    model_sd = model_runs.std(axis=0, ddof=1) if len(model_runs) > 1 else np.zeros_like(mu)
    band = np.sqrt(model_sd ** 2 + obs_sd ** 2)
    z = np.divide(mu - obs, band, out=np.full_like(mu, np.nan), where=band > 0)
    return {
        "dims": [n for n, k in zip(names, m) if k],
        "model_mean": mu[m], "model_sd": model_sd[m],
        "observed": obs[m], "obs_sd": obs_sd[m],
        "band": band[m], "z": z[m],
        "mean_abs_z": float(np.nanmean(np.abs(z[m]))),
        "frac_within_2": float(np.nanmean(np.abs(z[m]) < 2.0)),
    }


# ===========================================================================
# Reporting
# ===========================================================================

def _report(split, spec, fitted, ev, zip_table):
    print()
    print("=" * 82)
    print(f" {split.name.upper()}   (severity: {split.severity})")
    print(f" {split.description}")
    if split.caveat:
        print(f" CAVEAT: {split.caveat}")
    print("=" * 82)

    dr = split.dim_report(spec)
    if split.fit_zips is None:
        print(f" fit dims  ({len(dr['fit'])}): {dr['fit']}")
        print(f" test dims ({len(dr['test'])}): {dr['test']}")
    else:
        hh = zip_table["households"]
        print(f" fit  zips: {int(split.fit_zips.sum()):4d}  "
              f"({hh[split.fit_zips].sum()/hh.sum():.1%} of households)")
        print(f" test zips: {int(split.test_zips.sum()):4d}  "
              f"({hh[split.test_zips].sum()/hh.sum():.1%} of households)")

    print("\n posterior mode:")
    for p, v in zip(fitted["parameters_list"], fitted["best_sample"]):
        print(f"   {p['name']:<10s} {v.item(): .5f}   prior {p['bounds']}")

    for label, sc in (("FIT (seen)", ev["score_fit"]), ("TEST (held out)", ev["score_test"])):
        print(f"\n {label}")
        print(f"   {'dim':<24s} {'model':>11s} {'model sd':>10s} {'observed':>11s} "
              f"{'obs sd':>10s} {'z':>7s}")
        for i, d in enumerate(sc["dims"]):
            flag = "" if abs(sc["z"][i]) < 2 else "  <-- outside band"
            print(f"   {d:<24s} {sc['model_mean'][i]:11.6f} {sc['model_sd'][i]:10.6f} "
                  f"{sc['observed'][i]:11.6f} {sc['obs_sd'][i]:10.6f} {sc['z'][i]:7.2f}{flag}")
        print(f"   mean |z| = {sc['mean_abs_z']:.2f}   within band = {sc['frac_within_2']:.0%}")


def _summary_table(results):
    print()
    print("=" * 82)
    print(" SUMMARY")
    print("=" * 82)
    print(f" {'split':<26s} {'severity':<9s} {'fit |z|':>8s} {'test |z|':>9s} "
          f"{'test within band':>17s}")
    for r in results:
        print(f" {r['name']:<26s} {r['severity']:<9s} "
              f"{r['score_fit']['mean_abs_z']:8.2f} {r['score_test']['mean_abs_z']:9.2f} "
              f"{r['score_test']['frac_within_2']:16.0%}")
    print()
    print(" Read it in this order: extrapolate_* first (the only split that tests whether")
    print(" the covariate-to-preference map generalises, which every counterfactual needs),")
    print(" then temporal, then moment_type. random_zip is near-automatic; only a failure")
    print(" there means anything.")
    print()
    print(" A large test |z| with a large model sd is a NOISE problem, not necessarily a")
    print(" model problem: raise seed_repetitions and n_eval_seed before concluding anything.")
    print(" A large test |z| with a small model sd is a real miss.")


# ===========================================================================
# Driver
# ===========================================================================

def run(base_params_path=BASE_PARAMS,
        num_simulations=8, num_rounds=1, seed_repetitions=4, seeds_per_theta=1,
        n_eval_theta=2, n_eval_seed=4, n_boot=400,
        only=None, parameters_list=None, save=True):
    with open(base_params_path) as f:
        base_params = json.load(f)

    bundle = load_object("package/calibration_data", "calibration_data_output_zip")
    spec, zip_table = bundle["spec"], bundle["zip_table"]

    if bundle.get("IS_FAKE_DATA"):
        print("=" * 82)
        print(" FAKE PLACEHOLDER DATA. These z-scores test the PIPELINE, not the model.")
        print(" The fake zip EV shares were generated from a logistic surface with known")
        print(" gradients, which the model has no reason to reproduce; do not read the")
        print(" numbers below as evidence about California.")
        print("=" * 82)

    print("\nestimating observation noise on the target...")
    bundle["_obs_sd"] = target_uncertainty(bundle, n_boot=n_boot, verbose=False)["sd"]

    splits = all_splits(spec, zip_table)
    if only:
        splits = [s for s in splits if s.name == only or s.name.startswith(only)]
        if not splits:
            raise SystemExit(f"no split matching {only!r}; "
                             f"have {[s.name for s in all_splits(spec, zip_table)]}")

    total = len(splits) * (num_rounds * num_simulations * seed_repetitions * seeds_per_theta
                           + n_eval_theta * n_eval_seed)
    print(f"\n{len(splits)} split(s), roughly {total} simulations in total")

    results = []
    for split in splits:
        print(f"\n### calibrating on the fit side of {split.name} ...")
        fitted = calibrate(
            parameters_list=parameters_list or DEFAULT_PARAMETERS_LIST,
            BASE_PARAMS_LOAD=base_params_path,
            num_simulations=num_simulations,
            num_rounds=num_rounds,
            seed_repetitions=seed_repetitions,
            seeds_per_theta=seeds_per_theta,
            target_mask=split.fit_mask,
            zip_subset=split.fit_zips,
            root=f"validation_{split.name}",
            verbose=False,
        )
        ev = evaluate(split, fitted, bundle, base_params, n_eval_theta, n_eval_seed)
        _report(split, spec, fitted, ev, zip_table)
        results.append({
            "name": split.name, "severity": split.severity,
            "description": split.description, "caveat": split.caveat,
            "fit_mask": split.fit_mask, "test_mask": split.test_mask,
            "fit_zips": split.fit_zips, "test_zips": split.test_zips,
            "best_sample": fitted["best_sample"].numpy(),
            "parameters_list": fitted["parameters_list"],
            "calibration_folder": fitted["fileName"],
            "score_fit": ev["score_fit"], "score_test": ev["score_test"],
        })

    _summary_table(results)

    if save:
        out = produce_name_datetime("validation")
        createFolder(out)
        save_object({"results": results, "spec": spec,
                     "IS_FAKE_DATA": bundle.get("IS_FAKE_DATA", False),
                     "settings": {"num_simulations": num_simulations,
                                  "num_rounds": num_rounds,
                                  "seed_repetitions": seed_repetitions,
                                  "seeds_per_theta": seeds_per_theta,
                                  "n_eval_theta": n_eval_theta,
                                  "n_eval_seed": n_eval_seed}},
                    out + "/Data", "validation_results")
        print(f"\nsaved to {out}/Data/validation_results.pkl")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", action="store_true",
                    help="64 sims x 3 rounds x 32 seed reps, 8 posterior draws x 16 seeds "
                         "per split. Tens of thousands of simulations: cluster only.")
    ap.add_argument("--split", default=None, help="run only this split by name")
    ap.add_argument("--num-simulations", type=int, default=8)
    ap.add_argument("--num-rounds", type=int, default=1)
    ap.add_argument("--seed-repetitions", type=int, default=4)
    ap.add_argument("--seeds-per-theta", type=int, default=1,
                    help="average k simulations per theta; cuts noise SD by sqrt(k). "
                         "Run noise_screen.py first: on this model 8-16 is usually needed.")
    ap.add_argument("--n-eval-theta", type=int, default=2)
    ap.add_argument("--n-eval-seed", type=int, default=4)
    a = ap.parse_args()

    if a.production:
        run(num_simulations=64, num_rounds=3, seed_repetitions=32, seeds_per_theta=8,
            n_eval_theta=8, n_eval_seed=16, only=a.split)
    else:
        run(num_simulations=a.num_simulations, num_rounds=a.num_rounds,
            seed_repetitions=a.seed_repetitions, seeds_per_theta=a.seeds_per_theta,
            n_eval_theta=a.n_eval_theta, n_eval_seed=a.n_eval_seed, only=a.split)
