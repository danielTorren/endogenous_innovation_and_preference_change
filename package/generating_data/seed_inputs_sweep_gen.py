"""
Sweep seed_inputs and score each landscape against the calibration targets.

WHY THIS EXISTS. seed_inputs is not a nuisance seed. It is the ONLY seed the NK
technology landscapes, the agent population, and the firms' starting designs are
drawn from (Controller.handle_seed), and it is held FIXED across every replicate of
every run in this project -- seed_repetitions varies `seed`, not seed_inputs. So
one draw of it is baked into every result, and its effect on EV uptake is large:
measured across the landscape draw alone, the 2023 EV share moves over most of the
unit interval at unchanged parameters. Whatever value is pinned in
base_params_calibration.json is therefore a modelling choice on the same footing as
a calibrated parameter, and it should be made by measurement rather than inherited.

The value in the JSON, 22, was inherited from the rejection-sampler code path in
gen_chi, where a_chi silently moved the landscape (see gen_chi's Step 1 comment).
On the stream-stable inverse-CDF path now in place, seed_inputs = 22 selects a
DIFFERENT landscape than it did there -- the worst ICE design changes from
100100110010101 at $38.5M to 111011100010101 at $37.1M -- so the old value carries
no information about fit on this path. This script is how a value is chosen for it.

WHAT IS SCORED. Three things, all read from the final calibration year unless said
otherwise, and all three must be satisfied at once:

  ev_rmse   Root mean square error of the EV STOCK TRAJECTORY, 2010-2023, against
            the observed California series, in LOGIT units. The whole 14-year
            trajectory rather than the 4 years the SBI calibration conditions on,
            because a landscape that lands on 2023 by overshooting 2015 and
            falling back is not a landscape worth keeping, and the extra ten years
            cost nothing to score. Logit rather than raw because the series spans
            2.7e-5 to 3.8e-2, so a raw error is dominated entirely by the last two
            years. Sampled at APRIL of each year, and the logit is applied per seed
            BEFORE averaging over seeds, both to match the calibration convention
            in NN_multi_round_calibration_multi_gen.py exactly.
  hhi       New-car market concentration, mean over the final 12 months, against
            the observed band. Reported with an in-band flag, not folded into
            ev_rmse: the two are different quantities and adding them would need a
            weight nobody can justify.
  age_yr    Whole-fleet mean car age in years, mean over the final 12 months,
            against the observed band.

The ranking is by ev_rmse among the rows that sit inside BOTH bands. Rows failing a
band are still printed, so a near-miss is visible rather than silently dropped.

WHAT THIS IS NOT. It is not a calibration. The parameters stay exactly as the JSON
has them; only the landscape draw moves. If no seed_inputs in the grid satisfies
both bands, the answer is that the parameters need refitting on this code path, not
that the grid should be widened until something passes -- a value picked from a
long enough list will fit by chance, and it will not survive a change of any other
parameter. Treat a wide spread of ev_rmse across the grid as the finding it is: the
result depends more on the landscape draw than on the parameters.

Usage:
    python -m package.generating_data.seed_inputs_sweep_gen
    python -m package.generating_data.seed_inputs_sweep_gen --seeds 4          # quick check
    python -m package.generating_data.seed_inputs_sweep_gen --values 1 40      # grid range
"""
import sys

import numpy as np

from package.generating_data.sweep_hhi_age_gen import load_base_params
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder,
    load_object,
    params_list_with_seed,
    produce_name_datetime,
    save_object,
)

# Candidate landscapes. Contiguous from 1 rather than a scattered list so the grid
# carries no hidden selection: every integer in the range is scored and the whole
# table is saved, which is what makes the spread of ev_rmse readable as evidence
# about the model rather than about the search.
SEED_INPUTS_LIST = np.arange(1, 10)

# Replicates of `seed` per landscape. Seed noise has to be averaged down far enough
# that the RANKING is about the landscape, so this is the same 64 the other sweeps
# use. Cut it with --seeds for a screening pass, but do not pick a final value off
# a screening pass: at 8 seeds the ordering of near-neighbours is not resolved.
SEEDS = 64

# Copied from calibration/NN_multi_round_calibration_multi_gen.py rather than
# imported, because that module imports torch and sbi at module scope and this job
# has no use for either. Keep in step if the calibration convention changes.
STOCK_MONTH_OFFSET = 3      # APRIL, the month the EV stock (population) data refers to
LOGIT_EPS = 1e-4            # same floor the SBI runs use, so scores are comparable

# First year of calibration_data_output["EV Prop"], set by the 2010-2023 filter in
# calibration/calibration_data_outputs.py. The array carries no year labels, so
# this is the only thing tying its entries to model time steps.
DATA_START_YEAR = 2010

# From plotting_data/calibration_plot.py, itself from the manuscript's target table.
TARGET_HHI = (0.11, 0.18)
TARGET_FLEET_AGE_YEARS = (10, 12)


def _year_start_index(year, base_params):
    """Index of January of `year` in a monthly series that starts at the burn-in."""
    return (year - 2001) * 12 + base_params["duration_burn_in"]


def _logit(p):
    """Logit with the calibration's epsilon floor, so p = 0 is finite rather than -inf."""
    p = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def ev_stock_rmse(hist_prop_ev, base_params, observed):
    """
    RMSE of the modelled EV stock trajectory against `observed`, in logit units.

    hist_prop_ev is (seeds, T) and NOT burn-in trimmed, because _year_start_index
    counts from the start of the array. One April reading per observed year.

    ORDER OF OPERATIONS MATTERS and follows build_x() in the SBI generator: the
    logit is taken PER SEED and averaged afterwards. Averaging the shares first and
    transforming once is a different statistic, and near zero -- where most of this
    series lives -- a very different one.
    """
    years = np.arange(DATA_START_YEAR, DATA_START_YEAR + len(observed))
    idx = [_year_start_index(y, base_params) + STOCK_MONTH_OFFSET for y in years]
    model_logit = _logit(np.asarray(hist_prop_ev, dtype=float)[:, idx]).mean(axis=0)
    return float(np.sqrt(np.mean((model_logit - _logit(observed))**2)))


def final_year_mean(arr_2d):
    """Mean over seeds of the last 12 recorded steps, i.e. the 2023 calendar year."""
    return float(np.nanmean(np.asarray(arr_2d, dtype=float)[:, -12:]))


def in_band(value, band):
    return band[0] <= value <= band[1]


def parse_args(argv):
    """`--seeds N` and `--values LO HI`, so a screening pass needs no file edit."""
    args = list(argv)
    seeds, values = SEEDS, SEED_INPUTS_LIST
    if "--seeds" in args:
        seeds = int(args[args.index("--seeds") + 1])
    if "--values" in args:
        i = args.index("--values")
        values = np.arange(int(args[i + 1]), int(args[i + 2]) + 1)
    return values, seeds


def build_params(seed_inputs, seeds):
    """Base params with ONLY seed_inputs and the replicate count touched."""
    bp = load_base_params()
    bp["seed_inputs"] = int(seed_inputs)
    bp["seed_repetitions"] = seeds
    return bp


def run_sweep(values, seeds):
    values = np.asarray(values, dtype=int)
    observed = np.asarray(
        load_object("package/calibration_data", "calibration_data_output")["EV Prop"],
        dtype=float,
    )
    base_params = build_params(values[0], seeds)

    root = produce_name_datetime("seed_inputs_sweep")
    createFolder(root)

    print(f"fileName: {root}")
    print(f"seed_inputs: {len(values)} values from {values.min()} to {values.max()}, "
          f"seeds each: {seeds}, total runs: {len(values) * seeds}")
    print(f"scoring EV stock {DATA_START_YEAR}-"
          f"{DATA_START_YEAR + len(observed) - 1} in logit units, "
          f"observed 2023 = {observed[-1]:.4f}")
    print(f"the JSON currently pins seed_inputs = "
          f"{load_base_params()['seed_inputs']}\n")
    header = (f"{'seed_inputs':>12}{'ev_rmse':>9}{'EV 2023':>10}{'HHI':>9}"
              f"{'age yr':>9}{'bands':>10}")
    print(header)
    print(f"{'TARGET':>12}{'0':>9}{observed[-1]:>10.4f}"
          f"{'.11-.18':>9}{'10-12':>9}")
    print("-" * len(header))

    ev_all, hhi_all, age_all, rows = [], [], [], []
    for value in values:
        bp = build_params(value, seeds)
        outputs = parallel_run_multi_seed(params_list_with_seed(bp))

        ev = np.asarray(outputs["history_prop_EV"], dtype=float)
        hhi = np.asarray(outputs["history_market_concentration"], dtype=float)
        age = np.asarray(outputs["history_mean_car_age_fleet"], dtype=float)

        row = {
            "seed_inputs": int(value),
            "ev_rmse": ev_stock_rmse(ev, bp, observed),
            "ev_2023": final_year_mean(ev),
            "hhi": final_year_mean(hhi),
            "age_yr": final_year_mean(age) / 12,
        }
        row["hhi_ok"] = in_band(row["hhi"], TARGET_HHI)
        row["age_ok"] = in_band(row["age_yr"], TARGET_FLEET_AGE_YEARS)
        rows.append(row)

        # Trimmed to the calibration window for saving, so index 0 is the end of
        # burn-in and every saved array shares one time origin. Scoring above uses
        # the UNTRIMMED array, because _year_start_index counts from index 0.
        burn_in = bp["duration_burn_in"]
        ev_all.append(ev[:, burn_in:])
        hhi_all.append(hhi[:, burn_in:])
        age_all.append(age[:, burn_in:])

        # Dropped explicitly rather than left to rebinding on the next iteration:
        # `outputs` holds every MULTI_SEED_ARRAY_KEYS series plus the cars_on_sale
        # snapshot for every seed, and plain rebinding would build the next value's
        # dict while this one is still referenced, so the parent would peak at two
        # of them on top of the live workers.
        del outputs

        flags = ("HHI" if not row["hhi_ok"] else "") + ("age" if not row["age_ok"] else "")
        print(f"{row['seed_inputs']:>12d}{row['ev_rmse']:>9.3f}{row['ev_2023']:>10.4f}"
              f"{row['hhi']:>9.3f}{row['age_yr']:>9.1f}"
              f"{(flags or 'both ok'):>10}", flush=True)

    save_object(np.asarray(ev_all), root + "/Data", "data_array_ev_prop")
    save_object(np.asarray(hhi_all), root + "/Data", "data_array_hhi")
    save_object(np.asarray(age_all), root + "/Data", "data_array_age_fleet")
    save_object(values, root + "/Data", "param_list")
    save_object("seed_inputs", root + "/Data", "param_name")
    save_object(r"$\mathrm{seed\_inputs}$", root + "/Data", "param_label")
    save_object(base_params, root + "/Data", "base_params")
    save_object(observed, root + "/Data", "observed_ev_prop")
    save_object(rows, root + "/Data", "summary")

    report(rows, seeds)
    print(f"wrote {root}")
    return root, rows


def report(rows, seeds):
    """Rank by ev_rmse, band-passing rows first, and name the pick."""
    passing = sorted((r for r in rows if r["hhi_ok"] and r["age_ok"]),
                     key=lambda r: r["ev_rmse"])
    failing = sorted((r for r in rows if not (r["hhi_ok"] and r["age_ok"])),
                     key=lambda r: r["ev_rmse"])

    def line(r):
        flags = ("HHI " if not r["hhi_ok"] else "") + ("age" if not r["age_ok"] else "")
        return (f"  seed_inputs = {r['seed_inputs']:>3d}   ev_rmse {r['ev_rmse']:.3f}"
                f"   EV 2023 {r['ev_2023']:.4f}   HHI {r['hhi']:.3f}"
                f"   age {r['age_yr']:.1f} yr   {flags or 'both bands ok'}")

    print(f"\n{len(passing)} of {len(rows)} landscapes are inside BOTH bands.")
    print("\nbest 5 inside both bands, by EV trajectory fit:")
    print("\n".join(line(r) for r in passing[:5]) or "  none")
    if failing:
        print("\nbest 5 that miss a band, for reference:")
        print("\n".join(line(r) for r in failing[:5]))

    spread = max(r["ev_rmse"] for r in rows) - min(r["ev_rmse"] for r in rows)
    print(f"\nev_rmse spans {spread:.3f} logit units across the landscape draw "
          f"alone, at unchanged parameters.")
    if passing:
        print(f"\nPICK: seed_inputs = {passing[0]['seed_inputs']}")
    else:
        print("\nPICK: none. No landscape in this grid satisfies both bands, so the "
              "parameters need refitting on this code path. Do NOT widen the grid "
              "until something passes.")
    if seeds < SEEDS:
        print(f"NOTE: run at {seeds} seeds, not {SEEDS}. This is a screening pass; "
              f"confirm the pick at {SEEDS} before writing it into the JSON.")


if __name__ == "__main__":
    run_sweep(*parse_args(sys.argv[1:]))
