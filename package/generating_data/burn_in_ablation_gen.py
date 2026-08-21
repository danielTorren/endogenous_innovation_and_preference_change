"""
Cumulative ablation over the burn-in initialisation.

The model starts every one of the 3000 users on a brand-new car and every firm one
bit-flip from the same worst design. Both are shocks the burn-in then has to
absorb, and 180 months is not enough:

    end of burn-in   7.17 yr        (64-seed mean, run 22_32_39__12_08_2026)
    2005             9.50 yr
    2020            10.70 yr
    2023             9.50 yr

drifting +0.98 yr/decade across the whole calibration window. The cause is
structural rather than a matter of tuning: L_a_t starts at 0 for everyone on month
0 and ticks up one per month, so the oldest car that can exist at month t is t
months old. At the end of burn-in nothing is older than 15 years and the right
tail of the age distribution is simply absent. What 2001-2023 shows is that tail
filling in, which is why the drift runs for the entire window instead of settling.
Lengthening burn-in cannot fix this without going to 30+ years of it.

So the rungs are added cumulatively, cheapest first:

  research     duration_burn_in_research > 0. Already implemented and wired
               (controller -> firm_manager.next_step_burn_in) but set to 0 in every
               config, so it has never run. Firms innovate and re-choose production
               against each other with no consumers and no used market, which is
               far cheaper per step than a full step and lets the tech climb happen
               without simulating 3000 users.
  placement    firms drawn without replacement from the worst 5% of the sampled
               landscape, off the `seed` substream. The old draw takes 10 from the
               15 one-bit-flip neighbours of a single design with replacement and
               off seed_inputs -- so ~2.5 firms are exact duplicates and, because
               seed_inputs is pinned, every replicate starts identically. The NK
               landscape itself stays pinned to seed_inputs either way.
  fleet_age    starting ages drawn from a distribution instead of all zero, and the
               initial assignment priced accordingly (old cars are cheaper, and
               have decayed range and efficiency), so cheap old cars land with
               low-beta users.
  sellable     the starting fleet may be traded in rather than destroyed, and the
               12-month second-hand suppression is lifted. Only sensible after
               fleet_age: with 3000 simultaneous age-0 cars the trade-ins would all
               arrive at once, which is what the init_car flag was blocking.

Judge these on the SHAPE of the age transient, not on distance to the 10-12 yr
target. Mean fleet age is one of the calibration targets, so the free parameters
would move to absorb any level shift anyway; what cannot be absorbed is a series
that is still climbing when the calibration window ends. The columns to watch are
age_bi (age at the end of burn-in) and slope (yr per decade across calibration) --
the goal is age_bi already in range and slope flat or gently positive.

Usage:
    python -m package.generating_data.burn_in_ablation_gen                 # full ladder
    python -m package.generating_data.burn_in_ablation_gen b0_base b3_age  # named subset
    python -m package.generating_data.burn_in_ablation_gen --seeds 4
"""
import json
import sys

import numpy as np

from package.generating_data.calibration_gen import SAVED_KEYS
from package.resources.run import parallel_run_multi_seed
from package.resources.utility import (
    createFolder,
    load_object,
    params_list_with_seed,
    produce_name_datetime,
    save_object,
)

BASE_PARAMS_LOAD = "package/constants/base_params_calibration.json"
SEEDS = 14

# Mean starting age, in months. This must be the model's OWN equilibrium fleet age,
# not the 10-12 yr calibration target -- the point of seeding the distribution is to
# start where the burn-in would have ended, so that burn-in can then be shortened.
# Seeding below equilibrium just re-runs the same climb from a different offset.
#
# Measured at duration_burn_in = 360 (single seed, current base_params_calibration):
#   mean 16.0 yr, sd 11.4, median 14.0, p90 33.5, max 49.0
# so ~192 and ~129 months. RE-MEASURE THESE whenever delta_P, scrap_price,
# age_limit_second_hand or prob_switch_car change -- they are a property of those,
# and those are exactly what is being tuned.
INIT_AGE_MEAN = 192
INIT_AGE_STD = 129

# The shipped configs now carry the new initialisation switched on. Every rung is
# meant to be measured against the ORIGINAL behaviour, so reset all of it to off
# before applying features rather than inheriting whatever the config file happens
# to hold. Without this the ladder silently re-baselines whenever the JSON is
# edited -- which has already produced one mislabelled control row.
FEATURES_OFF = {
    "duration_burn_in_research": 0,
    "parameters_firm_manager": {
        "init_car_age_mean": 0,
        "init_car_age_std": 0,
        "init_car_sellable": False,
        "init_firm_placement": "hamming1",
    },
    "parameters_second_hand": {"burn_in_second_hand_market": 12},
}

# One entry per rung. Each is a set of overrides applied on top of the ones before
# it, so the configs below are prefixes of this list.
FEATURES = {
    "research": {
        "duration_burn_in_research": 180,
    },
    "placement": {
        "parameters_firm_manager": {
            "init_firm_placement": "worst_pool",
            "init_firm_pool_prop": 0.05,
        },
    },
    "fleet_age": {
        "parameters_firm_manager": {
            "init_car_age_mean": INIT_AGE_MEAN,
            "init_car_age_std": INIT_AGE_STD,
            "init_car_age_max": 600,
        },
    },
    "sellable": {
        "parameters_firm_manager": {"init_car_sellable": True},
        "parameters_second_hand": {"burn_in_second_hand_market": 0},
    },
}

LADDER = ["research", "placement", "fleet_age", "sellable"]

# (features, duration_burn_in override or None to keep the file's 180).
#
# The long-burn-in rows are the reference: at 360+ months the fleet has reached its
# own equilibrium, so they show what the initialisation is trying to reproduce. The
# whole point of the ladder is to hit those numbers at 180 months, or fewer.
CONFIGS = {
    "b0_base": ([], None),
    "b1_research": (LADDER[:1], None),
    "b2_placement": (LADDER[:2], None),
    "b3_age": (LADDER[:3], None),
    "b4_sellable": (LADDER[:4], None),
    # fleet_age on its own, to separate the age fix from the firm-side rungs.
    "age_only": (["fleet_age"], None),
    "age_sell_only": (["fleet_age", "sellable"], None),
    # Reference: converged fleet, no initialisation tricks, at 2x and 3.3x the cost.
    "long_360": ([], 360),
    "long_600": ([], 600),
    # The payoff, if the ladder works: equilibrium behaviour at a fraction of the
    # burn-in, which is what makes the calibration sweeps affordable.
    "short_90": (LADDER[:4], 90),
    "short_36": (LADDER[:4], 36),
    # How much firm-only research the tech climb actually needs, held at the
    # recommended duration_burn_in of 90. The full ladder fixes duration_burn_in;
    # this is the other half of the pair and is otherwise untested.
    "res000_bi90": (LADDER[:4], 90, {"duration_burn_in_research": 0}),
    "res060_bi90": (LADDER[:4], 90, {"duration_burn_in_research": 60}),
    "res360_bi90": (LADDER[:4], 90, {"duration_burn_in_research": 360}),
    "res720_bi90": (LADDER[:4], 90, {"duration_burn_in_research": 720}),
}

RESEARCH_SWEEP = ["res000_bi90", "res060_bi90", "short_90", "res360_bi90", "res720_bi90"]

# stretch_Cost sweep at the recommended configuration.
#
# stretch_Cost was raised to 2.0 while firms all started one bit-flip from a single
# design, which collapsed new-car price dispersion to 0.54x of target -- so the
# stretch was compensating for a placement artefact rather than describing the
# market. With worst_pool placement the dispersion is produced by the landscape
# itself (1.29x at stretch_Cost 2.0), so the two are now double-counting and the
# stretch should come back down. Target is priceIQR 1.00x.
for _sc in (1.5, 1.75, 2.0, 2.25, 2.5):
    CONFIGS[f"sc{int(_sc*100):03d}"] = (
        LADDER[:4], 90,
        {"parameters_ICE": {"stretch_Cost": _sc}, "parameters_EV": {"stretch_Cost": _sc}},
    )

# Range chosen upward: an earlier sweep put the 1.00x crossing at ~1.72, but that
# was measured before EVs were put on sale from 2001 (ev_production_start_time 0).
# At 1.75 the current config gives 0.81x, so the crossing has moved higher.
STRETCH_SWEEP = ["sc150", "sc175", "sc200", "sc225", "sc250"]

# Observed mean new-car TRANSACTION price in the model's 2020 dollars. See the
# price_med_x comment in summarise() for the derivation.
TARGET_NEW_PRICE_MEAN = 39644.0

# delta sweep: utility decay per month, against delta_P = 0.0087 for price decay.
#
# At the shipped delta = 0.00222 a 16-year-old car retains (1-delta)^192 = 0.653 of
# its utility while retaining only (1-delta_P)^192 = 0.187 of its price. A used car
# costs 19% of new and delivers 65% of the value, which is why 78% of purchases go
# used, the fleet ages to 16 years, and the only people still buying new are the
# wealthy tail -- so firms price for them and the new-car mass ends up at the
# expensive end with the skew backwards.
#
# Raising delta removes that bargain, and it is the one side of the ratio that is
# not off-limits. 0.0087 makes utility decay exactly track price decay. Note the
# controller copies delta onto the EV landscape, so this moves both drivetrains,
# and delta also enters the driving-cost term -- old cars get worse on fuel economy
# and on range at once, so the response will not be linear.
# HARD CEILING: controller.gen_gamma() requires r > delta/(1-delta), so with
# r = 0.00407412378 the model refuses any delta >= 0.0040576. Matching delta_P
# exactly is therefore impossible without also raising r -- the usable range is
# only 0.00222 to 0.00406, under a factor of two.
#
# That range is enough, because the response inside it is violent. At 0.00222 the
# model sits at 79% used purchases, a 16-yr fleet and EV 0.009; at 0.00400 it is at
# 7% used, a 5-yr fleet and EV 0.112. Every target of interest crosses its band
# somewhere in between, so the sweep is deliberately dense rather than wide.
#
# Note delta did NOT behave as a single fix: it moves fleet age, used share and EV
# uptake strongly, but the new-car price level gets WORSE as it rises (pMed 1.34x
# -> 1.59x), because collapsing the used market forces everyone into the new market
# and firms price into that demand. Price level needs its own lever.
DELTA_MAX = 0.0040576
for _d in (0.00222, 0.0025, 0.0028, 0.0031, 0.0034, 0.0037, 0.0040):
    assert _d < DELTA_MAX, f"delta {_d} exceeds the r-implied ceiling {DELTA_MAX}"
    CONFIGS[f"delta{int(round(_d*100000)):05d}"] = (
        LADDER[:4], 90, {"parameters_ICE": {"delta": _d}},
    )

for _d in (0.0029, 0.0031):
    for _sc in (1.75, 2.00, 2.25):
        CONFIGS[f"d{int(round(_d*100000)):04d}_sc{int(_sc*100):03d}"] = (
            LADDER[:4], 90,
            {"parameters_ICE": {"delta": _d, "stretch_Cost": _sc},
             "parameters_EV": {"stretch_Cost": _sc}},
        )

JOINT_SWEEP = [f"d{d:04d}_sc{s:03d}" for d in (290, 310) for s in (175, 200, 225)]

DELTA_SWEEP = ["delta00222", "delta00250", "delta00280", "delta00310",
               "delta00340", "delta00370", "delta00400"]

DEFAULT_CONFIGS = ["b0_base", "b1_research", "b2_placement", "b3_age", "b4_sellable",
                   "long_360", "short_90", "short_36"]


# Final tuning grid on the minimal-change config. EV max_Cost is raised on every
# row (battery packs give EVs a higher cost ceiling than ICE -- a one-sentence
# justification, unlike the correlation structure). The open questions are how
# much stretch_Cost is needed to stop the new-car price spread being 5x too
# narrow, and whether delta wants nudging down to lift fleet age back into band.
_EVMAX = {"parameters_EV": {"max_Cost": 100000}}
def _cfg(sc=None, delta=None):
    o = {"parameters_EV": dict(_EVMAX["parameters_EV"]), "parameters_ICE": {}}
    if sc is not None:
        o["parameters_ICE"]["stretch_Cost"] = sc; o["parameters_EV"]["stretch_Cost"] = sc
    if delta is not None:
        o["parameters_ICE"]["delta"] = delta
    return (LADDER[:4], 90, o)

CONFIGS["m1_evmax"]        = _cfg()
CONFIGS["m2_sc175"]        = _cfg(sc=1.75)
CONFIGS["m3_sc250"]        = _cfg(sc=2.50)
CONFIGS["m4_sc350"]        = _cfg(sc=3.50)
CONFIGS["m5_sc250_d29"]    = _cfg(sc=2.50, delta=0.0029)
CONFIGS["m6_sc250_d27"]    = _cfg(sc=2.50, delta=0.0027)
CONFIGS["m7_sc175_d27"] = _cfg(sc=1.75, delta=0.0027)
CONFIGS["m8_sc200_d27"] = _cfg(sc=2.00, delta=0.0027)
FINAL_GRID = ["m1_evmax","m2_sc175","m3_sc250","m4_sc350","m5_sc250_d29","m6_sc250_d27"]


# WTP_E_mean sweep: does environmental willingness-to-pay actually move EV uptake?
# It is the natural NN lever because it acts through preferences, not prices, so it
# cannot invert the ICE-cheaper-than-EV ordering the way EV.max_Cost can.
for _w in (0, 25000, 46647, 80000, 120000):
    CONFIGS[f"wtp{_w:06d}"] = (LADDER[:4], 90,
        {"parameters_social_network": {"WTP_E_mean": _w if _w else 1.0}})
WTP_SWEEP = ["wtp000000","wtp025000","wtp046647","wtp080000","wtp120000"]


# nu (battery/range weight) and zeta (its exponent). EVs and ICEs differ mainly in
# B -- battery pack vs fuel tank -- so this is the structural EV lever that acts
# through perceived range rather than through price, and so cannot invert the
# ICE-cheaper-than-EV ordering the way EV.max_Cost can.
for _n in (300, 1174, 3000, 8000):
    CONFIGS[f"nu{_n:05d}"] = (LADDER[:4], 90, {"parameters_social_network": {"nu": _n}})
NU_SWEEP = ["nu00300","nu01174","nu03000","nu08000"]


# EV.max_Cost sweep WITH the cost-quality correlation switched back on.
#
# rho[1]=0.5 was dropped because it pushed ICE/EV from 0.954 to 1.104 -- but that
# was measured at EV.max_Cost = 70000. With EV.max_Cost now at 100000 the ratio
# sits near 0.70, so the correlation's price-ordering cost may now be affordable,
# and it is worth +0.013 on EV uptake plus gains on pIQR and HHI. rho[3] stays at 0
# (measured null). This sweep asks how much EV.max_Cost headroom that leaves.
for _m in (70000, 85000, 100000, 120000, 140000):
    CONFIGS[f"corr_evmax{_m//1000:03d}"] = (LADDER[:4], 90, {
        "parameters_ICE": {"rho": [1, 0.5, 0]},
        "parameters_EV": {"rho": [1, 0.5, 0, 0], "max_Cost": _m},
    })
CORR_EVMAX_SWEEP = [f"corr_evmax{m//1000:03d}" for m in (70000,85000,100000,120000,140000)]


def build_params(name, seeds):
    with open(BASE_PARAMS_LOAD) as f:
        bp = json.load(f)
    features, burn_in, *rest = CONFIGS[name]
    extra = rest[0] if rest else None
    for key, val in FEATURES_OFF.items():
        if isinstance(val, dict):
            bp.setdefault(key, {}).update(val)
        else:
            bp[key] = val
    for feature in features:
        for key, val in FEATURES[feature].items():
            if isinstance(val, dict):
                bp.setdefault(key, {}).update(val)
            else:
                bp[key] = val
    if burn_in is not None:
        bp["duration_burn_in"] = burn_in
    for key, val in (extra or {}).items():
        if isinstance(val, dict):
            bp.setdefault(key, {}).update(val)
        else:
            bp[key] = val
    bp["seed_repetitions"] = seeds
    # Sampling stays off, matching ablation_gen, so the attribution is not
    # confounded by consideration-set noise.
    bp["parameters_social_network"].pop("num_new_considered", None)
    bp["parameters_social_network"].pop("num_second_hand_considered", None)
    return bp


def summarise(name, bp, outputs):
    """Calibration-target metrics plus the burn-in age-transient diagnostics."""
    b = bp["duration_burn_in"]
    A = lambda k: np.asarray(outputs[k], dtype=float)[:, b:]
    fy = lambda x: float(np.nanmean(x[..., -12:]))

    age_full = np.asarray(outputs["history_mean_car_age_fleet"], dtype=float)
    age_cal = age_full[:, b:] / 12
    # Age the fleet has reached by the time the calibration window opens. If this
    # is already in the 10-12 yr band the starting shock has been absorbed.
    age_bi = float(np.nanmean(age_full[:, b - 1]) / 12) if b > 0 else 0.0
    # Drift across the calibration window, per decade. Flat is the goal; the
    # baseline runs at about +1.0.
    months = np.arange(age_cal.shape[1])
    slope = float(np.polyfit(months, np.nanmean(age_cal, axis=0), 1)[0]*120)

    # [[new_ICE, new_EV], [used_ICE, used_EV]] mean prices over the cars on sale.
    # ICE is meant to be the cheaper drivetrain on average, so this ratio belongs
    # below 1; at 1.00 or above the model has EVs undercutting ICE, which inverts
    # the whole adoption story regardless of what the uptake number says.
    mp = np.asarray(outputs["history_mean_price_ICE_EV"], dtype=float)[:, b:, :, :]

    nq = np.asarray(outputs["history_new_car_price_quantiles"], dtype=float)[:, b:, :]
    uq = np.asarray(outputs["history_used_car_price_quantiles"], dtype=float)[:, b:, :]
    counts = np.asarray(outputs["history_purchase_counts"], dtype=float)[:, b:, :]
    n, u, opp = (counts[:, -12:, i].sum(axis=1) for i in range(3))

    return {
        "config": name,
        "age_bi": age_bi,
        "age_slope": slope,
        "age_yr": fy(A("history_mean_car_age_fleet")) / 12,
        "age_sd": float(np.nanstd(np.nanmean(age_cal[:, -12:], axis=1))),
        "ev_2023": fy(A("history_prop_EV")),
        "hhi": fy(A("history_market_concentration")),
        "price_iqr_x": fy(nq[:, :, 2] - nq[:, :, 0]) / (57784.66 - 32359.41),
        # Spread alone hides a distribution that is the right width but sitting in
        # the wrong place: a run can hit price_iqr_x 1.04x with both quartiles
        # inside 6% and still have its mass at the wrong end.
        #
        # Denominator is the OBSERVED MEAN, not the P25/P75 midpoint. Grieco, Murry
        # & Yurukoglu 2024 Fig II give a mean of $34,000 in 2015$, which the x1.166
        # deflator that reproduces the P25/P75 constants puts at $39,644 in the
        # model's 2020$. No median is published, but that mean sits well below the
        # midpoint ($45,072), so the real distribution is right skewed and its
        # median is BELOW its mean. Dividing by the mean therefore UNDERSTATES how
        # high the model sits -- treat price_med_x as a lower bound on the miss.
        "price_med_x": fy(nq[:, :, 1]) / TARGET_NEW_PRICE_MEAN,
        # Which side of the median is longer. Real new-car prices are right skewed
        # (cheap mass, expensive tail), so the observed value is well under 1 --
        # roughly 0.2-0.3. Above 1 means the model has it backwards: mass bunched
        # at the expensive end with a long cheap tail.
        "price_skew": (fy(nq[:, :, 1]) - fy(nq[:, :, 0])) / (fy(nq[:, :, 2]) - fy(nq[:, :, 1])),
        "ice_ev_new": fy(mp[:, :, 0, 0]) / fy(mp[:, :, 0, 1]),
        "qual_price_ratio": fy(A("history_used_stock_quality_spread"))
                            / fy(uq[:, :, 2] - uq[:, :, 0]),
        "used_share": float(np.mean(u / (n + u))),
        "prob_buy": float(np.mean((n + u) / opp)),
    }


# q/p is printed alongside priceIQR because the two move together and only make
# sense read as a pair. Compressing stretch_Cost pulls priceIQR down towards its
# 1.00x target, but it can do so by flattening the quality-cost ladder rather than
# by tightening the market -- and that shows up in q/p, not in priceIQR. A run that
# hits priceIQR 1.00x with q/p well off 1 has bought the price target by making the
# expensive end of the market not worth buying.
HEADER = (f"{'config':<16}{'age@BI':>8}{'slope':>8}{'age 23':>8}"
          f"{'EV 2023':>9}{'HHI':>7}{'pIQR':>7}{'pMed':>7}{'pSkew':>7}"
          f"{'q/p':>7}{'ICE/EV':>8}{'usedsh':>8}{'P(buy)':>8}")
TARGETS = (f"{'TARGET':<16}{'10-12':>8}{'~0':>8}{'10-12':>8}"
           f"{0.038:>9.3f}{'.11-.18':>7}{'1.00x':>7}{'1.00x':>7}{'0.25':>7}"
           f"{'~1':>7}{'<1':>8}{'0.67':>8}{'0.16':>8}")


def print_row(r):
    print(f"{r['config']:<16}{r['age_bi']:>8.2f}{r['age_slope']:>+8.2f}"
          f"{r['age_yr']:>8.2f}"
          f"{r['ev_2023']:>9.3f}{r['hhi']:>7.3f}"
          f"{r['price_iqr_x']:>6.2f}x{r['price_med_x']:>6.2f}x{r['price_skew']:>7.2f}"
          f"{r['qual_price_ratio']:>7.2f}{r['ice_ev_new']:>8.3f}"
          f"{r['used_share']:>8.3f}{r['prob_buy']:>8.3f}", flush=True)


def main(configs=None, seeds=SEEDS):
    configs = configs or DEFAULT_CONFIGS
    root = produce_name_datetime("burn_in_ablation_gen")
    createFolder(root)
    print(f"fileName: {root}\nseeds per config: {seeds}\n")
    print(HEADER)
    print(TARGETS)
    print("-" * len(HEADER))

    rows = []
    for name in configs:
        bp = build_params(name, seeds)
        outputs = parallel_run_multi_seed(params_list_with_seed(bp))
        saved = {k: outputs[k] for k in SAVED_KEYS}
        # Saved per config so calibration_plot can be pointed at any single row.
        sub = f"{root}/{name}"
        createFolder(sub)
        save_object(saved, sub + "/Data", "outputs")
        save_object(bp, sub + "/Data", "base_params")
        row = summarise(name, bp, saved)
        rows.append(row)
        print_row(row)

    save_object(rows, root + "/Data", "ablation_summary")
    print(f"\nwrote {root}")
    print("plot any single row with:")
    print(f"  python -c \"from package.plotting_data.calibration_plot import main;"
          f" main('{root}/b4_sellable')\"")
    return root


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    seeds = SEEDS
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = int(args[i + 1])
        del args[i:i + 2]
    unknown = [a for a in args if a not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}, choose from {sorted(CONFIGS)}")
    main(configs=args or None, seeds=seeds)
