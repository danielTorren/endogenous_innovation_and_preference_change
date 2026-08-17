"""
Dominance analysis for the endogenous policy-pair results.

A policy is DOMINATED if some other policy achieves, at the same time,
    lower cumulative emissions,
    lower cumulative net cost,
    higher cumulative utility.
The non-dominated policies form the Pareto front.

Two things make this non-trivial here, and both are handled explicitly:

1. ELIGIBILITY. Every candidate in these runs is the result of driving EV
   uptake to a target. Some candidates never reach the target. Comparing a
   cheap policy that misses the target against an expensive one that hits it is
   meaningless, so candidates outside the EV-uptake window are excluded from
   the dominance test and reported separately.

2. SEED NOISE. Each candidate is a mean over the same set of calibration
   seeds. Dominance on the means alone can flip under resampling, so the front
   is also bootstrapped over seeds. Because all candidates share the same
   seeds, the resampling is PAIRED, which removes most of the common noise.

Run it on one or more endog_pair folders from the gen script (or on a compiled
folder from endogenous_policy_intensity_pair_plot.main):

    python -m package.analysis.policy_dominance

Outputs, all written to the first input folder, in <folder>/Plots/dominance/:
    dominance_report.txt        everything printed to stdout
    candidates.csv              the full candidate table
    dominance_pairs.csv         every (dominator, dominated) relation
    fig1_pareto_scatter.png     criterion pairs, front vs dominated
    fig1b_pareto_scatter_zoom.png   same, extremes clipped
    fig2_membership.png         bootstrap Pareto-membership probability
    fig3_dominance_heatmap.png  seed-level P(A dominates B)
    fig4_instrument_survival.png
    fig5_dominator_counts.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from package.resources.utility import load_object, save_object
from package.analysis.endogenous_policy_intensity_pair_plot import (
    half_circle_marker, full_circle_marker
)

OKABE_ITO = ['#E69F00', '#009E73', '#56B4E9', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

POLICY_TITLES = {
    "Carbon_price": "Carbon Price",
    "Electricity_subsidy": "Electricity Subsidy",
    "Adoption_subsidy": "New Car Rebate",
    "Adoption_subsidy_used": "Used Car Rebate",
    "Production_subsidy": "Production Subsidy",
}

POLICY_ABBREV = {
    "Carbon_price": "CP",
    "Electricity_subsidy": "ES",
    "Adoption_subsidy": "NCR",
    "Adoption_subsidy_used": "UCR",
    "Production_subsidy": "PS",
}

# Criterion bookkeeping. sense = +1 means lower is better.
CRITERIA = [
    ("E", "Cumulative Emissions, MTCO2", +1),
    ("C", "Cumulative Net Cost, bn $", +1),
    ("U", "Cumulative Utility, bn $", -1),
]


def abbrev(policy):
    return POLICY_ABBREV.get(policy, policy[:3].upper())


# ----------------------------------------------------------------------------
# Candidate table
# ----------------------------------------------------------------------------

def _seed_arrays(entry, utility_scale):
    """Per-seed (E, C, U) for one outcome dict, in plotting units."""
    if "emissions_cumulative" in entry:
        em = np.asarray(entry["emissions_cumulative"], dtype=float)
    else:
        em = (np.asarray(entry["emissions_cumulative_driving"], dtype=float)
              + np.asarray(entry["emissions_cumulative_production"], dtype=float))
    E = em * 1e-9
    C = np.asarray(entry["net_cost"], dtype=float) * 1e-9
    U = np.asarray(entry["utility_cumulative"], dtype=float) * utility_scale * 1e-9
    return E, C, U


def build_candidate_table(base_params, pairwise_outcomes, single_outcomes,
                          outcomes_BAU, ev_min, ev_max, include_bau=True):
    """
    Flatten pair sweeps, single policies and BAU into one table.

    Returns (df, seeds) where df has one row per candidate and seeds is a dict
    of (n_candidates, n_seeds) arrays keyed "E"/"C"/"U", aligned to df.index.
    Rows without per-seed data (BAU) are dropped from the eligible set.
    """
    utility_scale = 1.0 / base_params["parameters_social_network"]["prob_switch_car"]

    rows, seed_E, seed_C, seed_U = [], [], [], []

    for (policy1, policy2), results in pairwise_outcomes.items():
        for step, entry in enumerate(results):
            E, C, U = _seed_arrays(entry, utility_scale)
            rows.append(dict(
                label=f"{abbrev(policy1)}+{abbrev(policy2)}[{step}]",
                kind="pair",
                policy1=policy1,
                policy2=policy2,
                step=step,
                p1_value=float(entry["policy1_value"]),
                p2_value=float(entry["policy2_value"]),
                ev=float(entry["mean_ev_uptake"]),
                E=float(np.mean(E)), C=float(np.mean(C)), U=float(np.mean(U)),
            ))
            seed_E.append(E); seed_C.append(C); seed_U.append(U)

    for policy, entry in single_outcomes.items():
        E, C, U = _seed_arrays(entry, utility_scale)
        rows.append(dict(
            label=f"{abbrev(policy)} only",
            kind="single",
            policy1=policy,
            policy2=None,
            step=-1,
            p1_value=float(entry["optimized_intensity"]),
            p2_value=np.nan,
            ev=float(entry["mean_EV_uptake"]),
            E=float(np.mean(E)), C=float(np.mean(C)), U=float(np.mean(U)),
        ))
        seed_E.append(E); seed_C.append(C); seed_U.append(U)

    n_seeds = max(len(a) for a in seed_E)

    if include_bau and outcomes_BAU is not None:
        rows.append(dict(
            label="BAU",
            kind="BAU",
            policy1=None, policy2=None, step=-1,
            p1_value=np.nan, p2_value=np.nan,
            ev=float(outcomes_BAU["mean_EV_uptake"]),
            E=float(outcomes_BAU["mean_emissions_cumulative"]) * 1e-9,
            C=float(outcomes_BAU["mean_net_cost"]) * 1e-9,
            U=float(outcomes_BAU["mean_utility_cumulative"]) * utility_scale * 1e-9,
        ))
        # BAU is saved without per-seed arrays, so it cannot be bootstrapped.
        nan = np.full(n_seeds, np.nan)
        seed_E.append(nan); seed_C.append(nan); seed_U.append(nan)

    df = pd.DataFrame(rows)

    bad = [len(a) for a in seed_E if len(a) != n_seeds]
    if bad:
        raise ValueError(
            f"candidates have different seed counts ({n_seeds} vs {set(bad)}); "
            "paired resampling assumes one common seed set"
        )

    seeds = {"E": np.vstack(seed_E), "C": np.vstack(seed_C), "U": np.vstack(seed_U)}
    df["has_seeds"] = ~np.isnan(seeds["E"]).any(axis=1)
    df["eligible"] = (df["ev"] >= ev_min) & (df["ev"] <= ev_max) & df["has_seeds"]
    return df, seeds, n_seeds


# ----------------------------------------------------------------------------
# Dominance
# ----------------------------------------------------------------------------

def dominance_matrix(E, C, U, eps=(0.0, 0.0, 0.0), margin_on="any"):
    """
    dom[i, j] is True when candidate i dominates candidate j.

    eps is a REQUIRED MARGIN, not a tolerance: raising it makes dominance
    harder to claim and so grows the front. i must never be worse than j on any
    criterion, and must beat it by more than eps on
        margin_on="any"  at least one criterion (a mild tightening), or
        margin_on="all"  every criterion (only clear-cut wins count).
    eps = 0 gives the textbook definition: weakly better on all three,
    strictly better on at least one. Passing the CI half-widths is a reasonable
    conservative setting.
    """
    E, C, U = np.asarray(E, float), np.asarray(C, float), np.asarray(U, float)
    eE, eC, eU = eps

    # i no worse than j on any criterion
    no_worse = ((E[:, None] <= E[None, :])
                & (C[:, None] <= C[None, :])
                & (U[:, None] >= U[None, :]))

    beats_E = E[:, None] < E[None, :] - eE
    beats_C = C[:, None] < C[None, :] - eC
    beats_U = U[:, None] > U[None, :] + eU

    if margin_on == "all":
        better = beats_E & beats_C & beats_U
    elif margin_on == "any":
        better = beats_E | beats_C | beats_U
    else:
        raise ValueError(f"margin_on must be 'any' or 'all', got {margin_on!r}")

    dom = no_worse & better
    np.fill_diagonal(dom, False)
    return dom


def bootstrap_membership(sE, sC, sU, n_boot=1000, seed=0, eps=(0.0, 0.0, 0.0),
                         margin_on="any"):
    """
    Probability each candidate stays on the Pareto front when the seeds are
    resampled. The same resampled seed index is used for every candidate, so
    the comparison stays paired and common noise cancels.
    """
    n_cand, n_seeds = sE.shape
    rng = np.random.default_rng(seed)
    hits = np.zeros(n_cand)
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, n_seeds)
        dom = dominance_matrix(sE[:, idx].mean(1), sC[:, idx].mean(1),
                              sU[:, idx].mean(1), eps=eps, margin_on=margin_on)
        hits += ~dom.any(axis=0)
    return hits / n_boot


def seedwise_dominance_probability(sE, sC, sU):
    """
    P[i, j] = fraction of seeds on which i beats j on all three criteria at
    once. This separates "dominated by a hair on the means" from "dominated in
    every single world". Paired by construction: seed k of i is compared with
    seed k of j.
    """
    win = ((sE[:, None, :] < sE[None, :, :])
           & (sC[:, None, :] < sC[None, :, :])
           & (sU[:, None, :] > sU[None, :, :]))
    P = win.mean(axis=2)
    np.fill_diagonal(P, 0.0)
    return P


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

class Reporter:
    """Print to stdout and collect the same text for a file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))

    def rule(self, title=None, char="="):
        if title is None:
            self(char * 78)
        else:
            self()
            self(char * 78)
            self(title)
            self(char * 78)

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def _fmt_table(df, cols, headers, floatfmt):
    widths = [max(len(h), *(len(floatfmt[c](df.iloc[i][c])) for i in range(len(df))))
              for c, h in zip(cols, headers)] if len(df) else [len(h) for h in headers]
    out = ["  ".join(h.rjust(w) for h, w in zip(headers, widths))]
    out.append("  ".join("-" * w for w in widths))
    for i in range(len(df)):
        row = df.iloc[i]
        out.append("  ".join(floatfmt[c](row[c]).rjust(w) for c, w in zip(cols, widths)))
    return out


def report(rep, df, dom, P_seed, membership, ev_min, ev_max, n_seeds, n_boot,
           eps, margin_on="any", marginal_lo=0.5, marginal_hi=0.95):
    """Print the whole story: eligibility, front, dominated set, robustness."""
    el = df[df["eligible"]].copy()

    rep.rule("DOMINANCE ANALYSIS")
    rep(f"Criteria      : lower emissions, lower net cost, higher utility")
    rep(f"EV window     : {ev_min:.3f} to {ev_max:.3f} mean uptake")
    rep(f"Seeds         : {n_seeds} shared across all candidates (paired comparisons)")
    rep(f"Bootstrap     : {n_boot} paired seed resamples")
    rep(f"Margin (eps)  : E {eps[0]:.4g}, C {eps[1]:.4g}, U {eps[2]:.4g}, "
        f"required on {margin_on} criteria")
    if not any(eps):
        rep("                eps = 0, so this is the textbook definition: "
            "weakly better on all three, strictly better on at least one.")

    # --- Eligibility -------------------------------------------------------
    rep.rule("1. ELIGIBILITY", "-")
    rep(f"Candidates total : {len(df)}  "
        f"({(df['kind'] == 'pair').sum()} pair points, "
        f"{(df['kind'] == 'single').sum()} single policies, "
        f"{(df['kind'] == 'BAU').sum()} BAU)")
    rep(f"Eligible         : {len(el)}")
    excluded = df[~df["eligible"]]
    rep(f"Excluded         : {len(excluded)}  (cannot be compared, they do not "
        f"reach the target)")
    if len(excluded):
        rep()
        rep("Excluded candidates, by how far they miss the window:")
        exc = excluded.sort_values("ev", ascending=False)
        for _, r in exc.iterrows():
            why = "no per-seed data" if not r["has_seeds"] else f"EV = {r['ev']:.3f}"
            rep(f"    {r['label']:<24} {why}")

    if len(el) == 0:
        rep("\nNothing eligible. Widen the EV window.")
        return el

    # --- Headline ----------------------------------------------------------
    n_front = int((~dom.any(axis=0)).sum())
    n_dominated = len(el) - n_front
    rep.rule("2. HEADLINE", "-")
    rep(f"{n_dominated} of {len(el)} eligible candidates are DOMINATED "
        f"({100 * n_dominated / len(el):.0f} %).")
    rep(f"{n_front} are non-dominated and form the Pareto front.")

    # --- The front ---------------------------------------------------------
    rep.rule("3. PARETO FRONT (non-dominated)", "-")
    front = el[el["n_dominators"] == 0].sort_values("E").copy()
    front["rank"] = np.arange(1, len(front) + 1)
    rep("'#' matches the numbers on the scatter figures.")
    rep()
    fmt = {
        "rank": lambda v: f"{int(v)}",
        "label": lambda v: str(v),
        "ev": lambda v: f"{v:.3f}",
        "E": lambda v: f"{v:.4f}",
        "C": lambda v: f"{v:+.4f}",
        "U": lambda v: f"{v:.3f}",
        "membership": lambda v: f"{v:.2f}",
        "n_beats": lambda v: f"{int(v)}",
        "p1_value": lambda v: f"{v:.3g}",
        "p2_value": lambda v: "-" if pd.isna(v) else f"{v:.3g}",
    }
    for line in _fmt_table(
            front,
            ["rank", "label", "ev", "E", "C", "U", "membership", "n_beats",
             "p1_value", "p2_value"],
            ["#", "candidate", "EV", "emissions", "net cost", "utility", "P(front)",
             "beats", "int.1", "int.2"], fmt):
        rep("    " + line)
    rep()
    rep("P(front) is the bootstrap probability the candidate stays non-dominated.")
    rep("'beats' counts how many other eligible candidates it dominates.")

    robust = front[front["membership"] >= marginal_hi]
    fragile = front[front["membership"] < marginal_hi]
    rep()
    rep(f"Robust front members (P >= {marginal_hi:.2f}) : {len(robust)}")
    if len(fragile):
        rep(f"Fragile front members (P <  {marginal_hi:.2f}) : {len(fragile)}  "
            "-- report these as uncertain, not as front members:")
        for _, r in fragile.sort_values("membership").iterrows():
            rep(f"    {r['label']:<24} P = {r['membership']:.2f}")

    near = el[(el["n_dominators"] > 0) & (el["membership"] >= marginal_lo)]
    if len(near):
        rep()
        rep(f"Dominated on the means but on the front in >= {marginal_lo:.0%} of "
            "resamples -- treat as effectively tied with the front:")
        for _, r in near.sort_values("membership", ascending=False).iterrows():
            rep(f"    {r['label']:<24} P = {r['membership']:.2f}, "
                f"{int(r['n_dominators'])} dominator(s)")

    # --- Dominated set -----------------------------------------------------
    rep.rule("4. DOMINATED CANDIDATES", "-")
    dominated = el[el["n_dominators"] > 0].sort_values(
        ["n_dominators", "best_dom_prob"], ascending=[False, False])
    rep(f"{len(dominated)} dominated. 'by' is the dominator with the highest "
        "seed-level win rate.")
    rep("'P(all 3)' is the fraction of seeds on which that dominator wins on "
        "all three criteria at once.")
    rep()
    header = (f"    {'candidate':<24} {'EV':>6} {'#dom':>5} {'by':>24} "
              f"{'P(all 3)':>9}  {'dE':>8} {'dC':>8} {'dU':>8}")
    rep(header)
    rep("    " + "-" * (len(header) - 4))
    for _, r in dominated.iterrows():
        rep(f"    {r['label']:<24} {r['ev']:>6.3f} {int(r['n_dominators']):>5} "
            f"{r['best_dominator']:>24} {r['best_dom_prob']:>9.2f}  "
            f"{r['best_dE']:>+8.4f} {r['best_dC']:>+8.4f} {r['best_dU']:>+8.3f}")
    rep()
    rep("dE, dC, dU are dominator minus dominated. Negative dE and dC and "
        "positive dU is the improvement the dominator delivers.")

    # --- Per-instrument survival ------------------------------------------
    rep.rule("5. SURVIVAL BY INSTRUMENT", "-")
    rep("Of the eligible candidates that use each instrument (in either "
        "position), how many are non-dominated?")
    rep()
    rep(f"    {'instrument':<22} {'eligible':>9} {'on front':>9} {'share':>7} "
        f"{'mean P(front)':>14}")
    rep("    " + "-" * 64)
    surv = instrument_survival(el)
    for _, r in surv.iterrows():
        rep(f"    {POLICY_TITLES.get(r['policy'], r['policy']):<22} "
            f"{int(r['n_eligible']):>9} {int(r['n_front']):>9} "
            f"{r['share']:>7.2f} {r['mean_membership']:>14.2f}")

    dead = surv[surv["n_front"] == 0]["policy"].tolist()
    if dead:
        rep()
        rep("Never on the front: " + ", ".join(POLICY_TITLES.get(p, p) for p in dead))

    # --- Best pair combinations -------------------------------------------
    rep.rule("6. SURVIVAL BY INSTRUMENT COMBINATION", "-")
    rep("Pair sweeps are directed: (A, B) sweeps A and re-optimises B. Both "
        "directions of the same unordered combination are pooled here, so a "
        "combination appearing in both directions is a consistency check.")
    rep()
    combo = combination_survival(el)
    rep(f"    {'combination':<34} {'eligible':>9} {'on front':>9} {'share':>7} "
        f"{'mean P(front)':>14}")
    rep("    " + "-" * 76)
    for _, r in combo.iterrows():
        rep(f"    {r['combination']:<34} {int(r['n_eligible']):>9} "
            f"{int(r['n_front']):>9} {r['share']:>7.2f} "
            f"{r['mean_membership']:>14.2f}")

    # --- Caveats -----------------------------------------------------------
    rep.rule("7. CAVEATS TO CARRY INTO ANY WRITE-UP", "-")
    rep("  - This is dominance over a SAMPLED set: a 5-point grid on policy 1 "
        "with policy 2 chosen by BO. A candidate can look non-dominated only "
        "because the sweep never sampled the mix that would beat it. It is not "
        "the true continuous 2-D frontier.")
    rep("  - The BO targets EV uptake, not emissions, cost or utility, so "
        "policy 2 is not cost- or utility-optimal at each step.")
    rep(f"  - Candidates inside the EV window still differ in uptake "
        f"({el['ev'].min():.3f} to {el['ev'].max():.3f}). A slightly higher-uptake "
        "candidate is doing more work for the same criteria. Tighten the window "
        "or add uptake as a fourth criterion if that matters.")
    rep("  - Utility is scaled by 1/prob_switch_car. That is a positive "
        "constant, so it changes no ordering.")
    rep("  - BAU carries no per-seed arrays, so it is a reference point only, "
        "never a dominator.")

    return el


def instrument_survival(el):
    rows = []
    for policy in sorted({p for p in list(el["policy1"]) + list(el["policy2"])
                          if isinstance(p, str)}):
        mask = (el["policy1"] == policy) | (el["policy2"] == policy)
        sub = el[mask]
        n_front = int((sub["n_dominators"] == 0).sum())
        rows.append(dict(policy=policy, n_eligible=len(sub), n_front=n_front,
                         share=n_front / len(sub) if len(sub) else np.nan,
                         mean_membership=sub["membership"].mean()))
    return pd.DataFrame(rows).sort_values("share", ascending=False)


def combination_survival(el):
    def key(r):
        if r["kind"] == "single":
            return f"{abbrev(r['policy1'])} only"
        return " + ".join(sorted([abbrev(r["policy1"]), abbrev(r["policy2"])]))

    tmp = el.copy()
    tmp["combination"] = tmp.apply(key, axis=1)
    g = tmp.groupby("combination").agg(
        n_eligible=("label", "size"),
        n_front=("n_dominators", lambda s: int((s == 0).sum())),
        mean_membership=("membership", "mean"),
    ).reset_index()
    g["share"] = g["n_front"] / g["n_eligible"]
    return g.sort_values(["n_front", "mean_membership"], ascending=False)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _policy_colors(el):
    policies = sorted({p for p in list(el["policy1"]) + list(el["policy2"])
                       if isinstance(p, str)})
    cmap = ListedColormap(OKABE_ITO)
    return {p: cmap(i) for i, p in enumerate(policies)}, policies


def _draw_candidate(ax, x, y, row, colors, size, on_front):
    """Two half discs, one per instrument, so the mix is readable."""
    if not on_front:
        ax.scatter(x, y, s=size * 0.35, marker=full_circle_marker(),
                   facecolor="none", edgecolor="0.65", linewidth=0.8, zorder=2)
        return
    c1 = colors.get(row["policy1"], "black")
    c2 = colors.get(row["policy2"], c1)
    ax.scatter(x, y, s=size, marker=half_circle_marker(0, 180),
               color=c1, edgecolor="black", linewidth=0.7, zorder=4)
    ax.scatter(x, y, s=size, marker=half_circle_marker(180, 360),
               color=c2, edgecolor="black", linewidth=0.7, zorder=4)


def front_ranks(el):
    """Front members numbered 1..k by emissions, for compact plot labels."""
    front = el[el["n_dominators"] == 0].sort_values("E")
    return {lab: i + 1 for i, lab in enumerate(front["label"])}


def plot_pareto_scatter(el, seeds, bau_row, out_dir, dpi=300, zoom=False,
                        zoom_quantile=0.92):
    """
    Every pair of criteria, front filled and numbered, dominated hollow.

    zoom=True clips the axes to the given quantile of each criterion, so a few
    very bad candidates do not squash the interesting region.
    """
    colors, policies = _policy_colors(el)
    panels = [("E", "C"), ("E", "U"), ("C", "U")]
    axis_labels = {k: lab for k, lab, _ in CRITERIA}
    ranks = front_ranks(el)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    idx = el.index.to_numpy()
    n_seeds = seeds["E"].shape[1]

    # Clip only the bad tail of each criterion, so no front member is cut off.
    lims = {}
    for k, _, sense in CRITERIA:
        v = el[k].to_numpy()
        lo, hi = v.min(), v.max()
        if zoom:
            if sense > 0:   # lower is better, so the high tail is the bad one
                hi = max(np.quantile(v, zoom_quantile), el.loc[el["n_dominators"] == 0, k].max())
            else:           # higher is better
                lo = min(np.quantile(v, 1 - zoom_quantile), el.loc[el["n_dominators"] == 0, k].min())
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        lims[k] = (lo - pad, hi + pad)

    for ax, (xk, yk) in zip(axes, panels):
        xs, ys = el[xk].to_numpy(), el[yk].to_numpy()
        xerr = 1.96 * seeds[xk][idx].std(axis=1) / np.sqrt(n_seeds)
        yerr = 1.96 * seeds[yk][idx].std(axis=1) / np.sqrt(n_seeds)

        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor="gray", alpha=0.35, zorder=1)

        for i, (_, row) in enumerate(el.iterrows()):
            _draw_candidate(ax, xs[i], ys[i], row, colors, 260,
                            row["n_dominators"] == 0)

        for i, (_, row) in enumerate(el.iterrows()):
            if row["label"] in ranks:
                ax.annotate(str(ranks[row["label"]]), (xs[i], ys[i]),
                            textcoords="offset points", xytext=(11, 7),
                            fontsize=8, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="0.6", lw=0.5, alpha=0.85))

        if bau_row is not None:
            ax.scatter(bau_row[xk], bau_row[yk], s=200, marker="*",
                       color="black", zorder=5)
            ax.annotate("BAU", (bau_row[xk], bau_row[yk]),
                        textcoords="offset points", xytext=(8, -11), fontsize=8)

        ax.set_xlim(*lims[xk])
        ax.set_ylim(*lims[yk])
        ax.set_xlabel(axis_labels[xk], fontsize=11)
        ax.set_ylabel(axis_labels[yk], fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5)

    handles = [Patch(facecolor=colors[p], edgecolor="black",
                     label=f"{POLICY_TITLES.get(p, p)} ({abbrev(p)})")
               for p in policies]
    handles += [
        plt.Line2D([0], [0], marker="o", color="0.65", markerfacecolor="none",
                   linestyle="None", markersize=7, label="dominated"),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="gray",
                   linestyle="None", markersize=9, label="non-dominated"),
        plt.Line2D([0], [0], marker="*", color="black", linestyle="None",
                   markersize=11, label="BAU (not eligible)"),
    ]
    leg1 = fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
                      frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.add_artist(leg1)

    key = [plt.Line2D([0], [0], linestyle="None", marker="None",
                      label=f"{r} = {lab}")
           for lab, r in sorted(ranks.items(), key=lambda kv: kv[1])]
    fig.legend(handles=key, loc="lower center", ncol=6, fontsize=8,
               frameon=False, handlelength=0, handletextpad=0,
               bbox_to_anchor=(0.5, -0.13),
               title="Pareto front, numbered by emissions", title_fontsize=9)

    suffix = " (zoomed, extremes clipped)" if zoom else ""
    fig.suptitle("Pareto front over emissions, net cost and utility"
                 f"{suffix} -- half discs show the two instruments", fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    name = "fig1b_pareto_scatter_zoom" if zoom else "fig1_pareto_scatter"
    fig.savefig(f"{out_dir}/{name}.png", dpi=dpi, bbox_inches="tight")
    return fig


def plot_membership(el, out_dir, dpi=300, marginal_hi=0.95):
    """Bootstrap Pareto-membership probability, worst to best."""
    d = el.sort_values("membership")
    n = len(d)
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.22 * n)))

    front = d["n_dominators"].to_numpy() == 0
    cols = np.where(front, "#009E73", "#D55E00")
    ax.barh(np.arange(n), d["membership"], color=cols, edgecolor="black",
            linewidth=0.4)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(d["label"], fontsize=7)
    ax.axvline(marginal_hi, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="0.5", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("P(non-dominated) under paired seed resampling", fontsize=11)
    ax.set_title("How robust is each verdict?", fontsize=12)
    ax.legend(handles=[
        Patch(facecolor="#009E73", edgecolor="black", label="non-dominated on the means"),
        Patch(facecolor="#D55E00", edgecolor="black", label="dominated on the means"),
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"P = {marginal_hi:.2f}"),
    ], loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig2_membership.png", dpi=dpi)
    return fig


def plot_dominance_heatmap(el, P_seed, out_dir, dpi=300):
    """
    Seed-level P(row dominates column). Rows and columns share one order,
    strongest dominator first, so the bright mass sits in the top-right and the
    candidates that beat nothing sink to the bottom.
    """
    order = np.lexsort((-el["membership"].to_numpy(), -el["n_beats"].to_numpy()))
    P = P_seed[np.ix_(order, order)]
    labels = el["label"].to_numpy()[order]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.16 * n + 3), max(6.0, 0.16 * n + 2)))
    im = ax.imshow(P, cmap="magma", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("dominated candidate", fontsize=11)
    ax.set_ylabel("dominating candidate", fontsize=11)
    ax.set_title("P(row beats column on emissions AND cost AND utility), per seed\n"
                 "ordered by how many candidates each one beats", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.75, label="fraction of seeds")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig3_dominance_heatmap.png", dpi=dpi)
    return fig


def plot_instrument_survival(el, out_dir, dpi=300):
    surv = instrument_survival(el)
    colors, _ = _policy_colors(el)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(surv))
    ax.barh(y, surv["share"], color=[colors[p] for p in surv["policy"]],
            edgecolor="black", linewidth=0.6)
    ax.plot(surv["mean_membership"], y, "kD", markersize=6,
            label="mean P(non-dominated)")
    for i, r in enumerate(surv.itertuples()):
        ax.text(max(r.share, r.mean_membership) + 0.02, i,
                f"{int(r.n_front)}/{int(r.n_eligible)}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([POLICY_TITLES.get(p, p) for p in surv["policy"]], fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("share of that instrument's eligible candidates on the front",
                  fontsize=10)
    ax.set_title("Which instruments survive the dominance test?", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig4_instrument_survival.png", dpi=dpi)
    return fig


def plot_dominator_counts(el, out_dir, dpi=300):
    """How deeply is each candidate beaten?"""
    d = el.sort_values("n_dominators")
    n = len(d)
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.22 * n)))
    cols = np.where(d["n_dominators"].to_numpy() == 0, "#009E73", "#0072B2")
    ax.barh(np.arange(n), d["n_dominators"], color=cols, edgecolor="black",
            linewidth=0.4)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(d["label"], fontsize=7)
    ax.set_xlabel("number of candidates that dominate it", fontsize=11)
    ax.set_title("Depth of domination (0 = on the Pareto front)", fontsize=12)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig5_dominator_counts.png", dpi=dpi)
    return fig


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def analyse(fileNames, ev_min=0.94, ev_max=0.96, n_boot=1000, boot_seed=0,
            eps=(0.0, 0.0, 0.0), margin_on="any", dpi=300, show=False,
            out_name="dominance", out_folder=None):
    """
    Run the whole analysis and write text, CSVs and figures.

    fileNames  : one endog_pair folder, or a list of them. Same convention as
                 endogenous_policy_intensity_pair_plot.main: base_params,
                 outcomes_BAU and single_policy_outcomes come from the FIRST
                 folder, pairwise_outcomes is merged across all of them.
    out_folder : where to write. Defaults to the first input folder, so the
                 output lands next to the data it came from.
    ev_min, ev_max : the eligibility window on mean EV uptake
    eps, margin_on : winning margin required before dominance is claimed, see
                 dominance_matrix. Raising eps grows the front.
    out_name   : subfolder of Plots/, so several settings can live side by side.
    """
    if isinstance(fileNames, str):
        fileNames = [fileNames]
    fileName = fileNames[0]

    base_params = load_object(f"{fileName}/Data", "base_params")
    single_outcomes = load_object(f"{fileName}/Data", "single_policy_outcomes")
    outcomes_BAU = load_object(f"{fileName}/Data", "outcomes_BAU")

    pairwise_outcomes = {}
    for folder in fileNames:
        pairwise_outcomes.update(load_object(f"{folder}/Data", "pairwise_outcomes"))

    if out_folder is None:
        out_folder = fileName
    out_dir = f"{out_folder}/Plots/{out_name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_folder}/Data", exist_ok=True)

    df, seeds, n_seeds = build_candidate_table(
        base_params, pairwise_outcomes, single_outcomes, outcomes_BAU,
        ev_min, ev_max)

    el_idx = df.index[df["eligible"]].to_numpy()
    sE, sC, sU = seeds["E"][el_idx], seeds["C"][el_idx], seeds["U"][el_idx]

    el = df.loc[el_idx].copy()
    dom = dominance_matrix(el["E"], el["C"], el["U"], eps=eps, margin_on=margin_on)
    P_seed = seedwise_dominance_probability(sE, sC, sU)
    membership = bootstrap_membership(sE, sC, sU, n_boot=n_boot, seed=boot_seed,
                                     eps=eps, margin_on=margin_on)

    # Attach per-candidate summaries.
    el["n_dominators"] = dom.sum(axis=0)
    el["n_beats"] = dom.sum(axis=1)
    el["membership"] = membership

    labels = el["label"].to_numpy()
    best_dom, best_p, best_dE, best_dC, best_dU = [], [], [], [], []
    for j in range(len(el)):
        doms = np.flatnonzero(dom[:, j])
        if len(doms) == 0:
            best_dom.append("-"); best_p.append(np.nan)
            best_dE.append(np.nan); best_dC.append(np.nan); best_dU.append(np.nan)
            continue
        i = doms[np.argmax(P_seed[doms, j])]
        best_dom.append(labels[i])
        best_p.append(P_seed[i, j])
        best_dE.append(el["E"].iloc[i] - el["E"].iloc[j])
        best_dC.append(el["C"].iloc[i] - el["C"].iloc[j])
        best_dU.append(el["U"].iloc[i] - el["U"].iloc[j])
    el["best_dominator"] = best_dom
    el["best_dom_prob"] = best_p
    el["best_dE"] = best_dE
    el["best_dC"] = best_dC
    el["best_dU"] = best_dU

    # --- text report -------------------------------------------------------
    rep = Reporter()
    el = report(rep, df.join(el[["n_dominators", "n_beats", "membership",
                                "best_dominator", "best_dom_prob",
                                "best_dE", "best_dC", "best_dU"]]),
                dom, P_seed, membership, ev_min, ev_max, n_seeds, n_boot, eps,
                margin_on=margin_on)
    rep.rule()
    rep(f"Written to {out_dir}")
    rep.write(f"{out_dir}/dominance_report.txt")

    # --- tables ------------------------------------------------------------
    full = df.join(el[["n_dominators", "n_beats", "membership", "best_dominator",
                       "best_dom_prob", "best_dE", "best_dC", "best_dU"]])
    full.to_csv(f"{out_dir}/candidates.csv", index=False)

    rel = []
    for i, j in zip(*np.nonzero(dom)):
        rel.append(dict(
            dominator=labels[i], dominated=labels[j],
            p_all_three_seedwise=P_seed[i, j],
            dE=el["E"].iloc[i] - el["E"].iloc[j],
            dC=el["C"].iloc[i] - el["C"].iloc[j],
            dU=el["U"].iloc[i] - el["U"].iloc[j],
        ))
    pd.DataFrame(rel).to_csv(f"{out_dir}/dominance_pairs.csv", index=False)

    save_object({"eps": eps, "margin_on": margin_on,
                 "ev_min": ev_min, "ev_max": ev_max,
                 "n_boot": n_boot, "n_seeds": n_seeds,
                 "labels": list(labels), "dominance_matrix": dom,
                 "P_seedwise": P_seed, "membership": membership},
                f"{out_folder}/Data", f"{out_name}_results")

    # --- figures -----------------------------------------------------------
    bau = df[df["kind"] == "BAU"]
    bau_row = bau.iloc[0] if len(bau) else None
    plot_pareto_scatter(el, seeds, bau_row, out_dir, dpi=dpi, zoom=False)
    plot_pareto_scatter(el, seeds, bau_row, out_dir, dpi=dpi, zoom=True)
    plot_membership(el, out_dir, dpi=dpi)
    plot_dominance_heatmap(el, P_seed, out_dir, dpi=dpi)
    plot_instrument_survival(el, out_dir, dpi=dpi)
    plot_dominator_counts(el, out_dir, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close("all")

    return el, dom, P_seed


if __name__ == "__main__":
    # One or more endog_pair folders. Output goes to the first one, in
    # <folder>/Plots/dominance.
    FOLDERS = ["results/endog_pair_15_10_05__13_08_2026"]

    # Headline result: textbook dominance, robustness from the bootstrap.
    analyse(
        fileNames=FOLDERS,
        ev_min=0.94,
        ev_max=0.96,
        n_boot=1000,
        eps=(0.0, 0.0, 0.0),
        out_name="dominance",
        show=True,
    )

    # Conservative variant: a dominator must clear the CI half-width on every
    # criterion. Only clear-cut wins count, so the front can only grow.
    # analyse(
    #     fileNames=FOLDERS,
    #     eps=(0.0063, 0.0089, 0.13),
    #     margin_on="all",
    #     out_name="dominance_conservative",
    #     show=False,
    # )
